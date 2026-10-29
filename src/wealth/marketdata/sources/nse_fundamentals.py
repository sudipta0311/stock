"""Official NSE Integrated Filing financials API and XBRL parser."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx

from wealth.marketdata.sources.fundamentals import FundamentalObservation, retention_cutoff

NSE_FILINGS_PAGE_URL = "https://www.nseindia.com/companies-listing/corporate-integrated-filing"
NSE_FILINGS_API_URL = "https://www.nseindia.com/api/integrated-filing-results"
NSE_SOURCE = "NSE_XBRL"
NSE_SOURCE_NAME = "NSE Integrated Filing Financials"
_XBRL_HOST = "nsearchives.nseindia.com"
_NSE_TIMEZONE = ZoneInfo("Asia/Kolkata")
_REQUEST_INTERVAL_SECONDS = 0.2
_SURVIVORSHIP_LABEL = "current constituents only (survivorship bias)"
_PERIOD_METRICS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromOperations",
        "Revenue",
        "RevenueFromOperationsFromContinuingOperations",
    ),
    "net_income": (
        "ProfitLossForPeriod",
        "ProfitLossForPeriodFromContinuingOperations",
        "ProfitLoss",
    ),
    "operating_income": (
        "ProfitBeforeInterestAndTax",
        "OperatingProfitBeforeInterestAndTax",
        "OperatingProfit",
    ),
    "eps_diluted": (
        "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "DilutedEarningsLossPerShareFromContinuingOperations",
        "DilutedEarningsPerShare",
    ),
}
_INSTANT_METRICS: dict[str, tuple[str, ...]] = {
    "total_assets": ("Assets", "TotalAssets"),
    "total_equity": ("Equity", "EquityAttributableToOwnersOfParent"),
    "total_debt": ("Borrowings", "TotalBorrowings", "Debt"),
    "current_liabilities": ("CurrentLiabilities", "LiabilitiesCurrent"),
}


@dataclass(frozen=True)
class NseFiling:
    symbol: str
    period_end: date
    filed_at: datetime
    xbrl_url: str
    consolidated: bool
    submission_type: str


class NseIntegratedClient:
    """Session client for NSE's public filings page, JSON index and XBRL archives."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; WealthCopilotMarketData/1.0)",
                "Accept": "application/json,text/xml,*/*",
                "Referer": NSE_FILINGS_PAGE_URL,
            },
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._last_request_at: float | None = None
        self._initialized = False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NseIntegratedClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _get(self, url: str, *, params: Mapping[str, str] | None = None) -> httpx.Response:
        if self._last_request_at is not None:
            remaining = _REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response

    def initialize(self) -> None:
        if not self._initialized:
            self._get(NSE_FILINGS_PAGE_URL)
            self._initialized = True

    def filings_for_symbol(self, symbol: str) -> list[NseFiling]:
        self.initialize()
        rows: list[Mapping[str, Any]] = []
        page = 1
        total_count = 0
        while page == 1 or len(rows) < total_count:
            response = self._get(
                NSE_FILINGS_API_URL,
                params={
                    "symbol": symbol,
                    "type": "Integrated Filing- Financials",
                    "page": str(page),
                    "size": "100",
                },
            )
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("NSE filings response must contain a JSON object")
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise ValueError("NSE filings response has no rows")
            rows.extend(row for row in data if isinstance(row, Mapping))
            total_count = int(payload.get("totalCount", len(rows)))
            if not data:
                break
            page += 1
        return parse_filing_rows(rows, symbol=symbol)

    def fetch_xbrl(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != _XBRL_HOST:
            raise ValueError("NSE XBRL URL must use the official archive host")
        return self._get(url).content


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    for pattern in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    for pattern in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(value.strip(), pattern)
            return parsed.replace(tzinfo=_NSE_TIMEZONE).astimezone(UTC)
        except ValueError:
            continue
    return None


def parse_filing_rows(rows: list[Mapping[str, Any]], *, symbol: str) -> list[NseFiling]:
    """Parse public filing-index rows, preferring consolidated reports per submission."""
    parsed: list[NseFiling] = []
    for row in rows:
        row_symbol = str(row.get("symbol") or symbol).strip().upper()
        period_end = _parse_date(row.get("qe_Date"))
        filed_at = _parse_timestamp(row.get("revised_Date")) or _parse_timestamp(
            row.get("broadcast_Date")
        )
        xbrl_url = row.get("xbrl")
        if (
            row_symbol != symbol.upper()
            or period_end is None
            or filed_at is None
            or not isinstance(xbrl_url, str)
            or not xbrl_url
        ):
            continue
        parsed.append(
            NseFiling(
                symbol=row_symbol,
                period_end=period_end,
                filed_at=filed_at,
                xbrl_url=xbrl_url,
                consolidated=str(row.get("consolidated", "")).casefold() == "consolidated",
                submission_type=str(row.get("type_Sub") or "Original"),
            )
        )

    by_submission: dict[tuple[date, str, date], list[NseFiling]] = defaultdict(list)
    for filing in parsed:
        filing_day = filing.filed_at.astimezone(_NSE_TIMEZONE).date()
        by_submission[(filing.period_end, filing.submission_type, filing_day)].append(filing)
    selected: list[NseFiling] = []
    for filings in by_submission.values():
        consolidated = [filing for filing in filings if filing.consolidated]
        selected.extend(consolidated or filings[:1])
    return sorted(selected, key=lambda filing: (filing.filed_at, filing.period_end))


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _decimal_fact(element: ET.Element) -> Decimal | None:
    text_value = (element.text or "").strip()
    if (
        not text_value
        or element.attrib.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true"
    ):
        return None
    try:
        value = Decimal(text_value)
        scale = int(element.attrib.get("scale", "0"))
        if scale:
            value *= Decimal(10) ** scale
        if element.attrib.get("sign") == "-" and value > 0:
            value = -value
        return value if value.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _context_period(context: ET.Element) -> tuple[date | None, date | None, date | None, bool]:
    period = next((child for child in context if _local_name(child) == "period"), None)
    if period is None:
        return None, None, None, False
    start = end = instant = None
    for element in period:
        name = _local_name(element)
        parsed = _parse_date(element.text)
        if name == "startDate":
            start = parsed
        elif name == "endDate":
            end = parsed
        elif name == "instant":
            instant = parsed
    has_dimensions = any(
        _local_name(element) in {"scenario", "segment"}
        for element in context.iter()
        if element is not context
    )
    return start, end, instant, has_dimensions


def _duration_period_type(start: date, end: date) -> str | None:
    days = (end - start).days + 1
    if 45 <= days <= 120:
        return "Q"
    if 300 <= days <= 380:
        return "FY"
    return None


def _fundamentals_cutoff(as_of: date) -> date:
    return retention_cutoff(as_of)


def _context_facts(
    root: ET.Element,
) -> tuple[
    dict[str, tuple[date | None, date | None, date | None, bool]],
    dict[str, dict[str, list[tuple[str, Decimal]]]],
]:
    contexts: dict[str, tuple[date | None, date | None, date | None, bool]] = {}
    for element in root.iter():
        if _local_name(element) == "context" and element.attrib.get("id"):
            contexts[element.attrib["id"]] = _context_period(element)
    facts: dict[str, dict[str, list[tuple[str, Decimal]]]] = defaultdict(lambda: defaultdict(list))
    for element in root.iter():
        context_ref = element.attrib.get("contextRef")
        if context_ref not in contexts:
            continue
        value = _decimal_fact(element)
        unit = element.attrib.get("unitRef")
        if value is not None and unit:
            facts[context_ref][_local_name(element)].append((unit, value))
    return contexts, facts


def _value(
    context_facts: Mapping[str, list[tuple[str, Decimal]]], tags: tuple[str, ...]
) -> tuple[str, Decimal] | None:
    for tag in tags:
        rows = context_facts.get(tag, [])
        if rows:
            return rows[0]
    return None


def _operating_income(
    context_facts: Mapping[str, list[tuple[str, Decimal]]],
) -> tuple[str, Decimal] | None:
    direct = _value(context_facts, _PERIOD_METRICS["operating_income"])
    if direct is not None:
        return direct
    profit_before_exceptional = _value(
        context_facts, ("ProfitBeforeExceptionalItemsAndTax", "ProfitBeforeTax")
    )
    finance_costs = _value(context_facts, ("FinanceCosts", "FinanceCost"))
    other_income = _value(context_facts, ("OtherIncome",))
    if profit_before_exceptional is None:
        return None
    unit, value = profit_before_exceptional
    if finance_costs is not None and finance_costs[0] == unit:
        value += finance_costs[1]
    if other_income is not None and other_income[0] == unit:
        value -= other_income[1]
    return unit, value


def _total_debt(
    context_facts: Mapping[str, list[tuple[str, Decimal]]],
) -> tuple[str, Decimal] | None:
    direct = _value(context_facts, _INSTANT_METRICS["total_debt"])
    if direct is not None:
        return direct
    current = _value(context_facts, ("CurrentBorrowings", "BorrowingsCurrent"))
    noncurrent = _value(context_facts, ("NonCurrentBorrowings", "BorrowingsNonCurrent"))
    if current is not None and noncurrent is not None and current[0] == noncurrent[0]:
        return current[0], current[1] + noncurrent[1]
    return None


def parse_xbrl_filing(
    xml_data: bytes | str,
    filing: NseFiling,
    *,
    as_of: date,
) -> list[FundamentalObservation]:
    """Read current-period SEBI XBRL facts, ignoring dimensional segment disclosures."""
    root = ET.fromstring(xml_data)
    contexts, all_facts = _context_facts(root)
    cutoff = _fundamentals_cutoff(as_of)
    if (
        filing.period_end < cutoff
        or filing.period_end > as_of
        or filing.filed_at.astimezone(_NSE_TIMEZONE).date() > as_of
    ):
        return []
    period_types = {
        period_type
        for start, end, _instant, _has_dimensions in contexts.values()
        if start is not None and end == filing.period_end
        if (period_type := _duration_period_type(start, end)) is not None
    }
    filing_period_type = "FY" if "FY" in period_types else "Q"
    observations: list[FundamentalObservation] = []

    def add(metric: str, period_type: str, unit_value: tuple[str, Decimal] | None) -> None:
        if unit_value is None:
            return
        unit, value = unit_value
        if metric == "eps_diluted" and not unit.casefold().endswith("pershare"):
            return
        if metric != "eps_diluted" and "pershare" in unit.casefold():
            return
        observations.append(
            FundamentalObservation(
                metric=metric,
                period_end=filing.period_end,
                period_type=period_type,
                value=value,
                unit=unit,
                filed_at=filing.filed_at,
                known_at=filing.filed_at,
                source=NSE_SOURCE,
                source_url=filing.xbrl_url,
            )
        )

    for context_id, (_start, end, instant, has_dimensions) in contexts.items():
        if has_dimensions:
            continue
        context_facts = all_facts[context_id]
        if end == filing.period_end and _start is not None:
            period_type = _duration_period_type(_start, end)
            if period_type is None:
                continue
            for metric in ("revenue", "net_income", "eps_diluted"):
                add(metric, period_type, _value(context_facts, _PERIOD_METRICS[metric]))
            add("operating_income", period_type, _operating_income(context_facts))
        if instant == filing.period_end:
            for metric, tags in _INSTANT_METRICS.items():
                value = (
                    _total_debt(context_facts)
                    if metric == "total_debt"
                    else _value(context_facts, tags)
                )
                add(metric, filing_period_type, value)

    unique: dict[tuple[str, str, Decimal, str], FundamentalObservation] = {}
    for observation in observations:
        unique[
            (observation.metric, observation.period_type, observation.value, observation.unit)
        ] = observation
    return list(unique.values())


def _select_recent(
    filings: list[NseFiling], *, as_of: date, after: datetime | None
) -> list[NseFiling]:
    cutoff = _fundamentals_cutoff(as_of)
    return [
        filing
        for filing in filings
        if cutoff <= filing.period_end <= as_of
        and filing.filed_at.astimezone(_NSE_TIMEZONE).date() <= as_of
        and (after is None or filing.filed_at > after)
    ]


def fetch_symbol_fundamentals(
    client: NseIntegratedClient,
    symbol: str,
    *,
    as_of: date,
    after: datetime | None = None,
) -> list[FundamentalObservation]:
    observations: list[FundamentalObservation] = []
    for filing in _select_recent(client.filings_for_symbol(symbol), as_of=as_of, after=after):
        xml_data = client.fetch_xbrl(filing.xbrl_url)
        observations.extend(parse_xbrl_filing(xml_data, filing, as_of=as_of))
    return observations


def survivorship_label() -> str:
    return _SURVIVORSHIP_LABEL
