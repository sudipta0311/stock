"""Point-in-time SEC EDGAR XBRL company-facts adapter."""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from wealth.marketdata.sources.fundamentals import FundamentalObservation, retention_cutoff

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SOURCE = "SEC_EDGAR"
_REQUEST_INTERVAL_SECONDS = 0.1
_EMAIL_PATTERN = re.compile(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+")
_FLOW_METRICS = ("revenue", "net_income", "operating_income", "operating_cash_flow", "capex")
# Per-share earnings differences approximate a quarter (share counts move a little), which is
# how a fourth-quarter EPS is usually read from a 10-K that reports only the full year.
_QUARTERLY_METRICS = (*_FLOW_METRICS, "eps_diluted")
# Tag choices are ordered from preferred modern concepts to older issuer fallbacks.
_SIMPLE_TAGS: dict[str, tuple[tuple[str, str], ...]] = {
    "revenue": (
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "SalesRevenueGoodsNet"),
    ),
    "net_income": (("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")),
    "operating_income": (("us-gaap", "OperatingIncomeLoss"),),
    "total_assets": (("us-gaap", "Assets"),),
    "total_equity": (
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ),
    "cash": (
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ),
    "current_liabilities": (("us-gaap", "LiabilitiesCurrent"),),
    "shares_outstanding": (("dei", "EntityCommonStockSharesOutstanding"),),
    "eps_diluted": (
        ("us-gaap", "EarningsPerShareDiluted"),
        ("us-gaap", "EarningsPerShareBasicAndDiluted"),
    ),
    "operating_cash_flow": (
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ),
    "capex": (
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ),
}
_UNITS = {
    "shares_outstanding": {"shares"},
    "eps_diluted": {"USD/shares", "USD / shares"},
}
_DEBT_TAG_GROUPS = (
    (
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    ),
    ("ShortTermBorrowings", "LongTermDebtCurrent", "LongTermDebtNoncurrent"),
    ("LongTermDebtCurrent", "LongTermDebtNoncurrent"),
    ("LongTermDebt",),
)


class EdgarConfigurationError(ValueError):
    """Raised when SEC access has no identifying contact user agent."""


@dataclass(frozen=True)
class EdgarCompany:
    ticker: str
    cik: str
    name: str


class SecEdgarClient:
    """Small synchronous SEC client that spaces requests to at most 10 per second."""

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        get: Callable[..., httpx.Response] | None = None,
    ) -> None:
        supplied = user_agent if user_agent is not None else os.getenv("SEC_USER_AGENT", "")
        self.user_agent = supplied.strip()
        if not self.user_agent or not _EMAIL_PATTERN.search(self.user_agent):
            raise EdgarConfigurationError(
                "Set SEC_USER_AGENT to an identifying client string with a contact email."
            )
        self._get = get or httpx.get
        self._last_request_at: float | None = None

    def get_json(self, url: str) -> Mapping[str, Any]:
        if self._last_request_at is not None:
            remaining = _REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()
        response = self._get(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("SEC response must contain a JSON object")
        return payload


def _normalize_ticker(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def parse_company_tickers(payload: Mapping[str, Any]) -> dict[str, EdgarCompany]:
    """Read SEC's numeric-keyed ticker file and normalize share-class punctuation."""
    companies: dict[str, EdgarCompany] = {}
    for value in payload.values():
        if not isinstance(value, Mapping):
            continue
        ticker = value.get("ticker")
        cik_value = value.get("cik_str")
        name = value.get("title")
        if (
            not isinstance(ticker, str)
            or not isinstance(name, str)
            or not isinstance(cik_value, int)
        ):
            continue
        company = EdgarCompany(ticker.upper(), f"{cik_value:010d}", name.strip())
        companies[_normalize_ticker(company.ticker)] = company
    return companies


def fetch_company_tickers(client: SecEdgarClient | None = None) -> dict[str, EdgarCompany]:
    sec_client = client or SecEdgarClient()
    return parse_company_tickers(sec_client.get_json(SEC_COMPANY_TICKERS_URL))


def fetch_companyfacts(cik: str, client: SecEdgarClient | None = None) -> Mapping[str, Any]:
    sec_client = client or SecEdgarClient()
    normalized_cik = cik.strip().removeprefix("CIK").zfill(10)
    if not normalized_cik.isdigit() or len(normalized_cik) != 10:
        raise ValueError("CIK must contain at most ten digits")
    return sec_client.get_json(SEC_COMPANYFACTS_URL.format(cik=normalized_cik))


def _date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _filed_at(value: Any) -> datetime | None:
    filed = _date(value)
    return datetime.combine(filed, datetime.min.time(), tzinfo=UTC) if filed else None


def _period_type(item: Mapping[str, Any]) -> str | None:
    end = _date(item.get("end"))
    start = _date(item.get("start"))
    if end is None:
        return None
    if start is None:
        form = str(item.get("form", ""))
        if form.startswith("10-K"):
            return "FY"
        if form.startswith("10-Q"):
            return "Q"
        return None
    days = (end - start).days + 1
    if 45 <= days <= 120:
        return "Q"
    if 300 <= days <= 380:
        return "FY"
    return None


def _entries(
    company_facts: Mapping[str, Any],
    tags: tuple[tuple[str, str], ...],
    *,
    unit_names: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    facts = company_facts.get("facts", {})
    if not isinstance(facts, Mapping):
        return []
    for taxonomy, tag in tags:
        taxonomy_facts = facts.get(taxonomy, {})
        if not isinstance(taxonomy_facts, Mapping):
            continue
        tagged = taxonomy_facts.get(tag)
        if not isinstance(tagged, Mapping):
            continue
        units = tagged.get("units", {})
        if not isinstance(units, Mapping):
            continue
        selected_units = [unit for unit in units if unit_names is None or unit in unit_names]
        if not selected_units:
            continue
        return [
            (unit, item)
            for unit in selected_units
            for item in units.get(unit, [])
            if isinstance(item, dict)
        ]
    return []


def _debt_entries(company_facts: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    for tags in _DEBT_TAG_GROUPS:
        components = [
            _entries(company_facts, (("us-gaap", tag),), unit_names={"USD"}) for tag in tags
        ]
        if any(not component for component in components):
            continue
        by_filing: dict[tuple[Any, ...], Decimal] = {}
        item_by_filing: dict[tuple[Any, ...], dict[str, Any]] = {}
        for component in components:
            for _unit, item in component:
                key = (
                    item.get("accn"),
                    item.get("filed"),
                    item.get("form"),
                    item.get("start"),
                    item.get("end"),
                )
                try:
                    value = Decimal(str(item["val"]))
                except (KeyError, InvalidOperation):
                    continue
                by_filing[key] = by_filing.get(key, Decimal(0)) + value
                item_by_filing[key] = item
        return [
            ("USD", item_by_filing[key] | {"val": str(value)}) for key, value in by_filing.items()
        ]
    return []


def _observation(
    metric: str,
    item: Mapping[str, Any],
    unit: str,
    *,
    source_url: str,
    cutoff: date,
    as_of: date,
) -> FundamentalObservation | None:
    period_end = _date(item.get("end"))
    filed_at = _filed_at(item.get("filed"))
    period_type = _period_type(item)
    form = str(item.get("form", ""))
    if (
        period_end is None
        or filed_at is None
        or period_type is None
        or period_end < cutoff
        or period_end > as_of
        or filed_at.date() > as_of
        or not (form.startswith("10-Q") or form.startswith("10-K"))
    ):
        return None
    try:
        value = Decimal(str(item["val"]))
    except (KeyError, InvalidOperation):
        return None
    if not value.is_finite():
        return None
    if metric == "capex":
        value = abs(value)
    return FundamentalObservation(
        metric=metric,
        period_end=period_end,
        period_type=period_type,
        value=value,
        unit=unit,
        filed_at=filed_at,
        known_at=filed_at,
        source=SEC_SOURCE,
        source_url=source_url,
    )


def _iter_metric_rows(
    company_facts: Mapping[str, Any],
    metric: str,
    *,
    source_url: str,
    cutoff: date,
    as_of: date,
) -> list[FundamentalObservation]:
    unit_names = _UNITS.get(metric)
    raw_entries = (
        _debt_entries(company_facts)
        if metric == "total_debt"
        else _entries(company_facts, _SIMPLE_TAGS[metric], unit_names=unit_names)
    )
    result: list[FundamentalObservation] = []
    for unit, item in raw_entries:
        if metric == "shares_outstanding" and not (
            str(item.get("form", "")).startswith("10-Q")
            or str(item.get("form", "")).startswith("10-K")
        ):
            continue
        observation = _observation(
            metric,
            item,
            unit,
            source_url=source_url,
            cutoff=cutoff,
            as_of=as_of,
        )
        if observation is not None:
            result.append(observation)
    return result


def _year_to_date_quarters(
    company_facts: Mapping[str, Any],
    metric: str,
    *,
    source_url: str,
    cutoff: date,
    as_of: date,
) -> list[FundamentalObservation]:
    """Derive later quarters from year-to-date totals that share a fiscal-year start.

    10-Q cash-flow statements report only 3-, 6- and 9-month totals, so Q2 and Q3 are the
    differences between consecutive totals. Each total keeps its first filing, and a derived
    quarter is known once both filings are.
    """
    totals: dict[tuple[str, date], dict[date, tuple[datetime, Decimal]]] = defaultdict(dict)
    for unit, item in _entries(company_facts, _SIMPLE_TAGS[metric], unit_names=_UNITS.get(metric)):
        start, end = _date(item.get("start")), _date(item.get("end"))
        filed_at = _filed_at(item.get("filed"))
        form = str(item.get("form", ""))
        if start is None or end is None or filed_at is None:
            continue
        if not (form.startswith("10-Q") or form.startswith("10-K")):
            continue
        if end < cutoff or end > as_of or filed_at.date() > as_of:
            continue
        if not 45 <= (end - start).days + 1 <= 290:
            continue
        try:
            value = Decimal(str(item["val"]))
        except (KeyError, InvalidOperation):
            continue
        if not value.is_finite():
            continue
        if metric == "capex":
            value = abs(value)
        existing = totals[(unit, start)].get(end)
        if existing is None or filed_at < existing[0]:
            totals[(unit, start)][end] = (filed_at, value)

    derived: list[FundamentalObservation] = []
    for (unit, start), by_end in totals.items():
        ends = sorted(by_end)
        for previous_end, end in zip(ends, ends[1:], strict=False):
            if (end - start).days + 1 < 150 or not 70 <= (end - previous_end).days <= 110:
                continue
            previous_filed, previous_value = by_end[previous_end]
            filed_at, value = by_end[end]
            known_at = max(filed_at, previous_filed)
            derived.append(
                FundamentalObservation(
                    metric=metric,
                    period_end=end,
                    period_type="Q",
                    value=value - previous_value,
                    unit=unit,
                    filed_at=known_at,
                    known_at=known_at,
                    source=SEC_SOURCE,
                    source_url=source_url,
                )
            )
    return derived


def _add_q4_observations(
    observations: list[FundamentalObservation],
) -> list[FundamentalObservation]:
    """Derive missing fourth quarters from annual filings and the three reported quarters."""
    by_metric: dict[str, list[FundamentalObservation]] = defaultdict(list)
    for observation in observations:
        by_metric[observation.metric].append(observation)
    derived: list[FundamentalObservation] = []
    for metric in _QUARTERLY_METRICS:
        metric_rows = by_metric.get(metric, [])
        annual_rows = [row for row in metric_rows if row.period_type == "FY"]
        quarters = [row for row in metric_rows if row.period_type == "Q"]
        for annual in annual_rows:
            if any(
                row.period_end == annual.period_end and row.known_at == annual.known_at
                for row in quarters
            ):
                continue
            by_quarter_end: dict[date, FundamentalObservation] = {}
            for row in quarters:
                if (
                    row.period_end < annual.period_end
                    and (annual.period_end - row.period_end).days <= 370
                    and row.known_at <= annual.filed_at
                    and row.unit == annual.unit
                ):
                    old = by_quarter_end.get(row.period_end)
                    if old is None or row.known_at > old.known_at:
                        by_quarter_end[row.period_end] = row
            last_three = sorted(by_quarter_end.values(), key=lambda row: row.period_end)[-3:]
            if len(last_three) != 3 or any(
                not 70 <= (right.period_end - left.period_end).days <= 110
                for left, right in zip(last_three, last_three[1:], strict=False)
            ):
                continue
            if not 70 <= (annual.period_end - last_three[-1].period_end).days <= 110:
                continue
            if any(row.known_at > annual.filed_at for row in last_three):
                continue
            derived.append(
                FundamentalObservation(
                    metric=metric,
                    period_end=annual.period_end,
                    period_type="Q",
                    value=annual.value - sum((row.value for row in last_three), Decimal(0)),
                    unit=annual.unit,
                    filed_at=annual.filed_at,
                    known_at=annual.known_at,
                    source=annual.source,
                    source_url=annual.source_url,
                )
            )
    return observations + derived


def parse_company_facts(
    company_facts: Mapping[str, Any],
    *,
    as_of: date,
    source_url: str | None = None,
) -> list[FundamentalObservation]:
    """Extract supported US-GAAP values with filing dates inside the retention window."""
    cik_value = company_facts.get("cik")
    cik_text = str(cik_value) if isinstance(cik_value, int | str) else ""
    cik = f"{int(cik_text):010d}" if cik_text.isdigit() else "0000000000"
    facts_url = source_url or SEC_COMPANYFACTS_URL.format(cik=cik)
    cutoff = retention_cutoff(as_of)
    metrics = (*_SIMPLE_TAGS.keys(), "total_debt")
    observations = [
        observation
        for metric in metrics
        for observation in _iter_metric_rows(
            company_facts,
            metric,
            source_url=facts_url,
            cutoff=cutoff,
            as_of=as_of,
        )
    ]
    reported_quarters = {
        (row.metric, row.period_end) for row in observations if row.period_type == "Q"
    }
    observations.extend(
        row
        for metric in _QUARTERLY_METRICS
        for row in _year_to_date_quarters(
            company_facts, metric, source_url=facts_url, cutoff=cutoff, as_of=as_of
        )
        if (row.metric, row.period_end) not in reported_quarters
    )
    observations = _add_q4_observations(observations)
    observations.sort(
        key=lambda row: (
            row.metric,
            row.period_end,
            row.period_type,
            row.filed_at,
            row.value,
        )
    )
    return observations


def ticker_lookup(companies: Mapping[str, EdgarCompany], ticker: str) -> EdgarCompany | None:
    return companies.get(_normalize_ticker(ticker))
