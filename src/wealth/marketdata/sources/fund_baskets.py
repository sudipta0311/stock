"""Public fund holdings and index-member snapshots used for stock look-through."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from wealth.core.ids import isin_check_digit_ok, isin_from_cusip
from wealth.marketdata.sources import universes
from wealth.marketdata.sources.http import get
from wealth.marketdata.sources.universes import Sp500Holdings, UniverseSecurity

SSGA_EMEA_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/emea/"
    "holdings-daily-emea-en-{ticker}-gy.xlsx"
)
INVESCO_NASDAQ100_URL = (
    "https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/QQQ/holdings/fund"
    "?idType=ticker&productType=ETF"
)
NIFTY_INDEXES = (
    "nifty50",
    "niftynext50",
    "nifty100",
    "niftymidcap150",
    "niftysmallcap250",
    "nifty500",
    "niftylargemidcap250",
    "niftybank",
    "niftyit",
    "niftypharma",
    "niftyfmcg",
    "niftyauto",
)
SSGA_TICKERS = {
    "msci_world": "sppw",
    "msci_acwi": "spyy",
    "msci_acwi_imi": "spyi",
    "msci_em": "spym",
}
_WEIGHT_QUANTUM = Decimal("0.000001")
_PERCENT = Decimal("100")


@dataclass(frozen=True)
class BasketWeight:
    market: str | None
    isin: str | None
    symbol: str | None
    name: str
    weight_pct: Decimal
    sedol: str | None = None


@dataclass(frozen=True)
class BasketSnapshot:
    id: str
    label: str
    as_of: date
    source: str
    source_url: str
    weights_estimated: bool
    weight_source: str | None
    members_total: int
    members_found: int
    weights: tuple[BasketWeight, ...]


@dataclass(frozen=True)
class FundBasketBatch:
    baskets: dict[str, BasketSnapshot]
    errors: dict[str, str]


@dataclass(frozen=True)
class ParsedHoldings:
    as_of: date
    weights: tuple[BasketWeight, ...]


@dataclass(frozen=True)
class NseIndexMember:
    symbol: str
    name: str
    isin: str | None


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _field(row: Mapping[str, Any], *names: str) -> Any:
    normalized = {_key(str(key)): value for key, value in row.items() if key is not None}
    for name in names:
        value = normalized.get(_key(name))
        if value is not None and str(value).strip():
            return value
    return None


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value).replace(",", "").replace("%", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _as_of(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    for pattern in ("%d-%b-%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(cleaned, pattern).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _country_market(value: Any) -> str | None:
    country = " ".join(str(value or "").casefold().split())
    if country in {"india", "in"}:
        return "IN"
    if country in {"united states", "united states of america", "us", "usa"}:
        return "US"
    return None


def parse_ssga_emea_holdings(content: bytes) -> ParsedHoldings:
    """Parse SSGA's public EMEA workbook and combine repeated valid ISIN rows."""
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    try:
        worksheet = next(
            (sheet for sheet in workbook.worksheets if sheet.title.casefold() == "holdings"),
            None,
        )
        if worksheet is None:
            raise ValueError("SSGA workbook has no holdings sheet")
        metadata: dict[str, Any] = {}
        header: list[str] | None = None
        combined: dict[str, tuple[BasketWeight, Decimal]] = {}
        for row in worksheet.iter_rows(values_only=True):
            if not row or not any(value is not None for value in row):
                continue
            first = str(row[0] or "").strip().casefold()
            if first.endswith(":") and first[:-1] in {
                "fund name",
                "isin",
                "ticker symbol",
                "holdings as of",
            }:
                metadata[first[:-1]] = row[1] if len(row) > 1 else None
                continue
            values = [str(value).strip() if value is not None else "" for value in row]
            if header is None:
                if {"isin", "percentoffund", "securityname"}.issubset({_key(v) for v in values}):
                    header = values
                continue
            raw = dict(zip(header, row, strict=False))
            isin = str(_field(raw, "ISIN") or "").strip().upper()
            weight = _decimal(_field(raw, "Percent of Fund"))
            name = str(_field(raw, "Security Name") or "").strip()
            if not isin_check_digit_ok(isin) or weight is None or weight <= 0 or not name:
                continue
            item = BasketWeight(
                market=_country_market(_field(raw, "Trade Country Name")),
                isin=isin,
                symbol=None,
                name=name[:160],
                weight_pct=weight,
                sedol=str(_field(raw, "SEDOL") or "").strip().upper() or None,
            )
            previous = combined.get(isin)
            retained = (
                item if previous is None or (not previous[0].sedol and item.sedol) else previous[0]
            )
            combined[isin] = (retained, weight + previous[1] if previous else weight)
        as_of = _as_of(metadata.get("holdings as of"))
        if as_of is None:
            raise ValueError("SSGA workbook has no recognized holdings date")
        if not combined:
            raise ValueError("SSGA workbook has no valid positive-weight holdings")
        weights = tuple(
            sorted(
                (item for item, _weight in combined.values()),
                key=lambda item: item.isin or "",
            )
        )
        return ParsedHoldings(
            as_of=as_of,
            weights=tuple(
                BasketWeight(
                    market=item.market,
                    isin=item.isin,
                    symbol=None,
                    name=item.name,
                    weight_pct=combined[item.isin or ""][1],
                    sedol=item.sedol,
                )
                for item in weights
            ),
        )
    finally:
        workbook.close()


def parse_invesco_nasdaq100(document: Mapping[str, Any] | bytes | str) -> ParsedHoldings:
    """Parse Invesco QQQ holdings, keeping share rows and summing duplicate tickers."""
    if isinstance(document, bytes):
        raw: Any = json.loads(document)
    elif isinstance(document, str):
        raw = json.loads(document)
    else:
        raw = document
    if not isinstance(raw, Mapping):
        raise ValueError("Invesco response is not an object")
    as_of = _as_of(raw.get("effectiveDate"))
    if as_of is None:
        raise ValueError("Invesco response has no effective date")
    holdings = raw.get("holdings")
    if not isinstance(holdings, list):
        raise ValueError("Invesco response has no holdings list")
    combined: dict[str, tuple[str, Decimal, str | None]] = {}
    for item in holdings:
        if not isinstance(item, Mapping) or str(item.get("securityTypeCode", "")).upper() != "COM":
            continue
        symbol = str(item.get("ticker") or "").strip().upper()
        name = str(item.get("issuerName") or "").strip()
        weight = _decimal(item.get("percentageOfTotalNetAssets"))
        if not symbol or not name or weight is None or weight <= 0:
            continue
        cusip_isin = isin_from_cusip(item.get("cusip"), country="US")
        previous = combined.get(symbol)
        combined[symbol] = (
            name[:160],
            weight + previous[1] if previous else weight,
            cusip_isin or (previous[2] if previous else None),
        )
    if not combined:
        raise ValueError("Invesco response has no positive-weight share holdings")
    return ParsedHoldings(
        as_of=as_of,
        weights=tuple(
            BasketWeight("US", isin, symbol, name, weight)
            for symbol, (name, weight, isin) in sorted(combined.items())
        ),
    )


def parse_nse_index_csv(text: str) -> list[NseIndexMember]:
    """Parse a public NSE index member archive without depending on its row order."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    members: dict[str, NseIndexMember] = {}
    for raw in reader:
        symbol = str(_field(raw, "Symbol") or "").strip().upper()
        name = str(_field(raw, "Company Name") or "").strip()
        isin_value = str(_field(raw, "ISIN Code") or "").strip().upper()
        isin = isin_value if isin_value not in {"", "-"} else None
        if not symbol or not name:
            continue
        members[symbol] = NseIndexMember(symbol, name[:160], isin)
    if not members:
        raise ValueError("NSE index archive has no members")
    return sorted(members.values(), key=lambda item: item.symbol)


def _renormalize(weights: Sequence[BasketWeight]) -> tuple[BasketWeight, ...]:
    positive = [item for item in weights if item.weight_pct > 0]
    total = sum((item.weight_pct for item in positive), Decimal(0))
    if total <= 0:
        raise ValueError("cannot normalize an empty basket")
    normalized: list[BasketWeight] = []
    for item in positive:
        normalized.append(
            BasketWeight(
                market=item.market,
                isin=item.isin,
                symbol=item.symbol,
                name=item.name,
                weight_pct=(item.weight_pct * _PERCENT / total).quantize(
                    _WEIGHT_QUANTUM, rounding=ROUND_HALF_UP
                ),
            )
        )
    rounding_delta = _PERCENT - sum((item.weight_pct for item in normalized), Decimal(0))
    largest = max(range(len(normalized)), key=lambda index: normalized[index].weight_pct)
    item = normalized[largest]
    normalized[largest] = BasketWeight(
        item.market,
        item.isin,
        item.symbol,
        item.name,
        item.weight_pct + rounding_delta,
    )
    return tuple(normalized)


def estimate_nifty_weights(
    members: Sequence[NseIndexMember],
    spym: Sequence[BasketWeight],
    spyi: Sequence[BasketWeight],
) -> tuple[tuple[BasketWeight, ...], int, Decimal]:
    """Estimate NIFTY weights from MSCI EM and IMI, then normalize to 100 percent."""
    em_by_isin = {item.isin: item.weight_pct for item in spym if item.isin}
    imi_by_isin = {item.isin: item.weight_pct for item in spyi if item.isin}
    common_indian = {
        isin
        for isin in em_by_isin.keys() & imi_by_isin.keys()
        if isin is not None and isin.startswith("IN")
    }
    em_total = sum((em_by_isin[isin] for isin in common_indian), Decimal(0))
    imi_total = sum((imi_by_isin[isin] for isin in common_indian), Decimal(0))
    if not common_indian or imi_total <= 0:
        raise ValueError("MSCI India weights have no shared reference members")
    scale = em_total / imi_total
    raw_rows: list[BasketWeight] = []
    found = 0
    for member in members:
        em_weight = em_by_isin.get(member.isin) if member.isin else None
        imi_weight = imi_by_isin.get(member.isin) if member.isin else None
        if em_weight is not None or imi_weight is not None:
            found += 1
        weight = em_weight if em_weight is not None else (imi_weight or Decimal(0)) * scale
        raw_rows.append(BasketWeight("IN", member.isin, member.symbol, member.name, weight))
    positive = [item for item in raw_rows if item.weight_pct > 0]
    if not positive:
        raise ValueError("NSE index has no members in the MSCI reference files")
    normalized = {item.symbol: item for item in _renormalize(positive)}
    rows = tuple(
        normalized.get(
            item.symbol, BasketWeight("IN", item.isin, item.symbol, item.name, Decimal(0))
        )
        for item in raw_rows
    )
    return rows, found, scale


def _snapshot(
    basket_id: str,
    label: str,
    as_of: date,
    source: str,
    source_url: str,
    weights: Sequence[BasketWeight],
    *,
    estimated: bool = False,
    weight_source: str | None = None,
    members_total: int | None = None,
    members_found: int | None = None,
) -> BasketSnapshot:
    return BasketSnapshot(
        id=basket_id,
        label=label,
        as_of=as_of,
        source=source,
        source_url=source_url,
        weights_estimated=estimated,
        weight_source=weight_source,
        members_total=len(weights) if members_total is None else members_total,
        members_found=len(weights) if members_found is None else members_found,
        weights=tuple(weights),
    )


def _sp500_snapshot(
    holdings: Sp500Holdings,
    isin_by_sedol: Mapping[str, str] | None = None,
) -> BasketSnapshot:
    weights = tuple(
        BasketWeight(
            "US",
            (isin_by_sedol or {}).get((item.sedol or "").strip().upper())
            or isin_from_cusip(item.cusip, country="US"),
            item.symbol,
            item.name,
            item.weight_pct,
        )
        for item in holdings.securities
        if item.weight_pct is not None and item.weight_pct > 0
    )
    if not weights:
        raise ValueError("SPY holdings have no positive weights")
    return _snapshot(
        "sp500",
        "S&P 500",
        holdings.as_of,
        "SSGA_SPY_HOLDINGS",
        universes.SP500_HOLDINGS_URL,
        weights,
    )


def _derived_msci(
    basket_id: str,
    label: str,
    source_basket: BasketSnapshot,
    rows: Sequence[BasketWeight],
    *,
    india: bool,
) -> BasketSnapshot:
    selected = [item for item in rows if (item.isin or "").startswith("IN") == india]
    normalized = _renormalize(selected)
    return _snapshot(
        basket_id,
        label,
        source_basket.as_of,
        "DERIVED_FROM_SSGA_EMEA",
        source_basket.source_url,
        normalized,
        weight_source=source_basket.id,
    )


def fetch_fund_baskets(
    on: date,
    *,
    fetch_ssga: Callable[[str], bytes] | None = None,
    fetch_invesco: Callable[[], Mapping[str, Any] | bytes | str] | None = None,
    fetch_nifty_csv: Callable[[str], str] | None = None,
    fetch_nifty500: Callable[[], Sequence[UniverseSecurity]] | None = None,
    fetch_sp500: Callable[[], Sp500Holdings] | None = None,
) -> FundBasketBatch:
    """Build every basket whose official source data is available, keeping per-source errors."""
    ssga_fetch = fetch_ssga or (lambda ticker: get(SSGA_EMEA_URL.format(ticker=ticker)).content)
    invesco_fetch = fetch_invesco or (lambda: get(INVESCO_NASDAQ100_URL).json())
    nifty_csv_fetch = fetch_nifty_csv or universes.fetch_nifty_index_csv
    nifty500_fetch = fetch_nifty500 or universes.fetch_nifty500
    sp500_fetch = fetch_sp500 or universes.fetch_sp500
    baskets: dict[str, BasketSnapshot] = {}
    errors: dict[str, str] = {}
    parsed: dict[str, BasketSnapshot] = {}

    for basket_id, ticker in SSGA_TICKERS.items():
        source_url = SSGA_EMEA_URL.format(ticker=ticker)
        try:
            holdings = parse_ssga_emea_holdings(ssga_fetch(ticker))
            parsed[basket_id] = _snapshot(
                basket_id,
                {
                    "msci_world": "MSCI World",
                    "msci_acwi": "MSCI ACWI",
                    "msci_acwi_imi": "MSCI ACWI IMI",
                    "msci_em": "MSCI Emerging Markets",
                }[basket_id],
                holdings.as_of,
                "SSGA_EMEA_HOLDINGS",
                source_url,
                holdings.weights,
                weight_source="ssga_emea_holdings",
            )
        except Exception as error:
            errors[basket_id] = type(error).__name__
    baskets.update(parsed)

    try:
        sedol_to_isin = {
            weight.sedol: weight.isin
            for basket in parsed.values()
            for weight in basket.weights
            if weight.sedol and weight.isin
        }
        baskets["sp500"] = _sp500_snapshot(sp500_fetch(), sedol_to_isin)
    except Exception as error:
        errors["sp500"] = type(error).__name__

    try:
        parsed_nasdaq = parse_invesco_nasdaq100(invesco_fetch())
        sp500_basket = baskets.get("sp500")
        sp500_by_symbol = {
            weight.symbol: weight.isin
            for weight in (sp500_basket.weights if sp500_basket is not None else ())
            if weight.symbol and weight.isin
        }
        nasdaq_weights = tuple(
            BasketWeight(
                row.market,
                row.isin or sp500_by_symbol.get(row.symbol),
                row.symbol,
                row.name,
                row.weight_pct,
                row.sedol,
            )
            for row in parsed_nasdaq.weights
        )
        baskets["nasdaq100"] = _snapshot(
            "nasdaq100",
            "NASDAQ 100",
            parsed_nasdaq.as_of,
            "INVESCO_QQQ_HOLDINGS",
            INVESCO_NASDAQ100_URL,
            nasdaq_weights,
            weight_source="invesco_qqq_holdings",
        )
    except Exception as error:
        errors["nasdaq100"] = type(error).__name__

    emerging = parsed.get("msci_em")
    if emerging is not None:
        for basket_id, label, india in (
            ("msci_india", "MSCI India", True),
            ("msci_em_ex_india", "Emerging markets excluding India", False),
        ):
            try:
                baskets[basket_id] = _derived_msci(
                    basket_id,
                    label,
                    emerging,
                    emerging.weights,
                    india=india,
                )
            except Exception as error:
                errors[basket_id] = type(error).__name__

    spyi = parsed.get("msci_acwi_imi")
    if emerging is None or spyi is None:
        missing_source = "MissingMSCIWeightSource"
        for basket_id in NIFTY_INDEXES:
            errors[basket_id] = missing_source
        return FundBasketBatch(baskets, errors)

    for index in NIFTY_INDEXES:
        try:
            source_url = universes.NIFTY_ARCHIVE_URL.format(index=index)
            if index == "nifty500":
                records = nifty500_fetch()
                members = [NseIndexMember(item.symbol, item.name, item.isin) for item in records]
            else:
                members = parse_nse_index_csv(nifty_csv_fetch(index))
            weights, found, _scale = estimate_nifty_weights(
                members,
                emerging.weights,
                spyi.weights,
            )
            basket = _snapshot(
                index,
                {
                    "nifty50": "NIFTY 50",
                    "niftynext50": "NIFTY Next 50",
                    "nifty100": "NIFTY 100",
                    "niftymidcap150": "NIFTY Midcap 150",
                    "niftysmallcap250": "NIFTY Smallcap 250",
                    "nifty500": "NIFTY 500",
                    "niftylargemidcap250": "NIFTY LargeMidcap 250",
                    "niftybank": "NIFTY Bank",
                    "niftyit": "NIFTY IT",
                    "niftypharma": "NIFTY Pharma",
                    "niftyfmcg": "NIFTY FMCG",
                    "niftyauto": "NIFTY Auto",
                }[index],
                on,
                "NSE_ARCHIVE+MSCI",
                source_url,
                weights,
                estimated=True,
                weight_source="msci_em+imi",
                members_total=len(members),
                members_found=found,
            )
            baskets[index] = basket
        except Exception as error:
            errors[index] = type(error).__name__
    return FundBasketBatch(baskets, errors)
