"""Official current constituent files for US and Indian broad-market universes."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from wealth.marketdata.sources.http import get
from wealth.marketdata.sources.nse import NseSecurity

SP500_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/institutional/library-content/products/fund-data/etfs/us/"
    "holdings-daily-us-en-spy.xlsx"
)
NIFTY500_ARCHIVE_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
SP500_SOURCE = "SSGA_SPY_HOLDINGS"
NIFTY500_SOURCE = "NSE_NIFTY500_ARCHIVE"
SURVIVORSHIP_LABEL = "current constituents only (survivorship bias)"
_TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]{1,15}$")


@dataclass(frozen=True)
class UniverseSecurity:
    symbol: str
    name: str
    isin: str | None = None
    series: str | None = None


@dataclass(frozen=True)
class Sp500Holdings:
    securities: list[UniverseSecurity]
    as_of: date


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _field(row: dict[str, Any], *names: str) -> str | None:
    normalized = {_key(str(key)): value for key, value in row.items() if key is not None}
    for name in names:
        value = normalized.get(_key(name))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_sp500_as_of(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().removeprefix("As of ").strip()
    for pattern in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, pattern).date()
        except ValueError:
            continue
    return None


def parse_sp500_holdings(content: bytes) -> Sp500Holdings:
    """Read SPY's daily issuer workbook and exclude cash/unassigned positions."""
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        rows = worksheet.iter_rows(values_only=True)
        metadata: dict[str, Any] = {}
        header: list[str] | None = None
        holdings: list[UniverseSecurity] = []
        for row in rows:
            if not any(value is not None for value in row):
                continue
            first = str(row[0] or "").strip().casefold()
            if first in {"fund name:", "ticker symbol:", "holdings:"}:
                metadata[first.rstrip(":")] = row[1] if len(row) > 1 else None
                continue
            values = [str(value).strip() if value is not None else "" for value in row]
            if header is None:
                if "ticker" in {_key(value) for value in values}:
                    header = values
                continue
            raw = dict(zip(header, row, strict=False))
            symbol = _field(raw, "Ticker", "Symbol")
            name = _field(raw, "Name", "Security Name")
            sector = (_field(raw, "Sector") or "").casefold()
            if not symbol or not name or sector == "unassigned":
                continue
            symbol = symbol.upper()
            if not _TICKER_PATTERN.fullmatch(symbol):
                continue
            holdings.append(UniverseSecurity(symbol=symbol, name=name[:160]))
        as_of = _parse_sp500_as_of(metadata.get("holdings"))
        if as_of is None:
            raise ValueError("SPY holdings workbook has no recognized as-of date")
        unique = {item.symbol: item for item in holdings}
        if not unique:
            raise ValueError("SPY holdings workbook has no equity constituents")
        return Sp500Holdings(
            securities=sorted(unique.values(), key=lambda item: item.symbol),
            as_of=as_of,
        )
    finally:
        workbook.close()


def fetch_sp500() -> Sp500Holdings:
    return parse_sp500_holdings(get(SP500_HOLDINGS_URL).content)


def parse_nifty500_csv(text: str) -> list[NseSecurity]:
    """Parse NSE's public index archive and deduplicate its current tickers."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    result: dict[str, NseSecurity] = {}
    for row in reader:
        symbol = _field(row, "Symbol", "Ticker")
        name = _field(row, "Company Name", "Name")
        isin = _field(row, "ISIN Code", "ISIN")
        series = _field(row, "Series")
        if not symbol or not name:
            continue
        symbol = symbol.upper()
        isin_value = isin.upper() if isin and isin != "-" else None
        series_value = series.upper() if series and series != "-" else None
        result[symbol] = NseSecurity(
            symbol=symbol,
            name=name[:160],
            isin=isin_value,
            series=series_value,
            security_type="equity",
            sleeve="equity_india",
            category="equity:india",
        )
    if not result:
        raise ValueError("NSE NIFTY 500 archive CSV has no constituents")
    return sorted(result.values(), key=lambda item: item.symbol)


def fetch_nifty500() -> list[NseSecurity]:
    return parse_nifty500_csv(get(NIFTY500_ARCHIVE_URL).text)
