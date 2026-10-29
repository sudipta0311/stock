"""NSE bhavcopy and securities-master parsers and HTTP adapters."""

from __future__ import annotations

import csv
import io
import re
import warnings
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import httpx

from wealth.marketdata.sources.http import get

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_ETF_LIST_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
NSE_SOURCE = "NSE"
NSE_SOURCE_URL = "https://www.nseindia.com/all-reports"
_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)
_LEGACY_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
)


@dataclass(frozen=True)
class NseClose:
    symbol: str
    close: Decimal
    as_of: date
    isin: str | None = None
    series: str | None = None
    volume: int | None = None
    traded_value: Decimal | None = None


@dataclass(frozen=True)
class NseSecurity:
    symbol: str
    name: str
    isin: str | None
    series: str | None
    security_type: str
    sleeve: str | None = None
    category: str | None = None
    sector: str = "Unclassified"
    industry: str = "Unclassified"


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _find(row: dict[str, str], *names: str) -> str | None:
    normalized = {_key(key): value.strip() for key, value in row.items() if key}
    return next((normalized[name] for name in names if normalized.get(name)), None)


def _date(value: str) -> date:
    value = value.strip()
    for pattern in ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    raise ValueError("unrecognized date")


def _reader(text: str) -> tuple[csv.DictReader, list[str]]:
    stream = io.StringIO(text.lstrip("\ufeff"))
    reader = csv.DictReader(stream)
    return reader, reader.fieldnames or []


def parse_bhavcopy(text: str, *, as_of: date | None = None) -> list[NseClose]:
    """Parse legacy NSE CM bhavcopy CSV or the UDiFF common bhavcopy CSV."""
    if not text.lstrip("\ufeff").strip():
        return []
    reader, fields = _reader(text)
    normalized_fields = {_key(field) for field in fields}
    if not (normalized_fields & {"symbol", "tckrsymb"}) or not (
        normalized_fields & {"closeprice", "close", "clspric"}
    ):
        warnings.warn(
            "NSE bhavcopy has an unrecognized header; rows were skipped",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    rows: list[NseClose] = []
    for line, raw in enumerate(reader, start=2):
        try:
            symbol = _find(raw, "symbol", "tckrsymb")
            close_text = _find(raw, "closeprice", "close", "clspric")
            date_text = _find(raw, "date1", "traddt", "bizdt", "date")
            isin = _find(raw, "isin")
            series = _find(raw, "series", "sctysrs")
            volume_text = _find(raw, "totaltradedquantity", "ttltradgvol", "tottrdqty")
            traded_value_text = _find(raw, "totaltradedvalue", "ttltrfval", "tottrdval")
            turnover_lacs_text = _find(raw, "turnoverlacs", "ttltrdvalinlacs")
            row_date = _date(date_text) if date_text else as_of
            if not symbol or not close_text or row_date is None:
                raise ValueError("missing symbol, close or date")
            close = Decimal(close_text.replace(",", ""))
            if not close.is_finite() or close <= 0:
                raise ValueError("invalid close")
            volume = int(volume_text.replace(",", "")) if volume_text else None
            if volume is not None and volume < 0:
                raise ValueError("invalid volume")
            traded_value = (
                Decimal(traded_value_text.replace(",", ""))
                if traded_value_text
                else Decimal(turnover_lacs_text.replace(",", "")) * Decimal(100_000)
                if turnover_lacs_text
                else None
            )
            if traded_value is not None and (not traded_value.is_finite() or traded_value < 0):
                raise ValueError("invalid traded value")
            rows.append(
                NseClose(
                    symbol.upper(),
                    close,
                    row_date,
                    isin.upper() if isin and isin != "-" else None,
                    series.upper() if series and series != "-" else None,
                    volume,
                    traded_value,
                )
            )
        except (InvalidOperation, ValueError):
            warnings.warn(f"NSE bhavcopy row {line} was skipped", RuntimeWarning, stacklevel=2)
    return rows


def parse_equity_list(text: str) -> list[NseSecurity]:
    """Parse NSE's public equity master CSV; malformed rows produce warnings."""
    if not text.strip():
        return []
    reader, fields = _reader(text)
    normalized_fields = {_key(field) for field in fields}
    if not ({"symbol", "tckrsymb"} & normalized_fields):
        warnings.warn(
            "NSE equity list has an unrecognized header; rows were skipped",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    rows: list[NseSecurity] = []
    for line, raw in enumerate(reader, start=2):
        try:
            symbol = _find(raw, "symbol", "tckrsymb")
            name = _find(raw, "nameofcompany", "companyname", "fininstrmnm", "name")
            isin = _find(raw, "isinnumber", "isin", "isinno")
            series = _find(raw, "series", "sctysrs")
            if not symbol or not name:
                raise ValueError("missing symbol or name")
            isin = isin.upper() if isin and isin != "-" else None
            series = series.upper() if series and series != "-" else None
            security_type = (
                "etf"
                if "etf" in name.casefold() or (series or "").casefold() == "etf"
                else "equity"
            )
            rows.append(NseSecurity(symbol.upper(), name.strip(), isin, series, security_type))
        except ValueError:
            warnings.warn(f"NSE equity row {line} was skipped", RuntimeWarning, stacklevel=2)
    return rows


def _classify_etf(underlying_asset: str, underlying_type: str, underlying_key: str) -> str:
    value = f"{underlying_asset} {underlying_type} {underlying_key}".casefold()
    if "gold" in value:
        return "gold"
    if any(term in value for term in ("debt", "liquid", "gilt", "bond", "government", "overnight")):
        return "debt"
    is_domestic_bse = "bse" in value
    if any(term in value for term in ("nasdaq", "global", "overseas", "international")) or (
        "s&p" in value and not is_domestic_bse
    ):
        return "equity_global"
    return "equity_india"


def parse_etf_list(text: str) -> list[NseSecurity]:
    """Parse NSE's ETF master and classify each fund to a policy sleeve."""
    if not text.strip():
        return []
    reader, fields = _reader(text)
    normalized_fields = {_key(field) for field in fields}
    if "symbol" not in normalized_fields:
        warnings.warn(
            "NSE ETF list has an unrecognized header; rows were skipped",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    rows: list[NseSecurity] = []
    for line, raw in enumerate(reader, start=2):
        try:
            symbol = _find(raw, "symbol", "tckrsymb")
            name = _find(raw, "securityname", "nameofcompany", "companyname", "name")
            isin = _find(raw, "isinnumber", "isin", "isinno")
            series = _find(raw, "series", "sctysrs")
            underlying_asset = _find(raw, "underlyingasset") or ""
            underlying_type = _find(raw, "etfunderlying") or ""
            underlying_key = _find(raw, "underlyingkey") or ""
            if not symbol or not name:
                raise ValueError("missing symbol or security name")
            isin = isin.upper() if isin and isin != "-" else None
            series = series.upper() if series and series != "-" else None
            sleeve = _classify_etf(underlying_asset, underlying_type, underlying_key)
            rows.append(
                NseSecurity(
                    symbol=symbol.upper(),
                    name=name.strip(),
                    isin=isin,
                    series=series,
                    security_type="etf",
                    sleeve=sleeve,
                    category=f"etf:{sleeve}",
                )
            )
        except ValueError:
            warnings.warn(f"NSE ETF row {line} was skipped", RuntimeWarning, stacklevel=2)
    return rows


def _download_bhavcopy(on: date) -> str | None:
    if on.weekday() >= 5:
        return None
    archive_url = _BHAVCOPY_URL.format(date=on.strftime("%Y%m%d"))
    try:
        response = get(archive_url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_name = next(name for name in archive.namelist() if name.casefold().endswith(".csv"))
            return archive.read(csv_name).decode("utf-8-sig")
    except (httpx.HTTPStatusError, zipfile.BadZipFile, StopIteration):
        # The legacy endpoint remains a fallback while NSE archive formats transition.
        legacy_url = _LEGACY_BHAVCOPY_URL.format(date=on.strftime("%d%m%Y"))
        try:
            return get(legacy_url).text
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                return None
            raise


def fetch_bhavcopy(on: date) -> list[NseClose]:
    """Fetch one official NSE trading-day archive and parse its closes."""
    text = _download_bhavcopy(on)
    return [] if text is None else parse_bhavcopy(text, as_of=on)


def fetch_equity_list() -> list[NseSecurity]:
    """Fetch the latest NSE equity securities master."""
    return parse_equity_list(get(NSE_EQUITY_LIST_URL).text)


def fetch_etf_list() -> list[NseSecurity]:
    """Fetch and parse NSE's ETF securities master."""
    return parse_etf_list(get(NSE_ETF_LIST_URL).text)
