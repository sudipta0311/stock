"""Free end-of-day closes from Alpha Vantage's global daily series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

from wealth.marketdata.sources import http

API_URL = "https://www.alphavantage.co/query"
SOURCE_URL = "https://www.alphavantage.co/documentation/"
TERMS_URL = "https://www.alphavantage.co/terms_of_service/"


@dataclass(frozen=True)
class DailyClose:
    """One positive, dated close returned by the daily time-series endpoint."""

    symbol: str
    as_of: date
    close: Decimal


class RateLimitResponse(ValueError):
    """Alpha Vantage returned a free-tier rate-limit response."""


class UnknownSymbolResponse(ValueError):
    """Alpha Vantage did not recognize the requested symbol."""


class NoSeriesResponse(ValueError):
    """Alpha Vantage returned no usable daily series."""


def parse_daily_series_points(payload: object, symbol: str) -> tuple[DailyClose, ...]:
    """Parse every valid positive close from a recorded daily-series response."""
    if not isinstance(payload, dict):
        raise NoSeriesResponse("Alpha Vantage returned an invalid response")
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series:
        raise NoSeriesResponse("Alpha Vantage returned no daily time series")
    valid: list[DailyClose] = []
    for raw_date, values in series.items():
        if not isinstance(raw_date, str) or not isinstance(values, dict):
            continue
        try:
            as_of = date.fromisoformat(raw_date)
            close = Decimal(str(values["4. close"]))
        except (KeyError, InvalidOperation, ValueError):
            continue
        if close.is_finite() and close > 0:
            valid.append(DailyClose(symbol=symbol, as_of=as_of, close=close))
    if not valid:
        raise NoSeriesResponse("Alpha Vantage returned no valid daily close")
    return tuple(sorted(valid, key=lambda item: item.as_of))


def parse_daily_series(payload: object, symbol: str) -> DailyClose:
    """Parse the newest valid close from a recorded daily-series response."""
    return parse_daily_series_points(payload, symbol)[-1]


def _fetch_daily_payload(
    symbol: str,
    api_key: str,
    *,
    timeout: float,
    retries: int,
) -> object:
    if not api_key.strip():
        raise ValueError("Alpha Vantage API key is not configured")
    url = (
        f"{API_URL}?function=TIME_SERIES_DAILY&symbol={quote(symbol, safe='')}&outputsize=compact"
        f"&apikey={quote(api_key, safe='')}"
    )
    response = http.get(url, timeout=timeout, retries=retries)
    try:
        payload = response.json()
    except ValueError as error:
        raise NoSeriesResponse("Alpha Vantage returned invalid JSON") from error
    if isinstance(payload, dict):
        if "Note" in payload or "Information" in payload:
            raise RateLimitResponse("Alpha Vantage request was rate limited")
        if "Error Message" in payload:
            raise UnknownSymbolResponse("Alpha Vantage did not recognize the symbol")
    return payload


def fetch_daily_series(
    symbol: str,
    api_key: str,
    *,
    timeout: float = 30.0,
    retries: int = 3,
) -> tuple[DailyClose, ...]:
    """Fetch the compact global daily series without exposing the key in errors."""
    payload = _fetch_daily_payload(symbol, api_key, timeout=timeout, retries=retries)
    return parse_daily_series_points(payload, symbol)


def fetch_daily(
    symbol: str,
    api_key: str,
    *,
    timeout: float = 30.0,
    retries: int = 3,
) -> DailyClose:
    """Fetch one compact global daily series without exposing the key in errors."""
    payload = _fetch_daily_payload(symbol, api_key, timeout=timeout, retries=retries)
    return parse_daily_series(payload, symbol)
