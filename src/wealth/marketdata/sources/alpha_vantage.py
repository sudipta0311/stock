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


def parse_daily_series(payload: object, symbol: str) -> DailyClose:
    """Parse the newest valid close from a recorded daily-series response."""
    if not isinstance(payload, dict):
        raise ValueError("Alpha Vantage returned an invalid response")
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series:
        raise ValueError("Alpha Vantage returned no daily time series")
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
        raise ValueError("Alpha Vantage returned no valid daily close")
    return max(valid, key=lambda item: item.as_of)


def fetch_daily(symbol: str, api_key: str) -> DailyClose:
    """Fetch one compact global daily series without exposing the key in errors."""
    if not api_key.strip():
        raise ValueError("Alpha Vantage API key is not configured")
    url = (
        f"{API_URL}?function=TIME_SERIES_DAILY&symbol={quote(symbol, safe='')}&outputsize=compact"
        f"&apikey={quote(api_key, safe='')}"
    )
    response = http.get(url)
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("Alpha Vantage returned invalid JSON") from error
    if isinstance(payload, dict) and any(
        key in payload for key in ("Note", "Information", "Error Message")
    ):
        raise ValueError("Alpha Vantage did not return a daily series")
    return parse_daily_series(payload, symbol)
