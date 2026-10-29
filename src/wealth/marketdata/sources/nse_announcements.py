"""Best-effort access to the official NSE corporate-announcement API."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any

import httpx

NSE_ANNOUNCEMENTS_PAGE_URL = (
    "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
)
NSE_ANNOUNCEMENTS_API_URL = "https://www.nseindia.com/api/corporate-announcements"
_REQUEST_INTERVAL_SECONDS = 0.2


class NseAnnouncementsClient:
    """Use NSE's public page session and API, without third-party aggregators."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; WealthCopilotMarketData/1.0)",
                "Accept": "application/json,*/*",
                "Referer": NSE_ANNOUNCEMENTS_PAGE_URL,
            },
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._initialized = False
        self._last_request_at: float | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NseAnnouncementsClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        if self._last_request_at is not None:
            remaining = _REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()
        response = self._client.get(url, **kwargs)
        response.raise_for_status()
        return response

    def announcements(self, symbol: str, *, days: int, as_of: date) -> list[dict[str, str]]:
        if not 1 <= days <= 30:
            raise ValueError("days must be between 1 and 30")
        if not self._initialized:
            self._get(NSE_ANNOUNCEMENTS_PAGE_URL)
            self._initialized = True
        payload = self._get(
            NSE_ANNOUNCEMENTS_API_URL,
            params={"index": "equities", "symbol": symbol.upper()},
        ).json()
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        if not isinstance(payload, list):
            raise ValueError("NSE announcements response must contain a list")
        cutoff = as_of - timedelta(days=days)
        result: list[dict[str, str]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_date = str(row.get("an_dt") or row.get("date") or row.get("broadcastDate") or "")
            try:
                announced_on = date.fromisoformat(raw_date[:10])
            except ValueError:
                for pattern in ("%d-%b-%Y", "%d-%m-%Y"):
                    try:
                        announced_on = datetime.strptime(raw_date[:11].strip(), pattern).date()
                        break
                    except ValueError:
                        announced_on = None
                if announced_on is None:
                    continue
            if not cutoff <= announced_on <= as_of:
                continue
            category = str(row.get("category") or row.get("desc") or row.get("subject") or "")
            details = str(row.get("attchmntText") or row.get("subject") or row.get("desc") or "")
            if category or details:
                result.append(
                    {
                        "symbol": symbol.upper(),
                        "date": announced_on.isoformat(),
                        "category": category[:200],
                        "text": details[:4000],
                    }
                )
        return result
