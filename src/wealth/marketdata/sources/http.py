"""Small HTTP helpers for market-data adapters."""

from __future__ import annotations

import time

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; WealthCopilotMarketData/1.0)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/csv,text/plain,text/html,*/*"}


def get(url: str, *, timeout: float = 30.0, retries: int = 3) -> httpx.Response:
    """Fetch a public source with a browser-like user agent and bounded backoff."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
            if response.status_code >= 500 and attempt + 1 < retries:
                time.sleep(0.25 * (2**attempt))
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(0.25 * (2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("HTTP retry loop ended without a response")
