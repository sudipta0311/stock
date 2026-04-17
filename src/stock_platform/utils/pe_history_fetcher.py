from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from stock_platform.data.db import connect_database

_log = logging.getLogger(__name__)


CACHE_TTL_DAYS = 7  # refresh PE history weekly

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
}


def get_pe_history(
    symbol: str,
    db_path: str,
    current_pe: float | None = None,
    neon_database_url: str = "",
) -> dict[str, Any]:
    """
    Fetch historical PE data for a stock.
    Tries four sources in sequence: Tickertape → Screener.in → Wisesheets → yfinance.
    Caches result in SQLite for 7 days.

    Returns dict with median_5yr, median_10yr, pe_low, pe_high, source,
    fetched_at — or empty dict if all sources fail.
    """
    clean = symbol.upper().replace(".NS", "").replace(".BO", "")

    cached = _get_from_cache(clean, db_path, neon_database_url)
    if cached:
        return cached

    # Source order: Tickertape (best bot tolerance) → Screener (richest when it works)
    # → Wisesheets → yfinance (computed from price × earnings)
    result = (
        _fetch_from_tickertape_pe(clean)
        or _fetch_from_screener(clean)
        or _fetch_from_wisesheets(clean)
        or _fetch_from_yfinance(clean)
    )

    if result:
        _save_to_cache(clean, result, db_path, neon_database_url)
        _log.info("PE history cached: %s | median=%.1f via %s",
                  clean, result.get("median_5yr") or 0, result.get("source", "?"))
        return result

    _log.error("PE history: all sources failed for %s", clean)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: Screener.in  (best for Indian stocks)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_screener(symbol: str) -> dict[str, Any]:
    """
    Fetch PE history from Screener.in chart API.

    Screener requires a real browser session with cookies before the API call
    will return data. Two steps:
      1. GET base URL to prime session cookies
      2. GET company page to extract the numeric company ID
      3. Hit /api/company/{id}/chart/?q=Price+to+Earning to get weekly PE values
    Falls back to standalone (non-consolidated) page if consolidated returns 403.
    """
    import random

    try:
        session = requests.Session()
        session.headers.update({
            **_BROWSER_HEADERS,
            "Referer": "https://www.screener.in/",
        })

        # Prime session cookies.
        session.get("https://www.screener.in/", timeout=10)
        time.sleep(random.uniform(1.5, 3.0))

        # Fetch company page to extract the numeric company ID.
        company_page = None
        for suffix in ("consolidated/", ""):
            url = f"https://www.screener.in/company/{symbol}/{suffix}"
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                company_page = r.text
                break
            if r.status_code == 403:
                _log.warning("Screener 403 for %s (%s)", symbol, suffix or "standalone")
                time.sleep(random.uniform(1.0, 2.0))

        if not company_page:
            return {}

        # Extract numeric company ID from the /api/company/{id}/add/ link.
        m = re.search(r"/api/company/(\d+)/", company_page)
        if not m:
            _log.warning("Screener: could not find company ID for %s", symbol)
            return {}

        company_id = m.group(1)
        chart_url = (
            f"https://www.screener.in/api/company/{company_id}/chart/"
            "?q=Price+to+Earning&days=1825"
        )
        session.headers["Accept"] = "application/json"
        session.headers["X-Requested-With"] = "XMLHttpRequest"

        cr = session.get(chart_url, timeout=15)
        if cr.status_code != 200:
            _log.warning("Screener chart API %s for %s", cr.status_code, symbol)
            return {}

        datasets = cr.json().get("datasets", [])
        if not datasets:
            return {}

        pe_values: list[float] = []
        for row in datasets[0].get("values", []):
            try:
                val = float(row[1])
                if 1 < val < 1000:
                    pe_values.append(val)
            except (TypeError, ValueError, IndexError):
                pass

        if len(pe_values) >= 3:
            return _compute_stats(pe_values, "screener.in")

        _log.warning("Screener: insufficient PE data for %s (%d values)", symbol, len(pe_values))
        return {}

    except Exception as exc:
        _log.error("Screener PE history error for %s: %r", symbol, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: Tickertape  (better bot tolerance than Screener for PE history)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_tickertape_pe(symbol: str) -> dict[str, Any]:
    """
    Fetch 5-year PE history from Tickertape's public ratios API.
    Search endpoint returns stock SID; ratios endpoint returns time-series PE values.
    """
    try:
        headers = {**_BROWSER_HEADERS, "Accept": "application/json"}

        sr = requests.get(
            f"https://api.tickertape.in/search?text={symbol}&type=stock",
            headers=headers,
            timeout=10,
        )
        if sr.status_code != 200:
            return {}

        stocks = sr.json().get("data", {}).get("stocks", [])
        if not stocks:
            return {}

        sid = stocks[0].get("sid")
        if not sid:
            return {}

        pr = requests.get(
            f"https://api.tickertape.in/stocks/{sid}/ratios?indicators=pe&duration=5y",
            headers=headers,
            timeout=15,
        )
        if pr.status_code != 200:
            return {}

        points = pr.json().get("data", {}).get("pe", {}).get("values", [])
        if not points:
            return {}

        pe_vals = [
            float(p["value"])
            for p in points
            if p.get("value") is not None and 1 < float(p["value"]) < 1000
        ]
        if len(pe_vals) < 3:
            return {}

        return _compute_stats(pe_vals, "tickertape")

    except Exception as exc:
        _log.warning("Tickertape PE failed for %s: %r", symbol, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: Wisesheets  (free, no auth, structured text)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_wisesheets(symbol: str) -> dict[str, Any]:
    """
    Wisesheets provides PE history summary text at wisesheets.io/pe-ratio/SYMBOL.NS
    """
    try:
        url = f"https://www.wisesheets.io/pe-ratio/{symbol}.NS"
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=12)
        if resp.status_code != 200:
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text()

        result: dict[str, Any] = {}

        mean_match = re.search(
            r"mean historical PE Ratio.*?is\s+([\d.]+)", text, re.IGNORECASE
        )
        median_match = re.search(
            r"median.*?was\s+([\d.]+)", text, re.IGNORECASE
        )
        low_match = re.search(
            r"lowest.*?at\s+([\d.]+)", text, re.IGNORECASE
        )
        high_match = re.search(
            r"highest.*?at\s+([\d.]+)", text, re.IGNORECASE
        )

        if mean_match:
            result["median_10yr"] = float(mean_match.group(1).rstrip('.'))
        if median_match:
            result["median_5yr"] = float(median_match.group(1).rstrip('.'))
        if low_match:
            result["pe_low"] = float(low_match.group(1).rstrip('.'))
        if high_match:
            result["pe_high"] = float(high_match.group(1).rstrip('.'))

        if result.get("median_10yr") or result.get("median_5yr"):
            result["source"] = "wisesheets.io"
            result["fetched_at"] = date.today().isoformat()
            return result

        return {}

    except Exception as exc:
        _log.error("Wisesheets PE error for %s: %r", symbol, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: yfinance — computed from price history + quarterly earnings
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_yfinance(symbol: str) -> dict[str, Any]:
    """
    Compute historical PE from yfinance 5-year price history and quarterly income statement.
    Uses income_stmt (Net Income + sharesOutstanding) instead of deprecated quarterly_earnings.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")

        hist = ticker.history(period="5y")
        if hist.empty:
            _log.warning("yfinance PE: no price history for %s", symbol)
            return {}

        # Use quarterly_income_stmt (Net Income row) — quarterly_earnings is deprecated
        income = None
        try:
            income = ticker.quarterly_income_stmt
        except Exception:
            pass
        if income is None or income.empty or "Net Income" not in income.index:
            _log.warning("yfinance PE: no quarterly income stmt for %s", symbol)
            return {}

        info = ticker.info or {}
        shares = (
            info.get("sharesOutstanding")
            or info.get("impliedSharesOutstanding")
            or info.get("floatShares")
        )
        if not shares or float(shares) <= 0:
            _log.warning("yfinance PE: no shares outstanding for %s", symbol)
            return {}

        import pandas as pd

        shares = float(shares)
        net_income_series = income.loc["Net Income"]

        pe_values: list[float] = []
        for dt, net_income in net_income_series.items():
            try:
                if pd.isna(net_income) or float(net_income) <= 0:
                    continue
                annualized_ni = float(net_income) * 4
                eps = annualized_ni / shares
                if eps <= 0:
                    continue
                date_str = str(dt)[:10]
                nearby = hist[hist.index.strftime("%Y-%m-%d") >= date_str].head(1)
                if not nearby.empty:
                    price = float(nearby["Close"].iloc[0])
                    pe = price / eps
                    if 0 < pe < 500:
                        pe_values.append(pe)
            except Exception:
                pass

        if len(pe_values) >= 4:
            return _compute_stats(pe_values, "yfinance computed")

        _log.warning("yfinance PE: insufficient PE data points (%d) for %s", len(pe_values), symbol)
        return {}

    except Exception as exc:
        _log.error("yfinance PE history error for %s: %r", symbol, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# STATS: compute median / low / high from a list of PE values
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stats(pe_values: list[float], source: str) -> dict[str, Any]:
    """
    Compute median, low, high from a list of PE values.
    Filters values outside 1–500 before computing.
    """
    clean = sorted(v for v in pe_values if 1 < v < 500)
    if len(clean) < 3:
        return {}

    n = len(clean)
    median = clean[n // 2]
    # Treat the upper half as "recent 5yr" if we have enough data points
    if n >= 8:
        recent = clean[n // 2:]
        med_5yr = recent[len(recent) // 2]
    else:
        med_5yr = median

    return {
        "median_5yr": round(med_5yr, 2),
        "median_10yr": round(median, 2),
        "pe_low": round(min(clean), 2),
        "pe_high": round(max(clean), 2),
        "data_points": n,
        "source": source,
        "fetched_at": date.today().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SQLite CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _get_from_cache(symbol: str, db_path: str, neon_database_url: str = "") -> dict[str, Any] | None:
    """Return cached PE history if still fresh (within CACHE_TTL_DAYS)."""
    conn = None
    try:
        conn = connect_database(db_path, neon_url=neon_database_url or None)
        row = conn.execute(
            "SELECT data, fetched_at FROM pe_history_cache WHERE symbol = ?",
            (symbol,),
        ).fetchone()

        if not row:
            return None

        raw_date = row["fetched_at"]
        if isinstance(raw_date, str):
            fetched = datetime.strptime(raw_date, "%Y-%m-%d").date()
        else:
            fetched = raw_date  # psycopg2 may return a date object

        if (date.today() - fetched).days > CACHE_TTL_DAYS:
            _log.warning("PE cache stale for %s (%dd) - refreshing", symbol, (date.today() - fetched).days)
            return None

        data = json.loads(row["data"])
        return data

    except Exception as exc:
        _log.warning("PE cache READ ERROR for %s: %r", symbol, exc)
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _save_to_cache(symbol: str, data: dict[str, Any], db_path: str, neon_database_url: str = "") -> None:
    """Upsert PE history into the pe_history_cache table."""
    try:
        conn = connect_database(db_path, neon_url=neon_database_url or None)
        conn.execute(
            """
            INSERT INTO pe_history_cache (symbol, data, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                data       = excluded.data,
                fetched_at = excluded.fetched_at
            """,
            (symbol, json.dumps(data), date.today().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        _log.error("PE cache save error for %s: %r", symbol, exc)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND PRE-FETCH
# ─────────────────────────────────────────────────────────────────────────────

def prefetch_pe_history_for_universe(
    symbols: list[str],
    db_path: str,
    neon_database_url: str = "",
) -> dict[str, int]:
    """
    Pre-fetch PE history for a list of stock symbols.
    Skips symbols that are already cached and fresh.
    Uses a random 2-4 s inter-request delay — Screener.in and Tickertape both
    rate-limit aggressive scrapers; the randomised delay avoids pattern detection.
    Intended to run in a background thread.

    Returns {"saved": N, "skipped": N, "failed": N}.
    """
    import random

    _log.info("Pre-fetching PE history for %d stocks...", len(symbols))
    saved = 0
    skipped = 0
    failed = 0
    for i, symbol in enumerate(symbols):
        clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "")
        if _get_from_cache(clean_symbol, db_path, neon_database_url):
            skipped += 1
            continue
        result = get_pe_history(symbol, db_path, neon_database_url=neon_database_url)
        if result:
            saved += 1
        else:
            failed += 1
        attempted = saved + failed
        if attempted % 5 == 0:
            _log.info(
                "PE pre-fetch: %d saved, %d failed (%d/%d processed)",
                saved, failed, i + 1, len(symbols),
            )
        time.sleep(random.uniform(2.0, 4.0))
    _log.info(
        "PE history pre-fetch complete — %d saved / %d failed / %d already cached / %d total",
        saved, failed, skipped, len(symbols),
    )
    return {"saved": saved, "skipped": skipped, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_pe_historical_context(
    symbol: str,
    current_pe: float | None,
    db_path: str,
    neon_database_url: str = "",
) -> dict[str, Any]:
    """
    Complete PE context computation with real-time history fetch.
    Fetches history from Tickertape → Screener → Wisesheets → yfinance, caches 7 days.
    Returns a pe_context dict compatible with the LLM prompt injector.
    """
    if not current_pe or current_pe <= 0:
        return {
            "pe_current": None,
            "pe_5yr_median": None,
            "pe_vs_median_pct": None,
            "pe_signal": "NEUTRAL",
            "pe_assessment": "PE data unavailable",
        }

    history = get_pe_history(symbol, db_path, current_pe, neon_database_url=neon_database_url)

    # Use median_5yr if available, fall back to median_10yr (Wisesheets often only has the mean)
    _median_val = history.get("median_5yr") or history.get("median_10yr") if history else None

    if not history or not _median_val:
        return {
            "pe_current": current_pe,
            "pe_5yr_median": None,
            "pe_vs_median_pct": None,
            "pe_signal": "NO_HISTORY",
            "pe_assessment": (
                f"PE {current_pe:.1f}x — historical data not yet cached. "
                "Run again after data loads."
            ),
            "pe_context_note": (
                "Sources tried: Tickertape, Screener.in, Wisesheets, yfinance. Will retry on next run."
            ),
        }

    median = float(_median_val)
    pe_low = history.get("pe_low")
    pe_high = history.get("pe_high")
    source = history.get("source", "")
    vs_pct = round((current_pe - median) / median * 100, 1)

    if vs_pct <= -30:
        signal = "VERY_CHEAP_VS_HISTORY"
        assessment = (
            f"PE {current_pe:.1f}x is {abs(vs_pct):.0f}% BELOW "
            f"5yr median {median:.1f}x. "
            "HISTORICALLY CHEAP — strong contrarian valuation support."
        )
    elif vs_pct <= -15:
        signal = "CHEAP_VS_HISTORY"
        assessment = (
            f"PE {current_pe:.1f}x is {abs(vs_pct):.0f}% below "
            f"5yr median {median:.1f}x. "
            "Below historical average — valuation support present."
        )
    elif vs_pct <= 10:
        signal = "FAIR_VS_HISTORY"
        assessment = (
            f"PE {current_pe:.1f}x is near 5yr median {median:.1f}x "
            f"({vs_pct:+.0f}%). Fair valued."
        )
    elif vs_pct <= 40:
        signal = "SLIGHT_PREMIUM"
        assessment = (
            f"PE {current_pe:.1f}x is {vs_pct:.0f}% above "
            f"5yr median {median:.1f}x. Moderate premium to history."
        )
    else:
        signal = "EXPENSIVE_VS_HISTORY"
        assessment = (
            f"PE {current_pe:.1f}x is {vs_pct:.0f}% ABOVE "
            f"5yr median {median:.1f}x. "
            "Significantly expensive vs own history."
        )

    # Near all-time PE low — flag it
    if pe_low and current_pe <= pe_low * 1.10:
        assessment += (
            f" Stock is near its all-time PE low of {pe_low:.1f}x — "
            "potential floor or capitulation."
        )

    if pe_low and pe_high:
        note = (
            f"Range: {pe_low:.1f}x–{pe_high:.1f}x | "
            f"5yr median: {median:.1f}x | "
            f"Source: {source} | "
            f"Cached: {history.get('fetched_at', '')}"
        )
    else:
        note = f"5yr median: {median:.1f}x | Source: {source}"

    return {
        "pe_current": current_pe,
        "pe_5yr_median": median,
        "pe_vs_median_pct": vs_pct,
        "pe_signal": signal,
        "pe_assessment": assessment,
        "pe_context_note": note,
    }
