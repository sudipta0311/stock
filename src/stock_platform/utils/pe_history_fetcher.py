from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from stock_platform.data.db import connect_database


CACHE_TTL_DAYS = 7  # refresh PE history weekly

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-research/1.0)"}


def get_pe_history(
    symbol: str,
    db_path: str,
    current_pe: float | None = None,
    neon_database_url: str = "",
) -> dict[str, Any]:
    """
    Fetch historical PE data for a stock.
    Tries three sources in sequence: Screener.in → Wisesheets → yfinance.
    Caches result in SQLite for 7 days.

    Returns dict with median_5yr, median_10yr, pe_low, pe_high, source,
    fetched_at — or empty dict if all sources fail.
    """
    clean = symbol.upper().replace(".NS", "").replace(".BO", "")

    cached = _get_from_cache(clean, db_path, neon_database_url)
    if cached:
        return cached

    result = (
        _fetch_from_screener(clean)
        or _fetch_from_wisesheets(clean)
        or _fetch_from_yfinance(clean)
    )

    if result:
        _save_to_cache(clean, result, db_path, neon_database_url)
        return result

    print(f"PE history: all sources failed for {clean}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: Screener.in  (best for Indian stocks)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_screener(symbol: str) -> dict[str, Any]:
    """
    Fetch PE ratio history from Screener.in annual ratios table.
    Tries consolidated first, falls back to standalone.
    """
    try:
        for suffix in ("consolidated/", ""):
            url = f"https://www.screener.in/company/{symbol}/{suffix}"
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                break
        else:
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        pe_values: list[float] = []

        # Method A: #ratios section — annual PE row
        ratio_section = soup.select_one("#ratios")
        if ratio_section:
            for table in ratio_section.select("table"):
                for row in table.select("tbody tr"):
                    label_cell = row.select_one("td:first-child")
                    if not label_cell or "P/E" not in label_cell.text:
                        continue
                    for cell in row.select("td")[1:]:
                        try:
                            val = float(cell.text.strip().replace(",", ""))
                            if 0 < val < 1000:
                                pe_values.append(val)
                        except (ValueError, AttributeError):
                            pass

        # Method B: #ten-year-summary section
        if not pe_values:
            ten_yr = soup.select_one("#ten-year-summary")
            if ten_yr:
                for row in ten_yr.select("tr"):
                    if "P/E" not in row.text:
                        continue
                    for cell in row.select("td"):
                        try:
                            val = float(cell.text.strip().replace(",", ""))
                            if 0 < val < 1000:
                                pe_values.append(val)
                        except (ValueError, AttributeError):
                            pass

        if len(pe_values) >= 3:
            return _compute_stats(pe_values, "screener.in")

        return {}

    except Exception as exc:
        print(f"Screener PE history error for {symbol}: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: Wisesheets  (free, no auth, structured text)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_wisesheets(symbol: str) -> dict[str, Any]:
    """
    Wisesheets provides PE history summary text at wisesheets.io/pe-ratio/SYMBOL.NS
    """
    try:
        url = f"https://www.wisesheets.io/pe-ratio/{symbol}.NS"
        resp = requests.get(url, headers=HEADERS, timeout=12)
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
        print(f"Wisesheets PE error for {symbol}: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: yfinance — computed from price history + quarterly earnings
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_yfinance(symbol: str) -> dict[str, Any]:
    """
    Compute historical PE from yfinance 5-year price history and quarterly earnings.
    Less accurate than Screener but always available.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")

        hist = ticker.history(period="5y")
        if hist.empty:
            return {}

        earnings = ticker.quarterly_earnings
        if earnings is None or earnings.empty:
            return {}

        pe_values: list[float] = []
        for idx, row in earnings.iterrows():
            try:
                eps_annual = float(row.get("Earnings", 0)) * 4
                if eps_annual <= 0:
                    continue
                date_str = str(idx)[:10]
                nearby = hist[hist.index.strftime("%Y-%m-%d") >= date_str].head(1)
                if not nearby.empty:
                    price = float(nearby["Close"].iloc[0])
                    pe = price / eps_annual
                    if 0 < pe < 500:
                        pe_values.append(pe)
            except Exception:
                pass

        if len(pe_values) >= 4:
            return _compute_stats(pe_values, "yfinance computed")

        return {}

    except Exception as exc:
        print(f"yfinance PE history error for {symbol}: {exc}")
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
    try:
        conn = connect_database(db_path, neon_url=neon_database_url or None)
        row = conn.execute(
            "SELECT data, fetched_at FROM pe_history_cache WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        fetched = datetime.strptime(row["fetched_at"], "%Y-%m-%d").date()
        if (date.today() - fetched).days > CACHE_TTL_DAYS:
            print(f"PE cache stale for {symbol} ({(date.today() - fetched).days}d) - refreshing")
            return None

        return json.loads(row["data"])

    except Exception:
        return None


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
        print(f"PE cache save error for {symbol}: {exc}")


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
    Uses 500 ms inter-request delay to be polite to upstream servers.
    Intended to run in a background thread.

    Returns {"saved": N, "skipped": N, "failed": N} so callers can surface
    the outcome — particularly to distinguish "all sources returned empty"
    (network/scraping block) from actual successes.
    """
    print(f"Pre-fetching PE history for {len(symbols)} stocks...")
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
        if attempted % 10 == 0:
            print(
                f"  PE pre-fetch: {saved} saved, {failed} failed "
                f"({i + 1}/{len(symbols)} processed)"
            )
        time.sleep(0.5)
    print(
        f"PE history pre-fetch complete — "
        f"{saved} saved / {failed} failed / {skipped} already cached / {len(symbols)} total"
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
    Fetches history from Screener → Wisesheets → yfinance, caches 7 days.
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

    if not history or not history.get("median_5yr"):
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
                "Sources tried: Screener.in, Wisesheets, yfinance. Will retry on next run."
            ),
        }

    median = float(history["median_5yr"])
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
