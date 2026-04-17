"""
result_date_fetcher.py
──────────────────────
Three-source chain for last quarterly result date on NSE stocks.
Priority: NSE official → Tickertape → yfinance.
Results cached in the platform database for 7 days.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "platform.db"

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS result_date_cache (
        symbol      TEXT PRIMARY KEY,
        result_date TEXT,
        days_stale  INTEGER,
        freshness   TEXT,
        source      TEXT,
        cached_at   TEXT
    )
"""
_CREATE_TABLE_SQL_PG = """
    CREATE TABLE IF NOT EXISTS result_date_cache (
        symbol      TEXT PRIMARY KEY,
        result_date TEXT,
        days_stale  INTEGER,
        freshness   TEXT,
        source      TEXT,
        cached_at   TEXT
    )
"""


# ── Source 1: NSE official corporate filings ─────────────────────────────────

def _fetch_from_nse(symbol: str) -> dict[str, Any]:
    """Query NSE's public corp-info API for the latest financial-result filing date."""
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
            ),
            "Accept":  "application/json",
            "Referer": "https://www.nseindia.com",
        })
        # NSE requires a valid session cookie — prime it first.
        session.get("https://www.nseindia.com", timeout=10)

        url = (
            "https://www.nseindia.com/api/"
            f"corp-info?symbol={symbol}&corpType=financials"
        )
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return {}

        filings = r.json().get("data", [])
        if not filings:
            return {}

        dates: list[datetime] = []
        for f in filings:
            raw = f.get("date") or f.get("bm_date") or ""
            for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
                try:
                    dates.append(datetime.strptime(raw, fmt))
                    break
                except ValueError:
                    continue

        if not dates:
            return {}

        return {
            "result_date": max(dates).strftime("%Y-%m-%d"),
            "source": "nse_official",
        }
    except Exception as exc:
        print(f"NSE fetch failed for {symbol}: {exc}")
        return {}


# ── Source 2: Tickertape ──────────────────────────────────────────────────────

def _fetch_from_tickertape(symbol: str) -> dict[str, Any]:
    """Query Tickertape's search + income API for the most recent quarter end date."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
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

        fr = requests.get(
            f"https://api.tickertape.in/stocks/{sid}/financials/income?period=quarterly",
            headers=headers,
            timeout=15,
        )
        if fr.status_code != 200:
            return {}

        statements = fr.json().get("data", {}).get("statements", [])
        if not statements:
            return {}

        latest = statements[0].get("periodEnd") or statements[0].get("date")
        if not latest:
            return {}

        return {"result_date": str(latest)[:10], "source": "tickertape"}
    except Exception as exc:
        print(f"Tickertape fetch failed for {symbol}: {exc}")
        return {}


# ── Source 3: yfinance fallback ───────────────────────────────────────────────

def _fetch_from_yfinance(symbol: str) -> dict[str, Any]:
    """yfinance three-path fallback: mrq timestamp → quarterly_financials → quarterly_income_stmt."""
    try:
        import yfinance as yf
        import pandas as pd

        t = yf.Ticker(f"{symbol}.NS")

        mrq = t.info.get("mostRecentQuarter")
        if mrq and int(mrq) > 0:
            return {
                "result_date": datetime.fromtimestamp(int(mrq)).strftime("%Y-%m-%d"),
                "source": "yfinance_mrq",
            }

        for attr in ("quarterly_financials", "quarterly_income_stmt"):
            try:
                df = getattr(t, attr)
                if df is not None and not df.empty:
                    return {
                        "result_date": pd.Timestamp(df.columns[0]).strftime("%Y-%m-%d"),
                        "source": f"yfinance_{attr}",
                    }
            except Exception:
                continue
    except Exception as exc:
        print(f"yfinance date fetch failed for {symbol}: {exc}")
    return {}


# ── Freshness bucket ──────────────────────────────────────────────────────────

def _freshness(days: int) -> str:
    if days <= 45:
        return "FRESH"
    if days <= 90:
        return "RECENT"
    if days <= 180:
        return "STALE"
    return "VERY_STALE"


# ── SQLite / Neon cache helpers ───────────────────────────────────────────────

def _db_connect(db_path: str | Path):
    from stock_platform.data.db import connect_database
    neon_url = os.environ.get("NEON_DATABASE_URL", "")
    return connect_database(db_path, neon_url=neon_url or None)


def _ensure_table(db_path: str | Path) -> None:
    try:
        conn = _db_connect(db_path)
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
        conn.close()
    except Exception:
        pass


def _delete_cached(symbol: str, db_path: str | Path) -> None:
    """Remove a cached entry so it gets re-fetched from live sources."""
    try:
        conn = _db_connect(db_path)
        conn.execute("DELETE FROM result_date_cache WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _get_cached(symbol: str, db_path: str | Path) -> dict[str, Any]:
    _ensure_table(db_path)
    try:
        conn = _db_connect(db_path)
        row = conn.execute(
            "SELECT result_date, freshness, source, cached_at "
            "FROM result_date_cache WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        conn.close()
        if not row:
            return {}
        # Use named key access — works for both NeonWrapper._NeonRow and sqlite3.Row
        cached_at = datetime.strptime(str(row["cached_at"]), "%Y-%m-%d").date()
        if (date.today() - cached_at).days > 7:
            return {}
        result_date = row["result_date"]
        # Validity gate on cached value — bad dates (e.g. 2014 artefacts) can be
        # persisted in Neon before this gate was added.  Evict and re-fetch live.
        if result_date and not _is_valid_result_date(result_date):
            print(f"Evicting invalid cached date for {symbol}: {result_date} — will re-fetch")
            _delete_cached(symbol, db_path)
            return {}
        days_stale = None
        if result_date:
            days_stale = (date.today() - datetime.strptime(result_date, "%Y-%m-%d").date()).days
        return {
            "result_date":       result_date,
            "result_days_stale": days_stale,
            "result_freshness":  _freshness(days_stale) if days_stale is not None else row["freshness"],
            "source":            str(row["source"]) + "_cache",
        }
    except Exception:
        return {}


def _save_cache(symbol: str, data: dict[str, Any], db_path: str | Path) -> None:
    _ensure_table(db_path)
    try:
        conn = _db_connect(db_path)
        # Use ? placeholders — NeonWrapper translates to %s for PostgreSQL.
        # ON CONFLICT DO UPDATE works for both PostgreSQL and SQLite 3.24+.
        conn.execute(
            "INSERT INTO result_date_cache "
            "(symbol, result_date, days_stale, freshness, source, cached_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET "
            "result_date=excluded.result_date, days_stale=excluded.days_stale, "
            "freshness=excluded.freshness, source=excluded.source, cached_at=excluded.cached_at",
            (
                symbol,
                data.get("result_date"),
                data.get("result_days_stale"),
                data.get("result_freshness"),
                data.get("source"),
                date.today().strftime("%Y-%m-%d"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"result_date_cache save failed for {symbol}: {exc}")


# ── Validity gate ────────────────────────────────────────────────────────────

def _is_valid_result_date(date_str: str) -> bool:
    """
    Rejects obviously wrong dates.
    Valid result dates must be within the last 3 years — anything older is
    a data artefact (e.g. yfinance returning 2014 IPO-era financials).
    """
    if not date_str:
        return False
    try:
        rd = datetime.strptime(date_str, "%Y-%m-%d").date()
        years_ago = (date.today() - rd).days / 365
        return 0 <= years_ago <= 3
    except Exception:
        return False


# ── Cache cleanup ────────────────────────────────────────────────────────────

def purge_invalid_result_dates(db_path: str | Path | None = None) -> int:
    """
    Delete all rows from result_date_cache whose result_date is outside the
    valid 3-year window (catches historical artefacts like 2014-06-30).

    Returns the number of rows deleted.  Safe to call on every app startup —
    it is a no-op when the cache is already clean.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH
    _ensure_table(db_path)
    try:
        today = date.today().strftime("%Y-%m-%d")
        cutoff = (date.today().replace(year=date.today().year - 3)).strftime("%Y-%m-%d")
        conn = _db_connect(db_path)
        cur = conn.execute(
            "DELETE FROM result_date_cache "
            "WHERE result_date < ? OR result_date > ?",
            (cutoff, today),
        )
        deleted = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
        conn.close()
        if deleted:
            print(f"purge_invalid_result_dates: removed {deleted} bad cache rows (outside {cutoff}–{today})")
        return deleted
    except Exception as exc:
        print(f"purge_invalid_result_dates failed: {exc}")
        return 0


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_last_result_date(
    symbol: str,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Fetch the most recent quarterly result date for an NSE stock.

    Chain: NSE official → Tickertape → yfinance.
    Results are cached in the platform database for 7 days.
    Never raises — returns a dict with result_date=None on total failure.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    clean = symbol.upper().replace(".NS", "").replace(".BO", "")

    cached = _get_cached(clean, db_path)
    if cached:
        return cached

    result = _fetch_from_nse(clean)
    if result and not _is_valid_result_date(result.get("result_date")):
        print(f"NSE returned invalid date for {clean}: {result.get('result_date')} — trying next source")
        result = None

    result = result or _fetch_from_tickertape(clean)
    if result and not _is_valid_result_date(result.get("result_date")):
        print(f"Tickertape returned invalid date for {clean}: {result.get('result_date')} — trying next source")
        result = None

    result = result or _fetch_from_yfinance(clean)
    if result and not _is_valid_result_date(result.get("result_date")):
        print(f"yfinance returned invalid date for {clean}: {result.get('result_date')} — discarding")
        result = None

    if result and result.get("result_date"):
        try:
            rd = datetime.strptime(result["result_date"], "%Y-%m-%d")
            days_stale = (date.today() - rd.date()).days
            result["result_days_stale"] = days_stale
            result["result_freshness"]  = _freshness(days_stale)
        except Exception:
            result["result_days_stale"] = None
            result["result_freshness"]  = "UNKNOWN"

        _save_cache(clean, result, db_path)
        print(
            f"Result date for {clean}: {result['result_date']} "
            f"({result.get('result_freshness')}, {result.get('result_days_stale')}d) "
            f"via {result.get('source')}"
        )
        return result

    print(f"Result date: all sources failed for {clean}")
    return {
        "result_date":       None,
        "result_days_stale": None,
        "result_freshness":  "NO_DATA",
        "source":            "unavailable",
    }
