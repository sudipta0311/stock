"""
fii_dii_fetcher.py
──────────────────
Fetch net FII/DII equity flow from NSE India's public API.
Results are cached in the platform database (Neon primary, SQLite fallback) for 24 hours.

NSE requires a session cookie obtained by hitting the base URL first.
The API returns per-date rows with net equity buy/sell by category.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_platform.data.db import connect_database

_log = logging.getLogger(__name__)

_CACHE_TTL_HOURS = 24

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "platform.db"

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS fii_dii_cache (
        key        TEXT PRIMARY KEY,
        payload    TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    )
"""

NSE_BASE_URL = "https://www.nseindia.com"
NSE_API_URL  = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json, text/plain, */*",
    "Referer":    "https://www.nseindia.com",
    "Accept-Language": "en-US,en;q=0.9",
}


def _ensure_table(conn: Any) -> None:
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()


def _load_cache(conn: Any, key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT payload, fetched_at FROM fii_dii_cache WHERE key = ?", (key,)
    ).fetchone()
    if not row:
        return None
    try:
        fetched_at = datetime.fromisoformat(row["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age_hours > _CACHE_TTL_HOURS:
            return None
        result = json.loads(row["payload"])
        result["cached_at"] = fetched_at.isoformat()
        return result
    except Exception:
        return None


def _save_cache(conn: Any, key: str, data: dict[str, Any]) -> None:
    payload = json.dumps(data)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO fii_dii_cache (key, payload, fetched_at) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET
               payload    = excluded.payload,
               fetched_at = excluded.fetched_at""",
        (key, payload, now),
    )
    conn.commit()


def _parse_flow(data: list[dict]) -> dict[str, Any]:
    """
    Aggregate net equity flow by category across all rows returned by the API.
    NSE fiidiiTradeReact returns one row per category per day with field netValue.
    """
    fii_net = 0.0
    dii_net = 0.0
    rows_seen = 0

    for row in data:
        category = str(row.get("category") or row.get("Category") or "")
        # NSE uses "netValue"; older API versions used other names — try all.
        net_val = (
            row.get("netValue")
            or row.get("netVal")
            or row.get("net_val")
            or row.get("NET")
            or row.get("netPurchSale")
            or 0.0
        )
        try:
            net_val = float(str(net_val).replace(",", ""))
        except (TypeError, ValueError):
            net_val = 0.0

        if "FII" in category.upper() or "FPI" in category.upper():
            fii_net += net_val
            rows_seen += 1
        elif "DII" in category.upper():
            dii_net += net_val
            rows_seen += 1

    # fiidiiTradeReact returns values already in ₹ Crore.
    fii_cr = round(fii_net, 0)
    dii_cr = round(dii_net, 0)

    # Flow signal — labels used in prompts
    if fii_cr > 3000:
        flow_signal = "STRONG_TAILWIND"
        fii_context = (
            "FIIs aggressively buying Indian equities — strong tailwind "
            "for large-cap valuations. Normal entry bar applies."
        )
    elif fii_cr > 500:
        flow_signal = "MILD_TAILWIND"
        fii_context = (
            "FIIs net buyers — mild positive flow supporting market sentiment. "
            "Entry bar unchanged."
        )
    elif fii_cr > -500:
        flow_signal = "NEUTRAL"
        fii_context = (
            "FII flow neutral — no strong directional bias from foreign investors."
        )
    elif fii_cr > -3000:
        flow_signal = "MILD_HEADWIND"
        fii_context = (
            "FIIs net sellers — mild headwind. Slightly higher entry conviction required."
        )
    else:
        flow_signal = "STRONG_HEADWIND"
        fii_context = (
            "FIIs aggressively selling Indian equities — significant headwind. "
            "Raise entry bar, widen stops, reduce position sizes."
        )

    # DII support signal
    if dii_cr > 2000:
        dii_signal = "STRONG_SUPPORT"
    elif dii_cr > 0:
        dii_signal = "MILD_SUPPORT"
    else:
        dii_signal = "WITHDRAWING"

    # Combined market signal for UI badge
    if flow_signal in ("STRONG_TAILWIND", "MILD_TAILWIND") and dii_signal in ("STRONG_SUPPORT", "MILD_SUPPORT"):
        market_signal = "RISK_ON"
    elif flow_signal in ("STRONG_TAILWIND", "MILD_TAILWIND"):
        market_signal = "RISK_ON"
    elif flow_signal == "STRONG_HEADWIND" and dii_signal == "WITHDRAWING":
        market_signal = "RISK_OFF"
    elif flow_signal in ("STRONG_HEADWIND", "MILD_HEADWIND"):
        market_signal = "CAUTIOUS"
    else:
        market_signal = "NEUTRAL"

    # Divergence override: FII buying + heavy DII selling
    # Domestic institutions distributing into foreign demand often precedes correction.
    if fii_cr > 0 and dii_cr < -2000:
        market_signal = "CAUTIOUS"
        fii_context = (
            fii_context
            + f" However, DIIs are selling heavily (net -{abs(dii_cr):,.0f}Cr)"
            " while FIIs buy — domestic institutions distributing into foreign demand."
            " This divergence warrants caution on entry timing."
        )

    from datetime import date as _date
    return {
        "fii_net_5d_cr": fii_cr,
        "dii_net_5d_cr": dii_cr,
        "flow_signal":   flow_signal,
        "dii_signal":    dii_signal,
        "market_signal": market_signal,
        "fii_context":   fii_context,
        "rows_parsed":   rows_seen,
        "as_of":         _date.today().isoformat(),
    }


def fetch_fii_dii_sector_flow(
    db_path: str | Path | None = None,
    neon_database_url: str = "",
) -> dict[str, Any]:
    """
    Fetch net FII/DII equity activity from NSE India for the last 5 sessions.
    Results are cached in the platform database (Neon primary, SQLite fallback) for 24 hours.

    Returns:
        fii_net_5d_cr  — net FII buy (+) / sell (-) in crores
        dii_net_5d_cr  — net DII buy (+) / sell (-) in crores
        flow_signal    — STRONG_TAILWIND | MILD_TAILWIND | MILD_HEADWIND | STRONG_HEADWIND
        source         — "nse_api" | "cache" | "unavailable"
    """
    import requests

    _db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    _neon_url = neon_database_url or os.environ.get("NEON_DATABASE_URL", "")
    cache_key = "fii_dii_5d"

    conn = None
    try:
        conn = connect_database(_db_path, neon_url=_neon_url or None)
        _ensure_table(conn)

        cached = _load_cache(conn, cache_key)
        if cached:
            _log.info("FII/DII flow: cache hit")
            conn.close()
            return {**cached, "source": "cache"}
    except Exception as exc:
        _log.warning("FII/DII cache read error: %s", exc)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        conn = None

    try:
        session = requests.Session()
        # NSE requires a cookie — hit the base page first
        session.get(NSE_BASE_URL, headers=NSE_HEADERS, timeout=10)
        resp = session.get(NSE_API_URL, headers=NSE_HEADERS, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        # API may return a list directly or a dict with a data key
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict):
            data = raw.get("data") or raw.get("result") or []
        else:
            data = []

        if not data:
            _log.warning("FII/DII: NSE returned empty payload")
            return _unavailable("empty NSE response")

        result = _parse_flow(data)

        # Write back to cache
        try:
            if conn is None:
                conn = connect_database(_db_path, neon_url=_neon_url or None)
                _ensure_table(conn)
            _save_cache(conn, cache_key, result)
        except Exception as exc:
            _log.warning("FII/DII cache write error: %s", exc)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        _log.info(
            "FII/DII flow fetched: FII=%.0fCr DII=%.0fCr signal=%s",
            result["fii_net_5d_cr"], result["dii_net_5d_cr"], result["flow_signal"],
        )
        return {**result, "source": "nse_api"}

    except Exception as exc:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        _log.warning("FII/DII fetch failed: %s", exc)
        return _unavailable(str(exc))


def _unavailable(reason: str) -> dict[str, Any]:
    from datetime import date as _date
    return {
        "fii_net_5d_cr": None,
        "dii_net_5d_cr": None,
        "flow_signal":   "UNKNOWN",
        "dii_signal":    "UNKNOWN",
        "market_signal": "UNKNOWN",
        "fii_context":   "FII/DII flow data unavailable today — treat as neutral.",
        "as_of":         _date.today().isoformat(),
        "source":        "unavailable",
        "error":         reason,
    }


def format_macro_flow_for_prompt(flow: dict[str, Any]) -> str:
    """
    Format FII/DII flow data as a macro context block for LLM prompts.
    Returns an empty string if data is unavailable so callers can safely inject it.
    """
    if flow.get("source") == "unavailable" or flow.get("fii_net_5d_cr") is None:
        return ""

    fii = float(flow["fii_net_5d_cr"])
    dii = float(flow.get("dii_net_5d_cr") or 0)
    flow_signal = flow.get("flow_signal", "UNKNOWN")
    market_signal = flow.get("market_signal", "UNKNOWN")
    fii_context = flow.get("fii_context", "")
    as_of = flow.get("as_of", "")

    fii_str = f"+₹{abs(fii):,.0f}Cr" if fii >= 0 else f"-₹{abs(fii):,.0f}Cr"
    dii_str = f"+₹{abs(dii):,.0f}Cr" if dii >= 0 else f"-₹{abs(dii):,.0f}Cr"

    headwind_instruction = (
        "\nINSTRUCTION: FII selling is active — raise your entry bar. "
        "Even a quality stock faces a timing headwind when FIIs are net sellers at scale. "
        "Reflect this in your confidence level and entry zone recommendation."
        if flow_signal in ("STRONG_HEADWIND", "MILD_HEADWIND") else ""
    )

    return (
        f"\nMACRO FLOW CONTEXT ({as_of or 'latest'}):\n"
        f"  FII net: {fii_str} | Flow signal: {flow_signal}\n"
        f"  DII net: {dii_str} | Market signal: {market_signal}\n"
        f"  Context: {fii_context}"
        f"{headwind_instruction}\n"
    )
