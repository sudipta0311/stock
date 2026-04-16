"""
fii_dii_fetcher.py
──────────────────
Fetch net FII/DII equity flow from NSE India's public API.
Results are cached in SQLite for 24 hours to avoid hammering NSE on every run.

NSE requires a session cookie obtained by hitting the base URL first.
The API returns per-date rows with net equity buy/sell by category.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_CACHE_TTL_HOURS = 24

# Standalone SQLite for this cache — doesn't need the full platform DB schema.
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "fii_dii_cache.db"

NSE_BASE_URL = "https://www.nseindia.com"
NSE_API_URL  = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json, text/plain, */*",
    "Referer":    "https://www.nseindia.com",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get_cache_conn(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS fii_dii_cache (
            key        TEXT PRIMARY KEY,
            payload    TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def _load_cache(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT payload, fetched_at FROM fii_dii_cache WHERE key = ?", (key,)
    ).fetchone()
    if not row:
        return None
    try:
        fetched_at = datetime.fromisoformat(row[1])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age_hours > _CACHE_TTL_HOURS:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def _save_cache(conn: sqlite3.Connection, key: str, data: dict[str, Any]) -> None:
    payload = json.dumps(data)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO fii_dii_cache (key, payload, fetched_at) VALUES (?, ?, ?)",
        (key, payload, now),
    )
    conn.commit()


def _parse_flow(data: list[dict]) -> dict[str, Any]:
    """
    Aggregate net equity flow by category across all rows returned by the API.
    NSE returns rows per trading date; we sum across the last 5 sessions.
    """
    fii_net = 0.0
    dii_net = 0.0
    rows_seen = 0

    for row in data:
        category = str(row.get("category") or row.get("Category") or "")
        # net equity value — field names vary slightly by NSE API version
        net_val = (
            row.get("netVal")
            or row.get("net_val")
            or row.get("NET")
            or row.get("netPurchSale")
            or 0.0
        )
        try:
            net_val = float(net_val)
        except (TypeError, ValueError):
            net_val = 0.0

        if "FII" in category.upper() or "FPI" in category.upper():
            fii_net += net_val
            rows_seen += 1
        elif "DII" in category.upper():
            dii_net += net_val
            rows_seen += 1

    # NSE reports in crores directly (some endpoints) or in lakhs — normalise
    # The fiidiiTradeReact endpoint returns values already in crores.
    fii_cr = round(fii_net, 0)
    dii_cr = round(dii_net, 0)

    if fii_cr > 5000:
        flow_signal = "STRONG_TAILWIND"
    elif fii_cr > 0:
        flow_signal = "MILD_TAILWIND"
    elif fii_cr > -5000:
        flow_signal = "MILD_HEADWIND"
    else:
        flow_signal = "STRONG_HEADWIND"

    return {
        "fii_net_5d_cr": fii_cr,
        "dii_net_5d_cr": dii_cr,
        "flow_signal":   flow_signal,
        "rows_parsed":   rows_seen,
    }


def fetch_fii_dii_sector_flow(
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Fetch net FII/DII equity activity from NSE India for the last 5 sessions.
    Results are cached in SQLite for 24 hours.

    Returns:
        fii_net_5d_cr  — net FII buy (+) / sell (-) in crores
        dii_net_5d_cr  — net DII buy (+) / sell (-) in crores
        flow_signal    — STRONG_TAILWIND | MILD_TAILWIND | MILD_HEADWIND | STRONG_HEADWIND
        source         — "nse_api" | "cache" | "unavailable"
    """
    import requests

    _cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH
    cache_key = "fii_dii_5d"

    conn = _get_cache_conn(_cache_path)
    try:
        cached = _load_cache(conn, cache_key)
        if cached:
            _log.info("FII/DII flow: cache hit")
            return {**cached, "source": "cache"}
    finally:
        pass  # keep conn open for potential write below

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
            conn.close()
            return _unavailable("empty NSE response")

        result = _parse_flow(data)
        _save_cache(conn, cache_key, result)
        conn.close()
        _log.info(
            "FII/DII flow fetched: FII=%.0fCr DII=%.0fCr signal=%s",
            result["fii_net_5d_cr"], result["dii_net_5d_cr"], result["flow_signal"],
        )
        return {**result, "source": "nse_api"}

    except Exception as exc:
        conn.close()
        _log.warning("FII/DII fetch failed: %s", exc)
        return _unavailable(str(exc))


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "fii_net_5d_cr": None,
        "dii_net_5d_cr": None,
        "flow_signal":   "UNKNOWN",
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

    fii = flow["fii_net_5d_cr"]
    dii = flow["dii_net_5d_cr"]
    signal = flow["flow_signal"]

    fii_str = f"+Rs.{abs(fii):.0f}Cr" if fii >= 0 else f"-Rs.{abs(fii):.0f}Cr"
    dii_str = f"+Rs.{abs(dii):.0f}Cr" if dii >= 0 else f"-Rs.{abs(dii):.0f}Cr"

    instruction = {
        "STRONG_TAILWIND": "FII buying is strong - sector re-ratings are more likely. Normal entry bar applies.",
        "MILD_TAILWIND":   "FII net buyers - mild macro support. Entry bar unchanged.",
        "MILD_HEADWIND":   "FII net sellers - mild headwind. Slightly higher entry conviction required.",
        "STRONG_HEADWIND": "FII STRONG_HEADWIND - even quality stocks face selling pressure. Raise entry bar. Prefer ACCUMULATE ON DIPS over immediate entry.",
        "UNKNOWN":         "FII/DII flow data unavailable - do not factor macro flow into this analysis.",
    }.get(signal, "")

    return (
        "\nMACRO FLOW (last 5 sessions):\n"
        f"  FII net: {fii_str} | Signal: {signal}\n"
        f"  DII net: {dii_str}\n"
        f"  {instruction}\n"
    )
