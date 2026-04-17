"""
cache_init.py
─────────────
Startup helpers for the three platform cache tables.
Call ensure_cache_tables() once at app startup so all fetchers can write
to Neon (primary) or SQLite (fallback) without worrying about DDL.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from stock_platform.data.db import connect_database

_log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "platform.db"

_DDL = [
    """CREATE TABLE IF NOT EXISTS result_date_cache (
        symbol      TEXT PRIMARY KEY,
        result_date TEXT,
        days_stale  INTEGER,
        freshness   TEXT,
        source      TEXT,
        cached_at   TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS pe_history_cache (
        symbol     TEXT PRIMARY KEY,
        data       TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS fii_dii_cache (
        key        TEXT PRIMARY KEY,
        payload    TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    )""",
]


def ensure_cache_tables(
    db_path: str | Path | None = None,
    neon_database_url: str = "",
) -> None:
    """
    Create all three cache tables if they don't exist.
    Safe to call on every startup (CREATE TABLE IF NOT EXISTS).
    """
    _db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    _neon_url = neon_database_url or os.environ.get("NEON_DATABASE_URL", "")
    conn = None
    try:
        conn = connect_database(_db_path, neon_url=_neon_url or None)
        for ddl in _DDL:
            conn.execute(ddl)
        conn.commit()
        _log.info("Cache tables verified/created OK (dialect=%s)", getattr(conn, "dialect", "?"))
    except Exception as exc:
        _log.warning("ensure_cache_tables failed: %s", exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def get_cache_row_counts(
    db_path: str | Path | None = None,
    neon_database_url: str = "",
) -> dict[str, int | str]:
    """
    Return row counts for all three cache tables.
    Used by the Streamlit sidebar health check widget.
    Returns {"result_date_cache": N, "pe_history_cache": N, "fii_dii_cache": N, "backend": "neon"|"sqlite"}.
    """
    _db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    _neon_url = neon_database_url or os.environ.get("NEON_DATABASE_URL", "")
    counts: dict[str, Any] = {
        "result_date_cache": "?",
        "pe_history_cache":  "?",
        "fii_dii_cache":     "?",
        "backend":           "unknown",
    }
    conn = None
    try:
        conn = connect_database(_db_path, neon_url=_neon_url or None)
        counts["backend"] = getattr(conn, "dialect", "sqlite")
        for table in ("result_date_cache", "pe_history_cache", "fii_dii_cache"):
            try:
                row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
                counts[table] = int(row["cnt"]) if row else 0
            except Exception:
                counts[table] = "?"
    except Exception as exc:
        _log.warning("get_cache_row_counts error: %s", exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return counts
