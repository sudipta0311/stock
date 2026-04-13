from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# DDL — dialect-aware
#   SQLite   : INTEGER PRIMARY KEY AUTOINCREMENT
#   PostgreSQL: BIGSERIAL PRIMARY KEY
# ─────────────────────────────────────────────────────────────────────────────

_DDL_SQLITE = [
    """
    CREATE TABLE IF NOT EXISTS app_state (
        key        TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_holdings (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        holding_type     TEXT    NOT NULL,
        instrument_name  TEXT    NOT NULL,
        symbol           TEXT,
        quantity         REAL    DEFAULT 0,
        market_value     REAL    DEFAULT 0,
        source           TEXT    NOT NULL,
        payload_json     TEXT    NOT NULL,
        updated_at       TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS normalized_exposure (
        symbol           TEXT PRIMARY KEY,
        company_name     TEXT NOT NULL,
        sector           TEXT NOT NULL,
        total_weight     REAL NOT NULL,
        source_mix_json  TEXT NOT NULL,
        attribution_json TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS direct_equity (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol        TEXT UNIQUE,
        quantity      REAL,
        avg_buy_price REAL,
        current_price REAL,
        buy_date      TEXT,
        source        TEXT,
        updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS overlap_scores (
        symbol           TEXT PRIMARY KEY,
        overlap_pct      REAL NOT NULL,
        band             TEXT NOT NULL,
        attribution_json TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS identified_gaps (
        sector          TEXT PRIMARY KEY,
        underweight_pct REAL NOT NULL,
        conviction      TEXT NOT NULL,
        score           REAL NOT NULL,
        reason          TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_family TEXT NOT NULL,
        signal_key    TEXT,
        sector        TEXT NOT NULL,
        conviction    TEXT NOT NULL,
        score         REAL NOT NULL,
        source        TEXT NOT NULL,
        horizon       TEXT NOT NULL,
        detail        TEXT NOT NULL,
        as_of_date    TEXT NOT NULL,
        payload_json  TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           TEXT NOT NULL,
        symbol           TEXT NOT NULL,
        company_name     TEXT NOT NULL,
        sector           TEXT NOT NULL,
        action           TEXT NOT NULL,
        score            REAL NOT NULL,
        confidence_band  TEXT NOT NULL,
        rationale        TEXT NOT NULL,
        payload_json     TEXT NOT NULL,
        created_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitoring_actions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       TEXT NOT NULL,
        symbol       TEXT NOT NULL,
        action       TEXT NOT NULL,
        severity     TEXT NOT NULL,
        urgency      TEXT NOT NULL DEFAULT 'LOW',
        rationale    TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitoring_watchlist (
        symbol       TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        sector       TEXT NOT NULL DEFAULT 'Unknown',
        note         TEXT NOT NULL DEFAULT '',
        added_at     TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cache_entries (
        cache_key    TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        expires_at   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS skipped_stocks (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id     TEXT NOT NULL,
        symbol     TEXT NOT NULL,
        status     TEXT NOT NULL,
        reason     TEXT NOT NULL,
        skipped_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pe_history_cache (
        symbol     TEXT PRIMARY KEY,
        data       TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    )
    """,
]

_DDL_PG = [
    """
    CREATE TABLE IF NOT EXISTS app_state (
        key        TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_holdings (
        id               BIGSERIAL PRIMARY KEY,
        holding_type     TEXT NOT NULL,
        instrument_name  TEXT NOT NULL,
        symbol           TEXT,
        quantity         DOUBLE PRECISION DEFAULT 0,
        market_value     DOUBLE PRECISION DEFAULT 0,
        source           TEXT NOT NULL,
        payload_json     TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS normalized_exposure (
        symbol           TEXT PRIMARY KEY,
        company_name     TEXT NOT NULL,
        sector           TEXT NOT NULL,
        total_weight     DOUBLE PRECISION NOT NULL,
        source_mix_json  TEXT NOT NULL,
        attribution_json TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS direct_equity (
        id            BIGSERIAL PRIMARY KEY,
        symbol        TEXT UNIQUE,
        quantity      DOUBLE PRECISION,
        avg_buy_price DOUBLE PRECISION,
        current_price DOUBLE PRECISION,
        buy_date      TEXT,
        source        TEXT,
        updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS overlap_scores (
        symbol           TEXT PRIMARY KEY,
        overlap_pct      DOUBLE PRECISION NOT NULL,
        band             TEXT NOT NULL,
        attribution_json TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS identified_gaps (
        sector          TEXT PRIMARY KEY,
        underweight_pct DOUBLE PRECISION NOT NULL,
        conviction      TEXT NOT NULL,
        score           DOUBLE PRECISION NOT NULL,
        reason          TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id            BIGSERIAL PRIMARY KEY,
        signal_family TEXT NOT NULL,
        signal_key    TEXT,
        sector        TEXT NOT NULL,
        conviction    TEXT NOT NULL,
        score         DOUBLE PRECISION NOT NULL,
        source        TEXT NOT NULL,
        horizon       TEXT NOT NULL,
        detail        TEXT NOT NULL,
        as_of_date    TEXT NOT NULL,
        payload_json  TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id               BIGSERIAL PRIMARY KEY,
        run_id           TEXT NOT NULL,
        symbol           TEXT NOT NULL,
        company_name     TEXT NOT NULL,
        sector           TEXT NOT NULL,
        action           TEXT NOT NULL,
        score            DOUBLE PRECISION NOT NULL,
        confidence_band  TEXT NOT NULL,
        rationale        TEXT NOT NULL,
        payload_json     TEXT NOT NULL,
        created_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitoring_actions (
        id           BIGSERIAL PRIMARY KEY,
        run_id       TEXT NOT NULL,
        symbol       TEXT NOT NULL,
        action       TEXT NOT NULL,
        severity     TEXT NOT NULL,
        urgency      TEXT NOT NULL DEFAULT 'LOW',
        rationale    TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitoring_watchlist (
        symbol       TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        sector       TEXT NOT NULL DEFAULT 'Unknown',
        note         TEXT NOT NULL DEFAULT '',
        added_at     TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cache_entries (
        cache_key    TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        expires_at   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS skipped_stocks (
        id         BIGSERIAL PRIMARY KEY,
        run_id     TEXT NOT NULL,
        symbol     TEXT NOT NULL,
        status     TEXT NOT NULL,
        reason     TEXT NOT NULL,
        skipped_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pe_history_cache (
        symbol     TEXT PRIMARY KEY,
        data       TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    )
    """,
]


def _get_ddl(connection: Any) -> list[str]:
    dialect = getattr(connection, "dialect", "sqlite")
    return _DDL_PG if dialect == "postgresql" else _DDL_SQLITE


def _ensure_column(
    connection: Any,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    dialect = getattr(connection, "dialect", "sqlite")

    if dialect == "postgresql":
        rows = connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = 'public'",
            (table_name,),
        ).fetchall()
        existing = {row["column_name"] for row in rows}
    else:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {row[1] for row in rows}

    if column_name not in existing:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
        )


def initialize_schema(connection: Any) -> None:
    cursor = connection.cursor() if hasattr(connection, "cursor") else None
    for statement in _get_ddl(connection):
        connection.execute(statement)
    _ensure_column(connection, "monitoring_actions", "urgency", "TEXT NOT NULL DEFAULT 'LOW'")
    connection.commit()
