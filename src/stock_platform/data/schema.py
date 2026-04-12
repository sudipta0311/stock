from __future__ import annotations

import sqlite3


DDL = [
    """
    CREATE TABLE IF NOT EXISTS app_state (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_holdings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        holding_type TEXT NOT NULL,
        instrument_name TEXT NOT NULL,
        symbol TEXT,
        quantity REAL DEFAULT 0,
        market_value REAL DEFAULT 0,
        source TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS normalized_exposure (
        symbol TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        sector TEXT NOT NULL,
        total_weight REAL NOT NULL,
        source_mix_json TEXT NOT NULL,
        attribution_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS direct_equity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT UNIQUE,
        quantity REAL,
        avg_buy_price REAL,
        current_price REAL,
        buy_date TEXT,
        source TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS overlap_scores (
        symbol TEXT PRIMARY KEY,
        overlap_pct REAL NOT NULL,
        band TEXT NOT NULL,
        attribution_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS identified_gaps (
        sector TEXT PRIMARY KEY,
        underweight_pct REAL NOT NULL,
        conviction TEXT NOT NULL,
        score REAL NOT NULL,
        reason TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_family TEXT NOT NULL,
        signal_key TEXT,
        sector TEXT NOT NULL,
        conviction TEXT NOT NULL,
        score REAL NOT NULL,
        source TEXT NOT NULL,
        horizon TEXT NOT NULL,
        detail TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        company_name TEXT NOT NULL,
        sector TEXT NOT NULL,
        action TEXT NOT NULL,
        score REAL NOT NULL,
        confidence_band TEXT NOT NULL,
        rationale TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitoring_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        severity TEXT NOT NULL,
        urgency TEXT NOT NULL DEFAULT 'LOW',
        rationale TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitoring_watchlist (
        symbol TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        sector TEXT NOT NULL DEFAULT 'Unknown',
        note TEXT NOT NULL DEFAULT '',
        added_at TEXT NOT NULL
    )
    """,
]


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    existing = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in existing:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def initialize_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    for statement in DDL:
        cursor.execute(statement)
    _ensure_column(connection, "monitoring_actions", "urgency", "TEXT NOT NULL DEFAULT 'LOW'")
    connection.commit()

