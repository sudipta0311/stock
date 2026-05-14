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
    """
    CREATE TABLE IF NOT EXISTS recommendation_history (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id                TEXT    NOT NULL,
        run_date              TEXT    NOT NULL,
        symbol                TEXT    NOT NULL,
        sector                TEXT,
        risk_profile          TEXT,
        verdict               TEXT,
        confidence            TEXT,
        quant_score           REAL,
        rr_ratio              REAL,
        entry_zone_low        REAL,
        entry_zone_high       REAL,
        stop_loss             REAL,
        target_1              REAL,
        target_2              REAL,
        cmp_at_recommendation REAL,
        pe_at_recommendation  REAL,
        fii_flow_cr           REAL,
        market_signal         TEXT,
        rationale_summary     TEXT,
        llm_provider          TEXT,
        llm_models            TEXT,
        llm_agreement         TEXT,
        user_acted            INTEGER DEFAULT 0,
        user_action_date      TEXT,
        user_entry_price      REAL,
        user_notes            TEXT,
        UNIQUE(run_id, symbol)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_history_symbol   ON recommendation_history(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_history_date     ON recommendation_history(run_date)",
    "CREATE INDEX IF NOT EXISTS idx_history_provider ON recommendation_history(llm_provider)",
    # ── Backtest tables ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS historical_fundamentals (
        -- Quarterly fundamental snapshots used by backtest replay.
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol           TEXT    NOT NULL,
        snapshot_date    TEXT    NOT NULL,
        roce             REAL,
        eps              REAL,
        debt_equity      REAL,
        revenue_growth   REAL,
        promoter_holding REAL,
        source           TEXT    NOT NULL DEFAULT 'yfinance',
        fetched_source   TEXT,
        UNIQUE(symbol, snapshot_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS historical_prices (
        -- Daily/weekly close prices used for forward-return computation.
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol       TEXT    NOT NULL,
        date         TEXT    NOT NULL,
        close_price  REAL    NOT NULL,
        volume       REAL,
        UNIQUE(symbol, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_runs (
        -- Summary row written by scorer.py after each replay+score cycle.
        run_id                TEXT PRIMARY KEY,
        start_date            TEXT NOT NULL,
        end_date              TEXT NOT NULL,
        weights_hash          TEXT,
        total_recommendations INTEGER DEFAULT 0,
        hit_rate_3m           REAL,
        hit_rate_6m           REAL,
        hit_rate_12m          REAL,
        alpha_3m              REAL,
        alpha_6m              REAL,
        alpha_12m             REAL,
        created_at            TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_recommendations (
        -- Per-symbol per-date recommendations from replay.py.
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id              TEXT    NOT NULL,
        symbol              TEXT    NOT NULL,
        recommendation_date TEXT    NOT NULL,
        action              TEXT    NOT NULL,
        confidence_band     TEXT,
        quality_score       REAL,
        forward_return_3m   REAL,
        forward_return_6m   REAL,
        forward_return_12m  REAL,
        hit                 INTEGER,
        UNIQUE(run_id, symbol, recommendation_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_hist_fund_symbol ON historical_fundamentals(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_hist_price_symbol ON historical_prices(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_bt_rec_run ON backtest_recommendations(run_id)",
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
    """
    CREATE TABLE IF NOT EXISTS recommendation_history (
        id                    BIGSERIAL PRIMARY KEY,
        run_id                TEXT             NOT NULL,
        run_date              TEXT             NOT NULL,
        symbol                TEXT             NOT NULL,
        sector                TEXT,
        risk_profile          TEXT,
        verdict               TEXT,
        confidence            TEXT,
        quant_score           DOUBLE PRECISION,
        rr_ratio              DOUBLE PRECISION,
        entry_zone_low        DOUBLE PRECISION,
        entry_zone_high       DOUBLE PRECISION,
        stop_loss             DOUBLE PRECISION,
        target_1              DOUBLE PRECISION,
        target_2              DOUBLE PRECISION,
        cmp_at_recommendation DOUBLE PRECISION,
        pe_at_recommendation  DOUBLE PRECISION,
        fii_flow_cr           DOUBLE PRECISION,
        market_signal         TEXT,
        rationale_summary     TEXT,
        llm_provider          TEXT,
        llm_models            TEXT,
        llm_agreement         TEXT,
        user_acted            BOOLEAN          DEFAULT FALSE,
        user_action_date      TEXT,
        user_entry_price      DOUBLE PRECISION,
        user_notes            TEXT,
        UNIQUE(run_id, symbol)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_history_symbol   ON recommendation_history(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_history_date     ON recommendation_history(run_date)",
    "CREATE INDEX IF NOT EXISTS idx_history_provider ON recommendation_history(llm_provider)",
    # ── Backtest tables ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS historical_fundamentals (
        -- Quarterly fundamental snapshots used by backtest replay.
        id               BIGSERIAL PRIMARY KEY,
        symbol           TEXT             NOT NULL,
        snapshot_date    TEXT             NOT NULL,
        roce             DOUBLE PRECISION,
        eps              DOUBLE PRECISION,
        debt_equity      DOUBLE PRECISION,
        revenue_growth   DOUBLE PRECISION,
        promoter_holding DOUBLE PRECISION,
        source           TEXT             NOT NULL DEFAULT 'yfinance',
        fetched_source   TEXT,
        UNIQUE(symbol, snapshot_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS historical_prices (
        -- Daily/weekly close prices used for forward-return computation.
        id           BIGSERIAL PRIMARY KEY,
        symbol       TEXT             NOT NULL,
        date         TEXT             NOT NULL,
        close_price  DOUBLE PRECISION NOT NULL,
        volume       DOUBLE PRECISION,
        UNIQUE(symbol, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_runs (
        -- Summary row written by scorer.py after each replay+score cycle.
        run_id                TEXT PRIMARY KEY,
        start_date            TEXT NOT NULL,
        end_date              TEXT NOT NULL,
        weights_hash          TEXT,
        total_recommendations INTEGER DEFAULT 0,
        hit_rate_3m           DOUBLE PRECISION,
        hit_rate_6m           DOUBLE PRECISION,
        hit_rate_12m          DOUBLE PRECISION,
        alpha_3m              DOUBLE PRECISION,
        alpha_6m              DOUBLE PRECISION,
        alpha_12m             DOUBLE PRECISION,
        created_at            TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_recommendations (
        -- Per-symbol per-date recommendations from replay.py.
        id                  BIGSERIAL PRIMARY KEY,
        run_id              TEXT    NOT NULL,
        symbol              TEXT    NOT NULL,
        recommendation_date TEXT    NOT NULL,
        action              TEXT    NOT NULL,
        confidence_band     TEXT,
        quality_score       DOUBLE PRECISION,
        forward_return_3m   DOUBLE PRECISION,
        forward_return_6m   DOUBLE PRECISION,
        forward_return_12m  DOUBLE PRECISION,
        hit                 BOOLEAN,
        UNIQUE(run_id, symbol, recommendation_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_hist_fund_symbol ON historical_fundamentals(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_hist_price_symbol ON historical_prices(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_bt_rec_run ON backtest_recommendations(run_id)",
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
    _ensure_column(connection, "historical_fundamentals", "fetched_source", "TEXT")
    connection.commit()
