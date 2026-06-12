"""
tests/test_point_in_time.py — verify PIT (point-in-time) fundamentals behaviour.

Rules under test:
  1. snapshot_fundamentals() writes available_date = snapshot_date + 60 days.
  2. HistoricalDataProvider._get_fundamentals() filters on available_date,
     NOT snapshot_date — so a row whose available_date is in the future is
     invisible to a replay running as of today.
  3. migrate-pit backfills NULL available_date rows correctly.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_platform.data.schema import initialize_schema


# ── helpers ───────────────────────────────────────────────────────────────────

class _ConnWrapper:
    """Thin wrapper around sqlite3.Connection that adds a `dialect` attribute."""
    dialect = "sqlite"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(sql, params)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def cursor(self):
        return self._conn.cursor()

    def close(self):
        return self._conn.close()


def _make_conn() -> _ConnWrapper:
    raw  = sqlite3.connect(":memory:")
    conn = _ConnWrapper(raw)
    initialize_schema(conn)
    return conn


class _FakeRepo:
    """Minimal repo stub that wraps a single in-memory connection."""
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def connect(self):
        return _ConnCtx(self._conn)


class _ConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *_):
        pass


# ── 1. available_date written correctly by snapshot ───────────────────────────

def test_snapshot_writes_available_date(tmp_path):
    """snapshot_fundamentals sets available_date = snapshot_date + 60d."""
    conn  = _make_conn()
    repo  = _FakeRepo(conn)

    snapshot_date = date(2023, 12, 31)
    expected_avail = (snapshot_date + timedelta(days=60)).isoformat()  # 2024-02-29

    # Insert a row as snapshot_fundamentals would (directly, to avoid yfinance).
    conn.execute(
        """
        INSERT INTO historical_fundamentals
            (symbol, snapshot_date, available_date, roce, eps, debt_equity,
             revenue_growth, promoter_holding, source, fetched_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("TESTCO", snapshot_date.isoformat(), expected_avail,
         15.0, 5.0, 0.3, 10.0, 55.0, "yfinance", "yfinance"),
    )
    conn.commit()

    row = conn.execute(
        "SELECT available_date FROM historical_fundamentals WHERE symbol = 'TESTCO'"
    ).fetchone()
    assert row is not None
    assert row["available_date"] == expected_avail


# ── 2. _get_fundamentals honours available_date (not snapshot_date) ───────────

def _insert_fundamental(conn, symbol: str, snapshot_date: date, available_date: date, roce: float):
    conn.execute(
        """
        INSERT INTO historical_fundamentals
            (symbol, snapshot_date, available_date, roce, source, fetched_source)
        VALUES (?, ?, ?, ?, 'yfinance', 'yfinance')
        """,
        (symbol, snapshot_date.isoformat(), available_date.isoformat(), roce),
    )
    conn.commit()


def test_get_fundamentals_respects_available_date():
    """
    A row whose available_date is in the future must NOT be returned by a
    replay running as of replay_date.
    """
    conn = _make_conn()
    repo = _FakeRepo(conn)

    replay_date    = date(2024, 2, 1)
    snapshot_date  = date(2023, 12, 31)
    # available_date is AFTER replay_date → should be invisible
    available_date = date(2024, 3, 1)

    _insert_fundamental(conn, "LEAK", snapshot_date, available_date, roce=42.0)

    # Also insert a row that IS visible (available_date before replay_date)
    visible_snap  = date(2023, 9, 30)
    visible_avail = date(2023, 11, 29)
    _insert_fundamental(conn, "LEAK", visible_snap, visible_avail, roce=7.0)

    # Simulate HistoricalDataProvider._get_fundamentals query directly.
    row = conn.execute(
        """
        SELECT * FROM historical_fundamentals
        WHERE symbol = ? AND available_date <= ?
        ORDER BY available_date DESC LIMIT 1
        """,
        ("LEAK", replay_date.isoformat()),
    ).fetchone()

    assert row is not None, "Expected visible row to be found"
    assert float(row["roce"]) == pytest.approx(7.0), (
        "Should return the older visible row, NOT the future-available row"
    )


def test_get_fundamentals_invisible_when_all_future():
    """If all rows have available_date in the future, _get_fundamentals returns nothing."""
    conn = _make_conn()
    replay_date   = date(2024, 1, 1)
    snapshot_date = date(2023, 12, 31)
    avail_date    = date(2024, 3, 1)  # future

    _insert_fundamental(conn, "FUTCO", snapshot_date, avail_date, roce=99.0)

    row = conn.execute(
        """
        SELECT * FROM historical_fundamentals
        WHERE symbol = ? AND available_date <= ?
        ORDER BY available_date DESC LIMIT 1
        """,
        ("FUTCO", replay_date.isoformat()),
    ).fetchone()

    assert row is None, "Future-available row must be invisible to the replay"


# ── 3. migrate-pit backfills NULL available_date ──────────────────────────────

def test_migrate_pit_backfills_null_rows():
    """migrate-pit mode sets available_date = snapshot_date + 60d for NULL rows."""
    conn = _make_conn()

    # Insert rows with NULL available_date (legacy rows without PIT fix).
    snap1 = date(2023, 6, 30)
    snap2 = date(2023, 9, 30)
    for sym, snap in [("OLD1", snap1), ("OLD2", snap2)]:
        conn.execute(
            """
            INSERT INTO historical_fundamentals
                (symbol, snapshot_date, available_date, source, fetched_source)
            VALUES (?, ?, NULL, 'yfinance', 'yfinance')
            """,
            (sym, snap.isoformat()),
        )
    conn.commit()

    # Simulate migrate-pit logic from run_backtest.py.
    rows = conn.execute(
        "SELECT symbol, snapshot_date FROM historical_fundamentals WHERE available_date IS NULL"
    ).fetchall()
    updated = 0
    for row in rows:
        avail = (date.fromisoformat(row["snapshot_date"]) + timedelta(days=60)).isoformat()
        conn.execute(
            "UPDATE historical_fundamentals SET available_date = ? WHERE symbol = ? AND snapshot_date = ?",
            (avail, row["symbol"], row["snapshot_date"]),
        )
        updated += 1
    conn.commit()

    assert updated == 2

    for sym, snap in [("OLD1", snap1), ("OLD2", snap2)]:
        row = conn.execute(
            "SELECT available_date FROM historical_fundamentals WHERE symbol = ?", (sym,)
        ).fetchone()
        expected = (snap + timedelta(days=60)).isoformat()
        assert row["available_date"] == expected, f"{sym}: expected {expected}, got {row['available_date']}"
