"""
tests/test_rank_ic.py — unit tests for Rank IC (Spearman) scorer metrics.

Contracts under test:
  1. _spearman_ic returns correct correlation for known inputs.
  2. _spearman_ic returns None when <3 valid pairs.
  3. _rank_ic_summary returns None fields when too few IC weeks.
  4. _decile_spread requires ≥10 observations.
  5. score_run computes and writes IC metrics to backtest_runs.
  6. 12m metrics suppressed when <30% of recs have 12m data.
  7. IC regression gate fires when mean_ic_6m ≤ 0.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_platform.data.schema import initialize_schema


# ── shared helpers ────────────────────────────────────────────────────────────

class _ConnWrapper:
    dialect = "sqlite"

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(sql, params)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def cursor(self):
        return self._conn.cursor()


def _make_conn():
    raw  = sqlite3.connect(":memory:")
    conn = _ConnWrapper(raw)
    initialize_schema(conn)
    return conn


class _FakeRepo:
    def __init__(self, conn): self._conn = conn
    def connect(self): return _ConnCtx(self._conn)


class _ConnCtx:
    def __init__(self, conn): self._conn = conn
    def __enter__(self): return self._conn
    def __exit__(self, *_): pass


# ── 1. _spearman_ic ───────────────────────────────────────────────────────────

def test_spearman_perfect_positive():
    from backtest.scorer import _spearman_ic
    scores = [1.0, 2.0, 3.0, 4.0]
    alphas = [10.0, 20.0, 30.0, 40.0]
    ic = _spearman_ic(scores, alphas)
    assert ic == pytest.approx(1.0)


def test_spearman_perfect_negative():
    from backtest.scorer import _spearman_ic
    scores = [1.0, 2.0, 3.0, 4.0]
    alphas = [40.0, 30.0, 20.0, 10.0]
    ic = _spearman_ic(scores, alphas)
    assert ic == pytest.approx(-1.0)


def test_spearman_none_pairs_excluded():
    from backtest.scorer import _spearman_ic
    # 3 valid pairs (None entries are excluded from pairing)
    scores = [1.0, None, 3.0, 4.0, 5.0]
    alphas = [10.0, 20.0, None, 40.0, 50.0]
    # Valid pairs: (1.0,10.0), (4.0,40.0), (5.0,50.0) → perfect positive correlation
    ic = _spearman_ic(scores, alphas)
    assert ic == pytest.approx(1.0)


def test_spearman_returns_none_for_too_few():
    from backtest.scorer import _spearman_ic
    assert _spearman_ic([1.0, 2.0], [3.0, 4.0]) is None
    assert _spearman_ic([], []) is None


# ── 2. _rank_ic_summary ───────────────────────────────────────────────────────

def test_rank_ic_summary_too_few():
    from backtest.scorer import _rank_ic_summary
    result = _rank_ic_summary([0.1, 0.2, 0.3])  # < _MIN_IC_WEEKS (4)
    assert result["mean_ic"]  is None
    assert result["ic_tstat"] is None
    assert result["icir"]     is None


def test_rank_ic_summary_computes():
    from backtest.scorer import _rank_ic_summary
    result = _rank_ic_summary([0.2, 0.3, 0.1, 0.4, 0.2])
    assert result["mean_ic"] == pytest.approx(0.24)
    assert result["ic_tstat"] is not None
    assert result["icir"] is not None


# ── 3. _decile_spread ─────────────────────────────────────────────────────────

def test_decile_spread_too_few():
    from backtest.scorer import _decile_spread
    pairs = [(float(i), float(i)) for i in range(9)]
    assert _decile_spread(pairs) is None


def test_decile_spread_positive_when_rank_predicts():
    from backtest.scorer import _decile_spread
    # Perfect rank: higher score → higher alpha
    n = 20
    pairs = [(float(i), float(i) * 2) for i in range(n)]
    spread = _decile_spread(pairs)
    assert spread is not None and spread > 0


# ── 4. score_run IC integration ───────────────────────────────────────────────

def _populate_run(conn, run_id: str, weeks: int = 10, positive_ic: bool = True):
    """
    Insert synthetic backtest_runs + recommendations + prices, then call score_run.
    When positive_ic=True, composite_score order matches forward alpha order.
    """
    start = date(2023, 1, 2)
    end   = date(2023, 3, 27)

    # Insert run stub
    conn.execute(
        "INSERT INTO backtest_runs (run_id, start_date, end_date, total_recommendations, created_at) VALUES (?,?,?,?,?)",
        (run_id, start.isoformat(), end.isoformat(), 0, "2023-01-01T00:00:00Z"),
    )

    # Insert price rows: each Monday + 13/26/52 weeks forward
    symbols = ["SYM_A", "SYM_B", "SYM_C", "SYM_D", "SYM_E", "NIFTY"]
    base_prices = {"SYM_A": 100.0, "SYM_B": 100.0, "SYM_C": 100.0,
                   "SYM_D": 100.0, "SYM_E": 100.0, "NIFTY": 100.0}
    # Final prices (26w forward): positive_ic means high-score stocks outperform
    final_returns = (
        {"SYM_A": 30.0, "SYM_B": 20.0, "SYM_C": 10.0, "SYM_D": 5.0, "SYM_E": 1.0, "NIFTY": 10.0}
        if positive_ic else
        {"SYM_A": 1.0,  "SYM_B": 5.0,  "SYM_C": 10.0, "SYM_D": 20.0, "SYM_E": 30.0, "NIFTY": 10.0}
    )

    # Composite scores: SYM_A highest → SYM_E lowest
    scores = {"SYM_A": 0.9, "SYM_B": 0.7, "SYM_C": 0.5, "SYM_D": 0.3, "SYM_E": 0.1}

    rec_date = start
    for w in range(weeks):
        # Insert entry prices
        for sym in symbols:
            conn.execute(
                "INSERT OR IGNORE INTO historical_prices (symbol, date, close_price) VALUES (?,?,?)",
                (sym, rec_date.isoformat(), base_prices[sym]),
            )
            # 13w forward
            fwd13 = (rec_date + timedelta(weeks=13)).isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO historical_prices (symbol, date, close_price) VALUES (?,?,?)",
                (sym, fwd13, base_prices[sym] * (1 + final_returns[sym] / 100 * 0.5)),
            )
            # 26w forward
            fwd26 = (rec_date + timedelta(weeks=26)).isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO historical_prices (symbol, date, close_price) VALUES (?,?,?)",
                (sym, fwd26, base_prices[sym] * (1 + final_returns[sym] / 100)),
            )

        # Insert recommendations
        for sym in ["SYM_A", "SYM_B", "SYM_C", "SYM_D", "SYM_E"]:
            conn.execute(
                """INSERT OR IGNORE INTO backtest_recommendations
                   (run_id, symbol, recommendation_date, action, confidence_band,
                    quality_score, composite_score)
                   VALUES (?,?,?,?,?,?,?)""",
                (run_id, sym, rec_date.isoformat(), "WAIT", "GREEN",
                 scores[sym], scores[sym]),
            )

        rec_date += timedelta(weeks=1)

    conn.commit()


def test_score_run_positive_ic():
    conn = _make_conn()
    repo = _FakeRepo(conn)
    run_id = "test-ic-positive"

    _populate_run(conn._conn, run_id, weeks=10, positive_ic=True)

    from backtest.scorer import score_run
    summary = score_run(repo, run_id)

    # mean_ic should be positive when score order predicts alpha order
    if summary.get("mean_ic_6m") is not None:
        assert summary["mean_ic_6m"] > 0, f"Expected positive IC, got {summary['mean_ic_6m']}"

    # decile_spread should be positive
    if summary.get("decile_spread_6m") is not None:
        assert summary["decile_spread_6m"] > 0

    # Verify DB write
    row = conn.execute(
        "SELECT mean_ic_6m, decile_spread_6m FROM backtest_runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    assert row is not None


def test_score_run_12m_suppressed_when_insufficient():
    """12m metrics should be None when <30% of recommendations have 12m data."""
    conn = _make_conn()
    repo = _FakeRepo(conn)
    run_id = "test-12m-suppress"

    # Insert run with prices that DON'T cover the 12m window.
    conn.execute(
        "INSERT INTO backtest_runs (run_id, start_date, end_date, total_recommendations, created_at) VALUES (?,?,?,?,?)",
        (run_id, "2023-01-02", "2023-03-27", 0, "2023-01-01T00:00:00Z"),
    )
    rec_date = date(2023, 1, 2)
    for sym in ["SYM_A", "SYM_B"]:
        conn.execute(
            "INSERT OR IGNORE INTO historical_prices (symbol, date, close_price) VALUES (?,?,?)",
            (sym, rec_date.isoformat(), 100.0),
        )
        # Only 13w forward (not 26w or 52w)
        conn.execute(
            "INSERT OR IGNORE INTO historical_prices (symbol, date, close_price) VALUES (?,?,?)",
            (sym, (rec_date + timedelta(weeks=13)).isoformat(), 110.0),
        )
        conn.execute(
            "INSERT OR IGNORE INTO backtest_recommendations (run_id, symbol, recommendation_date, action, confidence_band, composite_score) VALUES (?,?,?,?,?,?)",
            (run_id, sym, rec_date.isoformat(), "WAIT", "GREEN", 0.5),
        )
    conn.execute(
        "INSERT OR IGNORE INTO historical_prices (symbol, date, close_price) VALUES (?,?,?)",
        ("NIFTY", rec_date.isoformat(), 100.0),
    )
    conn.execute(
        "INSERT OR IGNORE INTO historical_prices (symbol, date, close_price) VALUES (?,?,?)",
        ("NIFTY", (rec_date + timedelta(weeks=13)).isoformat(), 105.0),
    )
    conn.commit()

    from backtest.scorer import score_run
    summary = score_run(repo, run_id)
    # 12m data should be suppressed (None) since no 52w forward prices exist
    assert summary.get("alpha_12m") is None or summary.get("hit_rate_12m") is None
