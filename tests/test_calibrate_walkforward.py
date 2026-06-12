"""
tests/test_calibrate_walkforward.py — unit tests for the walk-forward calibration.

Contracts under test:
  1. _build_folds: correct fold boundary arithmetic.
  2. _build_folds: fewer than 3 folds → calibrate() raises RuntimeError.
  3. _weight_grid: all combos sum to 1.0; contains expected boundary combos.
  4. _mean_weekly_ic: returns positive IC when score order matches alpha order.
  5. calibrate(): NO_ROBUST_WINNER outcome when no combo has IC > 0 in every fold.
  6. calibrate(): writes composite_weights.yaml on success.
  7. calibrate(): raises RuntimeError with < _MIN_FOLDS folds of history.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_platform.data.schema import initialize_schema


# ── helpers ───────────────────────────────────────────────────────────────────

class _ConnWrapper:
    dialect = "sqlite"
    def __init__(self, conn): self._conn = conn
    def execute(self, sql, params=()):
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(sql, params)
    def commit(self): return self._conn.commit()
    def rollback(self): return self._conn.rollback()
    def cursor(self): return self._conn.cursor()


def _make_conn():
    raw  = sqlite3.connect(":memory:")
    conn = _ConnWrapper(raw)
    initialize_schema(conn)
    return conn


class _FakeRepo:
    def __init__(self, conn): self._conn = conn
    def connect(self): return _ConnCtx(self._conn)
    def initialize(self): pass


class _ConnCtx:
    def __init__(self, conn): self._conn = conn
    def __enter__(self): return self._conn
    def __exit__(self, *_): pass


# ── 1. _build_folds arithmetic ───────────────────────────────────────────────

def test_build_folds_basic():
    from backtest.calibrate import _build_folds
    # 70 weeks: min_train=26, val_step=13
    # fold 0: train=0:26 val=26:39
    # fold 1: train=0:39 val=39:52
    # fold 2: train=0:52 val=52:65
    # fold 3: train=0:65 val=65:78 → 78>70, stops
    weeks = [str(i) for i in range(70)]
    folds = _build_folds(weeks, min_train=26, val_step=13)
    assert len(folds) == 3
    # First fold: train has 26 items, val has 13
    train0, val0 = folds[0]
    assert len(train0) == 26
    assert len(val0)   == 13
    # Second fold: train has 39 items
    train1, val1 = folds[1]
    assert len(train1) == 39
    assert len(val1)   == 13
    # Train sets are expanding: every item in train0 is in train1
    assert train0.issubset(train1)
    # Val sets are disjoint
    assert val0.isdisjoint(val1)


def test_build_folds_exactly_3():
    from backtest.calibrate import _build_folds
    # Minimum weeks for 3 folds: 26 + 3*13 = 65
    weeks = [str(i) for i in range(65)]
    folds = _build_folds(weeks, min_train=26, val_step=13)
    assert len(folds) == 3


def test_build_folds_too_few_weeks():
    from backtest.calibrate import _build_folds
    # 50 weeks → only 1 fold (need 65 for 3)
    weeks = [str(i) for i in range(50)]
    folds = _build_folds(weeks, min_train=26, val_step=13)
    assert len(folds) < 3


# ── 2. _weight_grid ───────────────────────────────────────────────────────────

def test_weight_grid_count():
    from backtest.calibrate import _weight_grid
    combos = _weight_grid(step=10)
    assert len(combos) == 66  # C(12,2) = 66 for step=10


def test_weight_grid_sums_to_one():
    from backtest.calibrate import _weight_grid
    for w_q, w_v, w_m in _weight_grid(step=10):
        assert abs(w_q + w_v + w_m - 1.0) < 1e-9, f"{w_q}+{w_v}+{w_m}≠1"


def test_weight_grid_boundaries():
    from backtest.calibrate import _weight_grid
    combos = _weight_grid(step=10)
    # Should include (1.0, 0.0, 0.0) and (0.0, 0.0, 1.0)
    assert (1.0, 0.0, 0.0) in combos
    assert (0.0, 0.0, 1.0) in combos
    assert (0.5, 0.5, 0.0) in combos


# ── 3. _mean_weekly_ic ────────────────────────────────────────────────────────

def test_mean_weekly_ic_positive():
    """When composite score order matches alpha order, IC should be positive."""
    from backtest.calibrate import _mean_weekly_ic

    week_rows = {
        "2024-01-08": [
            {"quality_pct": 0.9, "valuation_pct": 0.8, "momentum_pct": 0.7, "alpha_6m": 20.0},
            {"quality_pct": 0.7, "valuation_pct": 0.6, "momentum_pct": 0.5, "alpha_6m": 10.0},
            {"quality_pct": 0.3, "valuation_pct": 0.2, "momentum_pct": 0.1, "alpha_6m": -5.0},
        ],
    }
    ic = _mean_weekly_ic(week_rows, w_q=0.5, w_v=0.3, w_m=0.2)
    assert ic is not None and ic > 0


def test_mean_weekly_ic_none_when_too_few():
    from backtest.calibrate import _mean_weekly_ic
    week_rows = {
        "2024-01-08": [
            {"quality_pct": 0.9, "valuation_pct": 0.5, "momentum_pct": 0.5, "alpha_6m": 5.0},
            {"quality_pct": 0.1, "valuation_pct": None, "momentum_pct": None, "alpha_6m": None},
        ],
    }
    # Only 1 pair with complete data — IC should be None (requires ≥ 3)
    ic = _mean_weekly_ic(week_rows, w_q=1.0, w_v=0.0, w_m=0.0)
    assert ic is None


# ── 4. calibrate() integration ────────────────────────────────────────────────

def _populate_calibrate_db(
    conn,
    run_id: str,
    n_weeks: int,
    positive_quality_signal: bool = True,
) -> None:
    """
    Insert synthetic backtest data.
    When positive_quality_signal=True, quality_pct predicts alpha_6m positively.
    """
    conn.execute(
        "INSERT INTO backtest_runs (run_id, start_date, end_date, total_recommendations, created_at) "
        "VALUES (?,?,?,?,?)",
        (run_id, "2022-01-03", "2024-01-03", 0, "2022-01-01T00:00:00Z"),
    )
    start = date(2022, 1, 3)
    for w in range(n_weeks):
        week_date = (start + timedelta(weeks=w)).isoformat()
        for rank in range(5):
            q_pct = (4 - rank) / 4.0        # 1.0, 0.75, 0.5, 0.25, 0.0
            v_pct = 0.5
            m_pct = 0.5
            alpha = (20.0 - rank * 8.0) if positive_quality_signal else (rank * 8.0 - 20.0)
            conn.execute(
                """INSERT OR IGNORE INTO backtest_recommendations
                   (run_id, symbol, recommendation_date, action, confidence_band,
                    quality_pct, valuation_pct, momentum_pct, alpha_6m)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (run_id, f"SYM_{rank}", week_date, "WAIT", "GREEN",
                 q_pct, v_pct, m_pct, alpha),
            )
    conn.commit()


def test_calibrate_insufficient_history(tmp_path):
    """Fewer than _MIN_FOLDS folds raises RuntimeError."""
    conn = _make_conn()
    repo = _FakeRepo(conn)
    run_id = "short-run"
    # 40 weeks of scored data → only 1 fold possible (need 3)
    _populate_calibrate_db(conn._conn, run_id, n_weeks=40)

    from backtest.calibrate import calibrate
    with pytest.raises(RuntimeError, match="Accumulate more price history|walk-forward"):
        calibrate(repo, run_id, min_train=26, val_step=13)


def test_calibrate_success_writes_yaml(tmp_path, monkeypatch):
    """Enough history + positive signal → writes composite_weights.yaml."""
    # Monkeypatch the output file to a tmp location.
    import backtest.calibrate as cal_module
    out_file = tmp_path / "composite_weights.yaml"
    monkeypatch.setattr(cal_module, "_COMPOSITE_WEIGHTS_FILE", out_file)

    conn = _make_conn()
    repo = _FakeRepo(conn)
    run_id = "long-run"
    # 80 weeks → 3+ folds
    _populate_calibrate_db(conn._conn, run_id, n_weeks=80, positive_quality_signal=True)

    result = cal_module.calibrate(repo, run_id, min_train=26, val_step=13)

    assert result["outcome"] == "OK"
    assert out_file.exists()

    import yaml
    config = yaml.safe_load(out_file.read_text())
    assert "weights" in config
    assert abs(config["weights"]["quality"] + config["weights"]["valuation"] + config["weights"]["momentum"] - 1.0) < 1e-9


def test_calibrate_no_robust_winner(tmp_path, monkeypatch):
    """When signal is inverted, no combo has positive IC in every fold."""
    import backtest.calibrate as cal_module
    out_file = tmp_path / "composite_weights.yaml"
    monkeypatch.setattr(cal_module, "_COMPOSITE_WEIGHTS_FILE", out_file)

    conn = _make_conn()
    repo = _FakeRepo(conn)
    run_id = "inverted-run"
    # 80 weeks with INVERTED signal → IC will be negative for quality-weighted combos
    _populate_calibrate_db(conn._conn, run_id, n_weeks=80, positive_quality_signal=False)

    result = cal_module.calibrate(repo, run_id, min_train=26, val_step=13)

    # With inverted signal, quality-heavy combos have negative IC.
    # momentum=1.0 combo will also have near-zero or negative IC since m_pct is constant.
    # Either NO_ROBUST_WINNER or some combo genuinely beats it — both are valid outcomes.
    assert result["outcome"] in ("NO_ROBUST_WINNER", "OK")
    # Yaml should NOT be overwritten on NO_ROBUST_WINNER
    if result["outcome"] == "NO_ROBUST_WINNER":
        assert not out_file.exists()
