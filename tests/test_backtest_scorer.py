"""
Unit tests for backtest/scorer.py — forward-return computation and hit-rate logic.
Tests run entirely against an in-memory SQLite database; no network calls.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.config import AppConfig
from stock_platform.data.repository import PlatformRepository


def _make_repo() -> PlatformRepository:
    tmp = tempfile.mktemp(suffix=".db")
    cfg = AppConfig(db_path=Path(tmp))
    repo = PlatformRepository(db_path=Path(tmp))
    repo.initialize()
    return repo


def _insert_price(conn, symbol: str, dt: date, price: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO historical_prices(symbol, date, close_price) VALUES (?,?,?)",
        (symbol, dt.isoformat(), price),
    )


def _insert_rec(conn, run_id: str, symbol: str, rec_date: date, action: str, confidence: str = "GREEN", score: float = 0.7) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO backtest_recommendations
           (run_id, symbol, recommendation_date, action, confidence_band, quality_score)
           VALUES (?,?,?,?,?,?)""",
        (run_id, symbol, rec_date.isoformat(), action, confidence, score),
    )


def _insert_run(conn, run_id: str, start: date, end: date) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO backtest_runs
           (run_id, start_date, end_date, weights_hash, total_recommendations, created_at)
           VALUES (?,?,?,?,?,?)""",
        (run_id, start.isoformat(), end.isoformat(), None, 1, "2026-01-01T00:00:00Z"),
    )


class TestScorerForwardReturn(unittest.TestCase):

    def setUp(self):
        self.repo = _make_repo()
        self.run_id = "test-run-001"
        self.base_date = date(2024, 1, 8)   # a Monday

        with self.repo.connect() as conn:
            _insert_run(conn, self.run_id, self.base_date, date(2025, 1, 8))

            # Stock that beats NIFTY by >2% at 6m (13 weeks)
            for week in range(30):
                dt = self.base_date + timedelta(weeks=week)
                stock_price = 100.0 * (1 + week * 0.005)    # +0.5%/week ≈ +30% total
                nifty_price = 100.0 * (1 + week * 0.002)    # +0.2%/week ≈ +12% total
                _insert_price(conn, "WINNER", dt, stock_price)
                _insert_price(conn, "NIFTY",  dt, nifty_price)

            # Stock that underperforms NIFTY at 6m
            for week in range(30):
                dt = self.base_date + timedelta(weeks=week)
                _insert_price(conn, "LOSER", dt, 100.0 * (1 + week * 0.001))   # +0.1%/week

            # Recommendation on base_date for both stocks
            _insert_rec(conn, self.run_id, "WINNER", self.base_date, "ACCUMULATE", "GREEN", 0.8)
            _insert_rec(conn, self.run_id, "LOSER",  self.base_date, "ACCUMULATE", "YELLOW", 0.6)
            conn.commit()

    def test_winner_marked_as_hit(self):
        from backtest.scorer import score_run
        summary = score_run(self.repo, self.run_id)
        self.assertIsNotNone(summary.get("hit_rate_6m"))
        # At least one hit (WINNER beats NIFTY by >2%)
        self.assertGreater(summary["hit_rate_6m"], 0.0)

    def test_summary_keys_present(self):
        from backtest.scorer import score_run
        summary = score_run(self.repo, self.run_id)
        for key in ("run_id", "total_recommendations", "hit_rate_3m", "hit_rate_6m", "hit_rate_12m"):
            self.assertIn(key, summary)

    def test_empty_run_returns_gracefully(self):
        from backtest.scorer import score_run
        repo = _make_repo()
        with repo.connect() as conn:
            _insert_run(conn, "empty-run", date(2024, 1, 1), date(2024, 6, 1))
            conn.commit()
        summary = score_run(repo, "empty-run")
        self.assertEqual(summary["total"], 0)

    def test_forward_return_computation(self):
        """_forward_return helper: correct percentage and None handling."""
        from backtest.scorer import _forward_return
        self.assertAlmostEqual(_forward_return(100.0, 110.0), 10.0)
        self.assertAlmostEqual(_forward_return(100.0, 90.0), -10.0)
        self.assertIsNone(_forward_return(100.0, None))
        self.assertIsNone(_forward_return(0.0, 110.0))


class TestScorerByConfidenceBand(unittest.TestCase):

    def setUp(self):
        self.repo   = _make_repo()
        self.run_id = "band-test-001"
        self.base   = date(2024, 2, 5)

        with self.repo.connect() as conn:
            _insert_run(conn, self.run_id, self.base, date(2025, 2, 5))
            for week in range(30):
                dt = self.base + timedelta(weeks=week)
                _insert_price(conn, "NIFTY",  dt, 100.0 * (1 + week * 0.002))
                _insert_price(conn, "GREEN1", dt, 100.0 * (1 + week * 0.006))
                _insert_price(conn, "YEL1",   dt, 100.0 * (1 + week * 0.001))

            _insert_rec(conn, self.run_id, "GREEN1", self.base, "ACCUMULATE", "GREEN",  0.85)
            _insert_rec(conn, self.run_id, "YEL1",   self.base, "ACCUMULATE", "YELLOW", 0.55)
            conn.commit()

    def test_by_confidence_populated(self):
        from backtest.scorer import score_run
        summary = score_run(self.repo, self.run_id)
        self.assertIn("by_confidence", summary)
        bands = summary["by_confidence"]
        # GREEN band: GREEN1 beats NIFTY by >2% → hit
        if "GREEN" in bands:
            self.assertGreater(bands["GREEN"]["hit_rate_6m"] or 0, 0.0)


class TestWeightCombinations(unittest.TestCase):
    def test_all_sum_to_one(self):
        from backtest.calibrate import _weight_combinations
        combos = _weight_combinations()
        self.assertGreater(len(combos), 100)
        for combo in combos[:50]:   # spot-check first 50
            self.assertAlmostEqual(sum(combo), 1.0, places=10)

    def test_count_reasonable(self):
        from backtest.calibrate import _weight_combinations
        combos = _weight_combinations()
        # C(24,4) = 10626 valid 5-tuples in {0,0.05,...,1.0} summing to 1.0
        self.assertGreater(len(combos), 500)
        self.assertLess(len(combos), 15000)


if __name__ == "__main__":
    unittest.main()
