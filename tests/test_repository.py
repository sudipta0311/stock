from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.data.repository import PlatformRepository
from stock_platform.providers.demo import DemoDataProvider


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = ROOT / "data" / "test_tmp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.temp_dir / "test.db"
        self.repo = PlatformRepository(self.db_path)
        try:
            self.repo.initialize()
        except sqlite3.OperationalError as exc:
            self.skipTest(f"SQLite unavailable in this environment: {exc}")

    def tearDown(self) -> None:
        pass

    def test_state_round_trip(self) -> None:
        self.repo.set_state("user_preferences", {"macro_thesis": "Defence"})
        state = self.repo.get_state("user_preferences")
        self.assertEqual(state["macro_thesis"], "Defence")

    def test_normalized_exposure_round_trip(self) -> None:
        self.repo.replace_normalized_exposure(
            [
                {
                    "symbol": "BEL",
                    "company_name": "Bharat Electronics",
                    "sector": "Defence",
                    "total_weight": 4.2,
                    "source_mix": {"mutual_fund": 3.1, "etf": 1.1},
                    "attribution": [{"instrument_name": "HDFC Defence Fund", "lookthrough_weight": 3.1}],
                }
            ]
        )
        rows = self.repo.list_normalized_exposure()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "BEL")
        self.assertAlmostEqual(rows[0]["total_weight"], 4.2)


class DemoProviderTests(unittest.TestCase):
    def test_demo_provider_has_index_members(self) -> None:
        provider = DemoDataProvider()
        members = provider.get_index_members("NIFTY50")
        self.assertGreaterEqual(len(members), 5)
        self.assertIn("symbol", members[0])


if __name__ == "__main__":
    unittest.main()
