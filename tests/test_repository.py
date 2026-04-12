from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.data.repository import PlatformRepository
from stock_platform.models import MonitoringAction
from stock_platform.providers.live import LiveMarketDataProvider


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

    def test_monitoring_rows_include_joined_overlap_and_urgency(self) -> None:
        self.repo.replace_overlap_scores(
            [
                {
                    "symbol": "KWIL",
                    "overlap_pct": 2.75,
                    "band": "FLAG",
                    "attribution": [],
                }
            ]
        )
        self.repo.save_monitoring_actions(
            "monitor-test",
            [
                MonitoringAction(
                    symbol="KWIL",
                    action="WAIT 25 days then EXIT",
                    severity="MEDIUM",
                    urgency="MEDIUM",
                    rationale="Readable rationale",
                    payload={},
                )
            ],
        )

        rows = self.repo.list_monitoring_actions()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["urgency"], "MEDIUM")
        self.assertAlmostEqual(rows[0]["overlap_pct"], 2.75)


class LiveProviderTests(unittest.TestCase):
    def test_live_provider_normalizes_exchange_suffixes(self) -> None:
        provider = LiveMarketDataProvider()
        self.assertEqual(provider.normalize_symbol("bel.ns"), "BEL")
        self.assertEqual(provider.normalize_symbol(" tcs.nse "), "TCS")
        self.assertEqual(provider.normalize_symbol("TATAMOTORS"), "TMCV")
        self.assertEqual(provider.normalize_symbol("HDFBANEQ"), "HDFCBANK")
        self.assertEqual(provider.normalize_symbol("ICIBAN"), "ICICIBANK")
        self.assertEqual(provider.normalize_symbol("LARTOU"), "LT")

    def test_live_provider_uses_fallback_universe_when_index_download_is_empty(self) -> None:
        provider = LiveMarketDataProvider()
        fallback_rows = [{"symbol": "BEL", "company_name": "Bharat Electronics", "sector": "Capital Goods"}]

        with (
            patch.object(provider, "_download_index_csv", return_value=[]),
            patch.object(provider, "_load_stale_index_cache", return_value=[]),
            patch.object(provider, "_combined_universe", return_value=[]),
            patch.object(provider, "_fallback_index_members", return_value=fallback_rows),
        ):
            rows = provider.get_index_members("NIFTY200")

        self.assertEqual(rows, fallback_rows)

    def test_live_provider_raises_when_all_universes_are_unavailable(self) -> None:
        provider = LiveMarketDataProvider()

        with (
            patch.object(provider, "_download_index_csv", return_value=[]),
            patch.object(provider, "_load_stale_index_cache", return_value=[]),
            patch.object(provider, "_combined_universe", return_value=[]),
            patch.object(provider, "_fallback_index_members", return_value=[]),
        ):
            with self.assertRaisesRegex(ValueError, "Unable to load constituents"):
                provider.get_index_members("NIFTY200")


if __name__ == "__main__":
    unittest.main()
