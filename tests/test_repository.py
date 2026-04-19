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

    def test_cache_round_trip(self) -> None:
        self.repo.set_cache("index_constituents:NIFTY200", [{"symbol": "BEL"}], ttl_seconds=60)
        cached = self.repo.get_cache("index_constituents:NIFTY200")
        self.assertEqual(cached, [{"symbol": "BEL"}])

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

    def test_direct_equity_holdings_round_trip_uses_repository_store(self) -> None:
        saved = self.repo.upsert_direct_equity_holdings(
            [
                {
                    "symbol": "HDFBANEQ",
                    "quantity": 220.0,
                    "avg_buy_price": 712.2691,
                    "current_price": 810.30,
                    "buy_date": "2026-04-12",
                    "source": "broker_csv",
                }
            ]
        )

        self.assertEqual(saved, 1)

        rows = self.repo.list_direct_equity_holdings()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "HDFCBANK")
        self.assertAlmostEqual(rows[0]["avg_buy_price"], 712.2691)
        self.assertAlmostEqual(rows[0]["quantity"], 220.0)

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

    def test_monitoring_rows_prefer_payload_overlap_when_db_join_is_zero(self) -> None:
        self.repo.replace_overlap_scores(
            [
                {
                    "symbol": "HDFCBANK",
                    "overlap_pct": 0.0,
                    "band": "GREEN",
                    "attribution": [],
                }
            ]
        )
        self.repo.save_monitoring_actions(
            "monitor-test-payload",
            [
                MonitoringAction(
                    symbol="HDFCBANK",
                    action="HOLD",
                    severity="LOW",
                    urgency="LOW",
                    rationale="Readable rationale",
                    payload={"overlap_pct": 3.13},
                )
            ],
        )

        rows = self.repo.list_monitoring_actions()

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["overlap_pct"], 3.13)

    def test_replace_normalized_exposure_syncs_overlap_scores(self) -> None:
        # replace_normalized_exposure now atomically refreshes overlap scores so
        # load_portfolio_context can read them cheaply without recomputing.
        self.repo.replace_normalized_exposure(
            [
                {
                    "symbol": "HDFCBANK",
                    "company_name": "HDFC Bank",
                    "sector": "Banks",
                    "total_weight": 1.875,
                    "source_mix": {"direct_equity": 1.875},
                    "attribution": [
                        {
                            "instrument_name": "Direct",
                            "symbol": "HDFCBANK",
                            "lookthrough_weight": 1.875,
                            "source": "direct_equity",
                        },
                    ],
                },
                {
                    "symbol": "HDFCBANKLIMITED",
                    "company_name": "HDFCBANKLIMITED",
                    "sector": "Banks",
                    "total_weight": 3.13,
                    "source_mix": {"mutual_fund": 3.13},
                    "attribution": [
                        {
                            "instrument_name": "Fund A",
                            "symbol": "HDFCBANKLIMITED",
                            "lookthrough_weight": 3.13,
                            "source": "mutual_fund",
                        },
                    ],
                }
            ]
        )

        # Overlap scores are computed atomically inside replace_normalized_exposure.
        stored = self.repo.list_overlap_scores()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["symbol"], "HDFCBANK")
        self.assertAlmostEqual(stored[0]["overlap_pct"], 3.13)
        self.assertEqual(stored[0]["band"], "HARD_EXCLUDE")

        # load_portfolio_context returns the pre-computed scores without recomputing.
        context = self.repo.load_portfolio_context()
        self.assertEqual(len(context["overlap_scores"]), 1)
        self.assertEqual(context["overlap_scores"][0]["symbol"], "HDFCBANK")
        self.assertAlmostEqual(context["overlap_scores"][0]["overlap_pct"], 3.13)
        self.assertEqual(context["overlap_scores"][0]["band"], "HARD_EXCLUDE")


class LiveProviderTests(unittest.TestCase):
    def test_live_provider_normalizes_exchange_suffixes(self) -> None:
        provider = LiveMarketDataProvider()
        self.assertEqual(provider.normalize_symbol("bel.ns"), "BEL")
        self.assertEqual(provider.normalize_symbol(" tcs.nse "), "TCS")
        self.assertEqual(provider.normalize_symbol("TATAMOTORS"), "TMCV")
        self.assertEqual(provider.normalize_symbol("HDFBANEQ"), "HDFCBANK")
        self.assertEqual(provider.normalize_symbol("HDFCBANKLIMITED"), "HDFCBANK")
        self.assertEqual(provider.normalize_symbol("ICIBAN"), "ICICIBANK")
        self.assertEqual(provider.normalize_symbol("ICICIBANKLIMITED"), "ICICIBANK")
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
