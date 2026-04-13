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

from stock_platform.config import AppConfig
from stock_platform.services.engine import PlatformEngine


class EngineMonitoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = ROOT / "data" / "test_tmp" / "engine_monitoring"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.temp_dir / "platform.db"
        if self.db_path.exists():
            self.db_path.unlink()
        journal_path = self.temp_dir / "platform.db-journal"
        if journal_path.exists():
            journal_path.unlink()
        self.config = AppConfig(
            data_dir=self.temp_dir,
            db_path=self.db_path,
            turso_database_url="",
            turso_auth_token="",
            turso_sync_interval_seconds=0,
        )
        try:
            self.engine = PlatformEngine(self.config)
        except sqlite3.OperationalError as exc:
            self.skipTest(f"SQLite unavailable in this environment: {exc}")

    def test_ingest_clears_stale_monitoring_results(self) -> None:
        self.engine.seed_demo_data()
        self.engine.run_monitoring(llm_provider="anthropic")
        seeded_snapshot = self.engine.get_dashboard_snapshot()
        self.assertGreater(len(seeded_snapshot["monitoring_actions"]), 0)

        payload = self.engine.parse_portfolio_pdf(
            ROOT / "tests" / "NSDLe-CAS_109102284_FEB_2026.PDF",
            "AYFPS8467G",
        )
        self.engine.ingest_portfolio(payload)
        refreshed_snapshot = self.engine.get_dashboard_snapshot()

        self.assertEqual(refreshed_snapshot["monitoring_actions"], [])
        self.assertEqual(refreshed_snapshot["run_meta"]["monitoring"], {})

    def test_monitoring_uses_current_pdf_direct_holdings(self) -> None:
        payload = self.engine.parse_portfolio_pdf(
            ROOT / "tests" / "NSDLe-CAS_109102284_FEB_2026.PDF",
            "AYFPS8467G",
        )
        self.engine.ingest_portfolio(payload)
        self.engine.run_monitoring(llm_provider="anthropic")
        snapshot = self.engine.get_dashboard_snapshot()

        direct_symbols = {
            row["symbol"]
            for row in snapshot["portfolio"]["raw_holdings"]
            if row["holding_type"] == "direct_equity"
        }
        monitored_symbols = {row["symbol"] for row in snapshot["monitoring_actions"]}

        self.assertEqual(monitored_symbols, direct_symbols)
        self.assertNotIn("TITAN", monitored_symbols)

    def test_snapshot_restores_persisted_buy_comparison_result(self) -> None:
        comparison = {
            "anthropic": {"recommendations": [{"symbol": "BEL"}]},
            "openai": {"recommendations": [{"symbol": "HAL"}]},
            "synthesis": {"BEL": "COMBINED VERDICT: ACCUMULATE GRADUALLY."},
        }
        self.engine.repo.set_state("buy_comparison_result", comparison)

        snapshot = self.engine.get_dashboard_snapshot()

        self.assertEqual(snapshot["buy_comparison_result"], comparison)

    def test_ingest_clears_persisted_buy_comparison_result(self) -> None:
        self.engine.repo.set_state("buy_comparison_result", {"anthropic": {"recommendations": [{"symbol": "BEL"}]}})
        fake_graph = type("FakeGraph", (), {"invoke": lambda self, _: {"normalized_exposure": []}})()
        with (
            patch.object(self.engine, "run_signal_refresh", return_value={}),
            patch.object(self.engine, "_build_portfolio_graph", return_value=fake_graph),
        ):
            self.engine.ingest_portfolio({"macro_thesis": ""})
        snapshot = self.engine.get_dashboard_snapshot()

        self.assertEqual(snapshot["buy_comparison_result"], {})

    def test_comparison_synthesis_runs_for_union_of_provider_symbols(self) -> None:
        config = AppConfig(
            data_dir=self.temp_dir,
            db_path=self.db_path,
            turso_database_url="",
            turso_auth_token="",
            turso_sync_interval_seconds=0,
            anthropic_api_key="test-anthropic",
            openai_api_key="test-openai",
        )
        engine = PlatformEngine(config)

        def fake_run_buy_analysis(request: dict, llm_provider: str = "anthropic") -> dict:
            del request
            if llm_provider == "anthropic":
                return {
                    "recommendations": [
                        {
                            "symbol": "SUZLON",
                            "company_name": "Suzlon Energy",
                            "action": "BUY",
                            "rationale": "Risk analysis available.",
                            "payload": {
                                "current_price": 100.0,
                                "analyst_target": 120.0,
                                "entry_signal": "BUY",
                                "quality_score": 0.8,
                                "fin_data": {},
                            },
                        }
                    ],
                    "skipped_stocks": [],
                    "run_summary": {"recommendation_count": 1},
                }
            return {
                "recommendations": [
                    {
                        "symbol": "BEL",
                        "company_name": "Bharat Electronics",
                        "action": "BUY",
                        "rationale": "Catalyst analysis available.",
                        "payload": {
                            "current_price": 100.0,
                            "analyst_target": 120.0,
                            "entry_signal": "BUY",
                            "quality_score": 0.8,
                            "fin_data": {},
                        },
                    }
                ],
                "skipped_stocks": [],
                "run_summary": {"recommendation_count": 1},
            }

        captured_calls: list[tuple[str, str, str]] = []

        def fake_synthesise(*args, **kwargs) -> str:
            stock_name = kwargs["stock_name"]
            anthropic_rationale = kwargs["anthropic_rationale"]
            openai_rationale = kwargs["openai_rationale"]
            captured_calls.append((stock_name, anthropic_rationale, openai_rationale))
            return "• COMBINED VERDICT: ACCUMULATE GRADUALLY."

        with (
            patch.object(engine.repo, "list_signals", return_value=[{"sector": "Defence"}]),
            patch.object(engine, "run_buy_analysis", side_effect=fake_run_buy_analysis),
            patch("stock_platform.services.engine.PlatformLLM.synthesise_comparison", side_effect=fake_synthesise),
        ):
            result = engine.run_buy_analysis_comparison(
                {"index_name": "NIFTY 200", "horizon_months": 18, "risk_profile": "Aggressive", "top_n": 4}
            )

        self.assertEqual(set(result["synthesis"]), {"SUZLON", "BEL"})
        self.assertEqual(len(captured_calls), 2)
        suzlon_call = next(call for call in captured_calls if "SUZLON" in call[0])
        bel_call = next(call for call in captured_calls if "BEL" in call[0])
        self.assertEqual(suzlon_call[1], "Risk analysis available.")
        self.assertEqual(suzlon_call[2], "No catalyst analysis provided.")
        self.assertEqual(bel_call[1], "No risk analysis provided.")
        self.assertEqual(bel_call[2], "Catalyst analysis available.")


if __name__ == "__main__":
    unittest.main()
