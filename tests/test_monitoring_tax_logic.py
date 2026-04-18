from __future__ import annotations

import sys
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.agents.monitor_agents import MonitoringAgents, compute_monitoring_score, get_overlap_pct
from stock_platform.config import AppConfig

LOCAL_DB_CONFIG = {
    "turso_database_url": "",
    "turso_auth_token": "",
    "turso_sync_interval_seconds": 0,
}


class StubRepo:
    def __init__(self, context: dict | None = None) -> None:
        self.context = context or {}
        self.saved_rows = []

    def load_portfolio_context(self) -> dict:
        return self.context

    def save_monitoring_actions(self, run_id: str, rows: list) -> None:
        self.saved_rows = rows


class StubProvider:
    def __init__(self, analyst_target: float = 210.0) -> None:
        self.analyst_target = analyst_target

    def normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").upper()

    def get_price_context(self, symbol: str) -> dict[str, float]:
        return {"analyst_target": self.analyst_target}


class StubLLM:
    def monitoring_rationale(self, action_row, thesis, drawdown):  # pragma: no cover - not used in this test
        return None


class ShortRationaleLLM:
    def monitoring_rationale(self, action_row, thesis, drawdown):
        del thesis, drawdown
        return {
            "action": action_row["action"],
            "severity": action_row["severity"],
            "rationale": "Too short",
        }


class ErrorLLM:
    def monitoring_rationale(self, action_row, thesis, drawdown):
        del action_row, thesis, drawdown
        raise RuntimeError("monitoring LLM blew up")


class MonitoringTaxLogicTests(unittest.TestCase):
    def test_get_overlap_pct_uses_alias_variants(self) -> None:
        overlap = get_overlap_pct(
            "LT",
            {
                "LARSENTOUBRO": {"overlap_pct": 2.75},
            },
        )

        self.assertAlmostEqual(overlap, 2.75)

    def test_get_overlap_pct_uses_case_insensitive_alias_variants(self) -> None:
        overlap = get_overlap_pct(
            "HINDUNILVR",
            {
                "hul": {"overlap_pct": 1.8},
            },
        )

        self.assertAlmostEqual(overlap, 1.8)

    def test_get_overlap_pct_prefers_positive_alias_over_zero_exact_match(self) -> None:
        overlap = get_overlap_pct(
            "KWIL",
            {
                "KWIL": {"overlap_pct": 0.0},
                "KALYANJEWELS": {"overlap_pct": 3.13},
            },
        )

        self.assertAlmostEqual(overlap, 3.13)

    def test_compute_monitoring_score_returns_none_when_core_data_is_missing(self) -> None:
        score = compute_monitoring_score(
            "BEL",
            {
                "roce_ttm": None,
                "revenueGrowth": None,
                "debtToEquity": 0.9,
            },
        )

        self.assertIsNone(score)

    def test_compute_monitoring_score_uses_roe_for_banking_names(self) -> None:
        score = compute_monitoring_score(
            "HDFCBANK",
            {
                "returnOnEquity": 0.162,
                "revenueGrowth": 0.12,
                "debtToEquity": 8.5,
            },
            "Banks - Regional",
        )

        self.assertEqual(score, 0.75)

    def test_wait_then_exit_when_ltcg_is_close_and_upside_is_limited(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, StubLLM())
        buy_date = (date.today() - timedelta(days=340)).isoformat()
        state = {
            "portfolio_context": {
                "monitor_universe": [{"symbol": "KWIL", "monitor_source": "direct", "total_weight": 8.0}],
                "direct_equity_buy_map": {
                    "KWIL": {
                        "symbol": "KWIL",
                        "quantity": 1000,
                        "avg_buy_price": 200.0,
                        "buy_date": buy_date,
                    }
                },
            },
            "stock_reviews": [{"symbol": "KWIL", "sentiment_score": 0.2}],
            "thesis_reviews": [{"symbol": "KWIL", "status": "WEAKENED"}],
            "drawdown_alerts": [{"symbol": "KWIL", "severity": "LOW", "current_price": 260.0}],
            "quant_scores": [{"symbol": "KWIL", "quant_score": 0.84}],
        }

        with patch("stock_platform.agents.monitor_agents.fetch_analyst_consensus_target", return_value=270.0):
            result = agent.decide_actions(state)
        row = result["actions"][0]

        self.assertTrue(row["action"].startswith("WAIT "))
        self.assertIn("then EXIT", row["action"])
        self.assertEqual(row["urgency"], "MEDIUM")

    def test_quality_stock_in_loss_switches_to_buy_more(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, StubLLM())
        state = {
            "portfolio_context": {
                "monitor_universe": [{"symbol": "ASIANPAINT", "monitor_source": "direct", "total_weight": 8.0}],
                "direct_equity_buy_map": {
                    "ASIANPAINT": {
                        "symbol": "ASIANPAINT",
                        "quantity": 10,
                        "avg_buy_price": 3000.0,
                        "buy_date": "2025-01-01",
                    }
                },
            },
            "stock_reviews": [{"symbol": "ASIANPAINT", "sentiment_score": 0.1}],
            "thesis_reviews": [{"symbol": "ASIANPAINT", "status": "INTACT"}],
            "drawdown_alerts": [{"symbol": "ASIANPAINT", "severity": "LOW", "current_price": 2400.0}],
            "quant_scores": [{"symbol": "ASIANPAINT", "quant_score": 0.80}],
        }

        result = agent.decide_actions(state)
        row = result["actions"][0]

        self.assertEqual(row["action"], "BUY MORE")
        self.assertEqual(row["urgency"], "LOW")

    def test_medium_quality_stock_in_loss_switches_to_hold_review(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, StubLLM())
        state = {
            "portfolio_context": {
                "monitor_universe": [{"symbol": "SBICARD", "monitor_source": "direct", "total_weight": 8.0}],
                "direct_equity_buy_map": {
                    "SBICARD": {
                        "symbol": "SBICARD",
                        "quantity": 50,
                        "avg_buy_price": 740.0,
                        "buy_date": "2025-01-01",
                    }
                },
            },
            "stock_reviews": [{"symbol": "SBICARD", "sentiment_score": 0.1}],
            "thesis_reviews": [{"symbol": "SBICARD", "status": "INTACT"}],
            "drawdown_alerts": [{"symbol": "SBICARD", "severity": "LOW", "current_price": 677.5}],
            "quant_scores": [{"symbol": "SBICARD", "quant_score": 0.62}],
        }

        result = agent.decide_actions(state)
        row = result["actions"][0]

        self.assertEqual(row["action"], "HOLD - review next quarter")
        self.assertEqual(row["urgency"], "MEDIUM")

    def test_stale_provider_target_is_replaced_before_decision(self) -> None:
        provider = StubProvider(analyst_target=40.0)
        agent = MonitoringAgents(StubRepo(), provider, AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, StubLLM())
        state = {
            "portfolio_context": {
                "monitor_universe": [{"symbol": "JIOFIN", "monitor_source": "direct", "total_weight": 8.0}],
                "direct_equity_buy_map": {
                    "JIOFIN": {
                        "symbol": "JIOFIN",
                        "quantity": 100,
                        "avg_buy_price": 100.0,
                        "buy_date": "2025-01-01",
                    }
                },
            },
            "stock_reviews": [{"symbol": "JIOFIN", "sentiment_score": 0.1}],
            "thesis_reviews": [{"symbol": "JIOFIN", "status": "WEAKENED"}],
            "drawdown_alerts": [{"symbol": "JIOFIN", "severity": "LOW", "current_price": 120.0}],
            "quant_scores": [{"symbol": "JIOFIN", "quant_score": 0.43}],
        }

        with patch("stock_platform.agents.monitor_agents.fetch_analyst_consensus_target", return_value=138.0) as fresh:
            result = agent.decide_actions(state)

        row = result["actions"][0]
        fresh.assert_called_once_with("JIOFIN", 120.0)
        self.assertEqual(row["action"], "HOLD")
        self.assertEqual(row["urgency"], "LOW")

    def test_large_winner_with_intact_thesis_becomes_strong_winner_hold(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, StubLLM())
        state = {
            "portfolio_context": {
                "monitor_universe": [{"symbol": "ICICIBANK", "monitor_source": "direct", "total_weight": 8.0}],
                "direct_equity_buy_map": {
                    "ICICIBANK": {
                        "symbol": "ICICIBANK",
                        "quantity": 270,
                        "avg_buy_price": 491.3543,
                        "buy_date": "2024-01-01",
                    }
                },
            },
            "stock_reviews": [{"symbol": "ICICIBANK", "sentiment_score": 0.2}],
            "thesis_reviews": [{"symbol": "ICICIBANK", "status": "INTACT"}],
            "drawdown_alerts": [{"symbol": "ICICIBANK", "severity": "LOW", "current_price": 1321.9}],
            "quant_scores": [{"symbol": "ICICIBANK", "quant_score": 0.58}],
        }

        with patch("stock_platform.agents.monitor_agents.fetch_analyst_consensus_target", return_value=1400.0):
            result = agent.decide_actions(state)

        row = result["actions"][0]
        self.assertEqual(row["action"], "HOLD - strong winner")
        self.assertEqual(row["urgency"], "LOW")

    def test_overlap_suppresses_buy_more_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "platform.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE overlap_scores (
                    symbol TEXT PRIMARY KEY,
                    overlap_pct REAL NOT NULL,
                    band TEXT NOT NULL,
                    attribution_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO overlap_scores(symbol, overlap_pct, band, attribution_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("ASIANPAINT", 2.4, "FLAG", "[]", "2026-04-12T00:00:00Z"),
            )
            conn.commit()
            conn.close()

            config = AppConfig(db_path=db_path, **LOCAL_DB_CONFIG)
            agent = MonitoringAgents(StubRepo(), StubProvider(), config, lambda **kwargs: None, StubLLM())
            state = {
                "portfolio_context": {
                    "monitor_universe": [{"symbol": "ASIANPAINT", "monitor_source": "direct", "total_weight": 8.0}],
                    "direct_equity_buy_map": {
                        "ASIANPAINT": {
                            "symbol": "ASIANPAINT",
                            "quantity": 10,
                            "avg_buy_price": 3000.0,
                            "buy_date": "2025-01-01",
                        }
                    },
                },
                "stock_reviews": [{"symbol": "ASIANPAINT", "sentiment_score": 0.1}],
                "thesis_reviews": [{"symbol": "ASIANPAINT", "status": "INTACT"}],
                "drawdown_alerts": [{"symbol": "ASIANPAINT", "severity": "LOW", "current_price": 2400.0}],
                "quant_scores": [{"symbol": "ASIANPAINT", "quant_score": 0.80}],
            }

            result = agent.decide_actions(state)
            row = result["actions"][0]

            self.assertEqual(row["action"], "HOLD - already in MFs")
            self.assertEqual(row["urgency"], "LOW")
            self.assertAlmostEqual(row["overlap_pct"], 2.4)

    def test_data_unavailable_action_is_used_when_quant_score_is_missing(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, StubLLM())
        state = {
            "portfolio_context": {
                "monitor_universe": [{"symbol": "HDFCBANK", "monitor_source": "direct", "total_weight": 8.0, "overlap_pct": 3.13}],
                "direct_equity_buy_map": {},
            },
            "stock_reviews": [{"symbol": "HDFCBANK", "sentiment_score": 0.1}],
            "thesis_reviews": [{"symbol": "HDFCBANK", "status": "UNKNOWN"}],
            "drawdown_alerts": [{"symbol": "HDFCBANK", "severity": "LOW", "current_price": 1700.0}],
            "quant_scores": [{"symbol": "HDFCBANK", "quant_score": None}],
        }

        with patch("stock_platform.agents.monitor_agents.fetch_analyst_consensus_target", return_value=1900.0):
            result = agent.decide_actions(state)

        row = result["actions"][0]
        self.assertEqual(row["action"], "DATA_UNAVAILABLE")
        self.assertEqual(row["severity"], "UNKNOWN")
        self.assertAlmostEqual(row["overlap_pct"], 3.13)
        self.assertIn("Monitoring skipped - financial data not available", row["rationale"])

    def test_load_context_carries_overlap_and_buy_fields_into_monitor_universe(self) -> None:
        repo = StubRepo(
            {
                "normalized_exposure": [
                    {
                        "symbol": "HDFCBANK",
                        "company_name": "HDFC BANK LIMITED",
                        "source_mix": {"direct_equity": 1.875},
                    },
                    {
                        "symbol": "HDFCBANKLIMITED",
                        "company_name": "HDFCBANKLIMITED",
                        "source_mix": {"mutual_fund": 0.306},
                    },
                ],
                "raw_holdings": [
                    {
                        "holding_type": "direct_equity",
                        "instrument_name": "HDFC Bank",
                        "symbol": "HDFCBANK",
                        "market_value": 100000.0,
                    }
                ],
                "watchlist": [],
                "direct_equity_holdings": [
                    {
                        "symbol": "HDFCBANK",
                        "avg_buy_price": 1450.0,
                        "current_price": 1700.0,
                        "buy_date": "2025-01-01",
                    }
                ],
                "overlap_scores": [
                    {"symbol": "HDFCBANK", "overlap_pct": 0.0},
                    {"symbol": "HDFCBANKLIMITED", "overlap_pct": 3.13},
                ],
                "unified_signals": [],
                "user_preferences": {},
            }
        )
        agent = MonitoringAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, StubLLM())

        result = agent.load_context({})

        holding = result["portfolio_context"]["monitor_universe"][0]
        self.assertEqual(holding["symbol"], "HDFCBANK")
        self.assertAlmostEqual(holding["overlap_pct"], 3.13)
        self.assertEqual(holding["avg_buy_price"], 1450.0)
        self.assertEqual(holding["buy_date"], "2025-01-01")

    def test_replace_feedback_marks_empty_monitoring_llm_rationale(self) -> None:
        repo = StubRepo()
        agent = MonitoringAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, ShortRationaleLLM())
        state = {
            "actions": [
                {
                    "symbol": "HDFCBANK",
                    "action": "HOLD",
                    "severity": "LOW",
                    "urgency": "LOW",
                    "rationale": "Thesis intact (quant 0.72) | MF overlap 3.13%",
                    "overlap_pct": 3.13,
                    "pnl": None,
                    "exit_recommendation": None,
                    "analyst_target": 1900.0,
                }
            ],
            "thesis_reviews": [{"symbol": "HDFCBANK", "status": "INTACT", "geo_signal_change": "NEUTRAL"}],
            "drawdown_alerts": [{"symbol": "HDFCBANK", "severity": "LOW", "drawdown_pct": -4.0}],
            "behavioural_flags": [],
        }

        agent.replace_feedback(state)

        self.assertEqual(len(repo.saved_rows), 1)
        self.assertIn("[LLM analysis failed - data context may be empty]", repo.saved_rows[0].rationale)

    def test_replace_feedback_surfaces_monitoring_llm_errors(self) -> None:
        repo = StubRepo()
        agent = MonitoringAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), lambda **kwargs: None, ErrorLLM())
        state = {
            "actions": [
                {
                    "symbol": "HDFCBANK",
                    "action": "HOLD",
                    "severity": "LOW",
                    "urgency": "LOW",
                    "rationale": "Thesis intact (quant 0.72) | MF overlap 3.13%",
                    "overlap_pct": 3.13,
                    "pnl": None,
                    "exit_recommendation": None,
                    "analyst_target": 1900.0,
                }
            ],
            "thesis_reviews": [{"symbol": "HDFCBANK", "status": "INTACT", "geo_signal_change": "NEUTRAL"}],
            "drawdown_alerts": [{"symbol": "HDFCBANK", "severity": "LOW", "drawdown_pct": -4.0}],
            "behavioural_flags": [],
        }

        agent.replace_feedback(state)

        self.assertEqual(len(repo.saved_rows), 1)
        self.assertIn("[LLM error: RuntimeError]", repo.saved_rows[0].rationale)


if __name__ == "__main__":
    unittest.main()
