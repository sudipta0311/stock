from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.agents.monitor_agents import MonitoringAgents
from stock_platform.config import AppConfig


class StubRepo:
    pass


class StubProvider:
    def get_price_context(self, symbol: str) -> dict[str, float]:
        return {"analyst_target": 210.0}


class StubLLM:
    def monitoring_rationale(self, action_row, thesis, drawdown):  # pragma: no cover - not used in this test
        return None


class MonitoringTaxLogicTests(unittest.TestCase):
    def test_wait_then_exit_when_ltcg_is_close_and_upside_is_limited(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(), lambda **kwargs: None, StubLLM())
        buy_date = (date.today() - timedelta(days=340)).isoformat()
        state = {
            "portfolio_context": {
                "monitor_universe": [{"symbol": "KWIL", "monitor_source": "direct", "total_weight": 8.0}],
                "direct_equity_buy_map": {
                    "KWIL": {
                        "symbol": "KWIL",
                        "quantity": 1000,
                        "avg_buy_price": 100.0,
                        "buy_date": buy_date,
                    }
                },
            },
            "stock_reviews": [{"symbol": "KWIL", "sentiment_score": 0.2}],
            "thesis_reviews": [{"symbol": "KWIL", "status": "INTACT"}],
            "drawdown_alerts": [{"symbol": "KWIL", "severity": "LOW", "current_price": 200.0}],
            "quant_scores": [{"symbol": "KWIL", "quant_score": 0.84}],
        }

        result = agent.decide_actions(state)
        row = result["actions"][0]

        self.assertTrue(row["action"].startswith("WAIT "))
        self.assertIn("then EXIT", row["action"])
        self.assertEqual(row["urgency"], "MEDIUM")
        self.assertIn("P&L: +Rs100,000", row["rationale"])
        self.assertIn("[STCG]", row["rationale"])

    def test_quality_stock_in_loss_switches_to_buy_more(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(), lambda **kwargs: None, StubLLM())
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
        self.assertIn("BUY MORE", row["rationale"])
        self.assertIn("offset other gains", row["rationale"])

    def test_medium_quality_stock_in_loss_switches_to_hold_review(self) -> None:
        agent = MonitoringAgents(StubRepo(), StubProvider(), AppConfig(), lambda **kwargs: None, StubLLM())
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
        self.assertIn("HOLD - review next quarter", row["rationale"])
        self.assertIn("tax harvesting", row["rationale"])


if __name__ == "__main__":
    unittest.main()
