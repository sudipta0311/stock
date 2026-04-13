from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.agents.signal_agents import SignalAgents
from stock_platform.utils.signal_sources import get_tariff_signal


class StubRepo:
    def __init__(self) -> None:
        self.saved: dict[str, list[object]] = {}

    def replace_signals(self, family: str, rows: list[object]) -> None:
        self.saved[family] = rows

    def list_signals(self, family: str):
        return self.saved.get(family, [])


class StubProvider:
    today = date(2026, 4, 13)


class TariffSignalTests(unittest.TestCase):
    def test_get_tariff_signal_returns_negative_mapping(self) -> None:
        signal = get_tariff_signal("Consumer Durables")

        self.assertEqual(signal["impact"], "NEGATIVE")
        self.assertEqual(signal["source"], "US reciprocal tariff announcement")

    def test_aggregate_signals_applies_tariff_penalty(self) -> None:
        repo = StubRepo()
        agents = SignalAgents(repo, StubProvider())

        result = agents.aggregate_signals(
            {
                "geo_signals": [{"sector": "Consumer Durables", "score": 0.62}],
                "policy_signals": [{"sector": "Consumer Durables", "score": 0.62}],
                "flow_signals": [{"sector": "Consumer Durables", "score": 0.62}],
                "contrarian_signals": [{"sector": "Consumer Durables", "score": 0.62}],
            }
        )

        row = next(item for item in result["unified_signals"] if item["sector"] == "Consumer Durables")
        self.assertEqual(row["sector"], "Consumer Durables")
        self.assertEqual(row["score"], 0.47)
        self.assertEqual(row["conviction"], "NEUTRAL")
        self.assertEqual(row["payload"]["tariff_signal"]["impact"], "NEGATIVE")
        self.assertEqual(row["payload"]["tariff_penalty"], 0.15)
        self.assertIn("US 26% tariff", row["payload"]["tariff_warning"])


if __name__ == "__main__":
    unittest.main()
