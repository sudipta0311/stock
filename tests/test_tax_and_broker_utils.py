from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.broker_parser import parse_broker_csv
from utils.tax_calculator import calculate_pnl, should_exit


class BrokerParserTests(unittest.TestCase):
    def test_generic_csv_aliases_are_parsed(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as handle:
            handle.write("Trading Symbol,Qty,Avg Price,LTP\n")
            handle.write("infy,12,1450.5,1521.2\n")
            csv_path = Path(handle.name)
        try:
            holdings = parse_broker_csv(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["symbol"], "INFY")
        self.assertEqual(holdings[0]["quantity"], 12.0)
        self.assertEqual(holdings[0]["avg_buy_price"], 1450.5)


class TaxCalculatorTests(unittest.TestCase):
    def test_limited_upside_exit_recommendation(self) -> None:
        pnl = calculate_pnl(
            symbol="INFY",
            avg_buy_price=1000.0,
            current_price=1200.0,
            quantity=100.0,
            buy_date_str="2024-01-01",
        )
        decision = should_exit(pnl, analyst_target=1260.0, current_price=1200.0)

        self.assertEqual(decision["exit_recommendation"], "EXIT — limited upside")
        self.assertEqual(decision["urgency"], "MEDIUM")
        self.assertIn("Net proceeds", decision["tax_note"])


if __name__ == "__main__":
    unittest.main()
