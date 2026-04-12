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

    def test_icici_style_shifted_csv_is_realigned(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as handle:
            handle.write(
                "Sr. No.,Stock Name,Company Name,CMP,Change,% Change,NAV,NAV Chg,% Chg.,"
                "Portfolio Holdings,Invested Value,Average Cost Value,Unrealized Profit/Loss,"
                "% Chg,Current Value,Days Profit/Loss,% Chg.,Realized Profit/Loss,% Change,"
                "Qty,Long Term Qty,Short Term Qty,XIRR\n"
            )
            handle.write(
                "1,KWILEQ,KWALITY WALL'S (INDIA) LIMITED,24.85,0.91,3.80,-,-,-,Equity,"
                "3092.02,47.5695,-1476.77,-47.760687188310555,-,-,-,0.0,-,65.0,0.0,65.0,-,-\n"
            )
            csv_path = Path(handle.name)
        try:
            holdings = parse_broker_csv(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["symbol"], "KWIL")
        self.assertEqual(holdings[0]["quantity"], 65.0)
        self.assertAlmostEqual(holdings[0]["avg_buy_price"], 47.5695)
        self.assertAlmostEqual(holdings[0]["current_price"], 24.85)

    def test_broker_aliases_are_mapped_to_exchange_symbols(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as handle:
            handle.write("Stock Name,Company Name,CMP,Average Cost Value,Qty\n")
            handle.write("HDFBANEQ,HDFC BANK LTD,810.30,712.2691,220\n")
            handle.write("ICIBANEQ,ICICI BANK LTD.,1321.90,491.3543,270\n")
            handle.write("LARTOUEQ,LARSEN & TOUBRO LTD,3959.90,1030.0501,126\n")
            csv_path = Path(handle.name)
        try:
            holdings = parse_broker_csv(csv_path)
        finally:
            csv_path.unlink(missing_ok=True)

        self.assertEqual([row["symbol"] for row in holdings], ["HDFCBANK", "ICICIBANK", "LT"])


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
