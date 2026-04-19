import unittest

from stock_platform.utils.direct_equity_merge import apply_saved_buy_prices


class DirectEquityMergeTests(unittest.TestCase):
    def test_applies_saved_buy_price_when_editor_row_is_blank(self) -> None:
        rows = [{"instrument_name": "Infosys", "symbol": "INFY", "quantity": 10, "avg_buy_price": None}]
        holdings = [{"symbol": "INFY", "quantity": 10, "avg_buy_price": 1450.75, "buy_date": "2026-04-01"}]

        merged = apply_saved_buy_prices(rows, holdings)

        self.assertEqual(merged[0]["avg_buy_price"], 1450.75)

    def test_matches_saved_holdings_after_symbol_normalization(self) -> None:
        rows = [{"instrument_name": "Tata Motors", "symbol": "TATAMOTORS", "quantity": 4, "avg_buy_price": ""}]
        holdings = [{"symbol": "TMCV", "quantity": 4, "avg_buy_price": 712.4, "buy_date": "2026-04-01"}]

        merged = apply_saved_buy_prices(rows, holdings)

        self.assertEqual(merged[0]["avg_buy_price"], 712.4)

    def test_keeps_existing_buy_price_in_editor_rows(self) -> None:
        rows = [{"instrument_name": "HDFC Bank", "symbol": "HDFCBANK", "quantity": 3, "avg_buy_price": 1500.0}]
        holdings = [{"symbol": "HDFCBANK", "quantity": 3, "avg_buy_price": 1450.0, "buy_date": "2026-04-01"}]

        merged = apply_saved_buy_prices(rows, holdings)

        self.assertEqual(merged[0]["avg_buy_price"], 1500.0)


if __name__ == "__main__":
    unittest.main()
