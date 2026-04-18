from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.agents.portfolio_agents import PortfolioAgents


class _FakeRepo:
    def __init__(self) -> None:
        self.normalized_rows: list[dict] = []
        self.overlap_rows: list[dict] = []

    def replace_normalized_exposure(self, rows: list[dict]) -> None:
        self.normalized_rows = rows

    def replace_overlap_scores(self, rows: list[dict]) -> None:
        self.overlap_rows = rows

    def list_normalized_exposure(self) -> list[dict]:
        return self.normalized_rows


class _FakeProvider:
    def __init__(self) -> None:
        self.fund_holdings = {
            "Fund A": {"ICICIBANKLIMITED": 0.10},
            "Fund B": {"ICICIBANKLIMITED": 0.20},
        }
        self.etf_holdings = {
            "ETF A": {"BEL": 0.25},
        }

    def get_fund_holdings(self, instrument_name: str, month: str | None = None) -> tuple[dict[str, float], str]:
        del month
        return self.fund_holdings[instrument_name], "test-fund-source"

    def get_etf_holdings(self, instrument_name: str, month: str | None = None) -> tuple[dict[str, float], str]:
        del month
        return self.etf_holdings[instrument_name], "test-etf-source"

    def get_stock_snapshot(self, symbol: str) -> dict[str, str]:
        return {"company_name": symbol, "sector": "Test Sector"}

    def normalize_symbol(self, symbol: str) -> str:
        return {"ICICIBANKLIMITED": "ICICIBANK"}.get(symbol, symbol)

    def build_proxy_holding(self, *args, **kwargs) -> dict:
        raise AssertionError("Proxy holdings should not be used in this test")


class PortfolioAgentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = _FakeRepo()
        self.provider = _FakeProvider()
        self.agent = PortfolioAgents(self.repo, self.provider)

    def test_mutual_fund_lookthrough_uses_total_portfolio_assets(self) -> None:
        payload = {
            "mutual_funds": [
                {"instrument_name": "Fund A", "market_value": 40},
                {"instrument_name": "Fund B", "market_value": 60},
            ],
            "etfs": [],
            "direct_equities": [
                {"instrument_name": "HDFC BANK LIMITED", "symbol": "HDFCBANK", "market_value": 900},
            ],
        }

        state = {"payload": payload}
        mutual_fund_state = self.agent.parse_mutual_funds(state)
        normalized_state = self.agent.normalize_exposure({**state, **mutual_fund_state, "etf_exposure": []})
        overlap_state = self.agent.compute_overlap(normalized_state)

        icici_row = next(row for row in overlap_state["overlap_scores"] if row["symbol"] == "ICICIBANK")
        self.assertAlmostEqual(icici_row["overlap_pct"], 1.6)

        attribution_weights = [item["lookthrough_weight"] for item in icici_row["attribution"]]
        self.assertEqual(attribution_weights, [0.4, 1.2])

    def test_etf_lookthrough_uses_total_portfolio_assets(self) -> None:
        payload = {
            "mutual_funds": [],
            "etfs": [
                {"instrument_name": "ETF A", "market_value": 100},
            ],
            "direct_equities": [
                {"instrument_name": "HDFC BANK LIMITED", "symbol": "HDFCBANK", "market_value": 900},
            ],
        }

        state = {"payload": payload}
        etf_state = self.agent.decompose_etfs(state)

        self.assertEqual(len(etf_state["etf_exposure"]), 1)
        self.assertAlmostEqual(etf_state["etf_exposure"][0]["lookthrough_weight"], 2.5)


if __name__ == "__main__":
    unittest.main()
