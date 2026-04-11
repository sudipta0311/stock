from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.agents.quant_model import compute_quality_score
from stock_platform.config import AppConfig
from stock_platform.services.llm import PlatformLLM


class RecordingLLM(PlatformLLM):
    def __init__(self, config: AppConfig, provider: str) -> None:
        super().__init__(config, provider=provider)
        self.calls: list[dict[str, str]] = []

    def _call(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)
        return "ok"


class FakeAnthropicClient:
    last_request: dict[str, object] | None = None

    class _Messages:
        def create(self, **kwargs):
            FakeAnthropicClient.last_request = kwargs
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="synthesis")])

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.messages = self._Messages()


class BuyQualityScoreTests(unittest.TestCase):
    def test_missing_data_returns_unknown_not_perfect(self) -> None:
        self.assertEqual(compute_quality_score("TATAMOTORS", {}), 0.5)

    def test_expected_screener_rule_bands_for_tmcv_and_bel(self) -> None:
        tmcv_facts = {
            "symbol": "TMCV",
            "symbol_mapped": True,
            "roce_pct": -400.0,
            "eps": 11.84,
            "promoter_holding": 42.56,
            "debt_to_equity": 0.57,
        }
        bel_facts = {
            "symbol": "BEL",
            "roce_pct": 38.9,
            "eps": 8.14,
            "revenue_growth_pct": 13.0,
            "promoter_holding": 51.14,
            "debt_to_equity": 0.003,
        }

        self.assertLess(compute_quality_score("TATAMOTORS", tmcv_facts), 0.5)
        self.assertGreater(compute_quality_score("BEL", bel_facts), 0.6)

    def test_perfect_score_requires_all_five_rules(self) -> None:
        almost_perfect_live = {
            "roce_pct": 25,
            "eps": 30,
            "revenue_growth_pct": 20,
            "debt_to_equity": 0.10,
        }
        all_rules_live = almost_perfect_live | {"promoter_holding": 60}

        self.assertEqual(compute_quality_score("BEL", almost_perfect_live, {}), 1.0)
        self.assertEqual(compute_quality_score("IDEAL", all_rules_live, {}), 1.0)

    def test_negative_eps_caps_quality_score(self) -> None:
        stressed = {
            "roce_pct": 22,
            "eps": -5,
            "revenue_growth_pct": 12,
            "promoter_holding": 40,
            "debt_to_equity": 0.4,
        }
        self.assertEqual(compute_quality_score("LOSSMAKER", stressed), 0.35)


class BuyPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(anthropic_api_key="test-key", openai_api_key="test-key")
        self.item = {
            "company_name": "Bharat Electronics",
            "symbol": "BEL",
            "sector": "Defence",
            "entry_signal": "ACCUMULATE",
            "quality_score": 0.85,
            "overlap_pct": 0.0,
            "gap_reason": "Low existing defence exposure",
            "fund_attribution": [],
            "live_financials": {
                "roce_ttm": 0.266,
                "revenueGrowth": 0.24,
                "debtToEquity": 0.273,
                "trailingEps": 8.14,
                "targetMeanPrice": 488.64,
            },
            "price_context": {"analyst_target": 488.64},
        }
        self.portfolio_context = {"normalized_exposure": [{"symbol": "HDFCBANK"}]}

    def test_buy_rationale_prompts_are_different_for_each_provider(self) -> None:
        anthropic_llm = RecordingLLM(self.config, provider="anthropic")
        openai_llm = RecordingLLM(self.config, provider="openai")

        anthropic_llm.buy_rationale(self.item, self.portfolio_context)
        openai_llm.buy_rationale(self.item, self.portfolio_context)

        anthropic_prompt = anthropic_llm.calls[0]["system_prompt"]
        openai_prompt = openai_llm.calls[0]["system_prompt"]
        self.assertIn("single most important RISK", anthropic_prompt)
        self.assertIn("specific near-term catalyst", openai_prompt)
        self.assertIn("ROCE 26.6%", anthropic_llm.calls[0]["user_prompt"])
        self.assertIn("EXIT", openai_prompt)

    def test_synthesis_prompt_requests_combined_verdict(self) -> None:
        fake_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
        llm = PlatformLLM(self.config, provider="anthropic")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = llm.synthesise_comparison(
                stock_name="BEL",
                anthropic_rationale="RISK: margin compression.",
                openai_rationale="CATALYST: export order wins.",
            )

        self.assertEqual(result, "synthesis")
        request = FakeAnthropicClient.last_request or {}
        system_text = request["system"][0]["text"]  # type: ignore[index]
        user_text = request["messages"][0]["content"]  # type: ignore[index]
        self.assertIn("COMBINED VERDICT", system_text)
        self.assertIn("ACCUMULATE GRADUALLY", user_text)


if __name__ == "__main__":
    unittest.main()
