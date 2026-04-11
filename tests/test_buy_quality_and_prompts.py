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

from stock_platform.agents.buy_agents import compute_quality_score
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
    def test_missing_live_data_returns_zero_not_perfect(self) -> None:
        demo_facts = {
            "roce_5y": 18.9,
            "fcf_positive_years": 4,
            "revenue_consistency": 7.7,
            "de_ratio": 0.52,
        }
        self.assertEqual(compute_quality_score("TATAMOTORS", {}, demo_facts), 0.0)

    def test_expected_live_rule_bands_for_bel_and_divislab(self) -> None:
        bel_live = {
            "roce_ttm": 0.266,
            "freeCashflow": -4_246_600_000,
            "fcf_positive_years": 3,
            "revenueGrowth": 0.24,
            "debtToEquity": 0.003,
        }
        divislab_live = {
            "roce_ttm": 0.204,
            "freeCashflow": 2_150_000_000,
            "fcf_positive_years": 4,
            "revenueGrowth": 0.123,
            "debtToEquity": 0.0003,
        }

        self.assertAlmostEqual(compute_quality_score("BEL", bel_live, {}), 0.725, places=3)
        self.assertAlmostEqual(compute_quality_score("DIVISLAB", divislab_live, {}), 0.79, places=3)

    def test_perfect_score_requires_all_five_rules(self) -> None:
        almost_perfect_live = {
            "roce_ttm": 0.25,
            "freeCashflow": 1,
            "revenueGrowth": 0.20,
            "debtToEquity": 0.10,
        }
        all_rules_live = almost_perfect_live | {"promoter_holding_pct": 0.60}

        self.assertEqual(compute_quality_score("BEL", almost_perfect_live, {}), 0.85)
        self.assertEqual(compute_quality_score("IDEAL", all_rules_live, {}), 1.0)


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
