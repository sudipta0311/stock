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
from stock_platform.agents.buy_agents import compute_net_return, get_fresh_analyst_target, get_top_n_with_replacement
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
            # Fake response must contain "COMBINED VERDICT" — the guard in
            # synthesise_comparison() appends a fallback when it's absent.
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(
                    text="• WHERE THEY AGREE: both see defence tailwinds.\n"
                         "• WHERE THEY DISAGREE: margin timing differs.\n"
                         "• COMBINED VERDICT: ACCUMULATE GRADUALLY."
                )],
                stop_reason="end_turn",
            )

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.messages = self._Messages()


class FakeOpenAIAuthenticationError(Exception):
    pass


class FakeOpenAIRateLimitError(Exception):
    pass


class FakeOpenAITimeoutError(Exception):
    pass


class FakeOpenAIChat:
    last_request: dict[str, object] | None = None

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc

    def create(self, **kwargs):
        FakeOpenAIChat.last_request = kwargs
        if self.exc is not None:
            raise self.exc
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="OK"))]
        )


class FakeOpenAIClient:
    def __init__(self, exc: Exception | None = None) -> None:
        self.chat = types.SimpleNamespace(completions=FakeOpenAIChat(exc))


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

    def test_compute_net_return_never_returns_none(self) -> None:
        self.assertEqual(compute_net_return(0.0, None), 0.0)
        self.assertEqual(compute_net_return(100.0, None), 13.12)

    def test_top_n_replacement_backfills_after_skips(self) -> None:
        candidates = [
            {
                "symbol": "AAA",
                "quality_score": 0.95,
                "financials": {"roce_pct": 20.0, "eps": 10.0, "pe_ratio": 15.0},
                "current_price": 120.0,
            },
            {
                "symbol": "BBB",
                "quality_score": 0.90,
                "financials": {"roce_pct": 20.0, "eps": 10.0, "pe_ratio": 15.0},
                "current_price": None,
            },
            {
                "symbol": "CCC",
                "quality_score": 0.85,
                "financials": {"roce_pct": 18.0, "eps": 8.0, "pe_ratio": 14.0},
                "current_price": 110.0,
            },
            {
                "symbol": "DDD",
                "quality_score": 0.80,
                "financials": {"roce_pct": 16.0, "eps": 7.0, "pe_ratio": 13.0},
                "current_price": 105.0,
            },
        ]

        picks = get_top_n_with_replacement(candidates, 2, ["AAA"], "unused.db")

        self.assertEqual([row["symbol"] for row in picks], ["CCC", "DDD"])

    @patch("utils.screener_fetcher.fetch_screener_data")
    def test_fresh_target_falls_back_to_proxy_or_price_buffer(self, fetch_mock) -> None:
        fetch_mock.return_value = {"eps": 20.0, "sector_pe": 25.0}
        with patch.dict(sys.modules, {"yfinance": types.SimpleNamespace(Ticker=lambda *_: types.SimpleNamespace(info={}))}):
            target = get_fresh_analyst_target("BEL", 300.0)
        self.assertGreater(target, 300.0)


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
        openai_llm = PlatformLLM(self.config, provider="openai")
        openai_llm._client = FakeOpenAIClient()
        fake_openai = types.SimpleNamespace(
            AuthenticationError=FakeOpenAIAuthenticationError,
            RateLimitError=FakeOpenAIRateLimitError,
            APITimeoutError=FakeOpenAITimeoutError,
        )

        anthropic_llm.buy_rationale(self.item, self.portfolio_context)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            openai_llm.buy_rationale(self.item, self.portfolio_context)

        anthropic_prompt = anthropic_llm.calls[0]["system_prompt"]
        request = FakeOpenAIChat.last_request or {}
        openai_prompt = request["messages"][0]["content"]  # type: ignore[index]
        self.assertIn("single most important RISK", anthropic_prompt)
        self.assertIn("momentum and catalyst analyst", openai_prompt)
        self.assertIn("ROCE 26.6%", anthropic_llm.calls[0]["user_prompt"])
        self.assertIn("EXIT", str(request["messages"][0]["content"]))  # type: ignore[index]

    def test_synthesis_prompt_requests_combined_verdict(self) -> None:
        fake_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
        llm = PlatformLLM(self.config, provider="anthropic")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            result = llm.synthesise_comparison(
                stock_name="BEL",
                anthropic_rationale="RISK: margin compression.",
                openai_rationale="CATALYST: export order wins.",
            )

        self.assertIn("COMBINED VERDICT", result or "")
        request = FakeAnthropicClient.last_request or {}
        system_text = request["system"][0]["text"]  # type: ignore[index]
        user_text = request["messages"][0]["content"]  # type: ignore[index]
        self.assertIn("COMBINED VERDICT", system_text)
        self.assertIn("ACCUMULATE GRADUALLY", user_text)

    def test_openai_buy_rationale_surfaces_auth_error(self) -> None:
        fake_openai = types.SimpleNamespace(
            AuthenticationError=FakeOpenAIAuthenticationError,
            RateLimitError=FakeOpenAIRateLimitError,
            APITimeoutError=FakeOpenAITimeoutError,
        )
        llm = PlatformLLM(self.config, provider="openai")
        llm._client = FakeOpenAIClient(FakeOpenAIAuthenticationError("bad key"))

        with patch.dict(sys.modules, {"openai": fake_openai}):
            result = llm.buy_rationale(self.item, self.portfolio_context)

        self.assertEqual(result, "[OpenAI authentication failed - check API key]")

    def test_openai_connection_reports_rate_limit(self) -> None:
        fake_openai = types.SimpleNamespace(
            AuthenticationError=FakeOpenAIAuthenticationError,
            RateLimitError=FakeOpenAIRateLimitError,
            APITimeoutError=FakeOpenAITimeoutError,
        )
        llm = PlatformLLM(self.config, provider="openai")
        llm._client = FakeOpenAIClient(FakeOpenAIRateLimitError("slow down"))

        with patch.dict(sys.modules, {"openai": fake_openai}):
            ok, message = llm.test_openai_connection()

        self.assertFalse(ok)
        self.assertIn("Rate limited", message)


if __name__ == "__main__":
    unittest.main()
