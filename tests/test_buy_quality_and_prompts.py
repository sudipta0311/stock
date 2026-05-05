from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.agents.quant_model import compute_quality_score
from stock_platform.agents.buy_agents import (
    BuyAgents,
    FINAL_BUFFER_MULTIPLIER,
    MINIMUM_RR_RATIO,
    SHORTLIST_BUFFER_MULTIPLIER,
    buffered_top_n,
    compute_net_return,
    filter_by_risk_reward,
    get_top_n_with_replacement,
)
from stock_platform.config import AppConfig
from stock_platform.services.engine import _append_entry_summary
from stock_platform.services.llm import PlatformLLM
from stock_platform.utils.entry_calculator import (
    apply_momentum_override,
    calculate_entry_levels,
    fetch_analyst_consensus_target,
)
from stock_platform.utils.signal_sources import get_tariff_signal
from stock_platform.utils.screener_fetcher import (
    compute_pat_momentum,
    compute_revenue_momentum,
    fetch_recent_results,
    find_yoy_column,
)
from stock_platform.utils.stock_validator import check_recently_listed

LOCAL_DB_CONFIG = {
    "turso_database_url": "",
    "turso_auth_token": "",
    "turso_sync_interval_seconds": 0,
}


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


class _FakeAtAccessor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        row, column = key
        return self.data[row][column]


class FakeQuarterlyIncomeStatement:
    def __init__(self, columns, rows) -> None:
        self.columns = columns
        self.index = list(rows.keys())
        self.empty = False
        self.at = _FakeAtAccessor(rows)


class StubRepo:
    def __init__(self) -> None:
        self.saved_recommendations: list[object] = []

    def list_signals(self, family: str):
        if family == "unified":
            return [{"sector": "Defence", "conviction": "BUY", "score": 0.7}]
        return []

    def save_recommendations(self, run_id: str, rows: list[object]) -> None:
        self.saved_recommendations = rows

    def persist_recommendation_history(self, **kwargs) -> None:  # noqa: ARG002
        pass


class StubProvider:
    def __init__(self) -> None:
        self.today = __import__("datetime").date(2026, 4, 13)

    def get_stock_news(self, symbol: str):
        return {"headline": f"{symbol} update", "sentiment_score": 0.15}

    def get_price_context(self, symbol: str):
        return {"symbol": symbol, "price": 100.0, "analyst_target": 140.0}


class RejectingLLM:
    def qualitative_analysis(self, candidate, news, signal_context):
        return {"approved": False, "confidence": 0.2, "reasoning": "Too cautious"}


class StaticLLM:
    def buy_rationale(self, item, portfolio_context):
        return f"{item['symbol']} still clears the final gate."


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
    def test_consensus_target_falls_back_to_known_targets(self, fetch_mock) -> None:
        fetch_mock.return_value = {}
        with patch.dict(sys.modules, {"yfinance": types.SimpleNamespace(Ticker=lambda *_: types.SimpleNamespace(info={}))}):
            target = fetch_analyst_consensus_target("BEL", 300.0)
        self.assertGreater(target, 300.0)

    def test_qualitative_fallback_prevents_empty_shortlist(self) -> None:
        agent = BuyAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), RejectingLLM())
        state = {
            "request": {"top_n": 2},
            "shortlist": [
                {"symbol": "BEL", "sector": "Defence", "quality_score": 0.85, "selection_score": 0.90},
                {"symbol": "HAL", "sector": "Defence", "quality_score": 0.82, "selection_score": 0.88},
            ],
        }

        result = agent.validate_qualitative(state)

        self.assertEqual(len(result["shortlist"]), 2)
        self.assertEqual([row["symbol"] for row in result["shortlist"]], ["BEL", "HAL"])

    def test_buffered_top_n_keeps_extra_candidates_for_late_filters(self) -> None:
        self.assertEqual(buffered_top_n(3), 3 * FINAL_BUFFER_MULTIPLIER)
        self.assertEqual(buffered_top_n(1), 1 * FINAL_BUFFER_MULTIPLIER)

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=120.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_finalize_recommendation_backfills_after_do_not_enter(self, _gov_mock, _target_mock) -> None:
        repo = StubRepo()
        agent = BuyAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())
        base_item = {
            "company_name": "Demo Co",
            "sector": "Defence",
            "quality_score": 0.8,
            "gap_reason": "Gap fill",
            "overlap_pct": 0.0,
            "fund_attribution": [],
            "initial_tranche_pct": 8.0,
            "target_pct": 20.0,
            "initial_amount_inr": 8000.0,
            "target_amount_inr": 20000.0,
            "allocation_pct": 8.0,
            "allocation_amount": 8000.0,
            "tranches": 3,
            "differentiation_score": 0.8,
            "news": {"headline": "Positive update"},
            "price_context": {"price": 100.0, "analyst_target": 120.0},
            "live_financials": {"currentPrice": 100.0},
        }
        state = {
            "request": {"top_n": 3},
            "confidence": {"band": "GREEN"},
            "portfolio_context": {"normalized_exposure": []},
            "allocations": [
                base_item | {"symbol": "AAA", "entry_signal": "DO NOT ENTER"},
                base_item | {"symbol": "BBB", "entry_signal": "ACCUMULATE"},
                base_item | {"symbol": "CCC", "entry_signal": "ACCUMULATE"},
                base_item | {"symbol": "DDD", "entry_signal": "ACCUMULATE"},
            ],
        }

        result = agent.finalize_recommendation(state)

        self.assertEqual([row["symbol"] for row in result["recommendations"]], ["BBB", "CCC", "DDD"])
        self.assertEqual(len(repo.saved_recommendations), 3)
        first_payload = repo.saved_recommendations[0].payload
        self.assertEqual(first_payload["current_price"], 100.0)
        self.assertEqual(first_payload["analyst_target"], 120.0)
        self.assertEqual(first_payload["fin_data"]["currentPrice"], 100.0)
        self.assertGreaterEqual(first_payload["entry_levels"]["risk_reward"], MINIMUM_RR_RATIO)

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=110.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_finalize_recommendation_excludes_low_rr_candidates(self, _gov_mock, _target_mock) -> None:
        repo = StubRepo()
        agent = BuyAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())
        base_item = {
            "company_name": "Demo Co",
            "sector": "Defence",
            "quality_score": 0.8,
            "gap_reason": "Gap fill",
            "overlap_pct": 0.0,
            "fund_attribution": [],
            "initial_tranche_pct": 8.0,
            "target_pct": 20.0,
            "initial_amount_inr": 8000.0,
            "target_amount_inr": 20000.0,
            "allocation_pct": 8.0,
            "allocation_amount": 8000.0,
            "tranches": 3,
            "differentiation_score": 0.8,
            "news": {"headline": "Positive update"},
            "price_context": {"price": 100.0, "analyst_target": 110.0},
            "live_financials": {"currentPrice": 100.0},
        }
        state = {
            "request": {"top_n": 2},
            "confidence": {"band": "GREEN"},
            "portfolio_context": {"normalized_exposure": []},
            "allocations": [
                base_item | {"symbol": "LOWRR1", "entry_signal": "BUY"},
                base_item | {"symbol": "LOWRR2", "entry_signal": "ACCUMULATE"},
            ],
        }

        result = agent.finalize_recommendation(state)

        self.assertEqual(result["recommendations"], [])
        self.assertEqual(len(repo.saved_recommendations), 0)
        self.assertEqual(
            [row["status"] for row in result["skipped_stocks"]],
            ["LOW_RISK_REWARD", "LOW_RISK_REWARD"],
        )
        self.assertIn("R/R", result["skipped_stocks"][0]["reason"])
        self.assertIn("minimum risk/reward gate", result["run_summary"]["blocked_reason"])

    def test_filter_by_risk_reward_returns_only_valid_candidates(self) -> None:
        valid, excluded = filter_by_risk_reward(
            [
                {"symbol": "LOW", "entry_levels": {"risk_reward": 1.2}},
                {"symbol": "EDGE", "entry_levels": {"risk_reward": 1.5}},
                {"symbol": "HIGH", "entry_levels": {"risk_reward": 2.1}},
            ]
        )

        self.assertEqual([row["symbol"] for row in valid], ["EDGE", "HIGH"])
        self.assertEqual(excluded, ["LOW"])

    def test_shortlist_uses_larger_buffer_after_rr_gate(self) -> None:
        repo = StubRepo()
        agent = BuyAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())
        captured: dict[str, int] = {}

        def fake_replacement(scored_candidates, n, skipped_symbols, db_path):
            del scored_candidates, skipped_symbols, db_path
            captured["n"] = n
            return []

        with patch("stock_platform.agents.buy_agents.get_top_n_with_replacement", side_effect=fake_replacement):
            agent.shortlist(
                {
                    "request": {"top_n": 4},
                    "risk_filtered_candidates": [],
                    "skipped_candidates": [],
                }
            )

        self.assertEqual(captured["n"], 4 * SHORTLIST_BUFFER_MULTIPLIER)

    def test_recently_listed_check_flags_known_ipo(self) -> None:
        result = check_recently_listed("TMCV", current_date=__import__("datetime").date(2026, 4, 13))

        self.assertTrue(result["recently_listed"])
        self.assertEqual(result["recommendation"], "WAIT - lock-in risk")
        self.assertIn("Lock-in expiry overhang risk", result["warning"])

    def test_assess_timing_caps_recent_ipo_to_wait(self) -> None:
        agent = BuyAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())

        result = agent.assess_timing(
            {
                "request": {},
                "differentiated_shortlist": [{"symbol": "TMCV", "sector": "Defence", "quality_score": 0.8}],
            }
        )

        row = result["timing_assessments"][0]
        self.assertEqual(row["original_entry_signal"], "ACCUMULATE")
        self.assertEqual(row["entry_signal"], "WAIT")
        self.assertTrue(row["lock_in_check"]["recently_listed"])
        self.assertEqual(row["lock_in_multiplier"], 0.5)

    def test_apply_momentum_override_upgrades_wait_near_low(self) -> None:
        upgraded = apply_momentum_override(
            signal="WAIT",
            recent_results={"revenue_yoy_growth_pct": 36.0, "momentum": "STRONG"},
            current_price=100.0,
            week52_low=80.0,
        )

        self.assertEqual(upgraded, "ACCUMULATE")

    def test_assess_timing_upgrades_wait_when_recent_results_are_strong(self) -> None:
        class NeutralRepo(StubRepo):
            def list_signals(self, family: str):
                return []

        agent = BuyAgents(NeutralRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())

        result = agent.assess_timing(
            {
                "request": {},
                "differentiated_shortlist": [
                    {
                        "symbol": "BEL",
                        "sector": "Defence",
                        "quality_score": 0.8,
                        "financials": {
                            "recent_results": {"revenue_yoy_growth_pct": 42.0, "momentum": "STRONG"},
                            "week52_low": 80.0,
                            "last_result_date": "2026-04-01",
                            "result_days_stale": 30,
                        },
                    }
                ],
            }
        )

        row = result["timing_assessments"][0]
        self.assertEqual(row["original_entry_signal"], "WAIT")
        self.assertEqual(row["entry_signal"], "ACCUMULATE")
        self.assertTrue(row["momentum_override_applied"])

    @patch("stock_platform.utils.screener_fetcher._fetch_quarterly_income_stmt", return_value=None)
    @patch("stock_platform.utils.screener_fetcher.requests.get")
    def test_fetch_recent_results_parses_quarterly_sales(self, get_mock, _stmt_mock) -> None:
        get_mock.return_value = types.SimpleNamespace(
            status_code=200,
            text="""
            <div id="quarters">
              <table>
                <tr>
                  <th>Metric</th>
                  <th>Dec 2025</th>
                  <th>Sep 2025</th>
                  <th>Jun 2025</th>
                  <th>Mar 2025</th>
                  <th>Dec 2024</th>
                </tr>
                <tr><td>Sales +</td><td>126</td><td>110</td><td>104</td><td>101</td><td>100</td></tr>
              </table>
            </div>
            """,
        )

        result = fetch_recent_results("BEL")

        self.assertEqual(result["latest_quarter_revenue"], 126.0)
        self.assertEqual(result["prev_quarter_revenue"], 110.0)
        self.assertEqual(result["same_quarter_last_year_revenue"], 100.0)
        self.assertEqual(result["revenue_yoy_growth_pct"], 26.0)
        self.assertEqual(result["comparison_label"], "Q3 FY26 vs Q3 FY25")
        self.assertEqual(result["momentum"], "GOOD")

    def test_compute_revenue_momentum_uses_most_recent_quarter_vs_same_quarter_last_year(self) -> None:
        stmt = FakeQuarterlyIncomeStatement(
            columns=[
                datetime(2025, 12, 31),
                datetime(2025, 9, 30),
                datetime(2025, 6, 30),
                datetime(2025, 3, 31),
                datetime(2024, 12, 31),
            ],
            rows={
                "Total Revenue": {
                    datetime(2025, 12, 31): 148.0,
                    datetime(2025, 9, 30): 118.0,
                    datetime(2025, 6, 30): 112.0,
                    datetime(2025, 3, 31): 107.0,
                    datetime(2024, 12, 31): 100.0,
                }
            },
        )

        result = compute_revenue_momentum("SUZLON", {}, quarterly_income_stmt=stmt)

        self.assertEqual(result["momentum"], "STRONG")
        self.assertEqual(result["growth_pct"], 48.0)
        self.assertEqual(result["period"], "Q3 FY26 vs Q3 FY25")

    def test_find_yoy_column_uses_date_proximity_not_fixed_offset(self) -> None:
        columns = [
            datetime(2025, 12, 31),
            datetime(2025, 9, 30),
            datetime(2025, 6, 30),
            datetime(2025, 3, 31),
            datetime(2025, 6, 30),  # wrong fixed-offset candidate if position alone were used
            datetime(2024, 12, 31),
        ]

        result = find_yoy_column(columns, 0)

        self.assertEqual(result, 5)

    def test_compute_revenue_momentum_finds_correct_yoy_quarter_when_offset_four_is_wrong(self) -> None:
        stmt = FakeQuarterlyIncomeStatement(
            columns=[
                datetime(2025, 12, 31),
                datetime(2025, 9, 30),
                datetime(2025, 6, 30),
                datetime(2025, 3, 31),
                datetime(2024, 6, 30),
                datetime(2024, 12, 31),
            ],
            rows={
                "Total Revenue": {
                    datetime(2025, 12, 31): 148.0,
                    datetime(2025, 9, 30): 118.0,
                    datetime(2025, 6, 30): 112.0,
                    datetime(2025, 3, 31): 107.0,
                    datetime(2024, 6, 30): 95.0,
                    datetime(2024, 12, 31): 100.0,
                }
            },
        )

        result = compute_revenue_momentum("SUZLON", {}, quarterly_income_stmt=stmt)

        self.assertEqual(result["same_quarter_last_year_revenue"], 100.0)
        self.assertEqual(result["growth_pct"], 48.0)
        self.assertEqual(result["period"], "Q3 FY26 vs Q3 FY25")

    def test_compute_pat_momentum_detects_revenue_pat_divergence(self) -> None:
        stmt = FakeQuarterlyIncomeStatement(
            columns=[
                datetime(2025, 12, 31),
                datetime(2025, 9, 30),
                datetime(2025, 6, 30),
                datetime(2025, 3, 31),
                datetime(2024, 12, 31),
            ],
            rows={
                "Net Income": {
                    datetime(2025, 12, 31): 55.0,
                    datetime(2025, 9, 30): 80.0,
                    datetime(2025, 6, 30): 76.0,
                    datetime(2025, 3, 31): 73.0,
                    datetime(2024, 12, 31): 100.0,
                }
            },
        )

        result = compute_pat_momentum(
            "HINDALCO",
            {"revenue_growth_latest_qtr": 13.0},
            quarterly_income_stmt=stmt,
        )

        self.assertEqual(result["pat_momentum"], "COLLAPSING")
        self.assertEqual(result["pat_growth_pct"], -45.0)
        self.assertTrue(result["rev_pat_divergence"])

    def test_compute_pat_momentum_finds_correct_yoy_quarter_when_offset_four_is_wrong(self) -> None:
        stmt = FakeQuarterlyIncomeStatement(
            columns=[
                datetime(2025, 12, 31),
                datetime(2025, 9, 30),
                datetime(2025, 6, 30),
                datetime(2025, 3, 31),
                datetime(2024, 6, 30),
                datetime(2024, 12, 31),
            ],
            rows={
                "Net Income": {
                    datetime(2025, 12, 31): 145.0,
                    datetime(2025, 9, 30): 112.0,
                    datetime(2025, 6, 30): 108.0,
                    datetime(2025, 3, 31): 104.0,
                    datetime(2024, 6, 30): 92.0,
                    datetime(2024, 12, 31): 100.0,
                }
            },
        )

        result = compute_pat_momentum(
            "IRCTC",
            {"revenue_growth_latest_qtr": 16.0},
            quarterly_income_stmt=stmt,
        )

        self.assertEqual(result["pat_momentum"], "STRONG")
        self.assertEqual(result["pat_growth_pct"], 45.0)
        self.assertEqual(result["period"], "Q3 FY26 vs Q3 FY25")
        self.assertEqual(result["qualifier"], "")
        self.assertEqual(result["pat_abs_cr"], 0.0)

    def test_compute_pat_momentum_flags_high_growth_on_small_pat_base(self) -> None:
        stmt = FakeQuarterlyIncomeStatement(
            columns=[
                datetime(2025, 12, 31),
                datetime(2025, 9, 30),
                datetime(2025, 6, 30),
                datetime(2025, 3, 31),
                datetime(2024, 12, 31),
            ],
            rows={
                "Net Income": {
                    datetime(2025, 12, 31): 710000000.0,
                    datetime(2025, 9, 30): 640000000.0,
                    datetime(2025, 6, 30): 600000000.0,
                    datetime(2025, 3, 31): 580000000.0,
                    datetime(2024, 12, 31): 429800000.0,
                }
            },
        )

        result = compute_pat_momentum(
            "JUBLFOOD",
            {"revenue_growth_latest_qtr": 12.0},
            quarterly_income_stmt=stmt,
        )

        self.assertEqual(result["pat_momentum"], "STRONG")
        self.assertEqual(result["pat_growth_pct"], 65.2)
        self.assertEqual(result["period"], "Q3 FY26 vs Q3 FY25")
        self.assertEqual(
            result["qualifier"],
            "(Rs.71Cr absolute - high growth on small base)",
        )
        self.assertEqual(result["pat_abs_cr"], 71.0)

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_finalize_recommendation_preserves_lock_in_warning(self, _gov_mock, _target_mock) -> None:
        repo = StubRepo()
        agent = BuyAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())
        lock_in_check = check_recently_listed("TMCV", current_date=__import__("datetime").date(2026, 4, 13))
        state = {
            "request": {"top_n": 1},
            "confidence": {"band": "GREEN"},
            "portfolio_context": {"normalized_exposure": []},
            "allocations": [
                {
                    "symbol": "TMCV",
                    "company_name": "Demo Co",
                    "sector": "Defence",
                    "quality_score": 0.8,
                    "gap_reason": "Gap fill",
                    "overlap_pct": 0.0,
                    "fund_attribution": [],
                    "initial_tranche_pct": 0.0,
                    "target_pct": 5.5,
                    "initial_amount_inr": 0.0,
                    "target_amount_inr": 5500.0,
                    "allocation_pct": 0.0,
                    "allocation_amount": 0.0,
                    "tranches": 2,
                    "differentiation_score": 0.8,
                    "news": {"headline": "Positive update"},
                    "price_context": {"price": 100.0, "analyst_target": 140.0},
                    "live_financials": {"currentPrice": 100.0},
                    "financials": {"currentPrice": 100.0, "week52_low": 70.0},
                    "entry_signal": "WAIT",
                    "original_entry_signal": "ACCUMULATE",
                    "lock_in_check": lock_in_check,
                    "lock_in_multiplier": 0.5,
                }
            ],
        }

        result = agent.finalize_recommendation(state)

        self.assertEqual(len(result["recommendations"]), 1)
        payload = result["recommendations"][0]["payload"]
        self.assertTrue(payload["recently_listed"])
        self.assertEqual(payload["lock_in_multiplier"], 0.5)
        self.assertEqual(payload["entry_signal"], "WAIT")
        self.assertEqual(payload["original_entry_signal"], "ACCUMULATE")
        self.assertIn("Lock-in expiry overhang risk", payload["lock_in_warning"])

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_finalize_recommendation_preserves_recent_results_payload(self, _gov_mock, _target_mock) -> None:
        repo = StubRepo()
        agent = BuyAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())
        state = {
            "request": {"top_n": 1},
            "confidence": {"band": "GREEN"},
            "portfolio_context": {"normalized_exposure": []},
            "allocations": [
                {
                    "symbol": "BEL",
                    "company_name": "Demo Co",
                    "sector": "Defence",
                    "quality_score": 0.8,
                    "gap_reason": "Gap fill",
                    "overlap_pct": 0.0,
                    "fund_attribution": [],
                    "initial_tranche_pct": 8.0,
                    "target_pct": 20.0,
                    "initial_amount_inr": 8000.0,
                    "target_amount_inr": 20000.0,
                    "allocation_pct": 8.0,
                    "allocation_amount": 8000.0,
                    "tranches": 3,
                    "differentiation_score": 0.8,
                    "news": {"headline": "Positive update"},
                    "price_context": {"price": 100.0, "analyst_target": 140.0},
                    "live_financials": {"currentPrice": 100.0},
                    "financials": {"currentPrice": 100.0, "week52_low": 80.0},
                    "entry_signal": "ACCUMULATE",
                    "original_entry_signal": "WAIT",
                    "recent_results": {"revenue_yoy_growth_pct": 42.0, "momentum": "STRONG"},
                    "momentum_override_applied": True,
                    "lock_in_multiplier": 1.0,
                    "lock_in_check": {},
                }
            ],
        }

        result = agent.finalize_recommendation(state)

        payload = result["recommendations"][0]["payload"]
        self.assertTrue(payload["momentum_override_applied"])
        self.assertEqual(payload["recent_results"]["momentum"], "STRONG")
        self.assertEqual(payload["original_entry_signal"], "WAIT")
        self.assertEqual(payload["entry_signal"], "ACCUMULATE")

    def test_get_tariff_signal_returns_mapping_for_consumer_durables(self) -> None:
        tariff = get_tariff_signal("Consumer Durables")

        self.assertEqual(tariff["impact"], "NEGATIVE")
        self.assertEqual(tariff["date"], "2026-04-02")
        self.assertIn("US 26% tariff", tariff["reason"])

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_finalize_recommendation_reduces_first_tranche_for_tariff_hit_sector(self, _gov_mock, _target_mock) -> None:
        repo = StubRepo()
        agent = BuyAgents(repo, StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())
        state = {
            "request": {"top_n": 1},
            "confidence": {"band": "GREEN"},
            "portfolio_context": {"normalized_exposure": []},
            "allocations": [
                {
                    "symbol": "DURABLE1",
                    "company_name": "Demo Co",
                    "sector": "Consumer Durables",
                    "quality_score": 0.8,
                    "gap_reason": "Gap fill",
                    "overlap_pct": 0.0,
                    "fund_attribution": [],
                    "initial_tranche_pct": 8.0,
                    "target_pct": 20.0,
                    "initial_amount_inr": 8000.0,
                    "target_amount_inr": 20000.0,
                    "allocation_pct": 8.0,
                    "allocation_amount": 8000.0,
                    "tranches": 3,
                    "differentiation_score": 0.8,
                    "news": {"headline": "Positive update"},
                    "price_context": {"price": 100.0, "analyst_target": 140.0},
                    "live_financials": {"currentPrice": 100.0},
                    "financials": {"currentPrice": 100.0},
                    "entry_signal": "ACCUMULATE",
                    "original_entry_signal": "ACCUMULATE",
                    "recent_results": {},
                    "momentum_override_applied": False,
                    "lock_in_multiplier": 1.0,
                    "lock_in_check": {},
                }
            ],
        }

        result = agent.finalize_recommendation(state)

        payload = result["recommendations"][0]["payload"]
        self.assertEqual(payload["tariff_signal"]["impact"], "NEGATIVE")
        self.assertIn("US 26% tariff", payload["tariff_warning"])
        self.assertEqual(payload["initial_tranche_pct"], 5.6)
        self.assertEqual(payload["initial_amount_inr"], 5600.0)
        self.assertEqual(payload["allocation_pct"], 5.6)
        self.assertEqual(payload["allocation_amount"], 5600.0)
        self.assertEqual(payload["entry_levels"]["tranche_1_pct"], 28)


class TestExclusionGate(unittest.TestCase):
    """
    Regression tests for the exclusion-leak bug (2026-04-26).
    exclude_from_recommendations is set inside screener_fetcher but was silently
    dropped at get_financials() in live.py, meaning the gate in
    finalize_recommendation() never fired.  The fix (adding the three fields to the
    get_financials() result dict) is what makes these tests pass.  If they regress,
    the bug has returned.
    """

    def _base_item(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "company_name": "Test Co",
            "sector": "Defence",
            "quality_score": 0.8,
            "gap_reason": "Gap fill",
            "overlap_pct": 0.0,
            "fund_attribution": [],
            "initial_tranche_pct": 8.0,
            "target_pct": 20.0,
            "initial_amount_inr": 8000.0,
            "target_amount_inr": 20000.0,
            "allocation_pct": 8.0,
            "allocation_amount": 8000.0,
            "tranches": 3,
            "differentiation_score": 0.8,
            "entry_signal": "ACCUMULATE",
            "news": {"headline": "Update"},
            "price_context": {"price": 100.0, "analyst_target": 140.0},
            "live_financials": {"currentPrice": 100.0},
        }

    def _run_finalize(self, agent, items):
        return agent.finalize_recommendation({
            "request": {"top_n": len(items)},
            "confidence": {"band": "GREEN"},
            "portfolio_context": {"normalized_exposure": []},
            "allocations": items,
        })

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_excluded_stock_absent_from_recommendations(self, _gov, _target) -> None:
        item = self._base_item("TESTCO")
        item["live_financials"]["exclude_from_recommendations"] = True
        item["live_financials"]["yoy_source"] = "disagree_excluded"
        item["live_financials"]["yoy_confidence"] = "LOW"
        agent = BuyAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())

        result = self._run_finalize(agent, [item])

        rec_symbols = [r["symbol"] for r in result["recommendations"]]
        self.assertNotIn("TESTCO", rec_symbols, "Excluded stock leaked into recommendations")

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_excluded_stock_in_skipped_with_status_data_quality_low(self, _gov, _target) -> None:
        item = self._base_item("TESTCO")
        item["live_financials"]["exclude_from_recommendations"] = True
        item["live_financials"]["yoy_source"] = "disagree_excluded"
        agent = BuyAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())

        result = self._run_finalize(agent, [item])

        skipped = result.get("skipped_stocks", [])
        entry = next((s for s in skipped if s["symbol"] == "TESTCO"), None)
        self.assertIsNotNone(entry, "Excluded stock should appear in skipped_stocks")
        self.assertEqual(entry["status"], "DATA_QUALITY_LOW")

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_yoy_fields_propagate_into_fin_data(self, _gov, _target) -> None:
        """yoy_source + yoy_confidence must reach fin_data so UI confidence rendering works."""
        item = self._base_item("BEL")
        item["live_financials"].update({
            "exclude_from_recommendations": False,
            "yoy_source": "BSE_FILING",
            "yoy_confidence": "HIGH",
            "week52_low": 80.0,
        })
        agent = BuyAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())

        result = self._run_finalize(agent, [item])

        self.assertEqual(len(result["recommendations"]), 1)
        fin_data = result["recommendations"][0]["payload"]["fin_data"]
        self.assertEqual(fin_data.get("yoy_source"), "BSE_FILING")
        self.assertEqual(fin_data.get("yoy_confidence"), "HIGH")

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_low_confidence_passes_but_carries_marker(self, _gov, _target) -> None:
        """exclude_from_recommendations=False with LOW confidence passes through so the UI can warn."""
        item = self._base_item("BEL")
        item["live_financials"].update({
            "exclude_from_recommendations": False,
            "yoy_confidence": "LOW",
            "week52_low": 80.0,
        })
        agent = BuyAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())

        result = self._run_finalize(agent, [item])

        recs = result["recommendations"]
        bel = next((r for r in recs if r["symbol"] == "BEL"), None)
        self.assertIsNotNone(bel, "Stock with exclude=False should pass even with LOW confidence")
        self.assertEqual(bel["payload"]["fin_data"].get("yoy_confidence"), "LOW")

    @patch("stock_platform.agents.buy_agents.fetch_analyst_consensus_target", return_value=140.0)
    @patch("stock_platform.agents.buy_agents.governance_risk_blocks", return_value=(False, ""))
    def test_real_world_leak_symbols(self, _gov, _target) -> None:
        """Direct regression for the three stocks that leaked on 2026-04-26."""
        agent = BuyAgents(StubRepo(), StubProvider(), AppConfig(**LOCAL_DB_CONFIG), StaticLLM())
        for symbol in ("HAL", "ASHOKLEY", "ADANIENSOL"):
            with self.subTest(symbol=symbol):
                item = self._base_item(symbol)
                item["live_financials"]["exclude_from_recommendations"] = True
                item["live_financials"]["yoy_source"] = "disagree_excluded"
                item["live_financials"]["yoy_confidence"] = "LOW"

                result = self._run_finalize(agent, [item])

                rec_symbols = [r["symbol"] for r in result["recommendations"]]
                self.assertNotIn(symbol, rec_symbols, f"{symbol} leaked — same bug as 2026-04-26")


class TestExchangeFetcher:
    """Regression tests for the BSE→NSE migration on 2026-04-26 after BSE API
    silently started returning empty Table responses."""

    def test_nse_fetcher_returns_quarterly_data(self) -> None:
        """NSE fetcher must return populated dict for a known-good symbol (BEL).
        Production smoke check — fails loudly if NSE changes their API."""
        from stock_platform.utils.screener_fetcher import fetch_nse_quarterly
        import pytest
        result = fetch_nse_quarterly("BEL")
        if result is None:
            pytest.skip("NSE unavailable — re-run later")
        assert "latest" in result
        assert "prior_year" in result
        assert result["latest"].get("revenue", 0) > 0, (
            "Revenue is zero or missing — NSE parser regression"
        )

    def test_nse_yoy_resolves_bel_disagreement(self) -> None:
        """Real-world case from 2026-04-26 — yfinance said +24%, Screener said −0.2%,
        both wrong.  NSE should give ~38.6%."""
        from stock_platform.utils.screener_fetcher import fetch_nse_quarterly, compute_yoy_from_quarterly
        import pytest
        data = fetch_nse_quarterly("BEL")
        if data is None:
            pytest.skip("NSE unavailable — re-run later")
        yoy = compute_yoy_from_quarterly(data)
        assert yoy is not None
        assert 30 < yoy < 50, (
            f"BEL Q3 FY25 YoY expected ~38.6%, got {yoy:.1f}% — NSE parser may have regressed"
        )

    def test_resolve_yoy_uses_nse_when_available(self) -> None:
        """When NSE returns a definitive answer, resolve_yoy_disagreement must use NSE and
        NOT exclude the stock, even if yfinance and Screener disagree by >15pp."""
        from unittest.mock import patch
        from stock_platform.utils.screener_fetcher import resolve_yoy_disagreement
        with patch("stock_platform.utils.screener_fetcher.fetch_nse_quarterly") as mock_nse:
            mock_nse.return_value = {
                "latest":     {"revenue": 13860},
                "prior_year": {"revenue": 10000},
            }
            result = resolve_yoy_disagreement(
                symbol="BEL",
                yfinance_yoy=24.0,
                screener_consol_yoy=-0.2,
                standalone_yoy=None,
            )
        assert result["exclude_from_recommendations"] is False
        assert result["source"] == "NSE_FILING"
        assert result["yoy_confidence"] == "HIGH"
        assert 35 < result["yoy_pct"] < 42


class EntryCalculatorTests(unittest.TestCase):
    def test_buy_signal_calculates_entry_stop_and_tranches(self) -> None:
        entry = calculate_entry_levels(
            symbol="BEL",
            current_price=100.0,
            analyst_target=120.0,
            signal="BUY",
            quant_score=0.8,
            fin_data={},
        )

        self.assertEqual(entry["entry_price"], 97.0)
        self.assertEqual(entry["entry_zone_low"], 96.0)
        self.assertEqual(entry["entry_zone_high"], 100.0)
        self.assertEqual(entry["stop_loss"], 82.5)
        self.assertEqual(entry["stop_loss_pct"], 15.0)
        self.assertEqual(entry["risk_reward"], 1.6)
        self.assertEqual(entry["tranche_1_pct"], 40)
        self.assertEqual(entry["tranche_2_pct"], 35)
        self.assertEqual(entry["tranche_3_pct"], 25)

    def test_wait_signal_uses_52_week_low_when_available(self) -> None:
        entry = calculate_entry_levels(
            symbol="WAITCO",
            current_price=200.0,
            analyst_target=260.0,
            signal="WAIT",
            quant_score=0.4,
            fin_data={"week52_low": 120.0},
        )

        self.assertEqual(entry["entry_price"], 160.0)
        self.assertEqual(entry["entry_zone_low"], 155.2)
        self.assertEqual(entry["entry_zone_high"], 164.8)
        self.assertEqual(entry["stop_loss"], 147.2)
        self.assertIn("Wait for", entry["entry_note"])

    def test_synthesis_summary_appends_entry_line(self) -> None:
        synthesis = _append_entry_summary(
            "• COMBINED VERDICT: ENTER NOW.",
            {
                "symbol": "BEL",
                "action": "BUY",
                "payload": {
                    "current_price": 100.0,
                    "analyst_target": 120.0,
                    "entry_signal": "BUY",
                    "quality_score": 0.8,
                    "fin_data": {},
                },
            },
        )

        self.assertIn("ENTRY SUMMARY", synthesis)
        self.assertIn("Enter at ₹97", synthesis)
        self.assertIn("R/R 1.6x", synthesis)


class BuyPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            anthropic_api_key="test-key",
            openai_api_key="test-key",
            **LOCAL_DB_CONFIG,
        )
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
                "revenue_growth_ttm": 24.0,
                "revenue_growth_latest_qtr": 30.0,
                "revenue_growth_latest_qtr_label": "Q3 FY26 vs Q3 FY25",
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
        self.assertIn("single most likely way this thesis fails", anthropic_prompt)
        self.assertIn("BULL-BIASED catalyst analyst", openai_prompt)
        self.assertIn("ROCE:             26.6%", anthropic_llm.calls[0]["user_prompt"])
        self.assertIn("Revenue growth (TTM YoY): 24.0%", anthropic_llm.calls[0]["user_prompt"])
        self.assertIn("Revenue growth (Q3 FY26 vs Q3 FY25): 30.0%", anthropic_llm.calls[0]["user_prompt"])
        self.assertIn(
            "Use quarterly figure for momentum assessment, TTM figure for trend assessment.",
            anthropic_llm.calls[0]["user_prompt"],
        )
        self.assertIn("Exit signal", str(request["messages"][0]["content"]))  # type: ignore[index]

    def test_buy_rationale_includes_rev_pat_divergence_warning(self) -> None:
        anthropic_llm = RecordingLLM(self.config, provider="anthropic")
        stressed_item = self.item | {
            "live_financials": self.item["live_financials"] | {
                "revenue_growth_latest_qtr": 13.0,
                "revenue_growth_latest_qtr_label": "Q3 FY26 vs Q3 FY25",
                "recent_results": {
                    "revenue_yoy_growth_pct": 13.0,
                    "comparison_label": "Q3 FY26 vs Q3 FY25",
                },
                "pat_momentum": {
                    "pat_momentum": "COLLAPSING",
                    "pat_growth_pct": -45.0,
                    "rev_pat_divergence": True,
                    "period": "Q3 FY26 vs Q3 FY25",
                },
            }
        }

        anthropic_llm.buy_rationale(stressed_item, self.portfolio_context)

        user_prompt = anthropic_llm.calls[0]["user_prompt"]
        self.assertIn("CRITICAL DIVERGENCE DETECTED", user_prompt)
        self.assertIn("Revenue growing 13% YoY but PAT declining 45% YoY.", user_prompt)
        self.assertIn("downgrade to WATCHLIST", user_prompt)

    def test_buy_rationale_includes_pat_low_base_qualifier(self) -> None:
        anthropic_llm = RecordingLLM(self.config, provider="anthropic")
        qualified_item = self.item | {
            "live_financials": self.item["live_financials"] | {
                "revenue_growth_latest_qtr": 18.0,
                "revenue_growth_latest_qtr_label": "Q3 FY26 vs Q3 FY25",
                "recent_results": {
                    "revenue_yoy_growth_pct": 18.0,
                    "comparison_label": "Q3 FY26 vs Q3 FY25",
                    "pat_momentum": "STRONG",
                    "pat_growth_pct": 65.2,
                    "period": "Q3 FY26 vs Q3 FY25",
                    "qualifier": "(Rs.71Cr absolute - high growth on small base)",
                    "pat_abs_cr": 71.0,
                },
                "pat_momentum": {
                    "pat_momentum": "STRONG",
                    "pat_growth_pct": 65.2,
                    "period": "Q3 FY26 vs Q3 FY25",
                    "qualifier": "(Rs.71Cr absolute - high growth on small base)",
                    "pat_abs_cr": 71.0,
                },
            }
        }

        anthropic_llm.buy_rationale(qualified_item, self.portfolio_context)

        user_prompt = anthropic_llm.calls[0]["user_prompt"]
        self.assertIn(
            "PAT momentum (Q3 FY26 vs Q3 FY25): STRONG (+65.2% YoY) (Rs.71Cr absolute - high growth on small base)",
            user_prompt,
        )

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

    def test_synthesis_prompt_includes_news_alert_block(self) -> None:
        fake_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
        llm = PlatformLLM(self.config, provider="anthropic")

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            llm.synthesise_comparison(
                stock_name="INDIGO",
                anthropic_rationale="## RISK VERDICT: BUY",
                openai_rationale="## CATALYST VERDICT: BUY NOW",
                news_context={
                    "material_risks_found": True,
                    "summary": "CEO transition and regulatory probe are active.",
                    "flags": [
                        {
                            "severity": "HIGH",
                            "type": "CEO_CHANGE",
                            "headline": "CEO resigned and interim CEO appointed.",
                            "verdict_impact": "DOWNGRADE",
                        }
                    ],
                    "revised_verdict_suggestion": "WATCHLIST",
                    "news_override": True,
                    "news_override_verdict": "WATCHLIST",
                },
            )

        request = FakeAnthropicClient.last_request or {}
        user_text = request["messages"][0]["content"]  # type: ignore[index]
        self.assertIn("RECENT NEWS ALERTS", user_text)
        self.assertIn("CEO resigned and interim CEO appointed.", user_text)
        self.assertIn("WATCHLIST", user_text)
        self.assertIn("HIGH severity flags require explicit acknowledgement", user_text)

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

    def test_openai_requests_use_max_completion_tokens(self) -> None:
        fake_openai = types.SimpleNamespace(
            AuthenticationError=FakeOpenAIAuthenticationError,
            RateLimitError=FakeOpenAIRateLimitError,
            APITimeoutError=FakeOpenAITimeoutError,
        )
        llm = PlatformLLM(self.config, provider="openai")
        llm._client = FakeOpenAIClient()

        with patch.dict(sys.modules, {"openai": fake_openai}):
            llm.buy_rationale(self.item, self.portfolio_context)
            llm.test_openai_connection()

        self.assertIn("max_completion_tokens", FakeOpenAIChat.last_request or {})

    def test_openai_thesis_review_uses_fast_model_and_timeout(self) -> None:
        fake_openai = types.SimpleNamespace(
            AuthenticationError=FakeOpenAIAuthenticationError,
            RateLimitError=FakeOpenAIRateLimitError,
            APITimeoutError=FakeOpenAITimeoutError,
        )
        llm = PlatformLLM(self.config, provider="openai")
        llm._client = FakeOpenAIClient()

        with patch.dict(sys.modules, {"openai": fake_openai}):
            llm.thesis_review(
                holding={"symbol": "BEL", "sector": "Defence"},
                quant_score=0.72,
                sector_signal={"conviction": "BUY"},
                stock_news={"headline": "Order book remains strong", "sentiment_score": 0.2},
            )

        request = FakeOpenAIChat.last_request or {}
        self.assertEqual(request.get("model"), self.config.openai_fast_model)
        self.assertEqual(request.get("timeout"), self.config.openai_timeout_seconds)


class TestRecommendationHistory(unittest.TestCase):
    """Regression tests for the history persistence layer."""

    def _make_repo(self):
        import tempfile
        from stock_platform.data.repository import PlatformRepository
        tmp = tempfile.mkdtemp()
        repo = PlatformRepository(Path(tmp) / "test.db")
        repo.initialize()
        return repo

    def _make_rec(self, symbol: str = "TESTCO", verdict: str = "ACCUMULATE GRADUALLY"):
        from stock_platform.models import RecommendationRecord
        return RecommendationRecord(
            symbol=symbol,
            company_name="Test Co Ltd",
            sector="TestSector",
            action=verdict,
            score=0.75,
            confidence_band="GREEN",
            rationale="Test rationale",
            payload={
                "current_price": 100.0,
                "entry_levels": {
                    "entry_zone_low": 95.0,
                    "entry_zone_high": 100.0,
                    "stop_loss": 88.0,
                    "risk_reward": 2.0,
                    "analyst_target": 130.0,
                },
                "fin_data": {"pe_ratio": 20.0},
            },
        )

    def test_history_persisted_on_run(self):
        repo = self._make_repo()
        rec = self._make_rec()
        repo.persist_recommendation_history(
            run_id="buy-test001",
            recommendations=[rec],
            request={"risk_profile": "Aggressive"},
            macro_flow={"flow_signal": "RISK_ON", "fii_net_5d_cr": 500},
            llm_provider="anthropic",
            llm_models_json='{"anthropic_fast": "claude-haiku-4-5-20251001"}',
        )
        rows = repo.fetch_recommendation_history_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], "ACCUMULATE GRADUALLY")
        self.assertEqual(rows[0]["llm_provider"], "anthropic")
        self.assertEqual(rows[0]["llm_agreement"], "SINGLE_PROVIDER")
        self.assertAlmostEqual(rows[0]["fii_flow_cr"], 500.0)
        self.assertEqual(rows[0]["market_signal"], "RISK_ON")

    def test_duplicate_run_id_symbol_ignored(self):
        repo = self._make_repo()
        rec = self._make_rec()
        for _ in range(2):
            repo.persist_recommendation_history(
                run_id="buy-dup001",
                recommendations=[rec],
                request={"risk_profile": "Balanced"},
                macro_flow={},
                llm_provider="anthropic",
                llm_models_json="{}",
            )
        rows = repo.fetch_recommendation_history_rows()
        self.assertEqual(len(rows), 1)

    def test_lifecycle_aggregates_across_runs(self):
        repo = self._make_repo()
        for i in range(3):
            repo.persist_recommendation_history(
                run_id=f"buy-run{i:03d}",
                recommendations=[self._make_rec()],
                request={"risk_profile": "Balanced"},
                macro_flow={},
                llm_provider="anthropic",
                llm_models_json="{}",
            )
        rows = repo.fetch_recommendation_history_rows()
        self.assertEqual(len(rows), 3)
        symbols = [r["symbol"] for r in rows]
        self.assertEqual(symbols.count("TESTCO"), 3)

    def test_provider_filter_works(self):
        repo = self._make_repo()
        repo.persist_recommendation_history(
            run_id="buy-a001",
            recommendations=[self._make_rec("AAA")],
            request={},
            macro_flow={},
            llm_provider="anthropic",
            llm_models_json="{}",
        )
        repo.persist_recommendation_history(
            run_id="buy-o001",
            recommendations=[self._make_rec("OOO")],
            request={},
            macro_flow={},
            llm_provider="openai",
            llm_models_json="{}",
        )
        anthropic_rows = repo.fetch_recommendation_history_rows(provider_filter="anthropic")
        openai_rows = repo.fetch_recommendation_history_rows(provider_filter="openai")
        self.assertEqual([r["symbol"] for r in anthropic_rows], ["AAA"])
        self.assertEqual([r["symbol"] for r in openai_rows], ["OOO"])

    def test_mark_acted_updates_all_rows_for_symbol(self):
        repo = self._make_repo()
        for i in range(2):
            repo.persist_recommendation_history(
                run_id=f"buy-act{i}",
                recommendations=[self._make_rec("ACTCO")],
                request={},
                macro_flow={},
                llm_provider="anthropic",
                llm_models_json="{}",
            )
        repo.mark_recommendation_acted("ACTCO", 102.5, "bought on dip", "2026-05-05")
        rows = repo.fetch_recommendation_history_rows()
        for r in rows:
            self.assertTrue(r["user_acted"])
            self.assertAlmostEqual(r["user_entry_price"], 102.5)
            self.assertEqual(r["user_notes"], "bought on dip")

    def test_entry_levels_stored_correctly(self):
        repo = self._make_repo()
        rec = self._make_rec()
        repo.persist_recommendation_history(
            run_id="buy-el001",
            recommendations=[rec],
            request={},
            macro_flow={},
            llm_provider="anthropic",
            llm_models_json="{}",
        )
        rows = repo.fetch_recommendation_history_rows()
        r = rows[0]
        self.assertAlmostEqual(r["entry_zone_low"], 95.0)
        self.assertAlmostEqual(r["entry_zone_high"], 100.0)
        self.assertAlmostEqual(r["stop_loss"], 88.0)
        self.assertAlmostEqual(r["rr_ratio"], 2.0)
        self.assertAlmostEqual(r["target_1"], 130.0)
        self.assertAlmostEqual(r["cmp_at_recommendation"], 100.0)
        self.assertAlmostEqual(r["pe_at_recommendation"], 20.0)


if __name__ == "__main__":
    unittest.main()
