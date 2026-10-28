"""Golden and deterministic behaviour for the per-user weekly Personaliser."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

from wealth.policy.actions import Action, ActionList
from wealth.policy.exits import HoldingFacts, build_exit_card
from wealth.policy.personalise import (
    HoldingInput,
    HoldingLotInput,
    MarketContext,
    PolicyProfile,
    UserData,
    _exit_action,
    _merge_weekly_actions,
    personalise,
    plan_health_score,
)
from wealth.policy.tax import Lot, TaxContext
from wealth.schemas.assets import parse_asset
from wealth.schemas.common import ValuationMethod
from wealth.schemas.profile import FinancialContext, Goal, InvestingPrefs
from wealth.valuation import Valuation

_AS_OF = date(2026, 10, 28)
_WEEK = "2026-W44"


def synthetic_user_data() -> UserData:
    assets = (
        (
            parse_asset(
                {
                    "asset_type": "epf",
                    "balance": "1200000",
                    "balance_as_of": "2026-10",
                }
            ),
            Valuation(
                value_today=Decimal("1200000.00"),
                maturity_value=None,
                maturity_date=None,
                accrued_interest=Decimal("0.00"),
                principal_so_far=Decimal("1200000.00"),
                method=ValuationMethod.USER,
                notes=(),
            ),
        ),
        (
            parse_asset(
                {
                    "asset_type": "ppf",
                    "balance": "300000",
                    "opened_fy_start_year": 2018,
                    "yearly_contribution": "60000",
                    "balance_as_of": "2026-10",
                }
            ),
            Valuation(
                value_today=Decimal("300000.00"),
                maturity_value=None,
                maturity_date=None,
                accrued_interest=Decimal("0.00"),
                principal_so_far=Decimal("300000.00"),
                method=ValuationMethod.USER,
                notes=(),
            ),
        ),
        (
            parse_asset(
                {
                    "asset_type": "fd",
                    "bank": "Synthetic Bank",
                    "principal": "250000",
                    "rate_pct": "7.10",
                    "start": "2025-01",
                    "maturity": "2028-01",
                }
            ),
            Valuation(
                value_today=Decimal("250000.00"),
                maturity_value=None,
                maturity_date=None,
                accrued_interest=Decimal("0.00"),
                principal_so_far=Decimal("250000.00"),
                method=ValuationMethod.USER,
                notes=(),
            ),
        ),
        (
            parse_asset({"asset_type": "savings", "bank": "Synthetic Bank", "balance": "150000"}),
            Valuation(
                value_today=Decimal("150000.00"),
                maturity_value=None,
                maturity_date=None,
                accrued_interest=Decimal("0.00"),
                principal_so_far=Decimal("150000.00"),
                method=ValuationMethod.USER,
                notes=(),
            ),
        ),
    )
    goals = (
        Goal(
            name="Synthetic Home",
            goal_type="home",
            target_amount="5000000",
            target_year=2034,
            currency="INR",
            priority=1,
        ),
        Goal(
            name="Synthetic Education",
            goal_type="education_abroad",
            target_amount="60000",
            target_year=2040,
            currency="USD",
            priority=2,
        ),
    )
    holdings = (
        HoldingInput(
            security_id="synthetic-india-equity",
            symbol="SYNTH",
            market="IN",
            currency="INR",
            security_type="equity",
            sleeve="equity_india",
            category="equity:india",
            quantity=Decimal("20"),
            price=Decimal("450"),
            lots=(
                HoldingLotInput(
                    lot_id="synthetic-lot-in-1",
                    quantity=Decimal("20"),
                    cost_per_unit=Decimal("350"),
                    acquired_on=date(2025, 1, 15),
                ),
            ),
            price_as_of=date(2026, 10, 27),
            price_source="synthetic_fixture",
            sector="technology",
        ),
        HoldingInput(
            security_id="synthetic-us-etf",
            symbol="SYNTH-ETF",
            market="US",
            currency="USD",
            security_type="etf",
            sleeve="equity_global",
            category="etf:global",
            quantity=Decimal("2"),
            price=Decimal("200"),
            price_as_of=date(2026, 10, 27),
            price_source="synthetic_fixture",
        ),
    )
    prefs = InvestingPrefs.model_validate(
        {
            "monthly_sip_capacity": "40000",
            "existing_sips": [
                {
                    "sleeve": "equity_india",
                    "instrument_category": "nifty_index_fund",
                    "monthly_amount": "15000",
                },
                {
                    "sleeve": "equity_global",
                    "instrument_category": "international_index_fund",
                    "monthly_amount": "5000",
                },
            ],
            "new_money_this_week": "100000",
            "has_lrs_broker": True,
            "prefers_no_paperwork": False,
            "satellite_pct": "20",
            "birth_year": 1990,
            "lrs_used_this_fy": "100000",
        }
    )
    return UserData(
        profile=PolicyProfile(
            risk_profile="moderate",
            tax_slab_pct=Decimal("20"),
            tax_regime="new",
            city_tier=1,
        ),
        financial_context=FinancialContext.model_validate(
            {
                "monthly_expenses": "60000",
                "annual_income": "1500000",
                "tax_regime": "new",
                "dependants": 1,
                "employment_type": "salaried",
                "income_stability": "high",
                "city_tier": 1,
            }
        ),
        goals=goals,
        valued_assets=assets,
        holdings=holdings,
        investing=prefs,
    )


def synthetic_market() -> MarketContext:
    return MarketContext(
        fx_rates={"USD/INR": Decimal("83.50")},
        debt_yields={
            "sweep_fd": Decimal("6.00"),
            "bank_fd": Decimal("7.00"),
            "liquid_fund": Decimal("6.25"),
            "arbitrage_fund": Decimal("6.50"),
            "target_maturity_fund": Decimal("7.00"),
            "nsc": Decimal("7.70"),
            "ppf": Decimal("7.10"),
        },
        index_valuation_percentiles={"IN": Decimal("50"), "US": Decimal("50")},
        gold_price_per_gram=Decimal("7500"),
        yield_table_as_of=date(2026, 10, 28),
        yield_table_verified=False,
    )


def _plan_json(user_data: UserData | None = None, market: MarketContext | None = None) -> dict:
    plan = personalise(
        user_data or synthetic_user_data(), market or synthetic_market(), None, _WEEK, _AS_OF
    )
    return plan.model_dump(mode="json")


def test_personalise_matches_committed_weekly_plan_golden(request) -> None:
    expected_path = Path(__file__).parents[1] / "golden" / "weekly_plan_user1.json"
    actual_text = json.dumps(_plan_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if request.config.getoption("--update-golden"):
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(actual_text, encoding="utf-8")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    assert json.loads(actual_text) == expected


def test_personalise_is_deterministic_for_identical_inputs() -> None:
    first = _plan_json()
    second = _plan_json()
    assert first["inputs_digest"] == second["inputs_digest"]
    assert first == second


def test_missing_holding_price_is_excluded_and_reported() -> None:
    user_data = synthetic_user_data()
    missing = user_data.holdings[0].__class__(
        **{**user_data.holdings[0].__dict__, "price": None, "price_as_of": None}
    )
    user_data = UserData(
        profile=user_data.profile,
        financial_context=user_data.financial_context,
        goals=user_data.goals,
        valued_assets=user_data.valued_assets,
        holdings=(missing, *user_data.holdings[1:]),
        investing=user_data.investing,
    )
    plan = _plan_json(user_data=user_data)
    assert plan["allocation"]["equity_india"]["current"] == "0.00"
    assert "holding_price_missing:SYNTH" in plan["notes"]


def test_no_research_pack_cannot_create_satellite_stock_buy_actions() -> None:
    plan = _plan_json()
    assert all(action["kind"] != "BUY_STOCK_TRANCHE" for action in plan["actions"])


def test_personaliser_merges_triggered_exit_cards_with_buys() -> None:
    plan = _plan_json()
    actions = plan["actions"]
    sell = next(action for action in actions if action["kind"] == "SELL_TRIM")

    assert len(actions) <= 3
    assert [action["kind"] for action in actions] == [
        "BUILD_EMERGENCY_FUND",
        "REBALANCE_SIPS",
        "SELL_TRIM",
    ]
    assert sell["symbol"] == "SYNTH"
    assert Decimal(sell["quantity"]) > 0
    assert sell["lots_to_sell"]
    assert sell["tax_estimate"]["gross_proceeds"] == sell["amount"]
    assert isinstance(plan["carry_forward"], str)


def test_critical_and_high_exit_actions_precede_foundation_and_buy_actions() -> None:
    today = date(2026, 1, 10)
    lot = Lot(
        lot_id="synthetic-lot",
        tax_class="listed_equity",
        quantity=Decimal("10"),
        cost_per_unit=Decimal("500"),
        acquired_on=date(2025, 1, 22),
    )
    critical = build_exit_card(
        HoldingFacts(
            security_id="synthetic-critical",
            symbol="SYNTH-CRITICAL",
            sector="industrials",
            is_core=False,
            lots=[lot],
            price=Decimal("1000"),
            weight_pct_of_equity=Decimal("5"),
            target_weight_pct=Decimal("5"),
            governance_events=["AUDITOR_RESIGNATION"],
        ),
        today,
        TaxContext(),
    )
    high = build_exit_card(
        HoldingFacts(
            security_id="synthetic-high",
            symbol="SYNTH-HIGH",
            sector="industrials",
            is_core=False,
            lots=[lot],
            price=Decimal("1000"),
            weight_pct_of_equity=Decimal("5"),
            target_weight_pct=Decimal("5"),
            revenue_yoy_pct_last_quarters=[Decimal("-2"), Decimal("-1")],
        ),
        today,
        TaxContext(),
    )
    actions = ActionList(
        week=_WEEK,
        status="ACTIONS",
        actions=[
            Action(
                rank=1,
                kind="BUILD_EMERGENCY_FUND",
                sleeve="debt",
                instrument_category="sweep_fd",
                amount=Decimal("1000"),
                reason_code="emergency_fund_gap",
            ),
            Action(
                rank=2,
                kind="REBALANCE_SIPS",
                sleeve="multiple",
                instrument_category="monthly_sip",
                amount=Decimal("500"),
                reason_code="sip_capacity_redirected",
            ),
            Action(
                rank=3,
                kind="BUY_LUMPSUM",
                sleeve="equity_india",
                instrument_category="in_core_index",
                amount=Decimal("9000"),
                reason_code="target_gap",
            ),
        ],
        carry_forward=Decimal("50"),
    )

    merged, carry_forward = _merge_weekly_actions(actions, [high, critical])

    assert [action.kind for action in merged] == [
        "SELL_EXIT",
        "SELL_EXIT",
        "BUILD_EMERGENCY_FUND",
    ]
    assert [action.symbol for action in merged[:2]] == ["SYNTH-CRITICAL", "SYNTH-HIGH"]
    assert merged[0].quantity == Decimal("10")
    assert merged[0].lots_to_sell
    assert merged[0].tax_estimate is not None
    assert carry_forward == Decimal("9050.00")


def test_buy_actions_fill_remaining_slots_by_amount() -> None:
    buy_list = ActionList(
        week=_WEEK,
        status="ACTIONS",
        actions=[
            Action(
                rank=1,
                kind=kind,
                sleeve="equity_india",
                instrument_category="in_core_index",
                amount=Decimal(amount),
                reason_code="target_gap",
            )
            for kind, amount in (
                ("BUY_LUMPSUM", "100"),
                ("BUY_STP", "1000"),
                ("BUY_STOCK_TRANCHE", "500"),
                ("BUY_LUMPSUM", "800"),
            )
        ],
        carry_forward=Decimal("25"),
    )

    merged, carry_forward = _merge_weekly_actions(buy_list, [])

    assert [action.amount for action in merged] == [
        Decimal("1000"),
        Decimal("800"),
        Decimal("500"),
    ]
    assert carry_forward == Decimal("125.00")


def test_watch_and_wait_exit_cards_remain_informational() -> None:
    today = date(2026, 1, 10)
    wait_card = build_exit_card(
        HoldingFacts(
            security_id="synthetic-wait",
            symbol="SYNTH-WAIT",
            sector="industrials",
            is_core=False,
            lots=[
                Lot(
                    lot_id="synthetic-wait-lot",
                    tax_class="listed_equity",
                    quantity=Decimal("100"),
                    cost_per_unit=Decimal("500"),
                    acquired_on=date(2025, 1, 21),
                )
            ],
            price=Decimal("1000"),
            weight_pct_of_equity=Decimal("7"),
            target_weight_pct=Decimal("5"),
            pe_own_band_percentile=Decimal("93"),
            pe_vs_sector_multiple=Decimal("1.6"),
        ),
        today,
        TaxContext(),
    )
    watch_card = build_exit_card(
        HoldingFacts(
            security_id="synthetic-watch",
            symbol="SYNTH-WATCH",
            sector="industrials",
            is_core=False,
            lots=[],
            price=Decimal("1000"),
            weight_pct_of_equity=Decimal("4"),
            target_weight_pct=Decimal("4"),
            revenue_yoy_pct_last_quarters=[Decimal("-2"), Decimal("3")],
        ),
        today,
        TaxContext(),
    )
    empty = ActionList(week=_WEEK, status="NONE", actions=[], carry_forward=Decimal(0))

    merged, carry_forward = _merge_weekly_actions(empty, [wait_card, watch_card])

    assert wait_card.action == "WAIT_FOR_LONG_TERM"
    assert watch_card.triggers["thesis_break"].status == "watch"
    assert merged == []
    assert carry_forward == 0


def test_review_and_swap_exit_cards_use_sell_side_action_kinds() -> None:
    today = date(2026, 1, 10)
    review = build_exit_card(
        HoldingFacts(
            security_id="synthetic-review",
            symbol="SYNTH-REVIEW",
            sector="industrials",
            is_core=False,
            lots=[
                Lot(
                    lot_id="synthetic-review-lot",
                    tax_class="listed_equity",
                    quantity=Decimal("10"),
                    cost_per_unit=Decimal("1000"),
                    acquired_on=date(2024, 1, 1),
                )
            ],
            price=Decimal("700"),
            weight_pct_of_equity=Decimal("4"),
            target_weight_pct=Decimal("5"),
            below_200dma=True,
            thesis_status="watch",
        ),
        today,
        TaxContext(),
    )
    swap = build_exit_card(
        HoldingFacts(
            security_id="synthetic-swap",
            symbol="SYNTH-SWAP",
            sector="industrials",
            is_core=False,
            lots=[
                Lot(
                    lot_id="synthetic-swap-lot",
                    tax_class="foreign_equity",
                    quantity=Decimal("200"),
                    cost_per_unit=Decimal("871.79485"),
                    acquired_on=date(2024, 3, 1),
                )
            ],
            price=Decimal("1000"),
            weight_pct_of_equity=Decimal("4"),
            target_weight_pct=Decimal("5"),
            better_candidate={
                "security_id": "synthetic-candidate",
                "z_advantage": Decimal("1.3"),
                "weeks_ahead": 5,
                "expected_edge_pct": Decimal("6"),
            },
        ),
        today,
        TaxContext(),
    )

    review_action = _exit_action(review)
    swap_action = _exit_action(swap)

    assert review.action == "REVIEW_THEN_EXIT"
    assert review_action is not None and review_action.kind == "REVIEW_EXIT"
    assert swap.action == "SWAP"
    assert swap_action is not None and swap_action.kind == "SWAP"
    buy_list = ActionList(
        week=_WEEK,
        status="ACTIONS",
        actions=[
            Action(
                rank=1,
                kind="BUY_LUMPSUM",
                sleeve="equity_india",
                instrument_category="in_core_index",
                amount=Decimal("1000"),
                reason_code="target_gap",
            )
        ],
        carry_forward=Decimal(0),
    )

    merged, _carry_forward = _merge_weekly_actions(buy_list, [swap, review])

    assert [action.kind for action in merged] == ["REVIEW_EXIT", "SWAP", "BUY_LUMPSUM"]


def test_plan_health_weights_allocation_gaps_and_open_foundation_items() -> None:
    score = plan_health_score(
        {
            "equity": {"target_pct": "60", "current_pct": "50"},
            "debt": {"target_pct": "40", "current_pct": "50"},
        },
        [
            {"status": "ok", "gap": "0.00"},
            {"status": "gap", "gap": "100.00"},
        ],
    )
    assert score == 80


def test_plan_exposes_main_debt_picks_and_policy_reviews() -> None:
    user_data = synthetic_user_data()
    policy = parse_asset(
        {
            "asset_type": "traditional_policy",
            "insurer": "Synthetic Mutual",
            "plan": "Synthetic Endowment",
            "policy_kind": "endowment",
            "sum_assured": "1000000",
            "annual_premium": "50000",
            "start_year": 2016,
            "term_years": 21,
            "premium_paying_term": 21,
            "bonus_accrued": "300000",
            "maturity_illustration": "2000000",
        }
    )
    policy_valuation = Valuation(
        value_today=Decimal("300000.00"),
        maturity_value=Decimal("2000000.00"),
        maturity_date=None,
        accrued_interest=Decimal("0.00"),
        principal_so_far=Decimal("300000.00"),
        method=ValuationMethod.USER,
        notes=(),
    )
    expanded_user = UserData(
        profile=user_data.profile,
        financial_context=user_data.financial_context,
        goals=user_data.goals,
        valued_assets=(*user_data.valued_assets, (policy, policy_valuation)),
        holdings=user_data.holdings,
        investing=user_data.investing,
        realised_trades_this_fy=user_data.realised_trades_this_fy,
        notes=user_data.notes,
    )
    plan = personalise(expanded_user, synthetic_market(), None, _WEEK, _AS_OF)

    assert plan.plan_health in range(101)
    assert 0 < len(plan.debt_picks) <= 3
    assert {"instrument", "after_tax_yield_pct", "suggested_tenure", "reason_codes"} <= set(
        plan.debt_picks[0]
    )
    assert len(plan.insurance_reviews) == 1
    review = plan.insurance_reviews[0]
    assert {
        "irr_pct",
        "continue_value",
        "paid_up_value",
        "surrender_value",
        "comparison",
        "reason_codes",
        "recommendation_kind",
    } <= set(review)
    assert review["recommendation_kind"] == "compare_options"


def test_importing_personaliser_does_not_load_llm_or_research_packages() -> None:
    code = (
        "import sys; import wealth.policy.personalise; "
        "assert not any(name == 'wealth.research' or name.startswith('wealth.research.') "
        "or name == 'openai' or name.startswith('openai.') "
        "or name == 'langgraph' or name.startswith('langgraph.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
