"""Golden and deterministic behaviour for the per-user weekly Personaliser."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

import wealth.policy.personalise as personalise_module
from wealth.policy import NEUTRAL_VALUATION_PERCENTILE
from wealth.policy.actions import Action, ActionList, SatelliteIdea
from wealth.policy.denmark import ask_rule_for_year
from wealth.policy.exits import HoldingFacts, build_exit_card
from wealth.policy.holding_review import StockCheckFacts
from wealth.policy.household import MemberProfile
from wealth.policy.personalise import (
    HoldingInput,
    HoldingLotInput,
    MarketContext,
    PolicyProfile,
    ResearchPack,
    UserData,
    WeeklySpendInput,
    _acquisition_fx_notes,
    _digest_inputs,
    _exit_action,
    _merge_weekly_actions,
    _normalised_lots,
    personalise,
    plan_health_score,
)
from wealth.policy.residency import load_residency_rules
from wealth.policy.tax import Lot, TaxContext
from wealth.schemas.assets import parse_asset
from wealth.schemas.common import ValuationMethod
from wealth.schemas.profile import FinancialContext, Goal, InvestingPrefs, MemberTax, Residency
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
            domicile="US",
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
    stock_check = StockCheckFacts(
        security_id="synthetic-india-equity",
        week=_WEEK,
        as_of=_AS_OF,
        market="IN",
        quant_decile=6,
        pe_percentile=Decimal("50"),
        pe_vs_sector=Decimal("1.0"),
        price=Decimal("450"),
    )
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
        stock_checks={stock_check.security_id: stock_check},
        stock_check_week=_WEEK,
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


def test_weekly_spending_is_part_of_the_plan_digest() -> None:
    user_data = synthetic_user_data()
    with_spending = replace(
        user_data,
        weekly_spending=(WeeklySpendInput("self", "equity_global", Decimal("2000.00")),),
    )

    assert _plan_json(user_data)["inputs_digest"] != _plan_json(with_spending)["inputs_digest"]


def test_stock_check_week_is_part_of_the_plan_digest() -> None:
    market = synthetic_market()
    following_week = replace(market, stock_check_week="2026-W45")

    assert _digest_inputs(synthetic_user_data(), market, None, _WEEK, _AS_OF) != _digest_inputs(
        synthetic_user_data(), following_week, None, _WEEK, _AS_OF
    )


def test_new_ask_year_rule_changes_the_plan_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    user_data = synthetic_user_data()
    market = synthetic_market()
    as_of = date(2027, 1, 4)
    week = "2027-W01"
    old_rules = load_residency_rules()
    monkeypatch.setattr(personalise_module, "load_residency_rules", lambda: old_rules)
    fallback_digest = _digest_inputs(user_data, market, None, week, as_of)

    new_rules = deepcopy(old_rules)
    new_rules["denmark"]["ask"]["deposit_cap_dkk"][2027] = "180000"
    monkeypatch.setattr(personalise_module, "load_residency_rules", lambda: new_rules)

    assert ask_rule_for_year(new_rules, "deposit_cap_dkk", 2027) == ("180000", True)
    assert _digest_inputs(user_data, market, None, week, as_of) != fallback_digest


def test_fund_lookthrough_amounts_with_many_decimals_still_build_a_plan() -> None:
    # A fund's value x equity share x weight is rarely whole paise.
    idea = SatelliteIdea(
        security_id="synthetic-lookthrough-share",
        symbol="SYNLT",
        market="IN",
        sector="Industrials",
        price=Decimal("100"),
        quant_decile=9,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("40"),
        confidence="HIGH",
    )
    user_data = replace(
        synthetic_user_data(),
        through_fund_inr={idea.security_id: Decimal("1234.567891")},
        through_fund_by_member_security={("self", idea.security_id): Decimal("1234.567891")},
    )

    plan = personalise(
        user_data, synthetic_market(), ResearchPack(week=_WEEK, ideas=(idea,)), _WEEK, _AS_OF
    )

    assert plan.week == _WEEK


def test_us_market_percentile_removes_missing_data_note() -> None:
    market = replace(
        synthetic_market(),
        index_valuation_percentiles={"IN": Decimal("50"), "US": Decimal("85")},
    )

    plan = personalise(synthetic_user_data(), market, None, _WEEK, _AS_OF)

    assert "index_valuation_percentile_missing:US" not in plan.notes


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


def test_india_add_more_plan_uses_neutral_percentile_without_own_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_data = synthetic_user_data()
    original_holding = user_data.holdings[0]
    smaller_holding = replace(
        original_holding,
        quantity=Decimal("1"),
        lots=(replace(original_holding.lots[0], quantity=Decimal("1")),),
    )
    user_data = replace(user_data, holdings=(smaller_holding, *user_data.holdings[1:]))
    check = StockCheckFacts(
        security_id="synthetic-india-equity",
        week=_WEEK,
        as_of=_AS_OF,
        market="IN",
        quant_decile=9,
        pe_vs_sector=Decimal("1.1"),
        price=Decimal("1543.256789"),
    )
    market = replace(synthetic_market(), stock_checks={"synthetic-india-equity": check})
    captured_ideas: list[SatelliteIdea] = []
    build_action_list = personalise_module.build_action_list

    def capture_ideas(week_input, **kwargs):
        captured_ideas.extend(week_input.ideas)
        return build_action_list(week_input, **kwargs)

    monkeypatch.setattr(personalise_module, "build_action_list", capture_ideas)

    plan = personalise(user_data, market, None, _WEEK, _AS_OF)

    assert (
        next(
            review
            for review in plan.holding_reviews
            if review["security_id"] == "synthetic-india-equity"
        )["verdict"]
        == "add_more"
    )
    assert (
        next(
            idea.valuation_percentile
            for idea in captured_ideas
            if idea.security_id == "synthetic-india-equity"
        )
        == NEUTRAL_VALUATION_PERCENTILE
    )


def test_denmark_add_more_rounds_held_us_share_before_new_idea(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = synthetic_user_data()
    investing = base.investing.model_copy(
        update={
            "new_money_this_week": Decimal("2000000"),
            "monthly_sip_capacity": Decimal("0"),
            "existing_sips": [],
            "has_lrs_broker": True,
            "us_stock_picks_opt_in": True,
            "us_stock_picks_pct": Decimal("25"),
        }
    )
    held = HoldingInput(
        security_id="synthetic-held-us-share",
        symbol="HOLDX",
        market="US",
        currency="USD",
        security_type="equity",
        sleeve="equity_global",
        category="equity:global",
        quantity=Decimal("0.01"),
        price=Decimal("123.45"),
        lots=(
            HoldingLotInput(
                lot_id="synthetic-held-us-lot",
                quantity=Decimal("0.01"),
                cost_per_unit=Decimal("120"),
                acquired_on=date(2024, 1, 1),
                currency="USD",
            ),
        ),
        price_as_of=_AS_OF,
        sector="industrials",
        member_id="self",
        name="Synthetic Held Company",
        isin="US0000000101",
        exchange="NASDAQ",
    )
    user_data = replace(
        base,
        holdings=(base.holdings[1], held),
        financial_context=base.financial_context.model_copy(
            update={"monthly_expenses": Decimal("10000"), "dependants": 0}
        ),
        investing=investing,
        members=(
            MemberProfile(
                id="self",
                name="Synthetic Member",
                relationship="self",
                is_self=True,
                residency=Residency(status="NRI", country="DK"),
                tax=MemberTax(),
                investing=investing,
            ),
        ),
    )
    check = StockCheckFacts(
        security_id=held.security_id,
        week=_WEEK,
        as_of=_AS_OF,
        market="US",
        quant_decile=9,
        pe_vs_sector=Decimal("1.1"),
        price=Decimal("123.456789"),
    )
    market = replace(
        synthetic_market(),
        fx_rates={"USD/INR": Decimal("83.50"), "DKK/INR": Decimal("12.5")},
        stock_checks={
            **synthetic_market().stock_checks,
            held.security_id: check,
        },
    )
    new_idea = SatelliteIdea(
        security_id="synthetic-new-us-share",
        symbol="NEWX",
        market="US",
        sector="technology",
        price=Decimal("100"),
        quant_decile=10,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("35"),
        confidence="HIGH",
        name="Synthetic New Company",
        isin="US0000000102",
        exchange="NASDAQ",
        currency="USD",
    )
    captured_ideas: list[SatelliteIdea] = []
    approved_us_stock_buys = personalise_module.approved_us_stock_buys

    def capture_ideas(ideas, **kwargs):
        captured_ideas.extend(ideas)
        return approved_us_stock_buys(ideas, **kwargs)

    monkeypatch.setattr(personalise_module, "approved_us_stock_buys", capture_ideas)

    plan = personalise(
        user_data,
        market,
        ResearchPack(week=_WEEK, ideas=(new_idea,)),
        _WEEK,
        _AS_OF,
    )

    assert (
        next(
            review for review in plan.holding_reviews if review["security_id"] == held.security_id
        )["verdict"]
        == "add_more"
    )
    tickers = [line["instrument"]["ticker"] for line in plan.abroad_buy_list]
    assert held.symbol in tickers
    assert new_idea.symbol in tickers
    assert tickers.index(held.symbol) < tickers.index(new_idea.symbol)
    assert (
        next(
            idea.valuation_percentile
            for idea in captured_ideas
            if idea.security_id == held.security_id
        )
        == NEUTRAL_VALUATION_PERCENTILE
    )


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


def test_home_has_one_sell_step_and_keeps_both_holding_reviews() -> None:
    base = synthetic_user_data()
    holdings = tuple(
        HoldingInput(
            security_id=security_id,
            symbol=symbol,
            market="IN",
            currency="INR",
            security_type="equity",
            sleeve="equity_india",
            category="equity:india",
            quantity=Decimal(quantity),
            price=Decimal("1000"),
            lots=(
                HoldingLotInput(
                    lot_id=f"{security_id}-lot",
                    quantity=Decimal(quantity),
                    cost_per_unit=Decimal("500"),
                    acquired_on=date(2024, 1, 1),
                ),
            ),
            price_as_of=_AS_OF,
            sector="industrials",
        )
        for security_id, symbol, quantity in (
            ("synthetic-sale-small", "SELLS", "10"),
            ("synthetic-sale-large", "SELLL", "20"),
        )
    )
    user_data = replace(base, holdings=holdings)
    market = replace(
        synthetic_market(),
        stock_checks={
            holding.security_id: StockCheckFacts(
                security_id=holding.security_id,
                week=_WEEK,
                as_of=_AS_OF,
                market="IN",
                quant_decile=4,
                pe_percentile=Decimal("50"),
                price=Decimal("1000"),
                revenue_yoy_pct=(Decimal("-5"), Decimal("-3")),
            )
            for holding in holdings
        },
    )

    plan = _plan_json(user_data=user_data, market=market)

    sells = [action for action in plan["actions"] if action["kind"].startswith("SELL_")]
    assert len(sells) == 1
    assert sells[0]["symbol"] == "SELLL"
    assert {review["security_id"] for review in plan["holding_reviews"]} == {
        holding.security_id for holding in holdings
    }
    assert all(review["verdict"] == "sell" for review in plan["holding_reviews"])


def test_foreign_tax_lots_use_acquisition_fx_and_skip_missing_basis() -> None:
    holding = HoldingInput(
        security_id="synthetic-global-equity",
        symbol="SYNTH-US",
        market="US",
        currency="USD",
        security_type="equity",
        sleeve="equity_global",
        category="equity:global",
        quantity=Decimal("3"),
        price=Decimal("200"),
        domicile="US",
        lots=(
            HoldingLotInput(
                lot_id="priced-foreign-lot",
                quantity=Decimal("2"),
                cost_per_unit=Decimal("100"),
                acquired_on=date(2024, 1, 1),
                currency="USD",
                cost_per_unit_inr=Decimal("8300"),
            ),
            HoldingLotInput(
                lot_id="missing-foreign-fx-lot",
                quantity=Decimal("1"),
                cost_per_unit=Decimal("150"),
                acquired_on=date(2024, 2, 1),
                currency="USD",
                fx_missing=True,
            ),
        ),
    )
    notes: list[str] = []

    lots = _normalised_lots(holding, {"USD/INR": Decimal("90")}, notes)

    assert len(lots) == 1
    assert lots[0].cost_per_unit == Decimal("8300")
    assert lots[0].market_price == Decimal("18000")
    assert "holding_lot_fx_missing:SYNTH-US" in notes
    assert _acquisition_fx_notes(holding) == (
        "acquisition_fx_missing",
        "acquisition_fx_approximation",
    )


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
        "BUILD_EMERGENCY_FUND",
        "REBALANCE_SIPS",
    ]
    assert [action.symbol for action in merged[:1]] == ["SYNTH-CRITICAL"]
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

    assert [action.kind for action in merged] == ["SWAP", "BUY_LUMPSUM"]
    assert merged[0].security_id == "synthetic-swap"


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
