"""Household snapshots, member-specific limits, and deterministic placement."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

from wealth.policy.actions import Action, ActionList, WeekInput
from wealth.policy.allocation import AllocationPlan
from wealth.policy.data import required_fx_currencies
from wealth.policy.foundation import FoundationItem, FoundationReport
from wealth.policy.household import (
    MemberProfile,
    build_household_snapshot,
    build_member_contexts,
)
from wealth.policy.personalise import (
    HoldingInput,
    HoldingLotInput,
    MarketContext,
    _member_ltcg_harvest,
    _member_sip_actions,
    _merge_weekly_actions,
)
from wealth.policy.placement import place_actions
from wealth.policy.routes import RouteDecision
from wealth.schemas.assets import PPFAsset
from wealth.schemas.common import ValuationMethod
from wealth.schemas.profile import InvestingPrefs, MemberTax, Residency
from wealth.valuation import Valuation


def _valuation(value: str) -> Valuation:
    amount = Decimal(value)
    return Valuation(
        value_today=amount,
        maturity_value=None,
        maturity_date=None,
        accrued_interest=Decimal(0),
        principal_so_far=amount,
        method=ValuationMethod.USER,
        notes=(),
    )


def _members() -> tuple[MemberProfile, MemberProfile]:
    self_member = MemberProfile(
        id="member-self",
        name="Self",
        relationship="self",
        is_self=True,
        residency=Residency(status="NRI", country="DK"),
        tax=MemberTax(
            tax_slab_pct="30",
            annual_income="500000",
            annual_income_currency="DKK",
            employment_type="salaried",
        ),
        investing=InvestingPrefs.model_validate(
            {
                "new_money_this_week": "100000",
                "monthly_sip_capacity": "30000",
                "ask_used_this_fy_dkk": "50000",
                "lrs_used_this_fy": "0",
            }
        ),
    )
    spouse = MemberProfile(
        id="member-spouse",
        name="Spouse",
        relationship="spouse",
        is_self=False,
        residency=Residency(status="RI", country="IN"),
        tax=MemberTax(
            tax_regime="new",
            tax_slab_pct="20",
            annual_income="1000000",
            employment_type="salaried",
        ),
    )
    return self_member, spouse


def _household():
    members = _members()
    ppf = PPFAsset.model_validate(
        {
            "asset_type": "ppf",
            "member_id": members[1].id,
            "balance": "300000",
            "opened_fy_start_year": 2018,
            "yearly_contribution": "50000",
        }
    )
    assets = ((ppf, _valuation("300000")),)
    household = build_household_snapshot(
        assets,
        (),
        (),
        members,
        self_member_id=members[0].id,
    )
    contexts = build_member_contexts(
        members,
        household,
        assets,
        as_of_year=2026,
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
    )
    return members, assets, household, contexts


def test_member_contexts_keep_tax_and_annual_room_separate() -> None:
    members, _assets, household, contexts = _household()
    self_context, spouse_context = contexts

    assert household.net_worth == Decimal("300000.00")
    assert household.member(members[1].id).snapshot.net_worth == Decimal("300000.00")
    assert household.member(members[0].id).snapshot.net_worth == Decimal("0.00")
    assert self_context.tax_context.slab_pct == Decimal("30")
    assert spouse_context.tax_context.slab_pct == Decimal("20")
    assert self_context.financial_context.annual_income == Decimal("6000000.00")
    assert not self_context.eligibility["ppf"]
    assert self_context.eligibility_status["in_equity_direct"] == "conditional"
    assert self_context.eligibility_conditions["in_equity_direct"] == "direct_equity_nri_pis"
    assert self_context.eligibility_status["ucits_global_etf"] == "allowed"
    assert self_context.eligibility["ucits_global_etf"]
    assert self_context.limits_room["lrs_annual_usd"] == Decimal("0.00")
    assert self_context.limits_room["ppf_annual_inr"] == Decimal("0.00")
    assert self_context.limits_room["ssy_annual_inr"] == Decimal("0.00")
    assert self_context.limits_room["pomis_inr"] == Decimal("0.00")
    assert self_context.limits_room["nsc_inr"] == Decimal("0.00")
    assert spouse_context.limits_room["lrs_annual_usd"] == Decimal("250000.00")
    assert spouse_context.limits_room["lrs_tcs_threshold_inr"] == Decimal("1000000.00")
    assert spouse_context.limits_room["lrs_tcs_pct"] == Decimal("20.00")
    assert spouse_context.limits_room["ppf_annual_inr"] == Decimal("100000.00")
    assert self_context.limits_room["ask_deposit_dkk"] == Decimal("124200.00")


def test_india_equity_prefers_eligible_spouse_and_adds_clubbing_copy() -> None:
    members, _assets, _household_snapshot, contexts = _household()
    action = Action(
        rank=1,
        kind="BUY_LUMPSUM",
        sleeve="equity_india",
        instrument_category="in_core_index",
        amount=Decimal("100000"),
        reason_code="target_gap",
    )

    result = place_actions(
        [action],
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    )

    placed = result.actions[0]
    assert placed.member_id == members[1].id
    assert placed.member_name == members[1].name
    assert placed.member_relationship == "spouse"
    assert placed.instrument_category == "in_core_index"
    assert placed.clubbing_check == (
        "In India, income from money you give your spouse can be taxed as your income. "
        "Invest from their own income or savings, or keep records of a loan."
    )


def test_household_debt_placement_uses_spouse_ppf_room_only() -> None:
    _members_value, _assets, _household_snapshot, contexts = _household()
    action = Action(
        rank=1,
        kind="BUY_LUMPSUM",
        sleeve="debt",
        instrument_category="nre_fd",
        amount=Decimal("100000"),
        reason_code="target_gap",
    )

    result = place_actions(
        [action],
        contexts,
        debt_yields={
            "ppf": Decimal("7.10"),
            "bank_fd": Decimal("6.50"),
            "nre_fd": Decimal("6.50"),
            "nro_fd": Decimal("6.50"),
            "fcnr_b": Decimal("6.50"),
        },
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    )

    assert len(result.actions) == 1
    assert result.actions[0].instrument_category == "ppf"
    assert result.actions[0].amount == Decimal("100000.00")
    assert result.actions[0].member_relationship == "spouse"
    assert "after-tax yield 7.10%" in result.notes[0]


def test_danish_global_equity_uses_member_ask_room() -> None:
    members, _assets, _household_snapshot, contexts = _household()
    action = Action(
        rank=1,
        kind="BUY_LUMPSUM",
        sleeve="equity_global",
        instrument_category="ucits_global_etf",
        amount=Decimal("100000"),
        route="LOCAL",
        reason_code="target_gap",
    )

    result = place_actions(
        [action],
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    )

    assert len(result.actions) == 1
    assert result.actions[0].member_id == members[0].id
    assert result.actions[0].params["account_type"] == "ASK"
    assert result.actions[0].params["ask_tax_rate_pct"] == "17"


def _global_action(amount: str) -> Action:
    return Action(
        rank=1,
        kind="BUY_LUMPSUM",
        sleeve="equity_global",
        instrument_category="ucits_global_etf",
        amount=Decimal(amount),
        route="LOCAL",
        reason_code="target_gap",
    )


def test_ask_room_fits_full_global_purchase() -> None:
    members, assets, _snapshot, _contexts = _household()
    self_member = replace(
        members[0],
        investing=members[0].investing.model_copy(update={"ask_used_this_fy_dkk": Decimal("0")}),
    )
    members = (self_member, members[1])
    household = build_household_snapshot(assets, (), (), members, self_member_id=self_member.id)
    contexts = build_member_contexts(
        members,
        household,
        assets,
        as_of_year=2026,
        fx_rates={"DKK/INR": Decimal("12")},
    )
    assert contexts[0].limits_room["ask_deposit_dkk"] == Decimal("174200.00")

    result = place_actions(
        [_global_action("127500")],
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    )

    assert len(result.actions) == 1
    assert result.actions[0].member_id == members[0].id
    assert result.actions[0].amount == Decimal("127500.00")
    assert result.actions[0].params["account_type"] == "ASK"
    assert "global_equity_ask_room_spent_then_local_account" not in result.notes


def test_ask_room_spills_to_ordinary_account_only_when_exceeded() -> None:
    members, assets, _snapshot, _contexts = _household()
    self_member = replace(
        members[0],
        investing=members[0].investing.model_copy(
            update={"ask_used_this_fy_dkk": Decimal("169200")}
        ),
    )
    adjusted_members = (self_member, members[1])
    household = build_household_snapshot(
        assets, (), (), adjusted_members, self_member_id=self_member.id
    )
    contexts = build_member_contexts(
        adjusted_members,
        household,
        assets,
        as_of_year=2026,
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
    )

    result = place_actions(
        [_global_action("127500")],
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    )

    assert [item.params["account_type"] for item in result.actions] == ["ASK", "ORDINARY"]
    assert [item.amount for item in result.actions] == [
        Decimal("60000.00"),
        Decimal("67500.00"),
    ]
    assert any("global_equity_ask_room_spent_then_local_account" in note for note in result.notes)


def test_missing_dkk_fx_keeps_ask_first_and_marks_room_unchecked() -> None:
    members, _assets, _snapshot, contexts = _household()

    result = place_actions(
        [_global_action("127500")],
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83")},
        as_of_year=2026,
    )

    assert len(result.actions) == 1
    assert result.actions[0].member_id == members[0].id
    assert result.actions[0].params["account_type"] == "ASK"
    assert result.actions[0].amount == Decimal("127500.00")
    assert "ask_room_unchecked_fx_missing" in result.notes
    assert not any(
        "global_equity_ask_room_spent_then_local_account" in note for note in result.notes
    )


def test_required_fx_currencies_include_member_local_currency_without_goal() -> None:
    self_member, spouse = _members()
    self_member = replace(
        self_member,
        tax=MemberTax(annual_income="500000", annual_income_currency="INR"),
    )

    currencies = required_fx_currencies((), (self_member, spouse))

    assert currencies == ("DKK", "EUR", "USD")


def test_sip_redirections_use_each_members_own_capacity() -> None:
    members, _assets, _snapshot, _contexts = _household()
    members = (
        replace(
            members[0],
            investing=members[0].investing.model_copy(
                update={"monthly_sip_capacity": Decimal("40000")}
            ),
        ),
        replace(
            members[1],
            investing=members[1].investing.model_copy(
                update={"monthly_sip_capacity": Decimal("20000")}
            ),
        ),
    )
    gaps = {
        "cash": Decimal(0),
        "debt": Decimal(0),
        "gold": Decimal("10000"),
        "equity_global": Decimal("30000"),
        "equity_india": Decimal("60000"),
    }
    plan = AllocationPlan(
        target_rupees=gaps,
        target_percent={},
        current_rupees={},
        current_percent={},
        gap=gaps,
        buckets=(),
        investable_total=Decimal("100000"),
    )
    foundation = FoundationReport(
        items=(
            FoundationItem("high_interest_debt", "ok", Decimal(0), Decimal(0), Decimal(0)),
            FoundationItem("emergency_fund", "ok", Decimal(0), Decimal(0), Decimal(0)),
        )
    )
    week_input = WeekInput(
        plan=plan,
        foundation=foundation,
        new_money=Decimal(0),
        monthly_sip_capacity=Decimal("60000"),
        existing_sips=[],
        route=RouteDecision("A", "in_international_index_fund", [], [], []),
        satellite_pct=Decimal(0),
        week="2026-W44",
    )

    actions = _member_sip_actions(week_input, members)

    assert {action.member_id: action.amount for action in actions} == {
        members[0].id: Decimal("40000.00"),
        members[1].id: Decimal("20000.00"),
    }


def test_sip_placement_routes_by_member_and_preserves_eligibility() -> None:
    members, _assets, _snapshot, contexts = _household()
    self_action = Action(
        rank=1,
        kind="REBALANCE_SIPS",
        sleeve="multiple",
        instrument_category="monthly_sip",
        amount=Decimal("40000"),
        reason_code="sip_capacity_redirected",
        member_id=members[0].id,
        member_name=members[0].name,
        member_relationship=members[0].relationship,
        params={
            "changes": [
                {"sleeve": "equity_india", "from": "0.00", "to": "12500.00"},
                {"sleeve": "equity_global", "from": "0.00", "to": "27500.00"},
            ]
        },
    )
    spouse_action = Action(
        rank=2,
        kind="REBALANCE_SIPS",
        sleeve="multiple",
        instrument_category="monthly_sip",
        amount=Decimal("20000"),
        reason_code="sip_capacity_redirected",
        member_id=members[1].id,
        member_name=members[1].name,
        member_relationship=members[1].relationship,
        params={
            "changes": [
                {"sleeve": "equity_india", "from": "0.00", "to": "15000.00"},
                {"sleeve": "gold", "from": "0.00", "to": "5000.00"},
            ]
        },
    )

    result = place_actions(
        [self_action, spouse_action],
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    )

    by_member = {member.member_id: member for member in contexts}
    self_actions = [item for item in result.actions if item.member_id == members[0].id]
    spouse_actions = [item for item in result.actions if item.member_id == members[1].id]
    assert len(self_actions) == 1
    assert self_actions[0].params["changes"][0]["account_type"] == "ASK"
    assert all(
        change.get("instrument_category") != "in_core_index"
        for item in self_actions
        for change in item.params["changes"]
    )
    transferred = [
        (item, change)
        for item in spouse_actions
        for change in item.params["changes"]
        if change.get("sleeve") == "equity_india"
    ]
    assert transferred
    assert any(item.clubbing_check for item, _change in transferred)

    eligibility_key = {
        "in_core_index": "in_mf",
        "ucits_global_etf": "ucits_global_etf",
        "gold_etf": "gold_etf",
    }
    for item in result.actions:
        for change in item.params["changes"]:
            instrument = change.get("instrument_category")
            if instrument is not None:
                status = by_member[item.member_id].eligibility_status[eligibility_key[instrument]]
                assert status in {"allowed", "conditional"}

    extra_buy = Action(
        rank=3,
        kind="BUY_LUMPSUM",
        sleeve="equity_global",
        instrument_category="ucits_global_etf",
        amount=Decimal("1000"),
        reason_code="target_gap",
    )
    bounded, _carry = _merge_weekly_actions(
        ActionList(
            week="2026-W44",
            status="ACTIONS",
            actions=[*result.actions, extra_buy],
            carry_forward=Decimal(0),
        ),
        [],
    )
    assert len(bounded) <= 3


def test_new_instrument_buy_placements_match_member_eligibility_status() -> None:
    members, _assets, _snapshot, contexts = _household()
    actions = [
        Action(
            rank=1,
            kind="BUY_LUMPSUM",
            sleeve="equity_india",
            instrument_category="in_core_index",
            amount=Decimal("25000"),
            reason_code="target_gap",
        ),
        _global_action("25000"),
        Action(
            rank=3,
            kind="BUY_LUMPSUM",
            sleeve="gold",
            instrument_category="gold_etf",
            amount=Decimal("25000"),
            reason_code="target_gap",
        ),
    ]

    result = place_actions(
        actions,
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    )

    context_by_id = {context.member_id: context for context in contexts}
    vehicle_keys = {
        "in_core_index": "in_mf",
        "ucits_global_etf": "ucits_global_etf",
        "gold_etf": "gold_etf",
    }
    assert len(result.actions) == 3
    for action in result.actions:
        eligibility_key = vehicle_keys[action.instrument_category]
        status = context_by_id[action.member_id].eligibility_status[eligibility_key]
        assert status in {"allowed", "conditional"}
    assert (
        next(item for item in result.actions if item.sleeve == "equity_india").member_id
        == members[1].id
    )


def test_two_danish_members_keep_individual_limits_and_review_existing_funds() -> None:
    members, assets, _household_snapshot, _contexts = _household()
    both_dk = (
        members[0],
        replace(members[1], residency=Residency(status="NRI", country="DK")),
    )
    household = build_household_snapshot(assets, (), (), both_dk, self_member_id=both_dk[0].id)
    contexts = build_member_contexts(
        both_dk,
        household,
        assets,
        as_of_year=2026,
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        has_indian_funds_by_member={both_dk[0].id: True, both_dk[1].id: True},
    )

    self_context, spouse_context = contexts
    for context in contexts:
        assert not context.eligibility["ppf"]
        assert not context.eligibility["nsc"]
        assert not context.eligibility["scss"]
        assert "dk_non_abis_capital_income" in context.notes
        assert "dk_fund_tax_review" in context.notes
    assert self_context.limits_room["ask_deposit_dkk"] == Decimal("124200.00")
    assert spouse_context.limits_room["ask_deposit_dkk"] == Decimal("174200.00")

    india_action = Action(
        rank=1,
        kind="BUY_LUMPSUM",
        sleeve="equity_india",
        instrument_category="in_core_index",
        amount=Decimal("100000"),
        reason_code="target_gap",
    )
    placed = place_actions(
        [india_action],
        contexts,
        debt_yields={},
        fx_rates={"USD/INR": Decimal("83"), "DKK/INR": Decimal("12")},
        as_of_year=2026,
    ).actions[0]
    assert placed.instrument_category == "in_equity_direct_core"
    assert not contexts[0].eligibility["in_mf"]
    assert not contexts[0].eligibility["in_etf"]
    assert not contexts[1].eligibility["in_mf"]
    assert not contexts[1].eligibility["in_etf"]


def test_ltcg_harvesting_uses_each_members_lots_and_exemption_room() -> None:
    members = _members()
    members = (
        replace(members[0], realised_equity_ltcg=Decimal("120000")),
        replace(members[1], realised_equity_ltcg=Decimal("0")),
    )
    household = build_household_snapshot((), (), (), members, self_member_id=members[0].id)
    contexts = build_member_contexts(
        members,
        household,
        (),
        as_of_year=2027,
        fx_rates={"USD/INR": Decimal("83")},
    )
    holdings = (
        HoldingInput(
            security_id="self-security",
            symbol="SELF",
            market="IN",
            currency="INR",
            security_type="equity",
            sleeve="equity_india",
            category="equity:india",
            quantity=Decimal("100"),
            price=Decimal("150"),
            member_id=members[0].id,
            lots=(
                HoldingLotInput(
                    lot_id="self-lot",
                    quantity=Decimal("100"),
                    cost_per_unit=Decimal("100"),
                    acquired_on=date(2024, 1, 1),
                    member_id=members[0].id,
                ),
            ),
        ),
        HoldingInput(
            security_id="spouse-security",
            symbol="SPOUSE",
            market="IN",
            currency="INR",
            security_type="equity",
            sleeve="equity_india",
            category="equity:india",
            quantity=Decimal("100"),
            price=Decimal("150"),
            member_id=members[1].id,
            lots=(
                HoldingLotInput(
                    lot_id="spouse-lot",
                    quantity=Decimal("100"),
                    cost_per_unit=Decimal("50"),
                    acquired_on=date(2024, 1, 1),
                    member_id=members[1].id,
                ),
            ),
        ),
    )

    harvest = _member_ltcg_harvest(
        holdings,
        contexts,
        MarketContext(fx_rates={}, debt_yields={}, index_valuation_percentiles={}),
        date(2027, 2, 1),
        [],
    )

    assert harvest[members[0].id]["gains_booked"] == "5000.00"
    assert harvest[members[1].id]["gains_booked"] == "10000.00"
    assert harvest[members[0].id]["needs_review"] is True
    assert harvest[members[1].id]["needs_review"] is False
