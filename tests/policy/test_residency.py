from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from wealth.marketdata.sources import ecb
from wealth.policy.allocation import build_targets
from wealth.policy.debt_picker import DebtNeed, load_debt_rules, pick_debt
from wealth.policy.global_split import split_equity
from wealth.policy.personalise import (
    HoldingInput,
    MarketContext,
    PolicyProfile,
    UserData,
    personalise,
)
from wealth.policy.residency import (
    eligibility_rule,
    eligible_for_new_money,
    india_core_vehicle,
    jurisdiction_disclaimer,
    load_residency_rules,
    residency_note_codes,
)
from wealth.policy.routes import RouteContext, select_route
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import parse_asset
from wealth.schemas.common import LiquidityClass, ValuationMethod
from wealth.schemas.profile import (
    FinancialContext,
    Goal,
    InvestingPrefs,
    Residency,
    ResidencyInput,
    residency_from_stored,
)
from wealth.valuation import Valuation

_AS_OF = date(2026, 10, 28)
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"


def _allocation(residency: Residency, *, goals: tuple[Goal, ...] = ()) -> object:
    snapshot = WealthSnapshot(market_holdings=[("equity_india", Decimal("1000000"), "instant")])
    base = build_targets(
        snapshot,
        FinancialContext(monthly_expenses="0"),
        goals,
        "moderate",
        _AS_OF,
        fx_rates={"USD/INR": Decimal("95.1242"), "DKK/INR": Decimal("14.1906")},
    )
    return split_equity(
        base,
        goals,
        "moderate",
        residency=residency,
        fx_rates={"USD/INR": Decimal("95.1242"), "DKK/INR": Decimal("14.1906")},
        as_of=_AS_OF,
    )


def _policy_user(
    residency: Residency,
    *,
    dkk_goal: bool = False,
    monthly_expenses: str = "0",
    us_situs_usd: Decimal | None = None,
) -> UserData:
    asset = parse_asset({"asset_type": "savings", "bank": "Synthetic Bank", "balance": "100000"})
    asset_value = Valuation(
        value_today=Decimal("100000.00"),
        maturity_value=None,
        maturity_date=None,
        accrued_interest=Decimal("0.00"),
        principal_so_far=Decimal("100000.00"),
        method=ValuationMethod.USER,
        notes=(),
    )
    fund_holding = HoldingInput(
        security_id="synthetic-india-fund",
        symbol="SYNTHFUND",
        market="IN",
        currency="INR",
        security_type="mf",
        sleeve="equity_india",
        category="equity:india",
        quantity=Decimal("100"),
        price=None,
    )
    goals = (
        (
            Goal(
                name="Synthetic Denmark goal",
                goal_type="education_abroad",
                target_amount="1000",
                target_year=2032,
                currency="DKK",
                priority=1,
            ),
        )
        if dkk_goal
        else ()
    )
    us_holding = (
        HoldingInput(
            security_id="synthetic-us-stock",
            symbol="SYNTHUS",
            market="US",
            currency="USD",
            domicile="US",
            security_type="equity",
            sleeve="equity_global",
            category="equity:global",
            quantity=Decimal("1"),
            price=us_situs_usd,
        )
        if us_situs_usd is not None
        else None
    )
    return UserData(
        profile=PolicyProfile(
            risk_profile="moderate",
            tax_slab_pct=Decimal("20"),
            residency=residency,
        ),
        financial_context=FinancialContext(
            monthly_expenses=monthly_expenses, annual_income="1000000"
        ),
        goals=goals,
        valued_assets=((asset, asset_value),),
        holdings=(fund_holding, *((us_holding,) if us_holding is not None else ())),
        investing=InvestingPrefs(
            new_money_this_week="100000",
            monthly_sip_capacity="0",
            birth_year=1990,
        ),
    )


def _market() -> MarketContext:
    unavailable_yields = {"local_bank_deposit", "nre_fd", "fcnr_b", "nro_fd"}
    return MarketContext(
        fx_rates={"USD/INR": Decimal("95.1242"), "DKK/INR": Decimal("14.1906")},
        debt_yields={
            item["yield_key"]: Decimal("6.00")
            for item in load_debt_rules()["instruments"]
            if item["yield_key"] not in unavailable_yields
        },
        index_valuation_percentiles={"IN": Decimal("50"), "US": Decimal("50")},
    )


def test_residency_matrix_keeps_routes_allocation_and_new_debt_eligibility_consistent() -> None:
    """The supported residency matrix covers RI, NRI-DK, and RNOR households."""
    cases = [
        (
            "RI",
            Residency(),
            "A",
            Decimal("82.50"),
            {
                "liquid_fund",
                "sweep_fd",
                "bank_fd",
                "po_td_1y",
                "po_td_2y",
                "po_td_3y",
                "po_td_5y",
                "arbitrage_fund",
                "target_maturity_fund",
                "gsec_rbi_retail_direct",
                "nsc",
                "ppf",
                "vpf",
                "scss",
                "pomis",
                "rbi_frsb",
            },
        ),
        (
            "NRI-DK",
            Residency(status="NRI", country="DK"),
            "LOCAL",
            Decimal("25.00"),
            {"gsec_rbi_retail_direct", "nre_fd", "fcnr_b", "nro_fd", "local_bank_deposit"},
        ),
        ("RNOR", Residency(status="RNOR", country="IN"), "LOCAL", Decimal("82.50"), set()),
    ]
    debt_instruments = [item["key"] for item in load_debt_rules()["instruments"]]

    for label, residency, route_code, india_share, debt_eligible in cases:
        plan = _allocation(residency)
        equity_total = plan.target_rupees["equity_india"] + plan.target_rupees["equity_global"]
        actual_india_share = plan.target_rupees["equity_india"] * 100 / equity_total
        route = select_route(
            RouteContext(
                monthly_global_new_money="10000",
                has_lrs_broker=True,
                prefers_no_paperwork=False,
                lrs_used_this_fy="5000000",
                residency=residency,
            )
        )
        eligible = {key for key in debt_instruments if eligible_for_new_money(key, residency)}

        assert actual_india_share == india_share, label
        assert route.route == route_code, label
        assert eligible == debt_eligible, label


def test_three_year_return_glide_moves_nri_target_toward_ri_mix() -> None:
    """At three years to return, 4/7 of the moderate India-share gap closes: 25% to 82.5%."""
    residency = Residency(
        status="NRI",
        country="DK",
        planned_return_year=2029,
        settle_plan="india",
    )
    plan = _allocation(residency)
    equity_total = plan.target_rupees["equity_india"] + plan.target_rupees["equity_global"]
    india_share = (plan.target_rupees["equity_india"] * 100 / equity_total).quantize(
        Decimal("0.01")
    )
    assert india_share == Decimal("57.86")


def test_nri_small_savings_and_equal_yield_nre_rules() -> None:
    residency = Residency(status="NRI", country="DK")
    for key in ("ppf", "nsc", "scss"):
        assert not eligible_for_new_money(key, residency)
    assert eligibility_rule("ppf", residency)["existing"] == "continue_no_extension"

    picks = pick_debt(
        DebtNeed(
            amount=Decimal("100000"),
            horizon_years=Decimal("2"),
            liquidity_need=LiquidityClass.LOCKED,
        ),
        FinancialContext(monthly_expenses="0"),
        Decimal("30"),
        40,
        {"nre_fd": Decimal("5.00"), "nro_fd": Decimal("5.00")},
        WealthSnapshot(),
        residency=residency,
    )
    assert picks[0].instrument == "nre_fd"
    assert picks[0].after_tax_yield_pct == Decimal("5.00")
    assert next(
        pick for pick in picks if pick.instrument == "nro_fd"
    ).after_tax_yield_pct == Decimal("3.50")


def test_danish_policy_uses_ucits_direct_india_core_and_eu_copy() -> None:
    """A Denmark NRI uses UCITS and direct Indian core, with Danish notes and PRIIPs copy."""
    residency = Residency(status="NRI", country="DK")
    public_user = replace(_policy_user(residency), deployment_mode="public")
    plan = personalise(public_user, _market(), None, "2026-W44", _AS_OF)
    categories = {action["instrument_category"] for action in plan.actions}
    assert plan.route["route"] == "LOCAL"
    assert plan.route["vehicle"] == "ucits_global_etf"
    assert "estate_tax_us_domiciled_assets" not in plan.route["warnings"]
    assert "estate_tax_us_domiciled_assets" not in plan.residency_notes
    assert india_core_vehicle(residency) == "in_equity_direct_core"
    assert "in_equity_direct_core" in categories
    assert not categories.intersection({"in_mf", "in_etf"})
    assert "dk_non_abis_capital_income" in plan.residency_notes
    assert "dk_abis_check" in plan.residency_notes
    assert "eu_priips_kid" in plan.residency_notes
    assert "Danish residents" not in plan.disclaimer
    assert plan.disclaimer == jurisdiction_disclaimer(residency)
    assert "MiFID II" in plan.disclaimer
    assert "dk_fund_tax_review" in plan.residency_notes
    unranked = {item["instrument"]: item["reason"] for item in plan.unranked_options}
    assert {"local_bank_deposit", "nre_fd", "fcnr_b", "nro_fd"}.issubset(unranked)
    assert set(unranked.values()) == {"yield_unavailable"}
    assert not {item["instrument"] for item in plan.debt_picks}.intersection(unranked)
    assert all(item["after_tax_yield_pct"] != "0.00" for item in plan.debt_picks)


def test_dkk_goal_uses_live_format_ecb_reference_cross() -> None:
    dkk_goal = Goal(
        name="Synthetic DKK goal",
        goal_type="education_abroad",
        target_amount="1000",
        target_year=2032,
        currency="DKK",
        priority=1,
    )
    ecb_rows = ecb.parse_ecb_multi_fx_rates(
        (_FIXTURES / "ecb_multi_eur_cross.csv").read_text(encoding="utf-8"),
        fetched_on=_AS_OF,
    )
    fx_rates = {row.pair: row.rate for row in ecb_rows.observations}
    allocation = build_targets(
        WealthSnapshot(market_holdings=[("equity_india", Decimal("1000000"), "instant")]),
        FinancialContext(monthly_expenses="0"),
        [dkk_goal],
        "moderate",
        _AS_OF,
        fx_rates=fx_rates,
    )
    dkk_bucket = next(bucket for bucket in allocation.buckets if bucket.goal_name == dkk_goal.name)
    assert dkk_bucket.goal_target_rupees == Decimal("14190.60")
    assert dkk_bucket.fx_rate_to_inr == Decimal("14.1906")


def test_unsupported_goal_currency_has_clear_source_error() -> None:
    message = "Unsupported goal currency JPY: no ECB reference series"
    with pytest.raises(ValidationError, match=message):
        Goal(
            name="Synthetic unsupported currency goal",
            goal_type="education_abroad",
            target_amount="1000",
            target_year=2032,
            currency="JPY",
            priority=1,
        )


def test_missing_goal_fx_is_a_note_and_the_goal_is_not_converted() -> None:
    goal = Goal(
        name="Synthetic missing FX goal",
        goal_type="education_abroad",
        target_amount="1000",
        target_year=2032,
        currency="DKK",
        priority=1,
    )
    allocation = build_targets(
        WealthSnapshot(market_holdings=[("equity_india", Decimal("1000000"), "instant")]),
        FinancialContext(monthly_expenses="0"),
        [goal],
        "moderate",
        _AS_OF,
    )
    assert allocation.notes == ("goal_fx_missing:DKK",)
    assert all(bucket.goal_name != goal.name for bucket in allocation.buckets)


def test_supported_residency_input_and_legacy_country_normalization() -> None:
    assert ResidencyInput(status="RI", country="IN").country == "IN"
    assert ResidencyInput(status="NRI", country="DK").country == "DK"
    assert ResidencyInput(status="RNOR", country="IN").country == "IN"
    for country in ("US", "XX", "OTHER"):
        with pytest.raises(ValidationError):
            ResidencyInput(status="NRI", country=country)

    legacy_nri_us = residency_from_stored({"status": "NRI", "country": "US"})
    assert legacy_nri_us.country == "OTHER"
    assert "residency_country_unavailable" in residency_note_codes(legacy_nri_us)

    legacy_rows = [
        ({"status": "RI", "country": None}, "IN", []),
        ({"status": "RI"}, "IN", []),
        ({"status": "NRI", "country": None}, "OTHER", ["residency_country_unavailable"]),
        ({"status": "NRI"}, "OTHER", ["residency_country_unavailable"]),
        ({"status": "RNOR", "country": None}, "OTHER", ["residency_country_unavailable"]),
        ({"status": "RNOR"}, "OTHER", ["residency_country_unavailable"]),
        ({"status": "RNOR", "country": "US"}, "OTHER", ["residency_country_unavailable"]),
    ]
    for stored, expected_country, expected_notes in legacy_rows:
        normalized = residency_from_stored(stored)
        assert normalized.country == expected_country
        assert all(note in residency_note_codes(normalized) for note in expected_notes)


def test_legacy_us_person_and_nri_dk_us_situs_estate_warnings() -> None:
    legacy_nri_us_person = residency_from_stored(
        {"status": "NRI", "country": "US", "us_person": True}
    )
    legacy_plan = personalise(
        _policy_user(legacy_nri_us_person, us_situs_usd=Decimal("50000")),
        _market(),
        None,
        "2026-W44",
        _AS_OF,
    )
    assert legacy_nri_us_person.country == "OTHER"
    assert "estate_tax_us_domiciled_assets" not in legacy_plan.route["warnings"]
    assert "estate_tax_us_domiciled_assets" not in legacy_plan.residency_notes

    nri_dk = Residency(status="NRI", country="DK")
    dk_plan = personalise(
        _policy_user(nri_dk, us_situs_usd=Decimal("1000")),
        _market(),
        None,
        "2026-W44",
        _AS_OF,
    )
    assert "estate_tax_us_domiciled_assets" in dk_plan.route["warnings"]
    assert "estate_tax_us_domiciled_assets" in dk_plan.residency_notes


def test_nri_dk_salaried_emergency_fund_uses_nine_months() -> None:
    plan = personalise(
        _policy_user(Residency(status="NRI", country="DK"), monthly_expenses="10000"),
        _market(),
        None,
        "2026-W44",
        _AS_OF,
    )
    emergency = next(item for item in plan.foundation if item["key"] == "emergency_fund")
    assert emergency["required"] == "90000.00"


def test_plan_member_exposes_conditional_eligibility_to_the_pwa() -> None:
    plan = personalise(
        _policy_user(Residency(status="NRI", country="DK")),
        _market(),
        None,
        "2026-W44",
        _AS_OF,
    )

    self_member = plan.members[0]
    assert self_member.eligibility["in_equity_direct"] is False
    assert self_member.eligibility_status["in_equity_direct"] == "conditional"
    assert self_member.eligibility_conditions["in_equity_direct"] == "direct_equity_nri_pis"
    assert self_member.eligibility["ucits_global_etf"] is True
    assert self_member.eligibility_status["ucits_global_etf"] == "allowed"
    assert "ucits_global_etf" not in self_member.eligibility_conditions


def test_residency_rule_note_keys_have_copy_and_source_metadata() -> None:
    rules = load_residency_rules()
    copy = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "copy" / "en.yaml").read_text(encoding="utf-8")
    )
    note_keys = [
        item["note_key"]
        for section in ("instruments", "tax_notes", "resident_only_rules", "us_person_rules")
        for item in rules[section].values()
    ]
    note_keys.extend(
        item["note_key"]
        for item in rules["denmark"].values()
        if isinstance(item, dict) and "note_key" in item
    )
    for country_rules in rules.get("country_rules", {}).values():
        note_keys.extend(country_rules.get("note_keys", []))
        note_keys.extend(country_rules.get("note_keys_with_indian_funds", []))
    missing_copy = sorted(set(note_keys) - set(copy["residency_notes"]))
    assert missing_copy == []
    assert all(
        item["verified"] is False and item["last_checked"] == "2026-10-28"
        for item in rules["instruments"].values()
    )
