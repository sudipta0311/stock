"""Worked examples for weekly deterministic policy actions."""

from collections.abc import Mapping
from decimal import Decimal

import pytest

from wealth.policy.actions import (
    SatelliteHolding,
    SatelliteIdea,
    Sip,
    WeekInput,
    _proportional_shares,
    build_action_list,
    load_buying_rules,
)
from wealth.policy.allocation import AllocationPlan
from wealth.policy.debt_picker import DebtPick
from wealth.policy.foundation import FoundationItem, FoundationReport
from wealth.policy.routes import RouteDecision


def _plan(gaps: Mapping[str, Decimal] | None = None) -> AllocationPlan:
    targets = {
        "equity_india": Decimal("1905750.00"),
        "equity_global": Decimal("404250.00"),
        "debt": Decimal("2020000.00"),
        "gold": Decimal("420000.00"),
        "cash": Decimal("250000.00"),
    }
    default_gaps = {
        "equity_india": Decimal("105750.00"),
        "equity_global": Decimal("404250.00"),
        "debt": Decimal("-380000.00"),
        "gold": Decimal("120000.00"),
        "cash": Decimal("-250000.00"),
    }
    selected_gaps = default_gaps if gaps is None else dict(gaps)
    current = {sleeve: target - selected_gaps[sleeve] for sleeve, target in targets.items()}
    return AllocationPlan(
        target_rupees=targets,
        target_percent={sleeve: Decimal(0) for sleeve in targets},
        current_rupees=current,
        current_percent={sleeve: Decimal(0) for sleeve in targets},
        gap=selected_gaps,
        buckets=(),
        investable_total=Decimal("5000000.00"),
        current_equity_rupees={
            "equity_india": current["equity_india"],
            "equity_global": current["equity_global"],
        },
    )


def _foundation(debt_gap: str = "0", emergency_gap: str = "0") -> FoundationReport:
    debt = Decimal(debt_gap)
    emergency = Decimal(emergency_gap)
    return FoundationReport(
        (
            FoundationItem(
                key="high_interest_debt",
                status="gap" if debt else "ok",
                required=debt,
                have=Decimal(0),
                gap=debt,
            ),
            FoundationItem(
                key="emergency_fund",
                status="gap" if emergency else "ok",
                required=emergency,
                have=Decimal(0),
                gap=emergency,
            ),
            FoundationItem("term_cover", "ok", Decimal(0), Decimal(0), Decimal(0)),
            FoundationItem("health_cover", "ok", Decimal(0), Decimal(0), Decimal(0)),
        )
    )


def _route() -> RouteDecision:
    return RouteDecision(
        route="A",
        vehicle="in_international_index_fund",
        reasons=["simple_route"],
        warnings=[],
        alternatives=[],
    )


def _instant_pick() -> DebtPick:
    return DebtPick(
        instrument="sweep_fd",
        amount=Decimal("200000.00"),
        after_tax_yield_pct=Decimal("4.00"),
        reason_codes=["after_tax_yield"],
    )


def _week_input(
    *,
    plan: AllocationPlan | None = None,
    foundation: FoundationReport | None = None,
    new_money: str = "100000",
    capacity: str = "50000",
    sips: list[Sip] | None = None,
    debt_picks: list[DebtPick] | None = None,
    satellite_pct: str = "20",
    holdings: list[SatelliteHolding] | None = None,
    ideas: list[SatelliteIdea] | None = None,
    valuation_in: str = "50",
    valuation_us: str = "50",
) -> WeekInput:
    return WeekInput(
        plan=plan or _plan(),
        foundation=foundation or _foundation(),
        new_money=Decimal(new_money),
        monthly_sip_capacity=Decimal(capacity),
        existing_sips=[] if sips is None else sips,
        route=_route(),
        debt_picks=[_instant_pick()] if debt_picks is None else debt_picks,
        satellite_pct=Decimal(satellite_pct),
        satellite_holdings=[] if holdings is None else holdings,
        ideas=[] if ideas is None else ideas,
        index_valuation_percentile={
            "IN": Decimal(valuation_in),
            "US": Decimal(valuation_us),
        },
        week="2026-W45",
    )


def test_buying_rules_load_with_expected_caps() -> None:
    rules = load_buying_rules()

    assert rules["max_actions"] == 3
    assert rules["satellite"]["default_pct_of_india_equity"] == "20"


def test_positive_gaps_receive_rounded_proportional_lump_shares() -> None:
    """₹100,000 × the three gaps / ₹630,000, floored to ₹500, leaves ₹500 for global."""
    gaps = {
        "equity_india": Decimal("105750"),
        "equity_global": Decimal("404250"),
        "gold": Decimal("120000"),
    }

    shares, unapplied = _proportional_shares(Decimal("100000"), gaps, Decimal("500"))

    assert shares["equity_india"] == Decimal("16500")
    assert shares["equity_global"] == Decimal("64500")
    assert shares["gold"] == Decimal("19000")
    assert unapplied == 0


def test_sip_redirect_zeros_overweight_and_spreads_pool_over_gaps() -> None:
    """₹20,000 released from debt is split 3,000/13,500/3,500 after ₹500 floors."""
    result = build_action_list(
        _week_input(
            sips=[
                Sip(
                    sleeve="equity_india",
                    instrument_category="in_core_index",
                    monthly_amount="30000",
                ),
                Sip(sleeve="debt", instrument_category="sweep_fd", monthly_amount="20000"),
            ]
        )
    )
    sip_action = next(action for action in result.actions if action.kind == "REBALANCE_SIPS")

    assert sip_action.params["changes"] == [
        {"sleeve": "debt", "from": "20000.00", "to": "0.00"},
        {"sleeve": "equity_india", "from": "30000.00", "to": "33000.00"},
        {"sleeve": "equity_global", "from": "0.00", "to": "13500.00"},
        {"sleeve": "gold", "from": "0.00", "to": "3500.00"},
    ]


def test_action_cap_prioritizes_sip_redirect_then_largest_buys() -> None:
    """The SIP change keeps a slot; of 64,500/19,000/16,500 buys, the last is carried."""
    result = build_action_list(
        _week_input(
            sips=[
                Sip(
                    sleeve="equity_india",
                    instrument_category="in_core_index",
                    monthly_amount="30000",
                ),
                Sip(sleeve="debt", instrument_category="sweep_fd", monthly_amount="20000"),
            ]
        )
    )

    assert [action.kind for action in result.actions] == [
        "REBALANCE_SIPS",
        "BUY_LUMPSUM",
        "BUY_LUMPSUM",
    ]
    assert [action.amount for action in result.actions] == [
        Decimal("20000.00"),
        Decimal("64500.00"),
        Decimal("19000.00"),
    ]
    assert result.actions[1].sleeve == "equity_global"
    assert result.actions[2].sleeve == "gold"
    assert result.carry_forward == Decimal("16500.00")


@pytest.mark.parametrize(
    ("valuation", "expected_tranches", "expected_tranche_amount"),
    [("50", "3", "100000.00"), ("85", "6", "50000.00")],
)
def test_global_lumpsum_uses_stp_and_valuation_tranche_count(
    valuation: str, expected_tranches: str, expected_tranche_amount: str
) -> None:
    """₹300,000 exceeds 3×₹50,000; 50th percentile uses 3 parts and 85th uses 6."""
    only_global_gap = {
        "equity_india": Decimal(0),
        "equity_global": Decimal("300000"),
        "debt": Decimal(0),
        "gold": Decimal(0),
        "cash": Decimal(0),
    }
    sips = [
        Sip(
            sleeve="equity_global",
            instrument_category="in_international_index_fund",
            monthly_amount="50000",
        )
    ]
    result = build_action_list(
        _week_input(
            plan=_plan(only_global_gap),
            new_money="300000",
            capacity="50000",
            sips=sips,
            valuation_us=valuation,
        )
    )
    stp = next(action for action in result.actions if action.kind == "BUY_STP")

    assert stp.amount == Decimal("300000.00")
    assert stp.params["tranches"] == expected_tranches
    assert stp.params["tranche_amount"] == expected_tranche_amount
    assert stp.tranche == {"n": "1", "of": expected_tranches}


def test_satellite_buy_obeys_caps_and_rounds_core_remainder() -> None:
    """After ₹200,000 held, the ₹181,150 gap allows 27×₹1,420, then ₹11,500 core."""
    only_india_gap = {
        "equity_india": Decimal("50000"),
        "equity_global": Decimal(0),
        "debt": Decimal(0),
        "gold": Decimal(0),
        "cash": Decimal(0),
    }
    idea = SatelliteIdea(
        security_id="IN-SYNTH-01",
        symbol="SYNTH",
        market="IN",
        sector="Technology",
        price=Decimal("1420"),
        quant_decile=10,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("42"),
        confidence="HIGH",
    )
    result = build_action_list(
        _week_input(
            plan=_plan(only_india_gap),
            new_money="50000",
            holdings=[
                SatelliteHolding(
                    security_id="IN-SYNTH-05",
                    sector="Industrials",
                    value=Decimal("200000"),
                )
            ],
            ideas=[idea],
        )
    )
    stock = next(action for action in result.actions if action.kind == "BUY_STOCK_TRANCHE")
    core = next(
        action for action in result.actions if action.instrument_category == "in_core_index"
    )

    assert stock.quantity == Decimal("27")
    assert stock.amount == Decimal("38340.00")
    assert stock.tranche == {"n": "1", "of": "3"}
    assert core.amount == Decimal("11500.00")
    assert result.carry_forward == Decimal("160.00")


@pytest.mark.parametrize("gate", ["valuation", "dossier", "decile", "sector"])
def test_satellite_idea_gates_send_all_india_money_to_core(gate: str) -> None:
    """A failed approval, quant, valuation, or sector cap leaves all ₹50,000 for core."""
    only_india_gap = {
        "equity_india": Decimal("50000"),
        "equity_global": Decimal(0),
        "debt": Decimal(0),
        "gold": Decimal(0),
        "cash": Decimal(0),
    }
    idea_values: dict[str, object] = {
        "security_id": "IN-SYNTH-02",
        "symbol": "SAMPLE",
        "market": "IN",
        "sector": "Technology",
        "price": Decimal("1420"),
        "quant_decile": 10,
        "dossier_status": "APPROVED",
        "valuation_percentile": Decimal("42"),
        "confidence": "HIGH",
    }
    holdings: list[SatelliteHolding] = []
    if gate == "valuation":
        idea_values["valuation_percentile"] = Decimal("92")
    elif gate == "dossier":
        idea_values["dossier_status"] = "PENDING"
    elif gate == "decile":
        idea_values["quant_decile"] = 9
    else:
        holdings = [
            SatelliteHolding(
                security_id="IN-SYNTH-03",
                sector="Technology",
                value=Decimal("95287.50"),
            ),
            SatelliteHolding(
                security_id="IN-SYNTH-04",
                sector="Other",
                value=Decimal("104712.50"),
            ),
        ]
    idea = SatelliteIdea(**idea_values)
    result = build_action_list(
        _week_input(
            plan=_plan(only_india_gap),
            new_money="50000",
            holdings=holdings,
            ideas=[idea],
        )
    )

    assert not any(action.kind == "BUY_STOCK_TRANCHE" for action in result.actions)
    core = next(
        action for action in result.actions if action.instrument_category == "in_core_index"
    )
    assert core.amount == Decimal("50000.00")
    assert result.carry_forward == Decimal("0.00")


def test_foundation_uses_all_available_money_before_any_sleeve_buy() -> None:
    """A ₹200,000 emergency gap consumes the full ₹150,000 before portfolio gaps."""
    result = build_action_list(
        _week_input(
            foundation=_foundation(emergency_gap="200000"),
            new_money="150000",
            capacity="0",
        )
    )

    assert len(result.actions) == 1
    assert result.actions[0].kind == "BUILD_EMERGENCY_FUND"
    assert result.actions[0].instrument_category == "sweep_fd"
    assert result.actions[0].amount == Decimal("150000.00")


def test_high_interest_debt_is_paid_before_emergency_fund_and_sleeve_gaps() -> None:
    """₹80,000 debt is paid first; the remaining ₹20,000 starts the ₹50,000 reserve gap."""
    result = build_action_list(
        _week_input(
            foundation=_foundation(debt_gap="80000", emergency_gap="50000"),
            new_money="100000",
            capacity="0",
        )
    )

    assert [action.kind for action in result.actions] == [
        "PAY_DOWN_DEBT",
        "BUILD_EMERGENCY_FUND",
    ]
    assert [action.amount for action in result.actions] == [
        Decimal("80000.00"),
        Decimal("20000.00"),
    ]


def test_missing_instant_debt_pick_keeps_money_out_of_other_sleeves() -> None:
    result = build_action_list(
        _week_input(
            foundation=_foundation(emergency_gap="200000"),
            new_money="150000",
            capacity="0",
            debt_picks=[],
        )
    )

    assert result.status == "NONE"
    assert result.actions == []
    assert result.carry_forward == Decimal("150000.00")
    assert result.notes == ["emergency_pick_unavailable", "on_plan"]


def test_no_gaps_or_sip_pool_returns_none_on_plan() -> None:
    """No new money, positive gaps, or SIP capacity produces the explicit on-plan result."""
    no_gaps = {sleeve: Decimal("-1") for sleeve in _plan().gap}
    result = build_action_list(
        _week_input(
            plan=_plan(no_gaps),
            foundation=_foundation(),
            new_money="0",
            capacity="0",
        )
    )

    assert result.status == "NONE"
    assert result.actions == []
    assert result.notes == ["on_plan"]


def test_buy_side_actions_have_positive_amounts_and_no_sell_kinds() -> None:
    result = build_action_list(
        _week_input(
            sips=[
                Sip(
                    sleeve="equity_india",
                    instrument_category="in_core_index",
                    monthly_amount="30000",
                ),
                Sip(sleeve="debt", instrument_category="sweep_fd", monthly_amount="20000"),
            ]
        )
    )

    assert all(action.amount > 0 for action in result.actions)
    assert all(
        action.kind
        in {
            "BUY_LUMPSUM",
            "BUY_STP",
            "BUY_STOCK_TRANCHE",
            "BUILD_EMERGENCY_FUND",
            "PAY_DOWN_DEBT",
            "REBALANCE_SIPS",
        }
        for action in result.actions
    )
    encoded = result.model_dump(mode="json")
    assert isinstance(encoded["carry_forward"], str)
    assert isinstance(encoded["actions"][0]["amount"], str)
