"""C13 allocation-band and rebalancing examples."""

from datetime import date
from decimal import Decimal

from wealth.policy.actions import WeekInput
from wealth.policy.allocation import AllocationPlan
from wealth.policy.debt_picker import DebtPick
from wealth.policy.foundation import FoundationItem, FoundationReport
from wealth.policy.rebalance import band_breaches, rebalance_plan
from wealth.policy.routes import RouteDecision
from wealth.policy.tax import Lot, TaxContext


def _allocation_plan() -> AllocationPlan:
    targets = {
        "equity_india": Decimal("55000"),
        "equity_global": Decimal("0"),
        "debt": Decimal("35000"),
        "gold": Decimal("10000"),
        "cash": Decimal("0"),
    }
    current = {
        "equity_india": Decimal("61000"),
        "equity_global": Decimal("0"),
        "debt": Decimal("29000"),
        "gold": Decimal("10000"),
        "cash": Decimal("0"),
    }
    return AllocationPlan(
        target_rupees=targets,
        target_percent={
            "equity_india": Decimal("55"),
            "equity_global": Decimal("0"),
            "debt": Decimal("35"),
            "gold": Decimal("10"),
            "cash": Decimal("0"),
        },
        current_rupees=current,
        current_percent={
            "equity_india": Decimal("61"),
            "equity_global": Decimal("0"),
            "debt": Decimal("29"),
            "gold": Decimal("10"),
            "cash": Decimal("0"),
        },
        gap={sleeve: targets[sleeve] - current[sleeve] for sleeve in targets},
        buckets=(),
        investable_total=Decimal("100000"),
    )


def _week_input() -> WeekInput:
    foundation = FoundationReport(
        (
            FoundationItem("high_interest_debt", "ok", Decimal(0), Decimal(0), Decimal(0)),
            FoundationItem("emergency_fund", "ok", Decimal(0), Decimal(0), Decimal(0)),
            FoundationItem("term_cover", "ok", Decimal(0), Decimal(0), Decimal(0)),
            FoundationItem("health_cover", "ok", Decimal(0), Decimal(0), Decimal(0)),
        )
    )
    route = RouteDecision(
        route="A", vehicle="in_international_index_fund", reasons=[], warnings=[], alternatives=[]
    )
    pick = DebtPick(
        instrument="sweep_fd",
        amount=Decimal("1000"),
        after_tax_yield_pct=Decimal("4"),
        reason_codes=["instant"],
    )
    plan = _allocation_plan()
    return WeekInput(
        plan=plan,
        foundation=foundation,
        new_money=Decimal("500"),
        monthly_sip_capacity=Decimal(0),
        route=route,
        debt_picks=[pick],
        satellite_pct=Decimal("20"),
        week="2026-W45",
    )


def test_five_twenty_five_bands_flag_gold_relative_and_equity_absolute() -> None:
    """Gold at 7.4% is 26% below 10%; equity at 61% is 6pp above 55%."""
    plan = _allocation_plan()
    current_pct = dict(plan.current_percent)
    target_pct = dict(plan.target_percent)
    current_pct.update(
        {"gold": Decimal("7.4"), "equity_india": Decimal("61"), "debt": Decimal("31.6")}
    )
    target_pct.update({"gold": Decimal("10"), "equity_india": Decimal("55"), "debt": Decimal("35")})
    plan = AllocationPlan(
        target_rupees=plan.target_rupees,
        target_percent=target_pct,
        current_rupees=plan.current_rupees,
        current_percent=current_pct,
        gap=plan.gap,
        buckets=(),
        investable_total=plan.investable_total,
    )

    breaches = band_breaches(plan)

    assert [(item.sleeve, item.direction) for item in breaches] == [
        ("equity_india", "over"),
        ("gold", "under"),
    ]
    assert breaches[0].excess_amount == Decimal("1000.00")
    assert breaches[1].excess_amount == Decimal("100.00")


def test_rebalance_sells_only_the_amount_beyond_the_band_edge() -> None:
    plan = _allocation_plan()
    breaches = band_breaches(plan)
    lot = Lot(
        lot_id="core-lot",
        tax_class="listed_equity",
        quantity=Decimal("100"),
        cost_per_unit=Decimal("100"),
        acquired_on=date(2024, 1, 1),
    )

    result = rebalance_plan(
        breaches,
        _week_input(),
        {"equity_india": [(lot, Decimal("120"))]},
        date(2026, 10, 1),
        TaxContext(),
    )

    assert len(result.sells) == 1
    assert result.sells[0].sleeve == "equity_india"
    assert result.sells[0].gross_proceeds == Decimal("1000.00")
    assert result.sells[0].tax_estimate.total_tax == Decimal("0.00")
    assert result.sip_redirect is None


def test_unpriced_overweight_lots_do_not_create_a_sale() -> None:
    breaches = band_breaches(_allocation_plan())
    lot = Lot(
        lot_id="unpriced-core",
        tax_class="listed_equity",
        quantity=Decimal("100"),
        cost_per_unit=Decimal("100"),
        acquired_on=date(2024, 1, 1),
    )

    result = rebalance_plan(
        breaches,
        _week_input(),
        {"equity_india": [lot]},
        date(2026, 10, 1),
        TaxContext(),
    )

    assert result.sells == ()
    assert "sale_price_missing:equity_india" in result.notes
