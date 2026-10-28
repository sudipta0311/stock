from datetime import date
from decimal import Decimal
from typing import Literal

from wealth.policy.allocation import build_targets
from wealth.policy.rules import load_allocation_rules
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.profile import FinancialContext, Goal


def _goal(
    name: str,
    amount: str,
    year: int,
    priority: int = 1,
    currency: Literal["INR", "USD"] = "INR",
) -> Goal:
    return Goal(
        name=name,
        goal_type="other",
        target_amount=amount,
        target_year=year,
        currency=currency,
        priority=priority,
    )


def _snapshot(amount: str) -> WealthSnapshot:
    return WealthSnapshot(market_holdings=[("equity_india", Decimal(amount), "instant")])


def test_profile_mixes_in_versioned_yaml_each_sum_to_one_hundred() -> None:
    rules = load_allocation_rules()
    for mix in rules["profiles"].values():
        assert sum((Decimal(value) for value in mix.values()), Decimal(0)) == Decimal(100)


def test_allocation_worked_example_uses_whole_wealth_goal_buckets() -> None:
    """Worked example: 50 lakh investable wealth on 1 Nov 2026, moderate profile.

    Monthly expenses are 50,000, giving a 3 lakh emergency reserve. A 5 lakh
    car goal in 2028 uses the short mix. A 20 lakh education goal in 2032
    uses the moderate mix. The remaining 22 lakh uses the moderate profile.
    Final targets: equity 23.10 lakh, debt 20.20 lakh, gold 4.20 lakh,
    and cash 2.50 lakh.
    """
    context = FinancialContext(monthly_expenses="50000", employment_type="salaried")
    goals = [
        _goal("Car", "500000", 2028),
        _goal("Education", "2000000", 2032),
    ]

    plan = build_targets(
        _snapshot("5000000"),
        context,
        goals,
        "moderate",
        date(2026, 11, 1),
        fx_usd_inr=None,
    )

    assert plan.target_rupees == {
        "equity": Decimal("2310000.00"),
        "debt": Decimal("2020000.00"),
        "gold": Decimal("420000.00"),
        "cash": Decimal("250000.00"),
    }
    assert plan.current_equity_rupees == {
        "equity_india": Decimal("5000000.00"),
        "equity_global": Decimal("0.00"),
    }
    assert sum(plan.target_percent.values(), Decimal(0)) == Decimal(100)
    assert sum(plan.current_rupees.values(), Decimal(0)) == Decimal("5000000.00")
    assert sum(plan.target_rupees.values(), Decimal(0)) == Decimal("5000000.00")
    assert [bucket.kind for bucket in plan.buckets] == [
        "emergency",
        "short_goal",
        "medium_goal",
        "long_term",
    ]


def test_medium_goal_glide_shifts_equity_share_into_debt() -> None:
    context = FinancialContext(monthly_expenses="0")

    h4 = build_targets(
        _snapshot("1000000"),
        context,
        [_goal("Home", "1000000", 2030)],
        "moderate",
        date(2026, 1, 1),
        fx_usd_inr=None,
    )
    h6 = build_targets(
        _snapshot("1000000"),
        context,
        [_goal("Home", "1000000", 2032)],
        "moderate",
        date(2026, 1, 1),
        fx_usd_inr=None,
    )

    assert h4.target_rupees == {
        "equity": Decimal("275000.00"),
        "debt": Decimal("625000.00"),
        "gold": Decimal("100000.00"),
        "cash": Decimal("0.00"),
    }
    assert h6.target_rupees == {
        "equity": Decimal("550000.00"),
        "debt": Decimal("350000.00"),
        "gold": Decimal("100000.00"),
        "cash": Decimal("0.00"),
    }


def test_medium_goal_caps_growth_profile_at_moderate() -> None:
    context = FinancialContext(monthly_expenses="0")

    plan = build_targets(
        _snapshot("1000000"),
        context,
        [_goal("Study", "1000000", 2032)],
        "aggressive",
        date(2026, 1, 1),
        fx_usd_inr=None,
    )

    assert plan.target_rupees["equity"] == Decimal("550000.00")
    assert plan.target_rupees["debt"] == Decimal("350000.00")
    assert plan.buckets[1].profile_band == "moderate"


def test_usd_goal_missing_fx_is_not_silently_converted() -> None:
    context = FinancialContext(monthly_expenses="0")
    goal = _goal("Education", "25000", 2034, currency="USD")

    plan = build_targets(
        _snapshot("2112500"),
        context,
        [goal],
        "moderate",
        date(2026, 1, 1),
        fx_usd_inr=None,
    )
    assert plan.notes == ("goal_fx_missing:USD",)
    assert all(bucket.goal_name != goal.name for bucket in plan.buckets)


def test_usd_goal_is_converted_and_explained_at_profile_mix() -> None:
    context = FinancialContext(monthly_expenses="0")
    goal = _goal("Education", "25000", 2034, currency="USD")

    plan = build_targets(
        _snapshot("2112500"),
        context,
        [goal],
        "moderate",
        date(2026, 1, 1),
        fx_usd_inr=Decimal("84.50"),
    )

    assert plan.fx_usd_inr == Decimal("84.50")
    assert plan.target_rupees == {
        "equity": Decimal("1161875.00"),
        "debt": Decimal("739375.00"),
        "gold": Decimal("211250.00"),
        "cash": Decimal("0.00"),
    }
    bucket = next(bucket for bucket in plan.buckets if bucket.kind == "long_goal")
    assert bucket.amount == Decimal("2112500.00")
    assert bucket.goal_target_rupees == Decimal("2112500.00")
    assert bucket.goal_currency == "USD"
    assert bucket.fx_usd_inr == Decimal("84.50")
