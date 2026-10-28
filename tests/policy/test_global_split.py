from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from wealth.policy.allocation import AllocationPlan, build_targets
from wealth.policy.global_split import GeoTilts, split_equity
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.profile import FinancialContext, Goal


def _goal(
    amount: str,
    *,
    name: str = "Education",
    target_year: int = 2032,
    goal_type: str = "other",
    currency: str = "INR",
) -> Goal:
    return Goal(
        name=name,
        goal_type=goal_type,
        target_amount=amount,
        target_year=target_year,
        currency=currency,
        priority=1,
    )


def _plan(goals: list[Goal] | None = None, fx_usd_inr: Decimal | None = None) -> AllocationPlan:
    selected_goals = goals
    if selected_goals is None:
        selected_goals = [
            _goal("500000", name="Car", target_year=2028),
            _goal("2000000"),
        ]
    snapshot = WealthSnapshot(market_holdings=[("equity_india", Decimal("5000000"), "instant")])
    return build_targets(
        snapshot,
        FinancialContext(monthly_expenses="50000"),
        selected_goals,
        "moderate",
        date(2026, 11, 1),
        fx_usd_inr=fx_usd_inr,
    )


def test_global_split_uses_base_share_and_foreign_goal_bonus() -> None:
    """Worked example: 2,310,000 equity gives 404,250 global at 17.5%.

    A 20 lakh education-abroad goal is 40% of investable wealth, adding 10 pp and
    taking global equity to 635,250.
    """
    base_goals = [_goal("500000", name="Car", target_year=2028), _goal("2000000")]
    foreign_goals = [
        _goal("500000", name="Car", target_year=2028),
        _goal("2000000", goal_type="education_abroad"),
    ]
    base = split_equity(_plan(base_goals), base_goals, "moderate")
    abroad = split_equity(_plan(foreign_goals), foreign_goals, "moderate")

    assert base.target_rupees["equity_global"] == Decimal("404250.00")
    assert base.target_rupees["equity_india"] == Decimal("1905750.00")
    assert abroad.target_rupees["equity_global"] == Decimal("635250.00")
    assert abroad.target_rupees["equity_india"] == Decimal("1674750.00")
    assert sum(abroad.target_rupees.values(), Decimal(0)) == Decimal("5000000.00")
    assert sum(abroad.target_percent.values(), Decimal(0)) == Decimal("100.00")


def test_geo_tilts_are_bounded_evidence_cited_and_new_money_only() -> None:
    """Worked example: a +2 pp tilt plus a 40% foreign goal makes global equity 681,450.

    A +6 pp tilt or a +2 pp tilt with only one evidence id is rejected.
    """
    goals = [
        _goal("500000", name="Car", target_year=2028),
        _goal("2000000", goal_type="education_abroad"),
    ]
    tilted = GeoTilts(
        global_pp=Decimal("2"),
        evidence_ids=["source-a", "source-b"],
        as_of=date(2026, 10, 28),
    )
    result = split_equity(_plan(goals), goals, "moderate", tilted)
    assert result.target_rupees["equity_global"] == Decimal("681450.00")
    assert result.new_money_only is True

    with pytest.raises(ValidationError):
        GeoTilts(
            global_pp=Decimal("6"),
            evidence_ids=["source-a", "source-b"],
            as_of=date(2026, 10, 28),
        )
    with pytest.raises(ValidationError):
        GeoTilts(global_pp=Decimal("2"), evidence_ids=["source-a"], as_of=date(2026, 10, 28))


def test_gold_tilt_moves_points_between_gold_and_debt_without_changing_equity() -> None:
    """Worked example: +3 pp gold on 5,000,000 moves 150,000 from debt to gold.

    The 2,310,000 equity target is unchanged.
    """
    no_tilt = split_equity(_plan(), [], "moderate")
    tilt = GeoTilts(
        gold_pp=Decimal("3"),
        evidence_ids=["source-a", "source-b"],
        as_of=date(2026, 10, 28),
    )
    adjusted = split_equity(_plan(), [], "moderate", tilt)

    assert adjusted.target_rupees["gold"] == no_tilt.target_rupees["gold"] + Decimal("150000.00")
    assert adjusted.target_rupees["debt"] == no_tilt.target_rupees["debt"] - Decimal("150000.00")
    assert adjusted.target_rupees["equity_india"] + adjusted.target_rupees[
        "equity_global"
    ] == Decimal("2310000.00")


def test_usd_goal_without_fx_does_not_trigger_a_silent_foreign_goal_bonus() -> None:
    """A USD education goal with missing FX is excluded from the foreign-goal bonus."""
    usd_goal = _goal("10000", goal_type="education_abroad", currency="USD")
    plan = split_equity(_plan(), [usd_goal], "moderate")
    base = split_equity(_plan(), [], "moderate")
    assert plan.target_rupees["equity_global"] == base.target_rupees["equity_global"]
