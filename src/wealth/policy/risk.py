"""Deterministic risk tolerance and capacity scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from wealth.policy.rules import (
    AllocationRules,
    decimal_rule,
    integer_rule,
    resolve_rules,
)
from wealth.schemas.profile import FinancialContext


@dataclass(frozen=True)
class RiskResult:
    """Scores, resulting profile band, and which score limited the result."""

    tolerance: Decimal
    capacity: Decimal
    final: str
    limited_by: str | None


def risk_questions(rules: AllocationRules | None = None) -> tuple[Mapping[str, Any], ...]:
    """Return the six configured tolerance scenarios in display order."""
    selected_rules = resolve_rules(rules)
    questions = selected_rules["risk"]["questions"]
    return tuple(questions)


def tolerance_score(answers: dict[str, str], rules: AllocationRules | None = None) -> Decimal:
    """Return the mean score for one answer to every configured question."""
    selected_rules = resolve_rules(rules)
    questions = selected_rules["risk"]["questions"]
    question_ids = {question["id"] for question in questions}
    unknown_questions = set(answers) - question_ids
    missing_questions = question_ids - set(answers)
    if unknown_questions:
        raise ValueError(f"unknown risk question: {min(unknown_questions)}")
    if missing_questions:
        raise ValueError(f"missing risk answer: {min(missing_questions)}")

    scores: list[Decimal] = []
    for question in questions:
        question_id = question["id"]
        option_scores = {option["id"]: option["score"] for option in question["options"]}
        answer = answers[question_id]
        if answer not in option_scores:
            raise ValueError(f"unknown option {answer!r} for question {question_id!r}")
        scores.append(Decimal(integer_rule(option_scores[answer], f"{question_id}.{answer}.score")))
    return sum(scores, Decimal(0)) / Decimal(len(scores))


def capacity_score(
    ctx: FinancialContext,
    horizon_years: int,
    emergency_ok: bool,
    rules: AllocationRules | None = None,
) -> Decimal:
    """Score objective risk capacity, then clamp the additive result to 0–100."""
    if not isinstance(horizon_years, int) or isinstance(horizon_years, bool):
        raise ValueError("horizon_years must be an integer")
    if not isinstance(emergency_ok, bool):
        raise ValueError("emergency_ok must be a boolean")
    selected_rules = resolve_rules(rules)
    horizon = selected_rules["horizon"]
    capacity = selected_rules["risk"]["capacity"]
    short_years = integer_rule(horizon["short_years"], "horizon.short_years")
    medium_years = integer_rule(horizon["medium_years"], "horizon.medium_years")
    if horizon_years < short_years:
        score = integer_rule(
            capacity["horizon_points"]["short"], "risk.capacity.horizon_points.short"
        )
    elif horizon_years <= medium_years:
        score = integer_rule(
            capacity["horizon_points"]["medium"], "risk.capacity.horizon_points.medium"
        )
    else:
        score = integer_rule(
            capacity["horizon_points"]["long"], "risk.capacity.horizon_points.long"
        )

    score += integer_rule(
        capacity["income_stability_points"][ctx.income_stability],
        f"risk.capacity.income_stability_points.{ctx.income_stability}",
    )
    dependant_rules = capacity["dependants"]
    if ctx.dependants <= integer_rule(
        dependant_rules["none_max"], "risk.capacity.dependants.none_max"
    ):
        score += integer_rule(
            dependant_rules["none_points"], "risk.capacity.dependants.none_points"
        )
    elif ctx.dependants <= integer_rule(
        dependant_rules["one_to_two_max"], "risk.capacity.dependants.one_to_two_max"
    ):
        score += integer_rule(
            dependant_rules["one_to_two_points"], "risk.capacity.dependants.one_to_two_points"
        )
    else:
        score += integer_rule(
            dependant_rules["many_points"], "risk.capacity.dependants.many_points"
        )

    emergency_key = "ok" if emergency_ok else "missing"
    score += integer_rule(
        capacity["emergency_fund_points"][emergency_key],
        f"risk.capacity.emergency_fund_points.{emergency_key}",
    )
    bounds = capacity["score_bounds"]
    minimum = integer_rule(bounds["minimum"], "risk.capacity.score_bounds.minimum")
    maximum = integer_rule(bounds["maximum"], "risk.capacity.score_bounds.maximum")
    return Decimal(min(maximum, max(minimum, score)))


def risk_profile(
    ctx: FinancialContext,
    horizon_years: int,
    emergency_ok: bool,
    answers: dict[str, str],
    rules: AllocationRules | None = None,
) -> RiskResult:
    """Combine tolerance and capacity using the lower resulting profile band."""
    selected_rules = resolve_rules(rules)
    tolerance = tolerance_score(answers, selected_rules)
    capacity = capacity_score(ctx, horizon_years, emergency_ok, selected_rules)
    tolerance_band = _profile_band(tolerance, selected_rules)
    capacity_band = _profile_band(capacity, selected_rules)
    bands = selected_rules["risk"]["bands"]
    tolerance_rank = integer_rule(bands[tolerance_band], f"risk.bands.{tolerance_band}")
    capacity_rank = integer_rule(bands[capacity_band], f"risk.bands.{capacity_band}")
    if tolerance_rank < capacity_rank:
        final, limited_by = tolerance_band, "tolerance"
    elif capacity_rank < tolerance_rank:
        final, limited_by = capacity_band, "capacity"
    else:
        final, limited_by = tolerance_band, None
    return RiskResult(tolerance, capacity, final, limited_by)


def _profile_band(score: Decimal, rules: AllocationRules) -> str:
    bands = rules["risk"]["bands"]
    eligible = [
        (integer_rule(lower, f"risk.bands.{name}"), name)
        for name, lower in bands.items()
        if score >= decimal_rule(lower, f"risk.bands.{name}")
    ]
    if not eligible:
        return min(bands, key=lambda name: decimal_rule(bands[name], f"risk.bands.{name}"))
    return max(eligible)[1]
