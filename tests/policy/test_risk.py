from decimal import Decimal

import pytest

from wealth.policy.risk import capacity_score, risk_profile, risk_questions, tolerance_score
from wealth.schemas.profile import FinancialContext


def _answers(index: int) -> dict[str, str]:
    return {question["id"]: question["options"][index]["id"] for question in risk_questions()}


def test_tolerance_score_is_the_mean_of_six_configured_answers() -> None:
    answers = {}
    scores = (0, 33, 67, 100, 33, 67)
    for question, score_index in zip(risk_questions(), (0, 1, 2, 3, 1, 2), strict=True):
        answers[question["id"]] = question["options"][score_index]["id"]

    assert tolerance_score(answers) == Decimal(sum(scores)) / Decimal(6)


@pytest.mark.parametrize(
    "answers",
    [
        {"unknown_question": "anything"},
        {question["id"]: "unknown_option" for question in risk_questions()},
        {},
    ],
)
def test_tolerance_score_rejects_unknown_or_missing_answers(answers: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        tolerance_score(answers)


def test_capacity_score_adds_horizon_stability_dependants_and_reserve_points() -> None:
    context = FinancialContext(
        monthly_expenses="50000",
        dependants=1,
        income_stability="medium",
    )

    assert capacity_score(context, horizon_years=5, emergency_ok=False) == Decimal(35)
    assert capacity_score(context, horizon_years=8, emergency_ok=True) == Decimal(85)


def test_capacity_score_clamps_to_zero_for_short_horizon_and_missing_emergency_fund() -> None:
    context = FinancialContext(
        monthly_expenses="50000",
        dependants=3,
        income_stability="low",
    )

    assert capacity_score(context, horizon_years=2, emergency_ok=False) == Decimal(0)


def test_risk_profile_explains_when_capacity_sets_the_lower_band() -> None:
    context = FinancialContext(
        monthly_expenses="50000",
        dependants=3,
        income_stability="low",
    )

    result = risk_profile(context, 2, False, _answers(3))

    assert result.tolerance == Decimal(100)
    assert result.capacity == Decimal(0)
    assert result.final == "conservative"
    assert result.limited_by == "capacity"


def test_risk_profile_explains_when_tolerance_sets_the_lower_band() -> None:
    context = FinancialContext(
        monthly_expenses="50000",
        dependants=0,
        income_stability="high",
    )

    result = risk_profile(context, 10, True, _answers(0))

    assert result.capacity == Decimal(100)
    assert result.final == "conservative"
    assert result.limited_by == "tolerance"
