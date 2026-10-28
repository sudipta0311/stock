"""C13 worked review cases for traditional insurance policies."""

from datetime import date
from decimal import Decimal

from wealth.policy.insurance_review import review_policy
from wealth.schemas.assets import TraditionalPolicyAsset


def _policy(**overrides: object) -> TraditionalPolicyAsset:
    values: dict[str, object] = {
        "asset_type": "traditional_policy",
        "insurer": "Example Mutual",
        "plan": "Traditional Plan",
        "policy_kind": "endowment",
        "sum_assured": Decimal("1000000"),
        "annual_premium": Decimal("50000"),
        "start_year": 2016,
        "term_years": 21,
        "premium_paying_term": 21,
        "bonus_accrued": Decimal("300000"),
        "maturity_illustration": Decimal("2000000"),
    }
    values.update(overrides)
    return TraditionalPolicyAsset.model_validate(values)


def test_paid_up_value_uses_premiums_elapsed_capped_at_ppt() -> None:
    """1,000,000 × 10 / 21 + 300,000 accrued bonus rounds to 776,190.48."""
    review = review_policy(
        _policy(),
        date(2026, 10, 28),
        {"ppf": Decimal("7.10"), "fd": Decimal("7.00")},
    )

    assert review.paid_up_value == Decimal("776190.48")


def test_missing_surrender_value_requests_insurer_quote_without_recommendation() -> None:
    """Without an insurer quote, ask for that value and keep the result as a comparison."""
    review = review_policy(
        _policy(surrender_value=None),
        date(2026, 10, 28),
        {"ppf": Decimal("7.10"), "fd": Decimal("7.00")},
    )

    assert "ask_insurer_for_surrender_value" in review.reason_codes
    assert review.recommendation_kind == "compare_options"
    assert review.surrender_value is None
    assert review.continue_value == Decimal("2000000")


def test_review_compares_policy_irr_to_benchmark_percent_rates() -> None:
    """C04 returns an annual fraction, so review exposes its percentage against both rates."""
    review = review_policy(
        _policy(),
        date(2026, 10, 28),
        {"ppf": Decimal("7.10"), "fd": Decimal("7.00")},
    )

    assert review.irr_pct is not None
    assert review.comparison["ppf"].benchmark_pct == Decimal("7.10")
    assert review.comparison["ppf"].difference_pct == review.irr_pct - Decimal("7.10")
    assert review.recommendation_kind == "compare_options"


def test_missing_illustration_has_no_irr_and_low_cover_gets_a_reason_code() -> None:
    review = review_policy(
        _policy(
            sum_assured=Decimal("400000"),
            maturity_illustration=None,
        ),
        date(2026, 10, 28),
        {"ppf": Decimal("7.10"), "fd": Decimal("7.00")},
    )

    assert review.irr_pct is None
    assert "cover_low_vs_premium" in review.reason_codes
