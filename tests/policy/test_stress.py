from decimal import Decimal

import pytest

from wealth.policy.stress import load_stress_rules, stress_compare, stress_test

CURRENT = {
    "equity_india": Decimal("9000000"),
    "equity_global": Decimal("500000"),
    "debt": Decimal("7000000"),
    "gold": Decimal("1100000"),
    "cash": Decimal("1000000"),
}
TARGET = {
    "equity_india": Decimal("7812000"),
    "equity_global": Decimal("2232000"),
    "debt": Decimal("6696000"),
    "gold": Decimal("1860000"),
    "cash": Decimal("0"),
}


@pytest.mark.parametrize(
    ("scenario", "current_total", "target_total", "difference"),
    [
        ("inr_minus_10", "50000.00", "223200.00", "173200.00"),
        ("india_equity_minus_25", "-2250000.00", "-1953000.00", "297000.00"),
        ("us_equity_minus_30", "-150000.00", "-669600.00", "-519600.00"),
        ("oil_shock", "-1307000.00", "-978360.00", "328640.00"),
        ("global_risk_off", "-2180000.00", "-2098080.00", "81920.00"),
    ],
)
def test_stress_comparison_matches_rupee_examples(
    scenario: str, current_total: str, target_total: str, difference: str
) -> None:
    comparison = stress_compare(CURRENT, TARGET, scenario)

    assert comparison == {
        "current_total": Decimal(current_total),
        "target_total": Decimal(target_total),
        "difference": Decimal(difference),
    }


def test_global_equity_return_compounds_usd_move_and_fx() -> None:
    impact = stress_test(
        {"equity_global": Decimal("100000")},
        {"equity_india": "0", "equity_global_usd": "-0.30", "fx": "0.10", "gold": "0", "debt": "0"},
    )

    assert impact["by_sleeve"]["equity_global"] == Decimal("-23000.00")
    assert impact["total"] == Decimal("-23000.00")


def test_stress_scenarios_are_marked_as_assumptions() -> None:
    rules = load_stress_rules()

    assert all(item["assumption"] is True for item in rules["scenarios"].values())
