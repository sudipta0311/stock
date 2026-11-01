from decimal import Decimal

import pytest

from wealth.api.personalisation import valuation_label


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [("32", "cheap"), ("33", "cheap"), ("66", "fair"), ("67", "expensive")],
)
def test_valuation_label_uses_research_rule_boundaries(
    percentile: str,
    expected: str,
) -> None:
    assert valuation_label(Decimal(percentile)) == expected
