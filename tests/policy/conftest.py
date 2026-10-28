from collections.abc import Callable
from decimal import Decimal

import pytest

from wealth.schemas.common import ValuationMethod
from wealth.valuation.base import Valuation


@pytest.fixture
def make_valuation() -> Callable[[str], Valuation]:
    def create(amount: str) -> Valuation:
        value = Decimal(amount)
        return Valuation(
            value_today=value,
            maturity_value=None,
            maturity_date=None,
            accrued_interest=Decimal("0.00"),
            principal_so_far=value,
            method=ValuationMethod.USER,
            notes=(),
        )

    return create
