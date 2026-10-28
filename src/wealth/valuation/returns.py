"""Decimal-only annual return calculations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal, localcontext

from wealth.schemas.assets import TraditionalPolicyAsset

_DAY_COUNT = Decimal(365)
_RATE_EPSILON = Decimal("1e-24")
_LOWER_RATE = Decimal("-0.999999999999999999999999999999")


def xirr(cashflows: Sequence[tuple[date, Decimal]]) -> Decimal:
    """Find the annualised return fraction with Decimal bisection on NPV."""
    if not cashflows:
        raise ValueError("cashflows must not be empty")
    checked: list[tuple[date, Decimal]] = []
    for on, amount in cashflows:
        if not isinstance(on, date):
            raise ValueError("cashflow dates must be dates")
        if not isinstance(amount, Decimal) or not amount.is_finite():
            raise ValueError("cashflow amounts must be finite Decimals")
        checked.append((on, amount))
    if not any(amount < 0 for _, amount in checked) or not any(amount > 0 for _, amount in checked):
        raise ValueError("cashflows must change sign")

    origin = min(on for on, _ in checked)
    with localcontext() as context:
        context.prec = 50

        def npv(rate: Decimal) -> Decimal:
            base = Decimal(1) + rate
            total = Decimal(0)
            for on, amount in checked:
                years = Decimal((on - origin).days) / _DAY_COUNT
                total += amount / (base**years)
            return total

        low = _LOWER_RATE
        low_value = npv(low)
        high = Decimal(1)
        high_value = npv(high)
        expansions = 0
        while low_value * high_value > 0 and expansions < 100:
            high = high * Decimal(2) + Decimal(1)
            high_value = npv(high)
            expansions += 1
        if low_value == 0:
            return low
        if high_value == 0:
            return high
        if low_value * high_value > 0:
            raise ValueError("cashflows do not have a real XIRR root above -100%")

        while high - low > _RATE_EPSILON:
            middle = (low + high) / Decimal(2)
            middle_value = npv(middle)
            if middle_value == 0:
                return middle
            if (middle_value > 0) == (low_value > 0):
                low = middle
                low_value = middle_value
            else:
                high = middle
        return (low + high) / Decimal(2)


def policy_irr(asset: TraditionalPolicyAsset) -> Decimal | None:
    """Return the XIRR implied by annual policy premiums and illustration."""
    if asset.maturity_illustration is None:
        return None
    premium_years = min(asset.premium_paying_term or asset.term_years, asset.term_years)
    start = date(asset.start_year, 1, 1)
    cashflows: list[tuple[date, Decimal]] = [
        (start + timedelta(days=365 * year), -asset.annual_premium) for year in range(premium_years)
    ]
    cashflows.append((start + timedelta(days=365 * asset.term_years), asset.maturity_illustration))
    return xirr(cashflows)
