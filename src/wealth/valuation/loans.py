"""Decimal loan payment and amortisation calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext

from wealth.core.dates import MonthYear
from wealth.core.money import round_paise


@dataclass(frozen=True)
class LoanSchedule:
    months_to_payoff: int
    total_interest: Decimal
    payoff: MonthYear


def _non_negative_decimal(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a non-negative finite Decimal")
    return value


def emi(principal: Decimal, rate_pct: Decimal, months: int) -> Decimal:
    """Return the fixed monthly payment for a fully amortising loan."""
    principal = _non_negative_decimal(principal, "principal")
    rate_pct = _non_negative_decimal(rate_pct, "rate_pct")
    if not isinstance(months, int) or isinstance(months, bool) or months <= 0:
        raise ValueError("months must be a positive integer")
    with localcontext() as context:
        context.prec = 40
        monthly_rate = rate_pct / Decimal(1200)
        if monthly_rate == 0:
            payment = principal / months
        else:
            factor = (Decimal(1) + monthly_rate) ** months
            payment = principal * monthly_rate * factor / (factor - Decimal(1))
        return round_paise(payment)


def amortise(
    outstanding: Decimal,
    rate_pct: Decimal,
    emi: Decimal,
    *,
    start: MonthYear | None = None,
) -> LoanSchedule:
    """Amortise monthly payments and return the final payment month.

    ``start`` is the balance's current month; when omitted, the local current
    calendar month is used and the first payment is one month later.
    """
    outstanding = _non_negative_decimal(outstanding, "outstanding")
    rate_pct = _non_negative_decimal(rate_pct, "rate_pct")
    emi = _non_negative_decimal(emi, "emi")
    start_month = start or MonthYear(date.today().year, date.today().month)
    if not isinstance(start_month, MonthYear):
        raise ValueError("start must be a MonthYear")
    if outstanding == 0:
        return LoanSchedule(0, Decimal("0.00"), start_month)

    with localcontext() as context:
        context.prec = 40
        monthly_rate = rate_pct / Decimal(1200)
        balance = outstanding
        total_interest = Decimal(0)
        months = 0
        while balance > 0:
            interest = balance * monthly_rate
            if emi <= interest:
                raise ValueError("EMI must be greater than the monthly interest")
            payment = min(emi, balance + interest)
            total_interest += interest
            balance += interest - payment
            months += 1
        return LoanSchedule(
            months_to_payoff=months,
            total_interest=round_paise(total_interest),
            payoff=start_month.add_months(months),
        )
