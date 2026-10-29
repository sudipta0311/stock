"""Decimal loan payment and amortisation calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Literal

from wealth.core.dates import MonthYear
from wealth.core.money import round_paise
from wealth.schemas.assets import LoanAsset

LoanEventKind = Literal[
    "prepayment", "rate_change", "emi_change", "balance_update", "missed_emi", "closed"
]


@dataclass(frozen=True)
class LoanSchedule:
    months_to_payoff: int
    total_interest: Decimal
    payoff: MonthYear


@dataclass(frozen=True)
class LoanEvent:
    """A recorded loan change used by the deterministic schedule engine."""

    kind: LoanEventKind
    on_date: date
    amount: Decimal | None = None
    rate_pct: Decimal | None = None
    emi: Decimal | None = None
    note: str | None = None


@dataclass(frozen=True)
class LoanBalancePoint:
    month: MonthYear
    outstanding: Decimal


@dataclass(frozen=True)
class LoanPayment:
    on_date: date
    amount: Decimal
    principal: Decimal
    interest: Decimal
    outstanding: Decimal


@dataclass(frozen=True)
class LoanPosition:
    outstanding: Decimal
    principal_paid: Decimal
    interest_paid: Decimal
    months_left: int | None
    payoff_month: MonthYear | None
    interest_to_pay: Decimal | None
    next_emi_date: date | None
    percent_repaid: Decimal | None
    yearly_balances: tuple[LoanBalancePoint, ...]
    scheduled_payments: tuple[LoanPayment, ...]
    never_pays_off: bool


@dataclass(frozen=True)
class PrepayOptions:
    keep_emi_months_saved: int
    keep_emi_interest_saved: Decimal
    keep_end_date_new_emi: Decimal
    keep_end_date_interest_saved: Decimal


@dataclass(frozen=True)
class _MonthResult:
    balance: Decimal
    rate_pct: Decimal
    emi: Decimal | None
    principal_paid: Decimal
    interest_paid: Decimal
    total_paid: Decimal
    interest_accrued: Decimal
    closed: bool
    insufficient_emi: bool


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


def loan_position(loan: LoanAsset, events: Sequence[LoanEvent], as_of: date) -> LoanPosition:
    """Roll a loan from its last saved or statement balance using monthly Decimal math."""
    if not isinstance(loan, LoanAsset):
        raise ValueError("loan must be a LoanAsset")
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    normalized = tuple(sorted(events, key=lambda item: (item.on_date, item.kind)))
    if any(not isinstance(item, LoanEvent) for item in normalized):
        raise ValueError("events must contain LoanEvent values")

    anchor_updates = [
        event for event in normalized if event.kind == "balance_update" and event.on_date <= as_of
    ]
    anchor = max(anchor_updates, key=lambda item: item.on_date) if anchor_updates else None
    anchor_month = (
        MonthYear(anchor.on_date.year, anchor.on_date.month) if anchor else loan.outstanding_as_of
    )
    anchor_date = anchor.on_date if anchor else anchor_month.first_day()
    balance = _non_negative_decimal(
        anchor.amount if anchor is not None and anchor.amount is not None else loan.outstanding,
        "outstanding",
    )
    anchor_balance = balance
    rate = loan.rate_pct
    monthly_emi = loan.emi
    principal_paid = Decimal(0)
    interest_paid = Decimal(0)

    anchor_month_events = [
        event
        for event in normalized
        if event.on_date > anchor_date
        and event.on_date.year == anchor_month.year
        and event.on_date.month == anchor_month.month
    ]
    rate, monthly_emi = _apply_terms(anchor_month_events, rate, monthly_emi)
    for event in anchor_month_events:
        if event.kind == "prepayment":
            amount = _non_negative_decimal(event.amount or Decimal(0), "prepayment")
            applied = min(balance, amount)
            balance -= applied
            principal_paid += applied

    closed_on = next(
        (
            event
            for event in normalized
            if event.kind == "closed" and anchor_date < event.on_date <= as_of
        ),
        None,
    )
    if closed_on is not None:
        balance = Decimal(0)

    def events_in_month(month: MonthYear) -> list[LoanEvent]:
        return [
            event
            for event in normalized
            if (event.on_date.year, event.on_date.month) == (month.year, month.month)
            and event.on_date > anchor_date
        ]

    # The saved balance already includes its anchor month. Only later months roll forward.
    current_month = MonthYear(as_of.year, as_of.month)
    months_from_anchor = anchor_month.months_until(current_month)
    historical_points = [LoanBalancePoint(anchor_month, round_paise(anchor_balance))]
    for offset in range(1, max(0, months_from_anchor) + 1):
        month = anchor_month.add_months(offset)
        if closed_on is not None:
            break
        result = _roll_month(balance, rate, monthly_emi, events_in_month(month))
        balance, rate, monthly_emi = result.balance, result.rate_pct, result.emi
        principal_paid += result.principal_paid
        interest_paid += result.interest_paid
        if offset % 12 == 0 or month == current_month:
            historical_points.append(LoanBalancePoint(month, round_paise(balance)))
        if result.closed:
            closed_on = next(
                (event for event in events_in_month(month) if event.kind == "closed"),
                LoanEvent("closed", month.last_day()),
            )
            balance = Decimal(0)
            break

    current_balance = max(Decimal(0), balance)
    if current_balance == 0:
        return LoanPosition(
            outstanding=round_paise(current_balance),
            principal_paid=round_paise(principal_paid),
            interest_paid=round_paise(interest_paid),
            months_left=0,
            payoff_month=MonthYear(closed_on.on_date.year, closed_on.on_date.month)
            if closed_on is not None
            else current_month,
            interest_to_pay=Decimal("0.00"),
            next_emi_date=None,
            percent_repaid=_percent_repaid(loan, current_balance),
            yearly_balances=(LoanBalancePoint(current_month, Decimal("0.00")),),
            scheduled_payments=(),
            never_pays_off=False,
        )

    projected_balance = current_balance
    projected_rate = rate
    projected_emi = monthly_emi
    months_left: int | None = None
    payoff_month: MonthYear | None = None
    interest_to_pay: Decimal | None = None
    never_pays_off = False
    schedule: list[LoanPayment] = []
    balance_points = [*historical_points]
    if balance_points[-1].month != current_month:
        balance_points.append(LoanBalancePoint(current_month, round_paise(current_balance)))
    projected_interest = Decimal(0)
    future_events = [event for event in normalized if event.on_date > as_of]
    for offset in range(1, 1201):
        month = current_month.add_months(offset)
        month_events = [
            event
            for event in future_events
            if (event.on_date.year, event.on_date.month) == (month.year, month.month)
        ]
        if not projected_emi or projected_emi <= 0:
            never_pays_off = True
            break
        result = _roll_month(projected_balance, projected_rate, projected_emi, month_events)
        projected_balance, projected_rate, projected_emi = (
            result.balance,
            result.rate_pct,
            result.emi,
        )
        projected_interest += result.interest_accrued
        if result.closed:
            months_left = offset
            payoff_month = month
            break
        if result.insufficient_emi:
            has_later_term_change = any(
                event.kind in {"rate_change", "emi_change"}
                and (event.on_date.year, event.on_date.month) > (month.year, month.month)
                for event in future_events
            )
            if not has_later_term_change:
                never_pays_off = True
                break
        payment_date = date(month.year, month.month, min(loan.emi_day or 1, 28))
        if len(schedule) < 12:
            schedule.append(
                LoanPayment(
                    on_date=payment_date,
                    amount=round_paise(result.total_paid),
                    principal=round_paise(result.principal_paid),
                    interest=round_paise(result.interest_paid),
                    outstanding=round_paise(projected_balance),
                )
            )
        if offset % 12 == 0 or result.closed:
            balance_points.append(LoanBalancePoint(month, round_paise(projected_balance)))
        if projected_balance <= 0:
            months_left = offset
            payoff_month = month
            break
    if months_left is not None:
        interest_to_pay = round_paise(projected_interest)
    elif not never_pays_off:
        never_pays_off = True
    next_emi_date = _next_emi_date(as_of, loan.emi_day) if loan.emi_day is not None else None
    percent_repaid = _percent_repaid(loan, current_balance)
    if balance_points[-1].month != current_month:
        balance_points.append(LoanBalancePoint(current_month, round_paise(current_balance)))
        balance_points.sort(key=lambda point: point.month)
    return LoanPosition(
        outstanding=round_paise(current_balance),
        principal_paid=round_paise(principal_paid),
        interest_paid=round_paise(interest_paid),
        months_left=months_left,
        payoff_month=payoff_month,
        interest_to_pay=interest_to_pay,
        next_emi_date=next_emi_date,
        percent_repaid=percent_repaid,
        yearly_balances=tuple(_unique_balance_points(balance_points)),
        scheduled_payments=tuple(schedule),
        never_pays_off=never_pays_off,
    )


def prepay_options(
    loan: LoanAsset, events: Sequence[LoanEvent], amount: Decimal, as_of: date
) -> PrepayOptions:
    """Compare an extra payment with a shorter term or a lower monthly payment."""
    amount = _non_negative_decimal(amount, "amount")
    baseline = loan_position(loan, events, as_of)
    if baseline.never_pays_off or baseline.months_left is None:
        raise ValueError("prepayment options require a loan that pays off")
    extra = min(round_paise(amount), baseline.outstanding)
    remaining = round_paise(baseline.outstanding - extra)
    anchor = LoanEvent("balance_update", as_of, amount=remaining)
    option_a = loan_position(loan, [*events, anchor], as_of)
    months_saved = max(0, baseline.months_left - (option_a.months_left or 0))
    interest_saved_a = max(
        Decimal(0),
        round_paise(
            (baseline.interest_to_pay or Decimal(0)) - (option_a.interest_to_pay or Decimal(0))
        ),
    )
    if remaining == 0:
        new_emi = Decimal("0.00")
        option_b_interest = Decimal("0.00")
    else:
        current_rate = _effective_rate(loan, events, as_of)
        new_emi = emi(remaining, current_rate, baseline.months_left)
        next_month = MonthYear(as_of.year, as_of.month).add_months(1)
        new_payment = LoanEvent("emi_change", next_month.first_day(), emi=new_emi)
        option_b = loan_position(loan, [*events, anchor, new_payment], as_of)
        option_b_interest = option_b.interest_to_pay or Decimal(0)
    interest_saved_b = max(
        Decimal(0), round_paise((baseline.interest_to_pay or Decimal(0)) - option_b_interest)
    )
    return PrepayOptions(
        keep_emi_months_saved=months_saved,
        keep_emi_interest_saved=interest_saved_a,
        keep_end_date_new_emi=round_paise(new_emi),
        keep_end_date_interest_saved=interest_saved_b,
    )


def _apply_terms(
    events: Sequence[LoanEvent], rate_pct: Decimal, monthly_emi: Decimal | None
) -> tuple[Decimal, Decimal | None]:
    for event in events:
        if event.kind == "rate_change" and event.rate_pct is not None:
            rate_pct = _non_negative_decimal(event.rate_pct, "rate_pct")
        elif event.kind == "emi_change" and event.emi is not None:
            monthly_emi = _non_negative_decimal(event.emi, "emi")
    return rate_pct, monthly_emi


def _roll_month(
    opening_balance: Decimal,
    rate_pct: Decimal,
    monthly_emi: Decimal | None,
    events: Sequence[LoanEvent],
) -> _MonthResult:
    rate_pct, monthly_emi = _apply_terms(events, rate_pct, monthly_emi)
    balance = opening_balance
    if balance <= 0:
        return _MonthResult(
            balance,
            rate_pct,
            monthly_emi,
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            True,
            False,
        )
    interest = balance * rate_pct / Decimal(1200)
    missed = any(event.kind == "missed_emi" for event in events)
    paid_interest = Decimal(0)
    paid_principal = Decimal(0)
    total_paid = Decimal(0)
    insufficient = False
    if missed:
        balance += interest
    elif monthly_emi is not None:
        payment = _non_negative_decimal(monthly_emi, "emi")
        if payment <= interest:
            balance += interest - payment
            paid_interest = payment
            total_paid = payment
            insufficient = True
        else:
            total_paid = min(payment, balance + interest)
            paid_interest = min(interest, total_paid)
            paid_principal = max(Decimal(0), total_paid - interest)
            balance = max(Decimal(0), balance - paid_principal)
    else:
        balance += interest

    for event in events:
        if event.kind == "prepayment":
            extra = _non_negative_decimal(event.amount or Decimal(0), "prepayment")
            applied = min(balance, extra)
            balance -= applied
            paid_principal += applied
            total_paid += applied
        elif event.kind == "closed":
            balance = Decimal(0)
    return _MonthResult(
        balance,
        rate_pct,
        monthly_emi,
        paid_principal,
        paid_interest,
        total_paid,
        interest,
        balance <= 0,
        insufficient,
    )


def _effective_rate(loan: LoanAsset, events: Sequence[LoanEvent], as_of: date) -> Decimal:
    rate = loan.rate_pct
    for event in sorted(events, key=lambda item: item.on_date):
        if event.kind == "rate_change" and event.on_date <= as_of and event.rate_pct is not None:
            rate = event.rate_pct
    return rate


def _next_emi_date(as_of: date, emi_day: int) -> date:
    month = MonthYear(as_of.year, as_of.month).add_months(1)
    return date(month.year, month.month, min(emi_day, 28))


def _percent_repaid(loan: LoanAsset, balance: Decimal) -> Decimal | None:
    if loan.original_amount is None or loan.original_amount <= 0:
        return None
    percent = (loan.original_amount - balance) * Decimal(100) / loan.original_amount
    return max(Decimal(0), min(Decimal(100), percent)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _unique_balance_points(points: Sequence[LoanBalancePoint]) -> list[LoanBalancePoint]:
    by_month = {point.month: point for point in points}
    return [by_month[month] for month in sorted(by_month)]
