"""Decimal-only valuation formulas for deposit-style assets."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

from wealth.core.dates import MonthYear, add_months
from wealth.core.money import format_inr, round_paise
from wealth.rates.reference import RateBook, load_rates
from wealth.schemas.assets import (
    AnyAsset,
    FixedDepositAsset,
    PostOfficeAsset,
    RecurringDepositAsset,
)
from wealth.schemas.common import Compounding, PayoutType, ValuationMethod
from wealth.valuation.base import Valuation
from wealth.valuation.prices import PriceProvider

_ONE_HUNDRED = Decimal(100)
_DAYS_PER_YEAR = Decimal(365)
_MONTHS_PER_YEAR = 12
_RD_TENURE_MONTHS = 60
_COMPOUNDING_MONTHS = {
    Compounding.MONTHLY: 1,
    Compounding.QUARTERLY: 3,
    Compounding.HALF_YEARLY: 6,
    Compounding.YEARLY: 12,
}
_PAYOUT_MONTHS = {
    PayoutType.MONTHLY: 1,
    PayoutType.QUARTERLY: 3,
    PayoutType.HALF_YEARLY: 6,
    PayoutType.YEARLY: 12,
}


def _result(
    value_today: Decimal,
    maturity_value: Decimal | None,
    maturity_date: date | None,
    principal_so_far: Decimal,
    notes: tuple[str, ...],
) -> Valuation:
    with localcontext() as context:
        context.prec = 40
        rounded_value = round_paise(value_today)
        rounded_principal = round_paise(principal_so_far)
        accrued_interest = round_paise(rounded_value - rounded_principal)
    return Valuation(
        value_today=rounded_value,
        maturity_value=None if maturity_value is None else round_paise(maturity_value),
        maturity_date=maturity_date,
        accrued_interest=accrued_interest,
        principal_so_far=rounded_principal,
        method=ValuationMethod.FORMULA,
        notes=notes,
    )


def _start_date(
    start: MonthYear | None, assumed_start: date | None
) -> tuple[date | None, tuple[str, ...]]:
    if start is not None:
        return start.first_day(), ()
    if assumed_start is not None:
        return assumed_start, (f"start month missing: assumed {assumed_start:%Y-%m}",)
    return None, ("add start month for exact value",)


def _rate_for(
    rate_pct: Decimal | None,
    rates: RateBook | None,
    scheme: str,
    on: date,
) -> tuple[Decimal, tuple[str, ...]]:
    if rate_pct is not None:
        return rate_pct, ()
    if rates is None:
        rates = load_rates()
    selected = rates.rate_for(scheme, on)
    notes = () if selected.verified else ("rate from reference table (unverified)",)
    return selected.rate_pct, notes


def _annual_rate(rate_pct: Decimal) -> Decimal:
    return rate_pct / _ONE_HUNDRED


def _whole_months_between(start: date, end: date) -> int:
    """Count full calendar months from ``start`` through ``end``."""
    months = (end.year - start.year) * _MONTHS_PER_YEAR + end.month - start.month
    if add_months(start, months) > end:
        months -= 1
    return max(months, 0)


def _months_between(start: date, end: date) -> Decimal:
    """Return elapsed calendar months, including a fraction of the current month."""
    if end <= start:
        return Decimal(0)
    whole_months = (end.year - start.year) * _MONTHS_PER_YEAR + end.month - start.month
    anniversary = add_months(start, whole_months)
    if anniversary > end:
        whole_months -= 1
        anniversary = add_months(start, whole_months)
    if anniversary == end:
        return Decimal(whole_months)
    partial_year = Decimal((end - anniversary).days) / _DAYS_PER_YEAR
    fraction = partial_year * _MONTHS_PER_YEAR
    return Decimal(whole_months) + fraction


def _compound_amount(
    principal: Decimal,
    rate_pct: Decimal,
    start: date,
    end: date,
    compounding: Compounding,
) -> Decimal:
    if end <= start:
        return principal
    annual_rate = _annual_rate(rate_pct)
    if compounding is Compounding.SIMPLE:
        return principal * (Decimal(1) + annual_rate * Decimal((end - start).days) / _DAYS_PER_YEAR)

    months_per_period = _COMPOUNDING_MONTHS[compounding]
    periods_per_year = Decimal(_MONTHS_PER_YEAR // months_per_period)
    periods = _whole_months_between(start, end) // months_per_period
    boundary = add_months(start, periods * months_per_period)
    compounded = principal * (Decimal(1) + annual_rate / periods_per_year) ** periods
    leftover_days = (end - boundary).days
    if leftover_days:
        compounded *= Decimal(1) + annual_rate * Decimal(leftover_days) / _DAYS_PER_YEAR
    return compounded


def _payout_value(
    principal: Decimal,
    rate_pct: Decimal,
    start: date,
    as_of: date,
    payout: PayoutType,
) -> Decimal:
    months_per_payout = _PAYOUT_MONTHS[payout]
    periods = _whole_months_between(start, as_of) // months_per_payout
    last_payout = add_months(start, periods * months_per_payout)
    days_since_payout = (as_of - last_payout).days
    interest = principal * _annual_rate(rate_pct) * Decimal(days_since_payout) / _DAYS_PER_YEAR
    return principal + interest


def value_fd(
    asset: FixedDepositAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
) -> Valuation:
    """Value a fixed deposit, including cumulative and periodic-payout FDs."""
    del rates  # Bank FD rates are supplied by the user's certificate.
    with localcontext() as context:
        context.prec = 40
        principal = asset.principal
        maturity_date = asset.maturity.first_day()
        start, notes = _start_date(asset.start, assumed_start)
        if start is None:
            return _result(principal, None, maturity_date, principal, notes)
        if maturity_date < start:
            raise ValueError("FD maturity must be on or after its start date")

        if asset.payout is PayoutType.CUMULATIVE:
            maturity_value = _compound_amount(
                principal, asset.rate_pct, start, maturity_date, asset.compounding
            )
            if as_of < start:
                value_today = principal
            elif as_of >= maturity_date:
                value_today = maturity_value
            else:
                value_today = _compound_amount(
                    principal, asset.rate_pct, start, as_of, asset.compounding
                )
        else:
            maturity_value = principal
            if as_of < start or as_of >= maturity_date:
                value_today = principal
            else:
                value_today = _payout_value(principal, asset.rate_pct, start, as_of, asset.payout)

        return _result(value_today, maturity_value, maturity_date, principal, notes)


def _rd_value_at(
    monthly_amount: Decimal,
    rate_pct: Decimal,
    start: date,
    value_date: date,
    installment_count: int,
) -> Decimal:
    if installment_count <= 0:
        return Decimal(0)
    quarterly_month_factor = Decimal(1) + _annual_rate(rate_pct) / Decimal(4)
    total = Decimal(0)
    for offset in range(installment_count):
        paid_at = add_months(start, offset)
        months_remaining = _months_between(paid_at, value_date)
        total += monthly_amount * quarterly_month_factor ** (months_remaining / Decimal(3))
    return total


def value_rd(
    asset: RecurringDepositAsset | PostOfficeAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
) -> Valuation:
    """Value bank or post-office monthly recurring-deposit contributions."""
    if isinstance(asset, PostOfficeAsset):
        if asset.scheme != "rd":
            raise ValueError("value_rd requires an RD asset")
        monthly_amount = asset.amount
        start = asset.purchase.first_day()
        maturity_date = add_months(start, _RD_TENURE_MONTHS)
        tenure_months = _RD_TENURE_MONTHS
        rate_scheme = "rd_5y"
        rate_pct, rate_notes = _rate_for(asset.rate_pct, rates, rate_scheme, start)
        notes = rate_notes
    else:
        monthly_amount = asset.monthly_amount
        start, start_notes = _start_date(asset.start, assumed_start)
        maturity_date = asset.maturity.first_day()
        tenure_months = _whole_months_between(start, maturity_date) if start is not None else 0
        rate_pct = asset.rate_pct
        rate_notes = ()
        notes = start_notes

    if start is None:
        return _result(Decimal(0), None, maturity_date, Decimal(0), notes)
    if maturity_date < start:
        raise ValueError("RD maturity must be on or after its start date")
    if tenure_months <= 0:
        raise ValueError("RD maturity must be at least one month after its start")

    with localcontext() as context:
        context.prec = 40
        notes = notes + rate_notes
        installment_count = min(tenure_months, _RD_TENURE_MONTHS)
        maturity_value = _rd_value_at(
            monthly_amount, rate_pct, start, maturity_date, installment_count
        )
        if as_of < start:
            paid_count = 0
            value_today = Decimal(0)
        else:
            valuation_date = min(as_of, maturity_date)
            paid_count = min(
                installment_count,
                _whole_months_between(start, valuation_date) + 1,
            )
            value_today = _rd_value_at(monthly_amount, rate_pct, start, valuation_date, paid_count)
            if as_of >= maturity_date:
                value_today = maturity_value
        principal_so_far = monthly_amount * paid_count
        return _result(
            value_today,
            maturity_value,
            maturity_date,
            principal_so_far,
            notes,
        )


def po_td_annual_interest(principal: Decimal, rate_pct: Decimal) -> Decimal:
    """Return one year's post-office TD interest after quarterly compounding."""
    if not isinstance(principal, Decimal) or not principal.is_finite() or principal < 0:
        raise ValueError("principal must be a non-negative finite Decimal")
    if not isinstance(rate_pct, Decimal) or not rate_pct.is_finite() or rate_pct < 0:
        raise ValueError("rate_pct must be a non-negative finite Decimal")
    with localcontext() as context:
        context.prec = 40
        quarterly_rate = _annual_rate(rate_pct) / Decimal(4)
        annual_amount = principal * ((Decimal(1) + quarterly_rate) ** 4 - Decimal(1))
        return round_paise(annual_amount)


def _latest_anniversary(start: date, on: date) -> date:
    months = _whole_months_between(start, on)
    years = months // 12
    anniversary = add_months(start, years * 12)
    while anniversary > on:
        years -= 1
        anniversary = add_months(start, years * 12)
    return anniversary


def value_po_td(
    asset: PostOfficeAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
) -> Valuation:
    """Value a post-office time deposit whose quarterly interest is paid yearly."""
    if asset.scheme != "td" or asset.td_years is None:
        raise ValueError("value_po_td requires a post-office TD with td_years")
    start = asset.purchase.first_day()
    rate_pct, rate_notes = _rate_for(asset.rate_pct, rates, f"td_{asset.td_years}y", start)
    maturity_date = add_months(start, asset.td_years * 12)
    principal = asset.amount
    annual_payout = po_td_annual_interest(principal, rate_pct)
    notes = rate_notes + (f"annual payout: {format_inr(annual_payout)}",)

    with localcontext() as context:
        context.prec = 40
        if as_of < start or as_of >= maturity_date:
            value_today = principal
        else:
            last_anniversary = _latest_anniversary(start, as_of)
            days_since_payout = (as_of - last_anniversary).days
            quarterly_rate = _annual_rate(rate_pct) / Decimal(4)
            accrued = principal * (
                (Decimal(1) + quarterly_rate)
                ** (Decimal(4) * Decimal(days_since_payout) / _DAYS_PER_YEAR)
                - Decimal(1)
            )
            value_today = principal + accrued
        return _result(value_today, principal, maturity_date, principal, notes)


def _annual_compounded_amount(
    principal: Decimal, rate_pct: Decimal, start: date, end: date
) -> Decimal:
    if end <= start:
        return principal
    complete_years = _whole_months_between(start, end) // 12
    last_anniversary = add_months(start, complete_years * 12)
    compounded = principal * (Decimal(1) + _annual_rate(rate_pct)) ** complete_years
    leftover_days = (end - last_anniversary).days
    if leftover_days:
        compounded *= Decimal(1) + _annual_rate(rate_pct) * Decimal(leftover_days) / _DAYS_PER_YEAR
    return compounded


def value_nsc(
    asset: PostOfficeAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
) -> Valuation:
    """Value an NSC using the rate applicable when it was purchased."""
    del assumed_start
    if asset.scheme != "nsc":
        raise ValueError("value_nsc requires an NSC asset")
    start = asset.purchase.first_day()
    rate_pct, rate_notes = _rate_for(asset.rate_pct, rates, "nsc", start)
    maturity_date = add_months(start, 60)
    principal = asset.amount

    with localcontext() as context:
        context.prec = 40
        maturity_value = _annual_compounded_amount(principal, rate_pct, start, maturity_date)
        value_date = min(as_of, maturity_date)
        value_today = (
            principal
            if value_date <= start
            else _annual_compounded_amount(principal, rate_pct, start, value_date)
        )
        return _result(
            value_today,
            maturity_value,
            maturity_date,
            principal,
            rate_notes + ("rate locked at purchase",),
        )


def value_kvp(
    asset: PostOfficeAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
) -> Valuation:
    """Value a KVP by its purchase-date doubling term in the rate reference."""
    del assumed_start
    if asset.scheme != "kvp":
        raise ValueError("value_kvp requires a KVP asset")
    start = asset.purchase.first_day()
    if rates is None:
        rates = load_rates()
    selected = rates.rate_for("kvp", start)
    if selected.maturity_months is None:
        raise ValueError("KVP reference periods must include maturity_months")
    maturity_date = add_months(start, selected.maturity_months)
    principal = asset.amount
    notes = () if selected.verified else ("rate from reference table (unverified)",)
    notes += ("rate locked at purchase",)

    with localcontext() as context:
        context.prec = 40
        maturity_value = principal * Decimal(2)
        if as_of <= start:
            value_today = principal
        elif as_of >= maturity_date:
            value_today = maturity_value
        else:
            months_elapsed = _months_between(start, as_of)
            value_today = principal * Decimal(2) ** (
                months_elapsed / Decimal(selected.maturity_months)
            )
        return _result(value_today, maturity_value, maturity_date, principal, notes)


def value_asset(
    asset: AnyAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
    prices: PriceProvider | None = None,
) -> Valuation:
    """Dispatch asset models to their deterministic valuation engine."""
    if isinstance(asset, FixedDepositAsset):
        return value_fd(asset, as_of, rates=rates, assumed_start=assumed_start)
    if isinstance(asset, RecurringDepositAsset):
        return value_rd(asset, as_of, rates=rates, assumed_start=assumed_start)
    if isinstance(asset, PostOfficeAsset):
        if asset.scheme == "nsc":
            return value_nsc(asset, as_of, rates=rates, assumed_start=assumed_start)
        if asset.scheme == "kvp":
            return value_kvp(asset, as_of, rates=rates, assumed_start=assumed_start)
        if asset.scheme == "td":
            return value_po_td(asset, as_of, rates=rates, assumed_start=assumed_start)
        if asset.scheme == "rd":
            return value_rd(asset, as_of, rates=rates, assumed_start=assumed_start)

    from wealth.valuation.assets import value_other_asset

    return value_other_asset(
        asset,
        as_of,
        rates=rates,
        assumed_start=assumed_start,
        prices=prices,
    )
