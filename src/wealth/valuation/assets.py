"""Deterministic valuations for long-term, market-priced, and user-valued assets."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

from wealth.core.dates import MonthYear, add_months, financial_year, fy_end, fy_start
from wealth.core.money import format_inr, round_paise
from wealth.rates.reference import RateBook, RateUnavailable, load_rates
from wealth.schemas.assets import (
    AnyAsset,
    BondAsset,
    EPFAsset,
    GoldAsset,
    HealthInsuranceAsset,
    NPSAsset,
    PostOfficeAsset,
    PPFAsset,
    PropertyAsset,
    SavingsAsset,
    TermInsuranceAsset,
    TraditionalPolicyAsset,
    ULIPAsset,
)
from wealth.schemas.common import ValuationMethod
from wealth.valuation.base import Valuation
from wealth.valuation.prices import PriceProvider, PriceQuote, PriceUnavailable, quote_value

_DAYS_PER_YEAR = Decimal(365)
_ONE_HUNDRED = Decimal(100)


def _result(
    value_today: Decimal,
    maturity_value: Decimal | None,
    maturity_date: date | None,
    principal_so_far: Decimal,
    notes: tuple[str, ...] = (),
    method: ValuationMethod = ValuationMethod.FORMULA,
) -> Valuation:
    with localcontext() as context:
        context.prec = 40
        rounded_value = round_paise(value_today)
        rounded_principal = round_paise(principal_so_far)
        accrued_interest = round_paise(rounded_value - rounded_principal)
        rounded_maturity = None if maturity_value is None else round_paise(maturity_value)
    return Valuation(
        value_today=rounded_value,
        maturity_value=rounded_maturity,
        maturity_date=maturity_date,
        accrued_interest=accrued_interest,
        principal_so_far=rounded_principal,
        method=method,
        notes=notes,
    )


def _annual_rate(rate_pct: Decimal) -> Decimal:
    return rate_pct / _ONE_HUNDRED


def _simple_interest(principal: Decimal, rate_pct: Decimal, start: date, end: date) -> Decimal:
    if end <= start:
        return Decimal(0)
    return principal * _annual_rate(rate_pct) * Decimal((end - start).days) / _DAYS_PER_YEAR


def _selected_rate(
    rates: RateBook | None, scheme: str, on: date, *, use_latest: bool = False
) -> tuple[Decimal, tuple[str, ...]]:
    book = rates if rates is not None else load_rates()
    try:
        selected = book.rate_for(scheme, on)
        notes = () if selected.verified else ("rate from reference table (unverified)",)
        return selected.rate_pct, notes
    except RateUnavailable:
        if not use_latest:
            raise
        reference = book.schemes.get(scheme)
        earlier = (
            [period for period in reference.periods if period.effective_from <= on]
            if reference is not None
            else []
        )
        if not earlier:
            raise
        period = max(earlier, key=lambda item: item.effective_from)
        note = (
            f"using latest available rate from {period.effective_from} to "
            f"{period.effective_to or 'open-ended'}"
        )
        notes = (note,)
        if not period.verified:
            notes += ("rate from reference table (unverified)",)
        return period.rate_pct, notes


def _balance_start(
    balance_as_of: MonthYear | None, assumed_start: date | None
) -> tuple[date | None, tuple[str, ...]]:
    if balance_as_of is not None:
        return balance_as_of.first_day(), ()
    if assumed_start is not None:
        return assumed_start, (f"balance month missing: assumed {assumed_start:%Y-%m}",)
    return None, ("add balance-as-of month for exact value",)


def _years_until_next_april(as_of: date, maturity_date: date) -> int:
    if as_of >= maturity_date:
        return 0
    this_april = date(as_of.year, 4, 1)
    next_contribution = (
        this_april
        if as_of == this_april
        else date(as_of.year + (1 if as_of >= this_april else 0), 4, 1)
    )
    if next_contribution < as_of:
        next_contribution = date(as_of.year + 1, 4, 1)
    return max(0, maturity_date.year - next_contribution.year)


def _next_april(as_of: date) -> date:
    this_april = date(as_of.year, 4, 1)
    return this_april if as_of <= this_april else date(as_of.year + 1, 4, 1)


def _balance_at_projection_april(
    balance: Decimal,
    rate_pct: Decimal,
    balance_start: date | None,
    as_of: date,
    projection_april: date,
) -> Decimal:
    accrual_start = balance_start if balance_start is not None else as_of
    return balance + _simple_interest(balance, rate_pct, accrual_start, projection_april)


def project_ppf(
    balance: Decimal,
    yearly_contribution: Decimal,
    years: int,
    rate_pct: Decimal,
) -> Decimal:
    """Project PPF-like annual deposits, made on 1 April, at one annual rate."""
    if not isinstance(years, int) or isinstance(years, bool) or years < 0:
        raise ValueError("years must be a non-negative integer")
    if any(
        not isinstance(value, Decimal) or not value.is_finite() or value < 0
        for value in (balance, yearly_contribution, rate_pct)
    ):
        raise ValueError("balance, contribution, and rate must be non-negative Decimals")
    with localcontext() as context:
        context.prec = 40
        amount = balance
        annual_rate = _annual_rate(rate_pct)
        for _ in range(years):
            amount = (amount + yearly_contribution) * (Decimal(1) + annual_rate)
        return round_paise(amount)


def value_ppf(
    asset: PPFAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
) -> Valuation:
    maturity_date = date(asset.opened_fy_start_year + 16, 4, 1)
    balance_start, notes = _balance_start(asset.balance_as_of, assumed_start)
    rate_pct, rate_notes = _selected_rate(rates, "ppf", as_of, use_latest=True)
    with localcontext() as context:
        context.prec = 40
        value_today = asset.balance
        if balance_start is not None and as_of > balance_start:
            value_today += _simple_interest(asset.balance, rate_pct, balance_start, as_of)
        projection_april = _next_april(as_of)
        years = _years_until_next_april(as_of, maturity_date)
        projection_balance = _balance_at_projection_april(
            asset.balance, rate_pct, balance_start, as_of, projection_april
        )
        maturity_value = project_ppf(
            projection_balance,
            asset.yearly_contribution or Decimal(0),
            years,
            rate_pct,
        )
        return _result(
            value_today,
            maturity_value,
            maturity_date,
            asset.balance,
            notes + rate_notes + ("accrued, credited 31 Mar", "projection at current rate"),
        )


def value_epf(
    asset: EPFAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
) -> Valuation:
    balance_start, notes = _balance_start(asset.balance_as_of, assumed_start)
    if balance_start is None or as_of <= balance_start:
        return _result(
            asset.balance,
            None,
            None,
            asset.balance,
            notes,
        )

    book = rates if rates is not None else load_rates()
    monthly_contribution = sum(
        (
            asset.monthly_employee or Decimal(0),
            asset.monthly_employer or Decimal(0),
            asset.vpf_monthly or Decimal(0),
        ),
        Decimal(0),
    )
    with localcontext() as context:
        context.prec = 40
        principal = asset.balance
        accrued = Decimal(0)
        cursor = fy_start(balance_start)
        while cursor <= as_of:
            end_of_fy = fy_end(cursor)
            segment_start = max(balance_start, cursor)
            segment_end = min(as_of, end_of_fy)
            fy = financial_year(cursor)
            try:
                selected = book.epf_rate(fy)
                source_fy = fy
            except RateUnavailable:
                needed_year = int(fy[:4])
                earlier = [
                    period for period in book.epf.periods if int(period.fy[:4]) < needed_year
                ]
                if not earlier:
                    raise
                selected = max(earlier, key=lambda period: int(period.fy[:4]))
                source_fy = selected.fy
                notes += (f"using latest available rate from FY {source_fy}",)
            annual_rate = _annual_rate(selected.rate_pct)
            if not selected.verified:
                notes += ("EPF rate from reference table (unverified)",)

            contribution_dates: list[date] = []
            month = add_months(balance_start.replace(day=1), 1)
            while month < segment_start:
                month = add_months(month, 1)
            while month <= segment_end:
                contribution_dates.append(month)
                month = add_months(month, 1)
            contributions = monthly_contribution * len(contribution_dates)
            accrued += (
                principal
                * annual_rate
                * Decimal((segment_end - segment_start).days)
                / _DAYS_PER_YEAR
            )
            for contribution_date in contribution_dates:
                accrued += (
                    monthly_contribution
                    * annual_rate
                    * Decimal((segment_end - contribution_date).days)
                    / _DAYS_PER_YEAR
                )
            principal += contributions
            if segment_end == end_of_fy:
                principal += accrued
                accrued = Decimal(0)
            cursor = date(cursor.year + 1, 4, 1)
        return _result(
            principal + accrued,
            None,
            None,
            asset.balance + monthly_contribution * _months_after(balance_start, as_of),
            notes,
        )


def _months_after(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + end.month - start.month
    if end.day < start.day:
        months -= 1
    return max(0, months)


def _last_quarter_start(on: date) -> date:
    month = ((on.month - 1) // 3) * 3 + 1
    return date(on.year, month, 1)


def _last_monthly_payout(start: date, on: date) -> date:
    months = (on.year - start.year) * 12 + on.month - start.month
    payout = add_months(start, months)
    if payout > on:
        payout = add_months(payout, -1)
    return payout


def value_scss(asset: PostOfficeAsset, as_of: date, *, rates: RateBook | None = None) -> Valuation:
    start = asset.purchase.first_day()
    maturity_date = add_months(start, 60)
    rate_pct = asset.rate_pct
    notes: tuple[str, ...] = ("interest paid quarterly",)
    if rate_pct is None:
        rate_pct, rate_notes = _selected_rate(rates, "scss", start)
        notes += rate_notes + ("rate locked at purchase",)
    if as_of <= start or as_of >= maturity_date:
        days_since_payout = 0
    else:
        days_since_payout = (as_of - max(_last_quarter_start(as_of), start)).days
    with localcontext() as context:
        context.prec = 40
        accrued = (
            asset.amount * _annual_rate(rate_pct) * Decimal(days_since_payout) / _DAYS_PER_YEAR
        )
        return _result(
            asset.amount + accrued,
            asset.amount,
            maturity_date,
            asset.amount,
            notes,
        )


def value_mis(asset: PostOfficeAsset, as_of: date, *, rates: RateBook | None = None) -> Valuation:
    start = asset.purchase.first_day()
    maturity_date = add_months(start, 60)
    rate_pct = asset.rate_pct
    notes: tuple[str, ...] = ("interest paid monthly",)
    if rate_pct is None:
        rate_pct, rate_notes = _selected_rate(rates, "mis", start)
        notes += rate_notes + ("rate locked at purchase",)
    if as_of <= start or as_of >= maturity_date:
        last_payout = start
    else:
        last_payout = max(_last_monthly_payout(start, as_of), start)
    days_since_payout = max(0, (as_of - last_payout).days)
    if as_of >= maturity_date:
        days_since_payout = 0
    with localcontext() as context:
        context.prec = 40
        accrued = (
            asset.amount * _annual_rate(rate_pct) * Decimal(days_since_payout) / _DAYS_PER_YEAR
        )
        return _result(
            asset.amount + accrued,
            asset.amount,
            maturity_date,
            asset.amount,
            notes,
        )


def _validate_price(value: Decimal | PriceQuote, description: str) -> Decimal:
    price, _notes = quote_value(value)
    if not isinstance(price, Decimal) or not price.is_finite() or price < 0:
        raise ValueError(f"{description} must be a non-negative finite Decimal")
    return price


def _validated_quote(
    value: Decimal | PriceQuote, description: str
) -> tuple[Decimal, tuple[str, ...]]:
    price, notes = quote_value(value)
    if not isinstance(price, Decimal) or not price.is_finite() or price < 0:
        raise ValueError(f"{description} must be a non-negative finite Decimal")
    return price, notes


def value_gold(asset: GoldAsset, as_of: date, *, prices: PriceProvider | None = None) -> Valuation:
    if prices is None:
        raise PriceUnavailable("No price provider was supplied")
    maturity_date: date | None = None
    notes: tuple[str, ...] = ()
    with localcontext() as context:
        context.prec = 40
        if asset.kind in {"physical", "digital", "sgb"}:
            price, price_notes = _validated_quote(prices.gold_24k_per_gram(as_of), "gold price")
            notes += price_notes
            if asset.kind == "physical":
                if asset.grams is None or asset.purity is None:
                    raise ValueError("physical gold requires grams and purity")
                value = asset.grams * Decimal(asset.purity) / Decimal(24) * price
            else:
                if asset.kind == "digital":
                    if asset.grams is None:
                        raise ValueError("digital gold requires grams")
                    value = asset.grams * price
                    notes += ("digital gold is not regulated by SEBI",)
                else:
                    if asset.units is None:
                        raise ValueError(f"{asset.kind} gold requires units")
                    value = asset.units * price
                    if asset.kind == "sgb" and asset.issue is not None:
                        maturity_date = add_months(asset.issue.first_day(), 96)
                    if asset.kind == "sgb" and asset.issue_price_per_gram is not None:
                        coupon = asset.units * asset.issue_price_per_gram * Decimal("0.0125")
                        notes += (f"half-yearly SGB coupon: {format_inr(coupon)}",)
            return _result(value, None, maturity_date, value, notes)

        if asset.kind in {"etf", "mf"}:
            if asset.units is None:
                raise ValueError(f"{asset.kind} gold requires units")
            key = asset.series or asset.label
            if not key:
                raise PriceUnavailable("Gold ETF or MF needs a series or label for NAV lookup")
            nav, price_notes = _validated_quote(prices.nav(key, as_of), "NAV")
            notes += price_notes
            value = asset.units * nav
            return _result(value, None, None, value, notes, ValuationMethod.MARKET)
    raise ValueError(f"Unsupported gold kind: {asset.kind}")


def _last_coupon_date(as_of: date, maturity: date, months_per_coupon: int) -> date:
    months_to_maturity = (maturity.year - as_of.year) * 12 + maturity.month - as_of.month
    coupon_steps = max(0, (months_to_maturity + months_per_coupon - 1) // months_per_coupon)
    return add_months(maturity, -coupon_steps * months_per_coupon)


def value_bond(asset: BondAsset, as_of: date) -> Valuation:
    maturity_date = asset.maturity.first_day()
    if asset.kind == "tbill":
        return _result(
            asset.face_value,
            asset.face_value,
            maturity_date,
            asset.face_value,
            ("discount instrument; shown at face value",),
        )
    notes = ("approximate; listed bonds should use market price",)
    if asset.coupon_pct is None or as_of >= maturity_date:
        return _result(asset.face_value, asset.face_value, maturity_date, asset.face_value, notes)
    frequency = asset.coupon_frequency or (2 if asset.kind in {"gsec", "rbi_frsb"} else 1)
    months_per_coupon = 12 // frequency
    last_coupon = _last_coupon_date(as_of, maturity_date, months_per_coupon)
    accrued = _simple_interest(asset.face_value, asset.coupon_pct, last_coupon, as_of)
    return _result(
        asset.face_value + accrued,
        asset.face_value,
        maturity_date,
        asset.face_value,
        notes,
    )


def value_traditional_policy(asset: TraditionalPolicyAsset, as_of: date) -> Valuation:
    premium_term = min(asset.premium_paying_term or asset.term_years, asset.term_years)
    elapsed_years = max(0, as_of.year - asset.start_year)
    paid_years = min(elapsed_years, premium_term)
    premiums_paid = asset.annual_premium * paid_years
    maturity_date = date(asset.start_year + asset.term_years, 4, 1)
    if as_of >= maturity_date:
        if asset.maturity_illustration is not None:
            value = asset.maturity_illustration
            method = ValuationMethod.USER
            notes: tuple[str, ...] = ("matured: confirm the amount received and close this policy",)
        else:
            value = premiums_paid
            method = ValuationMethod.FORMULA
            notes = (
                "estimate: premiums paid; add surrender value for accuracy",
                "matured: confirm the amount received and close this policy",
            )
    elif asset.surrender_value is not None:
        value = asset.surrender_value
        method = ValuationMethod.USER
        notes: tuple[str, ...] = ()
    else:
        value = premiums_paid
        method = ValuationMethod.FORMULA
        notes = ("estimate: premiums paid; add surrender value for accuracy",)
    return _result(
        value,
        asset.maturity_illustration,
        maturity_date,
        premiums_paid,
        notes,
        method,
    )


def value_other_asset(
    asset: AnyAsset,
    as_of: date,
    *,
    rates: RateBook | None = None,
    assumed_start: date | None = None,
    prices: PriceProvider | None = None,
) -> Valuation:
    if isinstance(asset, PPFAsset):
        return value_ppf(asset, as_of, rates=rates, assumed_start=assumed_start)
    if isinstance(asset, EPFAsset):
        return value_epf(asset, as_of, rates=rates, assumed_start=assumed_start)
    if isinstance(asset, PostOfficeAsset):
        start = asset.purchase.first_day()
        if asset.scheme == "ssy":
            maturity_date = add_months(start, 252)
            balance_start, notes = _balance_start(asset.balance_as_of, assumed_start)
            rate_pct, rate_notes = _selected_rate(rates, "ssy", as_of, use_latest=True)
            value_today = asset.amount
            if balance_start is not None and as_of > balance_start:
                value_today += _simple_interest(asset.amount, rate_pct, balance_start, as_of)
            projection_april = _next_april(as_of)
            years = _years_until_next_april(as_of, maturity_date)
            projection_balance = _balance_at_projection_april(
                asset.amount, rate_pct, balance_start, as_of, projection_april
            )
            maturity_value = project_ppf(projection_balance, Decimal(0), years, rate_pct)
            return _result(
                value_today,
                maturity_value,
                maturity_date,
                asset.amount,
                notes
                + rate_notes
                + ("SSY deposits allowed for 15 years from opening", "projection at current rate"),
            )
        if asset.scheme == "scss":
            return value_scss(asset, as_of, rates=rates)
        if asset.scheme == "mis":
            return value_mis(asset, as_of, rates=rates)
    if isinstance(asset, GoldAsset):
        return value_gold(asset, as_of, prices=prices)
    if isinstance(asset, BondAsset):
        return value_bond(asset, as_of)
    if isinstance(asset, SavingsAsset):
        return _result(asset.balance, None, None, asset.balance, method=ValuationMethod.USER)
    if isinstance(asset, NPSAsset):
        value = asset.tier1_balance + (asset.tier2_balance or Decimal(0))
        return _result(value, None, None, value, method=ValuationMethod.USER)
    if isinstance(asset, (TermInsuranceAsset, HealthInsuranceAsset)):
        return _result(
            Decimal(0),
            None,
            None,
            Decimal(0),
            (f"protection, not an asset; cover amount {format_inr(asset.cover)}",),
            ValuationMethod.USER,
        )
    if isinstance(asset, TraditionalPolicyAsset):
        return value_traditional_policy(asset, as_of)
    if isinstance(asset, ULIPAsset):
        return _result(asset.fund_value, None, None, asset.fund_value, method=ValuationMethod.USER)
    if isinstance(asset, PropertyAsset):
        return _result(
            asset.estimated_value, None, None, asset.estimated_value, method=ValuationMethod.USER
        )
    raise NotImplementedError(f"Valuation for {asset.asset_type!r} is not implemented")
