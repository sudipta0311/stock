"""Pure price and point-in-time fundamental factor calculations."""

from __future__ import annotations

import calendar
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Protocol

_MAX_PLAUSIBLE_YIELD = Decimal("0.4")

PriceSeries = Iterable[tuple[date, Decimal]]


class FundamentalRow(Protocol):
    metric: str
    period_end: date
    period_type: str
    value: Decimal
    known_at: datetime
    filed_at: datetime


@dataclass(frozen=True)
class PriceFactors:
    momentum_12_1: Decimal | None
    volatility_1y: Decimal | None
    max_drawdown_1y: Decimal | None
    dist_from_200dma: Decimal | None


@dataclass(frozen=True)
class FundamentalFactors:
    roce: Decimal | None
    roe: Decimal | None
    revenue_yoy: Decimal | None
    earnings_yield: Decimal | None
    fcf_yield: Decimal | None
    debt_to_equity: Decimal | None
    accruals_ratio: Decimal | None
    pe_percentile_own_10y: Decimal | None


def _shift_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    year, month_index = divmod(index, 12)
    month = month_index + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _prices(prices: PriceSeries, as_of: date) -> list[tuple[date, Decimal]]:
    values: dict[date, Decimal] = {}
    for on, close in prices:
        if on <= as_of and close.is_finite() and close > 0:
            values[on] = close
    return sorted(values.items())


def _price_on_or_before(prices: Sequence[tuple[date, Decimal]], on: date) -> Decimal | None:
    for price_date, close in reversed(prices):
        if price_date <= on:
            return close
    return None


def momentum_12_1(prices: PriceSeries, as_of: date) -> Decimal | None:
    """Return the 12-to-1-month price change, omitting the latest month."""
    rows = _prices(prices, as_of)
    if len(rows) < 200:
        return None
    last_date, _last_close = rows[-1]
    start = _price_on_or_before(rows, _shift_months(last_date, -12))
    end = _price_on_or_before(rows, _shift_months(last_date, -1))
    if start is None or end is None or start <= 0:
        return None
    return end / start - Decimal(1)


def volatility_1y(prices: PriceSeries, as_of: date) -> Decimal | None:
    """Return annualized population standard deviation of trailing daily log returns."""
    rows = _prices(prices, as_of)
    if len(rows) < 200:
        return None
    closes = [close for _on, close in rows[-252:]]
    returns = [(right / left).ln() for left, right in zip(closes, closes[1:], strict=False)]
    if not returns:
        return None
    mean = sum(returns, Decimal(0)) / Decimal(len(returns))
    variance = sum(((value - mean) ** 2 for value in returns), Decimal(0)) / Decimal(len(returns))
    return variance.sqrt() * Decimal(252).sqrt()


def max_drawdown_1y(prices: PriceSeries, as_of: date) -> Decimal | None:
    """Return the largest peak-to-trough loss from the trailing 252 closes."""
    rows = _prices(prices, as_of)
    if len(rows) < 200:
        return None
    peak = Decimal(0)
    drawdown = Decimal(0)
    for _on, close in rows[-252:]:
        peak = max(peak, close)
        drawdown = min(drawdown, close / peak - Decimal(1))
    return drawdown


def dist_from_200dma(prices: PriceSeries, as_of: date) -> Decimal | None:
    """Return the close's distance from its trailing 200-session moving average."""
    rows = _prices(prices, as_of)
    if len(rows) < 200:
        return None
    closes = [close for _on, close in rows[-200:]]
    average = sum(closes, Decimal(0)) / Decimal(200)
    return closes[-1] / average - Decimal(1)


def price_factors(prices: PriceSeries, as_of: date) -> PriceFactors:
    """Calculate all price factors using observations dated no later than ``as_of``."""
    rows = _prices(prices, as_of)
    return PriceFactors(
        momentum_12_1=momentum_12_1(rows, as_of),
        volatility_1y=volatility_1y(rows, as_of),
        max_drawdown_1y=max_drawdown_1y(rows, as_of),
        dist_from_200dma=dist_from_200dma(rows, as_of),
    )


def _as_of_timestamp(as_of: date) -> datetime:
    return datetime.combine(as_of, time.max, tzinfo=UTC)


def _aware_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _facts_as_of(
    facts: Iterable[FundamentalRow], as_of: date
) -> dict[tuple[str, date, str], FundamentalRow]:
    cutoff = _as_of_timestamp(as_of)
    latest: dict[tuple[str, date, str], FundamentalRow] = {}
    for fact in facts:
        if _aware_utc(fact.known_at) > cutoff or fact.period_end > as_of:
            continue
        key = (fact.metric, fact.period_end, fact.period_type)
        existing = latest.get(key)
        if existing is None or (_aware_utc(fact.known_at), _aware_utc(fact.filed_at)) > (
            _aware_utc(existing.known_at),
            _aware_utc(existing.filed_at),
        ):
            latest[key] = fact
    return latest


def _metric_rows(
    facts: dict[tuple[str, date, str], FundamentalRow], metric: str, period_type: str | None = None
) -> list[FundamentalRow]:
    return sorted(
        (
            fact
            for (fact_metric, _period_end, fact_period_type), fact in facts.items()
            if fact_metric == metric and (period_type is None or fact_period_type == period_type)
        ),
        key=lambda fact: fact.period_end,
    )


def _last_four_quarters(rows: Sequence[FundamentalRow]) -> list[FundamentalRow] | None:
    if len(rows) < 4:
        return None
    selected = list(rows[-4:])
    if any(
        not 70 <= (right.period_end - left.period_end).days <= 110
        for left, right in zip(selected, selected[1:], strict=False)
    ):
        return None
    return selected


def _plausible_yield(value: Decimal | None) -> Decimal | None:
    # Beyond 40% the price and the per-share filings disagree, usually after a stock split
    # the filings have not caught up with, so the yield is missing rather than a bargain.
    return value if value is None or abs(value) <= _MAX_PLAUSIBLE_YIELD else None


def _ttm(facts: dict[tuple[str, date, str], FundamentalRow], metric: str) -> Decimal | None:
    rows = _last_four_quarters(_metric_rows(facts, metric, "Q"))
    return sum((row.value for row in rows), Decimal(0)) if rows is not None else None


def _latest_value(
    facts: dict[tuple[str, date, str], FundamentalRow], metric: str
) -> Decimal | None:
    rows = _metric_rows(facts, metric)
    return rows[-1].value if rows else None


def pe_percentile_own_10y(
    current_pe: Decimal | None,
    historical_pe: Sequence[tuple[date, Decimal]],
    as_of: date,
) -> Decimal | None:
    """Return the current P/E percentile against at least 120 months of own history."""
    if current_pe is None or current_pe <= 0:
        return None
    cutoff = _shift_months(as_of, -120)
    history = [
        value
        for on, value in historical_pe
        if cutoff <= on <= as_of and value.is_finite() and value > 0
    ]
    if len(history) < 120:
        return None
    return Decimal(sum(value <= current_pe for value in history)) / Decimal(len(history))


def fundamental_factors(
    facts: Iterable[FundamentalRow],
    *,
    as_of: date,
    price: Decimal | None,
    historical_pe: Sequence[tuple[date, Decimal]] = (),
) -> FundamentalFactors:
    """Calculate fundamental factors with each observation filtered by its known date."""
    selected = _facts_as_of(facts, as_of)
    ttm_operating_income = _ttm(selected, "operating_income")
    ttm_net_income = _ttm(selected, "net_income")
    ttm_eps = _ttm(selected, "eps_diluted")
    ttm_revenue = _last_four_quarters(_metric_rows(selected, "revenue", "Q"))
    prior_revenue = None
    if ttm_revenue is not None:
        revenue_rows = _metric_rows(selected, "revenue", "Q")
        if len(revenue_rows) >= 8:
            prior_rows = revenue_rows[-8:-4]
            if all(
                70 <= (right.period_end - left.period_end).days <= 110
                for left, right in zip(revenue_rows[-8:], revenue_rows[-7:], strict=False)
            ):
                prior_revenue = sum((row.value for row in prior_rows), Decimal(0))
    current_revenue = (
        sum((row.value for row in ttm_revenue), Decimal(0)) if ttm_revenue is not None else None
    )
    revenue_yoy = (
        current_revenue / prior_revenue - Decimal(1)
        if current_revenue is not None and prior_revenue is not None and prior_revenue != 0
        else None
    )

    total_assets = _latest_value(selected, "total_assets")
    current_liabilities = _latest_value(selected, "current_liabilities")
    equity = _latest_value(selected, "total_equity")
    debt = _latest_value(selected, "total_debt")
    invested_capital = (
        total_assets - current_liabilities
        if total_assets is not None and current_liabilities is not None
        else equity + debt
        if equity is not None and debt is not None
        else None
    )
    roce = (
        ttm_operating_income / invested_capital
        if ttm_operating_income is not None
        and invested_capital is not None
        and invested_capital > 0
        else None
    )
    roe = (
        ttm_net_income / equity
        if ttm_net_income is not None and equity is not None and equity > 0
        else None
    )
    earnings_yield = _plausible_yield(
        ttm_eps / price if ttm_eps is not None and price is not None and price > 0 else None
    )
    ttm_operating_cash_flow = _ttm(selected, "operating_cash_flow")
    ttm_capex = _ttm(selected, "capex")
    shares = _latest_value(selected, "shares_outstanding")
    market_cap = price * shares if price is not None and shares is not None and price > 0 else None
    fcf_yield = _plausible_yield(
        (ttm_operating_cash_flow - ttm_capex) / market_cap
        if ttm_operating_cash_flow is not None
        and ttm_capex is not None
        and market_cap is not None
        and market_cap > 0
        else None
    )
    debt_to_equity = (
        debt / equity if debt is not None and equity is not None and equity > 0 else None
    )
    cash_flow_accrual = (
        (ttm_net_income - ttm_operating_cash_flow)
        if ttm_net_income is not None and ttm_operating_cash_flow is not None
        else None
    )
    asset_rows = _metric_rows(selected, "total_assets")
    average_assets = (
        (asset_rows[-1].value + asset_rows[-2].value) / Decimal(2)
        if cash_flow_accrual is not None and len(asset_rows) >= 2
        else None
    )
    accruals_ratio = (
        cash_flow_accrual / average_assets
        if cash_flow_accrual is not None and average_assets is not None and average_assets > 0
        else None
    )
    current_pe = (
        price / ttm_eps if price is not None and ttm_eps is not None and ttm_eps > 0 else None
    )
    pe_percentile = pe_percentile_own_10y(current_pe, historical_pe, as_of)
    return FundamentalFactors(
        roce=roce,
        roe=roe,
        revenue_yoy=revenue_yoy,
        earnings_yield=earnings_yield,
        fcf_yield=fcf_yield,
        debt_to_equity=debt_to_equity,
        accruals_ratio=accruals_ratio,
        pe_percentile_own_10y=pe_percentile,
    )
