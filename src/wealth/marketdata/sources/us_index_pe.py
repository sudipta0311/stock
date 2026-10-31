"""Deterministic S&P 500 trailing P/E from SSGA holdings and point-in-time EDGAR facts."""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from wealth.marketdata.sources.macro_geo import IndexValuationObservation
from wealth.marketdata.sources.universes import (
    SP500_HOLDINGS_URL,
    SPY_NAV_HISTORY_URL,
    Sp500Holdings,
    UniverseSecurity,
)
from wealth.research.factors import (
    FundamentalRow,
    _plausible_yield,
    trailing_twelve_month_eps,
    trailing_twelve_month_eps_for_period,
    trailing_twelve_month_net_income,
    trailing_twelve_month_net_income_for_period,
)

INDEX_NAME = "S&P 500"
CURRENT_SOURCE = "COMPUTED_SPY_EDGAR"
BACKFILL_SOURCE = "COMPUTED_SPY_EDGAR_BACKFILL"
MIN_COVERAGE = Decimal("0.90")
MAX_PRICE_AGE_DAYS = 20
INDEX_RETENTION_DAYS = 1826
# A split moves implied diluted shares beyond this reciprocal 2x range.
_IMPLIED_DILUTED_SHARES_RATIO_BOUNDS = (Decimal("0.5"), Decimal("2"))


@dataclass(frozen=True)
class Sp500ConstituentData:
    """Stored closes and point-in-time facts for one current S&P 500 member."""

    prices: Sequence[tuple[date, Decimal]] = ()
    facts: Sequence[FundamentalRow] = ()


@dataclass(frozen=True)
class UsIndexPeBatch:
    observations: list[IndexValuationObservation]
    errors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _CoveredMember:
    member: UniverseSecurity
    data: Sp500ConstituentData
    close: Decimal
    shares_held: Decimal
    ttm_eps: Decimal
    implied_diluted_shares: Decimal | None

    @property
    def lookthrough_market_value(self) -> Decimal:
        return self.shares_held * self.close

    @property
    def lookthrough_earnings(self) -> Decimal:
        return self.shares_held * self.ttm_eps


def _implied_diluted_shares(net_income: Decimal | None, eps: Decimal | None) -> Decimal | None:
    if net_income is None or eps is None or net_income == 0 or eps == 0:
        return None
    return net_income / eps


def _share_basis_mismatch(historical: Decimal | None, current: Decimal | None) -> bool:
    if historical is None or current is None or historical == 0 or current == 0:
        return False
    if (historical < 0) != (current < 0):
        return False
    ratio = historical / current
    minimum_ratio, maximum_ratio = _IMPLIED_DILUTED_SHARES_RATIO_BOUNDS
    return ratio < minimum_ratio or ratio > maximum_ratio


def _latest_close(prices: Sequence[tuple[date, Decimal]], as_of: date) -> Decimal | None:
    cutoff = as_of - timedelta(days=MAX_PRICE_AGE_DAYS)
    available = [
        (price_date, close)
        for price_date, close in prices
        if cutoff <= price_date <= as_of and close.is_finite() and close > 0
    ]
    return max(available, default=(date.min, None), key=lambda item: item[0])[1]


def _month_end(day: date) -> date:
    return date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])


def _monthly_nav_points(
    nav_history: Mapping[date, Decimal], *, cutoff: date, as_of: date
) -> list[tuple[date, Decimal]]:
    by_month: dict[tuple[int, int], tuple[date, Decimal]] = {}
    for nav_date, nav in sorted(nav_history.items()):
        if nav_date > as_of or nav_date < cutoff or not nav.is_finite() or nav <= 0:
            continue
        month = (nav_date.year, nav_date.month)
        if month not in by_month or nav_date > by_month[month][0]:
            by_month[month] = (nav_date, nav)
    return [
        (_month_end(date(year, month, 1)), nav)
        for (year, month), (nav_date, nav) in sorted(by_month.items())
        if _month_end(date(year, month, 1)) < as_of and nav_date <= as_of
    ]


def build_sp500_pe_valuations(
    holdings: Sp500Holdings,
    constituents: Mapping[str, Sp500ConstituentData],
    nav_history: Mapping[date, Decimal],
    *,
    on: date,
) -> UsIndexPeBatch:
    """Build current and monthly S&P 500 P/E from SPY look-through earnings.

    Historical observations use today's constituents at each past date, so they have
    survivorship bias. Historical EPS uses later restated quarters known today, a small
    look-ahead that keeps older quarters on today's share basis. Each month's calibration
    and earnings sum use the same subset of today's covered constituents.
    """
    weighted_members = [
        member
        for member in holdings.securities
        if member.weight_pct is not None and member.weight_pct > 0
    ]
    total_weight = sum((member.weight_pct for member in weighted_members), Decimal(0))
    if total_weight <= 0:
        return UsIndexPeBatch([], {"coverage": "MissingSp500Weights"})

    current_rows: list[_CoveredMember] = []
    covered_weight = Decimal(0)
    for member in weighted_members:
        data = constituents.get(member.symbol)
        shares_held = member.shares_held
        if data is None or shares_held is None or shares_held <= 0:
            continue
        close = _latest_close(data.prices, holdings.as_of)
        ttm_eps = trailing_twelve_month_eps(data.facts, holdings.as_of)
        if close is None or ttm_eps is None:
            continue
        if _plausible_yield(ttm_eps / close) is None:
            continue
        ttm_net_income = trailing_twelve_month_net_income(data.facts, holdings.as_of)
        covered_weight += member.weight_pct
        current_rows.append(
            _CoveredMember(
                member=member,
                data=data,
                close=close,
                shares_held=shares_held,
                ttm_eps=ttm_eps,
                implied_diluted_shares=_implied_diluted_shares(ttm_net_income, ttm_eps),
            )
        )

    coverage = covered_weight / total_weight
    if coverage < MIN_COVERAGE:
        return UsIndexPeBatch([], {"coverage": "CoverageBelow90Pct"})

    current_nav = nav_history.get(holdings.as_of)
    if current_nav is None or not current_nav.is_finite() or current_nav <= 0:
        return UsIndexPeBatch([], {"current_nav": "MissingCurrentNav"})

    current_market_value = sum((row.lookthrough_market_value for row in current_rows), Decimal(0))
    current_earnings = sum((row.lookthrough_earnings for row in current_rows), Decimal(0))
    observations: list[IndexValuationObservation] = []
    errors: dict[str, str] = {}
    retention_cutoff = on - timedelta(days=INDEX_RETENTION_DAYS)
    for month_end, nav in _monthly_nav_points(
        nav_history, cutoff=retention_cutoff, as_of=holdings.as_of
    ):
        if month_end < retention_cutoff:
            continue
        historical_rows: list[tuple[_CoveredMember, Decimal]] = []
        historical_weight = Decimal(0)
        for current in current_rows:
            ttm_eps = trailing_twelve_month_eps_for_period(
                current.data.facts,
                period_end=month_end,
                known_as_of=holdings.as_of,
            )
            if ttm_eps is None:
                continue
            historical_net_income = trailing_twelve_month_net_income_for_period(
                current.data.facts,
                period_end=month_end,
                known_as_of=holdings.as_of,
            )
            historical_implied_shares = _implied_diluted_shares(historical_net_income, ttm_eps)
            if _share_basis_mismatch(historical_implied_shares, current.implied_diluted_shares):
                continue
            historical_rows.append((current, ttm_eps))
            historical_weight += current.member.weight_pct
        if historical_weight / total_weight < MIN_COVERAGE:
            continue
        historical_value = sum(
            (row.lookthrough_market_value for row, _eps in historical_rows), Decimal(0)
        )
        historical_earnings = sum(
            (row.shares_held * eps for row, eps in historical_rows), Decimal(0)
        )
        if historical_earnings <= 0:
            continue
        historical_scale = historical_value / current_nav
        observations.append(
            IndexValuationObservation(
                index_name=INDEX_NAME,
                as_of=month_end,
                pe=historical_scale * nav / historical_earnings,
                pb=None,
                div_yield=None,
                source=BACKFILL_SOURCE,
                source_url=SPY_NAV_HISTORY_URL,
            )
        )

    if current_earnings <= 0:
        errors["current_earnings"] = "NonPositiveCurrentEarnings"
    else:
        observations.append(
            IndexValuationObservation(
                index_name=INDEX_NAME,
                as_of=holdings.as_of,
                pe=current_market_value / current_earnings,
                pb=None,
                div_yield=None,
                source=CURRENT_SOURCE,
                source_url=SP500_HOLDINGS_URL,
            )
        )
    return UsIndexPeBatch(observations, errors)
