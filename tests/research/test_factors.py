from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from wealth.research.factors import (
    dist_from_200dma,
    fundamental_factors,
    max_drawdown_1y,
    momentum_12_1,
    pe_percentile_own_10y,
    volatility_1y,
)


@dataclass(frozen=True)
class Fact:
    metric: str
    period_end: date
    period_type: str
    value: Decimal
    known_at: datetime
    filed_at: datetime


def test_price_factors_use_12_to_1_month_and_trailing_200_sessions() -> None:
    first_day = date(2025, 1, 1)
    prices = [
        (first_day + timedelta(days=offset), Decimal("100") if offset <= 31 else Decimal("150"))
        for offset in range(400)
    ]
    prices.append((date(2026, 2, 2), Decimal("999")))

    assert momentum_12_1(prices, date(2026, 2, 1)) == Decimal("0.5")

    rising = [
        (date(2026, 1, 1) + timedelta(days=offset), Decimal(offset + 1)) for offset in range(200)
    ]
    expected_distance = Decimal(200) / Decimal("100.5") - Decimal(1)
    assert dist_from_200dma(rising, date(2026, 7, 19)) == expected_distance
    assert max_drawdown_1y(rising, date(2026, 7, 19)) == Decimal(0)


def test_constant_prices_have_zero_volatility_and_short_history_is_missing() -> None:
    day = date(2026, 1, 1)
    constant = [(day + timedelta(days=offset), Decimal("42")) for offset in range(200)]
    short = constant[:199]

    assert volatility_1y(constant, date(2026, 7, 19)) == Decimal(0)
    assert volatility_1y(short, date(2026, 7, 19)) is None
    assert momentum_12_1(short, date(2026, 7, 19)) is None
    assert max_drawdown_1y(short, date(2026, 7, 19)) is None
    assert dist_from_200dma(short, date(2026, 7, 19)) is None


def test_max_drawdown_reports_peak_to_trough_loss() -> None:
    day = date(2026, 1, 1)
    prices = [
        (day + timedelta(days=offset), Decimal("100") if offset < 199 else Decimal("80"))
        for offset in range(200)
    ]

    assert max_drawdown_1y(prices, date(2026, 7, 19)) == Decimal("-0.2")


def test_fundamental_factors_exclude_filings_not_yet_known() -> None:
    period_ends = [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]
    facts = [
        Fact(
            "net_income",
            period_end,
            "Q",
            Decimal("10"),
            datetime(2026, 1, 20, tzinfo=UTC),
            datetime(2026, 1, 20, tzinfo=UTC),
        )
        for period_end in period_ends
    ]
    facts.extend(
        [
            Fact(
                "net_income",
                period_ends[-1],
                "Q",
                Decimal("20"),
                datetime(2026, 2, 10, tzinfo=UTC),
                datetime(2026, 2, 10, tzinfo=UTC),
            ),
            Fact(
                "total_equity",
                period_ends[-1],
                "FY",
                Decimal("100"),
                datetime(2026, 1, 20, tzinfo=UTC),
                datetime(2026, 1, 20, tzinfo=UTC),
            ),
        ]
    )

    before = fundamental_factors(facts, as_of=date(2026, 2, 1), price=None)
    after = fundamental_factors(facts, as_of=date(2026, 2, 11), price=None)

    assert before.roe == Decimal("0.4")
    assert after.roe == Decimal("0.5")


def test_own_pe_percentile_waits_for_ten_years_of_monthly_history() -> None:
    as_of = date(2025, 12, 31)
    history = [
        (date(2016 + offset // 12, offset % 12 + 1, 1), Decimal(offset + 1))
        for offset in range(120)
    ]

    assert pe_percentile_own_10y(Decimal("60"), history[:119], as_of) is None
    assert pe_percentile_own_10y(Decimal("60"), history, as_of) == Decimal("0.5")


def test_a_split_sized_yield_is_treated_as_missing() -> None:
    known_at = datetime(2026, 1, 20, tzinfo=UTC)
    quarter_ends = [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]
    facts = [Fact("eps_diluted", on, "Q", Decimal("30"), known_at, known_at) for on in quarter_ends]

    # Pre-split EPS of 120 a year against a post-split price of 150 reads as an 80% yield.
    after_split = fundamental_factors(facts, as_of=date(2026, 1, 31), price=Decimal("150"))
    before_split = fundamental_factors(facts, as_of=date(2026, 1, 31), price=Decimal("3750"))

    assert after_split.earnings_yield is None
    assert before_split.earnings_yield == Decimal("0.032")


def test_fundamental_factor_formulas_use_quarters_and_available_balance_values() -> None:
    quarter_ends = [
        date(2024, 3, 31),
        date(2024, 6, 30),
        date(2024, 9, 30),
        date(2024, 12, 31),
        date(2025, 3, 31),
        date(2025, 6, 30),
        date(2025, 9, 30),
        date(2025, 12, 31),
    ]
    known_at = datetime(2026, 1, 20, tzinfo=UTC)
    facts = [
        Fact("revenue", on, "Q", Decimal(str(value)), known_at, known_at)
        for on, value in zip(quarter_ends, [80, 90, 100, 110, 100, 120, 110, 130], strict=True)
    ]
    for metric, value in (
        ("net_income", 10),
        ("operating_income", 20),
        ("eps_diluted", 2),
        ("operating_cash_flow", 30),
        ("capex", 5),
    ):
        facts.extend(
            Fact(metric, on, "Q", Decimal(value), known_at, known_at) for on in quarter_ends[-4:]
        )
    facts.extend(
        [
            Fact("total_assets", quarter_ends[-2], "Q", Decimal("300"), known_at, known_at),
            Fact("total_assets", quarter_ends[-1], "Q", Decimal("320"), known_at, known_at),
            Fact("current_liabilities", quarter_ends[-1], "Q", Decimal("100"), known_at, known_at),
            Fact("total_equity", quarter_ends[-1], "Q", Decimal("100"), known_at, known_at),
            Fact("total_debt", quarter_ends[-1], "Q", Decimal("50"), known_at, known_at),
            Fact("shares_outstanding", quarter_ends[-1], "Q", Decimal("10000"), known_at, known_at),
        ]
    )
    as_of = date(2026, 1, 31)
    historical_pe = [
        (
            date(2016 + (offset + 1) // 12, (offset + 1) % 12 + 1, 1),
            Decimal(offset + 1),
        )
        for offset in range(120)
    ]

    factors = fundamental_factors(
        facts,
        as_of=as_of,
        price=Decimal("50"),
        historical_pe=historical_pe,
    )

    assert factors.roce == Decimal("80") / Decimal("220")
    assert factors.roe == Decimal("0.4")
    assert factors.revenue_yoy == Decimal("460") / Decimal("380") - Decimal(1)
    assert factors.earnings_yield == Decimal("0.16")
    assert factors.fcf_yield == Decimal("0.0002")
    assert factors.debt_to_equity == Decimal("0.5")
    assert factors.accruals_ratio == Decimal("-80") / Decimal("310")
    assert factors.pe_percentile_own_10y == Decimal("0.05")
