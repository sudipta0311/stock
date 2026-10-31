from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from wealth.marketdata.sources.universes import Sp500Holdings, UniverseSecurity
from wealth.marketdata.sources.us_index_pe import (
    BACKFILL_SOURCE,
    CURRENT_SOURCE,
    Sp500ConstituentData,
    build_sp500_pe_valuations,
)


@dataclass(frozen=True)
class _Fact:
    metric: str
    period_end: date
    period_type: str
    value: Decimal
    known_at: datetime
    filed_at: datetime


def _fact(metric: str, period_end: date, value: str, filed_on: date) -> _Fact:
    filed_at = datetime.combine(filed_on, datetime.min.time(), tzinfo=UTC)
    return _Fact(metric, period_end, "Q", Decimal(value), filed_at, filed_at)


def _constituents() -> dict[str, Sp500ConstituentData]:
    quarters = (
        (date(2024, 12, 31), date(2025, 1, 20)),
        (date(2025, 3, 31), date(2025, 4, 20)),
        (date(2025, 6, 30), date(2025, 7, 20)),
        (date(2025, 9, 30), date(2025, 10, 20)),
        (date(2025, 12, 31), date(2026, 1, 20)),
        (date(2026, 3, 31), date(2026, 4, 20)),
        (date(2026, 6, 30), date(2026, 7, 20)),
        (date(2026, 9, 30), date(2026, 10, 20)),
    )
    profitable = [_fact("eps_diluted", end, "1", filed) for end, filed in quarters]
    profitable.extend(_fact("net_income", end, "10", filed) for end, filed in quarters)
    loss_maker = [_fact("eps_diluted", end, "-0.5", filed) for end, filed in quarters]
    loss_maker.extend(_fact("net_income", end, "-10", filed) for end, filed in quarters)
    return {
        "GAIN": Sp500ConstituentData(
            prices=[(date(2026, 10, 30), Decimal("10"))], facts=profitable
        ),
        "LOSS": Sp500ConstituentData(prices=[(date(2026, 10, 30), Decimal("5"))], facts=loss_maker),
    }


def _holdings() -> Sp500Holdings:
    return Sp500Holdings(
        securities=[
            UniverseSecurity(
                "GAIN",
                "Synthetic Growth Company",
                weight_pct=Decimal("95"),
                shares_held=Decimal("100"),
            ),
            UniverseSecurity(
                "LOSS",
                "Synthetic Loss Company",
                weight_pct=Decimal("5"),
                shares_held=Decimal("100"),
            ),
        ],
        as_of=date(2026, 10, 30),
    )


def test_current_index_pe_uses_spy_shares_and_includes_loss_makers() -> None:
    result = build_sp500_pe_valuations(
        _holdings(),
        _constituents(),
        {date(2026, 10, 30): Decimal("100")},
        on=date(2026, 10, 31),
    )

    current = next(row for row in result.observations if row.source == CURRENT_SOURCE)
    # Look-through value is 100×10 + 100×5; earnings include the loss-maker EPS.
    assert current.pe == Decimal("1500") / Decimal("200")
    assert current.pb is None
    assert current.div_yield is None


def test_sp500_history_calibrates_on_the_same_monthly_member_set() -> None:
    result = build_sp500_pe_valuations(
        _holdings(),
        _constituents(),
        {
            date(2025, 10, 31): Decimal("50"),
            date(2026, 9, 30): Decimal("75"),
            date(2026, 10, 30): Decimal("100"),
        },
        on=date(2026, 10, 31),
    )

    historical = next(row for row in result.observations if row.source == BACKFILL_SOURCE)
    # k = 1,500 / 100 = 15; historical earnings are 100×4 - 100×2.
    assert historical.as_of == date(2025, 10, 31)
    assert historical.pe == Decimal("15") * Decimal("50") / Decimal("200")


def test_history_requires_four_quarters_ending_by_the_month() -> None:
    latest_eps = [
        _fact("eps_diluted", end, "1", filed)
        for end, filed in (
            (date(2025, 12, 31), date(2026, 1, 20)),
            (date(2026, 3, 31), date(2026, 4, 20)),
            (date(2026, 6, 30), date(2026, 7, 20)),
            (date(2026, 9, 30), date(2026, 10, 20)),
        )
    ]
    constituents = _constituents()
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "GAIN",
                "Synthetic Growth Company",
                weight_pct=Decimal("90"),
                shares_held=Decimal("100"),
            ),
            UniverseSecurity(
                "RECENT",
                "Synthetic Recent Company",
                weight_pct=Decimal("10"),
                shares_held=Decimal("100"),
            ),
        ],
        as_of=date(2026, 10, 30),
    )
    constituents["RECENT"] = Sp500ConstituentData(
        prices=[(date(2026, 10, 30), Decimal("20"))], facts=latest_eps
    )

    result = build_sp500_pe_valuations(
        holdings,
        constituents,
        {date(2025, 10, 31): Decimal("50"), date(2026, 10, 30): Decimal("100")},
        on=date(2026, 10, 31),
    )

    historical = next(row for row in result.observations if row.source == BACKFILL_SOURCE)
    assert historical.pe == Decimal("1000") / Decimal("100") * Decimal("50") / Decimal("400")


def test_split_backfill_uses_later_restated_comparatives() -> None:
    holdings_date = date(2026, 10, 30)
    split_quarters = (
        date(2023, 11, 30),
        date(2024, 2, 29),
        date(2024, 5, 31),
        date(2024, 8, 31),
    )
    current_quarters = (
        date(2025, 12, 31),
        date(2026, 3, 31),
        date(2026, 6, 30),
        date(2026, 9, 30),
    )
    split_facts = (
        [_fact("eps_diluted", end, "10", end) for end in split_quarters]
        + [_fact("eps_diluted", end, "1", holdings_date) for end in split_quarters]
        + [_fact("eps_diluted", end, "1", holdings_date) for end in current_quarters]
    )
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "SPLT",
                "Synthetic Split Company",
                weight_pct=Decimal("100"),
                shares_held=Decimal("100"),
            )
        ],
        as_of=holdings_date,
    )
    constituents = {
        "SPLT": Sp500ConstituentData(prices=[(holdings_date, Decimal("10"))], facts=split_facts)
    }

    result = build_sp500_pe_valuations(
        holdings,
        constituents,
        {date(2024, 8, 31): Decimal("50"), holdings_date: Decimal("100")},
        on=date(2026, 10, 31),
    )

    historical = next(row for row in result.observations if row.source == BACKFILL_SOURCE)
    current = next(row for row in result.observations if row.source == CURRENT_SOURCE)
    assert historical.as_of == date(2024, 8, 31)
    assert historical.pe == Decimal("1.25")
    assert historical.pe > Decimal("0.125")  # Unrestated 10x EPS would depress this point.
    assert current.pe == Decimal("2.5")


def test_unrestated_ten_to_one_split_is_removed_from_monthly_calibration() -> None:
    holdings_date = date(2026, 10, 30)
    historical_cutoff = date(2025, 9, 30)
    constituents = _constituents()
    mismatch = constituents["GAIN"]
    mismatch_facts = [
        _Fact(
            fact.metric,
            fact.period_end,
            fact.period_type,
            Decimal("10")
            if fact.metric == "eps_diluted" and fact.period_end <= historical_cutoff
            else fact.value,
            fact.known_at,
            fact.filed_at,
        )
        for fact in mismatch.facts
    ]
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "STABLE",
                "Synthetic Stable Company",
                weight_pct=Decimal("90"),
                shares_held=Decimal("90"),
            ),
            UniverseSecurity(
                "MISMATCH",
                "Synthetic Ten To One Split",
                weight_pct=Decimal("10"),
                shares_held=Decimal("10"),
            ),
        ],
        as_of=holdings_date,
    )
    constituents["STABLE"] = Sp500ConstituentData(
        prices=[(holdings_date, Decimal("10"))], facts=mismatch.facts
    )
    constituents["MISMATCH"] = Sp500ConstituentData(
        prices=[(holdings_date, Decimal("10"))], facts=mismatch_facts
    )

    result = build_sp500_pe_valuations(
        holdings,
        constituents,
        {date(2025, 10, 31): Decimal("50"), holdings_date: Decimal("100")},
        on=date(2026, 10, 31),
    )

    historical = next(row for row in result.observations if row.source == BACKFILL_SOURCE)
    # The split member's implied diluted shares are one tenth of today's share basis.
    assert historical.pe == Decimal("900") / Decimal("100") * Decimal("50") / Decimal("360")


def test_tenfold_eps_growth_with_steady_shares_is_kept() -> None:
    holdings_date = date(2026, 10, 30)
    historical_cutoff = date(2025, 9, 30)
    growth_facts = [
        _Fact(
            fact.metric,
            fact.period_end,
            fact.period_type,
            Decimal("100")
            if fact.metric == "net_income" and fact.period_end > historical_cutoff
            else Decimal("10")
            if fact.metric == "eps_diluted" and fact.period_end > historical_cutoff
            else fact.value,
            fact.known_at,
            fact.filed_at,
        )
        for fact in _constituents()["GAIN"].facts
    ]
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "GROWTH",
                "Synthetic Tenfold EPS Growth",
                weight_pct=Decimal("100"),
                shares_held=Decimal("100"),
            )
        ],
        as_of=holdings_date,
    )
    constituents = {
        "GROWTH": Sp500ConstituentData(prices=[(holdings_date, Decimal("100"))], facts=growth_facts)
    }

    result = build_sp500_pe_valuations(
        holdings,
        constituents,
        {date(2025, 10, 31): Decimal("50"), holdings_date: Decimal("100")},
        on=date(2026, 10, 31),
    )

    historical = next(row for row in result.observations if row.source == BACKFILL_SOURCE)
    assert historical.pe == Decimal("100") * Decimal("50") / Decimal("400")


def test_missing_net_income_disables_the_share_basis_guard() -> None:
    holdings_date = date(2026, 10, 30)
    historical_cutoff = date(2025, 9, 30)
    eps_facts = [
        _fact("eps_diluted", end, "10" if end <= historical_cutoff else "1", filed)
        for end, filed in (
            (date(2024, 12, 31), date(2025, 1, 20)),
            (date(2025, 3, 31), date(2025, 4, 20)),
            (date(2025, 6, 30), date(2025, 7, 20)),
            (date(2025, 9, 30), date(2025, 10, 20)),
            (date(2025, 12, 31), date(2026, 1, 20)),
            (date(2026, 3, 31), date(2026, 4, 20)),
            (date(2026, 6, 30), date(2026, 7, 20)),
            (date(2026, 9, 30), date(2026, 10, 20)),
        )
    ]
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "UNKNOWN",
                "Synthetic Missing Net Income",
                weight_pct=Decimal("100"),
                shares_held=Decimal("100"),
            )
        ],
        as_of=holdings_date,
    )
    constituents = {
        "UNKNOWN": Sp500ConstituentData(prices=[(holdings_date, Decimal("10"))], facts=eps_facts)
    }

    result = build_sp500_pe_valuations(
        holdings,
        constituents,
        {date(2025, 10, 31): Decimal("50"), holdings_date: Decimal("100")},
        on=date(2026, 10, 31),
    )

    assert any(row.source == BACKFILL_SOURCE for row in result.observations)


def test_incomplete_constituent_weight_returns_coverage_error_without_rows() -> None:
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "GAIN",
                "Synthetic Growth Company",
                weight_pct=Decimal("80"),
                shares_held=Decimal("100"),
            ),
            UniverseSecurity(
                "LOSS",
                "Synthetic Loss Company",
                weight_pct=Decimal("20"),
                shares_held=Decimal("100"),
            ),
        ],
        as_of=date(2026, 10, 30),
    )

    result = build_sp500_pe_valuations(
        holdings,
        {"GAIN": _constituents()["GAIN"]},
        {date(2026, 10, 30): Decimal("100")},
        on=date(2026, 10, 31),
    )

    assert result.observations == []
    assert result.errors == {"coverage": "CoverageBelow90Pct"}


def test_implausible_per_share_eps_counts_as_uncovered() -> None:
    bad_eps = [
        _fact("eps_diluted", end, "2", filed)
        for end, filed in (
            (date(2025, 12, 31), date(2026, 1, 20)),
            (date(2026, 3, 31), date(2026, 4, 20)),
            (date(2026, 6, 30), date(2026, 7, 20)),
            (date(2026, 9, 30), date(2026, 10, 20)),
        )
    ]
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "BAD",
                "Synthetic Class Mismatch",
                weight_pct=Decimal("15"),
                shares_held=Decimal("100"),
            ),
            UniverseSecurity(
                "GOOD",
                "Synthetic Covered Company",
                weight_pct=Decimal("85"),
                shares_held=Decimal("100"),
            ),
        ],
        as_of=date(2026, 10, 30),
    )
    bad = Sp500ConstituentData(prices=[(date(2026, 10, 30), Decimal("10"))], facts=bad_eps)
    good = Sp500ConstituentData(
        prices=[(date(2026, 10, 30), Decimal("10"))], facts=_constituents()["GAIN"].facts
    )

    result = build_sp500_pe_valuations(
        holdings,
        {"BAD": bad, "GOOD": good},
        {date(2026, 10, 30): Decimal("100")},
        on=date(2026, 10, 31),
    )

    assert result.observations == []
    assert result.errors == {"coverage": "CoverageBelow90Pct"}


def test_nonpositive_current_earnings_skips_only_the_current_point() -> None:
    negative_eps = [
        _fact("eps_diluted", end, "-1", filed)
        for end, filed in (
            (date(2025, 12, 31), date(2026, 1, 20)),
            (date(2026, 3, 31), date(2026, 4, 20)),
            (date(2026, 6, 30), date(2026, 7, 20)),
            (date(2026, 9, 30), date(2026, 10, 20)),
        )
    ]
    holdings = Sp500Holdings(
        securities=[
            UniverseSecurity(
                "NEG", "Synthetic Loss One", weight_pct=Decimal("50"), shares_held=Decimal("100")
            ),
            UniverseSecurity(
                "NEG2", "Synthetic Loss Two", weight_pct=Decimal("50"), shares_held=Decimal("100")
            ),
        ],
        as_of=date(2026, 10, 30),
    )
    data = Sp500ConstituentData(prices=[(date(2026, 10, 30), Decimal("20"))], facts=negative_eps)

    result = build_sp500_pe_valuations(
        holdings,
        {"NEG": data, "NEG2": data},
        {date(2026, 10, 30): Decimal("100")},
        on=date(2026, 10, 31),
    )

    assert result.observations == []
    assert result.errors == {"current_earnings": "NonPositiveCurrentEarnings"}
