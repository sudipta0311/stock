from __future__ import annotations

import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from wealth.research.screener import ScreenerInput, alternate_markets, screen_universe


def _full_factors(value: str = "1") -> dict[str, Decimal]:
    number = Decimal(value)
    return {
        "roce": number,
        "roe": number,
        "accruals_ratio": number,
        "debt_to_equity": number,
        "earnings_yield": number,
        "fcf_yield": number,
        "momentum_12_1": number,
        "volatility_1y": number,
    }


def test_uniformly_high_roce_sector_does_not_dominate() -> None:
    rows = [
        ScreenerInput("a1", "A1", "IN", "Industry A", _full_factors("10")),
        ScreenerInput("a2", "A2", "IN", "Industry A", _full_factors("20")),
        ScreenerInput("b1", "B1", "IN", "Industry B", _full_factors("100")),
        ScreenerInput("b2", "B2", "IN", "Industry B", _full_factors("110")),
    ]

    scored = screen_universe(rows, market="IN")
    picks = {row.symbol: row for row in scored.ranked}

    assert picks["A2"].z_scores["roce"] == picks["B2"].z_scores["roce"]
    assert picks["A2"].composite == picks["B2"].composite


def test_full_coverage_requires_three_factor_groups() -> None:
    factors = {"roce": Decimal("0.2"), "earnings_yield": Decimal("0.1")}
    rows = [
        ScreenerInput("one", "ONE", "IN", "Services", factors),
        ScreenerInput("two", "TWO", "IN", "Services", factors),
    ]

    scored = screen_universe(rows, market="IN")

    assert scored.ranked == ()
    assert scored.excluded == {"one": "insufficient_data", "two": "insufficient_data"}


def test_price_only_coverage_requires_both_price_groups() -> None:
    rows = [
        ScreenerInput(
            "one",
            "ONE",
            "IN",
            "Services",
            {"momentum_12_1": Decimal("0.2"), "volatility_1y": Decimal("0.1")},
        ),
        ScreenerInput(
            "two",
            "TWO",
            "IN",
            "Services",
            {"momentum_12_1": Decimal("0.1"), "volatility_1y": Decimal("0.2")},
        ),
    ]

    scored = screen_universe(rows, market="IN", price_only=True)

    assert all(item.factor_coverage == "price_only" for item in scored.ranked)
    assert set(scored.ranked[0].groups) == {"momentum", "low_vol"}
    missing_low_vol = screen_universe(
        [
            ScreenerInput(
                "one",
                "ONE",
                "IN",
                "Services",
                {"momentum_12_1": Decimal("0.2")},
            )
        ],
        market="IN",
        price_only=True,
    )
    assert missing_low_vol.excluded == {"one": "insufficient_data"}


def test_us_fundamentals_only_requires_quality_and_value_groups() -> None:
    rows = [
        ScreenerInput(
            "one",
            "ONE",
            "US",
            "Industrials",
            {
                "roce": Decimal("0.2"),
                "earnings_yield": Decimal("0.1"),
                "momentum_12_1": Decimal("0.9"),
                "volatility_1y": Decimal("0.8"),
            },
            in_sp500=True,
        ),
        ScreenerInput(
            "two",
            "TWO",
            "US",
            "Industrials",
            {"roce": Decimal("0.1")},
            in_sp500=True,
        ),
    ]

    scored = screen_universe(rows, market="US")

    assert len(scored.ranked) == 1
    assert scored.ranked[0].factor_coverage == "fundamentals_only"
    assert set(scored.ranked[0].groups) == {"quality", "value"}
    assert any("S&P 500 membership" in reason for reason in scored.ranked[0].reasons)
    assert scored.excluded == {"two": "insufficient_data"}


def test_screener_import_does_not_load_model_frameworks() -> None:
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import wealth.research.screener; "
        "blocked = ('openai', 'langgraph'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in blocked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_liquidity_floor_excludes_indian_security_below_five_crore() -> None:
    row = ScreenerInput(
        "one",
        "ONE",
        "IN",
        "Services",
        _full_factors(),
        median_3m_traded_value=Decimal("49999999"),
    )

    scored = screen_universe([row], market="IN", liquidity_floor=Decimal("50000000"))

    assert scored.excluded == {"one": "liquidity_below_floor"}


def test_unclassified_securities_are_flagged_and_not_peer_pooled() -> None:
    rows = [
        ScreenerInput("one", "ONE", "IN", "Unclassified", _full_factors("1")),
        ScreenerInput("two", "TWO", "IN", "Unclassified", _full_factors("100")),
    ]

    scored = screen_universe(rows, market="IN")

    assert all(item.z_scores["roce"] == Decimal(0) for item in scored.ranked)
    assert all("Unclassified sector" in item.reasons[0] for item in scored.ranked)


def test_ties_use_symbol_order_deterministically() -> None:
    rows = [
        ScreenerInput(identifier, symbol, "IN", "Unclassified", _full_factors())
        for identifier, symbol in (("three", "ZED"), ("one", "ALPHA"), ("two", "BETA"))
    ]

    first = screen_universe(rows, market="IN")
    second = screen_universe(list(reversed(rows)), market="IN")

    assert [row.symbol for row in first.ranked] == ["ALPHA", "BETA", "ZED"]
    assert [row.symbol for row in first.shortlist] == ["ALPHA"]
    assert first == second


def test_twenty_security_decile_edges_and_three_hand_computed_composites() -> None:
    rows = [
        ScreenerInput(
            str(index),
            f"T{index:02d}",
            "IN",
            "Technology",
            {
                "roce": Decimal(index),
                "earnings_yield": Decimal(index),
                "fcf_yield": Decimal(index),
                "momentum_12_1": Decimal(index),
                "volatility_1y": Decimal(20 - index),
            },
        )
        for index in range(20)
    ]

    scored = screen_universe(rows, market="IN")
    picks = {row.symbol: row for row in scored.ranked}
    standard_deviation = Decimal("33.25").sqrt()

    assert len(scored.ranked) == 20
    assert [row.symbol for row in scored.shortlist] == ["T19", "T18"]
    assert picks["T19"].decile == picks["T18"].decile == 10
    assert picks["T17"].decile == 9
    for index in (0, 10, 19):
        expected = (Decimal(index) - Decimal("9.5")) / standard_deviation
        assert abs(picks[f"T{index:02d}"].composite - expected) < Decimal("1e-26")


def test_a_capped_research_run_takes_both_markets_in_turn() -> None:
    picks = {"IN": ["in-1", "in-2", "in-3"], "US": ["us-1", "us-2"]}

    ordered = alternate_markets(picks)

    assert ordered == ["in-1", "us-1", "in-2", "us-2", "in-3"]
    assert ordered[:4].count("us-1") + ordered[:4].count("us-2") == 2
    assert alternate_markets({"US": ["us-1"]}) == ["us-1"]
