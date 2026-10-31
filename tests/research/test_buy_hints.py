from __future__ import annotations

from decimal import Decimal

import pytest

from wealth.policy.actions import load_buying_rules
from wealth.research.buy_hints import entry_hint


@pytest.mark.parametrize(
    ("distance", "expected_code", "expected_reason"),
    [
        (Decimal("0.151"), "wait", "price_far_above_average"),
        (Decimal("0.15"), "buy_in_parts", "regular_entry_schedule"),
    ],
)
def test_price_distance_threshold_is_strict(
    distance: Decimal, expected_code: str, expected_reason: str
) -> None:
    result = entry_hint(
        market="IN",
        dist_from_200dma=distance,
        market_percentile="50",
        market_history_label="long_history",
    )

    assert result["code"] == expected_code
    assert expected_reason in result["reasons"]


def test_ninetieth_market_percentile_says_wait() -> None:
    result = entry_hint(
        market="IN",
        dist_from_200dma=Decimal("0"),
        market_percentile="90",
        market_history_label="long_history",
    )

    assert result == {
        "code": "wait",
        "parts": None,
        "reasons": ["market_valuation_high"],
    }


def test_eightieth_market_percentile_uses_expensive_schedule() -> None:
    result = entry_hint(
        market="IN",
        dist_from_200dma=Decimal("0"),
        market_percentile="80",
        market_history_label="long_history",
    )

    assert result["code"] == "buy_in_parts"
    assert result["parts"] == load_buying_rules()["lumpsum"]["stp_tranches_expensive"]
    assert result["reasons"] == ["market_valuation_elevated"]


def test_default_uses_satellite_parts_and_us_records_missing_price_history() -> None:
    default = entry_hint(
        market="IN",
        dist_from_200dma=Decimal("0"),
        market_percentile="79",
        market_history_label="long_history",
    )
    us = entry_hint(
        market="US",
        dist_from_200dma=None,
        market_percentile="79",
        market_history_label="long_history",
    )

    assert default["parts"] == load_buying_rules()["satellite"]["tranches"]
    assert default["reasons"] == ["regular_entry_schedule"]
    assert us["parts"] == default["parts"]
    assert us["reasons"] == [
        "us_price_history_unavailable",
        "regular_entry_schedule",
    ]


def test_short_history_ignores_the_percentile_but_long_history_uses_it() -> None:
    short = entry_hint(
        market="IN",
        dist_from_200dma=Decimal("0"),
        market_percentile="95",
        market_history_label="short_history",
    )
    long = entry_hint(
        market="IN",
        dist_from_200dma=Decimal("0"),
        market_percentile="95",
        market_history_label="long_history",
    )

    assert short == {
        "code": "buy_in_parts",
        "parts": load_buying_rules()["satellite"]["tranches"],
        "reasons": ["market_history_short", "regular_entry_schedule"],
    }
    assert long == {
        "code": "wait",
        "parts": None,
        "reasons": ["market_valuation_high"],
    }
