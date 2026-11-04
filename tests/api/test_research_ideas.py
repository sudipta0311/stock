from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from wealth.api.personalisation import (
    _error_location,
    _index_funds_for_household,
    _us_market_summary,
    _valuation_percentile_thresholds,
    valuation_label,
)
from wealth.marketdata.store import IndexValuationPercentile
from wealth.policy.actions import load_buying_rules
from wealth.policy.denmark_picks import global_fund_ideas
from wealth.policy.market_timing import map_us_market_timing
from wealth.schemas.common import Rupees


class _Amounts(BaseModel):
    by_security: dict[str, Rupees]


def test_error_location_names_fields_and_types_but_never_values() -> None:
    with pytest.raises(ValidationError) as caught:
        _Amounts(by_security={"5d7c0b0e-synthetic-key": "1234.5678"})

    where = _error_location(caught.value)

    assert where.startswith("ValidationError@")
    assert "by_security.*=" in where
    assert "1234.5678" not in where
    assert "synthetic-key" not in where


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [("32", "cheap"), ("33", "cheap"), ("66", "fair"), ("67", "expensive")],
)
def test_valuation_label_uses_research_rule_boundaries(
    percentile: str,
    expected: str,
) -> None:
    assert valuation_label(Decimal(percentile)) == expected


@pytest.mark.parametrize(
    ("percentile", "expected_label"),
    [
        ("33", "cheap"),
        ("34", "fair"),
        ("66", "fair"),
        ("67", "expensive"),
    ],
)
def test_us_market_labels_use_research_rule_boundaries(
    percentile: str,
    expected_label: str,
) -> None:
    cheap_max, expensive_min = _valuation_percentile_thresholds()
    buying = load_buying_rules()["lumpsum"]
    result = map_us_market_timing(
        Decimal(percentile),
        "long_history",
        cheap_pe_percentile_max=cheap_max,
        expensive_pe_percentile_min=expensive_min,
        expensive_valuation_percentile=Decimal(str(buying["expensive_valuation_percentile"])),
        stp_tranches_normal=int(buying["stp_tranches_normal"]),
        stp_tranches_expensive=int(buying["stp_tranches_expensive"]),
    )

    assert result.label == expected_label


@pytest.mark.parametrize(("percentile", "expected_parts"), [("79", 3), ("80", 6)])
def test_us_market_extra_parts_use_buying_rule_threshold(
    percentile: str,
    expected_parts: int,
) -> None:
    cheap_max, expensive_min = _valuation_percentile_thresholds()
    buying = load_buying_rules()["lumpsum"]
    result = map_us_market_timing(
        Decimal(percentile),
        "long_history",
        cheap_pe_percentile_max=cheap_max,
        expensive_pe_percentile_min=expensive_min,
        expensive_valuation_percentile=Decimal(str(buying["expensive_valuation_percentile"])),
        stp_tranches_normal=int(buying["stp_tranches_normal"]),
        stp_tranches_expensive=int(buying["stp_tranches_expensive"]),
    )

    assert result.extra_money_parts == expected_parts


def test_short_us_market_history_hides_percentile_and_uses_normal_parts() -> None:
    cheap_max, expensive_min = _valuation_percentile_thresholds()
    buying = load_buying_rules()["lumpsum"]
    result = map_us_market_timing(
        Decimal("90"),
        "short_history",
        cheap_pe_percentile_max=cheap_max,
        expensive_pe_percentile_min=expensive_min,
        expensive_valuation_percentile=Decimal(str(buying["expensive_valuation_percentile"])),
        stp_tranches_normal=int(buying["stp_tranches_normal"]),
        stp_tranches_expensive=int(buying["stp_tranches_expensive"]),
    )

    assert result.percentile is None
    assert result.label is None
    assert result.extra_money_parts == 3
    summary = _us_market_summary(
        IndexValuationPercentile(
            index_name="S&P 500",
            as_of=date(2026, 10, 30),
            pe=Decimal("25"),
            percentile=Decimal("90"),
            history_label="short_history",
            history_count=5,
            source="synthetic_test",
            source_url="https://example.test/us-index",
        )
    )
    assert summary is not None and summary.percentile is None


def test_us_market_is_null_without_a_stored_valuation() -> None:
    assert _us_market_summary(None) is None


def test_index_funds_are_empty_without_a_denmark_resident() -> None:
    session = Mock()
    session.scalar.return_value = None
    session.scalars.return_value.all.return_value = []

    assert _index_funds_for_household(session, uuid4()) == []


def test_api_index_funds_follow_the_danish_members_broker_choice() -> None:
    session = Mock()
    session.scalar.return_value = None
    session.scalars.return_value.all.return_value = [
        SimpleNamespace(
            is_self=True,
            residency={"country": "DK"},
            investing={"broker": "saxo"},
        )
    ]

    funds = _index_funds_for_household(session, uuid4())

    assert [(fund.exposure, fund.role) for fund in funds] == [
        ("global_all_world", "plan_main"),
        ("global_sp500", "option"),
        ("global_nasdaq100", "option"),
    ]
    assert funds[0].instrument.id == global_fund_ideas("DK", "saxo")[0].id
    expected = global_fund_ideas("DK", "saxo")[0]
    assert funds[0].instrument.isin == expected.isin
    assert funds[0].instrument.listed_name == expected.listed_name
    assert funds[0].instrument.accumulating is expected.accumulating
