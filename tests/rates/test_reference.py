from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from wealth.rates.reference import RateBook, RateUnavailable, load_rates


@pytest.fixture
def rates() -> RateBook:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "rates_fixture.yaml"
    return load_rates(fixture_path)


def test_rate_lookup_includes_both_period_boundaries(rates: RateBook) -> None:
    assert rates.rate_for("nsc", date(2025, 1, 1)).rate_pct == Decimal("9.99")
    assert rates.rate_for("nsc", date(2025, 12, 31)).rate_pct == Decimal("9.99")
    assert rates.rate_for("nsc", date(2026, 1, 1)).rate_pct == Decimal("10.01")
    assert rates.rate_for("nsc", date(2026, 12, 31)).rate_pct == Decimal("10.01")


def test_rate_lookup_raises_when_scheme_or_date_is_unavailable(rates: RateBook) -> None:
    with pytest.raises(RateUnavailable, match="nsc"):
        rates.rate_for("nsc", date(2024, 12, 31))
    with pytest.raises(RateUnavailable, match="td_5y"):
        rates.rate_for("td_5y", date(2024, 12, 31))
    with pytest.raises(RateUnavailable, match="unknown"):
        rates.rate_for("unknown", date(2025, 1, 1))


def test_kvp_rate_includes_maturity_term_and_epf_uses_financial_year(rates: RateBook) -> None:
    kvp_rate = rates.rate_for("kvp", date(2025, 1, 1))
    assert kvp_rate.maturity_months == 115
    assert rates.epf_rate("2024-25").rate_pct == Decimal("8.89")
    with pytest.raises(RateUnavailable, match="epf"):
        rates.epf_rate("2025-26")


def test_reference_rejects_overlapping_periods(rates: RateBook) -> None:
    data = rates.model_dump(mode="python", by_alias=True)
    data["schemes"]["nsc"]["periods"][1]["from"] = date(2025, 12, 31)
    with pytest.raises(ValidationError, match="overlap"):
        RateBook.model_validate(data)


def test_reference_only_allows_maturity_months_for_kvp(rates: RateBook) -> None:
    data = rates.model_dump(mode="python", by_alias=True)
    data["schemes"]["td_5y"]["periods"] = [
        {
            "from": date(2025, 1, 1),
            "to": date(2025, 12, 31),
            "rate_pct": "9.87",
            "maturity_months": 60,
            "source_url": "https://example.invalid/fake/td-5y",
            "verified": False,
        }
    ]
    with pytest.raises(ValidationError, match="only valid for KVP"):
        RateBook.model_validate(data)
