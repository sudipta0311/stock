from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from wealth.rates.reference import RateBook, RateUnavailable, load_rates
from wealth.schemas.assets import (
    FixedDepositAsset,
    PostOfficeAsset,
    RecurringDepositAsset,
    parse_asset,
)
from wealth.schemas.common import ValuationMethod
from wealth.valuation import (
    po_td_annual_interest,
    value_asset,
    value_fd,
    value_kvp,
    value_nsc,
    value_po_td,
    value_rd,
)


@pytest.fixture
def rates() -> RateBook:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "rates_fixture.yaml"
    return load_rates(fixture_path)


def _fd(
    *,
    principal: str = "100000",
    rate_pct: str = "7",
    start: str | None = "2025-01",
    maturity: str = "2026-01",
    payout: str = "cumulative",
    compounding: str = "quarterly",
) -> FixedDepositAsset:
    data: dict[str, object] = {
        "asset_type": "fd",
        "bank": "Example Bank",
        "principal": principal,
        "rate_pct": rate_pct,
        "maturity": maturity,
        "payout": payout,
        "compounding": compounding,
    }
    if start is not None:
        data["start"] = start
    return FixedDepositAsset.model_validate(data)


def _po(
    scheme: str,
    *,
    amount: str = "1000",
    purchase: str = "2025-01",
    rate_pct: str | None = None,
    td_years: int | None = None,
) -> PostOfficeAsset:
    data: dict[str, object] = {
        "asset_type": "po_scheme",
        "scheme": scheme,
        "amount": amount,
        "purchase": purchase,
    }
    if rate_pct is not None:
        data["rate_pct"] = rate_pct
    if td_years is not None:
        data["td_years"] = td_years
    return PostOfficeAsset.model_validate(data)


def _rd() -> RecurringDepositAsset:
    return RecurringDepositAsset.model_validate(
        {
            "asset_type": "rd",
            "institution": "Example Institution",
            "monthly_amount": "1000",
            "rate_pct": "6.7",
            "start": "2025-01",
            "maturity": "2030-01",
        }
    )


def test_fd_quarterly_compounding_matches_one_year_anchor() -> None:
    result = value_fd(_fd(), date(2026, 1, 1))
    assert result.value_today == Decimal("107185.90")
    assert result.maturity_value == Decimal("107185.90")
    assert result.method is ValuationMethod.FORMULA


def test_fd_quarterly_compounding_matches_five_year_anchor() -> None:
    asset = _fd(maturity="2030-01")
    result = value_fd(asset, date(2030, 1, 1))
    assert result.value_today == Decimal("141477.82")


def test_fd_non_cumulative_payout_only_includes_interest_since_last_payout() -> None:
    asset = _fd(payout="quarterly")
    result = value_fd(asset, date(2025, 5, 1))
    assert result.value_today == Decimal("100575.34")
    assert result.maturity_value == Decimal("100000.00")
    assert result.accrued_interest == Decimal("575.34")


def test_fd_missing_start_uses_certificate_assumption_or_returns_principal() -> None:
    asset = _fd(start=None)
    no_start = value_fd(asset, date(2025, 6, 1))
    assert no_start.value_today == Decimal("100000.00")
    assert no_start.maturity_value is None
    assert no_start.maturity_date == date(2026, 1, 1)
    assert "add start month for exact value" in no_start.notes

    assumed = value_fd(asset, date(2026, 1, 1), assumed_start=date(2025, 1, 1))
    assert assumed.value_today == Decimal("107185.90")
    assert "start month missing: assumed 2025-01" in assumed.notes


def test_fd_before_start_has_no_accrued_interest_and_after_maturity_is_capped() -> None:
    asset = _fd(start="2026-01", maturity="2027-01")
    before = value_fd(asset, date(2025, 12, 31))
    assert before.value_today == Decimal("100000.00")
    assert before.accrued_interest == Decimal("0.00")

    matured = value_fd(_fd(maturity="2026-01"), date(2027, 1, 1))
    assert matured.value_today == matured.maturity_value == Decimal("107185.90")


def test_nsc_five_year_anchors_match_published_maturity_value() -> None:
    small = value_nsc(_po("nsc", amount="1000", rate_pct="7.7"), date(2030, 1, 1))
    large = value_nsc(_po("nsc", amount="100000", rate_pct="7.7"), date(2030, 1, 1))
    assert small.value_today == Decimal("1449.03")
    assert large.value_today == Decimal("144903.38")


def test_nsc_locks_unverified_reference_rate_at_purchase(rates: RateBook) -> None:
    asset = _po("nsc", amount="1000")
    result = value_nsc(asset, date(2026, 1, 1), rates=rates)
    assert "rate from reference table (unverified)" in result.notes
    assert "rate locked at purchase" in result.notes
    with pytest.raises(RateUnavailable):
        value_nsc(_po("nsc", purchase="2024-12"), date(2025, 1, 1), rates=rates)


def test_po_time_deposit_annual_interest_matches_anchor() -> None:
    assert po_td_annual_interest(Decimal("100000"), Decimal("7.5")) == Decimal("7713.59")
    result = value_po_td(_po("td", amount="100000", rate_pct="7.5", td_years=5), date(2025, 2, 1))
    assert any("annual payout: ₹7,713.59" in note for note in result.notes)


def test_po_time_deposit_uses_rate_table_when_certificate_rate_is_missing(
    rates: RateBook,
) -> None:
    asset = _po("td", amount="100000", td_years=5)
    with pytest.raises(RateUnavailable):
        value_po_td(asset, date(2025, 2, 1))
    result = value_po_td(asset, date(2025, 2, 1), rates=rates)
    assert result.value_today > result.principal_so_far
    assert "rate from reference table (unverified)" in result.notes


def test_rd_anchor_matches_independent_month_by_month_simulation() -> None:
    result = value_rd(_rd(), date(2030, 1, 1))
    assert result.value_today == Decimal("71365.83")
    assert result.principal_so_far == Decimal("60000.00")

    with localcontext() as context:
        context.prec = 40
        monthly_factor = (Decimal(1) + Decimal("0.067") / 4) ** (Decimal(1) / 3)
        simulated = Decimal(0)
        for _ in range(60):
            simulated += Decimal(1000)
            simulated *= monthly_factor
    assert simulated.quantize(Decimal("0.01")) == result.value_today


def test_post_office_rd_uses_reference_rate_and_five_year_term(rates: RateBook) -> None:
    asset = _po("rd", amount="1000", purchase="2025-01")
    result = value_rd(asset, date(2025, 2, 1), rates=rates)
    assert result.principal_so_far == Decimal("2000.00")
    assert result.maturity_date == date(2030, 1, 1)
    assert "rate from reference table (unverified)" in result.notes


def test_kvp_doubles_on_table_maturity_and_stays_capped(rates: RateBook) -> None:
    asset = _po("kvp", amount="50000")
    maturity = value_kvp(asset, date(2034, 8, 1), rates=rates)
    after_maturity = value_kvp(asset, date(2041, 9, 1), rates=rates)
    assert maturity.value_today == Decimal("100000.00")
    assert maturity.maturity_date == date(2034, 8, 1)
    assert after_maturity.value_today == Decimal("100000.00")
    assert "rate from reference table (unverified)" in maturity.notes


def test_dispatcher_routes_supported_assets_and_user_entered_values(rates: RateBook) -> None:
    assert value_asset(_fd(), date(2026, 1, 1), rates=rates).value_today == Decimal("107185.90")
    assert value_asset(
        _po("kvp", amount="50000"), date(2034, 8, 1), rates=rates
    ).value_today == Decimal("100000.00")
    savings = value_asset(
        parse_asset({"asset_type": "savings", "bank": "Example", "balance": 100}),
        date(2025, 1, 1),
        rates=rates,
    )
    assert savings.value_today == Decimal("100.00")
