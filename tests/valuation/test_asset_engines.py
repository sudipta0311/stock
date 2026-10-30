from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from wealth.core.dates import MonthYear
from wealth.rates.reference import (
    EpfRatePeriod,
    EpfReference,
    RateBook,
    RateUnavailable,
    load_rates,
)
from wealth.schemas.assets import ASSET_MODELS, TraditionalPolicyAsset, parse_asset
from wealth.schemas.common import AssetType, ValuationMethod
from wealth.valuation import (
    PriceUnavailable,
    StaticPrices,
    amortise,
    emi,
    policy_irr,
    project_ppf,
    value_asset,
    xirr,
)

AS_OF = date(2026, 11, 1)
GOLD_PRICES = StaticPrices(
    gold_prices={AS_OF: Decimal("7500")},
    navs={("GOLD-ETF", AS_OF): Decimal("75.25")},
)


@pytest.fixture
def rates() -> RateBook:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "rates_fixture.yaml"
    return load_rates(path)


def _asset(asset_type: str, **fields: object):
    return parse_asset({"asset_type": asset_type, **fields})


def test_ppf_projection_anchor() -> None:
    assert project_ppf(Decimal("500000"), Decimal("150000"), 5, Decimal("7.1")) == Decimal(
        "1630260.43"
    )


def test_ppf_accrues_balance_and_uses_opening_financial_year_for_maturity(
    rates: RateBook,
) -> None:
    result = value_asset(
        _asset(
            "ppf",
            balance="500000",
            opened_fy_start_year=2020,
            yearly_contribution="150000",
            balance_as_of="2026-04",
        ),
        date(2026, 10, 1),
        rates=rates,
    )
    assert result.value_today > Decimal("500000.00")
    assert result.maturity_date == date(2036, 4, 1)
    assert result.maturity_value is not None
    assert "accrued, credited 31 Mar" in result.notes
    assert "projection at current rate" in result.notes


def test_ppf_projection_includes_interest_through_current_fy_end() -> None:
    result = value_asset(
        _asset(
            "ppf",
            balance="500000",
            opened_fy_start_year=2015,
            yearly_contribution="150000",
            balance_as_of="2026-04",
        ),
        date(2026, 11, 1),
    )
    assert result.maturity_value == Decimal("1418892.73")
    assert result.value_today == Decimal("520813.70")


def test_epf_credits_contributions_and_uses_latest_rate_when_fy_rate_missing(
    rates: RateBook,
) -> None:
    asset = _asset(
        "epf",
        balance="100000",
        monthly_employee="1000",
        monthly_employer="1000",
        vpf_monthly="500",
        balance_as_of="2024-04",
    )
    valued = value_asset(asset, date(2025, 3, 31), rates=rates)
    assert valued.value_today > valued.principal_so_far
    assert valued.principal_so_far == Decimal("130000.00")
    missing_rate = value_asset(asset, date(2026, 7, 1), rates=rates)
    assert "using latest available rate from FY 2024-25" in missing_rate.notes


def test_epf_uses_epfo_monthly_running_balance_interest(rates: RateBook) -> None:
    asset = _asset(
        "epf",
        balance="1000000",
        monthly_employee="13670",
        balance_as_of="2024-04",
    )

    epf_rates = rates.model_copy(
        update={
            "epf": EpfReference(
                periods=(
                    EpfRatePeriod(
                        fy="2024-25",
                        rate_pct=Decimal("8.25"),
                        source_url="https://example.invalid/fake/epf-example",
                        verified=False,
                    ),
                )
            )
        }
    )
    valued = value_asset(asset, date(2025, 3, 31), rates=epf_rates)

    assert valued.principal_so_far == Decimal("1164040.00")
    assert valued.accrued_interest == Decimal("88702.76")

    # Mid-May only April has ended: 10,00,000 × 8.25 / 1200 = 6,875.00.
    for as_of in (date(2024, 4, 30), date(2024, 5, 15)):
        part_year = value_asset(asset, as_of, rates=epf_rates)
        assert part_year.principal_so_far == Decimal("1013670.00")
        assert part_year.accrued_interest == Decimal("6875.00")


def test_scss_accrual_resets_on_quarterly_payout_date(rates: RateBook) -> None:
    asset = _asset("po_scheme", scheme="scss", amount="1000000", purchase="2025-01")
    payout_day = value_asset(asset, date(2026, 7, 1), rates=rates)
    after_payout = value_asset(asset, date(2026, 8, 16), rates=rates)
    expected_interest = Decimal("1000000") * Decimal("0.082") * Decimal(46) / Decimal(365)
    assert payout_day.accrued_interest == Decimal("0.00")
    assert after_payout.accrued_interest == expected_interest.quantize(Decimal("0.01"))
    assert after_payout.maturity_date == date(2030, 1, 1)


def test_mis_accrues_only_since_its_monthly_payout(rates: RateBook) -> None:
    asset = _asset("po_scheme", scheme="mis", amount="100000", purchase="2025-01")
    assert value_asset(asset, date(2026, 8, 1), rates=rates).accrued_interest == Decimal("0.00")
    assert value_asset(asset, date(2026, 8, 16), rates=rates).accrued_interest > Decimal("0.00")


def test_ssy_uses_current_balance_as_of_and_twenty_one_year_maturity(rates: RateBook) -> None:
    result = value_asset(
        _asset(
            "po_scheme",
            scheme="ssy",
            amount="200000",
            purchase="2020-04",
            balance_as_of="2025-04",
        ),
        date(2026, 10, 1),
        rates=rates,
    )
    assert result.value_today > Decimal("200000.00")
    assert result.maturity_date == date(2041, 4, 1)
    assert "SSY deposits allowed for 15 years from opening" in result.notes


def test_ssy_projection_includes_interest_through_current_fy_end() -> None:
    result = value_asset(
        _asset(
            "po_scheme",
            scheme="ssy",
            amount="200000",
            purchase="2010-04",
            balance_as_of="2026-04",
        ),
        date(2026, 11, 1),
    )
    expected = Decimal("216400")
    for _ in range(4):
        expected *= Decimal("1.082")
    assert result.maturity_value == expected.quantize(Decimal("0.01"))


def test_floating_rates_fall_back_to_the_latest_period_and_locked_nsc_does_not() -> None:
    ppf = value_asset(
        _asset(
            "ppf",
            balance="500000",
            opened_fy_start_year=2020,
            balance_as_of="2026-04",
        ),
        date(2027, 1, 15),
    )
    assert ppf.value_today == Decimal("528108.22")
    assert "using latest available rate from 2026-10-01 to 2026-12-31" in ppf.notes

    ssy = value_asset(
        _asset(
            "po_scheme",
            scheme="ssy",
            amount="200000",
            purchase="2020-04",
            balance_as_of="2026-04",
        ),
        date(2027, 1, 15),
    )
    assert ssy.value_today == Decimal("212985.21")
    assert "using latest available rate from 2026-10-01 to 2026-12-31" in ssy.notes

    with pytest.raises(RateUnavailable, match="ppf"):
        value_asset(
            _asset(
                "ppf",
                balance="500000",
                opened_fy_start_year=2020,
                balance_as_of="2024-04",
            ),
            date(2025, 1, 15),
        )
    with pytest.raises(RateUnavailable, match="nsc"):
        value_asset(
            _asset("po_scheme", scheme="nsc", amount="1000", purchase="2027-02"),
            date(2027, 2, 1),
        )


def test_epf_uses_latest_declared_fy_rate_and_raises_without_prior_data() -> None:
    asset = _asset("epf", balance="100000", balance_as_of="2026-04")
    result = value_asset(asset, date(2027, 1, 15))
    # 15 Jan: April to December have ended; January has not.
    expected_interest = Decimal("100000") * Decimal("0.0825") * Decimal(9) / Decimal(12)
    assert result.value_today == Decimal("100000.00") + expected_interest.quantize(Decimal("0.01"))
    assert "using latest available rate from FY 2024-25" in result.notes

    without_prior_rate = _asset("epf", balance="100000", balance_as_of="2023-04")
    with pytest.raises(RateUnavailable, match="epf"):
        value_asset(without_prior_rate, date(2024, 1, 15))


def test_gold_physical_digital_sgb_and_market_fund_values(rates: RateBook) -> None:
    physical = value_asset(
        _asset("gold", kind="physical", grams="10", purity=22), AS_OF, prices=GOLD_PRICES
    )
    assert physical.value_today == Decimal("68750.00")

    digital = value_asset(_asset("gold", kind="digital", grams="2"), AS_OF, prices=GOLD_PRICES)
    assert digital.value_today == Decimal("15000.00")
    assert "digital gold is not regulated by SEBI" in digital.notes

    sgb = value_asset(
        _asset(
            "gold",
            kind="sgb",
            units="2",
            issue="2020-11",
            issue_price_per_gram="5000",
        ),
        AS_OF,
        prices=GOLD_PRICES,
    )
    assert sgb.value_today == Decimal("15000.00")
    assert sgb.maturity_date == date(2028, 11, 1)
    assert "half-yearly SGB coupon: ₹125.00" in sgb.notes

    etf = value_asset(
        _asset("gold", kind="etf", units="10", series="GOLD-ETF"), AS_OF, prices=GOLD_PRICES
    )
    assert etf.value_today == Decimal("752.50")
    assert etf.method is ValuationMethod.MARKET


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "physical", "grams": "1", "purity": 24},
        {"kind": "etf", "units": "1", "series": "GOLD-ETF"},
    ],
)
def test_market_gold_requires_a_price(fields: dict[str, object]) -> None:
    with pytest.raises(PriceUnavailable):
        value_asset(_asset("gold", **fields), AS_OF)


def test_bond_coupon_schedule_and_tbill_face_value() -> None:
    bond = _asset(
        "bond",
        kind="gsec",
        face_value="100000",
        coupon_pct="8.2",
        maturity="2030-03",
    )
    assert bond.coupon_frequency == 2
    accrued = value_asset(bond, date(2025, 10, 16))
    expected = Decimal("100000") * Decimal("0.082") * Decimal(45) / Decimal(365)
    assert accrued.accrued_interest == expected.quantize(Decimal("0.01"))
    assert "approximate; listed bonds should use market price" in accrued.notes

    tbill = value_asset(
        _asset("bond", kind="tbill", face_value="100000", maturity="2027-01"), AS_OF
    )
    assert tbill.value_today == Decimal("100000.00")
    assert "discount instrument; shown at face value" in tbill.notes
    corporate = _asset("bond", kind="corporate", face_value="1000", maturity="2030-01")
    assert corporate.coupon_frequency == 1


def test_traditional_policy_estimate_and_irr_anchor() -> None:
    policy = _asset(
        "traditional_policy",
        insurer="Example Assurance",
        plan="Savings Plan",
        policy_kind="endowment",
        sum_assured="1500000",
        annual_premium="52000",
        start_year=2005,
        term_years=21,
        premium_paying_term=21,
        maturity_illustration="1500000",
    )
    result = value_asset(policy, date(2026, 1, 1))
    assert result.value_today == Decimal("1092000.00")
    assert result.maturity_date == date(2026, 4, 1)
    assert "estimate: premiums paid; add surrender value for accuracy" in result.notes
    irr = policy_irr(policy)
    assert irr is not None
    assert abs(irr - Decimal("0.027978")) <= Decimal("0.000005")
    assert (
        policy_irr(
            TraditionalPolicyAsset.model_validate(
                {**policy.model_dump(), "maturity_illustration": None}
            )
        )
        is None
    )


def test_matured_traditional_policy_uses_received_amount_or_estimate() -> None:
    common = {
        "asset_type": "traditional_policy",
        "insurer": "Example Assurance",
        "plan": "Savings Plan",
        "policy_kind": "endowment",
        "sum_assured": "1500000",
        "annual_premium": "52000",
        "start_year": 2005,
        "term_years": 21,
        "premium_paying_term": 21,
    }
    illustrated = value_asset(
        parse_asset({**common, "maturity_illustration": "1500000"}),
        date(2026, 11, 1),
    )
    assert illustrated.value_today == Decimal("1500000.00")
    assert "matured: confirm the amount received and close this policy" in illustrated.notes

    estimated = value_asset(parse_asset(common), date(2026, 11, 1))
    assert estimated.value_today == Decimal("1092000.00")
    assert "estimate: premiums paid; add surrender value for accuracy" in estimated.notes
    assert "matured: confirm the amount received and close this policy" in estimated.notes


def test_emi_anchor_and_amortisation_validation() -> None:
    assert emi(Decimal("1000000"), Decimal("9"), 240) == Decimal("8997.26")
    schedule = amortise(
        Decimal("1000000"),
        Decimal("9"),
        emi(Decimal("1000000"), Decimal("9"), 240),
        start=MonthYear(2025, 1),
    )
    assert schedule.months_to_payoff > 0
    assert schedule.payoff == MonthYear(2025, 1).add_months(schedule.months_to_payoff)
    with pytest.raises(ValueError, match="monthly interest"):
        amortise(Decimal("100000"), Decimal("12"), Decimal("1000"))


def test_xirr_requires_cashflows_to_change_sign() -> None:
    with pytest.raises(ValueError, match="change sign"):
        xirr([(date(2025, 1, 1), Decimal("100"))])


def test_value_asset_handles_every_asset_type_except_loan(rates: RateBook) -> None:
    examples: dict[AssetType, dict[str, object]] = {
        AssetType.SAVINGS: {"bank": "Test Bank", "balance": "5000"},
        AssetType.FD: {
            "bank": "Test Bank",
            "principal": "5000",
            "rate_pct": "7",
            "start": "2025-01",
            "maturity": "2027-01",
        },
        AssetType.RD: {
            "institution": "Test Institution",
            "monthly_amount": "1000",
            "rate_pct": "7",
            "start": "2025-01",
            "maturity": "2030-01",
        },
        AssetType.PPF: {
            "balance": "500000",
            "opened_fy_start_year": 2020,
            "balance_as_of": "2025-01",
        },
        AssetType.EPF: {"balance": "500000", "balance_as_of": "2025-01"},
        AssetType.NPS: {"tier1_balance": "500000", "tier2_balance": "10000"},
        AssetType.PO_SCHEME: {
            "scheme": "nsc",
            "amount": "1000",
            "purchase": "2026-10",
        },
        AssetType.BOND: {
            "kind": "gsec",
            "face_value": "100000",
            "coupon_pct": "7",
            "maturity": "2030-03",
        },
        AssetType.GOLD: {"kind": "physical", "grams": "10", "purity": 22},
        AssetType.TERM_INSURANCE: {
            "insurer": "Test Assurance",
            "cover": "5000000",
            "ends_year": 2040,
        },
        AssetType.HEALTH_INSURANCE: {
            "insurer": "Test Assurance",
            "cover": "500000",
            "renewal_month": 4,
        },
        AssetType.TRADITIONAL_POLICY: {
            "insurer": "Test Assurance",
            "plan": "Savings Plan",
            "policy_kind": "endowment",
            "sum_assured": "500000",
            "annual_premium": "25000",
            "start_year": 2020,
            "term_years": 20,
        },
        AssetType.ULIP: {
            "insurer": "Test Assurance",
            "fund_value": "50000",
            "annual_premium": "12000",
        },
        AssetType.PROPERTY: {"property_kind": "residential", "estimated_value": "8000000"},
    }
    assert set(examples) == {
        asset_type for asset_type in AssetType if asset_type is not AssetType.LOAN
    }
    for asset_type, fields in examples.items():
        asset = parse_asset({"asset_type": asset_type.value, **fields})
        prices = GOLD_PRICES if asset_type is AssetType.GOLD else None
        result = value_asset(asset, AS_OF, rates=rates, prices=prices)
        assert result.value_today >= Decimal(0)
    assert set(ASSET_MODELS) == {asset_type.value for asset_type in AssetType}


def test_nsc_uses_cached_default_rate_book_when_rates_are_omitted() -> None:
    asset = _asset("po_scheme", scheme="nsc", amount="1000", purchase="2026-11")
    result = value_asset(asset, AS_OF)
    assert result.value_today == Decimal("1000.00")
