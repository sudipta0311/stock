"""C13 worked tax cases with paise-exact expected results."""

from datetime import date
from decimal import Decimal

import pytest

from wealth.policy.tax import (
    Lot,
    TaxContext,
    days_to_long_term,
    estimate_sale_tax,
    is_long_term,
    load_tax_rules,
    ltcg_harvest,
    select_lots,
)


def _lot(
    lot_id: str,
    *,
    tax_class: str = "listed_equity",
    quantity: str = "100",
    cost: str = "1000",
    acquired: date = date(2025, 1, 10),
    fmv: str | None = None,
    maturity_date: date | None = None,
) -> Lot:
    return Lot(
        lot_id=lot_id,
        tax_class=tax_class,  # type: ignore[arg-type]
        quantity=Decimal(quantity),
        cost_per_unit=Decimal(cost),
        acquired_on=acquired,
        fmv_2018_01_31=None if fmv is None else Decimal(fmv),
        maturity_date=maturity_date,
    )


def test_listed_equity_same_anniversary_is_short_term_and_cess_is_added() -> None:
    """100 × (1,500 − 1,000) × 20% = 10,000; 4% cess is 400."""
    lot = _lot("st", quantity="100", acquired=date(2025, 1, 10))

    estimate = estimate_sale_tax(
        [(lot, Decimal("100"), Decimal("1500"))],
        date(2026, 1, 10),
        TaxContext(slab_pct=Decimal("30")),
    )

    assert estimate.lines[0].holding_period == "ST"
    assert estimate.total_tax == Decimal("10400.00")


def test_nri_estimate_exposes_listed_equity_tds_without_estimating_residence_tax() -> None:
    lot = _lot("nri-short-gain", quantity="10", acquired=date(2025, 1, 10))

    estimate = estimate_sale_tax(
        [(lot, Decimal("10"), Decimal("1500"))],
        date(2026, 1, 10),
        TaxContext(residency_status="NRI", us_person=True),
    )

    assert estimate.tds_estimate == estimate.total_tax == Decimal("1040.00")
    assert "tds_deducted_at_source" in estimate.notes
    assert "residence_country_tax_may_apply" in estimate.notes
    assert "us_tax_applies" in estimate.notes


def test_listed_equity_one_day_after_anniversary_is_long_term_and_exempt() -> None:
    """100 × 500 = 50,000, fully covered by the still-unused FY exemption."""
    lot = _lot("lt", quantity="100", acquired=date(2025, 1, 10))

    estimate = estimate_sale_tax(
        [(lot, Decimal("100"), Decimal("1500"))],
        date(2026, 1, 11),
        TaxContext(slab_pct=Decimal("30")),
    )

    assert estimate.lines[0].holding_period == "LT"
    assert estimate.total_tax == Decimal("0.00")
    assert estimate.exemption_used == Decimal("50000.00")


def test_equity_ltcg_exemption_is_shared_across_lots() -> None:
    """175,000 aggregate gain − 125,000 exemption = 50,000 × 12.5% plus cess."""
    first = _lot("first", quantity="1", cost="0", acquired=date(2023, 1, 1))
    second = _lot("second", quantity="1", cost="0", acquired=date(2023, 1, 1))

    estimate = estimate_sale_tax(
        [(first, Decimal(1), Decimal("100000")), (second, Decimal(1), Decimal("75000"))],
        date(2026, 8, 1),
        TaxContext(),
    )

    assert estimate.total_tax == Decimal("6500.00")
    assert estimate.exemption_used == Decimal("125000.00")


def test_prior_equity_ltcg_reduces_the_shared_fy_exemption() -> None:
    """A prior 100,000 gain leaves 25,000 exempt; 150,000 × 12.5% plus cess is 19,500."""
    first = _lot("first", quantity="1", cost="0", acquired=date(2023, 1, 1))
    second = _lot("second", quantity="1", cost="0", acquired=date(2023, 1, 1))

    estimate = estimate_sale_tax(
        [(first, Decimal(1), Decimal("100000")), (second, Decimal(1), Decimal("75000"))],
        date(2026, 8, 1),
        TaxContext(fy_realised_equity_ltcg=Decimal("100000")),
    )

    assert estimate.total_tax == Decimal("19500.00")
    assert estimate.exemption_used == Decimal("25000.00")


def test_pre_2018_equity_uses_grandfathered_cost() -> None:
    """max(500 actual, min(800 FMV, 1,200 sale)) gives an 800 basis and 400 gain."""
    lot = _lot(
        "old",
        quantity="1",
        cost="500",
        acquired=date(2017, 6, 1),
        fmv="800",
    )

    estimate = estimate_sale_tax(
        [(lot, Decimal(1), Decimal("1200"))],
        date(2024, 8, 1),
        TaxContext(),
    )

    assert estimate.lines[0].cost_basis_per_unit == Decimal("800")
    assert estimate.lines[0].gain == Decimal("400.00")


def test_missing_pre_2018_fmv_keeps_actual_cost_and_adds_a_note() -> None:
    lot = _lot("old-no-fmv", quantity="1", cost="500", acquired=date(2017, 6, 1))

    estimate = estimate_sale_tax(
        [(lot, Decimal(1), Decimal("1200"))],
        date(2024, 8, 1),
        TaxContext(),
    )

    assert estimate.lines[0].cost_basis_per_unit == Decimal("500")
    assert "grandfather_fmv_missing" in estimate.notes


def test_foreign_equity_switches_from_slab_to_ltcg_after_more_than_24_months() -> None:
    """30% slab plus cess is 31,200; 12.5% LTCG plus cess is 13,000."""
    lot = _lot(
        "foreign",
        tax_class="foreign_equity",
        quantity="1",
        cost="0",
        acquired=date(2024, 1, 1),
    )
    ctx = TaxContext(slab_pct=Decimal("30"))

    short = estimate_sale_tax([(lot, Decimal(1), Decimal("100000"))], date(2025, 12, 31), ctx)
    long = estimate_sale_tax([(lot, Decimal(1), Decimal("100000"))], date(2026, 1, 2), ctx)

    assert not is_long_term(lot, date(2025, 12, 31))
    assert is_long_term(lot, date(2026, 1, 2))
    assert short.total_tax == Decimal("31200.00")
    assert long.total_tax == Decimal("13000.00")


def test_post_2023_debt_mutual_fund_is_always_taxed_at_slab() -> None:
    """A 10,000 gain at 20% slab produces 2,000 tax and 80 cess after three years."""
    lot = _lot(
        "debt",
        tax_class="debt_mf_post_2023",
        quantity="1",
        cost="0",
        acquired=date(2023, 5, 1),
    )

    estimate = estimate_sale_tax(
        [(lot, Decimal(1), Decimal("10000"))],
        date(2026, 5, 2),
        TaxContext(slab_pct=Decimal("20")),
    )

    assert estimate.lines[0].holding_period == "ST"
    assert estimate.total_tax == Decimal("2080.00")


def test_days_to_long_term_includes_the_day_after_the_anniversary() -> None:
    """The first LT date is 2027-01-16, which is 76 days after 2026-11-01."""
    lot = _lot("wait", quantity="1", acquired=date(2026, 1, 15))

    assert days_to_long_term(lot, date(2026, 11, 1)) == 76


def test_select_lots_chooses_loss_then_lt_gain_before_st_gain() -> None:
    """A is a loss, B is LT, and C is ST, so selling two units consumes A then B."""
    loss = _lot("A", quantity="1", cost="1200", acquired=date(2025, 1, 1))
    long_gain = _lot("B", quantity="1", cost="1000", acquired=date(2024, 1, 1))
    short_gain = _lot("C", quantity="1", cost="1000", acquired=date(2026, 8, 1))

    selected = select_lots(
        [short_gain, long_gain, loss],
        Decimal(2),
        Decimal("1100"),
        date(2026, 10, 1),
        TaxContext(),
    )

    assert [(lot.lot_id, quantity) for lot, quantity in selected] == [
        ("A", Decimal(1)),
        ("B", Decimal(1)),
    ]


def test_short_term_loss_offsets_long_term_gain_before_tax() -> None:
    """20,000 ST loss offsets 50,000 LT gain; 30,000 taxable at 12.5% plus cess."""
    short_loss = _lot("loss", quantity="1", cost="20000", acquired=date(2026, 9, 15))
    long_gain = _lot("gain", quantity="1", cost="0", acquired=date(2023, 1, 1))

    estimate = estimate_sale_tax(
        [
            (short_loss, Decimal(1), Decimal("0")),
            (long_gain, Decimal(1), Decimal("50000")),
        ],
        date(2026, 10, 1),
        TaxContext(fy_realised_equity_ltcg=Decimal("125000")),
    )

    assert estimate.lines[1].taxable_gain == Decimal("30000.00")
    assert estimate.total_tax == Decimal("3900.00")


def test_ltcg_harvest_uses_remaining_exemption_and_whole_units() -> None:
    """100,000 remaining exemption ÷ 200 gain per share buys a 500-share harvest."""
    lot = _lot("harvest", quantity="2000", cost="100", acquired=date(2024, 1, 1))

    suggestion = ltcg_harvest(
        [lot],
        {"harvest": Decimal("300")},
        date(2027, 2, 10),
        TaxContext(fy_realised_equity_ltcg=Decimal("25000")),
    )

    assert suggestion is not None
    assert suggestion.lots == ((lot, Decimal("500")),)
    assert suggestion.gains_booked == Decimal("100000.00")
    assert suggestion.tax_saved == Decimal("12500.00")


def test_ltcg_harvest_is_not_suggested_outside_february_and_march() -> None:
    """November 10 is outside the 1 February to 31 March harvesting window."""
    lot = _lot("harvest", quantity="10", cost="100", acquired=date(2024, 1, 1))

    assert (
        ltcg_harvest(
            [lot],
            {"harvest": Decimal("300")},
            date(2026, 11, 10),
            TaxContext(),
        )
        is None
    )


def test_fixed_rate_surcharge_is_capped_at_fifteen_percent() -> None:
    lot = _lot("surcharge", quantity="1", cost="0", acquired=date(2025, 1, 1))

    estimate = estimate_sale_tax(
        [(lot, Decimal(1), Decimal("100000"))],
        date(2025, 10, 1),
        TaxContext(surcharge_pct=Decimal("25")),
    )

    assert estimate.tax_before_cess == Decimal("23000.00")
    assert estimate.total_tax == Decimal("23920.00")


def test_sgb_redemption_at_explicit_maturity_is_exempt() -> None:
    maturity = date(2026, 10, 1)
    lot = _lot(
        "sgb",
        tax_class="sgb",
        quantity="1",
        cost="1000",
        acquired=date(2018, 10, 1),
        maturity_date=maturity,
    )

    estimate = estimate_sale_tax([(lot, Decimal(1), Decimal("2000"))], maturity, TaxContext())

    assert estimate.total_tax == Decimal("0.00")


def test_sales_before_supported_equity_rates_raise_value_error() -> None:
    lot = _lot("old-sale", quantity="1", acquired=date(2023, 1, 1))

    with pytest.raises(ValueError, match="before 2024-07-23"):
        estimate_sale_tax(
            [(lot, Decimal(1), Decimal("1200"))],
            date(2024, 7, 22),
            TaxContext(),
        )


def test_tax_rule_entries_have_sources_and_recent_classes_need_review() -> None:
    rules = load_tax_rules()
    entries = [
        rules["supported_sales_from"],
        rules["cess"],
        rules["equity_ltcg_exemption"],
        rules["grandfathering"],
        *rules["tax_classes"].values(),  # type: ignore[union-attr]
    ]

    assert all(
        entry["effective_from"] and entry["source_url"] and entry["verified"] is False
        for entry in entries
    )
    classes = rules["tax_classes"]
    assert all(
        classes[tax_class]["needs_review"] is True
        for tax_class in ("intl_fof", "intl_etf_listed_in", "gold_etf", "sgb")
    )
