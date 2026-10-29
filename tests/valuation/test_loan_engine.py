from datetime import date
from decimal import Decimal

import pytest

from wealth.schemas.assets import LoanAsset
from wealth.valuation.loans import LoanEvent, loan_position, prepay_options


def loan(
    *,
    outstanding: str = "10000",
    rate: str = "12",
    monthly_payment: str | None = "1000",
    outstanding_as_of: str = "2026-01",
) -> LoanAsset:
    return LoanAsset.model_validate(
        {
            "asset_type": "loan",
            "loan_kind": "home",
            "outstanding": outstanding,
            "rate_pct": rate,
            "emi": monthly_payment,
            "outstanding_as_of": outstanding_as_of,
        }
    )


def test_loan_position_matches_hand_computed_monthly_schedule() -> None:
    position = loan_position(loan(), [], date(2026, 3, 1))

    assert position.outstanding == Decimal("8191.00")
    assert position.principal_paid == Decimal("1809.00")
    assert position.interest_paid == Decimal("191.00")
    assert position.scheduled_payments[0].interest == Decimal("81.91")


def test_prepayment_shortens_payoff_and_reduces_interest() -> None:
    base = loan_position(loan(), [], date(2026, 1, 31))
    event = LoanEvent("prepayment", date(2026, 2, 10), amount=Decimal("2000"))
    after = loan_position(loan(), [event], date(2026, 1, 31))

    assert after.months_left < base.months_left
    assert after.interest_to_pay < base.interest_to_pay


def test_rate_change_applies_from_its_month() -> None:
    event = LoanEvent("rate_change", date(2026, 2, 15), rate_pct=Decimal("24"))

    position = loan_position(loan(), [event], date(2026, 3, 1))

    assert position.outstanding == Decimal("8384.00")
    assert position.interest_paid == Decimal("384.00")


def test_balance_update_reanchors_the_position() -> None:
    event = LoanEvent("balance_update", date(2026, 2, 15), amount=Decimal("5000"))

    position = loan_position(loan(), [event], date(2026, 3, 1))

    assert position.outstanding == Decimal("4050.00")
    assert position.principal_paid == Decimal("950.00")


def test_missed_payment_capitalizes_monthly_interest() -> None:
    event = LoanEvent("missed_emi", date(2026, 2, 5))

    position = loan_position(loan(), [event], date(2026, 3, 1))

    assert position.outstanding == Decimal("9201.00")
    assert position.interest_paid == Decimal("101.00")


def test_payment_that_only_covers_interest_has_no_payoff_date() -> None:
    position = loan_position(loan(monthly_payment="100", rate="12"), [], date(2026, 1, 31))

    assert position.never_pays_off is True
    assert position.payoff_month is None
    assert position.months_left is None


def test_prepay_options_match_twenty_year_home_loan_calculation() -> None:
    monthly_payment = Decimal("26034.70")
    home_loan = loan(
        outstanding="3000000",
        rate="8.5",
        monthly_payment=str(monthly_payment),
        outstanding_as_of="2026-10",
    ).model_copy(update={"original_amount": Decimal("3000000")})

    options = prepay_options(home_loan, [], Decimal("200000"), date(2026, 10, 29))

    assert options.keep_emi_months_saved == 36
    assert options.keep_emi_interest_saved == Decimal("756578.72")
    assert options.keep_end_date_new_emi == Decimal("24299.05")
    assert options.keep_end_date_interest_saved == Decimal("216553.78")


@pytest.mark.parametrize("amount", [Decimal("-1"), Decimal("NaN")])
def test_prepay_preview_rejects_invalid_amount(amount: Decimal) -> None:
    with pytest.raises(ValueError):
        prepay_options(loan(), [], amount, date(2026, 1, 31))
