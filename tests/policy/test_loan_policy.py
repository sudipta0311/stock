from decimal import Decimal

from wealth.policy.loans import (
    LoanCandidate,
    LoanPlanContext,
    load_loan_policy_rules,
    load_tax_relief_rules,
    pay_extra_or_invest,
)
from wealth.schemas.assets import LoanAsset


def candidate(
    loan_kind: str = "personal",
    rate: str = "12",
    balance: str = "500000",
    *,
    relief: str = "none",
    regime: str | None = None,
    slab: str | None = None,
    currency: str = "INR",
    outstanding_inr: str | None = None,
    identifier: str = "loan-1",
) -> LoanCandidate:
    loan = LoanAsset.model_validate(
        {
            "asset_type": "loan",
            "loan_kind": loan_kind,
            "outstanding": balance,
            "rate_pct": rate,
            "emi": "10000",
            "currency": currency,
            "tax_relief": relief,
            "outstanding_as_of": "2026-10",
        }
    )
    return LoanCandidate(
        id=identifier,
        loan=loan,
        outstanding_inr=Decimal(outstanding_inr or balance),
        emi_inr=Decimal("10000"),
        tax_regime=regime,  # type: ignore[arg-type]
        tax_slab_pct=Decimal(slab) if slab is not None else None,
    )


def context(
    *, spare: str = "25000", safety_gap: str = "0", safe_return: str = "7"
) -> LoanPlanContext:
    return LoanPlanContext(
        spare_money_inr=Decimal(spare),
        emergency_fund_gap_inr=Decimal(safety_gap),
        monthly_income_inr=Decimal("100000"),
        expected_safe_return_pct=Decimal(safe_return),
        high_interest_threshold_pct=Decimal("30"),
        tax_rules=load_tax_relief_rules(),
        debt_load_rules=load_loan_policy_rules(),
    )


def test_credit_card_is_prioritized_even_over_a_higher_rate_loan() -> None:
    decision = pay_extra_or_invest(
        [candidate(rate="18", identifier="card", loan_kind="credit_card"), candidate(rate="45")],
        context(),
    )

    assert decision.action == "PAY_EXTRA"
    assert decision.loan_id == "card"
    assert "high_interest_debt_first" in decision.reason_codes


def test_safety_gap_comes_before_any_loan_payment() -> None:
    decision = pay_extra_or_invest(
        [candidate(rate="36")], context(spare="10000", safety_gap="3000")
    )

    assert decision.action == "BUILD_SAFETY_FIRST"
    assert decision.loan_id is None
    assert decision.suggested_amount_inr == Decimal("3000.00")


def test_home_loan_tax_relief_can_put_safe_return_ahead_of_loan_cost() -> None:
    old_regime = candidate(
        loan_kind="home",
        rate="8.5",
        balance="3000000",
        relief="in_home_self_occupied",
        regime="old",
        slab="30",
    )
    invest = pay_extra_or_invest([old_regime], context(safe_return="7"))
    pay_without_relief = pay_extra_or_invest(
        [
            candidate(
                loan_kind="home",
                rate="8.5",
                balance="3000000",
                relief="in_home_self_occupied",
                regime="new",
                slab="30",
            )
        ],
        context(safe_return="7"),
    )

    assert invest.action == "INVEST"
    assert invest.effective_rate_pct == Decimal("6.500000000000000000000000000")
    assert pay_without_relief.action == "PAY_EXTRA"
    assert pay_without_relief.effective_rate_pct == Decimal("8.5")


def test_avalanche_pays_the_highest_effective_rate_first() -> None:
    decision = pay_extra_or_invest(
        [candidate(rate="11", identifier="second"), candidate(rate="19", identifier="first")],
        context(),
    )

    assert decision.loan_id == "first"


def test_dkk_loan_candidate_uses_the_fx_converted_net_worth_amount() -> None:
    dkk_loan = candidate(
        loan_kind="car",
        rate="10",
        balance="2000",
        currency="DKK",
        outstanding_inr="25000",
        identifier="dkk-loan",
    )

    decision = pay_extra_or_invest([dkk_loan], context(spare="40000"))

    assert decision.action == "PAY_EXTRA"
    assert decision.suggested_amount_inr == Decimal("25000.00")
