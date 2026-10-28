from collections.abc import Callable
from decimal import Decimal

from wealth.policy.foundation import foundation_check
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import parse_asset
from wealth.schemas.profile import FinancialContext
from wealth.valuation.base import Valuation


def test_emergency_fund_uses_instant_cash_and_debt_but_ignores_locked_assets(
    make_valuation: Callable[[str], Valuation],
) -> None:
    ppf = parse_asset({"asset_type": "ppf", "balance": "1000000", "opened_fy_start_year": 2015})
    savings = parse_asset({"asset_type": "savings", "bank": "Demo Bank", "balance": "1000"})
    snapshot = WealthSnapshot(
        [(ppf, make_valuation("1000000")), (savings, make_valuation("1000"))],
        [("debt", Decimal("500"), "instant")],
    )
    context = FinancialContext(monthly_expenses="500", employment_type="salaried")

    report = foundation_check(snapshot, context)

    assert report["emergency_fund"].required == Decimal("3000.00")
    assert report["emergency_fund"].have == Decimal("1500.00")
    assert report["emergency_fund"].gap == Decimal("1500.00")
    assert report["emergency_fund"].status == "gap"


def test_exact_high_interest_debt_threshold_is_included(
    make_valuation: Callable[[str], Valuation],
) -> None:
    exact = parse_asset(
        {"asset_type": "loan", "loan_kind": "credit_card", "outstanding": "25000", "rate_pct": "12"}
    )
    above = parse_asset(
        {"asset_type": "loan", "loan_kind": "personal", "outstanding": "75000", "rate_pct": "12.01"}
    )
    below = parse_asset(
        {"asset_type": "loan", "loan_kind": "personal", "outstanding": "90000", "rate_pct": "11.99"}
    )
    snapshot = WealthSnapshot(
        [(exact, make_valuation("0")), (above, make_valuation("0")), (below, make_valuation("0"))]
    )

    item = foundation_check(snapshot, FinancialContext(monthly_expenses="0"))["high_interest_debt"]

    assert item.required == Decimal("100000.00")
    assert item.gap == Decimal("100000.00")
    assert item.status == "gap"


def test_credit_card_loan_at_thirty_six_percent_is_reported_as_high_interest(
    make_valuation: Callable[[str], Valuation],
) -> None:
    card = parse_asset(
        {
            "asset_type": "loan",
            "loan_kind": "credit_card",
            "outstanding": "25000",
            "rate_pct": "36",
        }
    )
    snapshot = WealthSnapshot([(card, make_valuation("0"))])

    item = foundation_check(snapshot, FinancialContext(monthly_expenses="0"))["high_interest_debt"]

    assert card.rate_pct == Decimal("36")
    assert item.status == "gap"
    assert item.gap == Decimal("25000.00")


def test_term_cover_is_unknown_without_income_when_there_are_dependants(
    make_valuation: Callable[[str], Valuation],
) -> None:
    term = parse_asset(
        {
            "asset_type": "term_insurance",
            "insurer": "Demo Cover",
            "cover": "500000",
            "ends_year": 2040,
        }
    )
    snapshot = WealthSnapshot([(term, make_valuation("0"))])
    context = FinancialContext(monthly_expenses="1000", dependants=1)

    item = foundation_check(snapshot, context)["term_cover"]

    assert item.status == "unknown"
    assert item.required is None
    assert item.have == Decimal("500000.00")
    assert item.gap is None


def test_term_cover_required_amount_counts_loans_and_liquid_assets(
    make_valuation: Callable[[str], Valuation],
) -> None:
    loan = parse_asset(
        {"asset_type": "loan", "loan_kind": "home", "outstanding": "200000", "rate_pct": "8"}
    )
    cash = parse_asset({"asset_type": "savings", "bank": "Demo Bank", "balance": "100000"})
    cover = parse_asset(
        {
            "asset_type": "term_insurance",
            "insurer": "Demo Cover",
            "cover": "6100000",
            "ends_year": 2040,
        }
    )
    snapshot = WealthSnapshot(
        [
            (loan, make_valuation("0")),
            (cash, make_valuation("100000")),
            (cover, make_valuation("0")),
        ]
    )
    context = FinancialContext(monthly_expenses="1000", annual_income="500000", dependants=1)

    item = foundation_check(snapshot, context)["term_cover"]

    assert item.required == Decimal("6100000.00")
    assert item.have == Decimal("6100000.00")
    assert item.gap == Decimal("0.00")
    assert item.status == "ok"


def test_health_cover_uses_city_tier_and_adds_top_up_cover(
    make_valuation: Callable[[str], Valuation],
) -> None:
    health = parse_asset(
        {
            "asset_type": "health_insurance",
            "insurer": "Demo Cover",
            "cover": "500000",
            "top_up_cover": "200000",
            "renewal_month": 1,
        }
    )
    snapshot = WealthSnapshot([(health, make_valuation("0"))])

    assert (
        foundation_check(snapshot, FinancialContext(monthly_expenses="0", city_tier=2))[
            "health_cover"
        ].status
        == "ok"
    )
