"""Table tests for deterministic Money Map quick-add parsing."""

from __future__ import annotations

import pytest

from wealth.importer.quick_add import QuickAddNotUnderstood, parse_quick_add


@pytest.mark.parametrize(
    ("text", "asset_type", "fields", "missing"),
    [
        (
            "FD 5L SBI 7.1% till Mar 2027",
            "fd",
            {
                "bank": "SBI",
                "principal": "500000.00",
                "rate_pct": "7.1",
                "maturity": "2027-03",
            },
            [],
        ),
        (
            "fd 2,00,000 hdfc 7.25 % mar-2028",
            "fd",
            {
                "bank": "HDFC Bank",
                "principal": "200000.00",
                "rate_pct": "7.25",
                "maturity": "2028-03",
            },
            [],
        ),
        ("PPF 6.5 lakh", "ppf", {"balance": "650000.00"}, ["opened_fy_start_year"]),
        ("EPF 12L", "epf", {"balance": "1200000.00"}, []),
        (
            "savings 80k icici",
            "savings",
            {"bank": "ICICI Bank", "balance": "80000.00"},
            [],
        ),
        (
            "NSC 50000 Jan 2025",
            "po_scheme",
            {"scheme": "nsc", "amount": "50000.00", "purchase": "2025-01"},
            [],
        ),
        (
            "gold 20 g 22k",
            "gold",
            {"kind": "physical", "grams": "20", "purity": 22},
            [],
        ),
        (
            "home loan 35L 8.5%",
            "loan",
            {"loan_kind": "home", "outstanding": "3500000.00", "rate_pct": "8.5"},
            [],
        ),
        (
            "credit card 60k 42%",
            "loan",
            {"loan_kind": "credit_card", "outstanding": "60000.00", "rate_pct": "42"},
            [],
        ),
        (
            "LIC 10L premium 52000",
            "traditional_policy",
            {"insurer": "LIC", "sum_assured": "1000000.00", "annual_premium": "52000.00"},
            ["plan", "start_year", "term_years", "policy_kind"],
        ),
        (
            "term 1cr hdfc life 2050",
            "term_insurance",
            {"insurer": "HDFC Life", "cover": "10000000.00", "ends_year": 2050},
            [],
        ),
        (
            "SCSS 15L Oct 2026",
            "po_scheme",
            {"scheme": "scss", "amount": "1500000.00", "purchase": "2026-10"},
            [],
        ),
    ],
)
def test_quick_add_examples_produce_deterministic_drafts(
    text: str,
    asset_type: str,
    fields: dict[str, object],
    missing: list[str],
) -> None:
    result = parse_quick_add(text)

    assert result.asset_type == asset_type
    assert {key: result.draft[key] for key in fields} == fields
    assert list(result.missing) == missing
    assert result.confidence in {"high", "medium", "low"}


def test_quick_add_accepts_rupee_sign_and_unknown_institution_as_typed() -> None:
    result = parse_quick_add("FD ₹1,25,000 Demo Credit Union 7.5% until 2028-03")

    assert result.draft["principal"] == "125000.00"
    assert result.draft["bank"] == "Demo Credit Union"
    assert result.draft["maturity"] == "2028-03"


def test_quick_add_raises_for_unrecognized_text() -> None:
    with pytest.raises(QuickAddNotUnderstood):
        parse_quick_add("random words")
