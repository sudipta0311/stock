from decimal import Decimal

import pytest

from wealth.core.money import format_inr, format_usd, parse_inr, pct, round_paise


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("500000", Decimal("500000")),
        ("5,00,000", Decimal("500000")),
        ("500,000", Decimal("500000")),
        ("1,23,45,678.50", Decimal("12345678.50")),
        ("₹5,00,000", Decimal("500000")),
        ("Rs5,00,000", Decimal("500000")),
        ("Rs. 5,00,000", Decimal("500000")),
        ("INR 500,000", Decimal("500000")),
        ("5L", Decimal("500000")),
        ("5 lakh", Decimal("500000")),
        ("1.2 cr", Decimal("12000000")),
        ("2.5Cr", Decimal("25000000")),
        ("50k", Decimal("50000")),
        ("1 thousand", Decimal("1000")),
        ("10 lakhs", Decimal("1000000")),
        ("1 crore", Decimal("10000000")),
        ("0.001k", Decimal("1.000")),
        ("0.00001k", Decimal("0.01000")),
        ("1.230", Decimal("1.230")),
    ],
)
def test_parse_inr_accepts_supported_forms(text: str, expected: Decimal) -> None:
    result = parse_inr(text)
    assert result == expected
    assert result.as_tuple().exponent == -2
    assert "E" not in str(result)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "abc",
        "5L5",
        "1..2",
        "5 lakh crore",
        "-5",
        "₹-5",
        "1,23,4",
        "1234,567",
        "1,000,00",
        "1.001",
        "0.00001",
    ],
)
def test_parse_inr_rejects_invalid_or_overprecise_forms(text: str) -> None:
    with pytest.raises(ValueError):
        parse_inr(text)


@pytest.mark.parametrize(
    ("amount", "style", "decimals", "expected"),
    [
        (Decimal("12345678.5"), "full", 2, "₹1,23,45,678.50"),
        (Decimal("12345678.5"), "full", 0, "₹1,23,45,679"),
        (Decimal("12345.675"), "full", 2, "₹12,345.68"),
        (Decimal("0"), "full", 2, "₹0.00"),
        (Decimal("99999"), "compact", 2, "₹99,999"),
        (Decimal("540000"), "compact", 2, "₹5.4 L"),
        (Decimal("1235000"), "compact", 2, "₹12.35 L"),
        (Decimal("12400000"), "compact", 2, "₹1.24 Cr"),
        (Decimal("120000000"), "compact", 2, "₹12 Cr"),
        (Decimal("-5000"), "full", 2, "-₹5,000.00"),
        (Decimal("-540000"), "compact", 2, "-₹5.4 L"),
        (Decimal("-12345678.5"), "full", 2, "-₹1,23,45,678.50"),
    ],
)
def test_format_inr_uses_expected_indian_formats(
    amount: Decimal, style: str, decimals: int, expected: str
) -> None:
    assert format_inr(amount, style=style, decimals=decimals) == expected


def test_format_usd_uses_international_grouping() -> None:
    assert format_usd(Decimal("1234.5")) == "$1,234.50"
    assert format_usd(Decimal("-1234.5"), decimals=0) == "-$1,235"


def test_round_paise_and_pct_validate_boundaries() -> None:
    assert round_paise(Decimal("1.005")) == Decimal("1.01")
    assert pct("7.10") == Decimal("7.10")
    assert pct(100) == Decimal(100)
    for invalid in ("-0.1", "100.01", "NaN", 3.14):
        with pytest.raises(ValueError):
            pct(invalid)  # type: ignore[arg-type]


def test_parse_inr_results_have_fixed_two_decimal_places() -> None:
    for text in ("5L", "1.2 cr", "50k"):
        result = parse_inr(text)
        assert result.as_tuple().exponent == -2
        assert "E" not in str(result)


@pytest.mark.parametrize(
    ("amount", "label"),
    [
        (Decimal("9999999"), "1 Cr"),
        (Decimal("99999.6"), "1 L"),
        (Decimal("9999999.995"), "1 Cr"),
        (Decimal("9999999999"), "1000 Cr"),
    ],
)
def test_format_inr_compact_selects_units_after_rounding(amount: Decimal, label: str) -> None:
    assert format_inr(amount, style="compact") == f"{chr(8377)}{label}"
