"""Tests for public identifier validation helpers."""

from wealth.core.ids import isin_check_digit_ok, isin_from_cusip


def test_isin_check_digit_accepts_public_samples() -> None:
    assert isin_check_digit_ok("INE002A01018")
    assert isin_check_digit_ok("US0378331005")


def test_isin_check_digit_rejects_bad_format_and_digit() -> None:
    assert not isin_check_digit_ok("INE002A01019")
    assert not isin_check_digit_ok("INE002A0101")
    assert not isin_check_digit_ok("not-an-isin")


def test_isin_from_numeric_cusips() -> None:
    assert isin_from_cusip("037833100") == "US0378331005"
    assert isin_from_cusip("594918104") == "US5949181045"
    assert isin_from_cusip("67066G104") == "US67066G1040"


def test_isin_from_cusip_rejects_cins_and_bad_lengths() -> None:
    assert isin_from_cusip("G1151C101") is None
    assert isin_from_cusip("03783310") is None
    assert isin_from_cusip("0037833100") is None
    assert isin_from_cusip("03-833100") is None
