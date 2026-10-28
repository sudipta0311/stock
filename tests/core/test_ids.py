"""Tests for public identifier validation helpers."""

from wealth.core.ids import isin_check_digit_ok


def test_isin_check_digit_accepts_public_samples() -> None:
    assert isin_check_digit_ok("INE002A01018")
    assert isin_check_digit_ok("US0378331005")


def test_isin_check_digit_rejects_bad_format_and_digit() -> None:
    assert not isin_check_digit_ok("INE002A01019")
    assert not isin_check_digit_ok("INE002A0101")
    assert not isin_check_digit_ok("not-an-isin")
