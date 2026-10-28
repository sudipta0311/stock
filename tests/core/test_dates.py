from datetime import date

import pytest

from wealth.core.dates import (
    MonthYear,
    add_months,
    financial_year,
    fy_end,
    fy_start,
    parse_month_year,
)


@pytest.mark.parametrize(
    ("text", "year", "month"),
    [
        ("Mar 2027", 2027, 3),
        ("March 2027", 2027, 3),
        ("03/2027", 2027, 3),
        ("3/2027", 2027, 3),
        ("2027-03", 2027, 3),
        ("Mar-27", 2027, 3),
        ("february 2024", 2024, 2),
        ("FEB-20", 2020, 2),
        (" 12/2024 ", 2024, 12),
        ("Dec 99", 2099, 12),
    ],
)
def test_parse_month_year_accepts_month_picker_formats(text: str, year: int, month: int) -> None:
    assert parse_month_year(text) == MonthYear(year, month)


@pytest.mark.parametrize("text", ["", "March", "Foo 2027", "13/2027", "2027-13", "03/27", "2027-3"])
def test_parse_month_year_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_month_year(text)


def test_month_year_helpers_are_ordered_and_clamp_to_month_end() -> None:
    february = MonthYear(2024, 2)
    assert february.first_day() == date(2024, 2, 1)
    assert february.last_day() == date(2024, 2, 29)
    assert MonthYear(2024, 1).add_months(1) == february
    assert MonthYear(2024, 1).months_until(february) == 1
    assert MonthYear(2024, 1) < february
    assert add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2024, 3, 31), -1) == date(2024, 2, 29)


def test_indian_financial_year_boundaries() -> None:
    march_end = date(2026, 3, 31)
    april_start = date(2026, 4, 1)
    assert financial_year(march_end) == "2025-26"
    assert fy_start(march_end) == date(2025, 4, 1)
    assert fy_end(march_end) == march_end
    assert financial_year(april_start) == "2026-27"
    assert fy_start(april_start) == april_start
    assert fy_end(april_start) == date(2027, 3, 31)
