"""Month-year and Indian financial-year date helpers."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

_MONTH_NAMES = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
_MONTH_NAMES.update(
    {name.lower(): number for number, name in enumerate(calendar.month_abbr) if name}
)


@dataclass(frozen=True, order=True)
class MonthYear:
    """A validated calendar year and month without a specific day."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if not isinstance(self.year, int) or isinstance(self.year, bool) or self.year < 1:
            raise ValueError("year must be a positive integer")
        if (
            not isinstance(self.month, int)
            or isinstance(self.month, bool)
            or not 1 <= self.month <= 12
        ):
            raise ValueError("month must be between 1 and 12")

    def first_day(self) -> date:
        return date(self.year, self.month, 1)

    def last_day(self) -> date:
        return date(self.year, self.month, calendar.monthrange(self.year, self.month)[1])

    def add_months(self, n: int) -> MonthYear:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("month offset must be an integer")
        total = self.year * 12 + self.month - 1 + n
        year, month_index = divmod(total, 12)
        if year < 1:
            raise ValueError("resulting year must be positive")
        return MonthYear(year, month_index + 1)

    def months_until(self, other: MonthYear) -> int:
        return (other.year - self.year) * 12 + other.month - self.month

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def parse_month_year(text: str) -> MonthYear:
    """Parse a month and year in a common month-picker format."""
    if not isinstance(text, str):
        raise ValueError("month-year must be text")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("month-year cannot be empty")

    patterns: tuple[tuple[str, str], ...] = (
        (r"(?P<month>\d{1,2})/(?P<year>\d{4})", "numeric"),
        (r"(?P<year>\d{4})-(?P<month>\d{2})", "numeric"),
        (r"(?P<month>[A-Za-z]+)[ -](?P<year>\d{4}|\d{2})", "named"),
    )
    for pattern, kind in patterns:
        match = re.fullmatch(pattern, cleaned)
        if not match:
            continue
        year_text = match.group("year")
        year = int(year_text)
        if kind == "named":
            month = _MONTH_NAMES.get(match.group("month").lower())
            if month is None:
                break
            if len(year_text) == 2:
                year += 2000
        else:
            month = int(match.group("month"))
        try:
            return MonthYear(year, month)
        except ValueError as error:
            raise ValueError(f"Invalid month-year: {text!r}") from error

    raise ValueError(f"Invalid month-year: {text!r}")


def financial_year(d: date) -> str:
    """Return the Indian financial year containing a date."""
    start_year = d.year if d.month >= 4 else d.year - 1
    return f"{start_year:04d}-{(start_year + 1) % 100:02d}"


def fy_start(d: date) -> date:
    """Return 1 April for the Indian financial year containing ``d``."""
    start_year = d.year if d.month >= 4 else d.year - 1
    return date(start_year, 4, 1)


def fy_end(d: date) -> date:
    """Return 31 March for the Indian financial year containing ``d``."""
    start_year = d.year if d.month >= 4 else d.year - 1
    return date(start_year + 1, 3, 31)


def add_months(d: date, n: int) -> date:
    """Add calendar months, clamping the day to the target month's end."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("month offset must be an integer")
    total = d.year * 12 + d.month - 1 + n
    year, month_index = divmod(total, 12)
    if year < 1:
        raise ValueError("resulting year must be positive")
    month = month_index + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
