"""Shared, privacy-safe helpers for CAS text parsers."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from wealth.core.ids import isin_check_digit_ok

_ISIN = re.compile(r"(?<![A-Z0-9])([A-Z]{2}[A-Z0-9]{9}[0-9])(?![A-Z0-9])", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d[\d,]*)(?:\.\d+)?")
_TRAILING_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d[\d,]*)(?:\.\d+)?(?![A-Za-z0-9])")
_DATE = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})[-/]?(?P<month>[A-Za-z]{3})[-/](?P<year>\d{4})(?!\d)",
    re.I,
)
_ISO_DATE = re.compile(r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?!\d)")
_PERIOD = re.compile(
    r"(?:period\s+from|period\s*:?)\s*(\d{1,2}[-/]?[A-Za-z]{3}[-/]\d{4})"
    r"\s+(?:to|through)\s+(\d{1,2}[-/]?[A-Za-z]{3}[-/]\d{4})",
    re.I,
)


class ParseError(ValueError):
    """A stable parser error that never includes document content or passwords."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail or _DEFAULT_DETAILS.get(code, "The statement could not be parsed.")
        super().__init__(self.detail)


_DEFAULT_DETAILS = {
    "wrong_password": "The PDF password is incorrect.",
    "not_a_pdf": "The uploaded file is not a readable PDF.",
    "too_large": "The uploaded PDF exceeds the size limit.",
    "unsupported_format": "This statement format is not supported.",
}


def clean_lines(lines: list[str]) -> list[str]:
    """Trim blank lines while preserving the order of the statement text."""
    return [line.strip() for line in lines if isinstance(line, str) and line.strip()]


def parse_statement_date(lines: list[str]) -> date | None:
    """Read the statement's period end or an explicitly labelled as-of date."""
    for line in lines:
        period = _PERIOD.search(line)
        if period:
            parsed = _parse_date(period.group(2))
            if parsed is not None:
                return parsed

    for line in lines:
        if not re.search(
            r"\b(statement\s+date|statement\s+as\s+of|as\s+on|valuation\s+as\s+on)\b", line, re.I
        ):
            continue
        parsed = _date_in_text(line)
        if parsed is not None:
            return parsed

    return None


def parse_date_text(value: str) -> date | None:
    """Parse a date token in the common CAS formats."""
    return _parse_date(value)


def _date_in_text(value: str) -> date | None:
    match = _DATE.search(value)
    if match is not None:
        return _make_date(int(match.group("day")), match.group("month"), int(match.group("year")))
    iso = _ISO_DATE.search(value)
    if iso is not None:
        try:
            return date(int(iso.group("year")), int(iso.group("month")), int(iso.group("day")))
        except ValueError:
            return None
    return None


def _parse_date(value: str) -> date | None:
    cleaned = value.strip()
    parsed = _date_in_text(cleaned)
    if parsed is not None:
        return parsed
    try:
        return datetime.strptime(cleaned, "%d-%b-%Y").date()
    except ValueError:
        return None


def _make_date(day: int, month_name: str, year: int) -> date | None:
    try:
        month = datetime.strptime(month_name[:3].title(), "%b").month
        return date(year, month, day)
    except ValueError:
        return None


def decimal_value(value: str) -> Decimal | None:
    """Parse an exact decimal token with Indian comma grouping."""
    cleaned = value.strip().replace(",", "")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    try:
        number = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def numeric_tokens(value: str) -> list[Decimal]:
    """Return finite numeric tokens from one table row."""
    result: list[Decimal] = []
    for match in _NUMBER.finditer(value):
        number = decimal_value(match.group())
        if number is not None:
            result.append(number)
    return result


def isin_from_line(value: str) -> str | None:
    """Find an ISIN-shaped token without deciding whether its check digit is valid."""
    match = _ISIN.search(value)
    return match.group(1).upper() if match is not None else None


def isin_warning(isin: str) -> str | None:
    return None if isin_check_digit_ok(isin) else "bad_isin"


def mask_account_ref(value: str | None) -> str | None:
    """Keep the last four alphanumeric characters and discard the rest."""
    if not value:
        return None
    tail = re.sub(r"[^A-Za-z0-9]", "", value)[-4:].upper()
    return f"XXXX{tail}" if len(tail) == 4 else None


def normalized_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t:-")


def clean_security_name(value: str, *, trailing_numeric_fields: int = 0) -> str:
    """Remove known trailing table columns and embedded ISINs from a security name."""
    if trailing_numeric_fields > 0:
        matches = list(_TRAILING_NUMBER.finditer(value))
        if len(matches) >= trailing_numeric_fields and not value[matches[-1].end() :].strip(
            " \t.,"
        ):
            value = value[: matches[-trailing_numeric_fields].start()]
    value = _ISIN.sub(" ", value)
    return normalized_name(value)
