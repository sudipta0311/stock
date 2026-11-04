"""Helpers for validating public financial identifiers."""

from __future__ import annotations

import re

_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_CUSIP_PATTERN = re.compile(r"^[0-9][A-Z0-9]{8}$")


def isin_check_digit_ok(isin: str) -> bool:
    """Return whether an ISIN has a valid format and ISO 6166 Luhn digit."""
    if not isinstance(isin, str):
        return False

    normalized = isin.strip().upper()
    if not _ISIN_PATTERN.fullmatch(normalized):
        return False

    expanded = "".join(
        character if character.isdigit() else str(ord(character) - ord("A") + 10)
        for character in normalized
    )
    checksum = 0
    double = False
    for character in reversed(expanded):
        digit = int(character) * (2 if double else 1)
        checksum += digit // 10 + digit % 10
        double = not double
    return checksum % 10 == 0


def isin_from_cusip(cusip: str | None, country: str = "US") -> str | None:
    """Convert a numeric-leading nine-character CUSIP to a checked ISIN."""
    if not isinstance(cusip, str):
        return None
    normalized_cusip = cusip.strip().upper()
    normalized_country = country.strip().upper() if isinstance(country, str) else ""
    if not _CUSIP_PATTERN.fullmatch(normalized_cusip) or not re.fullmatch(
        r"[A-Z]{2}", normalized_country
    ):
        return None
    prefix = normalized_country + normalized_cusip
    for digit in "0123456789":
        candidate = prefix + digit
        if isin_check_digit_ok(candidate):
            return candidate
    return None
