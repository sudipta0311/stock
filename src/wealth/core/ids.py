"""Helpers for validating public financial identifiers."""

from __future__ import annotations

import re

_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


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
