"""Money parsing, formatting, and percentage helpers for INR-first workflows."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Literal

_AMOUNT_PATTERN = re.compile(
    r"(?P<number>(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}))(?:\.(?P<fraction>\d+))?"
)
_SUFFIX_PATTERN = re.compile(
    r"\s*(?P<suffix>thousand|lakhs?|lac|cr|crores?|[kKlL])\s*$", re.IGNORECASE
)
_PAISE = Decimal("0.01")


def _decimal_places(value: Decimal) -> int:
    _, digits, exponent = value.as_tuple()
    while exponent < 0 and digits and digits[-1] == 0:
        digits = digits[:-1]
        exponent += 1
    return max(0, -exponent)


def _quantize_paise(value: Decimal) -> Decimal:
    digits = len(value.as_tuple().digits)
    integer_digits = max(1, value.adjusted() + 1)
    with localcontext() as context:
        context.prec = max(context.prec, digits + 2, integer_digits + 2)
        return value.quantize(_PAISE, rounding=ROUND_HALF_UP)


def _shift_decimal(value: Decimal, power: int) -> Decimal:
    sign, digits, exponent = value.as_tuple()
    return Decimal((sign, digits, exponent + power))


def parse_inr(text: str) -> Decimal:
    """Parse a rupee amount, including common Indian shorthand, exactly."""
    if not isinstance(text, str):
        raise ValueError("INR amount must be text")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("INR amount cannot be empty")
    if cleaned.startswith("-"):
        raise ValueError("INR amount cannot be negative")

    cleaned = re.sub(r"^(?:₹|rs\.?|inr)\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    if cleaned.startswith("-"):
        raise ValueError("INR amount cannot be negative")
    suffix_power = 0
    suffix_match = _SUFFIX_PATTERN.search(cleaned)
    if suffix_match:
        suffix_name = suffix_match.group("suffix").lower()
        cleaned = cleaned[: suffix_match.start()].rstrip()
        if suffix_name in {"k", "thousand"}:
            suffix_power = 3
        elif suffix_name in {"l", "lac", "lakh", "lakhs"}:
            suffix_power = 5
        else:
            suffix_power = 7

    match = _AMOUNT_PATTERN.fullmatch(cleaned)
    if match is None:
        raise ValueError(f"Invalid INR amount: {text!r}")

    number = match.group("number").replace(",", "")
    fraction = match.group("fraction")
    try:
        parsed = Decimal(f"{number}.{fraction}" if fraction is not None else number)
        sign, digits, exponent = parsed.as_tuple()
        amount = Decimal((sign, digits, exponent + suffix_power))
    except InvalidOperation as error:
        raise ValueError(f"Invalid INR amount: {text!r}") from error

    if not amount.is_finite():
        raise ValueError("INR amount must be finite")
    if _decimal_places(amount) > 2:
        raise ValueError("INR amount cannot have more than 2 decimal places after scaling")
    return _quantize_paise(amount)


def _group_indian(integer: str) -> str:
    if len(integer) <= 3:
        return integer
    last_three = integer[-3:]
    leading = integer[:-3]
    groups: list[str] = []
    while leading:
        groups.append(leading[-2:])
        leading = leading[:-2]
    return ",".join(reversed(groups)) + "," + last_three


def _rounded(value: Decimal, decimals: int) -> Decimal:
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    quantum = Decimal(1).scaleb(-decimals)
    try:
        with localcontext() as context:
            context.prec = max(context.prec, len(value.as_tuple().digits) + decimals + 2)
            return value.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation as error:
        raise ValueError("amount cannot be formatted at the requested precision") from error


def _format_number(
    value: Decimal, decimals: int, grouping: Literal["indian", "international"]
) -> str:
    rounded = _rounded(abs(value), decimals)
    whole, _, fraction = f"{rounded:.{decimals}f}".partition(".")
    grouped = _group_indian(whole) if grouping == "indian" else f"{int(whole):,}"
    return f"{grouped}.{fraction}" if decimals else grouped


def format_inr(
    amount: Decimal, style: Literal["full", "compact"] = "full", decimals: int = 2
) -> str:
    """Format rupees using Indian digit grouping and half-up rounding."""
    if not isinstance(amount, Decimal) or not amount.is_finite():
        raise ValueError("amount must be a finite Decimal")
    if style not in ("full", "compact"):
        raise ValueError("style must be 'full' or 'compact'")

    sign = "-" if amount < 0 else ""
    magnitude = abs(amount)
    if style == "full":
        return f"{sign}₹{_format_number(magnitude, decimals, 'indian')}"

    rounded_rupees = _rounded(magnitude, 0)
    if rounded_rupees < Decimal(100_000):
        return f"{sign}₹{_format_number(rounded_rupees, 0, 'indian')}"

    lakhs = _rounded(_shift_decimal(magnitude, -5), 2)
    if lakhs < Decimal(100):
        number = _trim_decimal(lakhs)
        return f"{sign}₹{number} L"

    crores = _rounded(_shift_decimal(magnitude, -7), 2)
    number = _trim_decimal(crores)
    return f"{sign}₹{number} Cr"


def _trim_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def format_usd(amount: Decimal, decimals: int = 2) -> str:
    """Format a USD amount using international digit grouping."""
    if not isinstance(amount, Decimal) or not amount.is_finite():
        raise ValueError("amount must be a finite Decimal")
    sign = "-" if amount < 0 else ""
    return f"{sign}${_format_number(amount, decimals, 'international')}"


def round_paise(amount: Decimal) -> Decimal:
    """Round an amount to paise using ROUND_HALF_UP."""
    if not isinstance(amount, Decimal) or not amount.is_finite():
        raise ValueError("amount must be a finite Decimal")
    with localcontext() as context:
        context.prec = max(context.prec, len(amount.as_tuple().digits) + 4)
        return amount.quantize(_PAISE, rounding=ROUND_HALF_UP)


def pct(value: str | int | Decimal) -> Decimal:
    """Convert a percentage to Decimal and require a value from 0 through 100."""
    if isinstance(value, (bool, float)):
        raise ValueError("percentage must be text, an integer, or Decimal")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("percentage must be a valid decimal number") from error
    if not result.is_finite() or not Decimal(0) <= result <= Decimal(100):
        raise ValueError("percentage must be between 0 and 100")
    return result
