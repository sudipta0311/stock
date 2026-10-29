"""Shared enums and validated field types for domain schemas."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator, Field, PlainSerializer, WithJsonSchema

from wealth.core.dates import MonthYear as MonthYearValue
from wealth.core.dates import parse_month_year
from wealth.core.money import parse_inr, round_paise

_RUPEES_JSON_SCHEMA = {
    "type": "string",
    "pattern": r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$",
    "format": "decimal",
    "x-unit": "INR",
}
_DANISH_KRONER_JSON_SCHEMA = {
    "type": "string",
    "pattern": r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$",
    "format": "decimal",
    "x-unit": "DKK",
}
_CURRENCY_AMOUNT_JSON_SCHEMA = {
    "type": "string",
    "pattern": r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$",
    "format": "decimal",
}
_RATE_PCT_JSON_SCHEMA = {
    "type": "string",
    "pattern": r"^(0|[1-9][0-9]?)(\.[0-9]{1,4})?$",
    "format": "decimal",
    "x-minimum": "0",
    "x-maximum": "30",
}
_LOAN_RATE_PCT_JSON_SCHEMA = {
    "type": "string",
    "pattern": r"^(0|[1-9][0-9]?)(\.[0-9]{1,4})?$",
    "format": "decimal",
    "x-minimum": "0",
    "x-maximum": "60",
}
_PERCENTAGE_JSON_SCHEMA = {
    "type": "string",
    "pattern": r"^(0|[1-9][0-9]?)(\.[0-9]{1,4})?$|^100(\.0{1,4})?$",
    "format": "decimal",
    "x-minimum": "0",
    "x-maximum": "100",
}


class _CaseInsensitiveStrEnum(StrEnum):
    @classmethod
    def _missing_(cls, value: object) -> _CaseInsensitiveStrEnum | None:
        if isinstance(value, str):
            normalized = value.lower()
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class AssetType(_CaseInsensitiveStrEnum):
    SAVINGS = "savings"
    FD = "fd"
    RD = "rd"
    PPF = "ppf"
    EPF = "epf"
    NPS = "nps"
    PO_SCHEME = "po_scheme"
    BOND = "bond"
    GOLD = "gold"
    TERM_INSURANCE = "term_insurance"
    HEALTH_INSURANCE = "health_insurance"
    TRADITIONAL_POLICY = "traditional_policy"
    ULIP = "ulip"
    PROPERTY = "property"
    LOAN = "loan"


class Sleeve(_CaseInsensitiveStrEnum):
    EQUITY_INDIA = "equity_india"
    EQUITY_GLOBAL = "equity_global"
    DEBT = "debt"
    GOLD = "gold"
    CASH = "cash"
    REAL_ESTATE = "real_estate"
    PROTECTION = "protection"


class LiquidityClass(_CaseInsensitiveStrEnum):
    INSTANT = "instant"
    WITHIN_1Y = "within_1y"
    LOCKED = "locked"


class ValuationMethod(_CaseInsensitiveStrEnum):
    FORMULA = "formula"
    USER = "user"
    STATEMENT = "statement"
    MARKET = "market"


class Compounding(_CaseInsensitiveStrEnum):
    SIMPLE = "simple"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    HALF_YEARLY = "half_yearly"
    YEARLY = "yearly"


class PayoutType(_CaseInsensitiveStrEnum):
    CUMULATIVE = "cumulative"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    HALF_YEARLY = "half_yearly"
    YEARLY = "yearly"


def _strict_decimal(value: Any) -> Decimal:
    if isinstance(value, (float, bool)):
        raise ValueError("decimal values must not be supplied as floats or booleans")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("value must be a valid decimal amount") from error
    if not result.is_finite():
        raise ValueError("value must be finite")
    return Decimal(format(result, "f"))


def _rupees_decimal(value: Any) -> Decimal:
    if isinstance(value, str):
        try:
            return parse_inr(value)
        except ValueError as error:
            raise ValueError("value must be a valid INR amount") from error
    return _strict_decimal(value)


def _currency_decimal(value: Any) -> Decimal:
    return _strict_decimal(value)


def _at_most_two_places(value: Decimal) -> Decimal:
    if _decimal_places(value) > 2:
        raise ValueError("amount cannot have more than 2 decimal places")
    return round_paise(value)


def _at_most_four_places(value: Decimal) -> Decimal:
    if max(0, -value.as_tuple().exponent) > 4:
        raise ValueError("rate cannot have more than 4 decimal places")
    return Decimal(format(value, "f"))


def _decimal_places(value: Decimal) -> int:
    _, digits, exponent = value.as_tuple()
    while exponent < 0 and digits and digits[-1] == 0:
        digits = digits[:-1]
        exponent += 1
    return max(0, -exponent)


def _format_rupees(value: Decimal) -> str:
    return format(value, ".2f")


def _format_decimal(value: Decimal) -> str:
    return format(value, "f")


def _month_year(value: Any) -> MonthYearValue:
    if isinstance(value, MonthYearValue):
        return value
    if isinstance(value, str):
        return parse_month_year(value)
    raise ValueError("month-year must be a MonthYear or a supported month-year string")


Rupees = Annotated[
    Decimal,
    BeforeValidator(_rupees_decimal),
    AfterValidator(_at_most_two_places),
    Field(ge=Decimal("0")),
    PlainSerializer(_format_rupees, return_type=str, when_used="json"),
    WithJsonSchema(_RUPEES_JSON_SCHEMA, mode="validation"),
    WithJsonSchema(_RUPEES_JSON_SCHEMA, mode="serialization"),
]
CurrencyAmount = Annotated[
    Decimal,
    BeforeValidator(_currency_decimal),
    AfterValidator(_at_most_two_places),
    Field(ge=Decimal("0")),
    PlainSerializer(_format_rupees, return_type=str, when_used="json"),
    WithJsonSchema(_CURRENCY_AMOUNT_JSON_SCHEMA, mode="validation"),
    WithJsonSchema(_CURRENCY_AMOUNT_JSON_SCHEMA, mode="serialization"),
]
DanishKroner = Annotated[
    Decimal,
    BeforeValidator(_strict_decimal),
    AfterValidator(_at_most_two_places),
    Field(ge=Decimal("0")),
    PlainSerializer(_format_rupees, return_type=str, when_used="json"),
    WithJsonSchema(_DANISH_KRONER_JSON_SCHEMA, mode="validation"),
    WithJsonSchema(_DANISH_KRONER_JSON_SCHEMA, mode="serialization"),
]
RatePct = Annotated[
    Decimal,
    BeforeValidator(_strict_decimal),
    AfterValidator(_at_most_four_places),
    Field(ge=Decimal("0"), le=Decimal("30")),
    PlainSerializer(_format_decimal, return_type=str, when_used="json"),
    WithJsonSchema(_RATE_PCT_JSON_SCHEMA, mode="validation"),
    WithJsonSchema(_RATE_PCT_JSON_SCHEMA, mode="serialization"),
]
LoanRatePct = Annotated[
    Decimal,
    BeforeValidator(_strict_decimal),
    AfterValidator(_at_most_four_places),
    Field(ge=Decimal("0"), le=Decimal("60")),
    PlainSerializer(_format_decimal, return_type=str, when_used="json"),
    WithJsonSchema(_LOAN_RATE_PCT_JSON_SCHEMA, mode="validation"),
    WithJsonSchema(_LOAN_RATE_PCT_JSON_SCHEMA, mode="serialization"),
]
Percentage = Annotated[
    Decimal,
    BeforeValidator(_strict_decimal),
    AfterValidator(_at_most_four_places),
    Field(ge=Decimal("0"), le=Decimal("100")),
    PlainSerializer(_format_decimal, return_type=str, when_used="json"),
    WithJsonSchema(_PERCENTAGE_JSON_SCHEMA, mode="validation"),
    WithJsonSchema(_PERCENTAGE_JSON_SCHEMA, mode="serialization"),
]
MonthYear = Annotated[
    MonthYearValue,
    BeforeValidator(_month_year),
    PlainSerializer(str, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string", "pattern": r"^\d{4}-(0[1-9]|1[0-2])$"}, mode="validation"),
    WithJsonSchema({"type": "string", "pattern": r"^\d{4}-(0[1-9]|1[0-2])$"}, mode="serialization"),
]
