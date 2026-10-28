"""Validated, normalised output from statement parsers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from wealth.schemas.common import Rupees

StatementSource = Literal["nsdl", "cdsl", "cams_kfin"]
HoldingKind = Literal["equity", "etf", "mf", "bond", "sgb", "other"]
TransactionKind = Literal["buy", "sell", "switch_in", "switch_out", "dividend_reinvest", "other"]


def _decimal_value(value: object) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError("decimal values must not be floats or booleans")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("value must be a decimal number") from error
    if not result.is_finite() or result < 0:
        raise ValueError("value must be a finite, non-negative decimal")
    return result


NonNegativeDecimal = Annotated[Decimal, BeforeValidator(_decimal_value)]


class _ImporterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ParsedHolding(_ImporterModel):
    """One holding as reported by a statement; invalid ISINs remain reviewable."""

    kind: HoldingKind
    isin: str = Field(min_length=1)
    name: str = Field(min_length=1)
    quantity: NonNegativeDecimal
    market_value: Rupees | None = None
    price: NonNegativeDecimal | None = None
    cost_value: Rupees | None = None
    account_ref: str | None = Field(default=None, pattern=r"^XXXX[A-Z0-9]{4}$")
    amc: str | None = None
    registrar: str | None = None


class ParsedTransaction(_ImporterModel):
    """A transaction row read from a detailed mutual-fund statement."""

    isin: str = Field(min_length=1)
    date: date
    kind: TransactionKind
    units: NonNegativeDecimal
    amount: Rupees
    nav: NonNegativeDecimal
    account_ref: str | None = Field(default=None, pattern=r"^XXXX[A-Z0-9]{4}$")


class ParsedStatement(_ImporterModel):
    """Holdings and transactions extracted from a single CAS statement."""

    source: StatementSource
    statement_date: date
    holdings: list[ParsedHolding]
    transactions: list[ParsedTransaction]
    warnings: list[str] = Field(default_factory=list)
