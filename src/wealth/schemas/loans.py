"""Input models shared by loan event and prepayment endpoints."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wealth.schemas.common import CurrencyAmount, LoanRatePct

LoanEventKind = Literal[
    "prepayment", "rate_change", "emi_change", "balance_update", "missed_emi", "closed"
]


class LoanEventWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: LoanEventKind
    on_date: date
    amount: CurrencyAmount | None = None
    rate_pct: LoanRatePct | None = None
    emi: CurrencyAmount | None = None
    note: str | None = Field(default=None, max_length=140)

    @model_validator(mode="after")
    def validate_event_fields(self) -> LoanEventWrite:
        if self.kind == "prepayment" and (self.amount is None or self.amount <= 0):
            raise ValueError("a prepayment needs an amount above zero")
        if self.kind == "balance_update" and self.amount is None:
            raise ValueError("a balance update needs an amount")
        if self.kind == "rate_change" and self.rate_pct is None:
            raise ValueError("a rate change needs a new rate")
        if self.kind == "emi_change" and self.emi is None:
            raise ValueError("a monthly payment change needs a new amount")
        if self.note is not None and any(ord(character) < 32 for character in self.note):
            raise ValueError("note must be plain text on one line")
        return self


class PrepayPreviewWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: CurrencyAmount = Field(gt=Decimal(0))
