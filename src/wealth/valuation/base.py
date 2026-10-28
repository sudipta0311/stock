"""Shared result type for deterministic asset valuation."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from wealth.schemas.common import ValuationMethod


@dataclass(frozen=True)
class Valuation:
    """A current valuation and, when known, its scheduled maturity value."""

    value_today: Decimal
    maturity_value: Decimal | None
    maturity_date: date | None
    accrued_interest: Decimal
    principal_so_far: Decimal
    method: ValuationMethod
    notes: tuple[str, ...]
