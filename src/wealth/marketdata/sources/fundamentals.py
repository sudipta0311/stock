"""Shared dated fundamental observations used by public-market adapters."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

# Neon free tier holds the whole database in 0.5 GB. Four years covers trailing-twelve-month
# values and two years of year-on-year comparisons for ~1,000 universe companies (~85 MB).
RETENTION_YEARS = 4


def retention_cutoff(as_of: date) -> date:
    """Return the earliest period end kept in the rolling fundamentals window."""
    year = as_of.year - RETENTION_YEARS
    return date(year, as_of.month, min(as_of.day, calendar.monthrange(year, as_of.month)[1]))


@dataclass(frozen=True)
class FundamentalObservation:
    metric: str
    period_end: date
    period_type: str
    value: Decimal
    unit: str
    filed_at: datetime
    known_at: datetime
    source: str
    source_url: str
