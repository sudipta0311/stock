"""FRED's public DEXINUS CSV parser and adapter."""

from __future__ import annotations

import csv
import io
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

from wealth.marketdata.sources.http import get

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS"
FRED_SOURCE = "FRED"
ECB_SOURCE = "ECB (EUR cross)"


@dataclass(frozen=True)
class FxObservation:
    pair: str
    rate: Decimal
    as_of: date
    publication_lag_days: int | None
    source: str = FRED_SOURCE
    source_url: str = FRED_CSV_URL


def parse_dexinus_csv(text: str, *, fetched_on: date | None = None) -> list[FxObservation]:
    """Parse FRED's date,value CSV, skipping missing observations and malformed rows."""
    if not text.strip():
        return []
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    headers = {header.strip().casefold() for header in reader.fieldnames or []}
    if not {"observation_date", "dexinus"}.issubset(headers):
        warnings.warn(
            "FRED CSV has an unrecognized header; rows were skipped", RuntimeWarning, stacklevel=2
        )
        return []
    rows: list[FxObservation] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            date_text = row.get("observation_date") or row.get("observation_date".upper())
            rate_text = row.get("DEXINUS") or row.get("dexinus")
            if not date_text or not rate_text:
                warnings.warn(f"FRED row {line_number} was skipped", RuntimeWarning, stacklevel=2)
                continue
            if rate_text.strip() == ".":
                continue
            as_of = date.fromisoformat(date_text.strip())
            rate = Decimal(rate_text.strip())
            if not rate.is_finite() or rate <= 0:
                raise ValueError("invalid rate")
            lag = max(0, (fetched_on - as_of).days) if fetched_on else None
            rows.append(FxObservation("USD/INR", rate, as_of, lag))
        except (InvalidOperation, ValueError):
            warnings.warn(f"FRED row {line_number} was skipped", RuntimeWarning, stacklevel=2)
    return rows


def fetch_dexinus(*, fetched_on: date | None = None) -> list[FxObservation]:
    """Fetch the public FRED DEXINUS observation series."""
    effective_fetch_date = fetched_on or datetime.now(UTC).date()
    return parse_dexinus_csv(get(FRED_CSV_URL).text, fetched_on=effective_fetch_date)
