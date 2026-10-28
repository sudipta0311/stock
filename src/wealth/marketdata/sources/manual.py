"""Manual gold-price fallback loader."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from wealth.marketdata.sources.ibja import GoldObservation

MANUAL_PRICES_PATH = Path(__file__).resolve().parents[4] / "rules" / "manual_prices.yaml"


def load_manual_gold_price(path: Path = MANUAL_PRICES_PATH) -> GoldObservation | None:
    """Load a configured manual 24k price, or return none when no value was configured."""
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        return None
    entry = payload.get("gold_24k_per_gram")
    if isinstance(entry, dict):
        value = entry.get("value")
        as_of = entry.get("as_of")
        source = entry.get("source", "manual")
    else:
        value = entry
        as_of = payload.get("as_of")
        source = payload.get("source", "manual")
    if value is None or as_of is None or source != "manual":
        return None
    try:
        price = Decimal(str(value))
        effective_date = date.fromisoformat(str(as_of))
        if not price.is_finite() or price <= 0:
            return None
    except (InvalidOperation, ValueError):
        return None
    return GoldObservation(price, effective_date, "rules/manual_prices.yaml")
