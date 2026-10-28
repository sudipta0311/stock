"""Combine indicative market yields with dated small-savings reference rates."""

from __future__ import annotations

from contextlib import suppress
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from wealth.policy.rules import decimal_rule, integer_rule
from wealth.rates.reference import RateUnavailable, load_rates

_YIELDS_PATH = Path(__file__).resolve().parents[3] / "rules" / "debt_yields.yaml"
_SCHEME_YIELDS = {
    "po_td_1y": "td_1y",
    "po_td_2y": "td_2y",
    "po_td_3y": "td_3y",
    "po_td_5y": "td_5y",
    "nsc": "nsc",
    "ppf": "ppf",
    "scss": "scss",
    "pomis": "mis",
}


@lru_cache(maxsize=1)
def _yield_table() -> dict[str, Any]:
    with _YIELDS_PATH.open(encoding="utf-8") as source:
        raw = yaml.safe_load(source)
    if not isinstance(raw, dict) or integer_rule(raw.get("version"), "version") != 1:
        raise ValueError("unsupported debt-yield rules")
    if not isinstance(raw.get("as_of"), str) or not isinstance(raw.get("verified"), bool):
        raise ValueError("debt-yield rules require as_of and verified fields")
    date.fromisoformat(raw["as_of"])
    yields = raw.get("yields_pct")
    if not isinstance(yields, dict) or not yields:
        raise ValueError("debt-yield rules require a yields_pct mapping")
    for key, value in yields.items():
        if not isinstance(key, str) or not key:
            raise ValueError("debt-yield keys must be non-empty text")
        if decimal_rule(value, f"yields_pct.{key}") < 0:
            raise ValueError(f"yields_pct.{key} must be non-negative")
    return raw


def load_debt_yields(on: date) -> tuple[dict[str, Decimal], date, bool]:
    """Return effective small-savings rates and the unverified yield table."""
    if not isinstance(on, date):
        raise ValueError("on must be a date")
    table = _yield_table()
    yields = {
        key: decimal_rule(value, f"yields_pct.{key}") for key, value in table["yields_pct"].items()
    }
    rate_book = load_rates()
    for yield_key, scheme in _SCHEME_YIELDS.items():
        try:
            yields[yield_key] = rate_book.rate_for(scheme, on).rate_pct
        except RateUnavailable:
            continue

    fy_start = on.year if on.month >= 4 else on.year - 1
    financial_year = f"{fy_start}-{(fy_start + 1) % 100:02d}"
    with suppress(RateUnavailable):
        yields["vpf"] = rate_book.epf_rate(financial_year).rate_pct

    return yields, date.fromisoformat(table["as_of"]), table["verified"]
