"""Code-owned entry timing hints for weekly research ideas."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, TypedDict

from wealth.policy.actions import load_buying_rules
from wealth.policy.rules import decimal_rule, integer_rule


class EntryHint(TypedDict):
    code: Literal["buy_in_parts", "wait"]
    parts: int | None
    reasons: list[str]


def _percentile(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() and Decimal(0) <= parsed <= Decimal(100) else None


def entry_hint(
    *,
    market: str,
    dist_from_200dma: Decimal | None,
    market_percentile: object,
    market_history_label: str | None,
    rules: Mapping[str, Any] | None = None,
) -> EntryHint:
    """Apply stored price and market history to a short, deterministic buy hint."""
    selected_rules = load_buying_rules() if rules is None else rules
    hint_rules = selected_rules["entry_hint"]
    distance_limit = decimal_rule(
        hint_rules["wait_above_200dma_pct"], "entry_hint.wait_above_200dma_pct"
    )
    wait_percentile = decimal_rule(
        hint_rules["wait_market_percentile"], "entry_hint.wait_market_percentile"
    )
    more_parts_percentile = decimal_rule(
        hint_rules["more_parts_market_percentile"], "entry_hint.more_parts_market_percentile"
    )
    selected_market = market.upper()
    reasons: list[str] = []
    wait = False

    if selected_market == "US":
        reasons.append("us_price_history_unavailable")
    elif dist_from_200dma is None:
        reasons.append("price_history_unavailable")
    elif dist_from_200dma * Decimal(100) > distance_limit:
        reasons.append("price_far_above_average")
        wait = True

    if market_history_label != "long_history":
        percentile = None
        reasons.append("market_history_short")
    else:
        percentile = _percentile(market_percentile)
        if percentile is None:
            reasons.append("market_valuation_unavailable")
        elif percentile >= wait_percentile:
            reasons.append("market_valuation_high")
            wait = True

    if wait:
        return {"code": "wait", "parts": None, "reasons": reasons}

    if percentile is not None and percentile >= more_parts_percentile:
        reasons.append("market_valuation_elevated")
        parts = integer_rule(
            selected_rules["lumpsum"]["stp_tranches_expensive"],
            "lumpsum.stp_tranches_expensive",
        )
    else:
        reasons.append("regular_entry_schedule")
        parts = integer_rule(selected_rules["satellite"]["tranches"], "satellite.tranches")
    return {"code": "buy_in_parts", "parts": parts, "reasons": reasons}
