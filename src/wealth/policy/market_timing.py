"""Pure mapping from stored US market percentiles to the plan's buying guidance."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

ValuationLabel = Literal["cheap", "fair", "expensive"]


@dataclass(frozen=True)
class UsMarketTiming:
    percentile: Decimal | None
    history_label: str
    label: ValuationLabel | None
    extra_money_parts: int


def valuation_label(
    percentile: Decimal,
    *,
    cheap_pe_percentile_max: Decimal,
    expensive_pe_percentile_min: Decimal,
) -> ValuationLabel:
    """Classify a percentile using the research rulebook's boundaries."""
    if percentile <= cheap_pe_percentile_max:
        return "cheap"
    if percentile >= expensive_pe_percentile_min:
        return "expensive"
    return "fair"


def map_us_market_timing(
    percentile: Decimal | None,
    history_label: str,
    *,
    cheap_pe_percentile_max: Decimal,
    expensive_pe_percentile_min: Decimal,
    expensive_valuation_percentile: Decimal,
    stp_tranches_normal: int,
    stp_tranches_expensive: int,
) -> UsMarketTiming:
    """Use only long-history percentiles to label the market or change extra-buy parts."""
    usable_percentile = percentile if history_label == "long_history" else None
    label = (
        valuation_label(
            usable_percentile,
            cheap_pe_percentile_max=cheap_pe_percentile_max,
            expensive_pe_percentile_min=expensive_pe_percentile_min,
        )
        if usable_percentile is not None
        else None
    )
    extra_money_parts = (
        stp_tranches_expensive
        if usable_percentile is not None and usable_percentile >= expensive_valuation_percentile
        else stp_tranches_normal
    )
    return UsMarketTiming(
        percentile=usable_percentile,
        history_label=history_label,
        label=label,
        extra_money_parts=extra_money_parts,
    )
