"""Code-owned valuation statistics and thresholds."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict

from wealth.research.factors import FundamentalFactors
from wealth.research.rules import ResearchRules


class ValuationStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str
    market: Literal["IN", "US"]
    price: Decimal | None
    earnings_yield_pct: Decimal | None
    sector_earnings_yield_median_pct: Decimal | None
    earnings_yield_vs_sector: Decimal | None
    pe_percentile: Decimal | None
    fcf_yield_pct: Decimal | None
    reverse_dcf_implied_growth_pct: Decimal | None
    discount_rate_pct: Decimal
    verdict: Literal["CHEAP", "FAIR", "EXPENSIVE", "UNKNOWN"]


def calculate_valuation_stats(
    *,
    security_id: str,
    market: str,
    price: Decimal | None,
    factors: FundamentalFactors,
    sector_earnings_yields: Sequence[Decimal],
    historical_pe: Sequence[Decimal],
    rules: ResearchRules,
) -> ValuationStats:
    """Compare earnings yield, own-history P/E, and FCF yield using Decimal math."""
    market_code = market.upper()
    if market_code not in {"IN", "US"}:
        raise ValueError("Research valuation supports India and US securities")
    rate_pct = rules.discount_rate(market_code)
    earnings_yield = factors.earnings_yield
    fcf_yield = factors.fcf_yield
    peer_median = Decimal(str(median(sector_earnings_yields))) if sector_earnings_yields else None
    relative_yield = (
        earnings_yield / peer_median
        if earnings_yield is not None and peer_median not in (None, Decimal(0))
        else None
    )
    current_pe = (
        Decimal(1) / earnings_yield if earnings_yield is not None and earnings_yield > 0 else None
    )
    pe_percentile = (
        Decimal(sum(value <= current_pe for value in historical_pe))
        / Decimal(len(historical_pe))
        * Decimal(100)
        if current_pe is not None and len(historical_pe) >= 36
        else None
    )
    if pe_percentile is not None:
        if pe_percentile <= rules.cheap_pe_percentile_max:
            verdict: Literal["CHEAP", "FAIR", "EXPENSIVE", "UNKNOWN"] = "CHEAP"
        elif pe_percentile >= rules.expensive_pe_percentile_min:
            verdict = "EXPENSIVE"
        else:
            verdict = "FAIR"
    elif relative_yield is not None and fcf_yield is not None:
        if relative_yield >= rules.cheap_earnings_yield_vs_sector and fcf_yield > 0:
            verdict = "CHEAP"
        elif relative_yield < rules.expensive_earnings_yield_vs_sector or fcf_yield < 0:
            verdict = "EXPENSIVE"
        else:
            verdict = "FAIR"
    else:
        verdict = "UNKNOWN"
    # Treat current trailing FCF per share as FCF1 for this intentionally simple reverse DCF.
    implied_growth_pct = rate_pct - fcf_yield * Decimal(100) if fcf_yield is not None else None
    return ValuationStats(
        security_id=security_id,
        market=market_code,
        price=price,
        earnings_yield_pct=(earnings_yield * Decimal(100) if earnings_yield is not None else None),
        sector_earnings_yield_median_pct=(
            peer_median * Decimal(100) if peer_median is not None else None
        ),
        earnings_yield_vs_sector=relative_yield,
        pe_percentile=pe_percentile,
        fcf_yield_pct=(fcf_yield * Decimal(100) if fcf_yield is not None else None),
        reverse_dcf_implied_growth_pct=implied_growth_pct,
        discount_rate_pct=rate_pct,
        verdict=verdict,
    )
