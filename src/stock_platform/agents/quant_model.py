from __future__ import annotations

from typing import Any

from stock_platform.utils.rules import clamp


def compute_quality_score(symbol: str, fin_data: dict[str, Any], _unused: dict[str, Any] | None = None) -> float:
    """
    Quality score based on standardized fundamentals.

    Returns 0.5 for fully unknown data and applies an extra cap for mapped/de-merged symbols
    when post-listing disclosures are still sparse.
    """
    if not fin_data:
        print(f"WARNING: No data for {symbol} - score = 0.5 (unknown)")
        return 0.5

    scores: list[float] = []
    weights: list[float] = []

    roce = fin_data.get("roce_pct")
    if roce is not None:
        scores.append(
            1.0 if roce > 18 else
            0.6 if roce > 10 else
            0.0 if roce > 0 else
            -0.5
        )
        weights.append(0.25)

    eps = fin_data.get("eps")
    if eps is not None:
        scores.append(1.0 if eps > 0 else 0.0)
        weights.append(0.25)

    revenue_growth = fin_data.get("revenue_growth_pct")
    if revenue_growth is not None:
        scores.append(
            1.0 if revenue_growth > 15 else
            0.7 if revenue_growth > 8 else
            0.3 if revenue_growth > 0 else
            0.0
        )
        weights.append(0.20)

    promoter = fin_data.get("promoter_holding")
    if promoter is not None:
        scores.append(
            1.0 if promoter > 50 else
            0.7 if promoter > 35 else
            0.3
        )
        weights.append(0.15)

    debt_to_equity = fin_data.get("debt_to_equity")
    if debt_to_equity is not None:
        scores.append(
            1.0 if debt_to_equity < 0.5 else
            0.5 if debt_to_equity < 1.0 else
            0.1 if debt_to_equity < 2.0 else
            0.0
        )
        weights.append(0.15)

    if not scores:
        print(f"WARNING: No usable data for {symbol} - score = 0.5 (unknown)")
        return 0.5

    raw_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
    final_score = round(clamp(raw_score, 0.0, 1.0), 2)

    if eps is not None and eps < 0:
        final_score = min(final_score, 0.35)
        print(f"{symbol}: negative EPS - capped quality at 0.35")

    critical_fields = sum(
        1
        for key in ("roce_pct", "revenue_growth_pct", "debt_to_equity")
        if fin_data.get(key) is not None
    )
    if fin_data.get("symbol_mapped") and critical_fields <= 1:
        final_score = min(final_score, 0.45)
        print(f"{symbol}: mapped to {fin_data.get('symbol')} with sparse post-demerger metrics - capped quality at 0.45")

    return round(final_score, 2)


# ── Entry signal vocabulary used in assess_timing() ──────────────────────────
_BULLISH_ENTRY_SIGNALS = {"STRONG ENTER", "ACCUMULATE", "SMALL INITIAL"}
_SOFTEN_MAP = {"STRONG ENTER": "ACCUMULATE", "ACCUMULATE": "SMALL INITIAL"}


def apply_freshness_cap(entry_signal: str, fin_data: dict[str, Any]) -> str:
    """
    Caps quant entry signal based on data freshness.
    Prevents overly bullish calls when earnings data is absent or price data is corrupt.

    entry_signal uses assess_timing() vocabulary:
        STRONG ENTER | ACCUMULATE | SMALL INITIAL | WAIT | DO NOT ENTER
    """
    if entry_signal not in _BULLISH_ENTRY_SIGNALS:
        return entry_signal

    result_date = fin_data.get("last_result_date")
    w52_quality = fin_data.get("52w_data_quality", "")

    # No earnings anchor — cannot sustain bullish call
    if not result_date or str(result_date).lower() in ("none", ""):
        print(
            f"Freshness cap: no result date - capping '{entry_signal}' to 'WAIT'"
        )
        return "WAIT"

    # Corrupt price data — soften but don't zero out
    if w52_quality in ("DATA_CORRUPT", "UNAVAILABLE"):
        softened = _SOFTEN_MAP.get(entry_signal, entry_signal)
        if softened != entry_signal:
            print(
                f"Freshness cap: 52W {w52_quality} - softening '{entry_signal}' to '{softened}'"
            )
        return softened

    return entry_signal
