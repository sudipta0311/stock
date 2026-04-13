from __future__ import annotations

from typing import Any


def compute_technical_signal(
    symbol: str,
    fin_data: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """
    Compute technical entry signals from fundamental and price data.
    These are medium-term signals suitable for 18-30 month investment horizon.
    NOT short-term trading indicators.
    """
    signals: list[dict[str, Any]] = []

    # Signal 1: 200 DMA position
    dma_200 = fin_data.get("dma_200")
    if dma_200 and current_price:
        pct_above_200dma = (current_price - dma_200) / dma_200 * 100
        if pct_above_200dma > 20:
            signals.append({
                "type":   "200DMA",
                "value":  f"+{pct_above_200dma:.1f}%",
                "signal": "CAUTION",
                "note":   f"Extended {pct_above_200dma:.1f}% above 200DMA — wait for pullback",
            })
        elif pct_above_200dma > 0:
            signals.append({
                "type":   "200DMA",
                "value":  f"+{pct_above_200dma:.1f}%",
                "signal": "POSITIVE",
                "note":   "Trading above 200DMA — uptrend intact",
            })
        elif pct_above_200dma > -10:
            signals.append({
                "type":   "200DMA",
                "value":  f"{pct_above_200dma:.1f}%",
                "signal": "WATCH",
                "note":   "Near 200DMA support — potential entry zone",
            })
        else:
            signals.append({
                "type":   "200DMA",
                "value":  f"{pct_above_200dma:.1f}%",
                "signal": "NEGATIVE",
                "note":   "Below 200DMA — downtrend, wait for recovery",
            })

    # Signal 2: 52-week low proximity
    week52_low = fin_data.get("week52_low") or fin_data.get("fiftyTwoWeekLow")
    week52_high = fin_data.get("week52_high") or fin_data.get("fiftyTwoWeekHigh")
    if week52_low and week52_high and current_price:
        pct_from_low = (current_price - week52_low) / week52_low * 100
        pct_from_high = (week52_high - current_price) / week52_high * 100

        if pct_from_low <= 15:
            signals.append({
                "type":   "52W_RANGE",
                "value":  f"{pct_from_low:.1f}% from low",
                "signal": "CONTRARIAN_BUY",
                "note":   (
                    f"Only {pct_from_low:.1f}% above 52-week low — "
                    "contrarian entry zone if thesis intact"
                ),
            })
        elif pct_from_high <= 5:
            signals.append({
                "type":   "52W_RANGE",
                "value":  f"{pct_from_high:.1f}% from high",
                "signal": "CAUTION",
                "note":   "Near 52-week high — momentum entry, not value entry",
            })

    # Signal 3: Promoter holding change
    promoter_change = fin_data.get("promoter_change")
    if promoter_change is not None:
        if promoter_change > 1.0:
            signals.append({
                "type":   "PROMOTER",
                "value":  f"+{promoter_change:.1f}%",
                "signal": "POSITIVE",
                "note":   (
                    f"Promoter increased holding by {promoter_change:.1f}% "
                    "last quarter — insider confidence signal"
                ),
            })
        elif promoter_change < -2.0:
            signals.append({
                "type":   "PROMOTER",
                "value":  f"{promoter_change:.1f}%",
                "signal": "NEGATIVE",
                "note":   (
                    f"Promoter reduced holding by {abs(promoter_change):.1f}% "
                    "— insider selling flag"
                ),
            })

    # Aggregate technical score
    signal_scores = {
        "CONTRARIAN_BUY": 1.0,
        "POSITIVE":       0.7,
        "WATCH":          0.5,
        "NEUTRAL":        0.5,
        "CAUTION":        0.3,
        "NEGATIVE":       0.1,
    }
    if signals:
        avg_score = sum(
            signal_scores.get(s["signal"], 0.5) for s in signals
        ) / len(signals)
    else:
        avg_score = 0.5

    return {
        "symbol":          symbol,
        "signals":         signals,
        "technical_score": round(avg_score, 2),
        "signal_count":    len(signals),
    }
