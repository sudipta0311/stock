from __future__ import annotations

from typing import Any


def validate_52w_range(fin_data: dict[str, Any], symbol: str) -> str:
    """
    Validate 52W high/low integrity in-place.

    Mutates fin_data: corrects week52_high/week52_low from 1Y history when
    corrupted, and always sets fin_data["52w_data_quality"] to one of:
      VERIFIED            — high > low, current price within range
      RANGE_MISMATCH      — high > low but current price outside stated range
      COMPUTED_FROM_HISTORY — high == low or either None; recomputed from yfinance 1Y history
      UNAVAILABLE         — history fallback also failed

    Returns the quality string.
    """
    high = fin_data.get("week52_high") or fin_data.get("fiftyTwoWeekHigh")
    low  = fin_data.get("week52_low")  or fin_data.get("fiftyTwoWeekLow")
    current = fin_data.get("current_price") or fin_data.get("currentPrice")

    try:
        high = float(high) if high is not None else None
        low  = float(low)  if low  is not None else None
    except (TypeError, ValueError):
        high = low = None

    corrupted = (high is None or low is None or high <= low)

    if corrupted:
        quality = "UNAVAILABLE"
        try:
            import yfinance as yf
            hist = yf.Ticker(f"{symbol}.NS").history(period="1y")
            if not hist.empty and "High" in hist.columns and "Low" in hist.columns:
                computed_high = round(float(hist["High"].max()), 2)
                computed_low  = round(float(hist["Low"].min()),  2)
                if computed_high > computed_low:
                    fin_data["week52_high"]      = computed_high
                    fin_data["week52_low"]        = computed_low
                    fin_data["fiftyTwoWeekHigh"]  = computed_high
                    fin_data["fiftyTwoWeekLow"]   = computed_low
                    quality = "COMPUTED_FROM_HISTORY"
        except Exception:
            pass
    else:
        try:
            cur = float(current) if current is not None else None
        except (TypeError, ValueError):
            cur = None
        if cur is not None and (cur > high * 1.05 or cur < low * 0.95):
            quality = "RANGE_MISMATCH"
        else:
            quality = "VERIFIED"

    fin_data["52w_data_quality"] = quality
    return quality


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
    # Validate 52W range before computing any range-relative signals.
    # Mutates fin_data to fix corrupted values; sets fin_data["52w_data_quality"].
    w52_quality = validate_52w_range(fin_data, symbol)

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

    # Signal 2: 52-week range proximity — only computed when data is trusted
    week52_low  = fin_data.get("week52_low")  or fin_data.get("fiftyTwoWeekLow")
    week52_high = fin_data.get("week52_high") or fin_data.get("fiftyTwoWeekHigh")
    if w52_quality in ("VERIFIED", "COMPUTED_FROM_HISTORY", "RANGE_MISMATCH"):
        if week52_low and week52_high and current_price and float(week52_high) > float(week52_low):
            pct_from_low  = (current_price - float(week52_low))  / float(week52_low)  * 100
            pct_from_high = (float(week52_high) - current_price) / float(week52_high) * 100

            if pct_from_low <= 15:
                signals.append({
                    "type":   "52W_RANGE",
                    "value":  f"{pct_from_low:.1f}% from low",
                    "signal": "CONTRARIAN_BUY",
                    "note":   (
                        f"Only {pct_from_low:.1f}% above 52-week low — "
                        "contrarian entry zone if thesis intact"
                        + (" [range from history]" if w52_quality == "COMPUTED_FROM_HISTORY" else "")
                    ),
                })
            elif pct_from_high <= 5:
                signals.append({
                    "type":   "52W_RANGE",
                    "value":  f"{pct_from_high:.1f}% from high",
                    "signal": "CAUTION",
                    "note":   "Near 52-week high — momentum entry, not value entry",
                })
    # UNAVAILABLE: skip 52W range signals entirely — don't generate false signals

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
        "symbol":           symbol,
        "signals":          signals,
        "technical_score":  round(avg_score, 2),
        "signal_count":     len(signals),
        "52w_data_quality": w52_quality,
    }
