from __future__ import annotations

from typing import Any


def compute_evidence_strength(
        fin_data: dict,
        pe_context: dict,
        val_reliability: dict,
        tech_signals: dict,
        sector_signal: dict,
        news_sentiment: float = 0.0) -> dict[str, Any]:
    """
    Score the evidence supporting this recommendation.
    Returns 0-1 score and component breakdown.

    This is NOT the quality score (ROCE/D/E etc).
    This scores HOW MUCH EVIDENCE supports the thesis.
    """
    score: float = 0.0
    components: dict[str, float] = {}

    # 1. Data freshness (0-0.20)
    try:
        age = int(fin_data.get("data_age_days", 99))
    except (TypeError, ValueError):
        age = 99

    freshness = (
        0.20 if age < 7  else
        0.15 if age < 30 else
        0.08 if age < 90 else
        0.02
    )
    score += freshness
    components["data_freshness"] = freshness

    # 2. Recent earnings quality (0-0.25)
    qtr = fin_data.get("recent_results") or {}
    momentum = str(qtr.get("momentum") or "").upper()
    earnings_score = (
        0.25 if momentum == "STRONG"   else
        0.18 if momentum == "GOOD"     else
        0.10 if momentum == "MODERATE" else
        0.03
    )
    score += earnings_score
    components["earnings_quality"] = earnings_score

    # 3. Valuation reliability (0-0.15)
    val_label = (val_reliability or {}).get("label", "LOW")
    val_score: float = {"HIGH": 0.15, "MEDIUM": 0.08, "LOW": 0.02}.get(val_label, 0.02)
    score += val_score
    components["valuation_reliability"] = val_score

    # 4. PE context clarity (0-0.10)
    pe_signal = pe_context.get("pe_signal", "")
    pe_score = (
        0.10 if pe_signal in ("VERY_CHEAP_VS_HISTORY", "EXPENSIVE_VS_HISTORY") else
        0.06 if pe_signal in ("CHEAP_VS_HISTORY", "SLIGHT_PREMIUM")            else
        0.02  # NO_HISTORY or NEUTRAL
    )
    score += pe_score
    components["pe_context_clarity"] = pe_score

    # 5. Sector signal strength (0-0.15)
    conviction = (sector_signal or {}).get("conviction", "NEUTRAL")
    sect_score: float = {
        "STRONG_BUY": 0.15,
        "BUY":        0.10,
        "NEUTRAL":    0.05,
        "SELL":       0.0,
        "AVOID":      0.0,
        "STRONG_AVOID": 0.0,
    }.get(conviction, 0.05)
    score += sect_score
    components["sector_signal"] = sect_score

    # 6. Technical signal (0-0.10)
    tech_score = 0.05  # default neutral
    try:
        pct_from_low = float(tech_signals.get("pct_from_52w_low") or 50)
    except (TypeError, ValueError):
        pct_from_low = 50.0
    if pct_from_low < 15:
        tech_score = 0.10  # near 52W low — contrarian support
    elif pct_from_low > 80:
        tech_score = 0.02  # near 52W high — stretched
    score += tech_score
    components["technical"] = tech_score

    # 7. News sentiment (0-0.05)
    try:
        ns = float(news_sentiment)
    except (TypeError, ValueError):
        ns = 0.0
    news_score = min(0.05, max(0.0, (ns + 1.0) / 2.0 * 0.05))
    score += news_score
    components["news_sentiment"] = round(news_score, 3)

    score = round(min(1.0, score), 2)

    label = (
        "STRONG"    if score >= 0.70 else
        "MODERATE"  if score >= 0.45 else
        "WEAK"      if score >= 0.25 else
        "VERY WEAK"
    )

    return {
        "score":      score,
        "label":      label,
        "components": components,
        "note": (
            "Strong evidence basis — thesis well-supported"
            if label == "STRONG"   else
            "Moderate evidence — thesis plausible but not confirmed"
            if label == "MODERATE" else
            "Weak evidence — thesis largely inferential"
            if label == "WEAK"     else
            "Very weak evidence — consider rejecting this candidate"
        ),
    }
