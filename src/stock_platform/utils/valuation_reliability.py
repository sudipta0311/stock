from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Sector reliability maps
# ─────────────────────────────────────────────────────────────────────────────

_LOW_RELIABILITY_SECTORS: dict[str, str] = {
    "Power":       "capital-intensive utility — earnings distorted by capex",
    "Defence":     "PSU / policy-driven — lumpy order execution distorts PE",
    "Real Estate": "cyclical — PE meaningless at cycle extremes",
    "Metals":      "commodities cyclical — PE at trough/peak distorts history",
    "Oil & Gas":   "commodity price dependent",
}

_MEDIUM_RELIABILITY_SECTORS: dict[str, str] = {
    "Capital Goods":     "execution-led, order book dependent",
    "Consumer Durables": "discretionary cyclical",
    "Pharma":            "R&D binary outcomes, regulatory dependent",
    "CDMO":              "client concentration risk",
}

_HIGH_RELIABILITY_SECTORS: dict[str, str] = {
    "Consumer Staples": "stable earnings compounder",
    "Private Banks":    "consistent ROA/ROE",
    "IT Services":      "recurring revenue, visible margins",
    "FMCG":             "stable, predictable",
}


def get_valuation_reliability(
        symbol: str,
        sector: str,
        fin_data: dict,
        pe_context: dict,
        years_listed: int = 5) -> dict[str, Any]:
    """
    Assess how reliable PE comparison is for this specific stock.
    Returns reliability score 0-1 and reason.

    High reliability (0.7-1.0):
      Stable compounder, consistent earnings, long listing history,
      no major restructuring.

    Medium reliability (0.4-0.7):
      Some cyclicality or policy dependency, earnings history
      available but lumpy.

    Low reliability (0.0-0.4):
      Turnaround, recently listed, PSU/policy-driven, monopoly with
      distorted earnings, cyclical at peak/trough, distorted history.
    """
    score: float = 0.7  # default medium-high
    flags: list[str] = []

    # Factor 1: Years of listing history
    try:
        yl = int(years_listed)
    except (TypeError, ValueError):
        yl = 5

    if yl < 3:
        score -= 0.4
        flags.append("recently listed — PE history unreliable")
    elif yl < 5:
        score -= 0.2
        flags.append("short listing history")

    # Factor 2: Sector-based reliability
    s_lower = (sector or "").lower()
    for sec, reason_txt in _LOW_RELIABILITY_SECTORS.items():
        if sec.lower() in s_lower:
            score -= 0.3
            flags.append(reason_txt)
            break
    else:
        for sec, reason_txt in _MEDIUM_RELIABILITY_SECTORS.items():
            if sec.lower() in s_lower:
                score -= 0.1
                flags.append(reason_txt)
                break
        else:
            for sec, reason_txt in _HIGH_RELIABILITY_SECTORS.items():
                if sec.lower() in s_lower:
                    score += 0.1
                    flags.append(reason_txt)
                    break

    # Factor 3: Earnings stability indicators
    try:
        roce = float(
            fin_data.get("roce_pct")
            or fin_data.get("roce_ttm")
            or fin_data.get("returnOnCapitalEmployed")
            or fin_data.get("roce")
            or 15
        )
        # Convert fraction if needed
        if roce != 0 and abs(roce) < 1.5:
            roce *= 100
    except (TypeError, ValueError):
        roce = 15.0

    try:
        rev_growth = float(
            fin_data.get("revenue_growth_pct")
            or fin_data.get("revenueGrowth")
            or fin_data.get("revenue_growth")
            or 10
        )
        if rev_growth != 0 and abs(rev_growth) < 2.0:
            rev_growth *= 100
    except (TypeError, ValueError):
        rev_growth = 10.0

    if roce > 40:
        flags.append("ROCE > 40% — check if cyclical peak or structural")
        score -= 0.1
    if rev_growth > 40:
        flags.append("revenue growth > 40% — high growth distorts PE history")
        score -= 0.1
    if rev_growth < 0:
        flags.append("negative revenue growth — possible earnings distortion")
        score -= 0.15

    # Factor 4: D/E ratio
    try:
        de = float(fin_data.get("debt_to_equity") or 0)
    except (TypeError, ValueError):
        de = 0.0

    if de > 2.0:
        flags.append(
            f"D/E {de:.1f}x — highly leveraged, earnings sensitive to rate changes"
        )
        score -= 0.15

    # Factor 5: PE history availability
    if pe_context.get("pe_signal") == "NO_HISTORY":
        score -= 0.25
        flags.append("no PE history — comparison impossible")

    score = max(0.0, min(1.0, round(score, 2)))

    if score >= 0.65:
        label = "HIGH"
    elif score >= 0.40:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "score": score,
        "label": label,
        "flags": flags,
        "note": (
            "PE comparison is reliable — use historical median with confidence"
            if label == "HIGH" else
            "PE comparison is approximate — use alongside earnings/order book data"
            if label == "MEDIUM" else
            "PE comparison is unreliable here — do NOT anchor thesis on historical PE"
        ),
    }
