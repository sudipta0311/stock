from __future__ import annotations

from typing import Any


US_TARIFF_IMPACT_MAP = {
    # Sector: (impact_level, reason)
    "Consumer Durables": (
        "NEGATIVE",
        "US 26% tariff on Indian goods April 2026 - FII selling pressure, margin risk on export-linked components",
    ),
    "Pharma Exports": (
        "HIGH_NEGATIVE",
        "Direct US generic pharma revenue at risk",
    ),
    "IT Services": (
        "MODERATE_NEGATIVE",
        "Indirect impact via client spending cuts",
    ),
    "Defence": (
        "NEUTRAL",
        "Domestic sector - not affected by US tariffs",
    ),
    "Capital Goods": (
        "NEUTRAL",
        "Primarily domestic demand driven",
    ),
    "CDMO": (
        "MODERATE_NEGATIVE",
        "US pharma outsourcing demand uncertainty",
    ),
}

TARIFF_SCORE_PENALTIES = {
    "HIGH_NEGATIVE": 0.25,
    "NEGATIVE": 0.15,
    "MODERATE_NEGATIVE": 0.10,
}


def get_tariff_signal(sector: str) -> dict[str, Any]:
    impact = US_TARIFF_IMPACT_MAP.get(str(sector or "").strip())
    if not impact:
        return {}
    level, reason = impact
    return {
        "type": "US_TARIFF",
        "sector": sector,
        "impact": level,
        "reason": reason,
        "date": "2026-04-02",
        "source": "US reciprocal tariff announcement",
    }


def get_tariff_penalty(impact: str) -> float:
    return float(TARIFF_SCORE_PENALTIES.get(str(impact or "").upper(), 0.0))
