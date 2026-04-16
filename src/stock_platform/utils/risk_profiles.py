from __future__ import annotations

from typing import Any

# ── Per-profile thresholds ────────────────────────────────────────────────────
# quant_cap_no_result_signal uses the quant-layer vocabulary:
#   "STRONG ENTER" | "ACCUMULATE" | "SMALL INITIAL" | "WAIT" | "DO NOT ENTER"
RISK_PROFILES: dict[str, dict[str, Any]] = {
    "Conservative": {
        "min_rr_ratio":               2.5,
        "max_pe_vs_median_pct":       10,
        "min_roce":                   22,
        "max_debt_equity":            0.3,
        "staleness_cap_days":         45,
        "quant_cap_no_result_signal": "WAIT",
        "min_revenue_growth":         15,
        "confidence_floor":           "MEDIUM",
        "llm_temperature_hint":       "cautious",
        "aggressive_entry_pct":       3,
    },
    "Balanced": {
        "min_rr_ratio":               2.0,
        "max_pe_vs_median_pct":       25,
        "min_roce":                   18,
        "max_debt_equity":            0.5,
        "staleness_cap_days":         90,
        "quant_cap_no_result_signal": "WAIT",
        "min_revenue_growth":         12,
        "confidence_floor":           "LOW",
        "llm_temperature_hint":       "balanced",
        "aggressive_entry_pct":       5,
    },
    "Aggressive": {
        "min_rr_ratio":               1.5,
        "max_pe_vs_median_pct":       50,
        "min_roce":                   15,
        "max_debt_equity":            0.8,
        "staleness_cap_days":         120,
        "quant_cap_no_result_signal": "SMALL INITIAL",
        "min_revenue_growth":         8,
        "confidence_floor":           "LOW",
        "llm_temperature_hint":       "growth_oriented",
        "aggressive_entry_pct":       7,
    },
}
# "Moderate" treated as "Balanced" for backward compat with stored UI value.
RISK_PROFILES["Moderate"] = RISK_PROFILES["Balanced"]

# ── LLM system-prompt injections per profile ──────────────────────────────────
RISK_PROMPT_HINTS: dict[str, str] = {
    "Conservative": (
        "RISK PROFILE: CONSERVATIVE investor.\n"
        "Apply strict criteria:\n"
        "- Reject any stock with PE more than 10% above its 5-year median\n"
        "- Require confirmed result date within 45 days\n"
        "- Minimum R/R of 2.5x before recommending entry\n"
        "- Default to WATCHLIST unless ALL criteria are met\n"
        "- Flag any governance, pledge, or concentration risk as disqualifying"
    ),
    "Balanced": (
        "RISK PROFILE: BALANCED investor.\n"
        "Apply standard criteria:\n"
        "- Tolerate PE up to 25% above 5-year median if growth justifies it\n"
        "- Accept result date up to 90 days old\n"
        "- Minimum R/R of 2.0x before recommending entry\n"
        "- WATCHLIST is appropriate when data gaps exist\n"
        "- Flag risks but do not auto-disqualify on single factors"
    ),
    "Aggressive": (
        "RISK PROFILE: AGGRESSIVE, GROWTH-ORIENTED investor with a 24-36 month horizon.\n"
        "Apply growth-first criteria:\n"
        "- Tolerate PE premium if revenue growth > 20% YoY and ROCE > 15%\n"
        "- Accept result date up to 120 days old for high-quality businesses"
        " -- data freshness is a caution, not a veto\n"
        "- Minimum R/R of 1.5x is acceptable for high-conviction ideas\n"
        "- ACCUMULATE GRADUALLY is your preferred verdict when business quality"
        " is confirmed and sector gap is real\n"
        "- WATCHLIST only when there is NO identifiable catalyst OR the valuation"
        " is indefensible by any growth metric\n"
        "- Do NOT default to WATCHLIST just because data is slightly stale --"
        " aggressive investors tolerate information uncertainty in exchange for early entry"
    ),
}
RISK_PROMPT_HINTS["Moderate"] = RISK_PROMPT_HINTS["Balanced"]


def get_risk_config(risk_profile: str) -> dict[str, Any]:
    """Return the threshold config for the given profile, defaulting to Balanced."""
    return RISK_PROFILES.get(risk_profile, RISK_PROFILES["Balanced"])
