from __future__ import annotations

from typing import Any

# ── TIER 1: Universal hard exclusions ─────────────────────────────────────────
# Apply to ALL profiles and ALL runs. These reduce the universe before scoring.
# Change these only when a new regulatory or structural reason emerges.
# Do NOT add profile-specific tightening here — use TIER 2 instead.
UNIVERSAL_HARD_EXCLUDE: dict[str, float] = {
    "min_avg_daily_volume_cr": 5.0,    # illiquid — cannot exit position safely
    "max_promoter_pledge_pct": 50.0,   # extreme pledge — margin call overhang
    "max_debt_equity":         5.0,    # extreme leverage — solvency risk
    "min_market_cap_cr":       1000.0, # micro-cap — too thin for retail entry
}

# ── TIER 2: Per-profile thresholds ────────────────────────────────────────────
# These are SCORING MODIFIERS and CONFIDENCE INPUTS only.
# They NEVER reduce the candidate pool — they downgrade verdicts and confidence.
# Field names use "preferred_*" to signal scoring intent, not exclusion gates.
#
# quant_cap_no_result_signal uses the quant-layer vocabulary:
#   "STRONG ENTER" | "ACCUMULATE" | "SMALL INITIAL" | "WAIT" | "DO NOT ENTER"
RISK_PROFILES: dict[str, dict[str, Any]] = {
    "Conservative": {
        "min_rr_ratio":                    2.5,
        "preferred_max_pe_vs_median_pct":  10,   # scoring modifier — not a hard gate
        "preferred_min_roce":              22,   # scoring modifier — not a hard gate
        "preferred_max_de":                0.3,  # scoring modifier — not a hard gate
        "staleness_cap_days":              45,
        "quant_cap_no_result_signal":      "WAIT",
        "preferred_min_revenue_growth":    15,   # scoring modifier — not a hard gate
        "confidence_floor":                "MEDIUM",
        "llm_temperature_hint":            "cautious",
        "aggressive_entry_pct":            3,
    },
    "Balanced": {
        "min_rr_ratio":                    2.0,
        "preferred_max_pe_vs_median_pct":  25,
        "preferred_min_roce":              18,
        "preferred_max_de":                0.5,
        "staleness_cap_days":              90,
        "quant_cap_no_result_signal":      "WAIT",
        "preferred_min_revenue_growth":    12,
        "confidence_floor":                "LOW",
        "llm_temperature_hint":            "balanced",
        "aggressive_entry_pct":            5,
    },
    "Aggressive": {
        "min_rr_ratio":                    1.5,
        "preferred_max_pe_vs_median_pct":  50,
        "preferred_min_roce":              15,
        "preferred_max_de":                0.8,
        "staleness_cap_days":              120,
        "quant_cap_no_result_signal":      "SMALL INITIAL",
        "preferred_min_revenue_growth":    8,
        "confidence_floor":                "LOW",
        "llm_temperature_hint":            "growth_oriented",
        "aggressive_entry_pct":            7,
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
        " aggressive investors tolerate information uncertainty in exchange for early entry\n\n"
        "OVERRIDE RULE FOR AGGRESSIVE PROFILE — AVOID THRESHOLD:\n"
        "If the entry plan R/R >= 2.5x AND ROCE > 10% (or ROCE weakness is explained"
        " by holding-company structure), you MUST NOT recommend AVOID."
        " Maximum downgrade allowed is WATCHLIST.\n"
        "AVOID is reserved ONLY for:\n"
        "- Governance red flags (promoter pledge > 30%, SEBI action, fraud history)\n"
        "- D/E > 2.0x with deteriorating earnings\n"
        "- Revenue growth negative YoY\n"
        "- PE > 100% above 5-year median with no growth justification\n"
        "For everything else, WATCHLIST is your floor."
        " The synthesis layer will adjudicate — your job is to surface risks, not to veto."
    ),
}
RISK_PROMPT_HINTS["Moderate"] = RISK_PROMPT_HINTS["Balanced"]


def get_risk_config(risk_profile: str) -> dict[str, Any]:
    """Return the threshold config for the given profile, defaulting to Balanced."""
    return RISK_PROFILES.get(risk_profile, RISK_PROFILES["Balanced"])
