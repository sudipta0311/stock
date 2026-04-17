"""
recommendation_resolver.py
──────────────────────────
Governance layer: parse LLM verdicts, run consistency checks,
and produce a single canonical recommendation state that drives
the UI.  No LLM calls — pure deterministic logic.
"""
from __future__ import annotations

import re
from datetime import date as _date
from typing import Any

from stock_platform.utils.risk_profiles import get_risk_config


# ── Canonical state ordering (lower index = more conservative) ───────────────
_CANONICAL_ORDER: list[str] = [
    "AVOID",
    "WATCHLIST",
    "BUY ONLY AFTER CONFIRMATION",
    "ACCUMULATE ON DIPS",
    "ACCUMULATE GRADUALLY",
    "ACTIONABLE BUY",
]

_CANONICAL_RANK: dict[str, int] = {s: i for i, s in enumerate(_CANONICAL_ORDER)}


def rank_canonical_state(state: str) -> int:
    """Public rank helper for recommendation states (higher = more actionable)."""
    return _CANONICAL_RANK.get((state or "").upper().strip(), 99)


# ── Verdict maps: LLM output token → canonical state ─────────────────────────

# Anthropic risk analyst outputs  ## RISK VERDICT: <token>
_RISK_VERDICT_MAP: dict[str, str] = {
    "AVOID":                      "AVOID",
    "WAIT":                       "WATCHLIST",
    "ACCUMULATE WITH CONDITIONS": "ACCUMULATE ON DIPS",
    "BUY":                        "ACCUMULATE GRADUALLY",
}

# OpenAI catalyst analyst outputs  ## CATALYST VERDICT: <token>
_CATALYST_VERDICT_MAP: dict[str, str] = {
    "AVOID":      "AVOID",
    "WATCHLIST":  "WATCHLIST",
    "ACCUMULATE": "ACCUMULATE GRADUALLY",
    "BUY NOW":    "ACTIONABLE BUY",
}

# Synthesis verdict stance tokens  ## SYNTHESIS VERDICT: <stance> | Confidence: <level>
_SYNTHESIS_STANCE_MAP: dict[str, str] = {
    "AVOID":                      "AVOID",
    "WATCHLIST":                  "WATCHLIST",
    "BUY ONLY AFTER CONFIRMATION":"BUY ONLY AFTER CONFIRMATION",
    "ACCUMULATE ON DIPS":         "ACCUMULATE ON DIPS",
    "ACCUMULATE GRADUALLY":       "ACCUMULATE GRADUALLY",
    "ACTIONABLE BUY":             "ACTIONABLE BUY",
    # tolerate quant-style labels that may leak in
    "ACCUMULATE":                 "ACCUMULATE GRADUALLY",
    "STRONG ENTER":               "ACTIONABLE BUY",
    "BUY":                        "ACCUMULATE GRADUALLY",
    "WAIT":                       "WATCHLIST",
    "STRONG AVOID":               "AVOID",
}

# Quant action labels → canonical
_QUANT_ACTION_MAP: dict[str, str] = {
    "STRONG ENTER":     "ACTIONABLE BUY",
    "ACCUMULATE":       "ACCUMULATE GRADUALLY",
    "WAIT":             "WATCHLIST",
    "WATCH":            "WATCHLIST",
    "AVOID":            "AVOID",
    "STRONG AVOID":     "AVOID",
    # Improvement-6 may produce these
    "ACCUMULATE GRADUALLY": "ACCUMULATE GRADUALLY",
    "ACCUMULATE ON DIPS":   "ACCUMULATE ON DIPS",
    "ACTIONABLE BUY":       "ACTIONABLE BUY",
    "BUY ONLY AFTER CONFIRMATION": "BUY ONLY AFTER CONFIRMATION",
    "WATCHLIST":            "WATCHLIST",
}


# ── Verdict parsers ───────────────────────────────────────────────────────────

def extract_risk_verdict(rationale_text: str) -> str | None:
    """
    Parse '## RISK VERDICT: AVOID' from Anthropic risk analyst output.
    Returns canonical state string or None if not found.
    """
    if not rationale_text:
        return None
    m = re.search(r"##\s*RISK VERDICT\s*:\s*([A-Z /]+)", rationale_text, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).strip().upper()
    # try exact match first, then prefix match
    if token in _RISK_VERDICT_MAP:
        return _RISK_VERDICT_MAP[token]
    for key, val in _RISK_VERDICT_MAP.items():
        if token.startswith(key):
            return val
    return None


def extract_catalyst_verdict(rationale_text: str) -> str | None:
    """
    Parse '## CATALYST VERDICT: BUY NOW' from OpenAI catalyst analyst output.
    Returns canonical state string or None if not found.
    """
    if not rationale_text:
        return None
    m = re.search(r"##\s*CATALYST VERDICT\s*:\s*([A-Z /]+)", rationale_text, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).strip().upper()
    if token in _CATALYST_VERDICT_MAP:
        return _CATALYST_VERDICT_MAP[token]
    for key, val in _CATALYST_VERDICT_MAP.items():
        if token.startswith(key):
            return val
    return None


def extract_synthesis_verdict(synthesis_text: str) -> tuple[str, str]:
    """
    Parse '## SYNTHESIS VERDICT: ACCUMULATE ON DIPS | Confidence: MODERATE'
    Returns (canonical_state, confidence_level). Both default to "" if absent.
    """
    if not synthesis_text:
        return "", ""
    m = re.search(
        r"##\s*SYNTHESIS VERDICT\s*:\s*([^|]+?)(?:\s*\|\s*Confidence\s*:\s*([A-Z]+))?(?:\n|$)",
        synthesis_text, re.IGNORECASE
    )
    if not m:
        return "", ""
    stance_raw = m.group(1).strip().upper()
    confidence = (m.group(2) or "").strip().upper()
    canonical = _SYNTHESIS_STANCE_MAP.get(stance_raw, "")
    if not canonical:
        # partial match
        for key, val in _SYNTHESIS_STANCE_MAP.items():
            if stance_raw.startswith(key) or key.startswith(stance_raw):
                canonical = val
                break
    return canonical, confidence


def determine_preliminary_verdict(
    anthropic_rationale: str,
    openai_rationale: str,
) -> str:
    """
    Pick the most actionable analyst verdict before synthesis.

    This is used by the news gate: if either analyst is already approaching an
    actionable stance, fetch recent risk news before asking the synthesis model
    for a final verdict.
    """
    candidates = [
        verdict
        for verdict in (
            extract_risk_verdict(anthropic_rationale),
            extract_catalyst_verdict(openai_rationale),
        )
        if verdict
    ]
    if not candidates:
        return "WATCHLIST"
    return max(candidates, key=rank_canonical_state)


# ── Signal reconciliation ────────────────────────────────────────────────────

def reconcile_signals(
    quant_canonical: str,
    llm_canonical: str,
    risk_profile: str = "Balanced",
) -> dict[str, str | None]:
    """
    Compare quant canonical state vs the best available LLM canonical state.

    Gap tolerance is risk-profile-aware:
      Conservative  — gap must be 0 to be ALIGNED (strict)
      Balanced      — gap <= 1 is ALIGNED (default)
      Aggressive    — gap <= 2 is ALIGNED (tolerant)

    status values:
      ALIGNED           — within profile tolerance
      MINOR_DIVERGENCE  — one tier beyond tolerance
      CONFLICT          — two+ tiers beyond tolerance
      NO_SYNTHESIS      — llm_canonical absent; nothing to compare
      UNKNOWN           — unmapped state; skip display
    """
    if not llm_canonical:
        return {"status": "NO_SYNTHESIS", "message": "", "badge_type": "info", "override": None}

    q_rank = _CANONICAL_RANK.get(quant_canonical, 99)
    l_rank = _CANONICAL_RANK.get(llm_canonical, 99)

    if q_rank == 99 or l_rank == 99:
        return {"status": "UNKNOWN", "message": "", "badge_type": "info", "override": None}

    gap = abs(q_rank - l_rank)

    # Profile-driven alignment threshold
    _aligned_threshold = {"Conservative": 0, "Balanced": 1, "Aggressive": 2}.get(risk_profile, 1)

    if gap <= _aligned_threshold:
        msg = (
            f"Quant and LLM analysis agree: {quant_canonical}"
            if gap == 0
            else (
                f"Within {risk_profile} tolerance: "
                f"Quant {quant_canonical} / LLM {llm_canonical}"
            )
        )
        return {"status": "ALIGNED", "message": msg, "badge_type": "success", "override": None}

    if gap == _aligned_threshold + 1:
        if q_rank > l_rank:
            return {
                "status":     "MINOR_DIVERGENCE",
                "message":    (
                    f"Quant: {quant_canonical} | LLM: {llm_canonical} — "
                    "minor gap. LLM adds qualitative caution; LLM verdict governs."
                ),
                "badge_type": "info",
                "override":   "LLM_DEFERRED",
            }
        else:
            return {
                "status":     "MINOR_DIVERGENCE",
                "message":    (
                    f"Quant: {quant_canonical} | LLM: {llm_canonical} — "
                    "minor gap. Quant factor scores are cautious; quant conservatism governs."
                ),
                "badge_type": "info",
                "override":   "QUANT_DEFERRED",
            }

    # gap >= _aligned_threshold + 2 — material conflict
    if q_rank > l_rank:
        return {
            "status":     "CONFLICT",
            "direction":  "QUANT_BULLISH_LLM_CAUTIOUS",
            "message":    (
                f"Quant: {quant_canonical} (factor scores) vs "
                f"LLM: {llm_canonical} (qualitative analysis) — "
                f"{gap}-tier gap. LLM identifies risks that downgrade the idea. "
                "LLM verdict governs entry decision."
            ),
            "badge_type": "warning",
            "override":   "LLM_VERDICT",
        }
    else:
        return {
            "status":     "CONFLICT",
            "direction":  "LLM_BULLISH_QUANT_CAUTIOUS",
            "message":    (
                f"Quant: {quant_canonical} (factor scores) vs "
                f"LLM: {llm_canonical} (qualitative analysis) — "
                f"{gap}-tier gap. Factor scores are weak despite positive LLM view. "
                "Quant conservatism governs."
            ),
            "badge_type": "warning",
            "override":   "QUANT_VERDICT",
        }


# ── Data freshness scoring ────────────────────────────────────────────────────

def compute_data_freshness_score(
    last_result_date: str,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    """
    Returns a freshness multiplier (0.0-1.0) that scales confidence.
    Plugs into final confidence calculation in resolve_final_recommendation().

    snapshot_date defaults to today when omitted.
    """
    today_str = snapshot_date or _date.today().strftime("%Y-%m-%d")

    if not last_result_date or str(last_result_date).lower() in ("none", "unknown", ""):
        return {
            "freshness_score": 0.0,
            "days_stale": None,
            "label": "NO_RESULT_DATE",
            "confidence_cap": "LOW",
        }

    try:
        from datetime import datetime
        result_dt   = datetime.strptime(str(last_result_date), "%Y-%m-%d")
        snapshot_dt = datetime.strptime(today_str, "%Y-%m-%d")
        days_stale  = (snapshot_dt - result_dt).days
    except Exception:
        return {"freshness_score": 0.3, "days_stale": None, "label": "PARSE_ERROR", "confidence_cap": "LOW"}

    if days_stale <= 30:
        return {"freshness_score": 1.0,  "days_stale": days_stale, "label": "FRESH",      "confidence_cap": "HIGH"}
    elif days_stale <= 60:
        return {"freshness_score": 0.75, "days_stale": days_stale, "label": "ACCEPTABLE",  "confidence_cap": "MEDIUM"}
    elif days_stale <= 90:
        return {"freshness_score": 0.5,  "days_stale": days_stale, "label": "STALE",       "confidence_cap": "LOW"}
    else:
        return {"freshness_score": 0.2,  "days_stale": days_stale, "label": "VERY_STALE",  "confidence_cap": "LOW",
                "force_watchlist": True}


# ── Consistency checks ────────────────────────────────────────────────────────

def run_consistency_checks(
    payload: dict[str, Any],
    current_price: float,
    analyst_target: float,
    quant_action: str,
    synthesis_canonical: str,
) -> list[dict[str, str]]:
    """
    Return list of {check, severity, message} dicts.
    severity: CRITICAL | WARNING
    """
    failures: list[dict[str, str]] = []

    # 1. Data freshness — only flag if explicitly known > 30 days
    #    (data_age_days defaults to 99 when absent — treat as unknown, not stale)
    fin_data = payload.get("fin_data") or {}
    raw_age = fin_data.get("data_age_days")
    if raw_age is not None:
        try:
            age = int(raw_age)
            # 99 is the sentinel default for "unknown" — do not flag it
            if age != 99 and age > 60:
                failures.append({
                    "check": "data_freshness",
                    "severity": "CRITICAL",
                    "message": f"Financial data is {age} days old (>60 days — stale).",
                })
            elif age != 99 and age > 30:
                failures.append({
                    "check": "data_freshness",
                    "severity": "WARNING",
                    "message": f"Financial data is {age} days old (>30 days).",
                })
        except (TypeError, ValueError):
            pass

    # 2. Analyst target below CMP — invalidates trade math
    if analyst_target and current_price and float(analyst_target) < float(current_price) * 0.98:
        failures.append({
            "check": "target_below_cmp",
            "severity": "CRITICAL",
            "message": (
                f"Analyst target ₹{analyst_target:,.0f} is below CMP ₹{current_price:,.0f}. "
                "Trade mechanics are invalid."
            ),
        })

    # 3. Impossible 52W price range (keys: week52_low / week52_high — no underscore between digits)
    try:
        low_52w  = float(fin_data.get("week52_low")  or fin_data.get("fiftyTwoWeekLow")  or 0)
        high_52w = float(fin_data.get("week52_high") or fin_data.get("fiftyTwoWeekHigh") or 0)
        if low_52w > 0 and high_52w > 0 and low_52w >= high_52w:
            failures.append({
                "check": "impossible_52w_range",
                "severity": "WARNING",
                "message": (
                    f"52W low (₹{low_52w:,.0f}) ≥ 52W high (₹{high_52w:,.0f}) — data corrupt. "
                    "52W range signals suppressed."
                ),
            })
    except (TypeError, ValueError):
        pass

    # (verdict_mismatch check removed — replaced by reconcile_signals() in resolver return)

    # 4. Valuation reliability failure
    val_rel = (payload.get("val_reliability") or {}).get("label", "")
    if val_rel == "LOW":
        failures.append({
            "check": "low_valuation_reliability",
            "severity": "WARNING",
            "message": "Valuation reliability is LOW — PE/target comparisons are unreliable for this sector.",
        })

    # 5. Missing execution data
    entry = payload.get("entry_levels") or {}
    if not entry or not entry.get("entry_price"):
        failures.append({
            "check": "missing_execution_data",
            "severity": "WARNING",
            "message": "Entry price could not be calculated — trade mechanics unavailable.",
        })

    return failures


# ── Main resolver ─────────────────────────────────────────────────────────────

def resolve_final_recommendation(
    quant_action: str,
    anthropic_rationale: str,
    openai_rationale: str,
    synthesis_text: str,
    payload: dict[str, Any],
    provider: str = "",
    risk_profile: str = "Balanced",
) -> dict[str, Any]:
    """
    Most-conservative-wins aggregation across quant + analyst verdicts + synthesis.

    Returns:
        canonical_state    — final UI-facing recommendation label
        actionability      — ACTIONABLE | NON_ACTIONABLE | DEGRADED
        suppressed_reasons — list[str] explaining what was suppressed and why
        upgrade_trigger    — what would move the recommendation up a tier
        consistency_failures — list from run_consistency_checks()
        data_status        — OK | STALE | UNKNOWN
        confidence_level   — HIGH | MODERATE | LOW (from synthesis)
        confidence_reason  — human-readable string
        synthesis_stance   — raw parsed synthesis canonical
        risk_verdict       — parsed Anthropic verdict canonical
        catalyst_verdict   — parsed OpenAI verdict canonical
    """
    # ── 1. Parse each source ─────────────────────────────────────────────────
    quant_canon      = _QUANT_ACTION_MAP.get((quant_action or "").upper().strip(), "WATCHLIST")
    risk_canon       = extract_risk_verdict(anthropic_rationale) if provider in ("anthropic", "") else None
    catalyst_canon   = extract_catalyst_verdict(openai_rationale) if provider in ("openai", "") else None
    synthesis_canon, confidence_level = extract_synthesis_verdict(synthesis_text)

    # When running single-provider mode, only consider verdicts from that provider
    if provider == "anthropic":
        catalyst_canon = None
    elif provider == "openai":
        risk_canon = None

    # ── 2. Most-conservative-wins ────────────────────────────────────────────
    candidates: list[str] = [quant_canon]
    if risk_canon:
        candidates.append(risk_canon)
    if catalyst_canon:
        candidates.append(catalyst_canon)
    if synthesis_canon:
        candidates.append(synthesis_canon)

    def _rank(s: str) -> int:
        return _CANONICAL_RANK.get(s, 99)

    canonical_state = min(candidates, key=_rank)

    # ── 3. Consistency checks ────────────────────────────────────────────────
    current_price   = float(payload.get("current_price") or 0)
    analyst_target  = float(payload.get("analyst_target") or 0)
    consistency_failures = run_consistency_checks(
        payload, current_price, analyst_target, quant_action, synthesis_canon or canonical_state
    )
    critical_failures = [f for f in consistency_failures if f["severity"] == "CRITICAL"]

    # Critical failures auto-downgrade to at most WATCHLIST
    if critical_failures:
        if _rank(canonical_state) > _rank("WATCHLIST"):
            canonical_state = "WATCHLIST"

    # ── 4. Data freshness scoring + confidence cap ───────────────────────────
    fin_data = payload.get("fin_data") or {}
    freshness_score = compute_data_freshness_score(
        fin_data.get("last_result_date") or "",
    )

    # Profile staleness tolerance — Aggressive allows up to 120 days without suppression
    _profile_cfg      = get_risk_config(risk_profile)
    _profile_stale_cap = int(_profile_cfg.get("staleness_cap_days", 90))
    _days_stale_actual = freshness_score.get("days_stale") or 0

    # force_watchlist: suppress only when data exceeds the profile's own staleness cap
    if freshness_score.get("force_watchlist") and _days_stale_actual > _profile_stale_cap:
        if _rank(canonical_state) > _rank("WATCHLIST"):
            canonical_state = "WATCHLIST"

    # confidence_cap: stale/missing data caps LLM confidence regardless of its output
    _cap = freshness_score.get("confidence_cap", "HIGH")
    if _cap == "LOW" and (confidence_level or "").upper() not in ("LOW", ""):
        confidence_level = "LOW"
    elif _cap == "MEDIUM" and (confidence_level or "").upper() == "HIGH":
        confidence_level = "MODERATE"

    # data_status: map freshness label to OK/STALE/UNKNOWN for the actionability gate.
    # Use the profile's staleness cap so Aggressive doesn't get NON_ACTIONABLE on data
    # that is within its declared tolerance.
    _fl = freshness_score.get("label", "")
    if _fl in ("FRESH", "ACCEPTABLE") or _days_stale_actual <= _profile_stale_cap:
        data_status = "OK"
    elif _fl in ("STALE", "VERY_STALE"):
        data_status = "STALE"
    else:
        # NO_RESULT_DATE / PARSE_ERROR — use data_age_days sentinel fallback
        raw_age = fin_data.get("data_age_days")
        try:
            age_int = int(raw_age) if raw_age is not None else None
        except (TypeError, ValueError):
            age_int = None
        if age_int is None or age_int == 99:
            data_status = "UNKNOWN"
        elif age_int > 60:
            data_status = "STALE"
        else:
            data_status = "OK"

    # ── 5. Actionability gate ────────────────────────────────────────────────
    evidence_label  = (payload.get("evidence") or {}).get("label", "")
    val_label       = (payload.get("val_reliability") or {}).get("label", "")
    target_src      = str(payload.get("target_source_label") or "")
    is_model_target = "model" in target_src.lower() or not target_src

    non_actionable_states = {"AVOID", "WATCHLIST", "BUY ONLY AFTER CONFIRMATION"}
    degraded_conditions = [
        is_model_target,
        evidence_label in ("WEAK",),
        val_label == "LOW",
        (confidence_level or "").upper() in ("LOW",),
        bool([f for f in consistency_failures if f["severity"] == "WARNING"]),
    ]

    if canonical_state in non_actionable_states or data_status == "STALE":
        actionability = "NON_ACTIONABLE"
    elif analyst_target and current_price and float(analyst_target) < float(current_price) * 0.98:
        actionability = "NON_ACTIONABLE"
    elif any(degraded_conditions):
        actionability = "DEGRADED"
    else:
        actionability = "ACTIONABLE"

    # ── 6. Suppression reasons ───────────────────────────────────────────────
    suppressed_reasons: list[str] = []
    if canonical_state == "AVOID":
        suppressed_reasons.append("AVOID verdict — no trade plan shown")
    elif canonical_state == "WATCHLIST":
        suppressed_reasons.append("WATCHLIST — monitoring only, no entry plan shown")
    elif canonical_state == "BUY ONLY AFTER CONFIRMATION":
        suppressed_reasons.append("Confirmation required before entry")
    if data_status == "STALE":
        suppressed_reasons.append("Data is stale (>60 days old)")
    if analyst_target and current_price and float(analyst_target) < float(current_price) * 0.98:
        suppressed_reasons.append("Analyst target below CMP — trade math invalid")
    for f in critical_failures:
        if f["check"] not in ("data_freshness", "target_below_cmp"):
            suppressed_reasons.append(f["message"])

    # ── 7. Confidence reason ─────────────────────────────────────────────────
    confidence_parts: list[str] = []
    if confidence_level:
        confidence_parts.append(f"Synthesis confidence: {confidence_level}")
    if evidence_label:
        confidence_parts.append(f"{evidence_label} evidence basis")
    if val_label:
        confidence_parts.append(f"{val_label} valuation reliability")
    confidence_reason = " · ".join(confidence_parts) if confidence_parts else "Confidence not assessed"

    # ── 8. Upgrade trigger ───────────────────────────────────────────────────
    upgrade_trigger = _upgrade_trigger(canonical_state, consistency_failures, payload)

    # ── 9. Signal reconciliation — quant vs best available LLM verdict ───────
    # Compare against synthesis first; fall back to single-analyst verdict.
    # Deliberately NOT compared against canonical_state (that's already the merged result).
    _llm_for_reconcile = synthesis_canon or risk_canon or catalyst_canon or ""
    reconciliation = reconcile_signals(quant_canon, _llm_for_reconcile, risk_profile)

    # ── 10. UI flags — derived from resolved state, drive renderer decisions ──
    ui_flags = {
        "show_trade_plan":       actionability != "NON_ACTIONABLE",
        "show_target_table":     actionability != "NON_ACTIONABLE",
        "show_stop_loss":        actionability != "NON_ACTIONABLE",
        "show_tranche_plan":     actionability != "NON_ACTIONABLE",
        "show_rr":               actionability != "NON_ACTIONABLE",
        "show_warning_banner":   bool(consistency_failures),
        "grey_out_trade_fields": (
            data_status == "STALE" and (confidence_level or "").upper() == "LOW"
        ),
    }

    # final_verdict: the signal that governs the entry decision per reconciliation rules
    _override = reconciliation.get("override")
    if _override in (None, "LLM_DEFERRED", "LLM_VERDICT"):
        final_verdict = _llm_for_reconcile or canonical_state
    else:
        final_verdict = quant_canon or canonical_state

    return {
        "canonical_state":       canonical_state,
        "actionability":         actionability,
        "suppressed_reasons":    suppressed_reasons,
        "upgrade_trigger":       upgrade_trigger,
        "consistency_failures":  consistency_failures,
        "data_status":           data_status,
        "confidence_level":      confidence_level or "UNKNOWN",
        "confidence_reason":     confidence_reason,
        "synthesis_stance":      synthesis_canon,
        "risk_verdict":          risk_canon,
        "catalyst_verdict":      catalyst_canon,
        "reconciliation":        reconciliation,
        "final_verdict":         final_verdict,
        "freshness_score":       freshness_score,
        "ui_flags":              ui_flags,
    }


def _upgrade_trigger(
    canonical_state: str,
    consistency_failures: list[dict[str, str]],
    payload: dict[str, Any],
) -> str:
    """Return a brief sentence describing what would move this up one tier."""
    if canonical_state == "AVOID":
        return "Requires material improvement in fundamentals or management credibility before re-evaluation."
    if canonical_state == "WATCHLIST":
        fail_checks = {f["check"] for f in consistency_failures}
        triggers: list[str] = []
        if "data_freshness" in fail_checks:
            triggers.append("fresh financial data (< 30 days)")
        if "target_below_cmp" in fail_checks:
            triggers.append("analyst target above CMP")
        triggers.append("earnings catalyst or price dip to entry zone")
        return "Moves to ACCUMULATE ON DIPS when: " + "; ".join(triggers) + "."
    if canonical_state == "BUY ONLY AFTER CONFIRMATION":
        return "Moves to ACCUMULATE GRADUALLY on confirmed earnings beat or volume breakout above resistance."
    if canonical_state == "ACCUMULATE ON DIPS":
        return "Moves to ACCUMULATE GRADUALLY on price reaching entry zone with sustained volume."
    if canonical_state == "ACCUMULATE GRADUALLY":
        return "Moves to ACTIONABLE BUY on strong earnings beat or sector re-rating catalyst."
    return ""  # ACTIONABLE BUY — already top
