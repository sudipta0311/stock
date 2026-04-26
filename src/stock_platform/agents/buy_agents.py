from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import timedelta
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

# Ideal sector allocation targets for an aggressive growth-oriented portfolio.
# Used to override index-composition-based targets when risk_profile is "Aggressive",
# ensuring gap analysis reflects what an aggressive investor should hold — not just
# what the index holds.
AGGRESSIVE_SECTOR_TARGETS: dict[str, float] = {
    "Capital Goods":            12.0,
    "Defence":                  10.0,
    "Industrials":              10.0,
    "Chemicals":                 8.0,
    "Auto":                      8.0,
    "Auto Components":           5.0,
    "Banking":                  20.0,
    "Financial Services":       15.0,
    "IT":                       15.0,
    "Technology":               15.0,
    "FMCG":                      8.0,
    "Consumer":                  8.0,
    "Healthcare":                6.0,
    "Pharma":                    6.0,
    "Energy":                    5.0,
    "Infrastructure":            8.0,
    "Metals":                    5.0,
    "Real Estate":               4.0,
    "Telecom":                   4.0,
    "Power":                     6.0,
    "Cement":                    4.0,
}


def _compute_aggressive_gaps(
    identified_gaps: list[dict],
    sector_weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    """
    Re-derive gap entries for an aggressive portfolio using AGGRESSIVE_SECTOR_TARGETS.
    Supplements existing gap entries; sectors already present keep their scores.
    Returns a dict keyed by sector.
    """
    result: dict[str, dict] = {g["sector"]: g for g in identified_gaps}

    # Build a minimal sector_weights map from what's already in gap entries
    _existing_exposure: dict[str, float] = {}
    for g in identified_gaps:
        _existing_exposure[g["sector"]] = g.get("underweight_pct", 0.0)
    if sector_weights:
        _existing_exposure.update(sector_weights)

    for sector, target_pct in AGGRESSIVE_SECTOR_TARGETS.items():
        existing_pct = _existing_exposure.get(sector, 0.0)
        gap_pct = round(max(0.0, target_pct - existing_pct), 2)
        gap_status = (
            "WIDE" if gap_pct > 5
            else "MODERATE" if gap_pct > 2
            else "MINIMAL"
        )
        if gap_pct == 0 and existing_pct == 0:
            _log.warning(
                "GAP_ANOMALY %s: target sector has 0 portfolio exposure and 0 gap — "
                "check sector classification in holdings data",
                sector,
            )
        overlap_note = (
            f"GAP: only {existing_pct:.1f}% existing — {gap_status} opportunity"
        )
        reason = f"{overlap_note} | target {target_pct:.1f}% (aggressive) | gap {gap_pct:.1f}%"
        score = round(min(1.0, gap_pct / max(target_pct, 1.0)), 3)
        conviction = (
            "STRONG_BUY" if gap_pct > 5
            else "BUY" if gap_pct > 2
            else "NEUTRAL"
        )
        if sector not in result:
            result[sector] = {
                "sector":          sector,
                "underweight_pct": gap_pct,
                "gap_pct":         gap_pct,
                "target_pct":      target_pct,
                "conviction":      conviction,
                "score":           score,
                "reason":          reason,
            }
        else:
            # Update existing entry with aggressive-calibrated reason, score, and target
            result[sector]["reason"]     = reason
            result[sector]["target_pct"] = target_pct
            result[sector]["gap_pct"]    = max(result[sector].get("gap_pct", 0.0), gap_pct)
            result[sector]["score"]      = max(result[sector].get("score", 0.0), score)

    return result


def _track_quant_llm_disagreement(results: list[dict]) -> None:
    """Log a warning when quant and LLM verdicts disagree systematically."""
    if not results:
        return
    disagreements = [
        r for r in results
        if r.get("quant_verdict") and r.get("llm_verdict")
        and r["quant_verdict"] != r["llm_verdict"]
    ]
    if not disagreements:
        return
    ratio = len(disagreements) / len(results)
    if ratio > 0.5:
        _log.warning(
            "QUANT_LLM_MISCALIBRATION: %d/%d stocks disagree (%d%%). "
            "One model is systematically biased — review prompts.",
            len(disagreements), len(results), int(ratio * 100),
        )
    quant_more_aggressive = sum(
        1 for r in disagreements
        if r.get("quant_verdict") in {"ACCUMULATE GRADUALLY", "ACCUMULATE", "SMALL INITIAL"}
        and r.get("llm_verdict") in {"WATCHLIST", "WAIT"}
    )
    if disagreements and quant_more_aggressive / len(disagreements) > 0.7:
        _log.warning(
            "QUANT consistently more aggressive than LLM (%d/%d disagreements). "
            "Either quant scoring threshold is too low or LLM prompt is too cautious.",
            quant_more_aggressive, len(disagreements),
        )

from stock_platform.config import AppConfig
from stock_platform.models import RecommendationRecord
from stock_platform.agents.quant_model import apply_freshness_cap, compute_quality_score as compute_quality_score_v2
from stock_platform.utils.risk_profiles import UNIVERSAL_HARD_EXCLUDE, get_risk_config
from stock_platform.utils.entry_calculator import (
    KNOWN_ANALYST_TARGETS,
    apply_momentum_override,
    calculate_entry_levels,
    fetch_analyst_consensus_target,
)
from stock_platform.utils.rules import clamp, parse_iso_datetime
from stock_platform.utils.pe_history_fetcher import get_pe_historical_context
from stock_platform.utils.sector_config import governance_risk_blocks
from stock_platform.utils.signal_sources import get_tariff_signal
from stock_platform.utils.stock_validator import (
    ValidationResult,
    check_recently_listed,
    log_skipped_stock,
    validate_stock,
)
from stock_platform.utils.technical_signals import compute_technical_signal
from stock_platform.utils.fii_dii_fetcher import fetch_fii_dii_sector_flow
from stock_platform.utils.valuation_reliability import get_valuation_reliability
from stock_platform.utils.evidence_scoring import compute_evidence_strength
from concurrent.futures import ThreadPoolExecutor, as_completed
from stock_platform.utils.stock_context import (
    build_factual_snapshot,
    format_snapshot_for_prompt,
)


MINIMUM_RR_RATIO = 1.2
SHORTLIST_BUFFER_MULTIPLIER = 8
FINAL_BUFFER_MULTIPLIER = 6


def _quality_sort_key(candidate: dict[str, Any]) -> float:
    """Quality-focused sort for the Anthropic (bear-biased risk analyst) candidate pool.
    Prioritises ROCE, low leverage, and overall fundamental quality score."""
    fin = candidate.get("financials") or candidate.get("live_financials") or {}
    roce = float(fin.get("roce_pct") or 0)
    debt = max(float(fin.get("debt_to_equity") or fin.get("debtToEquity") or 1), 0.01)
    qs   = float(candidate.get("quality_score") or 0)
    return roce * 0.40 + qs * 0.30 + (1.0 / debt) * 0.30


def _momentum_sort_key(candidate: dict[str, Any]) -> float:
    """Momentum/catalyst-focused sort for the OpenAI (bull-biased catalyst analyst) pool.
    Prioritises revenue growth, technical momentum score, and entry timing."""
    fin        = candidate.get("financials") or candidate.get("live_financials") or {}
    rev_growth = float(fin.get("revenue_growth_pct") or 0)
    tech_score = float(candidate.get("technical_score") or 0)
    sel_score  = float(candidate.get("selection_score") or 0)
    return rev_growth * 0.40 + tech_score * 0.35 + sel_score * 0.25


def _apply_momentum_exclusions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hard exclusion for the OpenAI momentum pool.

    Rejects stocks where revenue AND earnings are declining while the stock
    sits near its 52W high — a sign that price momentum has decoupled from
    fundamentals and the thesis rests on price action alone.
    All four conditions must hold to exclude (conservative gate).
    """
    eligible: list[dict[str, Any]] = []
    for c in candidates:
        fin        = c.get("live_financials") or c.get("financials") or {}
        rev_growth = float(fin.get("revenue_growth_pct") or 0)
        pat_info   = fin.get("pat_momentum") or {}
        pat_growth = float(pat_info.get("pat_growth_pct") or 0)
        current    = float(fin.get("currentPrice") or fin.get("current_price") or 0)
        w52_high   = float(fin.get("week52_high") or fin.get("fiftyTwoWeekHigh") or 0)
        near_high  = (
            w52_high > 0 and current > 0
            and (w52_high - current) / w52_high < 0.05
        )
        if rev_growth < -5.0 and pat_growth < -10.0 and near_high:
            print(
                f"MOMENTUM EXCLUDED: {c.get('symbol')} — "
                f"rev {rev_growth:.1f}% PAT {pat_growth:.1f}% "
                f"within 5% of 52W high ({current:.0f}/{w52_high:.0f})"
            )
            continue
        eligible.append(c)
    return eligible


PROMOTER_GROUPS: dict[str, list[str]] = {
    "ADANI": [
        "ADANIENT", "ADANIPORTS", "ADANIPOWER", "ADANIGREEN",
        "ADANITRANS", "ADANIGAS", "ADANIWILMAR", "NDTV",
    ],
    "TATA": [
        "TCS", "TATAMOTORS", "TATASTEEL", "TATAPOWER",
        "TATACONSUM", "TITAN", "TATACOMM", "TATACHEM",
    ],
    "RELIANCE": ["RELIANCE", "JIOFINANCE"],
    "BAJAJ": ["BAJFINANCE", "BAJAJFINSV", "BAJAJ-AUTO", "BAJAJELEC"],
}

_SYMBOL_TO_GROUP: dict[str, str] = {
    sym: grp for grp, syms in PROMOTER_GROUPS.items() for sym in syms
}


def apply_group_concentration_check(
    candidates: list[dict[str, Any]],
    max_per_group: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep at most max_per_group stocks from the same promoter group.

    Candidates must already be sorted by score descending (highest first).
    Returns (filtered, deferred) preserving original order within each bucket.
    """
    seen_groups: dict[str, int] = {}
    filtered: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    sorted_cands = sorted(
        candidates,
        key=lambda x: x.get("differentiation_score", x.get("quality_score", 0)),
        reverse=True,
    )
    for stock in sorted_cands:
        symbol = stock.get("symbol", "")
        group = _SYMBOL_TO_GROUP.get(symbol)
        if group:
            if seen_groups.get(group, 0) < max_per_group:
                seen_groups[group] = seen_groups.get(group, 0) + 1
                filtered.append(stock)
            else:
                kept = [s["symbol"] for s in filtered if _SYMBOL_TO_GROUP.get(s["symbol"]) == group]
                deferred.append(
                    stock | {
                        "deferred_reason": (
                            f"Group concentration: {group} already represented by {kept}"
                        )
                    }
                )
        else:
            filtered.append(stock)
    return filtered, deferred


def get_top_n_with_replacement(
    scored_candidates: list[dict[str, Any]],
    n: int,
    skipped_symbols: list[str],
    db_path: str,
    sort_key=None,
) -> list[dict[str, Any]]:
    """
    Keep fetching the next-best validated candidates until we have N rows
    or exhaust the candidate list.
    sort_key: optional callable(candidate)->float; defaults to selection_score.
    """
    from stock_platform.utils.screener_fetcher import fetch_screener_data

    recommendations: list[dict[str, Any]] = []
    attempted = {str(symbol).upper() for symbol in skipped_symbols}

    _key = sort_key or (lambda row: row.get("selection_score", row.get("quality_score", 0)))
    sorted_candidates = sorted(scored_candidates, key=_key, reverse=True)

    for candidate in sorted_candidates:
        if len(recommendations) >= n:
            break

        symbol = str(candidate.get("symbol", "")).upper().strip()
        if not symbol or symbol in attempted:
            continue

        attempted.add(symbol)
        fin_data = candidate.get("financials") or candidate.get("live_financials") or {}
        if not fin_data:
            fin_data = fetch_screener_data(symbol)
        current_price = (
            candidate.get("current_price")
            or candidate.get("live_financials", {}).get("currentPrice")
            or candidate.get("live_financials", {}).get("current_price")
            or fin_data.get("current_price")
        )

        result = validate_stock(symbol, fin_data, current_price)
        if result.can_recommend:
            recommendations.append(
                candidate
                | {
                    "financials": fin_data,
                    "live_financials": candidate.get("live_financials") or fin_data,
                    "current_price": current_price,
                }
            )
        else:
            print(f"Skipped {symbol}: {result.reason}")

    if len(recommendations) < n:
        print(
            f"WARNING: Only found {len(recommendations)} valid candidates out of requested {n}. "
            "Try a broader index like NIFTY 200."
        )

    return recommendations


def compute_position_size(
    entry_signal: str,
    conviction_score: float,
    direct_equity_corpus: float,
    conviction_multiplier: float = 1.0,
    risk_profile: str = "Balanced",
) -> dict[str, Any]:
    """
    Returns initial_tranche_pct and target_pct separately.
    Initial tranche is what to deploy NOW.
    Target is the full intended position over 2-3 months.

    Profile sizing multipliers:
      Conservative — smaller initial tranches, lower target caps (preservation over growth)
      Balanced     — standard sizing (1× base)
      Aggressive   — larger initial tranches, higher target caps (concentration tolerated)
    """
    # Map timing signals from assess_timing to conviction tiers
    signal_map = {
        "STRONG ENTER": "STRONG_BUY",
        "ACCUMULATE":   "BUY",
        "SMALL INITIAL": "ACCUMULATE",
        "WAIT":          "WAIT",
    }
    conviction_key = signal_map.get(entry_signal, "ACCUMULATE")

    # Base sizing rules (Balanced / default)
    sizing_rules: dict[str, dict[str, float]] = {
        "STRONG_BUY": {"initial": 0.10, "target": 0.28},
        "BUY":        {"initial": 0.08, "target": 0.22},
        "ACCUMULATE": {"initial": 0.06, "target": 0.18},
        "WAIT":       {"initial": 0.00, "target": 0.10},
    }

    # Profile-specific size scaling and per-position caps
    _PROFILE_SCALE: dict[str, dict[str, float]] = {
        "Conservative": {"initial_scale": 0.65, "target_scale": 0.75, "hard_cap_pct": 20.0},
        "Balanced":     {"initial_scale": 1.00, "target_scale": 1.00, "hard_cap_pct": 25.0},
        "Aggressive":   {"initial_scale": 1.30, "target_scale": 1.20, "hard_cap_pct": 30.0},
    }
    profile_cfg = _PROFILE_SCALE.get(risk_profile, _PROFILE_SCALE["Balanced"])

    rule = sizing_rules.get(conviction_key, {"initial": 0.05, "target": 0.10})

    conviction_multiplier = max(0.0, conviction_multiplier) * (0.85 + (conviction_score * 0.30))

    initial_pct = round(
        rule["initial"] * conviction_multiplier * profile_cfg["initial_scale"] * 100, 1
    )
    target_pct = round(
        rule["target"] * conviction_multiplier * profile_cfg["target_scale"] * 100, 1
    )

    hard_cap_pct = profile_cfg["hard_cap_pct"]
    initial_pct = min(initial_pct, hard_cap_pct)
    target_pct  = min(target_pct,  hard_cap_pct)

    initial_amount = round(direct_equity_corpus * initial_pct / 100, 0)
    target_amount  = round(direct_equity_corpus * target_pct  / 100, 0)

    return {
        "initial_tranche_pct": initial_pct,
        "target_pct": target_pct,
        "initial_amount_inr": initial_amount,
        "target_amount_inr": target_amount,
        "rule_applied": entry_signal,
    }


def compute_net_return(
    current_price: float,
    analyst_target: float | None,
    holding_months: int = 24,
) -> float:
    """Stock-specific net return (%) after LTCG/STCG tax, based on analyst target price."""
    if not current_price or current_price <= 0:
        return 0.0
    if not analyst_target or analyst_target <= 0:
        analyst_target = current_price * 1.15
    gross_return = (analyst_target - current_price) / current_price
    tax_rate = 0.125 if holding_months >= 12 else 0.20
    return round(gross_return * (1 - tax_rate) * 100, 2)


def filter_by_risk_reward(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    valid = []
    excluded = []
    for candidate in candidates:
        entry = candidate.get("entry_levels", {})
        rr = float(entry.get("risk_reward", 0) or 0)
        if rr < MINIMUM_RR_RATIO:
            excluded.append(candidate["symbol"])
            print(
                f"EXCLUDED {candidate['symbol']}: "
                f"R/R {rr}x below minimum {MINIMUM_RR_RATIO}x"
            )
        else:
            valid.append(candidate)
    return valid, excluded


def buffered_top_n(top_n: int) -> int:
    """Keep extra candidates alive so strict entry/RR filters can still backfill Top N."""
    return max(top_n * FINAL_BUFFER_MULTIPLIER, top_n)


def ensure_minimum_candidates(
        candidates: list[dict[str, Any]],
        top_n: int,
        risk_profile: str,
        all_scored: list[dict[str, Any]],
        sort_key=None) -> list[dict[str, Any]]:
    """
    Guarantees at least top_n candidates reach the LLM layer.

    When strict universal filtering leaves fewer than top_n candidates,
    backfills from the broader scored universe (stocks that passed quant
    scoring but were removed by filter_risk).  Backfill candidates are
    flagged so the LLM knows to apply extra scrutiny.
    """
    if len(candidates) >= top_n:
        return candidates
    shortfall = top_n - len(candidates)
    existing_symbols = {c.get("symbol") for c in candidates}
    _key = sort_key or (lambda row: row.get("selection_score", row.get("quality_score", 0)))
    backfill_pool = sorted(
        [s for s in all_scored if s.get("symbol") not in existing_symbols],
        key=_key,
        reverse=True,
    )[:shortfall]
    for s in backfill_pool:
        s["backfill_flag"] = True
        s["backfill_reason"] = (
            f"Added to meet Top N={top_n} — did not pass universal hard-exclusion "
            f"filter for {risk_profile} profile. Apply extra scrutiny."
        )
    result = candidates + backfill_pool
    if backfill_pool:
        _log.info(
            "Candidate pool backfill: %d primary + %d backfill = %d total "
            "(top_n=%d, profile=%s)",
            len(candidates), len(backfill_pool), len(result), top_n, risk_profile,
        )
    return result


class BuyAgents:
    def __init__(self, repo: Any, provider: Any, config: AppConfig, llm: Any) -> None:
        self.repo = repo
        self.provider = provider
        self.config = config
        self.llm = llm

    def load_portfolio_gate(self, state: dict[str, Any]) -> dict[str, Any]:
        context = self.repo.load_portfolio_context()
        updated = parse_iso_datetime(context["portfolio_meta"].get("portfolio_last_updated"))
        cutoff = self.provider.today - timedelta(days=self.config.max_portfolio_age_days)
        if not context["normalized_exposure"]:
            raise ValueError("Portfolio data is missing. Upload holdings before running buy analysis.")
        if not updated or updated.date() < cutoff:
            raise ValueError(
                f"Portfolio data is older than {self.config.max_portfolio_age_days} days. Refresh ingestion first."
            )
        return {"portfolio_context": context}

    def discover_universe(self, state: dict[str, Any]) -> dict[str, Any]:
        request = state["request"]
        return {"universe": self.provider.get_index_members(request["index_name"])}

    def recommend_industries(self, state: dict[str, Any]) -> dict[str, Any]:
        context = state["portfolio_context"]
        gaps = context["identified_gaps"]
        exposure = context["normalized_exposure"]
        sector_weights: dict[str, float] = {}
        for row in exposure:
            sector_weights[row["sector"]] = sector_weights.get(row["sector"], 0.0) + row["total_weight"]
        industries: list[dict[str, Any]] = []
        for gap in gaps:
            sector = gap["sector"]
            if sector_weights.get(sector, 0.0) > 25:
                continue
            if gap["conviction"] == "STRONG_AVOID":
                continue
            market_signal = self.provider.get_current_market_signal(sector)
            score = round(gap["score"] * 0.35 + market_signal * 0.25 + min(1.0, gap["score"] + 0.2) * 0.4, 3)
            industries.append(
                {
                    "sector": sector,
                    "score": score,
                    "market_signal": market_signal,
                    "reason": gap["reason"],
                }
            )
        industries.sort(key=lambda row: row["score"], reverse=True)
        top_industries = industries[:6]

        # LLM:Sonnet — one-shot sector prioritization narrative for the investor.
        macro_thesis = context.get("user_preferences", {}).get("macro_thesis", "")
        portfolio_summary = (
            f"{len(exposure)} holdings across "
            f"{len({r['sector'] for r in exposure})} sectors"
        )
        llm_narrative = self.llm.industry_reasoning(top_industries, macro_thesis, portfolio_summary)
        return {
            "preferred_industries": top_industries,
            "industry_narrative": llm_narrative or "",
        }

    def generate_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
        preferred = {row["sector"] for row in state["preferred_industries"]}
        context = state["portfolio_context"]
        held_symbols = {row["symbol"] for row in context["normalized_exposure"]}
        overlap_map = {row["symbol"]: row for row in context["overlap_scores"]}
        candidates: list[dict[str, Any]] = []
        for stock in state["universe"]:
            if stock["sector"] not in preferred:
                continue
            if stock["symbol"] in held_symbols:
                continue
            if overlap_map.get(stock["symbol"], {}).get("overlap_pct", 0.0) > 3.0:
                continue
            candidates.append(stock)
        return {"candidates": candidates}

    def score_quality(self, state: dict[str, Any]) -> dict[str, Any]:
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        flow = self.repo.list_signals("flow")
        accumulation_bonus = 0.05 if any(float(row.get("score", 0.0)) >= 0.65 for row in flow) else 0.0
        validation_run_id = f"val-{uuid4().hex[:8]}"
        scored: list[dict[str, Any]] = []
        skipped: list[ValidationResult] = []
        # Parallel pre-fetch: warm the provider's in-memory financial cache before the serial scoring loop.
        _all_syms = [c["symbol"] for c in state["candidates"]]
        with ThreadPoolExecutor(max_workers=8) as _pex:
            _pfuts = {_pex.submit(self.provider.get_financials, s): s for s in _all_syms}
            for _pf in as_completed(_pfuts):
                try:
                    _pf.result()
                except Exception as _pe:
                    _log.debug("pre-fetch financials %s: %s", _pfuts[_pf], _pe)
        for candidate in state["candidates"]:
            live_facts = self.provider.get_financials(candidate["symbol"])
            current_price = live_facts.get("currentPrice") or live_facts.get("current_price")

            # Safety gate: block unresolvable stocks before any scoring or LLM call.
            vr = validate_stock(candidate["symbol"], live_facts, current_price)
            if not vr.can_recommend:
                skipped.append(vr)
                log_skipped_stock(
                    str(self.config.db_path),
                    vr,
                    validation_run_id,
                    neon_database_url=self.config.neon_database_url,
                )
                print(f"SKIPPED {candidate['symbol']}: {vr.reason}")
                continue

            base_score = compute_quality_score_v2(candidate["symbol"], live_facts)

            # Technical timing signals — 20% blend on top of fundamental quality.
            tech = compute_technical_signal(candidate["symbol"], live_facts, current_price or 0.0)
            blended_quality = base_score * 0.80 + tech["technical_score"] * 0.20

            # Geo signal bonus — applied on top of the computed base (small additive).
            geo_bonus = 0.04 if unified.get(candidate["sector"], {}).get("conviction") in {"BUY", "STRONG_BUY"} else 0.0
            # Flow / accumulation bonus.
            flow_bonus = accumulation_bonus
            selection_score = clamp(blended_quality + geo_bonus + flow_bonus, 0.0, 1.0)
            scored.append(candidate | {
                "quality_score": round(base_score, 3),
                "selection_score": round(selection_score, 3),
                "technical_signals": tech["signals"],
                "technical_score": tech["technical_score"],
                "financials": live_facts,
                "live_financials": live_facts,
            })
        scored.sort(key=lambda row: row.get("selection_score", row["quality_score"]), reverse=True)
        return {
            "scored_candidates": scored,
            "skipped_candidates": skipped,
            "validation_run_id": validation_run_id,
        }

    def filter_risk(self, state: dict[str, Any]) -> dict[str, Any]:
        from stock_platform.utils.risk_profiles import get_risk_config
        sector_weights: dict[str, float] = {}
        for row in state["portfolio_context"]["normalized_exposure"]:
            sector_weights[row["sector"]] = sector_weights.get(row["sector"], 0.0) + row["total_weight"]
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        filtered = []
        universe_size = len(state.get("universe", []))
        scored_size = len(state.get("scored_candidates", []))

        risk_profile = state["request"].get("risk_profile", "Balanced")
        _risk_cfg = get_risk_config(risk_profile)
        _preferred_min_roce   = float(_risk_cfg.get("preferred_min_roce", 18))
        _preferred_max_de     = float(_risk_cfg.get("preferred_max_de", 0.5))
        _preferred_max_pe_pct = float(_risk_cfg.get("preferred_max_pe_vs_median_pct", 25))
        _preferred_min_rev_gr = float(_risk_cfg.get("preferred_min_revenue_growth", 12))

        for candidate in state["scored_candidates"]:
            risk = self.provider.get_risk_metrics(candidate["symbol"])

            # ── TIER 1: UNIVERSAL HARD EXCLUSIONS (same for all profiles) ─────
            if (risk.get("avg_daily_value_cr") is not None
                    and risk["avg_daily_value_cr"] < UNIVERSAL_HARD_EXCLUDE["min_avg_daily_volume_cr"]):
                continue
            if risk.get("beta") is not None and risk["beta"] > 2.0:
                continue
            pledge = risk.get("pledge_pct") or risk.get("promoter_pledge_pct")
            if pledge is not None and pledge > UNIVERSAL_HARD_EXCLUDE["max_promoter_pledge_pct"]:
                candidate["hard_exclude"] = True
                candidate["exclude_reason"] = (
                    f"Promoter pledge {pledge:.1f}% exceeds universal 50% threshold — "
                    f"extreme margin call risk"
                )
                continue
            # Extreme leverage — solvency risk regardless of profile
            fin = candidate.get("financials") or candidate.get("live_financials") or {}
            _de = fin.get("debt_to_equity") or fin.get("debtToEquity")
            if _de is not None:
                try:
                    if float(_de) > UNIVERSAL_HARD_EXCLUDE["max_debt_equity"]:
                        continue
                except (TypeError, ValueError):
                    pass
            # Micro-cap — too thin for meaningful position sizing
            _mcap = fin.get("market_cap_cr") or fin.get("marketCapCr")
            if _mcap is not None:
                try:
                    if float(_mcap) < UNIVERSAL_HARD_EXCLUDE["min_market_cap_cr"]:
                        continue
                except (TypeError, ValueError):
                    pass

            if risk.get("sebi_flag"):
                continue
            if sector_weights.get(candidate["sector"], 0.0) > self.config.max_sector_pct:
                continue
            if unified.get(candidate["sector"], {}).get("conviction") == "STRONG_AVOID":
                continue

            # ── TIER 2: PROFILE SCORING MODIFIERS (never exclude — only downgrade score) ─
            # Penalties are larger for Conservative, smaller for Aggressive — reflecting
            # how strictly each profile cares about these quality thresholds.
            _penalty_scale = {"Conservative": 0.08, "Balanced": 0.05, "Aggressive": 0.02}
            _penalty = _penalty_scale.get(risk_profile, 0.05)
            _profile_score_adj = 0.0

            _roce = fin.get("roce") or fin.get("return_on_capital_employed")
            if _roce is not None:
                try:
                    if float(_roce) < _preferred_min_roce:
                        _profile_score_adj -= _penalty
                except (TypeError, ValueError):
                    pass

            if _de is not None:
                try:
                    if float(_de) > _preferred_max_de:
                        _profile_score_adj -= _penalty
                except (TypeError, ValueError):
                    pass

            _rev_gr = (
                fin.get("revenue_growth_yoy")
                or fin.get("revenue_yoy_growth_pct")
                or (fin.get("recent_results") or {}).get("revenue_yoy_growth_pct")
            )
            if _rev_gr is not None:
                try:
                    if float(_rev_gr) < _preferred_min_rev_gr:
                        _profile_score_adj -= _penalty * 0.5
                except (TypeError, ValueError):
                    pass

            # PE premium vs median — only apply when PE history is meaningful
            _pe_signal = (candidate.get("pe_signal") or "").upper()
            if _pe_signal == "EXPENSIVE_VS_HISTORY":
                _pe_premium = fin.get("pe_premium_vs_median_pct")
                if _pe_premium is not None:
                    try:
                        if float(_pe_premium) > _preferred_max_pe_pct:
                            _profile_score_adj -= _penalty
                    except (TypeError, ValueError):
                        pass

            _adj_selection_score = round(
                clamp(candidate.get("selection_score", candidate.get("quality_score", 0.0)) + _profile_score_adj, 0.0, 1.0),
                3,
            )

            filtered.append(
                candidate
                | {
                    "risk_metrics": risk,
                    "profile_score_adj": round(_profile_score_adj, 3),
                    "selection_score": _adj_selection_score,
                }
            )

        _log.info(
            "filter_risk: universe=%d scored=%d post_exclusion=%d",
            universe_size, scored_size, len(filtered),
        )
        return {
            "risk_filtered_candidates": filtered,
            "pipeline_stats": {
                "universe_size": universe_size,
                "post_scoring_count": scored_size,
                "post_exclusion_count": len(filtered),
            },
        }

    def shortlist(self, state: dict[str, Any]) -> dict[str, Any]:
        top_n = int(state["request"]["top_n"])
        risk_profile = state["request"].get("risk_profile", "Balanced")
        skipped_symbols = [result.symbol for result in state.get("skipped_candidates", [])]
        # Each model gets a pool ordered by its own analytical lens so genuine
        # divergence can surface: Anthropic (risk/quality) vs OpenAI (momentum/catalyst).
        sort_key = (
            _momentum_sort_key if self.llm.provider == "openai"
            else _quality_sort_key
        )
        target = max(top_n * SHORTLIST_BUFFER_MULTIPLIER, top_n)
        primary = get_top_n_with_replacement(
            state["risk_filtered_candidates"],
            target,
            skipped_symbols,
            str(self.config.db_path),
            sort_key=sort_key,
        )
        # Guarantee at least top_n candidates reach the LLM layer even when
        # universal hard exclusions left too few candidates in the primary pool.
        full_shortlist = ensure_minimum_candidates(
            primary,
            top_n,
            risk_profile,
            state.get("scored_candidates", []),
            sort_key=sort_key,
        )
        if self.llm.provider == "openai":
            full_shortlist = _apply_momentum_exclusions(full_shortlist)
        return {"shortlist": full_shortlist}

    def validate_qualitative(self, state: dict[str, Any]) -> dict[str, Any]:
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        top_n = int(state["request"]["top_n"])
        target_pool_size = buffered_top_n(top_n)
        approved = []
        fallback_pool = []
        for candidate in state["shortlist"]:
            news = self.provider.get_stock_news(candidate["symbol"])
            signal_context = unified.get(candidate["sector"], {})

            # LLM:Sonnet — structured qualitative validation with JSON output.
            llm_result = self.llm.qualitative_analysis(candidate, news, signal_context)
            if llm_result is not None:
                if not llm_result["approved"]:
                    if news["sentiment_score"] >= -0.1:
                        fallback_pool.append(
                            candidate | {
                                "validation_confidence": round(
                                    clamp(
                                        candidate["quality_score"] * 0.7
                                        + max(news["sentiment_score"], 0) * 0.3,
                                        0.0,
                                        1.0,
                                    ),
                                    3,
                                ),
                                "news": news,
                                "validation_reasoning": (
                                    llm_result["reasoning"]
                                    or "Fallback used after qualitative rejection."
                                ),
                            }
                        )
                    continue
                confidence = round(clamp(llm_result["confidence"], 0.0, 1.0), 3)
            else:
                # Deterministic fallback: rule-based sentiment scoring.
                if news["sentiment_score"] < -0.1:
                    continue
                confidence = round(
                    clamp(candidate["quality_score"] * 0.7 + max(news["sentiment_score"], 0) * 0.3, 0.0, 1.0),
                    3,
                )

            approved.append(
                candidate | {
                    "validation_confidence": confidence,
                    "news": news,
                    "validation_reasoning": llm_result["reasoning"] if llm_result else "",
                }
            )
        if len(approved) < target_pool_size and fallback_pool:
            used_symbols = {row["symbol"] for row in approved}
            fallback_pool.sort(
                key=lambda row: (
                    row.get("validation_confidence", 0.0),
                    row.get("selection_score", row.get("quality_score", 0.0)),
                ),
                reverse=True,
            )
            for candidate in fallback_pool:
                if len(approved) >= target_pool_size:
                    break
                if candidate["symbol"] in used_symbols:
                    continue
                approved.append(candidate)
                used_symbols.add(candidate["symbol"])

        if not approved:
            print("WARNING: qualitative validation approved zero candidates - buy feed will be empty.")

        return {"shortlist": approved[:target_pool_size]}

    def differentiate_portfolio(self, state: dict[str, Any]) -> dict[str, Any]:
        overlaps = {row["symbol"]: row for row in state["portfolio_context"]["overlap_scores"]}
        gaps = {row["sector"]: row for row in state["portfolio_context"]["identified_gaps"]}
        top_n = int(state["request"]["top_n"])
        target_pool_size = buffered_top_n(top_n)
        differentiated = []
        overlap_filtered: list[str] = []
        for candidate in state["shortlist"]:
            overlap = overlaps.get(candidate["symbol"], {"overlap_pct": 0.0, "band": "GREEN", "attribution": []})
            if overlap["overlap_pct"] > 3:
                overlap_filtered.append(candidate["symbol"])
                continue
            differentiation_score = round(
                clamp(
                    candidate.get("selection_score", candidate["quality_score"])
                    - overlap["overlap_pct"] * 0.08
                    + gaps.get(candidate["sector"], {}).get("score", 0.3) * 0.2,
                    0.0,
                    1.0,
                ),
                3,
            )
            differentiated.append(
                candidate
                | {
                    "overlap_pct": overlap["overlap_pct"],
                    "fund_attribution": overlap.get("attribution", []),
                    "differentiation_score": differentiation_score,
                    "gap_reason": gaps.get(candidate["sector"], {}).get("reason", "Sector diversification benefit"),
                }
            )
        differentiated.sort(key=lambda row: row["differentiation_score"], reverse=True)
        filtered_diff, group_deferred = apply_group_concentration_check(
            differentiated[:target_pool_size]
        )
        return {
            "differentiated_shortlist": filtered_diff,
            "overlap_filtered": overlap_filtered,
            "group_deferred": group_deferred,
        }

    def assess_timing(self, state: dict[str, Any]) -> dict[str, Any]:
        risk_profile = state["request"].get("risk_profile", "Balanced")
        contrarian = {row["sector"]: row for row in self.repo.list_signals("contrarian")}
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        timing_rows = []
        for candidate in state["differentiated_shortlist"]:
            price = self.provider.get_price_context(candidate["symbol"])
            contra = contrarian.get(candidate["sector"], {})
            geo = unified.get(candidate["sector"], {})
            if geo.get("conviction") == "STRONG_AVOID":
                entry = "DO NOT ENTER"
            elif geo.get("conviction") in {"BUY", "STRONG_BUY"} and contra.get("conviction") == "BUY":
                entry = "STRONG ENTER"
            elif geo.get("conviction") in {"BUY", "STRONG_BUY"}:
                entry = "ACCUMULATE"
            elif contra.get("conviction") == "BUY":
                entry = "SMALL INITIAL"
            else:
                entry = "WAIT"
            baseline_entry_signal = entry
            fin_data = candidate.get("financials") or candidate.get("live_financials") or {}
            recent_results = fin_data.get("recent_results") or {}
            entry = apply_momentum_override(
                signal=entry,
                recent_results=recent_results,
                current_price=price.get("price"),
                week52_low=fin_data.get("week52_low") or fin_data.get("fiftyTwoWeekLow"),
            )
            lock_in_check = check_recently_listed(candidate["symbol"], current_date=self.provider.today)
            lock_in_multiplier = 1.0
            if lock_in_check.get("recently_listed"):
                lock_in_multiplier = 0.5
                if entry in {"STRONG ENTER", "ACCUMULATE", "SMALL INITIAL"}:
                    entry = "WAIT"
            # Freshness cap: thresholds driven by risk_profile
            entry = apply_freshness_cap(entry, fin_data, risk_profile)
            timing_rows.append(
                candidate
                | {
                    "entry_signal": entry,
                    "original_entry_signal": baseline_entry_signal,
                    "price_context": price,
                    "recent_results": recent_results,
                    "revenue_momentum": fin_data.get("revenue_momentum") or recent_results,
                    "pat_momentum": fin_data.get("pat_momentum") or {},
                    "momentum_override_applied": baseline_entry_signal == "WAIT" and entry == "ACCUMULATE",
                    "lock_in_check": lock_in_check,
                    "lock_in_multiplier": lock_in_multiplier,
                }
            )
        return {"timing_assessments": timing_rows}

    def size_positions(self, state: dict[str, Any]) -> dict[str, Any]:
        prefs = state["portfolio_context"]["user_preferences"]
        corpus = float(prefs.get("direct_equity_corpus", 0) or 0) or 100.0
        risk_profile = state["request"].get("risk_profile", "Balanced")
        allocations = []
        for item in state["timing_assessments"]:
            sizing = compute_position_size(
                item["entry_signal"],
                item["quality_score"],
                corpus,
                conviction_multiplier=float(item.get("lock_in_multiplier", 1.0) or 1.0),
                risk_profile=risk_profile,
            )
            allocations.append(
                item
                | sizing
                | {
                    # Keep legacy key for backward-compat with any stored recommendation reads.
                    "allocation_pct": sizing["initial_tranche_pct"],
                    "allocation_amount": sizing["initial_amount_inr"],
                    "tranches": 3 if item["entry_signal"] in {"STRONG ENTER", "ACCUMULATE"} else 2,
                }
            )
        return {"allocations": allocations}

    def assess_tax_costs(self, state: dict[str, Any]) -> dict[str, Any]:
        horizon_months = int(state["request"].get("horizon_months", 24))
        applicable_rate = 12.5 if horizon_months >= 12 else 20.0
        tax_regime = "LTCG (12.5%)" if horizon_months >= 12 else "STCG (20%)"
        return {
            "tax_assessment": {
                "stcg_rate": 20.0,
                "ltcg_rate": 12.5,
                "applicable_rate": applicable_rate,
                "horizon_months": horizon_months,
                "cost_drag_pct": 0.15,
                "note": (
                    f"Horizon {horizon_months}m → {tax_regime} applies on exit. "
                    "New-buy flow applies entry-side cost drag only."
                ),
            }
        }

    def check_confidence(self, state: dict[str, Any]) -> dict[str, Any]:
        unified = self.repo.list_signals("unified")
        signal_count = len(unified)
        exposure_count = len(state["portfolio_context"]["normalized_exposure"])

        # Market-level confidence: look at the distribution of unified signal scores.
        if unified:
            avg_score = sum(float(s.get("score", 0.5)) for s in unified) / signal_count
            avoid_count = sum(
                1 for s in unified
                if s.get("conviction", "NEUTRAL") in ("AVOID", "STRONG_AVOID")
            )
            avoid_ratio = avoid_count / signal_count
        else:
            avg_score = 0.5
            avoid_count = 0
            avoid_ratio = 0.0

        # RED: market broadly weak — more than half of signals in avoid territory
        # OR average score below NEUTRAL threshold (0.35). Do not recommend.
        if avg_score < 0.35 or avoid_ratio > 0.5:
            band = "RED"
            market_note = (
                f"Market signals too weak for confident buy recommendations "
                f"(avg score {avg_score:.2f}, {avoid_count}/{signal_count} sectors in avoid territory). "
                "Wait for clearer conditions before adding new positions."
            )
        elif signal_count >= 5 and exposure_count > 0:
            band = "GREEN"
            market_note = ""
        else:
            band = "YELLOW"
            market_note = ""

        return {
            "confidence": {
                "band": band,
                "signal_count": signal_count,
                "exposure_count": exposure_count,
                "avg_market_score": round(avg_score, 3),
                "market_note": market_note,
            }
        }

    def finalize_recommendation(self, state: dict[str, Any]) -> dict[str, Any]:
        confidence_band = state["confidence"]["band"]
        requested_top_n = int(state["request"]["top_n"])
        risk_profile    = state["request"].get("risk_profile", "Balanced")
        horizon_months  = int(state["request"].get("horizon_months", 24))
        _risk_cfg       = get_risk_config(risk_profile)
        _min_rr         = float(_risk_cfg["min_rr_ratio"])
        run_id = f"buy-{uuid4().hex[:10]}"
        # Pre-load unified signals once for the whole loop (used by evidence scoring).
        unified_by_sector = {row["sector"]: row for row in self.repo.list_signals("unified")}

        # RED band: market signals too weak — save empty run and return early.
        if confidence_band == "RED":
            self.repo.save_recommendations(run_id, [])
            return {
                "recommendations": [],
                "skipped_stocks": [],
                "run_summary": {
                    "run_id": run_id,
                    "recommendation_count": 0,
                    "blocked_reason": state["confidence"].get("market_note", "Market signals too weak."),
                },
            }

        # Fetch FII/DII macro flow once per run — cached 24h in Neon (SQLite fallback).
        try:
            macro_flow = fetch_fii_dii_sector_flow(
                neon_database_url=self.config.neon_database_url,
            )
            _log.info(
                "FII/DII macro flow: signal=%s FII=%.0fCr DII=%.0fCr (source=%s)",
                macro_flow.get("flow_signal"), macro_flow.get("fii_net_5d_cr") or 0,
                macro_flow.get("dii_net_5d_cr") or 0, macro_flow.get("source"),
            )
        except Exception as _fii_exc:
            macro_flow = {"flow_signal": "UNKNOWN", "source": "unavailable", "error": str(_fii_exc)}
            _log.warning("FII/DII fetch failed: %s", _fii_exc)

        recommendations = []
        gov_skipped: list[dict] = []
        _log.info(
            "finalize_recommendation: processing %d allocations for top_n=%d (band=%s)",
            len(state["allocations"]), requested_top_n, confidence_band,
        )

        # ── Parallel pre-fetch: analyst consensus targets ─────────────────────
        def _safe_item_price(it: dict[str, Any]) -> float:
            pc = dict(it.get("price_context") or {})
            return float(
                pc.get("price") or it.get("current_price")
                or it.get("live_financials", {}).get("currentPrice")
                or it.get("live_financials", {}).get("current_price")
                or 0.0
            )

        def _existing_analyst_target(it: dict[str, Any], current_price: float) -> float | None:
            for raw in (
                dict(it.get("price_context") or {}).get("analyst_target"),
                it.get("live_financials", {}).get("targetMeanPrice"),
                it.get("analyst_target"),
            ):
                try:
                    target = float(raw)
                except (TypeError, ValueError):
                    continue
                if target > current_price > 0:
                    return target
            return None

        _analyst_cache: dict[str, float] = {}
        _prefetch_items: list[tuple[str, float]] = []
        for it in state["allocations"]:
            if it.get("entry_signal") == "DO NOT ENTER":
                continue
            _price = _safe_item_price(it)
            if _price <= 0:
                continue
            _target = _existing_analyst_target(it, _price)
            if _target is not None:
                _analyst_cache[it["symbol"]] = _target
            else:
                _prefetch_items.append((it["symbol"], _price))
        with ThreadPoolExecutor(max_workers=8) as _aex:
            _afuts = {
                _aex.submit(fetch_analyst_consensus_target, sym, price): sym
                for sym, price in _prefetch_items if price > 0
            }
            for _af in as_completed(_afuts):
                _asym = _afuts[_af]
                try:
                    _analyst_cache[_asym] = _af.result()
                except Exception as _ae:
                    _log.warning("analyst target pre-fetch %s: %s", _asym, _ae)

        _raw_gaps = state["portfolio_context"].get("identified_gaps", [])
        if risk_profile == "Aggressive":
            gaps_by_sector = _compute_aggressive_gaps(_raw_gaps)
        else:
            gaps_by_sector = {g["sector"]: g for g in _raw_gaps}

        # ── Phase 1: serial gate checks + context prep (no LLM calls) ─────────
        _prepared: list[dict[str, Any]] = []

        for item in state["allocations"]:
            if item["entry_signal"] == "DO NOT ENTER":
                _log.info("SKIPPED %s: entry_signal=DO NOT ENTER", item["symbol"])
                continue
            price_ctx = dict(item["price_context"])
            current_price = (
                price_ctx.get("price")
                or item.get("current_price")
                or item.get("live_financials", {}).get("currentPrice")
                or item.get("live_financials", {}).get("current_price")
                or 0.0
            )
            if not current_price or float(current_price) <= 0:
                _log.warning(
                    "SKIPPED %s: current_price=0 or missing — price_ctx=%s",
                    item["symbol"], price_ctx,
                )
                gov_skipped.append({
                    "symbol": item["symbol"],
                    "status": "NO_PRICE",
                    "resolved_symbol": item["symbol"],
                    "reason": "current_price unavailable",
                })
                continue
            analyst_target = (
                _existing_analyst_target(item, float(current_price))
                or _analyst_cache.get(item["symbol"])
                or fetch_analyst_consensus_target(item["symbol"], current_price)
            )
            price_ctx["analyst_target"] = analyst_target
            net_return_pct = compute_net_return(current_price, analyst_target, holding_months=horizon_months)

            if net_return_pct is None or net_return_pct <= 0:
                _log.warning(
                    "EXCLUDED %s: net_return=%.1f%% — target=%.1f below current=%.1f",
                    item["symbol"], net_return_pct or 0, analyst_target or 0, current_price,
                )
                gov_skipped.append({
                    "symbol": item["symbol"],
                    "status": "NEGATIVE_NET_RETURN",
                    "resolved_symbol": item["symbol"],
                    "reason": f"net_return={net_return_pct}% — analyst target is at or below current price",
                })
                continue

            gov_blocked, gov_reason = governance_risk_blocks(item["symbol"], net_return_pct)
            if gov_blocked:
                _log.warning("GOVERNANCE FILTER: %s excluded — %s", item["symbol"], gov_reason)
                gov_skipped.append({
                    "symbol": item["symbol"],
                    "status": "GOVERNANCE_RISK",
                    "resolved_symbol": item["symbol"],
                    "reason": gov_reason,
                })
                continue

            fin_data = dict(item.get("live_financials") or {})
            fin_data.update({key: value for key, value in (item.get("financials") or {}).items() if value is not None})

            # Guard against stale cached data that pre-dates the exclusion flag:
            # check both the explicit flag AND the yoy_source field.  Stale cache
            # entries lacking the flag but carrying a bad yoy_source are still caught.
            _yoy_src = fin_data.get("yoy_source", "")
            _data_excluded = (
                bool(fin_data.get("exclude_from_recommendations"))
                or _yoy_src in ("disagree_excluded", "standalone_only", "no_data")
            )
            if _data_excluded:
                _log.warning(
                    "DATA_QUALITY EXCLUDED %s: flag=%s yoy_source=%s yoy_confidence=%s",
                    item["symbol"],
                    fin_data.get("exclude_from_recommendations"),
                    _yoy_src or "unknown",
                    fin_data.get("yoy_confidence", "unknown"),
                )
                gov_skipped.append({
                    "symbol": item["symbol"],
                    "status": "DATA_QUALITY_LOW",
                    "resolved_symbol": item["symbol"],
                    "reason": (
                        f"Unresolvable YoY revenue disagreement across all sources "
                        f"(confidence={fin_data.get('yoy_confidence', 'LOW')}, "
                        f"source={_yoy_src or 'unknown'}). "
                        "Stock excluded to avoid buy ideas on questionable numbers."
                    ),
                })
                continue

            _current_pe = fin_data.get("pe_ratio") or (
                current_price / fin_data["eps"]
                if fin_data.get("eps") and float(fin_data["eps"]) > 0
                else None
            )
            pe_context = get_pe_historical_context(
                symbol=item["symbol"],
                current_pe=_current_pe,
                db_path=str(self.config.db_path),
                neon_database_url=self.config.neon_database_url,
            )
            val_reliability = get_valuation_reliability(
                symbol=item["symbol"],
                sector=item.get("sector", ""),
                fin_data=fin_data,
                pe_context=pe_context,
                years_listed=int(fin_data.get("years_listed") or 5),
            )

            _w52_low = fin_data.get("week52_low") or fin_data.get("fiftyTwoWeekLow")
            _pct_from_low: float | None = None
            try:
                if _w52_low and float(_w52_low) > 0 and current_price:
                    _pct_from_low = round(
                        (float(current_price) - float(_w52_low)) / float(_w52_low) * 100, 1
                    )
            except (TypeError, ValueError, ZeroDivisionError):
                pass
            tech_signals_dict: dict[str, Any] = {}
            if _pct_from_low is not None:
                tech_signals_dict["pct_from_52w_low"] = _pct_from_low
            try:
                _dma = float(fin_data.get("dma_200") or 0)
                if _dma > 0 and current_price:
                    tech_signals_dict["above_200dma"] = float(current_price) > _dma
            except (TypeError, ValueError):
                pass

            sector_signal_row = unified_by_sector.get(item.get("sector", ""), {})
            evidence = compute_evidence_strength(
                fin_data=fin_data,
                pe_context=pe_context,
                val_reliability=val_reliability,
                tech_signals=tech_signals_dict,
                sector_signal=sector_signal_row,
                news_sentiment=float((item.get("news") or {}).get("sentiment_score", 0.0)),
            )
            if evidence["label"] == "VERY WEAK":
                _log.warning(
                    "REJECTED %s: evidence too weak (%.2f) — %s",
                    item["symbol"], evidence["score"], evidence["note"],
                )
                gov_skipped.append({
                    "symbol": item["symbol"],
                    "status": "WEAK_EVIDENCE",
                    "resolved_symbol": item["symbol"],
                    "reason": f"Evidence too weak ({evidence['score']:.2f}) — {evidence['note']}",
                })
                continue

            # WEAK evidence: stock passes through but entry plan is withheld.
            # Insufficient data to generate reliable entry parameters.
            _entry_withheld = evidence["label"] == "WEAK"
            _entry_withheld_note: str | None = (
                "⚠️ Entry plan withheld — WEAK evidence basis. "
                "Insufficient data to generate reliable entry parameters. "
                "Do not treat as trade-ready."
                if _entry_withheld else None
            )
            if _entry_withheld:
                _log.info(
                    "ENTRY WITHHELD %s: WEAK evidence (%.2f) — entry plan suppressed, stock retained",
                    item["symbol"], evidence["score"],
                )

            _item_sector = item.get("sector", "")
            sector_gap_row = gaps_by_sector.get(_item_sector, {})
            _log.debug(
                "GAP_TRACE %s: sector=%r, gap_found=%s, target_pct=%.1f, gap_pct=%.1f, "
                "known_sectors=%s",
                item["symbol"],
                _item_sector,
                bool(sector_gap_row),
                sector_gap_row.get("target_pct", 0.0),
                sector_gap_row.get("gap_pct", 0.0),
                sorted(gaps_by_sector.keys()) if not sector_gap_row else "—",
            )
            _snap = build_factual_snapshot(
                symbol=item["symbol"],
                fin_data=fin_data,
                current_price=float(current_price),
                pe_context=pe_context,
                tech_signals=tech_signals_dict,
                portfolio_overlap=float(item.get("overlap_pct", 0)),
                sector_gap=sector_gap_row,
            )
            _snap["val_reliability"]       = val_reliability["label"]
            _snap["val_reliability_note"]  = val_reliability["note"]
            _snap["val_reliability_flags"] = val_reliability["flags"]
            snapshot_text = format_snapshot_for_prompt(_snap)

            _is_fallback = (
                analyst_target > 0
                and current_price > 0
                and abs(analyst_target - float(current_price) * 1.15) < float(current_price) * 0.005
                and item["symbol"] not in KNOWN_ANALYST_TARGETS
            )
            if _is_fallback:
                target_source_label = "model estimate — treat as indicative"
            elif item["symbol"] in KNOWN_ANALYST_TARGETS:
                target_source_label = "known analyst target"
            else:
                target_source_label = "screener/broker data"

            # Compute entry levels and R/R before LLM to avoid wasted LLM calls.
            entry_levels = calculate_entry_levels(
                symbol=item["symbol"],
                current_price=current_price,
                analyst_target=analyst_target,
                signal=item["entry_signal"],
                quant_score=item["quality_score"],
                fin_data=fin_data,
            )
            tariff_signal = get_tariff_signal(item.get("sector", ""))
            tariff_warning = None
            initial_tranche_pct = float(item["initial_tranche_pct"])
            initial_amount_inr = float(item["initial_amount_inr"])
            allocation_pct = float(item["allocation_pct"])
            allocation_amount = float(item["allocation_amount"])
            if tariff_signal.get("impact") in {"NEGATIVE", "HIGH_NEGATIVE"}:
                tariff_warning = (
                    f"{tariff_signal['reason']} - FII selling pressure may delay re-rating. "
                    "Consider reducing first tranche size."
                )
                initial_tranche_pct = round(initial_tranche_pct * 0.70, 1)
                initial_amount_inr = round(initial_amount_inr * 0.70, 0)
                allocation_pct = initial_tranche_pct
                allocation_amount = initial_amount_inr
                if entry_levels.get("tranche_1_pct") is not None:
                    entry_levels["tranche_1_pct"] = int(entry_levels["tranche_1_pct"] * 0.70)
            rr_value = float(entry_levels.get("risk_reward", 0) or 0)
            _log.info(
                "%s: signal=%s price=%.1f target=%.1f net_return=%.1f%% rr=%.1fx",
                item["symbol"], item["entry_signal"], float(current_price),
                float(analyst_target or 0), net_return_pct, rr_value,
            )
            if rr_value < _min_rr:
                _log.warning(
                    "EXCLUDED %s: R/R %.1fx below %s minimum %.1fx (price=%.1f target=%.1f signal=%s)",
                    item["symbol"], rr_value, risk_profile, _min_rr,
                    float(current_price), float(analyst_target or 0), item["entry_signal"],
                )
                gov_skipped.append({
                    "symbol": item["symbol"],
                    "status": "LOW_RISK_REWARD",
                    "resolved_symbol": item["symbol"],
                    "reason": f"R/R {rr_value}x below {risk_profile} minimum {_min_rr}x",
                })
                continue

            _prepared.append({
                "item": item,
                "risk_profile": risk_profile,
                "current_price": current_price,
                "analyst_target": analyst_target,
                "net_return_pct": net_return_pct,
                "fin_data": fin_data,
                "pe_context": pe_context,
                "val_reliability": val_reliability,
                "evidence": evidence,
                "snapshot_text": snapshot_text,
                "target_source_label": target_source_label,
                "entry_levels": entry_levels,
                "entry_withheld": _entry_withheld,
                "entry_withheld_note": _entry_withheld_note,
                "tariff_signal": tariff_signal,
                "tariff_warning": tariff_warning,
                "initial_tranche_pct": initial_tranche_pct,
                "initial_amount_inr": initial_amount_inr,
                "allocation_pct": allocation_pct,
                "allocation_amount": allocation_amount,
                "rr_value": rr_value,
            })

        # ── Phase 1b: fetch recent news context (one web_search per stock) ──────
        # Done serially before the parallel LLM phase so both analyst paths
        # (Anthropic + OpenAI) share the same cached headline text.
        def _fetch_news_context_safe(symbol: str, company_name: str) -> str:
            if not hasattr(self.llm, "fetch_stock_news_context"):
                return ""
            try:
                return self.llm.fetch_stock_news_context(symbol, company_name)
            except Exception as exc:
                _log.warning("News fetch failed for %s: %s", symbol, exc)
                return ""

        if _prepared:
            with ThreadPoolExecutor(max_workers=min(4, len(_prepared))) as _nex:
                _nfuts = {
                    _nex.submit(
                        _fetch_news_context_safe,
                        p["item"].get("symbol", ""),
                        p["item"].get("company_name", ""),
                    ): idx
                    for idx, p in enumerate(_prepared)
                }
                for _nf in as_completed(_nfuts):
                    _nidx = _nfuts[_nf]
                    _prepared[_nidx]["news_context"] = _nf.result()

        # ── Phase 2: parallel LLM calls ───────────────────────────────────────
        def _call_rationale(prep: dict[str, Any]) -> str | None:
            it = prep["item"]
            return self.llm.buy_rationale(
                it | {
                    "pe_context":            prep["pe_context"],
                    "val_reliability":       prep["val_reliability"],
                    "evidence":              prep["evidence"],
                    "factual_snapshot_text": prep["snapshot_text"],
                    "target_source_label":   prep["target_source_label"],
                    "analyst_target":        prep["analyst_target"],
                    "macro_flow":            macro_flow,
                    "risk_profile":          prep["risk_profile"],
                    "horizon_months":        horizon_months,
                    "news_context":          prep.get("news_context", ""),
                },
                state["portfolio_context"],
            )

        _llm_results: list[str | None] = [None] * len(_prepared)
        if _prepared:
            with ThreadPoolExecutor(max_workers=min(4, len(_prepared))) as _lex:
                _lfuts = {_lex.submit(_call_rationale, p): i for i, p in enumerate(_prepared)}
                for _lf in as_completed(_lfuts):
                    _lidx = _lfuts[_lf]
                    try:
                        _llm_results[_lidx] = _lf.result()
                    except Exception as _le:
                        _log.warning(
                            "LLM rationale error for %s: %s",
                            _prepared[_lidx]["item"]["symbol"], _le,
                        )

        # ── Phase 3: serial assembly ───────────────────────────────────────────
        for prep, llm_rationale in zip(_prepared, _llm_results):
            if len(recommendations) >= requested_top_n:
                break
            item = prep["item"]
            if not llm_rationale:
                _log.warning("LLM rationale unavailable for %s — proceeding without narrative", item["symbol"])
            rationale = llm_rationale or "[LLM analysis unavailable for this stock]"
            lock_in_check = item.get("lock_in_check") or {}
            payload = {
                "investment_thesis": f"{item['sector']} benefits from current signal stack and fills a real portfolio gap.",
                "why_for_portfolio": item["gap_reason"],
                "overlap_pct": item["overlap_pct"],
                "fund_attribution": item["fund_attribution"],
                "entry_signal": item["entry_signal"],
                "original_entry_signal": item.get("original_entry_signal"),
                "recent_results": item.get("recent_results") or {},
                "revenue_momentum": item.get("revenue_momentum") or prep["fin_data"].get("revenue_momentum") or {},
                "pat_momentum": item.get("pat_momentum") or prep["fin_data"].get("pat_momentum") or {},
                "momentum_override_applied": bool(item.get("momentum_override_applied")),
                "tranche_schedule": f"{item['tranches']} tranches over 4-8 weeks",
                "initial_tranche_pct": prep["initial_tranche_pct"],
                "target_pct": item["target_pct"],
                "initial_amount_inr": prep["initial_amount_inr"],
                "target_amount_inr": item["target_amount_inr"],
                "current_price": round(float(prep["current_price"]), 2) if prep["current_price"] else None,
                "analyst_target": round(float(prep["analyst_target"]), 2) if prep["analyst_target"] else None,
                "fin_data": prep["fin_data"],
                "entry_levels": (
                    prep["entry_levels"] | {
                        "withheld": True,
                        "withheld_note": prep["entry_withheld_note"],
                    }
                    if prep.get("entry_withheld")
                    else prep["entry_levels"]
                ),
                "entry_plan_withheld": bool(prep.get("entry_withheld")),
                "entry_withheld_note": prep.get("entry_withheld_note"),
                # Legacy key kept for backward compatibility with stored recommendations.
                "allocation_pct": prep["allocation_pct"],
                "allocation_amount": prep["allocation_amount"],
                "quality_score": item["quality_score"],
                "technical_signals": item.get("technical_signals", []),
                "technical_score": item.get("technical_score"),
                "pe_context": prep["pe_context"],
                "tariff_signal": prep["tariff_signal"],
                "tariff_warning": prep["tariff_warning"],
                "net_of_tax_return_pct": prep["net_return_pct"],
                "horizon_months": horizon_months,
                # Legacy key for backward compat.
                "net_of_tax_return_projection": round(prep["net_return_pct"] / 100, 4),
                "confidence_band": confidence_band,
                "headline": item["news"]["headline"],
                "validation_reasoning": item.get("validation_reasoning", ""),
                "industry_narrative": state.get("industry_narrative", ""),
                "recently_listed": bool(lock_in_check.get("recently_listed")),
                "months_since_ipo": lock_in_check.get("months_since_ipo"),
                "lock_in_warning": lock_in_check.get("warning"),
                "lock_in_recommendation": lock_in_check.get("recommendation"),
                "lock_in_multiplier": item.get("lock_in_multiplier", 1.0),
                "llm_used": bool(llm_rationale),
                "val_reliability":       prep["val_reliability"],
                "evidence":              prep["evidence"],
                "factual_snapshot_text": prep["snapshot_text"],
                "target_source_label":   prep["target_source_label"],
                "macro_flow":            macro_flow,
                "risk_profile":          risk_profile,
            }
            recommendations.append(
                RecommendationRecord(
                    symbol=item["symbol"],
                    company_name=item["company_name"],
                    sector=item["sector"],
                    action=item["entry_signal"],
                    score=round(item["differentiation_score"], 3),
                    confidence_band=confidence_band,
                    rationale=rationale,
                    payload=payload,
                )
            )
        if len(recommendations) < requested_top_n:
            _log.warning(
                "Final recommendation count %d below requested Top N %d after filtering.",
                len(recommendations), requested_top_n,
            )

        # Diagnose systematic quant vs LLM disagreement (logged as warning; no behaviour change).
        if risk_profile == "Aggressive":
            _disagreement_input = [
                {
                    "quant_verdict": rec.payload.get("entry_signal", ""),
                    "llm_verdict":   rec.action,
                }
                for rec in recommendations
            ]
            _track_quant_llm_disagreement(_disagreement_input)

        # Leak detection: log an error if any excluded stock reached this point.
        for _r in recommendations:
            _r_fin = (_r.payload or {}).get("fin_data", {})
            _r_yoy_src = _r_fin.get("yoy_source", "")
            if (
                _r_fin.get("exclude_from_recommendations")
                or _r_yoy_src in ("disagree_excluded", "standalone_only", "no_data")
            ):
                _log.error(
                    "EXCLUSION LEAK: %s reached recommendations "
                    "(flag=%s yoy_source=%s) — pipeline bug",
                    _r.symbol,
                    _r_fin.get("exclude_from_recommendations"),
                    _r_yoy_src,
                )

        self.repo.save_recommendations(run_id, recommendations)
        # Merge validation-gate skips (ValidationResult objects) + governance-filter skips (dicts).
        validation_skipped: list[ValidationResult] = state.get("skipped_candidates", [])
        overlap_filtered: list[str] = state.get("overlap_filtered", [])
        group_deferred: list[dict] = state.get("group_deferred", [])
        skipped_stocks: list[dict] = [
            {
                "symbol": vr.symbol,
                "status": vr.status.value,
                "resolved_symbol": vr.resolved_symbol,
                "reason": vr.reason,
            }
            for vr in validation_skipped
        ] + gov_skipped + [
            {
                "symbol": sym,
                "status": "OVERLAP_FILTERED",
                "resolved_symbol": sym,
                "reason": "Already >3% represented in your mutual fund / ETF holdings.",
            }
            for sym in overlap_filtered
        ] + [
            {
                "symbol": s.get("symbol", ""),
                "status": "GROUP_CONCENTRATION",
                "resolved_symbol": s.get("symbol", ""),
                "reason": s.get("deferred_reason", "Promoter group already represented."),
            }
            for s in group_deferred
        ]
        blocked_reason = ""
        if not recommendations:
            low_rr_count = sum(1 for row in skipped_stocks if row.get("status") == "LOW_RISK_REWARD")
            negative_return_count = sum(1 for row in skipped_stocks if row.get("status") == "NEGATIVE_NET_RETURN")
            weak_evidence_count = sum(1 for row in skipped_stocks if row.get("status") == "WEAK_EVIDENCE")
            overlap_count = len(overlap_filtered)
            do_not_enter_count = sum(
                1 for item in state.get("allocations", [])
                if item.get("entry_signal") == "DO NOT ENTER"
            )
            shortlist_empty = not state.get("shortlist")
            if negative_return_count:
                blocked_reason = (
                    f"{negative_return_count} shortlisted stock(s) were excluded because the analyst "
                    "target price is at or below the current market price — no positive net-of-tax return. "
                    "Prices may have run ahead of targets. Try refreshing market signals, using a broader "
                    "index universe, or waiting for a better entry point."
                )
            elif low_rr_count:
                blocked_reason = (
                    f"All shortlisted candidates failed the minimum risk/reward gate of "
                    f"{_min_rr}x ({risk_profile} profile). Try a broader universe or rerun when prices/targets improve."
                )
            elif weak_evidence_count:
                _stale_cap = _risk_cfg.get("staleness_cap_days", 90)
                blocked_reason = (
                    f"{weak_evidence_count} candidate(s) were rejected due to insufficient evidence quality — "
                    f"financial data is likely stale (>{_stale_cap} days) for the {risk_profile} profile. "
                    "This is common at end-of-quarter before new results are published. "
                    "Try switching to Balanced profile (90-day tolerance), using a broader universe, "
                    "or waiting for Q4 results to be published."
                )
            elif overlap_count:
                blocked_reason = (
                    f"{overlap_count} shortlisted stock(s) were skipped because they are already "
                    f">3% represented in your mutual fund / ETF holdings. "
                    "Try a broader index (e.g. NIFTY 500) or reduce top-N to allow the next-best candidates through."
                )
            elif do_not_enter_count:
                blocked_reason = (
                    f"{do_not_enter_count} candidate(s) flagged DO NOT ENTER — their sectors have "
                    "STRONG_AVOID market signals. Rerun after a signal refresh or wait for conditions to improve."
                )
            elif shortlist_empty:
                blocked_reason = (
                    "No candidates passed qualitative validation — all shortlisted stocks were rejected "
                    "by the news/sentiment filter or LLM analysis. Current market momentum may be broadly "
                    "negative. Try refreshing market signals, using a broader index, or switching to a "
                    "less restrictive risk profile."
                )
            else:
                blocked_reason = (
                    "No buy candidates survived the full pipeline. "
                    "This can happen when qualitative validation rejects all shortlisted stocks, "
                    "or when current market momentum is broadly negative. "
                    "Try refreshing market signals or re-uploading your portfolio."
                )
        _pipeline_stats = state.get("pipeline_stats") or {}
        _pipeline_stats["shortlist_count"] = len(state.get("shortlist") or [])
        return {
            "recommendations": [asdict(record) for record in recommendations],
            "run_summary": {
                "run_id": run_id,
                "recommendation_count": len(recommendations),
                "requested_top_n": requested_top_n,
                "blocked_reason": blocked_reason,
                "pipeline_stats": _pipeline_stats,
            },
            "skipped_stocks": skipped_stocks,
        }
