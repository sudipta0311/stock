from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any
from uuid import uuid4

from stock_platform.config import AppConfig
from stock_platform.models import RecommendationRecord
from stock_platform.agents.quant_model import compute_quality_score as compute_quality_score_v2
from stock_platform.utils.rules import clamp, parse_iso_datetime
from stock_platform.utils.sector_config import governance_risk_blocks
from stock_platform.utils.stock_validator import ValidationResult, log_skipped_stock, validate_stock


def get_top_n_with_replacement(
    scored_candidates: list[dict[str, Any]],
    n: int,
    skipped_symbols: list[str],
    db_path: str,
) -> list[dict[str, Any]]:
    """
    Keep fetching the next-best validated candidates until we have N rows
    or exhaust the candidate list.
    """
    from stock_platform.utils.screener_fetcher import fetch_screener_data

    recommendations: list[dict[str, Any]] = []
    attempted = {str(symbol).upper() for symbol in skipped_symbols}

    sorted_candidates = sorted(
        scored_candidates,
        key=lambda row: row.get("selection_score", row.get("quality_score", 0)),
        reverse=True,
    )

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


def get_fresh_analyst_target(
    symbol: str,
    current_price: float,
    seed_target: float | None = None,
    live_financials: dict[str, Any] | None = None,
) -> float:
    if not current_price or current_price <= 0:
        return 0.0

    def _valid_target(value: Any) -> bool:
        try:
            return value is not None and float(value) > current_price * 0.5
        except (TypeError, ValueError):
            return False

    for candidate_target in (
        seed_target,
        (live_financials or {}).get("targetMeanPrice"),
        (live_financials or {}).get("target_mean_price"),
        (live_financials or {}).get("target_price"),
    ):
        if _valid_target(candidate_target):
            return float(candidate_target)

    try:
        import yfinance as yf
        from utils.symbol_resolver import resolve_nse_symbol

        ticker = yf.Ticker(resolve_nse_symbol(symbol))
        info = ticker.info or {}
        target = info.get("targetMeanPrice")
        if _valid_target(target):
            return float(target)
    except Exception:
        pass

    try:
        from utils.screener_fetcher import fetch_screener_data

        data = fetch_screener_data(symbol) or {}
        target = data.get("target_price") or data.get("target_mean_price")
        if _valid_target(target):
            return float(target)

        eps = data.get("eps")
        sector_pe = data.get("sector_pe") or data.get("pe_ratio")
        if eps and eps > 0 and sector_pe and sector_pe > 0:
            proxy_target = float(eps) * float(sector_pe) * 1.15
            if proxy_target > current_price * 0.5:
                print(f"{symbol}: using PE proxy target Rs.{proxy_target:.0f}")
                return float(proxy_target)
    except Exception:
        pass

    fallback = current_price * 1.15
    print(f"{symbol}: no target found - using Rs.{fallback:.0f} (15% upside assumption)")
    return float(fallback)


def compute_position_size(
    entry_signal: str,
    conviction_score: float,
    direct_equity_corpus: float,
) -> dict[str, Any]:
    """
    Returns initial_tranche_pct and target_pct separately.
    Initial tranche is what to deploy NOW.
    Target is the full intended position over 2-3 months.
    """
    # Map timing signals from assess_timing to conviction tiers
    signal_map = {
        "STRONG ENTER": "STRONG_BUY",
        "ACCUMULATE":   "BUY",
        "SMALL INITIAL": "ACCUMULATE",
        "WAIT":          "WAIT",
    }
    conviction_key = signal_map.get(entry_signal, "ACCUMULATE")

    sizing_rules: dict[str, dict[str, float]] = {
        "STRONG_BUY": {"initial": 0.10, "target": 0.28},
        "BUY":        {"initial": 0.08, "target": 0.22},
        "ACCUMULATE": {"initial": 0.06, "target": 0.18},
        "WAIT":       {"initial": 0.00, "target": 0.10},
    }
    rule = sizing_rules.get(conviction_key, {"initial": 0.05, "target": 0.10})

    conviction_multiplier = 0.85 + (conviction_score * 0.30)

    initial_pct = round(rule["initial"] * conviction_multiplier * 100, 1)
    target_pct  = round(rule["target"]  * conviction_multiplier * 100, 1)

    initial_amount = round(direct_equity_corpus * initial_pct / 100, 0)
    target_amount  = round(direct_equity_corpus * target_pct  / 100, 0)

    hard_cap_pct = 30.0
    initial_pct = min(initial_pct, hard_cap_pct)
    target_pct  = min(target_pct,  hard_cap_pct)

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
        for candidate in state["candidates"]:
            live_facts = self.provider.get_financials(candidate["symbol"])
            current_price = live_facts.get("currentPrice") or live_facts.get("current_price")

            # Safety gate: block unresolvable stocks before any scoring or LLM call.
            vr = validate_stock(candidate["symbol"], live_facts, current_price)
            if not vr.can_recommend:
                skipped.append(vr)
                log_skipped_stock(str(self.config.db_path), vr, validation_run_id)
                print(f"SKIPPED {candidate['symbol']}: {vr.reason}")
                continue

            base_score = compute_quality_score_v2(candidate["symbol"], live_facts)

            # Geo signal bonus — applied on top of the computed base (small additive).
            geo_bonus = 0.04 if unified.get(candidate["sector"], {}).get("conviction") in {"BUY", "STRONG_BUY"} else 0.0
            # Flow / accumulation bonus.
            flow_bonus = accumulation_bonus
            selection_score = clamp(base_score + geo_bonus + flow_bonus, 0.0, 1.0)
            scored.append(candidate | {
                "quality_score": round(base_score, 3),
                "selection_score": round(selection_score, 3),
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
        sector_weights: dict[str, float] = {}
        for row in state["portfolio_context"]["normalized_exposure"]:
            sector_weights[row["sector"]] = sector_weights.get(row["sector"], 0.0) + row["total_weight"]
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        filtered = []
        for candidate in state["scored_candidates"]:
            risk = self.provider.get_risk_metrics(candidate["symbol"])
            if risk.get("avg_daily_value_cr") is not None and risk["avg_daily_value_cr"] < 5:
                continue
            if risk.get("beta") is not None and risk["beta"] > 2.0:
                continue
            if risk.get("promoter_pledge_pct") is not None and risk["promoter_pledge_pct"] > 50:
                continue
            if risk.get("sebi_flag"):
                continue
            if sector_weights.get(candidate["sector"], 0.0) > self.config.max_sector_pct:
                continue
            if unified.get(candidate["sector"], {}).get("conviction") == "STRONG_AVOID":
                continue
            filtered.append(candidate | {"risk_metrics": risk})
        return {"risk_filtered_candidates": filtered}

    def shortlist(self, state: dict[str, Any]) -> dict[str, Any]:
        top_n = int(state["request"]["top_n"])
        skipped_symbols = [result.symbol for result in state.get("skipped_candidates", [])]
        shortlist = get_top_n_with_replacement(
            state["risk_filtered_candidates"],
            max(top_n * 4, top_n),
            skipped_symbols,
            str(self.config.db_path),
        )
        return {"shortlist": shortlist}

    def validate_qualitative(self, state: dict[str, Any]) -> dict[str, Any]:
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        top_n = int(state["request"]["top_n"])
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
        if len(approved) < top_n and fallback_pool:
            used_symbols = {row["symbol"] for row in approved}
            fallback_pool.sort(
                key=lambda row: (
                    row.get("validation_confidence", 0.0),
                    row.get("selection_score", row.get("quality_score", 0.0)),
                ),
                reverse=True,
            )
            for candidate in fallback_pool:
                if len(approved) >= top_n:
                    break
                if candidate["symbol"] in used_symbols:
                    continue
                approved.append(candidate)
                used_symbols.add(candidate["symbol"])

        if not approved:
            print("WARNING: qualitative validation approved zero candidates - buy feed will be empty.")

        return {"shortlist": approved[:top_n]}

    def differentiate_portfolio(self, state: dict[str, Any]) -> dict[str, Any]:
        overlaps = {row["symbol"]: row for row in state["portfolio_context"]["overlap_scores"]}
        gaps = {row["sector"]: row for row in state["portfolio_context"]["identified_gaps"]}
        differentiated = []
        for candidate in state["shortlist"]:
            overlap = overlaps.get(candidate["symbol"], {"overlap_pct": 0.0, "band": "GREEN", "attribution": []})
            if overlap["overlap_pct"] > 3:
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
        return {"differentiated_shortlist": differentiated[: int(state["request"]["top_n"])]}

    def assess_timing(self, state: dict[str, Any]) -> dict[str, Any]:
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
            timing_rows.append(candidate | {"entry_signal": entry, "price_context": price})
        return {"timing_assessments": timing_rows}

    def size_positions(self, state: dict[str, Any]) -> dict[str, Any]:
        prefs = state["portfolio_context"]["user_preferences"]
        corpus = float(prefs.get("direct_equity_corpus", 0) or 0) or 100.0
        allocations = []
        for item in state["timing_assessments"]:
            sizing = compute_position_size(item["entry_signal"], item["quality_score"], corpus)
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
        return {
            "tax_assessment": {
                "stcg_rate": 20.0,
                "ltcg_rate": 12.5,
                "cost_drag_pct": 0.15,
                "note": "New-buy flow applies entry-side cost drag only because no exit is required.",
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
        run_id = f"buy-{uuid4().hex[:10]}"

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

        recommendations = []
        gov_skipped: list[dict] = []   # governance-blocked stocks collected during this loop
        for item in state["allocations"]:
            if item["entry_signal"] == "DO NOT ENTER":
                continue
            price_ctx = dict(item["price_context"])
            current_price = (
                price_ctx.get("price")
                or item.get("current_price")
                or item.get("live_financials", {}).get("currentPrice")
                or item.get("live_financials", {}).get("current_price")
                or 0.0
            )
            analyst_target = get_fresh_analyst_target(
                item["symbol"],
                current_price,
                seed_target=price_ctx.get("analyst_target"),
                live_financials=item.get("live_financials", {}),
            )
            price_ctx["analyst_target"] = analyst_target
            net_return_pct = compute_net_return(current_price, analyst_target, holding_months=24)

            # Governance risk gate: Adani Group and peers need ≥10% net return.
            gov_blocked, gov_reason = governance_risk_blocks(item["symbol"], net_return_pct)
            if gov_blocked:
                print(f"GOVERNANCE FILTER: {item['symbol']} excluded — {gov_reason}")
                gov_skipped.append({
                    "symbol": item["symbol"],
                    "status": "GOVERNANCE_RISK",
                    "resolved_symbol": item["symbol"],
                    "reason": gov_reason,
                })
                continue

            llm_rationale = self.llm.buy_rationale(item, state["portfolio_context"])
            if not llm_rationale:
                print(f"LLM rationale unavailable for {item['symbol']} — proceeding without narrative")
            rationale = llm_rationale or "[LLM analysis unavailable for this stock]"

            payload = {
                "investment_thesis": f"{item['sector']} benefits from current signal stack and fills a real portfolio gap.",
                "why_for_portfolio": item["gap_reason"],
                "overlap_pct": item["overlap_pct"],
                "fund_attribution": item["fund_attribution"],
                "entry_signal": item["entry_signal"],
                "tranche_schedule": f"{item['tranches']} tranches over 4-8 weeks",
                # Position sizing — initial tranche (deploy now) and full target separately.
                "initial_tranche_pct": item["initial_tranche_pct"],
                "target_pct": item["target_pct"],
                "initial_amount_inr": item["initial_amount_inr"],
                "target_amount_inr": item["target_amount_inr"],
                # Legacy key kept for backward compatibility with stored recommendations.
                "allocation_pct": item["allocation_pct"],
                "allocation_amount": item["allocation_amount"],
                "quality_score": item["quality_score"],
                "net_of_tax_return_pct": net_return_pct,
                # Legacy key for backward compat.
                "net_of_tax_return_projection": round(net_return_pct / 100, 4),
                "confidence_band": confidence_band,
                "headline": item["news"]["headline"],
                "validation_reasoning": item.get("validation_reasoning", ""),
                "industry_narrative": state.get("industry_narrative", ""),
                "llm_used": bool(llm_rationale),
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
        self.repo.save_recommendations(run_id, recommendations)
        # Merge validation-gate skips (ValidationResult objects) + governance-filter skips (dicts).
        validation_skipped: list[ValidationResult] = state.get("skipped_candidates", [])
        skipped_stocks: list[dict] = [
            {
                "symbol": vr.symbol,
                "status": vr.status.value,
                "resolved_symbol": vr.resolved_symbol,
                "reason": vr.reason,
            }
            for vr in validation_skipped
        ] + gov_skipped
        return {
            "recommendations": [asdict(record) for record in recommendations],
            "run_summary": {"run_id": run_id, "recommendation_count": len(recommendations)},
            "skipped_stocks": skipped_stocks,
        }
