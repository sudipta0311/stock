from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any
from uuid import uuid4

from stock_platform.config import AppConfig
from stock_platform.models import RecommendationRecord
from stock_platform.utils.rules import clamp, parse_iso_datetime


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
        accumulation_bonus = 0.05 if flow and flow[0]["payload"].get("label") == "ACCUMULATION" else 0.0
        scored = []
        for candidate in state["candidates"]:
            facts = self.provider.get_financials(candidate["symbol"])
            geo_bonus = 0.05 if unified.get(candidate["sector"], {}).get("conviction") in {"BUY", "STRONG_BUY"} else 0.0
            raw = (
                min(facts["roce_5y"] / 20, 1.0) * 0.25
                + min(facts["fcf_positive_years"] / 5, 1.0) * 0.25
                + min(facts["revenue_consistency"] / 10, 1.0) * 0.20
                + (1.0 if facts["promoter_trend"] in {"stable", "rising"} else 0.6) * 0.15
                + (1.0 if facts["de_ratio"] < 0.5 else 0.4) * 0.15
                + geo_bonus
                + accumulation_bonus
            )
            scored.append(candidate | {"quality_score": round(min(raw, 1.0), 3), "financials": facts})
        scored.sort(key=lambda row: row["quality_score"], reverse=True)
        return {"scored_candidates": scored}

    def filter_risk(self, state: dict[str, Any]) -> dict[str, Any]:
        sector_weights: dict[str, float] = {}
        for row in state["portfolio_context"]["normalized_exposure"]:
            sector_weights[row["sector"]] = sector_weights.get(row["sector"], 0.0) + row["total_weight"]
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        filtered = []
        for candidate in state["scored_candidates"]:
            risk = self.provider.get_risk_metrics(candidate["symbol"])
            if risk["avg_daily_value_cr"] < 5:
                continue
            if risk["beta"] > 2.0:
                continue
            if risk["promoter_pledge_pct"] > 50:
                continue
            if risk["sebi_flag"]:
                continue
            if sector_weights.get(candidate["sector"], 0.0) > self.config.max_sector_pct:
                continue
            if unified.get(candidate["sector"], {}).get("conviction") == "STRONG_AVOID":
                continue
            filtered.append(candidate | {"risk_metrics": risk})
        return {"risk_filtered_candidates": filtered}

    def shortlist(self, state: dict[str, Any]) -> dict[str, Any]:
        top_n = int(state["request"]["top_n"])
        shortlist = state["risk_filtered_candidates"][: max(top_n * 2, top_n)]
        return {"shortlist": shortlist}

    def validate_qualitative(self, state: dict[str, Any]) -> dict[str, Any]:
        unified = {row["sector"]: row for row in self.repo.list_signals("unified")}
        approved = []
        for candidate in state["shortlist"]:
            news = self.provider.get_stock_news(candidate["symbol"])
            signal_context = unified.get(candidate["sector"], {})

            # LLM:Sonnet — structured qualitative validation with JSON output.
            llm_result = self.llm.qualitative_analysis(candidate, news, signal_context)
            if llm_result is not None:
                if not llm_result["approved"]:
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
        return {"shortlist": approved[: int(state["request"]["top_n"])]}

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
                    candidate["quality_score"] - overlap["overlap_pct"] * 0.08 + gaps.get(candidate["sector"], {}).get("score", 0.3) * 0.2,
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
        corpus = float(prefs.get("direct_equity_corpus", 0) or 0)
        surplus = float(prefs.get("investable_surplus", 0) or 0)
        deployable = max(corpus, surplus) or 100.0
        allocations = []
        base_map = {
            "STRONG ENTER": 0.28,
            "ACCUMULATE": 0.22,
            "SMALL INITIAL": 0.12,
            "WAIT": 0.06,
            "DO NOT ENTER": 0.0,
        }
        for item in state["timing_assessments"]:
            pct = min(base_map[item["entry_signal"]], self.config.max_single_stock_pct / 100)
            allocations.append(
                item
                | {
                    "allocation_pct": round(pct * 100, 1),
                    "allocation_amount": round(deployable * pct, 2),
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
        signal_count = len(self.repo.list_signals("unified"))
        exposure_count = len(state["portfolio_context"]["normalized_exposure"])
        band = "GREEN" if signal_count >= 5 and exposure_count > 0 else "YELLOW"
        return {"confidence": {"band": band, "signal_count": signal_count, "exposure_count": exposure_count}}

    def finalize_recommendation(self, state: dict[str, Any]) -> dict[str, Any]:
        recommendations = []
        confidence_band = state["confidence"]["band"]
        run_id = f"buy-{uuid4().hex[:10]}"
        for item in state["allocations"]:
            if item["entry_signal"] == "DO NOT ENTER":
                continue
            llm_rationale = self.llm.buy_rationale(item, state["portfolio_context"])
            rationale = llm_rationale or (
                f"{item['company_name']} adds differentiated {item['sector']} exposure with "
                f"{item['entry_signal'].lower()} timing."
            )
            payload = {
                "investment_thesis": f"{item['sector']} benefits from current signal stack and fills a real portfolio gap.",
                "why_for_portfolio": item["gap_reason"],
                "overlap_pct": item["overlap_pct"],
                "fund_attribution": item["fund_attribution"],
                "entry_signal": item["entry_signal"],
                "tranche_schedule": f"{item['tranches']} tranches over 4-8 weeks",
                "allocation_pct": item["allocation_pct"],
                "allocation_amount": item["allocation_amount"],
                "quality_score": item["quality_score"],
                "net_of_tax_return_projection": round(13 + item["quality_score"] * 8 - state["tax_assessment"]["cost_drag_pct"], 2),
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
        return {
            "recommendations": [asdict(record) for record in recommendations],
            "run_summary": {"run_id": run_id, "recommendation_count": len(recommendations)},
        }
