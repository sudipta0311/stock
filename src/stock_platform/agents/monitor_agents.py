from __future__ import annotations

from typing import Any
from uuid import uuid4

from stock_platform.config import AppConfig
from stock_platform.models import MonitoringAction
from utils.tax_calculator import calculate_pnl, should_exit


def format_monitoring_rationale(
    row: dict[str, Any],
    pnl: dict[str, Any] | None,
    exit_rec: dict[str, Any] | None,
) -> str:
    parts = []
    thesis = row.get("thesis_status", "INTACT")
    quant = row.get("quant_score", 0.0)

    if thesis == "BREACHED":
        parts.append("THESIS BREACHED — exit regardless of price")
    elif thesis == "WEAKENED":
        parts.append(f"Thesis weakening (quant {quant:.2f})")
    else:
        parts.append(f"Thesis intact (quant {quant:.2f})")

    if pnl:
        days_held = pnl["days_held"] if pnl["days_held"] is not None else "n/a"
        parts.append(
            f"P&L: {'+' if pnl['gross_pnl'] > 0 else ''}"
            f"Rs{pnl['gross_pnl']:,.0f} ({pnl['pnl_pct']:+.1f}%) "
            f"| Held {days_held}d [{pnl['tax_type']}]"
        )
    if exit_rec:
        parts.append(exit_rec["exit_recommendation"])
        parts.append(exit_rec["tax_note"])

    return " | ".join(parts)


class MonitoringAgents:
    def __init__(self, repo: Any, provider: Any, config: AppConfig, signal_refresh_runner: Any, llm: Any) -> None:
        self.repo = repo
        self.provider = provider
        self.config = config
        self.signal_refresh_runner = signal_refresh_runner
        self.llm = llm

    def refresh_signals(self, state: dict[str, Any]) -> dict[str, Any]:
        self.signal_refresh_runner(trigger="monitoring")
        return {}

    def load_context(self, state: dict[str, Any]) -> dict[str, Any]:
        ctx = self.repo.load_portfolio_context()

        # Build monitor universe: direct equity holdings + watchlist only.
        # MF / ETF indirect holdings are excluded — the user monitors those
        # through the fund manager, not stock-by-stock.
        norm_by_symbol = {row["symbol"]: row for row in ctx["normalized_exposure"]}

        total_direct_value = sum(
            row["market_value"]
            for row in ctx["raw_holdings"]
            if row["holding_type"] == "direct_equity" and row.get("market_value")
        ) or 1.0

        monitor_universe: list[dict[str, Any]] = []
        seen: set[str] = set()

        for row in ctx["raw_holdings"]:
            if row["holding_type"] != "direct_equity":
                continue
            sym = self.provider.normalize_symbol(row.get("symbol") or row.get("instrument_name") or "")
            if not sym or sym in seen:
                continue
            seen.add(sym)
            entry = dict(norm_by_symbol[sym]) if sym in norm_by_symbol else {
                "symbol": sym,
                "company_name": row["instrument_name"],
                "sector": "Unknown",
                "total_weight": round(row["market_value"] / total_direct_value * 100, 2),
            }
            entry["monitor_source"] = "direct"
            monitor_universe.append(entry)

        for row in ctx["watchlist"]:
            sym = self.provider.normalize_symbol(row["symbol"])
            if sym in seen:
                continue
            seen.add(sym)
            entry = dict(norm_by_symbol[sym]) if sym in norm_by_symbol else {
                "symbol": sym,
                "company_name": row["company_name"],
                "sector": row["sector"],
                "total_weight": 0.0,
            }
            entry["monitor_source"] = "watchlist"
            monitor_universe.append(entry)

        broker_map = {
            self.provider.normalize_symbol(item["symbol"]): item
            for item in ctx.get("direct_equity_holdings", [])
            if item.get("symbol")
        }
        ctx["monitor_universe"] = monitor_universe
        ctx["direct_equity_buy_map"] = broker_map
        return {"portfolio_context": ctx}

    def monitor_industries(self, state: dict[str, Any]) -> dict[str, Any]:
        reviews = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            sector_news = self.provider.get_sector_news(holding["sector"])
            unified = next(
                (row for row in state["portfolio_context"]["unified_signals"] if row["sector"] == holding["sector"]),
                None,
            )
            reviews.append(
                {
                    "symbol": holding["symbol"],
                    "sector": holding["sector"],
                    "industry_signal_score": sector_news["signal_score"],
                    "aligned_conviction": unified["conviction"] if unified else "NEUTRAL",
                    "summary": sector_news["summary"],
                }
            )
        return {"industry_reviews": reviews}

    def monitor_stocks(self, state: dict[str, Any]) -> dict[str, Any]:
        reviews = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            news = self.provider.get_stock_news(holding["symbol"])
            reviews.append(
                {
                    "symbol": holding["symbol"],
                    "headline": news["headline"],
                    "sentiment_score": news["sentiment_score"],
                }
            )
        return {"stock_reviews": reviews}

    def rescore_quant(self, state: dict[str, Any]) -> dict[str, Any]:
        scores = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            financials = self.provider.get_financials(holding["symbol"])

            pe_t = financials.get("pe_trailing")
            pe_5y = financials.get("pe_5yr_avg")
            pe_sec = financials.get("sector_pe")
            pe_fwd = financials.get("pe_forward")

            pe_components: list[tuple[float, float]] = []
            if pe_t and pe_5y:
                pe_components.append((0.40, max(0.0, min(1.0, 1.0 - (pe_t - pe_5y) / pe_5y))))
            if pe_t and pe_sec:
                pe_components.append((0.35, max(0.0, min(1.0, 1.0 - (pe_t - pe_sec) / pe_sec))))
            if pe_t and pe_fwd:
                pe_components.append((0.25, max(0.0, min(1.0, 2.0 - pe_fwd / pe_t))))
            pe_weight = sum(weight for weight, _ in pe_components)
            pe_score = sum(weight * score for weight, score in pe_components) / pe_weight if pe_weight > 0 else 0.5

            quality_components: list[tuple[float, float]] = []
            roce = financials.get("roce_ttm") or financials.get("returnOnCapitalEmployed")
            if roce is not None:
                quality_components.append((0.35, min(roce / 0.20, 1.0)))

            fcf_positive_years = financials.get("fcf_positive_years")
            free_cashflow = financials.get("freeCashflow") or financials.get("free_cashflow")
            if isinstance(fcf_positive_years, int):
                quality_components.append((0.25, min(fcf_positive_years / 5, 1.0)))
            elif free_cashflow is not None:
                quality_components.append((0.25, 1.0 if free_cashflow > 0 else 0.0))

            revenue_growth = financials.get("revenueGrowth") or financials.get("revenue_growth")
            if revenue_growth is not None:
                quality_components.append((0.20, min(max(revenue_growth, 0.0) / 0.15, 1.0)))

            quality_components.append((0.20, pe_score))
            total_weight = sum(weight for weight, _ in quality_components)
            quant = round(
                sum(weight * score for weight, score in quality_components) / total_weight if total_weight > 0 else 0.5,
                3,
            )
            scores.append({"symbol": holding["symbol"], "quant_score": quant})
        return {"quant_scores": scores}

        scores = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            financials = self.provider.get_financials(holding["symbol"])

            # P/E score: weighted combination of three relative comparisons.
            # NOT an absolute threshold — each sub-component is scored 0–1.
            pe_t = financials["pe_trailing"]
            pe_5y = financials["pe_5yr_avg"]
            pe_sec = financials["sector_pe"]
            pe_fwd = financials["pe_forward"]

            # 40%: stock vs own 5yr avg P/E (discount to own history = good)
            vs_own = max(0.0, min(1.0, 1.0 - (pe_t - pe_5y) / pe_5y))
            # 35%: stock vs sector peer P/E (discount to peers = good)
            vs_sector = max(0.0, min(1.0, 1.0 - (pe_t - pe_sec) / pe_sec))
            # 25%: forward vs trailing P/E (forward < trailing = earnings growth = good)
            fwd_vs_trail = max(0.0, min(1.0, 2.0 - pe_fwd / pe_t))

            pe_score = 0.40 * vs_own + 0.35 * vs_sector + 0.25 * fwd_vs_trail

            # Overall quant score: quality + valuation blend.
            quant = round(
                min(financials["roce_5y"] / 20, 1.0) * 0.35
                + min(financials["fcf_positive_years"] / 5, 1.0) * 0.25
                + min(financials["revenue_consistency"] / 10, 1.0) * 0.20
                + pe_score * 0.20,
                3,
            )
            scores.append({"symbol": holding["symbol"], "quant_score": quant})
        return {"quant_scores": scores}

    def review_thesis(self, state: dict[str, Any]) -> dict[str, Any]:
        unified = {row["sector"]: row for row in state["portfolio_context"]["unified_signals"]}
        quant_scores = {row["symbol"]: row["quant_score"] for row in state["quant_scores"]}
        stock_news = {row["symbol"]: row for row in state["stock_reviews"]}
        thesis = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            sector_signal = unified.get(holding["sector"], {})
            quant = quant_scores.get(holding["symbol"], 0.5)

            # Hard rules always override LLM — enforce first.
            if sector_signal.get("conviction") == "STRONG_AVOID":
                status = "BREACHED"
                llm_reasoning = ""
            else:
                # LLM:Sonnet — nuanced multi-signal thesis assessment.
                news = stock_news.get(holding["symbol"], {})
                llm_result = self.llm.thesis_review(holding, quant, sector_signal, news)
                if llm_result is not None:
                    status = llm_result["status"]
                    llm_reasoning = llm_result["reasoning"]
                else:
                    # Deterministic fallback.
                    status = "WEAKENED" if quant < 0.55 else "INTACT"
                    llm_reasoning = ""

            thesis.append(
                {
                    "symbol": holding["symbol"],
                    "sector": holding["sector"],
                    "status": status,
                    "geo_signal_change": sector_signal.get("conviction", "NEUTRAL"),
                    "llm_reasoning": llm_reasoning,
                }
            )
        return {"thesis_reviews": thesis}

    def drawdown_risk(self, state: dict[str, Any]) -> dict[str, Any]:
        buy_map = state["portfolio_context"].get("direct_equity_buy_map", {})
        alerts = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            buy_info = buy_map.get(holding["symbol"], {})
            entry_price = buy_info.get("avg_buy_price")
            series = self.provider.get_monitoring_price_series(holding["symbol"], entry_price=entry_price)
            drawdown = series["drawdown_pct"]
            if drawdown <= -35:
                severity = "CRITICAL"
            elif drawdown <= -20:
                severity = "HIGH"
            elif holding["total_weight"] > 35:
                severity = "HIGH"
            else:
                severity = "LOW"
            alerts.append(
                {
                    "symbol": holding["symbol"],
                    "drawdown_pct": drawdown,
                    "severity": severity,
                    "entry_price": series["entry_price"],
                    "current_price": series["current_price"],
                }
            )
        return {"drawdown_alerts": alerts}

    @staticmethod
    def _severity_from_context(thesis: str, drawdown: str, urgency: str) -> str:
        if thesis == "BREACHED" or drawdown == "CRITICAL":
            return "CRITICAL"
        if drawdown == "HIGH" or urgency == "HIGH":
            return "HIGH"
        if thesis == "WEAKENED" or urgency == "MEDIUM":
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _fallback_urgency(action: str, severity: str) -> str:
        if action in {"SELL", "REPLACE"} or severity in {"CRITICAL", "HIGH"}:
            return "HIGH"
        if action in {"TRIM", "BUY MORE"} or severity == "MEDIUM":
            return "MEDIUM"
        return "LOW"

    def decide_actions(self, state: dict[str, Any]) -> dict[str, Any]:
        stock_reviews = {row["symbol"]: row for row in state["stock_reviews"]}
        thesis_map = {row["symbol"]: row for row in state["thesis_reviews"]}
        drawdown_map = {row["symbol"]: row for row in state["drawdown_alerts"]}
        quant_map = {row["symbol"]: row["quant_score"] for row in state["quant_scores"]}
        buy_map = state["portfolio_context"].get("direct_equity_buy_map", {})
        actions = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            symbol = holding["symbol"]
            thesis = thesis_map[symbol]["status"]
            drawdown = drawdown_map[symbol]["severity"]
            sentiment = stock_reviews[symbol]["sentiment_score"]
            quant = quant_map[symbol]
            pnl = None
            exit_rec = None
            urgency = "LOW"

            buy_info = buy_map.get(symbol)
            analyst_target = self.provider.get_price_context(symbol).get("analyst_target")
            current_price = drawdown_map[symbol].get("current_price")

            if buy_info and current_price:
                pnl = calculate_pnl(
                    symbol=symbol,
                    avg_buy_price=float(buy_info.get("avg_buy_price") or 0),
                    current_price=float(current_price),
                    quantity=float(buy_info.get("quantity") or 0),
                    buy_date_str=str(buy_info.get("buy_date") or "unknown"),
                )
                exit_rec = should_exit(
                    pnl=pnl,
                    analyst_target=analyst_target,
                    current_price=float(current_price),
                    thesis_status=thesis,
                    quant_score=quant,
                )
                urgency = exit_rec["urgency"]

            if thesis == "BREACHED":
                action = "EXIT - thesis breached"
                urgency = exit_rec["urgency"] if exit_rec else "CRITICAL"
                if exit_rec is None:
                    exit_rec = {
                        "exit_recommendation": action,
                        "reasoning": "Thesis breached",
                        "tax_note": "Re-underwrite before continuing to hold",
                        "urgency": urgency,
                    }
                severity = "CRITICAL"
            elif exit_rec is not None:
                action = exit_rec["exit_recommendation"]
                severity = self._severity_from_context(thesis, drawdown, urgency)
            else:
                if drawdown == "HIGH" and quant < 0.6:
                    action = "TRIM"
                    severity = "HIGH"
                elif sentiment > 0.5 and quant > 0.72:
                    action = "BUY MORE"
                    severity = "MEDIUM"
                else:
                    action = "HOLD"
                    severity = "LOW"
                urgency = self._fallback_urgency(action, severity)

            rationale = format_monitoring_rationale(
                {
                    "symbol": symbol,
                    "thesis_status": thesis,
                    "quant_score": quant,
                },
                pnl,
                exit_rec,
            )
            actions.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "severity": severity,
                    "urgency": urgency,
                    "rationale": rationale,
                    "pnl": pnl,
                    "exit_recommendation": exit_rec,
                    "analyst_target": analyst_target,
                }
            )
        return {"actions": actions}

    def behavioural_guard(self, state: dict[str, Any]) -> dict[str, Any]:
        prefs = state["portfolio_context"]["user_preferences"]
        current_direct = len(
            [row for row in state["portfolio_context"]["raw_holdings"] if row["holding_type"] == "direct_equity"]
        )
        flags = []
        for action in state["actions"]:
            if action["action"] == "SELL" or str(action["action"]).startswith("EXIT"):
                flags.append(
                    {
                        "symbol": action["symbol"],
                        "flag": "SELL_OVERRIDE_BLOCK",
                        "message": "Breached thesis cannot be overridden without re-underwriting evidence.",
                    }
                )
        if current_direct > self.config.max_direct_stocks:
            flags.append(
                {
                    "symbol": "PORTFOLIO",
                    "flag": "MAX_DIRECT_STOCKS",
                    "message": f"Direct stock count exceeds platform limit of {self.config.max_direct_stocks}.",
                }
            )
        if prefs.get("monitoring_runs_today", 0) > 3:
            flags.append(
                {
                    "symbol": "PORTFOLIO",
                    "flag": "BEHAVIOURAL_REMINDER",
                    "message": "Check less, earn more. Monitoring more than 3 times a day adds noise.",
                }
            )
        return {"behavioural_flags": flags}

    def replace_feedback(self, state: dict[str, Any]) -> dict[str, Any]:
        sell_like = [
            row
            for row in state["actions"]
            if row["action"] in {"SELL", "REPLACE"} or str(row["action"]).startswith("EXIT")
        ]
        prompt = {
            "should_prompt": bool(sell_like),
            "message": "Portfolio gap created. Run new buy recommendation with refreshed context."
            if sell_like
            else "No replacement cycle needed.",
        }
        run_id = f"monitor-{uuid4().hex[:10]}"
        rows = []
        for row in state["actions"]:
            thesis = next(item for item in state["thesis_reviews"] if item["symbol"] == row["symbol"])
            drawdown = next(item for item in state["drawdown_alerts"] if item["symbol"] == row["symbol"])
            llm_result = None
            if row["action"] in {"BUY MORE", "HOLD", "TRIM", "SELL", "REPLACE"}:
                llm_result = self.llm.monitoring_rationale(row, thesis, drawdown)
            # Use LLM-confirmed action/severity/rationale if parsing succeeded;
            # fall back to deterministic values so a JSON failure never drops a row.
            final_action = llm_result["action"] if llm_result else row["action"]
            final_severity = llm_result["severity"] if llm_result else row["severity"]
            final_rationale = row["rationale"]
            rows.append(
                MonitoringAction(
                    symbol=row["symbol"],
                    action=final_action,
                    severity=final_severity,
                    urgency=row.get("urgency", "LOW"),
                    rationale=final_rationale,
                    payload={
                        "behavioural_flags": [
                            flag for flag in state["behavioural_flags"] if flag["symbol"] in {row["symbol"], "PORTFOLIO"}
                        ],
                        "drawdown": drawdown,
                        "thesis": thesis,
                        "thesis_llm_reasoning": thesis.get("llm_reasoning", ""),
                        "pnl": row.get("pnl"),
                        "exit_recommendation": row.get("exit_recommendation"),
                        "analyst_target": row.get("analyst_target"),
                        "llm_used": bool(llm_result),
                    },
                )
            )
        self.repo.save_monitoring_actions(run_id, rows)
        return {
            "replacement_prompt": prompt,
            "run_summary": {"run_id": run_id, "action_count": len(rows)},
        }
