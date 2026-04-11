from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any
from uuid import uuid4

from stock_platform.config import AppConfig
from stock_platform.models import RecommendationRecord
from stock_platform.utils.rules import clamp, parse_iso_datetime


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


def fetch_financial_data(symbol: str) -> dict[str, Any]:
    """
    Fetch live financial data from yfinance for quality scoring.
    Returns {} on any failure — callers must treat empty dict as "no data".
    Uses .NS suffix for NSE symbols.
    """
    try:
        import yfinance as yf  # lazy import — only needed when live data is requested
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        if not info or info.get("regularMarketPrice") is None:
            return {}
        # Fast financial metrics from info dict
        result: dict[str, Any] = {
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "roe": info.get("returnOnEquity"),          # decimal (0.18 = 18%)
            "roa": info.get("returnOnAssets"),          # decimal
            "de_ratio": info.get("debtToEquity"),       # percentage in yfinance (150 = 150%)
            "revenue_growth": info.get("revenueGrowth"),  # decimal
            "earnings_growth": info.get("earningsGrowth"),
            "free_cashflow": info.get("freeCashflow"),
            "operating_cashflow": info.get("operatingCashflow"),
            "profit_margins": info.get("profitMargins"),
            "gross_margins": info.get("grossMargins"),
            "price": info.get("regularMarketPrice"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "promoter_holding": None,  # not available via yfinance for Indian stocks
        }
        return {k: v for k, v in result.items() if v is not None}
    except Exception:
        return {}


def compute_quality_score(symbol: str, live: dict[str, Any], demo: dict[str, Any]) -> float:
    """
    5-rule quality score using live yfinance data where available,
    falling back to demo data for missing fields.
    Returns 0.5 when no reliable data exists (never 1.0 by default).

    Rules (total weight = 1.0):
      1. Return on equity (ROE)          25%  — 15%+ ROE target
      2. Free cash flow positive          20%  — positive FCF is a pass/fail gate
      3. Debt-to-equity ratio             20%  — <0.5 ideal (yfinance: <50%)
      4. Revenue growth                   20%  — >10% annual growth
      5. Profit margin                    15%  — >10% margin
    """
    if not live and not demo:
        return 0.5

    def get(live_key: str, demo_key: str, default: Any = None) -> Any:
        """Prefer live data; fall back to demo; then default."""
        v = live.get(live_key)
        if v is not None:
            return v
        v = demo.get(demo_key)
        if v is not None:
            return v
        return default

    score = 0.0
    data_points = 0

    # Rule 1 — Return on equity (weight 25%)
    roe = get("roe", "roce_5y")
    if roe is not None:
        # yfinance returns decimal (0.18); demo returns percentage-like (18.0 as roce_5y)
        # Normalise: if abs(roe) > 1.0 it's in percentage form
        roe_pct = roe * 100 if abs(roe) <= 1.0 else roe
        score += clamp(roe_pct / 20.0, 0.0, 1.0) * 0.25
        data_points += 1

    # Rule 2 — Free cash flow positive (weight 20%)
    fcf = live.get("free_cashflow") or live.get("operating_cashflow")
    if fcf is not None:
        score += (1.0 if fcf > 0 else 0.2) * 0.20
        data_points += 1
    elif demo.get("fcf_positive_years") is not None:
        score += clamp(demo["fcf_positive_years"] / 5.0, 0.0, 1.0) * 0.20
        data_points += 1

    # Rule 3 — Debt-to-equity (weight 20%)
    de = live.get("de_ratio")
    if de is not None:
        # yfinance D/E is expressed as percentage (e.g. 150 means 150%)
        de_ratio = de / 100.0 if de > 10 else de
        score += (1.0 if de_ratio < 0.5 else 0.4 if de_ratio < 1.5 else 0.1) * 0.20
        data_points += 1
    elif demo.get("de_ratio") is not None:
        score += (1.0 if demo["de_ratio"] < 0.5 else 0.4) * 0.20
        data_points += 1

    # Rule 4 — Revenue growth (weight 20%)
    rev_growth = live.get("revenue_growth")
    if rev_growth is not None:
        rev_pct = rev_growth * 100 if abs(rev_growth) <= 1.0 else rev_growth
        score += clamp(rev_pct / 15.0, 0.0, 1.0) * 0.20
        data_points += 1
    elif demo.get("revenue_consistency") is not None:
        score += clamp(demo["revenue_consistency"] / 10.0, 0.0, 1.0) * 0.20
        data_points += 1

    # Rule 5 — Profit margin (weight 15%)
    margin = live.get("profit_margins")
    if margin is not None:
        margin_pct = margin * 100 if abs(margin) <= 1.0 else margin
        score += clamp(margin_pct / 15.0, 0.0, 1.0) * 0.15
        data_points += 1
    elif demo.get("roce_5y") is not None:
        # Rough proxy: high ROCE implies reasonable margin
        score += clamp(demo["roce_5y"] / 25.0, 0.0, 1.0) * 0.15
        data_points += 1

    if data_points == 0:
        return 0.5  # No data — neutral, not perfect

    # Scale up score proportionally to the weight of rules that fired
    max_possible = 0.25 + 0.20 + 0.20 + 0.20 + 0.15  # = 1.0, but kept explicit
    return round(clamp(score / max_possible if max_possible > 0 else score, 0.0, 1.0), 3)


def fetch_financial_data(symbol: str) -> dict[str, Any]:
    """
    Fetch live financial data from yfinance.
    Appends .NS for NSE symbols automatically.
    Returns empty dict on failure and never raises.
    """
    try:
        import yfinance as yf

        nse_symbol = symbol if "." in symbol else f"{symbol}.NS"
        ticker = yf.Ticker(nse_symbol)

        def _as_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _series_value(frame: Any, labels: list[str], *, latest_two: bool = False) -> float | list[float] | None:
            if frame is None or getattr(frame, "empty", True):
                return None
            for label in labels:
                if label not in frame.index:
                    continue
                values = [_as_float(value) for value in frame.loc[label].tolist()]
                values = [value for value in values if value is not None]
                if not values:
                    continue
                return values[:2] if latest_two else values[0]
            return None

        def _normalise_ratio(value: float | None) -> float | None:
            if value is None:
                return None
            return value / 100.0 if abs(value) > 10 else value

        def _series_values(frame: Any, labels: list[str], *, limit: int | None = None) -> list[float]:
            if frame is None or getattr(frame, "empty", True):
                return []
            for label in labels:
                if label not in frame.index:
                    continue
                values = [_as_float(value) for value in frame.loc[label].tolist()]
                values = [value for value in values if value is not None]
                return values[:limit] if limit is not None else values
            return []

        def _consecutive_negative_quarters(frame: Any) -> int | None:
            if frame is None or getattr(frame, "empty", True):
                return None
            for label in [
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
            ]:
                if label not in frame.index:
                    continue
                negatives = 0
                for value in frame.loc[label].tolist():
                    numeric = _as_float(value)
                    if numeric is None:
                        continue
                    if numeric < 0:
                        negatives += 1
                    else:
                        break
                return negatives
            return None

        info = ticker.info or {}
        income_stmt = ticker.income_stmt
        quarterly_income_stmt = ticker.quarterly_income_stmt
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow

        ebit = _series_value(income_stmt, ["EBIT", "Operating Income", "Normalized EBITDA"])
        invested_capital_values = _series_value(balance_sheet, ["Invested Capital"], latest_two=True)
        roce_ttm = None
        if ebit is not None and isinstance(invested_capital_values, list) and invested_capital_values:
            avg_invested_capital = sum(invested_capital_values) / len(invested_capital_values)
            if avg_invested_capital > 0:
                roce_ttm = ebit / avg_invested_capital

        fcf_values = _series_values(cashflow, ["Free Cash Flow"], limit=5)
        free_cashflow = _as_float(info.get("freeCashflow"))
        if free_cashflow is None and fcf_values:
            free_cashflow = fcf_values[0]
        fcf_positive_years = sum(1 for value in fcf_values if value > 0)

        revenue_growth = _as_float(info.get("revenueGrowth"))
        if revenue_growth is None:
            revenue_values = _series_value(income_stmt, ["Total Revenue"], latest_two=True)
            if isinstance(revenue_values, list) and len(revenue_values) == 2 and revenue_values[1]:
                revenue_growth = (revenue_values[0] - revenue_values[1]) / revenue_values[1]

        debt_to_equity = None
        total_debt = _series_value(balance_sheet, ["Total Debt"])
        equity = _series_value(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
        if total_debt is not None and equity and equity > 0:
            debt_to_equity = total_debt / equity
        if debt_to_equity is None:
            debt_to_equity = _normalise_ratio(_as_float(info.get("debtToEquity")))

        current_price = _as_float(info.get("regularMarketPrice")) or _as_float(info.get("currentPrice"))
        if current_price is None:
            try:
                current_price = _as_float(ticker.fast_info.get("lastPrice"))
            except Exception:
                current_price = None

        result: dict[str, Any] = {
            "symbol": nse_symbol,
            "roce_ttm": roce_ttm,
            "returnOnCapitalEmployed": roce_ttm,
            "freeCashflow": free_cashflow,
            "free_cashflow": free_cashflow,
            "fcf_positive_years": fcf_positive_years,
            "revenueGrowth": revenue_growth,
            "revenue_growth": revenue_growth,
            "debtToEquity": debt_to_equity,
            "debt_to_equity": debt_to_equity,
            "promoter_holding_pct": None,
            "trailingEps": _as_float(info.get("trailingEps")),
            "currentPrice": current_price,
            "targetMeanPrice": _as_float(info.get("targetMeanPrice")),
            "negative_pat_quarters": _consecutive_negative_quarters(quarterly_income_stmt),
        }
        populated = {key: value for key, value in result.items() if value is not None}
        if not any(
            key in populated
            for key in ("roce_ttm", "freeCashflow", "revenueGrowth", "debtToEquity", "trailingEps")
        ):
            print(f"WARNING: yfinance returned empty data for {nse_symbol}")
            return {}
        return populated
    except Exception as exc:
        print(f"ERROR fetching data for {symbol}: {exc}")
        return {}


def compute_quality_score(symbol: str, live: dict[str, Any], demo: dict[str, Any]) -> float:
    """
    Quality moat score from live financial data only.
    Missing rules contribute no points, and a total live-data failure returns 0.0.
    """
    _ = demo

    if not live:
        print(f"WARNING: No financial data available for {symbol} - returning 0.0")
        return 0.0

    score = 0.0
    scores_applied = 0

    roce = live.get("roce_ttm") or live.get("returnOnCapitalEmployed")
    if roce is not None:
        score += (
            1.0 if roce > 0.18 else
            0.5 if roce > 0.10 else
            0.0 if roce > 0.0 else
            -0.5
        ) * 0.25
        scores_applied += 1

    fcf_positive_years = live.get("fcf_positive_years")
    fcf = live.get("freeCashflow") or live.get("free_cashflow")
    if isinstance(fcf_positive_years, int):
        score += (
            1.0 if fcf_positive_years >= 4 else
            0.5 if fcf_positive_years >= 3 else
            0.0
        ) * 0.25
        scores_applied += 1
    elif fcf is not None:
        score += (1.0 if fcf > 0 else 0.0) * 0.25
        scores_applied += 1

    rev_growth = live.get("revenueGrowth") or live.get("revenue_growth")
    if rev_growth is not None:
        score += (
            1.0 if rev_growth > 0.15 else
            0.7 if rev_growth > 0.08 else
            0.3 if rev_growth > 0.0 else
            0.0
        ) * 0.20
        scores_applied += 1

    promoter = live.get("promoter_holding_pct")
    if promoter is not None:
        score += (
            1.0 if promoter > 0.50 else
            0.7 if promoter > 0.35 else
            0.3
        ) * 0.15
        scores_applied += 1

    de = live.get("debtToEquity") or live.get("debt_to_equity")
    if de is not None:
        score += (
            1.0 if de < 0.5 else
            0.5 if de < 1.0 else
            0.0 if de < 2.0 else
            -0.5
        ) * 0.15
        scores_applied += 1

    if scores_applied == 0:
        print(f"WARNING: No usable financial rules fired for {symbol} - returning 0.0")
        return 0.0

    return round(clamp(score, 0.0, 1.0), 3)


def compute_net_return(
    current_price: float,
    analyst_target: float,
    holding_months: int = 24,
) -> float | None:
    """Stock-specific net return (%) after LTCG/STCG tax, based on analyst target price."""
    if not current_price or not analyst_target or current_price == 0:
        return None
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
        accumulation_bonus = 0.05 if flow and flow[0]["payload"].get("label") == "ACCUMULATION" else 0.0
        scored = []
        for candidate in state["candidates"]:
            demo_facts = self.provider.get_financials(candidate["symbol"])
            live_facts = fetch_financial_data(candidate["symbol"])

            # compute_quality_score uses 5 live-data rules; falls back to demo fields.
            base_score = compute_quality_score(candidate["symbol"], live_facts, demo_facts)

            # Geo signal bonus — applied on top of the computed base (small additive).
            geo_bonus = 0.04 if unified.get(candidate["sector"], {}).get("conviction") in {"BUY", "STRONG_BUY"} else 0.0
            # Flow / accumulation bonus.
            flow_bonus = accumulation_bonus
            selection_score = clamp(base_score + geo_bonus + flow_bonus, 0.0, 1.0)
            scored.append(candidate | {
                "quality_score": round(base_score, 3),
                "selection_score": round(selection_score, 3),
                "financials": demo_facts,
                "live_financials": live_facts,
            })
        scored.sort(key=lambda row: row.get("selection_score", row["quality_score"]), reverse=True)
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
                "run_summary": {
                    "run_id": run_id,
                    "recommendation_count": 0,
                    "blocked_reason": state["confidence"].get("market_note", "Market signals too weak."),
                },
            }

        recommendations = []
        for item in state["allocations"]:
            if item["entry_signal"] == "DO NOT ENTER":
                continue
            llm_rationale = self.llm.buy_rationale(item, state["portfolio_context"])
            rationale = llm_rationale or (
                f"{item['company_name']} adds differentiated {item['sector']} exposure with "
                f"{item['entry_signal'].lower()} timing."
            )
            price_ctx = item["price_context"]
            current_price = price_ctx["price"]
            analyst_target = price_ctx.get("analyst_target") or current_price * 1.25
            net_return_pct = compute_net_return(current_price, analyst_target, holding_months=24)

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
                "net_of_tax_return_projection": round(net_return_pct / 100, 4) if net_return_pct else None,
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
