from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from stock_platform.config import AppConfig
from stock_platform.data.db import database_connection
from stock_platform.models import MonitoringAction
from stock_platform.utils.entry_calculator import fetch_analyst_consensus_target
from utils.tax_calculator import calculate_pnl, should_exit

BANKING_SECTORS = {
    "BANKS",
    "BANKS - REGIONAL",
    "PRIVATE BANKING",
    "FINANCIAL SERVICES",
    "NBFC",
    "INSURANCE",
}

SYMBOL_ALIASES = {
    "HINDUNILVR": ["HUL", "HINDUNILVR", "HINDUSTAN UNILEVER"],
    "KOTAKBANK": ["KOTAK", "KOTAKBANK", "KOTAK MAHINDRA"],
    "LT": ["LT", "LARSENTOUBRO", "LARSEN"],
    "TCS": ["TCS", "TATACONSULTANCY"],
    "KWIL": ["KWIL", "KALYANJEWELS", "KALYAN JEWELLERS"],
    "SBICARD": ["SBICARD", "SBI CARDS", "SBI CARD"],
    "AXITA": ["AXITA", "AXITA COTTON"],
}

# Remaps truncated company-name symbols (produced by mf_lookup._symbol_from_name
# before the normalisation fix) to proper NSE tickers. Applied in
# _build_overlap_lookup() so existing DB rows with old keys resolve correctly.
_NAME_TO_TICKER: dict[str, str] = {
    "HDFCBANKLIMITED":    "HDFCBANK",
    "ICICIBANKLIMITED":   "ICICIBANK",
    "RELIANCEINDUSTRIES": "RELIANCE",
    "AXISBANKLIMITED":    "AXISBANK",
    "TATACONSULTANCYSER": "TCS",
    "INFOSYSLIMITED":     "INFY",
    "BHARTIAIRTELLIMITE": "BHARTIARTL",
    "STATEBANKOFINDIA":   "SBIN",
    "SUNPHARMACEUTICALI": "SUNPHARMA",
    "HCLTECHNOLOGIESLIM": "HCLTECH",
    "BIOCONLIMITED":      "BIOCON",
    "SBILIFEINSURANCECO": "SBILIFE",
    "95MUTHOOTFINANCELI": "MUTHOOTFIN",
}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_ratio(value: Any) -> float | None:
    raw = _as_float(value)
    if raw is None:
        return None
    return raw / 100.0 if abs(raw) > 1.5 else raw


def _metric_ratio(stock_data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        ratio = _as_ratio(stock_data.get(key))
        if ratio is not None:
            return ratio
    return None


def _metric_float(stock_data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        numeric = _as_float(stock_data.get(key))
        if numeric is not None:
            return numeric
    return None


def debug_monitoring_data(symbol: str, stock_data: dict[str, Any]) -> None:
    print(f"MONITORING DATA CHECK {symbol}:")
    print(f"  roce:           {stock_data.get('roce')}")
    print(f"  roe:            {stock_data.get('roe')}")
    print(f"  revenue_growth: {stock_data.get('revenue_growth')}")
    print(f"  debt_equity:    {stock_data.get('debt_equity')}")
    print(f"  current_price:  {stock_data.get('current_price')}")
    print(f"  overlap_pct:    {stock_data.get('overlap_pct')}")


def _clean_company_key(value: Any) -> str:
    text = "".join(ch for ch in str(value or "").upper() if ch.isalnum())
    for suffix in ("LIMITED", "LTD"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _score_debt_to_equity(debt_equity: float) -> float:
    if debt_equity <= 0.25:
        return 1.0
    if debt_equity >= 2.0:
        return 0.0
    return max(0.0, 1.0 - ((debt_equity - 0.25) / 1.75))


def _overlap_value(entry: Any) -> float:
    if isinstance(entry, dict):
        return float(entry.get("overlap_pct", 0.0) or 0.0)
    return float(entry or 0.0)


def get_overlap_pct(symbol: str, portfolio_ctx: dict[str, Any]) -> float:
    """
    Look up overlap_pct with alias and case-insensitive fallback.
    Returns the highest matching overlap so a zero exact match does not mask
    a positive alias match from the portfolio DB.
    """
    normalized_symbol = str(symbol or "").upper().strip()
    candidates = [normalized_symbol, *SYMBOL_ALIASES.get(normalized_symbol, [])]
    found = 0.0

    for candidate in candidates:
        if candidate in portfolio_ctx:
            found = max(found, _overlap_value(portfolio_ctx[candidate]))

    for key, value in portfolio_ctx.items():
        key_upper = str(key or "").upper()
        key_clean = _clean_company_key(key)
        for candidate in candidates:
            if key_upper == str(candidate).upper() or key_clean == _clean_company_key(candidate):
                found = max(found, _overlap_value(value))

    return found


def get_monitoring_metrics(symbol: str, stock_data: dict[str, Any], sector: str | None = None) -> dict[str, Any]:
    sector_name = str(sector or stock_data.get("sector") or "").strip().upper()
    symbol_name = str(symbol or "").upper()
    is_bank = sector_name in BANKING_SECTORS or any(
        token in symbol_name for token in ["BANK", "FIN", "CARD", "HDFC", "ICICI", "KOTAK", "SBI", "AXIS"]
    )
    if is_bank:
        return {
            "quality_metric": _as_ratio(
                stock_data.get("returnOnEquity")
                or stock_data.get("roe")
                or stock_data.get("roe_pct")
                or stock_data.get("roce")
                or stock_data.get("roce_ttm")
            ),
            "growth_metric": _as_ratio(
                stock_data.get("revenueGrowth")
                or stock_data.get("revenue_growth")
                or stock_data.get("earningsGrowth")
                or stock_data.get("pat_growth_pct")
            ),
            "leverage_metric": _as_float(
                stock_data.get("debtToEquity")
                or stock_data.get("debt_equity")
                or stock_data.get("debt_to_equity")
                or 1.0
            ),
            "metric_type": "banking",
        }
    return {
        "quality_metric": _as_ratio(
            stock_data.get("roce")
            or stock_data.get("roce_ttm")
            or stock_data.get("returnOnCapitalEmployed")
            or stock_data.get("returnOnEquity")
            or stock_data.get("roe")
            or stock_data.get("roe_pct")
        ),
        "growth_metric": _as_ratio(
            stock_data.get("revenue_growth")
            or stock_data.get("revenueGrowth")
            or stock_data.get("earningsGrowth")
            or stock_data.get("pat_growth_pct")
        ),
        "leverage_metric": _as_float(
            stock_data.get("debt_equity")
            or stock_data.get("debt_to_equity")
            or stock_data.get("debtToEquity")
            or stock_data.get("de_ratio")
        ),
        "metric_type": "standard",
    }


def compute_monitoring_score(symbol: str, stock_data: dict[str, Any], sector: str | None = None) -> float | None:
    try:
        metrics = get_monitoring_metrics(symbol, stock_data, sector)
        quality_metric = metrics["quality_metric"]
        growth_metric = metrics["growth_metric"]
        leverage_metric = metrics["leverage_metric"]
        metric_type = metrics["metric_type"]

        if quality_metric is None and growth_metric is None:
            print(
                f"MONITORING SCORE FAIL {symbol}: no quality or growth metric found "
                f"sector={sector} metric_type={metric_type}"
            )
            return None

        score = 0.5
        if quality_metric is not None:
            if metric_type == "banking":
                score += 0.2 if quality_metric > 0.12 else 0.1 if quality_metric > 0.08 else -0.1
            else:
                score += 0.2 if quality_metric > 0.15 else 0.1 if quality_metric > 0.10 else -0.1
        if growth_metric is not None:
            score += 0.15 if growth_metric > 0.15 else 0.05 if growth_metric > 0.08 else -0.1
        if leverage_metric is not None and metric_type != "banking":
            score += 0.1 if leverage_metric < 0.5 else 0.0 if leverage_metric < 1.0 else -0.1
        return round(min(max(score, 0.0), 1.0), 2)
    except Exception as exc:
        print(f"MONITORING SCORE ERROR {symbol}: {exc}")
        return None


def apply_overlap_override(
    symbol: str,
    exit_rec: dict[str, Any],
    db_path: str,
    *,
    overlap_pct: float | None = None,
    turso_database_url: str = "",
    turso_auth_token: str = "",
    turso_sync_interval_seconds: int | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Suppress BUY MORE when the same stock is already owned meaningfully via MFs.
    """
    overlap = float(overlap_pct or 0.0)
    if overlap <= 0:
        with database_connection(
            db_path,
            turso_url=turso_database_url,
            turso_token=turso_auth_token,
            sync_interval=turso_sync_interval_seconds,
        ) as conn:
            row = conn.execute(
                "SELECT overlap_pct FROM overlap_scores "
                "WHERE UPPER(TRIM(symbol)) = UPPER(TRIM(?))",
                (symbol,),
            ).fetchone()
            overlap = float(row["overlap_pct"]) if row else 0.0

    if overlap >= 2.0 and exit_rec["exit_recommendation"] == "BUY MORE":
        exit_rec = dict(exit_rec)
        exit_rec["exit_recommendation"] = "HOLD - already in MFs"
        exit_rec["reasoning"] += (
            f" However {overlap:.1f}% already held via mutual funds - "
            "direct purchase adds concentration not diversification."
        )
        exit_rec["urgency"] = "LOW"

    return exit_rec, overlap

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
    overlap_pct = _as_float(row.get("overlap_pct"))
    if overlap_pct and overlap_pct > 0:
        if overlap_pct > 10:
            parts.append(
                f"MF overlap HIGH at {overlap_pct:.1f}% — MFs already give substantial exposure; "
                "direct holding adds concentration, not diversification"
            )
        elif overlap_pct > 5:
            parts.append(
                f"MF overlap MODERATE at {overlap_pct:.1f}% — direct holding adds marginal incremental exposure"
            )
        else:
            parts.append(
                f"MF overlap LOW at {overlap_pct:.1f}% — direct holding genuinely diversifies"
            )

    caution_flag = str(row.get("caution_flag") or "").strip()
    if caution_flag:
        parts.append(f"Caution flag: {caution_flag}")

    return " | ".join(parts)


def _build_overlap_lookup(ctx: dict[str, Any], normalize_symbol: Any) -> dict[str, float]:
    exact_overlap: dict[str, float] = {}
    company_overlap: dict[str, float] = {}
    exposure_by_symbol = {
        normalize_symbol(row.get("symbol") or ""): row
        for row in ctx.get("normalized_exposure", [])
        if row.get("symbol")
    }

    for overlap_row in ctx.get("overlap_scores", []):
        symbol = normalize_symbol(overlap_row.get("symbol") or "")
        overlap_pct = float(overlap_row.get("overlap_pct", 0.0) or 0.0)
        if symbol:
            exact_overlap[symbol] = max(exact_overlap.get(symbol, 0.0), overlap_pct)
            # Remap legacy truncated company-name symbols to proper NSE tickers
            # so existing DB rows resolve correctly before re-ingestion.
            ticker = _NAME_TO_TICKER.get(symbol)
            if ticker:
                exact_overlap[ticker] = max(exact_overlap.get(ticker, 0.0), overlap_pct)
        exposure_row = exposure_by_symbol.get(symbol, {})
        company_key = _clean_company_key(exposure_row.get("company_name"))
        if company_key:
            company_overlap[company_key] = max(company_overlap.get(company_key, 0.0), overlap_pct)

    resolved: dict[str, float] = dict(exact_overlap)
    for exposure_row in ctx.get("normalized_exposure", []):
        symbol = normalize_symbol(exposure_row.get("symbol") or "")
        company_key = _clean_company_key(exposure_row.get("company_name"))
        if not symbol or not company_key:
            continue
        resolved[symbol] = max(resolved.get(symbol, 0.0), company_overlap.get(company_key, 0.0))
    return resolved


# ── Valuation floor helpers (Fix 1 & 3) ─────────────────────────────────────

def _valuation_floor_active(d: dict) -> bool:
    """
    Returns True when a stock is near its 52W low AND trading at a PE discount
    to its own 5-year average, signalling a sector-wide correction rather than
    a thesis breach.  All four conditions must be met:
      - quant_score >= 0.55 (fundamentals not broken)
      - price <= week52_low * 1.15 (within 15% of 52W low)
      - current_pe < pe_5yr_avg * 0.85 (PE at 15%+ discount to own history)
      - thesis_status in ("INTACT", "WEAKENED") — not "BREACHED"
    Returns False if any required field is missing so the guard never fires
    on incomplete data.
    """
    pe      = d.get("current_pe") or 0
    avg_pe  = d.get("pe_5yr_avg") or d.get("sector_avg_pe") or 0
    low52w  = d.get("week52_low") or 0
    price   = d.get("current_price") or 0
    quant   = d.get("quant_score", 0)
    thesis  = d.get("thesis_status", "INTACT")

    if not all([pe, avg_pe, low52w, price]):
        return False
    return (
        quant >= 0.55
        and price <= low52w * 1.15
        and pe < avg_pe * 0.85
        and thesis in ("INTACT", "WEAKENED")
    )


def _ltcg_guard(d: dict) -> dict:
    """
    Returns {"block": True, ...} when selling under STCG would burn more than
    40% of the gross gain as tax and the holding is within 90 days of LTCG
    qualification.  Does NOT fire if the thesis is BREACHED or the position
    is already in a loss / already LTCG.
    """
    pnl_pct      = d.get("pnl_pct", 0)
    holding_days = d.get("holding_days")
    gross_pnl    = d.get("gross_pnl", 0)

    if pnl_pct <= 0 or holding_days is None or holding_days >= 365:
        return {"block": False}

    stcg_tax = gross_pnl * 0.20
    ltcg_tax = max(0.0, gross_pnl - 125000) * 0.125
    days_to_ltcg   = 365 - holding_days
    tax_pct_of_gain = (stcg_tax / gross_pnl * 100) if gross_pnl > 0 else 0

    block = (
        pnl_pct > 5
        and days_to_ltcg < 90
        and tax_pct_of_gain > 40
        and d.get("thesis_status") != "BREACHED"
    )
    return {
        "block": block,
        "reason": f"STCG exit {days_to_ltcg}d before LTCG",
        "tax": round(stcg_tax, 0),
        "tax_pct": round(tax_pct_of_gain, 1),
        "ltcg_date": (date.today() + timedelta(days=days_to_ltcg)).strftime("%d %b %Y"),
    }


def check_sector_correction(sector: str, holdings: list) -> bool:
    """
    Returns True when ≥2 stocks in the same sector are >10% below their
    52-week high simultaneously — a signal of a macro/sector-wide pullback
    rather than company-specific deterioration.
    Expects each holding dict to contain a ``pct_from_52w_high`` key
    (negative value, e.g. -14 means 14% below the 52W high).
    """
    sector_stocks = [s for s in holdings if (s.get("sector") or "") == sector]
    down_count = sum(
        1 for s in sector_stocks
        if (s.get("pct_from_52w_high") or 0) < -10
    )
    return down_count >= 2


LEGACY_AUTOMATIC_REVIEW_TRIGGERS = {
    "revenue_degrowth_severe": lambda s: (s.get("revenue_growth", 0) or 0) < -0.20,
    "near_zero_margin": lambda s: (s.get("pat_margin", 1) or 1) < 0.01,
    "pe_extreme_low_profit": lambda s: (
        (s.get("pe_ratio", 0) or 0) > 50 and (s.get("pat_margin", 1) or 1) < 0.02
    ),
    "promoter_selling": lambda s: (s.get("promoter_holding_change_3yr", 0) or 0) < -0.15,
}


def _legacy_check_auto_review(symbol: str, stock_data: dict) -> dict:
    triggers_fired = []
    for name, condition in LEGACY_AUTOMATIC_REVIEW_TRIGGERS.items():
        try:
            if condition(stock_data):
                triggers_fired.append(name)
        except Exception:
            pass
    if len(triggers_fired) >= 2:
        return {
            "override_action": "REVIEW",
            "override_severity": "HIGH",
            "override_reason": (
                f"Auto-review triggered: {', '.join(triggers_fired)}. "
                "Standard HOLD threshold does not apply — manual review needed."
            ),
        }
    return {}


AUTOMATIC_REVIEW_TRIGGERS: dict[str, dict[str, Any]] = {
    "revenue_degrowth_severe": {
        "condition": lambda s: (
            (_metric_ratio(s, "revenue_growth", "revenueGrowth", "revenue_growth_pct", "revenue_growth_ttm") or 0.0)
            < -0.20
        ),
        "message": "Revenue declining >20% YoY - severe top-line deterioration",
    },
    "near_zero_margin": {
        "condition": lambda s: (
            0.0
            < (_metric_ratio(s, "pat_margin", "net_margin", "profit_margins", "pat_margin_pct", "net_margin_pct") or 0.0)
            < 0.01
        ),
        "message": "PAT margin below 1% - business generating no profit",
    },
    "pe_extreme_low_profit": {
        "condition": lambda s: (
            (_metric_float(s, "pe_ratio", "pe_trailing") or 0.0) > 80.0
            and (_metric_ratio(s, "pat_margin", "net_margin", "profit_margins", "pat_margin_pct", "net_margin_pct") or 0.0)
            < 0.02
        ),
        "message": "PE >80x with <2% margin - valuation entirely speculative",
    },
    "promoter_selling_sustained": {
        "condition": lambda s: (_metric_ratio(s, "promoter_holding_change_3yr") or 0.0) < -0.15,
        "message": "Promoter holding down >15% over 3 years - sustained insider selling",
    },
    "pat_momentum_collapsing": {
        "condition": lambda s: (_metric_ratio(s, "pat_growth_pct", "earningsGrowth") or 0.0) < -0.40,
        "message": "PAT down >40% YoY - earnings collapsing",
    },
    "micro_cap_commodity": {
        "condition": lambda s: (
            (_metric_float(s, "market_cap_cr", "marketCapCr") or 999999.0) < 500.0
            and any(
                token in str(s.get("sector", "") or "").lower()
                for token in ("textile", "textiles", "cotton", "commodity", "agri", "agriculture", "sugar", "paper")
            )
        ),
        "message": "Micro-cap commodity processor - standard thresholds unreliable",
    },
}


def check_auto_review(symbol: str, stock_data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate warning conditions before the monitoring LLM path.
    Two or more triggers force REVIEW/HIGH; one trigger becomes a caution flag.
    """
    fired: list[dict[str, str]] = []

    for name, trigger in AUTOMATIC_REVIEW_TRIGGERS.items():
        try:
            if trigger["condition"](stock_data):
                fired.append({"name": name, "message": str(trigger["message"])})
                print(f"AUTO-REVIEW TRIGGER: {symbol} - {name}: {trigger['message']}")
        except Exception as exc:
            print(f"Trigger check error {symbol}/{name}: {exc}")

    if len(fired) >= 2:
        trigger_summary = " | ".join(item["message"] for item in fired)
        return {
            "override": True,
            "override_action": "REVIEW",
            "override_severity": "HIGH",
            "override_urgency": "HIGH",
            "override_rationale": (
                f"WARNING AUTO-REVIEW: {len(fired)} warning conditions fired - {trigger_summary}. "
                "Standard HOLD threshold does not apply. Manual review required before next portfolio rebalancing."
            ),
            "triggers": fired,
        }
    if len(fired) == 1:
        return {
            "override": False,
            "caution_flag": fired[0]["message"],
            "triggers": fired,
        }
    return {}


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
        print(
            "Portfolio context loaded: "
            f"{len(ctx.get('direct_equity_holdings', []))} direct holdings, "
            f"{len(ctx.get('raw_holdings', []))} raw rows, "
            f"{len(ctx.get('overlap_scores', []))} overlap rows"
        )
        sample_holdings = [
            {
                "symbol": row.get("symbol"),
                "avg_buy_price": row.get("avg_buy_price"),
                "buy_date": row.get("buy_date"),
            }
            for row in ctx.get("direct_equity_holdings", [])[:3]
        ]
        print(f"Portfolio context sample: {sample_holdings}")
        if (
            not ctx.get("direct_equity_holdings")
            or not any(float(row.get("overlap_pct", 0.0) or 0.0) > 0 for row in ctx.get("overlap_scores", []))
        ):
            diagnostics = getattr(self.repo, "portfolio_table_diagnostics", lambda: {})()
            if diagnostics:
                print(f"Portfolio table diagnostic: {diagnostics}")

        # Build monitor universe: direct equity holdings + watchlist only.
        # MF / ETF indirect holdings are excluded — the user monitors those
        # through the fund manager, not stock-by-stock.
        norm_by_symbol = {row["symbol"]: row for row in ctx["normalized_exposure"]}

        total_direct_value = sum(
            row["market_value"]
            for row in ctx["raw_holdings"]
            if row["holding_type"] == "direct_equity" and row.get("market_value")
        ) or 1.0

        overlap_lookup = _build_overlap_lookup(ctx, self.provider.normalize_symbol)

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
            entry["overlap_pct"] = get_overlap_pct(sym, overlap_lookup)
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
            entry["overlap_pct"] = get_overlap_pct(sym, overlap_lookup)
            monitor_universe.append(entry)

        broker_map = {
            self.provider.normalize_symbol(item["symbol"]): item
            for item in ctx.get("direct_equity_holdings", [])
            if item.get("symbol")
        }
        for entry in monitor_universe:
            buy_info = broker_map.get(entry["symbol"], {})
            entry["avg_buy_price"] = buy_info.get("avg_buy_price")
            entry["buy_date"] = buy_info.get("buy_date")
            entry["current_price"] = buy_info.get("current_price")
        ctx["monitor_universe"] = monitor_universe
        ctx["direct_equity_buy_map"] = broker_map
        ctx["portfolio_overlap_map"] = overlap_lookup
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
        for idx, holding in enumerate(state["portfolio_context"]["monitor_universe"]):
            financials = dict(self.provider.get_financials(holding["symbol"]) or {})
            financials["overlap_pct"] = holding.get("overlap_pct", 0.0)
            financials["current_price"] = holding.get("current_price")
            financials["sector"] = holding.get("sector")
            financials["roe"] = (
                financials.get("returnOnEquity")
                or financials.get("roe")
                or financials.get("roe_pct")
            )
            financials["roce"] = (
                financials.get("roce_ttm")
                or financials.get("returnOnCapitalEmployed")
                or financials.get("roce_5y")
            )
            financials["revenue_growth"] = (
                financials.get("revenueGrowth")
                or financials.get("revenue_growth")
            )
            financials["debt_equity"] = (
                financials.get("debt_equity")
                or financials.get("debt_to_equity")
                or financials.get("debtToEquity")
                or financials.get("de_ratio")
            )
            auto_review = check_auto_review(holding["symbol"], financials)
            if idx < 3:
                debug_monitoring_data(holding["symbol"], financials)
            scores.append(
                {
                    "symbol": holding["symbol"],
                    "quant_score": compute_monitoring_score(
                        holding["symbol"],
                        financials,
                        holding.get("sector"),
                    ),
                    "financials": financials,
                    "auto_review": auto_review,
                }
            )
        return {"quant_scores": scores}

    def review_thesis(self, state: dict[str, Any]) -> dict[str, Any]:
        unified = {row["sector"]: row for row in state["portfolio_context"]["unified_signals"]}
        quant_rows = {row["symbol"]: row for row in state["quant_scores"]}
        quant_scores = {symbol: row["quant_score"] for symbol, row in quant_rows.items()}
        stock_news = {row["symbol"]: row for row in state["stock_reviews"]}

        # Fix 2: build per-holding pct_from_52w_high so we can detect sector corrections
        _universe = state["portfolio_context"]["monitor_universe"]
        _enriched: list[dict[str, Any]] = []
        for _h in _universe:
            try:
                _pctx = self.provider.get_price_context(_h["symbol"])
                _dd_from_high = float(_pctx.get("drawdown_from_52w") or 0)
            except Exception:
                _dd_from_high = 0.0
            _enriched.append({**_h, "pct_from_52w_high": -_dd_from_high})
        sector_correction_cache: dict[str, bool] = {}
        for _h in _enriched:
            _sec = _h.get("sector") or ""
            if _sec and _sec not in sector_correction_cache:
                sector_correction_cache[_sec] = check_sector_correction(_sec, _enriched)

        thesis = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            sector_signal = unified.get(holding["sector"], {})
            quant = quant_scores.get(holding["symbol"])
            auto_review = quant_rows.get(holding["symbol"], {}).get("auto_review", {})

            # Hard rules always override LLM — enforce first.
            if sector_signal.get("conviction") == "STRONG_AVOID":
                status = "BREACHED"
                llm_reasoning = ""
            elif auto_review.get("override"):
                status = "WEAKENED"
                llm_reasoning = auto_review["override_rationale"]
            elif quant is None:
                status = "UNKNOWN"
                llm_reasoning = "Financial data unavailable for monitoring score."
            else:
                # LLM:Sonnet — nuanced multi-signal thesis assessment.
                news = stock_news.get(holding["symbol"], {})
                _sector_flag = sector_correction_cache.get(holding.get("sector") or "", False)
                llm_result = self.llm.thesis_review(
                    holding, quant, sector_signal, news,
                    sector_correction_flag=_sector_flag,
                )
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
        quant_rows = {row["symbol"]: row for row in state["quant_scores"]}
        quant_map = {symbol: row["quant_score"] for symbol, row in quant_rows.items()}
        auto_review_map = {symbol: row.get("auto_review", {}) for symbol, row in quant_rows.items()}
        buy_map = state["portfolio_context"].get("direct_equity_buy_map", {})
        actions = []
        for holding in state["portfolio_context"]["monitor_universe"]:
            symbol = holding["symbol"]
            thesis = thesis_map[symbol]["status"]
            drawdown = drawdown_map[symbol]["severity"]
            sentiment = stock_reviews[symbol]["sentiment_score"]
            quant = quant_map.get(symbol)
            pnl = None
            exit_rec = None
            urgency = "LOW"
            overlap_pct = float(holding.get("overlap_pct", 0.0) or 0.0)
            auto = auto_review_map.get(symbol, {})
            caution_flag = str(auto.get("caution_flag") or "").strip()

            buy_info = buy_map.get(symbol)
            current_price = drawdown_map[symbol].get("current_price")
            analyst_target = (
                fetch_analyst_consensus_target(symbol, float(current_price))
                if current_price
                else self.provider.get_price_context(symbol).get("analyst_target")
            )

            if auto.get("override"):
                actions.append(
                    {
                        "symbol": symbol,
                        "action": auto["override_action"],
                        "severity": auto["override_severity"],
                        "urgency": auto["override_urgency"],
                        "rationale": auto["override_rationale"],
                        "overlap_pct": overlap_pct,
                        "pnl": pnl,
                        "exit_recommendation": exit_rec,
                        "analyst_target": analyst_target,
                        "auto_review_override": True,
                        "auto_review": auto,
                    }
                )
                continue

            if quant is None:
                actions.append(
                    {
                        "symbol": symbol,
                        "action": "DATA_UNAVAILABLE",
                        "severity": "UNKNOWN",
                        "urgency": "LOW",
                        "rationale": (
                            f"Monitoring skipped - financial data not available for {symbol}. "
                            "Re-run after data refresh."
                        ),
                        "overlap_pct": overlap_pct,
                        "pnl": pnl,
                        "exit_recommendation": exit_rec,
                        "analyst_target": analyst_target,
                        "caution_flag": caution_flag,
                        "auto_review": auto,
                    }
                )
                continue

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
                exit_rec, overlap_pct = apply_overlap_override(
                    symbol=symbol,
                    exit_rec=exit_rec,
                    db_path=str(self.config.db_path),
                    overlap_pct=overlap_pct,
                    turso_database_url=self.config.turso_database_url,
                    turso_auth_token=self.config.turso_auth_token,
                    turso_sync_interval_seconds=self.config.turso_sync_interval_seconds,
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

            # ── Fix 1: Valuation floor gate ──────────────────────────────────
            # Blocks EXIT/TRIM on quality stocks that are near their 52W low
            # AND trading at a PE discount vs own history — indicates a
            # sector-wide macro correction, not a thesis breach.
            vf_override_rationale: str | None = None
            action_upper = str(action).upper()
            if thesis != "BREACHED" and ("EXIT" in action_upper or "TRIM" in action_upper):
                _fin = quant_rows.get(symbol, {}).get("financials", {})
                _vf_data = {
                    "current_pe":   _as_float(_fin.get("pe_trailing") or _fin.get("pe_ratio")),
                    "pe_5yr_avg":   _as_float(_fin.get("pe_5yr_avg")),
                    "sector_avg_pe": _as_float(_fin.get("sector_pe")),
                    "week52_low":   _as_float(
                        _fin.get("week52_low") or _fin.get("fiftyTwoWeekLow")
                    ),
                    "current_price": float(current_price) if current_price else None,
                    "quant_score":  quant,
                    "thesis_status": thesis,
                }
                if _valuation_floor_active(_vf_data):
                    _pe_disp   = _vf_data["current_pe"] or 0
                    _low52w_dp = _vf_data["week52_low"] or 0
                    action    = "HOLD"
                    severity  = "LOW"
                    urgency   = "LOW"
                    vf_override_rationale = (
                        f"Valuation floor active: PE {_pe_disp:.1f}× "
                        f"near 52W low ₹{_low52w_dp:,.0f}. "
                        "Sector-wide correction, not thesis breach."
                    )

            # ── Fix 3: LTCG guard ────────────────────────────────────────────
            # Downgrades EXIT/TRIM to HOLD when STCG tax erodes >40% of gain
            # and LTCG threshold is within 90 days.
            ltcg_addendum: str = ""
            if pnl and ("EXIT" in str(action).upper() or "TRIM" in str(action).upper()):
                _guard_data = {
                    "pnl_pct":      pnl.get("pnl_pct", 0),
                    "holding_days": pnl.get("days_held"),
                    "gross_pnl":    pnl.get("gross_pnl", 0),
                    "thesis_status": thesis,
                }
                _tax_check = _ltcg_guard(_guard_data)
                if _tax_check["block"]:
                    action   = "HOLD"
                    severity = "LOW"
                    urgency  = "LOW"
                    ltcg_addendum = (
                        f" [LTCG GUARD: {_tax_check['reason']}. "
                        f"Exit tax cost ₹{_tax_check['tax']:,.0f} "
                        f"({_tax_check['tax_pct']:.0f}% of gain). "
                        f"Review at LTCG threshold on {_tax_check['ltcg_date']}]"
                    )

            if vf_override_rationale:
                rationale = vf_override_rationale + ltcg_addendum
            else:
                rationale = format_monitoring_rationale(
                    {
                        "symbol": symbol,
                        "thesis_status": thesis,
                        "quant_score": quant,
                        "overlap_pct": overlap_pct,
                        "caution_flag": caution_flag,
                    },
                    pnl,
                    exit_rec,
                )
                if ltcg_addendum:
                    rationale += ltcg_addendum
            actions.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "severity": severity,
                    "urgency": urgency,
                    "rationale": rationale,
                    "overlap_pct": overlap_pct,
                    "pnl": pnl,
                    "exit_recommendation": exit_rec,
                    "analyst_target": analyst_target,
                    "caution_flag": caution_flag,
                    "auto_review": auto,
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
            llm_fallback_note = ""
            if row.get("auto_review_override"):
                pass  # Auto-override is sufficient — skip LLM call.
            elif row["action"] in {"BUY MORE", "HOLD", "TRIM", "SELL", "REPLACE"}:
                try:
                    llm_result = self.llm.monitoring_rationale(row, thesis, drawdown)
                    llm_rationale = str((llm_result or {}).get("rationale", "")).strip()
                    if not llm_rationale or len(llm_rationale) < 20:
                        print(f"MONITORING LLM EMPTY: {row['symbol']}")
                        llm_fallback_note = "[LLM analysis failed - data context may be empty]"
                        if llm_result is not None:
                            llm_result = dict(llm_result)
                            llm_result["rationale"] = ""
                except Exception as exc:
                    print(f"MONITORING LLM ERROR {row['symbol']}: {exc}")
                    llm_result = None
                    llm_fallback_note = f"[LLM error: {type(exc).__name__}]"
            # Use LLM-confirmed action/severity/rationale if parsing succeeded;
            # fall back to deterministic values so a JSON failure never drops a row.
            final_action = llm_result["action"] if llm_result else row["action"]
            final_severity = llm_result["severity"] if llm_result else row["severity"]
            final_rationale = row["rationale"]
            if llm_result and str(llm_result.get("rationale", "")).strip():
                final_rationale = str(llm_result["rationale"]).strip()
            elif llm_fallback_note:
                final_rationale = f"{row['rationale']} {llm_fallback_note}".strip()
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
                        "overlap_pct": row.get("overlap_pct", 0.0),
                        "exit_recommendation": row.get("exit_recommendation"),
                        "analyst_target": row.get("analyst_target"),
                        "caution_flag": row.get("caution_flag"),
                        "auto_review": row.get("auto_review", {}),
                        "llm_used": bool(llm_result),
                    },
                )
            )
        self.repo.save_monitoring_actions(run_id, rows)
        return {
            "replacement_prompt": prompt,
            "run_summary": {"run_id": run_id, "action_count": len(rows)},
        }
