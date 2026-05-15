"""
replay.py — replay FLOW 2 against historical snapshots.

Key design:
- HistoricalDataProvider mimics LiveMarketDataProvider's interface but reads
  from historical_prices and historical_fundamentals instead of live APIs.
- today is a *settable* attribute updated each iteration of the replay loop.
  WARNING: do not construct HistoricalDataProvider once and forget to update
  today — every replay step will silently use the same date, producing results
  that look plausible but are garbage.
- LLM nodes (validate_qualitative, finalize_recommendation) are skipped via
  request["skip_llm_nodes"] = True; only the quant pipeline is replayed.
- Known wall-clock leaks in the buy path (stock_validator.check_recently_listed,
  result_date_fetcher) are accepted as v1 limitations and documented in README.md.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from stock_platform.agents.buy_agents import BuyAgents
from stock_platform.config import AppConfig
from stock_platform.data.repository import PlatformRepository
from stock_platform.graphs.buy_graph import build_buy_graph
from stock_platform.services.llm import PlatformLLM

_log = logging.getLogger(__name__)

# Simple seeded test portfolio — enough to drive sector-gap logic.
_SEED_PORTFOLIO_CONTEXT = {
    "portfolio_meta": {"portfolio_last_updated": "2020-01-01T00:00:00Z"},
    "raw_holdings": [],
    "normalized_exposure": [
        {"symbol": "HDFCBANK",  "company_name": "HDFC Bank",  "sector": "Banking",           "total_weight": 8.0,  "source_mix_json": "{}", "attribution_json": "[]", "updated_at": "2020-01-01"},
        {"symbol": "INFY",      "company_name": "Infosys",    "sector": "IT",                "total_weight": 6.0,  "source_mix_json": "{}", "attribution_json": "[]", "updated_at": "2020-01-01"},
        {"symbol": "RELIANCE",  "company_name": "Reliance",   "sector": "Energy",            "total_weight": 7.0,  "source_mix_json": "{}", "attribution_json": "[]", "updated_at": "2020-01-01"},
    ],
    "overlap_scores": [],
    "identified_gaps": [
        # "Unknown" catches all HistoricalDataProvider stocks (no live sector data in snapshots).
        {"sector": "Unknown",       "underweight_pct": 20.0, "conviction": "BUY",      "score": 0.80, "reason": "Backtest universe"},
        {"sector": "Capital Goods", "underweight_pct": 8.0,  "conviction": "BUY",      "score": 0.75, "reason": "Infra push"},
        {"sector": "Healthcare",    "underweight_pct": 5.0,  "conviction": "POSITIVE",  "score": 0.60, "reason": "Pharma tailwinds"},
        {"sector": "Auto",          "underweight_pct": 6.0,  "conviction": "POSITIVE",  "score": 0.65, "reason": "EV cycle"},
        {"sector": "Chemicals",     "underweight_pct": 4.0,  "conviction": "NEUTRAL",   "score": 0.50, "reason": "Import substitution"},
        {"sector": "FMCG",          "underweight_pct": 3.0,  "conviction": "POSITIVE",  "score": 0.55, "reason": "Rural demand"},
        {"sector": "IT",            "underweight_pct": 3.0,  "conviction": "POSITIVE",  "score": 0.58, "reason": "AI exports"},
    ],
    "unified_signals": [],
    "user_preferences": {"macro_thesis": "India infra + domestic consumption"},
    "watchlist": [],
    "direct_equity_holdings": [],
    "source_health_warning": None,
}


class HistoricalDataProvider:
    """
    Drop-in replacement for LiveMarketDataProvider that reads from snapshot tables.

    ``today`` MUST be updated before each replay iteration:
        provider.today = replay_date
    """

    def __init__(self, repo: PlatformRepository, replay_date: date) -> None:
        self.repo  = repo
        self.today = replay_date   # updated per-iteration by the replay loop
        self._price_cache: dict[str, float | None] = {}
        self._external_call_count: int = 0  # increments when snapshot has no data (live fallback risk)

    # ── price helpers ──────────────────────────────────────────────────────────

    def _get_price(self, symbol: str) -> float | None:
        key = f"{symbol}:{self.today}"
        if key in self._price_cache:
            return self._price_cache[key]
        with self.repo.connect() as conn:
            row = conn.execute(
                "SELECT close_price FROM historical_prices WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (symbol, self.today.isoformat()),
            ).fetchone()
        price = float(row["close_price"]) if row else None
        if price is None:
            self._external_call_count += 1
        self._price_cache[key] = price
        return price

    def _get_fundamentals(self, symbol: str) -> dict[str, Any]:
        with self.repo.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM historical_fundamentals
                WHERE symbol = ? AND snapshot_date <= ?
                ORDER BY snapshot_date DESC LIMIT 1
                """,
                (symbol, self.today.isoformat()),
            ).fetchone()
        if not row:
            self._external_call_count += 1
            return {}
        return dict(row)

    # ── LiveMarketDataProvider interface ──────────────────────────────────────

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.upper().strip()

    def get_index_members(self, index_name: str) -> list[dict[str, Any]]:
        """Return symbols that have price data as of replay_date."""
        with self.repo.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM historical_prices WHERE date <= ? AND symbol != 'NIFTY'",
                (self.today.isoformat(),),
            ).fetchall()
        return [{"symbol": row["symbol"], "company_name": row["symbol"], "sector": "Unknown"} for row in rows]

    def get_financials(self, symbol: str) -> dict[str, Any]:
        fund   = self._get_fundamentals(symbol)
        price  = self._get_price(symbol)
        result = {
            "symbol":              symbol,
            "currentPrice":        price,
            "current_price":       price,
            "roce_pct":            fund.get("roce"),
            "eps":                 fund.get("eps"),
            "debt_to_equity":      fund.get("debt_equity"),
            "revenue_growth_pct":  fund.get("revenue_growth"),
            "promoter_holding":    fund.get("promoter_holding"),
        }
        return {k: v for k, v in result.items() if v is not None}

    def get_stock_snapshot(self, symbol: str) -> dict[str, Any]:
        price = self._get_price(symbol)
        return {
            "symbol":           symbol,
            "company_name":     symbol,
            "sector":           "Unknown",
            "price":            price,
            "analyst_target":   None,
            "market_cap_cr":    None,
            "beta":             None,
            "avg_daily_value_cr": None,
            "pe_trailing":      None,
            "pe_forward":       None,
            "drawdown_from_52w": None,
        }

    def get_risk_metrics(self, symbol: str) -> dict[str, Any]:
        fund = self._get_fundamentals(symbol)
        return {
            "avg_daily_value_cr":   5.0,   # assume liquid — no volume data
            "beta":                 1.0,
            "promoter_pledge_pct":  0.0,
            "pledge_trend":         "stable",
            "sebi_flag":            False,
            "debt_to_equity":       fund.get("debt_equity"),
        }

    def get_price_context(self, symbol: str) -> dict[str, Any]:
        price = self._get_price(symbol)
        return {"price": price, "analyst_target": None, "drawdown_from_52w": None,
                "price_change_1m": None, "price_change_6m": None}

    def get_stock_news(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "sentiment_score": 0.0, "headline": "[backtest — no news]"}

    def get_geopolitical_signals(self) -> list[dict[str, Any]]:
        return []

    def get_policy_signals(self) -> list[dict[str, Any]]:
        return []

    def get_flow_signals(self) -> list[dict[str, Any]]:
        return []

    def get_contrarian_signals(self) -> list[dict[str, Any]]:
        return []

    def get_monitoring_price_series(self, symbol: str, entry_price: float | None = None) -> dict[str, Any]:
        return {}

    def get_current_market_signal(self, sector: str) -> float:
        return 0.5


class _NoOpLLM:
    """Stub LLM used in backtest mode — returns deterministic empty responses."""
    provider = "backtest"

    def industry_reasoning(self, *a, **kw) -> str:
        return "[backtest]"

    def qualitative_analysis(self, *a, **kw) -> dict:
        return {"approved": True, "confidence": 0.7, "reasoning": "[backtest]"}

    def buy_rationale(self, *a, **kw) -> str:
        return "[backtest — LLM skipped]"

    def monitoring_rationale(self, *a, **kw) -> dict:
        return {}

    def thesis_review(self, *a, **kw) -> dict:
        return {"status": "INTACT", "reasoning": "[backtest]"}

    def synthesise_comparison(self, *a, **kw) -> str:
        return "[backtest]"


def _monday_range(start: date, end: date) -> list[date]:
    """Return every Monday between start (inclusive) and end (exclusive)."""
    mondays = []
    current = start + timedelta(days=(7 - start.weekday()) % 7)
    while current < end:
        mondays.append(current)
        current += timedelta(weeks=1)
    return mondays


def replay(
    repo: PlatformRepository,
    config: AppConfig,
    start_date: date,
    end_date: date,
    run_id: str | None = None,
    top_n: int = 5,
    index_name: str = "NIFTY200",
) -> str:
    """
    Replay FLOW 2 for every Monday in [start_date, end_date).

    Returns the backtest run_id (written to backtest_runs).
    LLM nodes are skipped; only the quant pipeline runs.
    """
    run_id = run_id or f"bt-{uuid4().hex[:10]}"
    mondays = _monday_range(start_date, end_date)
    _log.info("replay: run_id=%s  %d weeks (%s → %s)", run_id, len(mondays), start_date, end_date)

    provider = HistoricalDataProvider(repo, replay_date=start_date)
    llm      = _NoOpLLM()
    agents   = BuyAgents(repo=repo, provider=provider, config=config, llm=llm)  # type: ignore[arg-type]
    graph    = build_buy_graph(agents)

    total_recs = 0

    for replay_date in mondays:
        t_start = time.perf_counter()

        # ── Phase (a): provider setup ─────────────────────────────────────────
        provider.today = replay_date          # ← critical: update per iteration
        provider._price_cache.clear()         # evict stale per-date cache
        provider._external_call_count = 0     # reset per-week counter

        ctx = dict(_SEED_PORTFOLIO_CONTEXT)
        ctx["portfolio_meta"] = {
            "portfolio_last_updated": (replay_date - timedelta(days=1)).isoformat()
        }

        request: dict[str, Any] = {
            "index_name":      index_name,
            "top_n":           top_n,
            "risk_profile":    "Balanced",
            "horizon_months":  12,
            "corpus":          1_000_000,
            "skip_llm_nodes":  True,          # skip validate_qualitative + finalize LLM calls
        }

        initial_state: dict[str, Any] = {
            "request":           request,
            "portfolio_context": ctx,
        }

        t_setup_done = time.perf_counter()

        # ── Phase (b): graph execution ────────────────────────────────────────
        try:
            result = graph.invoke(initial_state)
        except Exception as exc:
            _log.warning("replay %s on %s: %r", run_id, replay_date, exc)
            continue

        t_graph_done = time.perf_counter()

        recs = result.get("recommendations", [])
        market_band = result.get("confidence", {}).get("band", "YELLOW")

        # ── Phase (c): DB writes ──────────────────────────────────────────────
        # check_confidence returns a single market-level band (same for every rec in
        # the week). In backtest the production DB always satisfies signal_count≥5 so
        # every week is GREEN. Derive per-rec confidence_band from within-week rank
        # instead (recs are already sorted desc by quality_score by the graph):
        #   rank 1      → GREEN   (best pick of the week)
        #   ranks 2–3   → YELLOW
        #   ranks 4+    → RED
        _RANK_BANDS = ["GREEN", "YELLOW", "YELLOW", "RED", "RED"]

        score_log: list[tuple[str, float]] = []
        for rank, rec in enumerate(recs):
            symbol = getattr(rec, "symbol", None) or rec.get("symbol") if isinstance(rec, dict) else rec.symbol
            action = getattr(rec, "action", None) or rec.get("action", "WAIT") if isinstance(rec, dict) else rec.action
            score  = getattr(rec, "score", 0.5) if hasattr(rec, "score") else rec.get("score", 0.5) if isinstance(rec, dict) else 0.5
            rec_band = _RANK_BANDS[rank] if rank < len(_RANK_BANDS) else "RED"

            score_log.append((symbol, round(float(score), 3)))
            with repo.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO backtest_recommendations
                        (run_id, symbol, recommendation_date, action, confidence_band, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, symbol, recommendation_date) DO NOTHING
                    """,
                    (run_id, symbol, replay_date.isoformat(), action, rec_band, score),
                )
                conn.commit()
            total_recs += 1

        t_db_done = time.perf_counter()

        _log.info(
            "replay %s: %s → %d recs (market_band=%s)  "
            "total=%.1fs setup=%.3fs graph=%.1fs db=%.3fs http_fallbacks=%d  "
            "quality_scores=%s",
            run_id, replay_date, len(recs), market_band,
            t_db_done   - t_start,
            t_setup_done - t_start,
            t_graph_done - t_setup_done,
            t_db_done   - t_graph_done,
            provider._external_call_count,
            score_log,
        )

    # Write summary stub to backtest_runs (scorer.py fills in hit rates later).
    created_at = __import__("datetime").datetime.now(__import__("datetime").UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with repo.connect() as conn:
        conn.execute(
            """
            INSERT INTO backtest_runs
                (run_id, start_date, end_date, weights_hash, total_recommendations, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                total_recommendations=excluded.total_recommendations,
                created_at=excluded.created_at
            """,
            (run_id, start_date.isoformat(), end_date.isoformat(), None, total_recs, created_at),
        )
        conn.commit()

    _log.info("replay complete: run_id=%s  total_recommendations=%d", run_id, total_recs)
    return run_id
