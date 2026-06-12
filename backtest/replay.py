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
from stock_platform.agents.composite_score import compute_composite_scores
from stock_platform.config import AppConfig
from stock_platform.data.repository import PlatformRepository
from stock_platform.graphs.buy_graph import build_buy_graph
from stock_platform.services.llm import PlatformLLM

_log = logging.getLogger(__name__)

# Minimal seed portfolio context — drives the buy graph's sector-gap analysis.
# All backtest stocks return sector "Unknown" (HistoricalDataProvider), so only
# the "Unknown" gap entry matters; it keeps the gap score high for every candidate.
_SEED_PORTFOLIO_CONTEXT = {
    "portfolio_meta": {"portfolio_last_updated": "2020-01-01T00:00:00Z"},
    "raw_holdings": [],
    "normalized_exposure": [
        {"symbol": "HDFCBANK", "company_name": "HDFC Bank", "sector": "Banking",
         "total_weight": 8.0, "source_mix_json": "{}", "attribution_json": "[]", "updated_at": "2020-01-01"},
        {"symbol": "INFY",     "company_name": "Infosys",   "sector": "IT",
         "total_weight": 6.0, "source_mix_json": "{}", "attribution_json": "[]", "updated_at": "2020-01-01"},
        {"symbol": "RELIANCE", "company_name": "Reliance",  "sector": "Energy",
         "total_weight": 7.0, "source_mix_json": "{}", "attribution_json": "[]", "updated_at": "2020-01-01"},
    ],
    "overlap_scores": [],
    "identified_gaps": [
        {"sector": "Unknown", "underweight_pct": 20.0, "conviction": "BUY", "score": 0.80, "reason": "Backtest universe"},
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
                WHERE symbol = ? AND available_date <= ?
                ORDER BY available_date DESC LIMIT 1
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


def _rebalance_dates(start: date, end: date, freq: str = "weekly") -> list[date]:
    """
    Return rebalance dates between start (inclusive) and end (exclusive).

    freq="weekly"  : every Monday (same as _monday_range)
    freq="monthly" : first Monday of each calendar month
    """
    if freq not in ("weekly", "monthly"):
        raise ValueError(f"freq must be 'weekly' or 'monthly', got {freq!r}")

    all_mondays = _monday_range(start, end)
    if freq == "weekly":
        return all_mondays

    # Monthly: keep only the first Monday of each (year, month) pair.
    seen: set[tuple[int, int]] = set()
    result: list[date] = []
    for d in all_mondays:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def replay(
    repo: PlatformRepository,
    config: AppConfig,
    start_date: date,
    end_date: date,
    run_id: str | None = None,
    top_n: int = 5,
    index_name: str = "NIFTY200",
    freq: str = "weekly",
) -> str:
    """
    Replay FLOW 2 for each rebalance date in [start_date, end_date).

    freq="weekly"  : rebalance every Monday
    freq="monthly" : rebalance on the first Monday of each month

    Returns the backtest run_id (written to backtest_runs).
    LLM nodes are skipped; only the quant pipeline runs.
    """
    run_id = run_id or f"bt-{uuid4().hex[:10]}"
    mondays = _rebalance_dates(start_date, end_date, freq=freq)
    _log.info(
        "replay: run_id=%s  freq=%s  %d dates (%s → %s)",
        run_id, freq, len(mondays), start_date, end_date,
    )

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

        # ── Phase (c): composite scoring + DB writes ──────────────────────────
        # Per-rec confidence_band is derived from within-week composite rank
        # (not market-level band, which is GREEN for all backtest weeks):
        #   rank 1      → GREEN
        #   ranks 2–3   → YELLOW
        #   ranks 4+    → RED
        _RANK_BANDS = ["GREEN", "YELLOW", "YELLOW", "RED", "RED"]

        # Build a universe-of-week list for cross-sectional composite scoring.
        # Each entry carries financials fetched from the snapshot tables.
        week_candidates: list[dict[str, Any]] = []
        for rec in recs:
            sym = getattr(rec, "symbol", None) or (rec.get("symbol") if isinstance(rec, dict) else None)
            act = getattr(rec, "action", None) or (rec.get("action", "WAIT") if isinstance(rec, dict) else "WAIT")
            scr = getattr(rec, "score", 0.5) if hasattr(rec, "score") else (rec.get("score", 0.5) if isinstance(rec, dict) else 0.5)
            if sym:
                fin = provider.get_financials(sym)
                week_candidates.append({
                    "symbol":        sym,
                    "action":        act,
                    "quality_score": float(scr),
                    "financials":    fin,
                    "sector":        "Unknown",  # HistoricalDataProvider always returns Unknown
                })

        scored = compute_composite_scores(week_candidates)
        scored.sort(key=lambda c: c.get("composite_score", 0.0), reverse=True)

        for rank, cand in enumerate(scored):
            rec_band = _RANK_BANDS[rank] if rank < len(_RANK_BANDS) else "RED"
            with repo.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO backtest_recommendations
                        (run_id, symbol, recommendation_date, action, confidence_band,
                         quality_score, composite_score, quality_pct, valuation_pct, momentum_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, symbol, recommendation_date) DO NOTHING
                    """,
                    (
                        run_id, cand["symbol"], replay_date.isoformat(),
                        cand["action"], rec_band,
                        cand["quality_score"],
                        cand.get("composite_score"),
                        cand.get("quality_pct"),
                        cand.get("valuation_pct"),
                        cand.get("momentum_pct"),
                    ),
                )
                conn.commit()
            total_recs += 1

        t_db_done = time.perf_counter()

        _log.info(
            "replay %s: %s → %d recs (market_band=%s)  "
            "total=%.1fs setup=%.3fs graph=%.1fs db=%.3fs http_fallbacks=%d  "
            "composite_scores=%s",
            run_id, replay_date, len(scored), market_band,
            t_db_done    - t_start,
            t_setup_done - t_start,
            t_graph_done - t_setup_done,
            t_db_done    - t_graph_done,
            provider._external_call_count,
            [(c["symbol"], round(c.get("composite_score", 0.0), 3)) for c in scored],
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
