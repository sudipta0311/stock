"""
scorer.py — compute hit rate and alpha from backtest_recommendations.

After replay.py has populated backtest_recommendations, this module:
1. Fetches forward prices from historical_prices for 3m/6m/12m windows.
2. Computes hit = recommendation beat NIFTY by >2% in the window.
3. Aggregates by run_id and confidence band.
4. Writes results back to backtest_runs.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_platform.data.repository import PlatformRepository

_log = logging.getLogger(__name__)

# A recommendation is a "hit" if it beats NIFTY by this many percentage points.
_HIT_ALPHA_THRESHOLD_PCT = 2.0

# Forward windows in trading weeks (approx months × 4.33).
_WINDOWS = {
    "3m":  13,
    "6m":  26,
    "12m": 52,
}


def _fetch_price(conn: Any, symbol: str, from_date: str, weeks_ahead: int) -> float | None:
    """Fetch the closest available price roughly `weeks_ahead` weeks after `from_date`."""
    target = (date.fromisoformat(from_date) + timedelta(weeks=weeks_ahead)).isoformat()
    row = conn.execute(
        """
        SELECT close_price FROM historical_prices
        WHERE symbol = ? AND date >= ?
        ORDER BY date ASC LIMIT 1
        """,
        (symbol, target),
    ).fetchone()
    return float(row["close_price"]) if row else None


def _forward_return(entry_price: float, forward_price: float | None) -> float | None:
    """Percentage return from entry to forward price."""
    if forward_price is None or entry_price <= 0:
        return None
    return round((forward_price / entry_price - 1) * 100, 4)


def score_run(repo: PlatformRepository, run_id: str) -> dict[str, Any]:
    """
    Compute hit rates and alpha for a completed replay run.

    Returns a summary dict and writes it to backtest_runs.
    """
    with repo.connect() as conn:
        recs = conn.execute(
            "SELECT * FROM backtest_recommendations WHERE run_id = ?",
            (run_id,),
        ).fetchall()

    if not recs:
        _log.warning("score_run: no recommendations found for run_id=%s", run_id)
        return {"run_id": run_id, "total": 0}

    results: list[dict[str, Any]] = []

    with repo.connect() as conn:
        for rec in recs:
            rec_date  = rec["recommendation_date"]
            symbol    = rec["symbol"]
            action    = rec["action"]
            confidence = rec["confidence_band"] or "YELLOW"

            entry_price = _fetch_price(conn, symbol, rec_date, 0)
            nifty_entry = _fetch_price(conn, "NIFTY",  rec_date, 0)

            if entry_price is None or nifty_entry is None:
                continue

            row: dict[str, Any] = {
                "run_id":           run_id,
                "symbol":           symbol,
                "recommendation_date": rec_date,
                "action":           action,
                "confidence_band":  confidence,
                "quality_score":    rec["quality_score"],
            }

            for window_label, weeks in _WINDOWS.items():
                fwd_price   = _fetch_price(conn, symbol, rec_date, weeks)
                nifty_fwd   = _fetch_price(conn, "NIFTY",  rec_date, weeks)
                stock_ret   = _forward_return(entry_price,  fwd_price)
                nifty_ret   = _forward_return(nifty_entry, nifty_fwd)
                alpha       = round(stock_ret - nifty_ret, 4) if (stock_ret is not None and nifty_ret is not None) else None
                hit         = bool(alpha is not None and alpha > _HIT_ALPHA_THRESHOLD_PCT)
                row[f"forward_return_{window_label}"] = stock_ret
                row[f"alpha_{window_label}"]          = alpha
                row[f"hit_{window_label}"]            = hit

            results.append(row)

        # Write forward returns back to backtest_recommendations.
        for row in results:
            conn.execute(
                """
                UPDATE backtest_recommendations SET
                    forward_return_3m=?, forward_return_6m=?, forward_return_12m=?, hit=?
                WHERE run_id=? AND symbol=? AND recommendation_date=?
                """,
                (
                    row.get("forward_return_3m"),
                    row.get("forward_return_6m"),
                    row.get("forward_return_12m"),
                    row.get("hit_6m"),        # primary hit = 6m window
                    run_id, row["symbol"], row["recommendation_date"],
                ),
            )
        conn.commit()

    # Aggregate overall and per-band stats.
    def _rate(rows: list[dict], window: str) -> float | None:
        hits   = [r for r in rows if r.get(f"hit_{window}") is not None]
        if not hits:
            return None
        return round(sum(r[f"hit_{window}"] for r in hits) / len(hits), 4)

    def _alpha(rows: list[dict], window: str) -> float | None:
        alphas = [r[f"alpha_{window}"] for r in rows if r.get(f"alpha_{window}") is not None]
        return round(sum(alphas) / len(alphas), 4) if alphas else None

    # In backtest mode all emitted recs are the top-N candidates (LLM action labels
    # are skipped), so score all of them rather than filtering on action string.
    # For production runs with real action labels, WAIT recs are excluded by the filter below.
    buy_actions = {"ACCUMULATE", "STRONG ENTER", "SMALL INITIAL"}
    has_buy_actions = any(r["action"] in buy_actions for r in results)
    buy_results = results if not has_buy_actions else [r for r in results if r["action"] in buy_actions]

    summary: dict[str, Any] = {
        "run_id":               run_id,
        "total_recommendations": len(results),
        "buy_recommendations":   len(buy_results),
        "hit_rate_3m":          _rate(buy_results, "3m"),
        "hit_rate_6m":          _rate(buy_results, "6m"),
        "hit_rate_12m":         _rate(buy_results, "12m"),
        "alpha_3m":             _alpha(buy_results, "3m"),
        "alpha_6m":             _alpha(buy_results, "6m"),
        "alpha_12m":            _alpha(buy_results, "12m"),
        "by_confidence": {},
    }

    for band in ("GREEN", "YELLOW", "RED"):
        band_rows = [r for r in buy_results if r["confidence_band"] == band]
        if band_rows:
            summary["by_confidence"][band] = {
                "count":      len(band_rows),
                "hit_rate_6m": _rate(band_rows, "6m"),
                "alpha_6m":    _alpha(band_rows, "6m"),
            }

    # Write hit rates to backtest_runs.
    with repo.connect() as conn:
        conn.execute(
            """
            UPDATE backtest_runs SET
                total_recommendations=?,
                hit_rate_3m=?, hit_rate_6m=?, hit_rate_12m=?,
                alpha_3m=?,  alpha_6m=?,  alpha_12m=?
            WHERE run_id=?
            """,
            (
                summary["total_recommendations"],
                summary["hit_rate_3m"],  summary["hit_rate_6m"],  summary["hit_rate_12m"],
                summary["alpha_3m"],     summary["alpha_6m"],     summary["alpha_12m"],
                run_id,
            ),
        )
        conn.commit()

    _log.info(
        "score_run %s: total=%d hit_6m=%s alpha_6m=%s",
        run_id, summary["total_recommendations"],
        summary["hit_rate_6m"], summary["alpha_6m"],
    )
    return summary
