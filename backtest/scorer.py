"""
scorer.py — compute hit rate, alpha, and rank IC from backtest_recommendations.

After replay.py has populated backtest_recommendations, this module:
1. Fetches forward prices from historical_prices for 3m/6m/12m windows.
2. Computes hit = recommendation beat NIFTY by >2% in the window.
3. Computes Spearman rank IC per week between composite_score and forward alpha.
4. Aggregates by run_id and confidence band.
5. Writes results back to backtest_runs.

Rank IC metrics:
  mean_ic_6m  : mean weekly Spearman IC over the 6m window
  ic_tstat    : t-stat = mean_ic / (std_ic / sqrt(n_weeks))
  icir        : IC information ratio = mean_ic / std_ic
  decile_spread_6m : 6m alpha of top decile minus bottom decile
  median_alpha_6m  : median per-rec alpha at 6m
"""
from __future__ import annotations

import logging
import math
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

# Minimum number of weeks with valid IC before reporting mean_ic (else None).
_MIN_IC_WEEKS = 4

# Minimum fraction of 12m-window completions required before reporting 12m metrics.
_MIN_12M_COVERAGE = 0.30


def _spearman_ic(scores: list[float], alphas: list[float]) -> float | None:
    """
    Compute Spearman rank correlation (IC) between composite_scores and forward_alphas.
    Returns None when fewer than 3 paired observations are available.
    """
    pairs = [(s, a) for s, a in zip(scores, alphas) if s is not None and a is not None]
    n = len(pairs)
    if n < 3:
        return None

    def _rank_list(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[order[j]] == vals[order[j + 1]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    s_vals = [p[0] for p in pairs]
    a_vals = [p[1] for p in pairs]
    rs = _rank_list(s_vals)
    ra = _rank_list(a_vals)
    mean_rs = sum(rs) / n
    mean_ra = sum(ra) / n
    cov = sum((rs[i] - mean_rs) * (ra[i] - mean_ra) for i in range(n))
    var_s = sum((r - mean_rs) ** 2 for r in rs)
    var_a = sum((r - mean_ra) ** 2 for r in ra)
    denom = (var_s * var_a) ** 0.5
    return round(cov / denom, 4) if denom > 0 else None


def _rank_ic_summary(ic_values: list[float]) -> dict[str, float | None]:
    """
    Compute mean IC, t-stat, and ICIR from a list of per-week IC values.
    Returns None for each metric when fewer than _MIN_IC_WEEKS values exist.
    """
    valid = [v for v in ic_values if v is not None]
    if len(valid) < _MIN_IC_WEEKS:
        return {"mean_ic": None, "ic_tstat": None, "icir": None}

    n = len(valid)
    mean_ic = sum(valid) / n
    if n < 2:
        return {"mean_ic": round(mean_ic, 4), "ic_tstat": None, "icir": None}
    variance = sum((v - mean_ic) ** 2 for v in valid) / (n - 1)
    std_ic = variance ** 0.5 or 1e-9
    ic_tstat = round(mean_ic / (std_ic / math.sqrt(n)), 4)
    icir     = round(mean_ic / std_ic, 4)
    return {"mean_ic": round(mean_ic, 4), "ic_tstat": ic_tstat, "icir": icir}


def _decile_spread(scores_alphas: list[tuple[float, float]]) -> float | None:
    """
    Alpha spread between top and bottom decile of composite_score.
    Returns None when fewer than 10 obs are present.
    """
    valid = [(s, a) for s, a in scores_alphas if s is not None and a is not None]
    if len(valid) < 10:
        return None
    valid.sort(key=lambda x: x[0])
    n = len(valid)
    decile_size = max(1, n // 10)
    bottom_alpha = sum(a for _, a in valid[:decile_size]) / decile_size
    top_alpha    = sum(a for _, a in valid[-decile_size:]) / decile_size
    return round(top_alpha - bottom_alpha, 4)


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
    Compute hit rates, alpha, and rank IC for a completed replay run.

    Returns a summary dict and writes it to backtest_runs.
    12m metrics are suppressed (set to None) when fewer than _MIN_12M_COVERAGE
    of recommendations have a completed 12m window, to avoid misleading stats.
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
            rec_date   = rec["recommendation_date"]
            symbol     = rec["symbol"]
            action     = rec["action"]
            confidence = rec["confidence_band"] or "YELLOW"

            entry_price = _fetch_price(conn, symbol, rec_date, 0)
            nifty_entry = _fetch_price(conn, "NIFTY",  rec_date, 0)

            if entry_price is None or nifty_entry is None:
                continue

            row: dict[str, Any] = {
                "run_id":              run_id,
                "symbol":              symbol,
                "recommendation_date": rec_date,
                "action":              action,
                "confidence_band":     confidence,
                "quality_score":       rec["quality_score"],
                "composite_score":     rec["composite_score"],
            }

            for window_label, weeks in _WINDOWS.items():
                fwd_price = _fetch_price(conn, symbol, rec_date, weeks)
                nifty_fwd = _fetch_price(conn, "NIFTY",  rec_date, weeks)
                stock_ret = _forward_return(entry_price, fwd_price)
                nifty_ret = _forward_return(nifty_entry, nifty_fwd)
                alpha     = round(stock_ret - nifty_ret, 4) if (stock_ret is not None and nifty_ret is not None) else None
                hit       = bool(alpha is not None and alpha > _HIT_ALPHA_THRESHOLD_PCT)
                row[f"forward_return_{window_label}"] = stock_ret
                row[f"alpha_{window_label}"]          = alpha
                row[f"hit_{window_label}"]            = hit

            results.append(row)

        # Write forward returns (including alpha_6m) back to backtest_recommendations.
        for row in results:
            conn.execute(
                """
                UPDATE backtest_recommendations SET
                    forward_return_3m=?, forward_return_6m=?, forward_return_12m=?,
                    alpha_6m=?, hit=?
                WHERE run_id=? AND symbol=? AND recommendation_date=?
                """,
                (
                    row.get("forward_return_3m"),
                    row.get("forward_return_6m"),
                    row.get("forward_return_12m"),
                    row.get("alpha_6m"),
                    row.get("hit_6m"),
                    run_id, row["symbol"], row["recommendation_date"],
                ),
            )
        conn.commit()

    # Aggregate overall and per-band stats.
    def _rate(rows: list[dict], window: str) -> float | None:
        hits = [r for r in rows if r.get(f"hit_{window}") is not None]
        return round(sum(r[f"hit_{window}"] for r in hits) / len(hits), 4) if hits else None

    def _alpha_mean(rows: list[dict], window: str) -> float | None:
        alphas = [r[f"alpha_{window}"] for r in rows if r.get(f"alpha_{window}") is not None]
        return round(sum(alphas) / len(alphas), 4) if alphas else None

    def _alpha_median(rows: list[dict], window: str) -> float | None:
        alphas = sorted(r[f"alpha_{window}"] for r in rows if r.get(f"alpha_{window}") is not None)
        if not alphas:
            return None
        n = len(alphas)
        mid = n // 2
        return round((alphas[mid - 1] + alphas[mid]) / 2 if n % 2 == 0 else alphas[mid], 4)

    # In backtest mode all emitted recs are top-N candidates (no LLM action labels).
    buy_actions = {"ACCUMULATE", "STRONG ENTER", "SMALL INITIAL"}
    has_buy_actions = any(r["action"] in buy_actions for r in results)
    buy_results = results if not has_buy_actions else [r for r in results if r["action"] in buy_actions]

    # Suppress 12m metrics when insufficient data.
    n_with_12m = sum(1 for r in buy_results if r.get("alpha_12m") is not None)
    report_12m = (n_with_12m / len(buy_results)) >= _MIN_12M_COVERAGE if buy_results else False

    summary: dict[str, Any] = {
        "run_id":                run_id,
        "total_recommendations": len(results),
        "buy_recommendations":   len(buy_results),
        "hit_rate_3m":           _rate(buy_results, "3m"),
        "hit_rate_6m":           _rate(buy_results, "6m"),
        "hit_rate_12m":          _rate(buy_results, "12m") if report_12m else None,
        "alpha_3m":              _alpha_mean(buy_results, "3m"),
        "alpha_6m":              _alpha_mean(buy_results, "6m"),
        "alpha_12m":             _alpha_mean(buy_results, "12m") if report_12m else None,
        "median_alpha_6m":       _alpha_median(buy_results, "6m"),
        "by_confidence":         {},
    }

    for band in ("GREEN", "YELLOW", "RED"):
        band_rows = [r for r in buy_results if r["confidence_band"] == band]
        if band_rows:
            summary["by_confidence"][band] = {
                "count":       len(band_rows),
                "hit_rate_6m": _rate(band_rows, "6m"),
                "alpha_6m":    _alpha_mean(band_rows, "6m"),
            }

    # ── Rank IC (Spearman) per week ───────────────────────────────────────────
    # Group results by recommendation_date, compute per-week IC, then aggregate.
    from collections import defaultdict
    week_groups: dict[str, list[dict]] = defaultdict(list)
    for r in buy_results:
        week_groups[r["recommendation_date"]].append(r)

    ic_values_6m: list[float] = []
    score_alpha_pairs_6m: list[tuple[float, float]] = []

    for week_date, week_rows in week_groups.items():
        scores = [r.get("composite_score") for r in week_rows]
        alphas = [r.get("alpha_6m") for r in week_rows]
        ic = _spearman_ic(scores, alphas)
        if ic is not None:
            ic_values_6m.append(ic)
        for s, a in zip(scores, alphas):
            if s is not None and a is not None:
                score_alpha_pairs_6m.append((s, a))

    ic_stats = _rank_ic_summary(ic_values_6m)
    summary["mean_ic_6m"]     = ic_stats["mean_ic"]
    summary["ic_tstat"]       = ic_stats["ic_tstat"]
    summary["icir"]           = ic_stats["icir"]
    summary["decile_spread_6m"] = _decile_spread(score_alpha_pairs_6m)

    # Write all metrics to backtest_runs.
    with repo.connect() as conn:
        conn.execute(
            """
            UPDATE backtest_runs SET
                total_recommendations=?,
                hit_rate_3m=?,  hit_rate_6m=?,  hit_rate_12m=?,
                alpha_3m=?,     alpha_6m=?,     alpha_12m=?,
                mean_ic_6m=?,   ic_tstat=?,     icir=?,
                decile_spread_6m=?, median_alpha_6m=?
            WHERE run_id=?
            """,
            (
                summary["total_recommendations"],
                summary["hit_rate_3m"],  summary["hit_rate_6m"],  summary["hit_rate_12m"],
                summary["alpha_3m"],     summary["alpha_6m"],     summary["alpha_12m"],
                summary["mean_ic_6m"],   summary["ic_tstat"],     summary["icir"],
                summary["decile_spread_6m"], summary["median_alpha_6m"],
                run_id,
            ),
        )
        conn.commit()

    _log.info(
        "score_run %s: total=%d hit_6m=%s alpha_6m=%s mean_ic_6m=%s decile_spread=%s",
        run_id, summary["total_recommendations"],
        summary["hit_rate_6m"], summary["alpha_6m"],
        summary["mean_ic_6m"],  summary["decile_spread_6m"],
    )
    return summary
