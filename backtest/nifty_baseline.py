"""
nifty_baseline.py — NIFTY-ETF buy-and-hold baseline for backtest comparison.

Answers: does the system's alpha (+X pp vs NIFTY) represent genuine signal,
or is it indistinguishable from noise?

Method:
1. For each Monday in a run's date range, record NIFTY entry and forward prices.
2. Compute NIFTY's own 3m / 6m raw return (same logic as scorer.py).
3. Load each stock rec's raw forward return from backtest_recommendations.
4. Compare portfolio average return vs NIFTY average return per window.
5. Run a paired one-sample t-test on alpha = stock_return - nifty_return.
   H0: mean alpha == 0.   Reject H0 if p < 0.05.

Run:
    python -m backtest.nifty_baseline --run-id <run_id>
"""
from __future__ import annotations

import argparse
import json
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

from stock_platform.config import AppConfig, load_app_env
from stock_platform.data.repository import PlatformRepository

_log = logging.getLogger(__name__)

_WINDOWS: dict[str, int] = {"3m": 13, "6m": 26, "12m": 52}


# ── Price helpers ─────────────────────────────────────────────────────────────

def _nifty_price(conn: Any, as_of: str, weeks_ahead: int = 0) -> float | None:
    target = (date.fromisoformat(as_of) + timedelta(weeks=weeks_ahead)).isoformat()
    if weeks_ahead == 0:
        row = conn.execute(
            "SELECT close_price FROM historical_prices "
            "WHERE symbol='NIFTY' AND date <= ? ORDER BY date DESC LIMIT 1",
            (target,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT close_price FROM historical_prices "
            "WHERE symbol='NIFTY' AND date >= ? ORDER BY date ASC LIMIT 1",
            (target,),
        ).fetchone()
    return float(row["close_price"]) if row else None


def _pct_return(entry: float | None, fwd: float | None) -> float | None:
    if entry is None or fwd is None or entry <= 0:
        return None
    return round((fwd / entry - 1) * 100, 4)


# ── t-test (no scipy dependency) ─────────────────────────────────────────────

def _t_test_one_sample(values: list[float]) -> dict[str, float | None]:
    """
    One-sample t-test against H0: mean == 0.
    Returns {mean, std, t_stat, p_value (two-tailed), n}.
    p-value is approximated via the t-distribution CDF using a series expansion
    sufficient for n ≥ 10.
    """
    n = len(values)
    if n < 2:
        return {"n": n, "mean": values[0] if values else None, "std": None,
                "t_stat": None, "p_value": None}
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / (n - 1)
    std  = math.sqrt(var)
    if std == 0:
        return {"n": n, "mean": mean, "std": 0.0, "t_stat": None, "p_value": None}
    t    = mean / (std / math.sqrt(n))
    df   = n - 1
    p    = _t_pvalue(t, df)
    return {"n": n, "mean": round(mean, 4), "std": round(std, 4),
            "t_stat": round(t, 4), "p_value": round(p, 4)}


def _t_pvalue(t: float, df: int) -> float:
    """
    Two-tailed p-value for t-statistic with df degrees of freedom.
    Uses the regularised incomplete beta function via its continued-fraction
    approximation (Numerical Recipes).  Accurate to ~4 decimal places.
    """
    x = df / (df + t * t)
    # Regularised incomplete beta I_x(df/2, 1/2)
    p_one_tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return round(min(1.0, 2.0 * p_one_tail), 6)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta via continued fraction (Lentz's method)."""
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    # Compute log of the beta function factor
    lbeta  = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    factor = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Continued fraction via Lentz
    fpmin  = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d   = 1.0 / d
    h   = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d  = 1.0 + aa * d
        c  = 1.0 + aa / c
        if abs(d) < fpmin: d = fpmin
        if abs(c) < fpmin: c = fpmin
        d  = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d  = 1.0 + aa * d
        c  = 1.0 + aa / c
        if abs(d) < fpmin: d = fpmin
        if abs(c) < fpmin: c = fpmin
        d  = 1.0 / d
        delta = d * c
        h    *= delta
        if abs(delta - 1.0) < 3e-7:
            break
    return factor * h


# ── Main comparison function ──────────────────────────────────────────────────

def nifty_baseline(repo: PlatformRepository, run_id: str) -> dict[str, Any]:
    """
    Compare portfolio raw returns to NIFTY raw returns for a completed run.
    Returns a report dict with per-window stats and t-test results.
    """
    with repo.connect() as conn:
        run_row = conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not run_row:
            raise ValueError(f"run_id {run_id!r} not found")

        recs = conn.execute(
            """
            SELECT symbol, recommendation_date,
                   forward_return_3m, forward_return_6m, forward_return_12m
            FROM backtest_recommendations
            WHERE run_id = ? ORDER BY recommendation_date
            """,
            (run_id,),
        ).fetchall()

    recs_list = [dict(r) for r in recs]
    if not recs_list:
        raise ValueError(f"No recommendations for run_id={run_id}")

    # All unique recommendation dates (one NIFTY price lookup per week).
    unique_dates = sorted({r["recommendation_date"] for r in recs_list})

    report: dict[str, Any] = {
        "run_id":        run_id,
        "total_recs":    len(recs_list),
        "unique_weeks":  len(unique_dates),
        "windows":       {},
    }

    with repo.connect() as conn:
        for label, weeks in _WINDOWS.items():
            col = f"forward_return_{label}"

            # Portfolio returns (all scored recs in this window)
            port_returns = [
                float(r[col]) for r in recs_list
                if r.get(col) is not None
            ]
            if not port_returns:
                report["windows"][label] = {"scored_recs": 0, "note": "no forward prices yet"}
                continue

            # NIFTY returns for the same weeks that have stock data
            scored_dates = sorted({
                r["recommendation_date"] for r in recs_list if r.get(col) is not None
            })
            nifty_returns_map: dict[str, float | None] = {}
            for d in scored_dates:
                entry = _nifty_price(conn, d, 0)
                fwd   = _nifty_price(conn, d, weeks)
                nifty_returns_map[d] = _pct_return(entry, fwd)

            # Paired alpha: stock_return - nifty_return (same week)
            alphas: list[float] = []
            for r in recs_list:
                if r.get(col) is None:
                    continue
                nifty_ret = nifty_returns_map.get(r["recommendation_date"])
                if nifty_ret is None:
                    continue
                alphas.append(float(r[col]) - nifty_ret)

            nifty_returns = [v for v in nifty_returns_map.values() if v is not None]

            ttest = _t_test_one_sample(alphas) if alphas else {}

            avg_port   = round(sum(port_returns) / len(port_returns), 4)
            avg_nifty  = round(sum(nifty_returns) / len(nifty_returns), 4) if nifty_returns else None
            avg_alpha  = round(sum(alphas) / len(alphas), 4) if alphas else None
            hit_rate   = (
                round(sum(1 for a in alphas if a > 2.0) / len(alphas), 4)
                if alphas else None
            )

            p = ttest.get("p_value")
            significant = p is not None and p < 0.05

            report["windows"][label] = {
                "scored_recs":       len(port_returns),
                "avg_portfolio_ret": avg_port,
                "avg_nifty_ret":     avg_nifty,
                "avg_alpha_pp":      avg_alpha,
                "hit_rate":          hit_rate,
                "t_stat":            ttest.get("t_stat"),
                "p_value":           p,
                "alpha_significant": significant,
                "interpretation": (
                    f"alpha {avg_alpha:+.2f} pp is statistically significant "
                    f"(p={p:.3f} < 0.05) — the system has detectable edge vs NIFTY."
                    if significant else
                    f"alpha {avg_alpha:+.2f} pp is NOT statistically significant "
                    f"(p={p:.3f}) — cannot distinguish from noise at this sample size."
                ) if avg_alpha is not None and p is not None else "insufficient data",
            }

    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="NIFTY baseline comparison")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    load_app_env()
    config = AppConfig()
    repo   = PlatformRepository(db_path=config.db_path, neon_database_url=config.neon_database_url)
    repo.initialize()

    result = nifty_baseline(repo, args.run_id)

    print(json.dumps({k: v for k, v in result.items() if k != "windows"}, indent=2))
    print("\nPer-window comparison (portfolio vs NIFTY buy-and-hold):")
    print(f"{'Window':<6}  {'Recs':>5}  {'Port%':>6}  {'NIFTY%':>7}  {'Alpha':>6}  {'HitRate':>8}  {'t':>6}  {'p':>6}  significant?")
    print("-" * 90)
    for label, w in result["windows"].items():
        if "note" in w:
            print(f"{label:<6}  {'—':>5}  {w['note']}")
            continue
        print(
            f"{label:<6}  {w['scored_recs']:>5}  "
            f"{w['avg_portfolio_ret']:>+6.2f}  "
            f"{w['avg_nifty_ret']:>+7.2f}  "
            f"{w['avg_alpha_pp']:>+6.2f}  "
            f"{w['hit_rate']:>8.1%}  "
            f"{w['t_stat'] or 0:>+6.3f}  "
            f"{w['p_value'] or 1:>6.3f}  "
            f"{'YES' if w['alpha_significant'] else 'no'}"
        )
        print(f"         {w['interpretation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
