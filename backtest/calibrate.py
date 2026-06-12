"""
calibrate.py — walk-forward calibration for composite pillar weights (w_q, w_v, w_m).

Algorithm
---------
1. Load backtest_recommendations for a completed run that has composite pillar
   percentiles (quality_pct, valuation_pct, momentum_pct) and alpha_6m populated.
   These columns are written by replay.py (Task 1) and scorer.py (Task 2).
2. Walk-forward expanding-window folds:
       fold k : train = weeks[0 : 26 + 13k]   (min 26-week train)
                val   = weeks[26 + 13k : 26 + 13(k+1)]
   Requires >= 3 folds; aborts with CALIBRATION_INSUFFICIENT_HISTORY otherwise.
3. For each of the 66 (w_q, w_v, w_m) combinations summing to 1.0 in 0.1 steps:
       for each fold:
           composite_i = w_q * quality_pct_i + w_v * valuation_pct_i + w_m * momentum_pct_i
           IC_fold = mean Spearman rho(composite, alpha_6m) over val weeks
       mean_val_IC = mean of IC_fold across folds
4. Selection: highest mean_val_IC among combos where IC_fold > 0 in EVERY fold.
   If no combo qualifies, keep current weights and log CALIBRATION_NO_ROBUST_WINNER.
5. Write best weights to rules/composite_weights.yaml.

Run:
    python -m backtest.calibrate --run-id <run_id>
    python -m backtest.calibrate --run-id <run_id> --min-train 26 --val-step 13
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

from stock_platform.config import AppConfig, load_app_env
from stock_platform.data.repository import PlatformRepository

_log = logging.getLogger(__name__)

_COMPOSITE_WEIGHTS_FILE = _ROOT / "rules" / "composite_weights.yaml"
_MIN_FOLDS              = 3    # abort if fewer than this many valid folds
_MIN_IC_PAIRS           = 3    # minimum pairs for a Spearman IC to be computed


# ── Pure-Python Spearman IC (no scipy dependency) ────────────────────────────

def _spearman_ic(scores: list[float | None], alphas: list[float | None]) -> float | None:
    pairs = [(s, a) for s, a in zip(scores, alphas) if s is not None and a is not None]
    n = len(pairs)
    if n < _MIN_IC_PAIRS:
        return None

    def _rank(vals: list[float]) -> list[float]:
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

    sv = [p[0] for p in pairs]
    av = [p[1] for p in pairs]
    rs = _rank(sv)
    ra = _rank(av)
    m_rs = sum(rs) / n
    m_ra = sum(ra) / n
    cov  = sum((rs[i] - m_rs) * (ra[i] - m_ra) for i in range(n))
    vs   = sum((r - m_rs) ** 2 for r in rs)
    va   = sum((r - m_ra) ** 2 for r in ra)
    denom = (vs * va) ** 0.5
    return round(cov / denom, 4) if denom > 0 else None


def _mean_weekly_ic(
    week_rows: dict[str, list[dict[str, Any]]],
    w_q: float, w_v: float, w_m: float,
) -> float | None:
    """Compute mean Spearman IC across weeks for given pillar weights."""
    ic_vals: list[float] = []
    for week_recs in week_rows.values():
        scores = [
            w_q * (r["quality_pct"] or 0.5) + w_v * (r["valuation_pct"] or 0.5) + w_m * (r["momentum_pct"] or 0.5)
            for r in week_recs
            if r.get("quality_pct") is not None
        ]
        alphas = [
            r["alpha_6m"]
            for r in week_recs
            if r.get("quality_pct") is not None
        ]
        ic = _spearman_ic(scores, alphas)
        if ic is not None:
            ic_vals.append(ic)
    if not ic_vals:
        return None
    return round(sum(ic_vals) / len(ic_vals), 4)


# ── Weight grid ───────────────────────────────────────────────────────────────

def _weight_grid(step: int = 10) -> list[tuple[float, float, float]]:
    """
    All (w_q, w_v, w_m) triples where each is a multiple of 1/step and sum = 1.0.
    Default step=10 → 0.1 increments → 66 combinations.
    """
    n = step
    combos: list[tuple[float, float, float]] = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            combos.append((a / n, b / n, c / n))
    return combos


# ── Walk-forward fold builder ─────────────────────────────────────────────────

def _build_folds(
    scored_weeks: list[str],
    min_train: int = 26,
    val_step: int  = 13,
) -> list[tuple[set[str], set[str]]]:
    """
    Expanding-window folds.
    fold k: train = scored_weeks[0 : min_train + k*val_step]
             val   = scored_weeks[min_train + k*val_step : min_train + (k+1)*val_step]
    """
    n      = len(scored_weeks)
    folds: list[tuple[set[str], set[str]]] = []
    k = 0
    while True:
        train_end = min_train + k * val_step
        val_end   = train_end + val_step
        if val_end > n:
            break
        folds.append((
            set(scored_weeks[:train_end]),
            set(scored_weeks[train_end:val_end]),
        ))
        k += 1
    return folds


# ── Main calibration function ─────────────────────────────────────────────────

def calibrate(
    repo: PlatformRepository,
    run_id: str,
    min_train: int = 26,
    val_step: int  = 13,
) -> dict[str, Any]:
    """
    Walk-forward calibration of composite pillar weights for a completed replay run.

    Raises RuntimeError when:
    - run_id not found
    - fewer than _MIN_FOLDS folds available (history too short)
    - quality_pct / valuation_pct / momentum_pct columns not present
      (run predates Task 1 — re-run the backtest with the current code)
    """
    with repo.connect() as conn:
        run_row = conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not run_row:
            raise ValueError(f"run_id {run_id!r} not found in backtest_runs")

        all_recs = conn.execute(
            """
            SELECT symbol, recommendation_date,
                   quality_pct, valuation_pct, momentum_pct,
                   alpha_6m
            FROM   backtest_recommendations
            WHERE  run_id = ?
            ORDER  BY recommendation_date
            """,
            (run_id,),
        ).fetchall()

    all_recs_list = [dict(r) for r in all_recs]
    if not all_recs_list:
        raise ValueError(f"No recommendations found for run_id={run_id}")

    # Guard: require composite columns (Task 1 run)
    sample = all_recs_list[0]
    if sample.get("quality_pct") is None and sample.get("valuation_pct") is None:
        # Check whether the column is simply NULL or genuinely absent
        if "quality_pct" not in sample:
            raise RuntimeError(
                "backtest_recommendations lacks quality_pct / valuation_pct / momentum_pct. "
                "Re-run the backtest with the current code (Task 1 must have been applied)."
            )

    # Weeks that have at least one rec with alpha_6m available
    scored_weeks = sorted({
        r["recommendation_date"]
        for r in all_recs_list
        if r.get("alpha_6m") is not None
        and r.get("quality_pct") is not None
    })

    folds = _build_folds(scored_weeks, min_train=min_train, val_step=val_step)

    if len(folds) < _MIN_FOLDS:
        msg = (
            f"Only {len(folds)} walk-forward fold(s) available "
            f"(need {_MIN_FOLDS}). Have {len(scored_weeks)} scored weeks; "
            f"need at least {min_train + _MIN_FOLDS * val_step}. "
            "Accumulate more price history and re-run."
        )
        _log.error("CALIBRATION_INSUFFICIENT_HISTORY: %s", msg)
        raise RuntimeError(msg)

    _log.info(
        "calibrate: run=%s  scored_weeks=%d  folds=%d  "
        "(min_train=%d, val_step=%d)",
        run_id, len(scored_weeks), len(folds), min_train, val_step,
    )

    # Group recs by week
    by_week: dict[str, list[dict]] = defaultdict(list)
    for r in all_recs_list:
        by_week[r["recommendation_date"]].append(r)

    combos = _weight_grid()
    results: list[dict[str, Any]] = []

    for w_q, w_v, w_m in combos:
        fold_ics: list[float | None] = []

        for train_dates, val_dates in folds:
            val_weeks = {d: by_week[d] for d in val_dates if d in by_week}
            ic = _mean_weekly_ic(val_weeks, w_q, w_v, w_m)
            fold_ics.append(ic)

        valid_ics = [ic for ic in fold_ics if ic is not None]
        if not valid_ics:
            continue

        mean_val_ic   = round(sum(valid_ics) / len(valid_ics), 4)
        all_positive  = all(ic > 0 for ic in valid_ics)  # every fold must be > 0

        results.append({
            "w_q":         w_q,
            "w_v":         w_v,
            "w_m":         w_m,
            "mean_val_ic": mean_val_ic,
            "fold_ics":    fold_ics,
            "all_positive": all_positive,
        })

    if not results:
        raise RuntimeError("No valid weight combinations produced results")

    results.sort(key=lambda r: r["mean_val_ic"], reverse=True)

    # Best: highest mean_val_IC with positive IC in EVERY fold
    robust = [r for r in results if r["all_positive"]]

    if robust:
        best = robust[0]
        winner_source = "robust"
    else:
        _log.warning(
            "CALIBRATION_NO_ROBUST_WINNER: no combo has IC > 0 in every fold. "
            "Keeping current composite_weights.yaml unchanged."
        )
        return {
            "run_id":        run_id,
            "outcome":       "NO_ROBUST_WINNER",
            "best":          None,
            "scored_weeks":  len(scored_weeks),
            "folds":         len(folds),
            "top10":         results[:10],
        }

    # ── Write composite_weights.yaml ─────────────────────────────────────────
    calibrated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    w_hash = hashlib.md5(
        json.dumps({"w_q": best["w_q"], "w_v": best["w_v"], "w_m": best["w_m"]}).encode()
    ).hexdigest()[:8]

    config_out = {
        "version":      f"v{calibrated_at[:10].replace('-', '')}",
        "calibrated_at": calibrated_at,
        "weights_hash": w_hash,
        "weights": {
            "quality":   best["w_q"],
            "valuation": best["w_v"],
            "momentum":  best["w_m"],
        },
        "folds":        len(folds),
        "per_fold_ic":  best["fold_ics"],
        "objective":    best["mean_val_ic"],
        "winner_source": winner_source,
        "min_train_weeks": min_train,
        "val_step_weeks":  val_step,
    }
    _COMPOSITE_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _COMPOSITE_WEIGHTS_FILE.open("w", encoding="utf-8") as fh:
        yaml.dump(config_out, fh, default_flow_style=False, sort_keys=False)

    _log.info(
        "calibrate: best w_q=%.2f w_v=%.2f w_m=%.2f  mean_val_ic=%.4f  "
        "fold_ics=%s  → %s (version=%s)",
        best["w_q"], best["w_v"], best["w_m"], best["mean_val_ic"],
        best["fold_ics"], _COMPOSITE_WEIGHTS_FILE, config_out["version"],
    )

    return {
        "run_id":       run_id,
        "outcome":      "OK",
        "best":         best,
        "weights_file": str(_COMPOSITE_WEIGHTS_FILE),
        "config":       config_out,
        "scored_weeks": len(scored_weeks),
        "folds":        len(folds),
        "top10":        results[:10],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Walk-forward composite weight calibration")
    parser.add_argument("--run-id",     required=True,   help="backtest_runs.run_id to calibrate on")
    parser.add_argument("--min-train",  type=int, default=26,
                        help="Minimum training weeks per fold (default 26)")
    parser.add_argument("--val-step",   type=int, default=13,
                        help="Validation fold width in weeks (default 13)")
    args = parser.parse_args(argv)

    load_app_env()
    config = AppConfig()
    repo   = PlatformRepository(db_path=config.db_path, neon_database_url=config.neon_database_url)
    repo.initialize()

    result = calibrate(repo, args.run_id,
                       min_train=args.min_train,
                       val_step=args.val_step)

    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("top10",)},
        indent=2, default=str,
    ))
    if result.get("top10"):
        print("\nTop-10 combos (sorted by mean validation IC):")
        print(f"{'#':>3}  {'mean_val_ic':>12}  {'all_pos?':>8}  w_q   w_v   w_m")
        print("-" * 60)
        for i, row in enumerate(result["top10"], 1):
            print(
                f"{i:>3}.  {row['mean_val_ic']:>12.4f}  "
                f"{'YES' if row['all_positive'] else ' no':>8}  "
                f"{row['w_q']:.2f}  {row['w_v']:.2f}  {row['w_m']:.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
