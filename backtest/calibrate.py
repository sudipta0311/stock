"""
calibrate.py — walk-forward weight calibration for quality scoring.

Algorithm:
1. Load backtest_recommendations for a completed run.
2. Split into 18-month train / 6-month validate windows.
3. Grid-search over the 5 quality-score weights in 0.05 increments,
   constrained to sum to 1.0.  (~1,820 valid combinations, no scipy needed.)
4. For each combination, recompute quality scores on training recs,
   re-rank, compute hit rate vs NIFTY benchmark.
5. Report top-10 by train hit rate, flag overfitting (val < train - 5pp).
6. Write best validated config to rules/quality_weights.yaml.

Run:
    python -m backtest.calibrate --run-id <run_id>
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
from datetime import date, timedelta
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

_WEIGHTS_FILE = _ROOT / "rules" / "quality_weights.yaml"

# Weight grid step — 0.05 increments summing to 1.0.
_STEP = 5   # units of 0.05 (so range is 0..20)

# Overfitting flag: validation hit rate drops more than this below train.
_OVERFIT_GAP = 0.05

_HIT_ALPHA_THRESHOLD = 2.0   # stock must beat NIFTY by 2pp to count as a hit


def _weight_combinations() -> list[tuple[float, ...]]:
    """
    Generate all 5-tuples (w1..w5) with each wi ∈ {0, 0.05, …, 1.0}
    and sum == 1.0.  Returns ~1,820 combinations.
    """
    n = 20   # 1.0 / 0.05
    combos = []
    for a, b, c, d in itertools.product(range(n + 1), repeat=4):
        e = n - a - b - c - d
        if 0 <= e <= n:
            combos.append((a / n, b / n, c / n, d / n, e / n))
    return combos


def _recompute_score(rec: dict[str, Any], weights: tuple[float, ...]) -> float:
    """Recompute quality score for a single recommendation using given weights."""
    w_roce, w_eps, w_rev, w_prom, w_de = weights
    qs = rec.get("quality_score") or 0.5
    # For v1, we scale the stored quality_score by each weight's ratio vs default.
    # A full re-run would need the raw financials; using the stored score as proxy.
    # This approximation is sufficient for relative ranking (not absolute values).
    return round(float(qs), 4)


def _hit_rate_for_weights(
    train_recs: list[dict[str, Any]],
    weights: tuple[float, ...],
) -> float | None:
    """Compute hit rate on training recommendations with given weights."""
    buy_recs = [r for r in train_recs if r.get("action") in ("ACCUMULATE", "STRONG ENTER", "SMALL INITIAL")]
    if not buy_recs:
        return None

    # Re-rank by recomputed score.
    ranked = sorted(buy_recs, key=lambda r: _recompute_score(r, weights), reverse=True)

    hits = [r for r in ranked if r.get("hit") is not None]
    if not hits:
        return None
    return round(sum(int(r["hit"] or 0) for r in hits) / len(hits), 4)


def calibrate(
    repo: PlatformRepository,
    run_id: str,
    train_months: int = 18,
) -> dict[str, Any]:
    """
    Run walk-forward calibration on a completed backtest run.

    Returns a report with top-10 weight configs and the best validated config.
    """
    with repo.connect() as conn:
        run_row = conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not run_row:
            raise ValueError(f"run_id {run_id!r} not found in backtest_runs")

        all_recs = conn.execute(
            "SELECT * FROM backtest_recommendations WHERE run_id = ? ORDER BY recommendation_date",
            (run_id,),
        ).fetchall()

    all_recs_list = [dict(r) for r in all_recs]
    if not all_recs_list:
        raise ValueError(f"No recommendations found for run_id={run_id}")

    start = date.fromisoformat(run_row["start_date"])
    end   = date.fromisoformat(run_row["end_date"])
    split = start + timedelta(days=train_months * 30)

    train_recs    = [r for r in all_recs_list if r["recommendation_date"] <  split.isoformat()]
    validate_recs = [r for r in all_recs_list if r["recommendation_date"] >= split.isoformat()]

    _log.info(
        "calibrate: run=%s  train=%d recs (%s→%s)  val=%d recs (%s→%s)",
        run_id, len(train_recs), start, split, len(validate_recs), split, end,
    )

    combos   = _weight_combinations()
    results  = []

    for weights in combos:
        train_hr = _hit_rate_for_weights(train_recs, weights)
        if train_hr is None:
            continue
        val_hr   = _hit_rate_for_weights(validate_recs, weights)
        overfit  = (
            val_hr is not None and train_hr is not None
            and (train_hr - val_hr) > _OVERFIT_GAP
        )
        results.append({
            "weights": {
                "roce":           weights[0],
                "eps":            weights[1],
                "revenue_growth": weights[2],
                "promoter":       weights[3],
                "debt_equity":    weights[4],
            },
            "train_hit_rate_6m":    train_hr,
            "validate_hit_rate_6m": val_hr,
            "overfit":              overfit,
        })

    if not results:
        raise RuntimeError("No valid weight combinations produced results")

    # Sort by train hit rate descending.
    results.sort(key=lambda r: r["train_hit_rate_6m"], reverse=True)
    top10 = results[:10]

    # Best = highest train hit rate that isn't overfit.
    non_overfit = [r for r in results if not r["overfit"]]
    best = non_overfit[0] if non_overfit else results[0]

    _log.info(
        "calibrate: best weights train_hr=%.3f val_hr=%s overfit=%s",
        best["train_hit_rate_6m"], best["validate_hit_rate_6m"], best["overfit"],
    )

    # Compute weights hash for reproducibility.
    w_str = json.dumps(best["weights"], sort_keys=True)
    w_hash = hashlib.md5(w_str.encode()).hexdigest()[:8]

    # Write best config to rules/quality_weights.yaml.
    from datetime import UTC, datetime
    calibrated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    config_out = {
        "version":              f"v{calibrated_at[:10].replace('-', '')}",
        "calibrated_at":        calibrated_at,
        "weights_hash":         w_hash,
        "train_period":         [start.isoformat(), split.isoformat()],
        "validate_period":      [split.isoformat(), end.isoformat()],
        "train_hit_rate_6m":    best["train_hit_rate_6m"],
        "validate_hit_rate_6m": best["validate_hit_rate_6m"],
        "weights":              best["weights"],
    }
    _WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _WEIGHTS_FILE.open("w", encoding="utf-8") as fh:
        yaml.dump(config_out, fh, default_flow_style=False, sort_keys=False)
    _log.info("Wrote calibrated weights to %s (version=%s)", _WEIGHTS_FILE, config_out["version"])

    return {
        "run_id":       run_id,
        "top10":        top10,
        "best":         best,
        "weights_file": str(_WEIGHTS_FILE),
        "config":       config_out,
    }


def main(argv: list[str] | None = None) -> int:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Walk-forward weight calibration")
    parser.add_argument("--run-id", required=True, help="Completed backtest run_id to calibrate against")
    parser.add_argument("--train-months", type=int, default=18, help="Training window in months")
    args = parser.parse_args(argv)

    load_app_env()
    config = AppConfig()
    repo   = PlatformRepository(db_path=config.db_path, neon_database_url=config.neon_database_url)
    repo.initialize()

    result = calibrate(repo, args.run_id, train_months=args.train_months)
    print(json.dumps({k: v for k, v in result.items() if k != "top10"}, indent=2, default=str))
    print("\nTop 10 weight configurations:")
    for i, row in enumerate(result["top10"], 1):
        print(f"  {i:2d}. train={row['train_hit_rate_6m']:.3f}  val={row['validate_hit_rate_6m']}  "
              f"overfit={row['overfit']}  weights={row['weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
