"""
calibrate.py — walk-forward weight calibration for quality scoring.

Algorithm:
1. Load backtest_recommendations for a completed run.
2. Split into train / validate windows by unique week dates.
   Default: first 36 weeks train, last 16 validate (user-configurable).
3. For each (symbol, date) in recs, fetch raw fundamentals from
   historical_fundamentals and compute per-field component scores using
   the same step functions as quant_model.compute_quality_score.
4. Grid-search over the 5 quality-score weights in 0.05 increments,
   constrained to sum to 1.0 (~1,820 valid combinations).
   For each combination, re-rank candidates WITHIN each week, take top_n,
   compute hit rate vs NIFTY benchmark.
5. Report top-10 by validated hit rate; flag overfit (val < train - 5 pp).
6. Write best validated config to rules/quality_weights.yaml.

Run:
    python -m backtest.calibrate --run-id <run_id>
    python -m backtest.calibrate --run-id <run_id> --train-weeks 36 --val-weeks 16
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

_WEIGHTS_FILE        = _ROOT / "rules" / "quality_weights.yaml"
_HIT_ALPHA_THRESHOLD = 2.0   # stock must beat NIFTY by 2 pp to count as hit
_OVERFIT_GAP         = 0.05  # flag overfit when val drops more than 5 pp below train
_STEP                = 5     # weight units: each unit = 0.05; total = 20 units = 1.0


# ── Component score step functions (mirror quant_model exactly) ───────────────

def _component_scores(fund: dict[str, Any]) -> dict[str, float | None]:
    """Map raw fundamental values to their per-field step scores (before weighting)."""
    roce  = fund.get("roce")
    eps   = fund.get("eps")
    rev   = fund.get("revenue_growth")
    promo = fund.get("promoter_holding")
    de    = fund.get("debt_equity")

    return {
        "roce": (
            None if roce is None else
            1.0  if roce > 18 else
            0.6  if roce > 10 else
            0.0  if roce > 0  else -0.5
        ),
        "eps": (
            None if eps is None else
            1.0  if eps > 0 else 0.0
        ),
        "revenue_growth": (
            None if rev is None else
            1.0  if rev > 15 else
            0.7  if rev > 8  else
            0.3  if rev > 0  else 0.0
        ),
        "promoter": (
            None if promo is None else
            1.0  if promo > 50 else
            0.7  if promo > 35 else 0.3
        ),
        "debt_equity": (
            None if de is None else
            1.0  if de < 0.5 else
            0.5  if de < 1.0 else
            0.1  if de < 2.0 else 0.0
        ),
    }


def _recompute_score(
    components: dict[str, float | None],
    weights: tuple[float, ...],
) -> float:
    """Weighted sum of component scores using given weights; absent fields are skipped."""
    w_roce, w_eps, w_rev, w_prom, w_de = weights
    field_weights = [
        ("roce",           w_roce),
        ("eps",            w_eps),
        ("revenue_growth", w_rev),
        ("promoter",       w_prom),
        ("debt_equity",    w_de),
    ]
    total_s = total_w = 0.0
    for field, w in field_weights:
        s = components.get(field)
        if s is not None and w > 0:
            total_s += s * w
            total_w += w
    if total_w == 0.0:
        return 0.5
    return round(max(0.0, min(1.0, total_s / total_w)), 4)


# ── Batch fundamentals loader ─────────────────────────────────────────────────

def _load_fundamentals_batch(
    conn: Any,
    recs: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float | None]]:
    """
    For each (symbol, recommendation_date) pair, fetch the closest
    historical_fundamentals snapshot at or before that date.
    Returns a dict keyed by (symbol, date_str).
    """
    pairs  = {(r["symbol"], r["recommendation_date"]) for r in recs}
    result: dict[tuple[str, str], dict[str, float | None]] = {}
    for symbol, rec_date in pairs:
        row = conn.execute(
            """
            SELECT roce, eps, debt_equity, revenue_growth, promoter_holding
            FROM historical_fundamentals
            WHERE symbol = ? AND snapshot_date <= ?
            ORDER BY snapshot_date DESC LIMIT 1
            """,
            (symbol, rec_date),
        ).fetchone()
        result[(symbol, rec_date)] = dict(row) if row else {}
    return result


def _load_nifty_prices(
    conn: Any,
    dates: list[str],
    weeks_ahead: int,
) -> dict[str, float | None]:
    """For each date, fetch NIFTY price weeks_ahead weeks later (for alpha computation)."""
    result: dict[str, float | None] = {}
    for d in dates:
        target = (date.fromisoformat(d) + timedelta(weeks=weeks_ahead)).isoformat()
        row = conn.execute(
            "SELECT close_price FROM historical_prices WHERE symbol = 'NIFTY' AND date >= ? ORDER BY date ASC LIMIT 1",
            (target,),
        ).fetchone()
        entry_row = conn.execute(
            "SELECT close_price FROM historical_prices WHERE symbol = 'NIFTY' AND date <= ? ORDER BY date DESC LIMIT 1",
            (d,),
        ).fetchone()
        entry = float(entry_row["close_price"]) if entry_row else None
        fwd   = float(row["close_price"])       if row       else None
        result[d] = (
            round((fwd / entry - 1) * 100, 4)
            if (fwd and entry and entry > 0) else None
        )
    return result


# ── Weight grid ───────────────────────────────────────────────────────────────

def _weight_combinations() -> list[tuple[float, ...]]:
    """~1,820 5-tuples summing to 1.0 in steps of 0.05."""
    n      = 20
    combos = []
    for a, b, c, d in itertools.product(range(n + 1), repeat=4):
        e = n - a - b - c - d
        if 0 <= e <= n:
            combos.append((a / n, b / n, c / n, d / n, e / n))
    return combos


# ── Hit-rate computation with per-week re-ranking ─────────────────────────────

def _hit_rate_for_weights(
    weeks_to_recs: dict[str, list[dict[str, Any]]],
    components_map: dict[tuple[str, str], dict[str, float | None]],
    nifty_6m: dict[str, float | None],
    weights: tuple[float, ...],
    top_n: int = 5,
) -> float | None:
    """
    For each week:
      - Recompute quality score for each rec with given weights.
      - Re-rank within the week; take top_n.
      - A rec is a hit if its 6m return beats NIFTY by _HIT_ALPHA_THRESHOLD pp.
    Returns overall hit rate, or None if no scored recs in this split.
    """
    hit_count = total_with_outcome = 0

    for week_date, week_recs in weeks_to_recs.items():
        nifty_ret = nifty_6m.get(week_date)

        # Recompute score for each rec; sort desc; take top_n
        scored = []
        for r in week_recs:
            comp  = components_map.get((r["symbol"], r["recommendation_date"]), {})
            score = _recompute_score(comp, weights)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)

        for _, rec in scored[:top_n]:
            fwd = rec.get("forward_return_6m")
            if fwd is None or nifty_ret is None:
                continue
            alpha = fwd - nifty_ret
            total_with_outcome += 1
            if alpha > _HIT_ALPHA_THRESHOLD:
                hit_count += 1

    return round(hit_count / total_with_outcome, 4) if total_with_outcome else None


# ── Main calibration function ─────────────────────────────────────────────────

def calibrate(
    repo: PlatformRepository,
    run_id: str,
    train_weeks: int = 36,
    val_weeks: int   = 16,
    top_n: int       = 5,
) -> dict[str, Any]:
    """
    Walk-forward calibration on a completed backtest run.

    Splits unique recommendation dates into train / val windows.
    Falls back to a 70/30 date split when fewer than train_weeks + val_weeks
    scored weeks exist.
    """
    with repo.connect() as conn:
        run_row = conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not run_row:
            raise ValueError(f"run_id {run_id!r} not found in backtest_runs")

        all_recs = conn.execute(
            """
            SELECT * FROM backtest_recommendations
            WHERE run_id = ? ORDER BY recommendation_date
            """,
            (run_id,),
        ).fetchall()

    all_recs_list = [dict(r) for r in all_recs]
    if not all_recs_list:
        raise ValueError(f"No recommendations found for run_id={run_id}")

    # ── Unique week dates with at least one 6m-scored rec ────────────────────
    scored_weeks = sorted({
        r["recommendation_date"]
        for r in all_recs_list
        if r.get("forward_return_6m") is not None
    })

    if len(scored_weeks) < 4:
        raise RuntimeError(
            f"Only {len(scored_weeks)} weeks have 6m forward returns — "
            "need at least 4 to calibrate. Re-run after more price history accumulates."
        )

    # Honour requested split; fall back to 70/30 when not enough scored weeks.
    if len(scored_weeks) >= train_weeks + val_weeks:
        train_dates = set(scored_weeks[:train_weeks])
        val_dates   = set(scored_weeks[train_weeks: train_weeks + val_weeks])
    else:
        split_idx   = max(1, int(len(scored_weeks) * 0.7))
        train_dates = set(scored_weeks[:split_idx])
        val_dates   = set(scored_weeks[split_idx:])
        _log.warning(
            "calibrate: only %d scored weeks — using 70/30 split "
            "(%d train, %d val) instead of requested %d/%d",
            len(scored_weeks), len(train_dates), len(val_dates),
            train_weeks, val_weeks,
        )

    _log.info(
        "calibrate: run=%s  scored_weeks=%d  train=%d weeks  val=%d weeks",
        run_id, len(scored_weeks), len(train_dates), len(val_dates),
    )

    # Group recs by week for each split.
    def _group(dates_set: set[str]) -> dict[str, list[dict]]:
        weeks: dict[str, list[dict]] = {}
        for r in all_recs_list:
            d = r["recommendation_date"]
            if d in dates_set:
                weeks.setdefault(d, []).append(r)
        return weeks

    train_weeks_map = _group(train_dates)
    val_weeks_map   = _group(val_dates)

    # ── Batch-load fundamentals and NIFTY forward returns ────────────────────
    with repo.connect() as conn:
        all_scored_recs = [
            r for r in all_recs_list
            if r["recommendation_date"] in train_dates | val_dates
        ]
        components_map = {
            k: _component_scores(v)
            for k, v in _load_fundamentals_batch(conn, all_scored_recs).items()
        }
        all_dates  = list(train_dates | val_dates)
        nifty_6m   = _load_nifty_prices(conn, all_dates, weeks_ahead=26)

    # Log coverage
    present = sum(1 for v in components_map.values() if any(s is not None for s in v.values()))
    _log.info(
        "calibrate: %d/%d (symbol,date) pairs have at least one fundamentals field",
        present, len(components_map),
    )

    # ── Grid search ──────────────────────────────────────────────────────────
    combos  = _weight_combinations()
    results = []

    for weights in combos:
        train_hr = _hit_rate_for_weights(
            train_weeks_map, components_map, nifty_6m, weights, top_n
        )
        if train_hr is None:
            continue
        val_hr  = _hit_rate_for_weights(
            val_weeks_map, components_map, nifty_6m, weights, top_n
        )
        overfit = (
            val_hr is not None
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

    # Sort by validation hit rate when available; fall back to train.
    results_with_val = [r for r in results if r["validate_hit_rate_6m"] is not None]
    if results_with_val:
        results_with_val.sort(
            key=lambda r: (r["validate_hit_rate_6m"], r["train_hit_rate_6m"]),
            reverse=True,
        )
        non_overfit = [r for r in results_with_val if not r["overfit"]]
        best = (non_overfit or results_with_val)[0]
    else:
        # No validation data — sort by train, warn.
        _log.warning(
            "calibrate: no validation hit rates available "
            "(val weeks lack 6m forward returns). Selecting by train hit rate only — "
            "check for overfitting manually when more price history accumulates."
        )
        results.sort(key=lambda r: r["train_hit_rate_6m"], reverse=True)
        best = results[0]

    top10 = (results_with_val or results)[:10]

    _log.info(
        "calibrate: best train_hr=%.4f val_hr=%s overfit=%s  weights=%s",
        best["train_hit_rate_6m"], best["validate_hit_rate_6m"],
        best["overfit"], best["weights"],
    )

    # ── Write best config to rules/quality_weights.yaml ──────────────────────
    from datetime import UTC, datetime
    calibrated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    start = run_row["start_date"]
    end   = run_row["end_date"]
    train_end = sorted(train_dates)[-1]
    val_end   = sorted(val_dates)[-1] if val_dates else train_end

    w_hash = hashlib.md5(
        json.dumps(best["weights"], sort_keys=True).encode()
    ).hexdigest()[:8]

    config_out = {
        "version":              f"v{calibrated_at[:10].replace('-', '')}",
        "calibrated_at":        calibrated_at,
        "weights_hash":         w_hash,
        "train_period":         [start, train_end],
        "validate_period":      [train_end, val_end],
        "train_hit_rate_6m":    best["train_hit_rate_6m"],
        "validate_hit_rate_6m": best["validate_hit_rate_6m"],
        "weights":              best["weights"],
    }
    _WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _WEIGHTS_FILE.open("w", encoding="utf-8") as fh:
        yaml.dump(config_out, fh, default_flow_style=False, sort_keys=False)
    _log.info("Wrote calibrated weights → %s (version=%s)", _WEIGHTS_FILE, config_out["version"])

    return {
        "run_id":       run_id,
        "top10":        top10,
        "best":         best,
        "weights_file": str(_WEIGHTS_FILE),
        "config":       config_out,
        "scored_weeks": len(scored_weeks),
        "train_weeks":  len(train_dates),
        "val_weeks":    len(val_dates),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Walk-forward weight calibration")
    parser.add_argument("--run-id",      required=True)
    parser.add_argument("--train-weeks", type=int, default=36,
                        help="Weeks to use as training window (default 36)")
    parser.add_argument("--val-weeks",   type=int, default=16,
                        help="Weeks to use as validation window (default 16)")
    parser.add_argument("--top-n",       type=int, default=5,
                        help="Top-N recs per week to score (default 5)")
    args = parser.parse_args(argv)

    load_app_env()
    config = AppConfig()
    repo   = PlatformRepository(db_path=config.db_path, neon_database_url=config.neon_database_url)
    repo.initialize()

    result = calibrate(repo, args.run_id,
                       train_weeks=args.train_weeks,
                       val_weeks=args.val_weeks,
                       top_n=args.top_n)

    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("top10",)},
        indent=2, default=str,
    ))
    print(f"\nTop-10 weight configurations (sorted by val hit rate):")
    print(f"{'#':>3}  {'train':>6}  {'val':>6}  {'overfit':>7}  weights")
    print("-" * 80)
    for i, row in enumerate(result["top10"], 1):
        w = row["weights"]
        val_s = f"{row['validate_hit_rate_6m']:.4f}" if row["validate_hit_rate_6m"] is not None else "  N/A "
        print(
            f"{i:>3}.  {row['train_hit_rate_6m']:.4f}  {val_s}  "
            f"{'YES' if row['overfit'] else ' no':>7}  "
            f"roce={w['roce']:.2f} eps={w['eps']:.2f} "
            f"rev={w['revenue_growth']:.2f} promo={w['promoter']:.2f} "
            f"de={w['debt_equity']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
