"""
run_backtest.py — CLI entry point for the backtest harness.

Usage:
    python -m backtest.run_backtest --mode full \\
        --start-date 2024-05-09 --end-date 2026-05-09

Modes:
    snapshot  — fetch and store historical prices + fundamentals
    replay    — replay FLOW 2 against snapshots, write backtest_recommendations
    score     — compute hit rates from an existing replay run
    full      — snapshot → replay → score in sequence

Exit codes:
    0 — success (or hit rate above floor)
    1 — hit rate below floor (configurable via --hit-rate-floor)
    2 — fatal error
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_platform.config import AppConfig, load_app_env
from stock_platform.data.repository import PlatformRepository

from backtest.snapshot import run_snapshot
from backtest.replay import replay
from backtest.scorer import score_run

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
_log = logging.getLogger("backtest.run_backtest")

_DEFAULT_HIT_RATE_FLOOR = 0.45


def _emit(event: str, **kwargs) -> None:
    """Emit a structured JSON log line (GitHub Actions compatible)."""
    print(json.dumps({"event": event, **kwargs}), flush=True)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stock platform backtester")
    parser.add_argument(
        "--mode", choices=["snapshot", "replay", "score", "full"], default="full",
        help="Which phase(s) to run",
    )
    parser.add_argument("--start-date", type=_parse_date, required=False,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date",   type=_parse_date, required=False,
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--run-id",     default=None,
                        help="Existing replay run_id (score mode only)")
    parser.add_argument("--top-n",      type=int, default=5,
                        help="Recommendations per replay step")
    parser.add_argument("--hit-rate-floor", type=float, default=_DEFAULT_HIT_RATE_FLOOR,
                        help="Minimum acceptable 6m hit rate before exit code 1")
    parser.add_argument("--weights-version", default=None,
                        help="Override quality_weights.yaml version label")
    args = parser.parse_args(argv)

    # ── environment + repo setup ──────────────────────────────────────────────
    try:
        load_app_env()
        config = AppConfig()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        repo = PlatformRepository(
            db_path=config.db_path,
            neon_database_url=config.neon_database_url,
        )
        repo.initialize()
    except Exception as exc:
        _emit("INIT_ERROR", error=str(exc))
        return 2

    start = args.start_date or date(date.today().year - 2, date.today().month, date.today().day)
    end   = args.end_date   or date.today()

    _emit("START", mode=args.mode, start_date=start.isoformat(), end_date=end.isoformat())

    run_id = args.run_id

    # ── snapshot ──────────────────────────────────────────────────────────────
    if args.mode in ("snapshot", "full"):
        try:
            counts = run_snapshot(repo)
            _emit("SNAPSHOT_DONE", **counts)
        except Exception as exc:
            _emit("SNAPSHOT_ERROR", error=str(exc))
            return 2

    # ── replay ────────────────────────────────────────────────────────────────
    if args.mode in ("replay", "full"):
        try:
            run_id = replay(
                repo=repo,
                config=config,
                start_date=start,
                end_date=end,
                top_n=args.top_n,
            )
            _emit("REPLAY_DONE", run_id=run_id)
        except Exception as exc:
            _emit("REPLAY_ERROR", error=str(exc))
            return 2

    # ── score ─────────────────────────────────────────────────────────────────
    if args.mode in ("score", "full"):
        if not run_id:
            _emit("SCORE_ERROR", error="--run-id required in score mode")
            return 2
        try:
            summary = score_run(repo=repo, run_id=run_id)
            _emit("SCORE_DONE", **{k: v for k, v in summary.items() if k != "by_confidence"})
            _emit("SCORE_BY_CONFIDENCE", by_confidence=summary.get("by_confidence", {}))
        except Exception as exc:
            _emit("SCORE_ERROR", error=str(exc))
            return 2

        # Exit non-zero if hit rate is below floor.
        hit_rate_6m = summary.get("hit_rate_6m")
        if hit_rate_6m is not None and hit_rate_6m < args.hit_rate_floor:
            _emit(
                "HIT_RATE_REGRESSION",
                hit_rate_6m=hit_rate_6m,
                floor=args.hit_rate_floor,
                message=f"Hit rate {hit_rate_6m:.1%} < floor {args.hit_rate_floor:.1%}",
            )
            return 1

    _emit("DONE", run_id=run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
