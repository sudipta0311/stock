"""Start, resume, and inspect resumable research runs."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy.engine import make_url

from wealth.db.session import make_engine
from wealth.research.graph import build_research_graph
from wealth.research.router import BudgetExceeded, ModelRouter, RunBudget, RunContext
from wealth.research.store import create_run, get_run, update_run

_WEEK_PATTERN = re.compile(r"^(\d{4})-W(\d{2})$")
_MAX_RUN_CALLS = 10_000


def _valid_week(week: str) -> bool:
    match = _WEEK_PATTERN.fullmatch(week)
    if match is None:
        return False
    year, number = (int(value) for value in match.groups())
    try:
        from datetime import date

        date.fromisocalendar(year, number, 1)
    except ValueError:
        return False
    return True


def _owner_url() -> str:
    url = os.getenv("OWNER_DATABASE_URL")
    if not url:
        raise RuntimeError("OWNER_DATABASE_URL is required for research jobs")
    return url


def _checkpointer_url(database_url: str) -> str:
    """PostgresSaver owns its migrations; it receives a plain psycopg URL."""
    parsed = make_url(database_url)
    if parsed.drivername in {"postgres", "postgresql", "postgresql+psycopg"}:
        return parsed.set(drivername="postgresql").render_as_string(hide_password=False)
    raise ValueError("OWNER_DATABASE_URL must point to PostgreSQL")


def _json_print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str, sort_keys=True))


def _graph_run(
    *,
    engine: Any,
    checkpointer: Any,
    run_id: UUID,
    record: dict[str, Any],
    initial: bool,
) -> tuple[dict[str, Any], int]:
    budget = RunBudget(
        max_usd=Decimal(record["budget_usd"]),
        max_calls=_MAX_RUN_CALLS,
        spent_usd=Decimal(record["spent_usd"]),
        calls=int(record.get("calls", 0)),
    )

    def persist_budget(spent_usd: Decimal) -> None:
        update_run(engine, run_id, spent_usd=spent_usd)

    run_ctx = RunContext(
        budget=budget,
        run_id=run_id,
        mode=str(record["mode"]),
        owner_engine=engine,
        on_budget_change=persist_budget,
    )
    graph = build_research_graph(
        ModelRouter(owner_engine=engine), run_ctx=run_ctx, checkpointer=checkpointer
    )
    config = {"configurable": {"thread_id": str(run_id)}}
    if initial:
        state = graph.invoke(
            {
                "run_id": str(run_id),
                "dossier_ids": [],
                "budget": {
                    "max_usd": format(budget.max_usd, "f"),
                    "spent_usd": format(budget.spent_usd, "f"),
                    "max_calls": budget.max_calls,
                    "calls": budget.calls,
                },
                "stage": "QUEUED",
                "status": "running",
                "handback_count": 0,
            },
            config,
        )
    else:
        state = graph.invoke(None, config)
    return state, budget.calls


def _run_start(args: argparse.Namespace) -> int:
    if not _valid_week(args.week):
        _json_print({"status": "failed", "error": "invalid_week"})
        return 3
    try:
        budget_usd = Decimal(args.budget_usd)
    except InvalidOperation:
        _json_print({"status": "failed", "error": "invalid_budget"})
        return 3
    if not budget_usd.is_finite() or budget_usd < 0:
        _json_print({"status": "failed", "error": "invalid_budget"})
        return 3

    engine = None
    run_id = uuid4()
    try:
        database_url = _owner_url()
        engine = make_engine(database_url)
        create_run(engine, run_id=run_id, week=args.week, mode=args.mode, budget_usd=budget_usd)
        with PostgresSaver.from_conn_string(_checkpointer_url(database_url)) as checkpointer:
            checkpointer.setup()
            record = get_run(engine, run_id)
            if record is None:
                raise RuntimeError("Research run was not stored")
            state, _calls = _graph_run(
                engine=engine,
                checkpointer=checkpointer,
                run_id=run_id,
                record=record,
                initial=True,
            )
        update_run(engine, run_id, stage=state["stage"], status="waiting")
        _json_print({"run_id": str(run_id), "stage": state["stage"], "status": "waiting"})
        return 0
    except BudgetExceeded:
        if engine is not None:
            update_run(engine, run_id, status="budget_exceeded", finished=True)
        _json_print({"run_id": str(run_id), "status": "budget_exceeded"})
        return 2
    except Exception as error:
        if engine is not None:
            update_run(engine, run_id, status="failed", finished=True)
        _json_print({"run_id": str(run_id), "status": "failed", "error": type(error).__name__})
        return 3
    finally:
        if engine is not None:
            engine.dispose()


def _run_continue(args: argparse.Namespace) -> int:
    try:
        run_id = UUID(args.run_id)
    except ValueError:
        _json_print({"status": "failed", "error": "invalid_run_id"})
        return 3

    engine = None
    try:
        database_url = _owner_url()
        engine = make_engine(database_url)
        record = get_run(engine, run_id)
        if record is None:
            _json_print({"run_id": str(run_id), "status": "failed", "error": "run_not_found"})
            return 3
        with PostgresSaver.from_conn_string(_checkpointer_url(database_url)) as checkpointer:
            checkpointer.setup()
            state, _calls = _graph_run(
                engine=engine,
                checkpointer=checkpointer,
                run_id=run_id,
                record=record,
                initial=False,
            )
        status = "completed" if state.get("status") == "completed" else "waiting"
        update_run(
            engine,
            run_id,
            stage=state.get("stage"),
            status=status,
            finished=status == "completed",
        )
        _json_print({"run_id": str(run_id), "stage": state.get("stage"), "status": status})
        return 0
    except BudgetExceeded:
        if engine is not None:
            update_run(engine, run_id, status="budget_exceeded", finished=True)
        _json_print({"run_id": str(run_id), "status": "budget_exceeded"})
        return 2
    except Exception as error:
        if engine is not None:
            update_run(engine, run_id, status="failed", finished=True)
        _json_print({"run_id": str(run_id), "status": "failed", "error": type(error).__name__})
        return 3
    finally:
        if engine is not None:
            engine.dispose()


def _run_status(args: argparse.Namespace) -> int:
    try:
        run_id = UUID(args.run_id)
        engine = make_engine(_owner_url())
    except (ValueError, RuntimeError) as error:
        _json_print({"status": "failed", "error": type(error).__name__})
        return 3
    try:
        record = get_run(engine, run_id)
        if record is None:
            _json_print({"run_id": str(run_id), "status": "not_found"})
            return 3
        _json_print(record)
        return 0
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m wealth.research.run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--week", required=True)
    start.add_argument("--mode", choices=("weekly",), default="weekly")
    start.add_argument("--budget-usd", required=True)
    start.set_defaults(handler=_run_start)
    continue_run = subparsers.add_parser("continue")
    continue_run.add_argument("--run-id", required=True)
    continue_run.set_defaults(handler=_run_continue)
    status = subparsers.add_parser("status")
    status.add_argument("--run-id", required=True)
    status.set_defaults(handler=_run_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
