"""Start, resume, and inspect resumable weekly research runs."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from wealth.db.models import PriceEod, Security
from wealth.db.session import make_engine, system_session
from wealth.research.dossier import Dossier
from wealth.research.graph import build_research_graph
from wealth.research.router import BudgetExceeded, ModelRouter, RunBudget, RunContext
from wealth.research.rules import ResearchRules
from wealth.research.screener import run_screener
from wealth.research.store import (
    create_run,
    get_open_run_for_week,
    get_run,
    update_run,
)
from wealth.research.tools import ToolRegistry

_WEEK_PATTERN = re.compile(r"^(\d{4})-W(\d{2})$")
_MAX_RUN_CALLS = 10_000


def _valid_week(week: str) -> bool:
    match = _WEEK_PATTERN.fullmatch(week)
    if match is None:
        return False
    year, number = (int(value) for value in match.groups())
    try:
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


def load_shortlist_candidates(
    engine: Any,
    week: str,
    security_ids: Sequence[str],
    *,
    maximum: int,
) -> list[dict[str, Any]]:
    """Load persisted screener rows in rank order and attach their source snapshots."""
    candidates: list[dict[str, Any]] = []
    as_of = date.fromisocalendar(int(week[:4]), int(week[-2:]), 5)
    with system_session(engine) as session:
        for raw_security_id in security_ids[:maximum]:
            try:
                security_id = UUID(str(raw_security_id))
            except ValueError:
                continue
            security = session.scalar(select(Security).where(Security.id == security_id))
            if security is None or not security.symbol or security.market not in {"IN", "US"}:
                continue
            dossier_id = f"{week}/{security.market}/{security.symbol}"
            raw_dossier = session.scalar(
                text(
                    "SELECT json FROM dossiers "
                    "WHERE id = :dossier_id AND week = :week AND stage = 'SCREENED'"
                ),
                {"dossier_id": dossier_id, "week": week},
            )
            if raw_dossier is None:
                continue
            dossier = Dossier.model_validate(raw_dossier)
            evidence = next(
                (item for claim in dossier.claims for item in claim.evidence if item.tool_call_id),
                None,
            )
            if evidence is None:
                continue
            try:
                call_uuid = UUID(evidence.tool_call_id)
            except ValueError:
                continue
            snapshot_row = session.execute(
                text(
                    "SELECT result_json, source_url, retrieved_at "
                    "FROM tool_calls WHERE tool_call_id = :tool_call_id"
                ),
                {"tool_call_id": call_uuid},
            ).one_or_none()
            if snapshot_row is None:
                continue
            snapshot = snapshot_row.result_json
            if isinstance(snapshot, str):
                snapshot = json.loads(snapshot)
            if not isinstance(snapshot, dict):
                continue
            price = snapshot.get("price_used_for_value")
            if price is None:
                price = session.scalar(
                    select(PriceEod.close)
                    .where(
                        PriceEod.security_id == security_id,
                        PriceEod.as_of <= as_of,
                    )
                    .order_by(PriceEod.as_of.desc())
                    .limit(1)
                )
            if price is None or Decimal(str(price)) <= 0:
                continue
            tool_id = evidence.tool_call_id
            candidates.append(
                {
                    "dossier_id": dossier_id,
                    "security_id": str(security.id),
                    "symbol": security.symbol,
                    "name": security.name,
                    "market": security.market,
                    "sector": str(snapshot.get("sector") or security.sector),
                    "as_of": str(snapshot.get("as_of") or as_of.isoformat()),
                    "price": format(Decimal(str(price)), "f"),
                    "quant_decile": int(snapshot.get("decile", 0)),
                    "composite": str(snapshot.get("composite", "0")),
                    "dossier": dossier.model_dump(mode="json", by_alias=True),
                    "tool_results": {
                        tool_id: {
                            "tool_call_id": tool_id,
                            "result": snapshot,
                            "source_url": snapshot_row.source_url,
                            "retrieved_at": snapshot_row.retrieved_at.isoformat(),
                        }
                    },
                }
            )
    return candidates


def _graph_run(
    *,
    engine: Any,
    checkpointer: Any,
    run_id: UUID,
    record: dict[str, Any],
    initial: bool,
    shortlist: list[dict[str, Any]] | None = None,
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
    registry = ToolRegistry(owner_engine=engine)
    try:
        graph = build_research_graph(
            ModelRouter(owner_engine=engine),
            run_ctx=run_ctx,
            owner_engine=engine,
            tools=registry,
            shortlist=shortlist,
            rules=ResearchRules.load(),
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": str(run_id)}}
        if initial:
            state = graph.invoke(
                {
                    "run_id": str(run_id),
                    "week": str(record["week"]),
                    "shortlist": shortlist or [],
                    "shortlist_ids": [],
                    "dossiers": [],
                    "tool_results_by_dossier": {},
                    "budget": {
                        "max_usd": format(budget.max_usd, "f"),
                        "spent_usd": format(budget.spent_usd, "f"),
                        "max_calls": budget.max_calls,
                        "calls": budget.calls,
                    },
                    "stage": "QUEUED",
                    "status": "running",
                },
                config,
            )
        else:
            state = graph.invoke(None, config)
        return state, budget.calls
    finally:
        registry.close()


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
        existing = get_open_run_for_week(engine, args.week)
        if existing is not None:
            _json_print(
                {
                    "run_id": str(existing["run_id"]),
                    "stage": existing["stage"],
                    "status": existing["status"],
                }
            )
            return 0
        create_run(engine, run_id=run_id, week=args.week, mode=args.mode, budget_usd=budget_usd)
        screened = run_screener(engine, args.week)
        rules = ResearchRules.load()
        shortlist = load_shortlist_candidates(
            engine,
            args.week,
            screened["security_ids"],
            maximum=rules.max_dossiers_per_run,
        )
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
                shortlist=shortlist,
            )
        update_run(engine, run_id, stage=state["stage"], status="waiting")
        _json_print(
            {
                "run_id": str(run_id),
                "stage": state["stage"],
                "status": "waiting",
                "shortlisted": len(shortlist),
            }
        )
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


def _resolve_continue_run(engine: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.run_id:
        try:
            return get_run(engine, UUID(args.run_id))
        except ValueError:
            return None
    if not _valid_week(args.week):
        return None
    return get_open_run_for_week(engine, args.week)


def _run_continue(args: argparse.Namespace) -> int:
    if args.week and not _valid_week(args.week):
        _json_print({"status": "failed", "error": "invalid_week"})
        return 3
    try:
        database_url = _owner_url()
        engine = make_engine(database_url)
    except (ValueError, RuntimeError):
        _json_print({"status": "failed", "error": "database_unavailable"})
        return 3

    try:
        record = _resolve_continue_run(engine, args)
        if record is None:
            _json_print(
                {
                    "run_id": args.run_id,
                    "week": args.week,
                    "status": "no_open_run",
                }
            )
            return 0 if args.week else 3
        run_id = UUID(str(record["run_id"]))
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
        _json_print({"run_id": str(run_id), "stage": state["stage"], "status": status})
        return 0
    except BudgetExceeded:
        _json_print({"run_id": args.run_id, "status": "budget_exceeded"})
        return 2
    except Exception as error:
        _json_print({"run_id": args.run_id, "status": "failed", "error": type(error).__name__})
        return 3
    finally:
        engine.dispose()


def _run_status(args: argparse.Namespace) -> int:
    try:
        run_id = UUID(args.run_id)
        engine = make_engine(_owner_url())
    except (ValueError, RuntimeError):
        _json_print({"status": "failed", "error": "database_unavailable"})
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


def _default_budget() -> str:
    return os.getenv("RESEARCH_BUDGET_USD", "5")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m wealth.research.run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--week", required=True)
    start.add_argument("--mode", choices=("weekly",), default="weekly")
    start.add_argument("--budget-usd", default=_default_budget())
    start.set_defaults(handler=_run_start)
    continue_run = subparsers.add_parser("continue")
    selector = continue_run.add_mutually_exclusive_group(required=True)
    selector.add_argument("--run-id")
    selector.add_argument("--week")
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
