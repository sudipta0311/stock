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
from wealth.research.dossier import Dossier, Stage
from wealth.research.graph import (
    DECIDED_DOSSIER_STAGES,
    build_research_graph,
    build_week_pack_content,
    keep_market_statuses,
    merge_week_decisions,
)
from wealth.research.router import BudgetExceeded, ModelRouter, RunBudget, RunContext
from wealth.research.rules import ResearchRules
from wealth.research.screener import run_screener
from wealth.research.store import (
    create_run,
    get_open_run_for_week,
    get_run,
    load_research_pack,
    save_research_pack,
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
    security_ids: Sequence[str] | None = None,
    *,
    maximum: int | None = None,
    stages: Sequence[Stage | str] = (Stage.SCREENED,),
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Load persisted dossiers and attach their stored screener snapshots."""
    candidates: list[dict[str, Any]] = []
    as_of = date.fromisocalendar(int(week[:4]), int(week[-2:]), 5)
    stage_values = list(dict.fromkeys(Stage(stage).value for stage in stages))
    if not stage_values:
        return candidates
    stage_params = {f"stage_{index}": stage for index, stage in enumerate(stage_values)}
    stage_clause = ", ".join(f":{name}" for name in stage_params)
    with system_session(engine) as session:
        dossier_rows: list[Any] = []
        if security_ids is None:
            dossier_rows = list(
                session.execute(
                    text(
                        "SELECT id, security_id, json FROM dossiers "
                        f"WHERE week = :week AND stage IN ({stage_clause}) ORDER BY id"
                    ),
                    {"week": week, **stage_params},
                ).all()
            )
        else:
            selected_ids = security_ids if maximum is None else security_ids[:maximum]
            for raw_security_id in selected_ids:
                try:
                    security_id = UUID(str(raw_security_id))
                except ValueError:
                    continue
                security = session.scalar(select(Security).where(Security.id == security_id))
                if security is None or not security.symbol or security.market not in {"IN", "US"}:
                    continue
                dossier_id = f"{week}/{security.market}/{security.symbol}"
                row = session.execute(
                    text(
                        "SELECT id, security_id, json FROM dossiers "
                        f"WHERE id = :dossier_id AND week = :week AND stage IN ({stage_clause})"
                    ),
                    {"dossier_id": dossier_id, "week": week, **stage_params},
                ).one_or_none()
                if row is not None:
                    dossier_rows.append(row)

        for dossier_row in dossier_rows:
            dossier_id = str(dossier_row.id)
            try:
                security_id = UUID(str(dossier_row.security_id))
            except (TypeError, ValueError):
                continue
            security = session.scalar(select(Security).where(Security.id == security_id))
            if security is None or not security.symbol or security.market not in {"IN", "US"}:
                continue
            raw_dossier = dossier_row.json
            dossier = Dossier.model_validate(raw_dossier)
            evidence = next(
                (item for claim in dossier.claims for item in claim.evidence if item.tool_call_id),
                None,
            )
            snapshot: dict[str, Any] = {}
            snapshot_row = None
            if evidence is not None:
                try:
                    call_uuid = UUID(evidence.tool_call_id)
                except ValueError:
                    call_uuid = None
                if call_uuid is not None:
                    snapshot_row = session.execute(
                        text(
                            "SELECT result_json, source_url, retrieved_at "
                            "FROM tool_calls WHERE tool_call_id = :tool_call_id"
                        ),
                        {"tool_call_id": call_uuid},
                    ).one_or_none()
                    if snapshot_row is not None:
                        raw_snapshot = snapshot_row.result_json
                        if isinstance(raw_snapshot, str):
                            raw_snapshot = json.loads(raw_snapshot)
                        if isinstance(raw_snapshot, dict):
                            snapshot = raw_snapshot
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
            try:
                parsed_price = Decimal(str(price)) if price is not None else None
            except (ArithmeticError, TypeError, ValueError):
                parsed_price = None
            if parsed_price is not None and (not parsed_price.is_finite() or parsed_price <= 0):
                parsed_price = None
            tool_results: dict[str, dict[str, Any]] = {}
            if evidence is not None and snapshot_row is not None:
                tool_id = evidence.tool_call_id
                tool_results[tool_id] = {
                    "tool_call_id": tool_id,
                    "result": snapshot,
                    "source_url": snapshot_row.source_url,
                    "retrieved_at": snapshot_row.retrieved_at.isoformat(),
                }
            try:
                quant_decile = int(snapshot["decile"]) if "decile" in snapshot else None
            except (TypeError, ValueError):
                quant_decile = None
            if not allow_incomplete and (
                evidence is None
                or snapshot_row is None
                or not snapshot
                or parsed_price is None
                or quant_decile is None
            ):
                continue
            candidates.append(
                {
                    "dossier_id": dossier_id,
                    "security_id": str(security.id),
                    "symbol": security.symbol,
                    "name": security.name,
                    "market": security.market,
                    "sector": str(snapshot.get("sector") or security.sector),
                    "as_of": str(snapshot.get("as_of") or as_of.isoformat()),
                    "price": format(parsed_price, "f") if parsed_price is not None else None,
                    "quant_decile": quant_decile,
                    "composite": str(snapshot.get("composite", "0")),
                    "dossier": dossier.model_dump(mode="json", by_alias=True),
                    "tool_results": tool_results,
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


def _run_repack(args: argparse.Namespace) -> int:
    if not _valid_week(args.week):
        _json_print({"status": "failed", "error": "invalid_week"})
        return 3
    try:
        engine = make_engine(_owner_url())
    except (ValueError, RuntimeError):
        _json_print({"status": "failed", "error": "database_unavailable"})
        return 3

    try:
        previous = load_research_pack(engine, args.week)
        stored_candidates = load_shortlist_candidates(
            engine,
            args.week,
            stages=DECIDED_DOSSIER_STAGES,
            allow_incomplete=True,
        )
        has_approved = any(
            Dossier.model_validate(candidate["dossier"]).stage == Stage.APPROVED
            for candidate in stored_candidates
        )
        if previous is None and not has_approved:
            _json_print(
                {
                    "status": "failed",
                    "error": "empty_week",
                    "message": f"No pack or approved dossiers were found for {args.week}.",
                }
            )
            return 3

        dossiers, candidates = merge_week_decisions([], {}, stored_candidates)
        as_of = date.fromisocalendar(int(args.week[:4]), int(args.week[-2:]), 5)
        from wealth.research.agents.macro_geo import MacroGeoToolbox

        index_percentiles, index_history = MacroGeoToolbox(engine, as_of).policy_index_percentiles()
        content = build_week_pack_content(
            dossiers,
            candidates,
            index_percentiles=index_percentiles,
            index_history=index_history,
        )
        pack = {
            **(previous or {}),
            **content,
            "index_valuation_percentiles": index_percentiles,
        }
        pack.setdefault("summary", "No ideas passed the checks this week.")
        pack.setdefault("run_id", "")
        pack.setdefault("cio_agenda", {})
        pack.setdefault("macro", {})
        pack = keep_market_statuses(pack, previous or {})
        save_research_pack(engine, args.week, pack)
        _json_print(
            {
                "week": args.week,
                "idea_count": {
                    market: content["markets"][market]["idea_count"] for market in ("IN", "US")
                },
            }
        )
        return 0
    except Exception as error:
        _json_print({"status": "failed", "error": type(error).__name__})
        return 3
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
    repack = subparsers.add_parser("repack")
    repack.add_argument("--week", required=True)
    repack.set_defaults(handler=_run_repack)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
