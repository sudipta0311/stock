"""Owner-only persistence helpers for research runs and dossiers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text

from wealth.db.session import system_session
from wealth.research.dossier import Dossier


def create_run(
    engine: Engine,
    *,
    run_id: UUID,
    week: str,
    mode: str,
    budget_usd: Decimal,
) -> None:
    with system_session(engine) as session:
        session.execute(
            text(
                "INSERT INTO research_runs "
                "(run_id, week, mode, stage, status, budget_usd, spent_usd) "
                "VALUES (:run_id, :week, :mode, 'QUEUED', 'running', :budget_usd, 0)"
            ),
            {"run_id": run_id, "week": week, "mode": mode, "budget_usd": budget_usd},
        )


def update_run(
    engine: Engine,
    run_id: UUID,
    *,
    stage: str | None = None,
    status: str | None = None,
    spent_usd: Decimal | None = None,
    finished: bool = False,
) -> None:
    updates: dict[str, Any] = {}
    assignments: list[str] = []
    for key, value in (("stage", stage), ("status", status), ("spent_usd", spent_usd)):
        if value is not None:
            assignments.append(f"{key} = :{key}")
            updates[key] = value
    if finished:
        assignments.append("finished_at = :finished_at")
        updates["finished_at"] = datetime.now(UTC)
    if not assignments:
        return
    updates["run_id"] = run_id
    with system_session(engine) as session:
        session.execute(
            text(f"UPDATE research_runs SET {', '.join(assignments)} WHERE run_id = :run_id"),
            updates,
        )


def get_run(engine: Engine, run_id: UUID) -> dict[str, Any] | None:
    with system_session(engine) as session:
        row = (
            session.execute(
                text(
                    "SELECT run_id, week, mode, stage, status, budget_usd, spent_usd, "
                    "started_at, finished_at, "
                    "(SELECT count(*) FROM llm_calls WHERE llm_calls.run_id = "
                    "research_runs.run_id) AS calls "
                    "FROM research_runs WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .one_or_none()
        )
    return dict(row) if row is not None else None


def save_dossier(engine: Engine, dossier: Dossier, *, security_id: UUID | None = None) -> None:
    payload = json.dumps(dossier.model_dump(mode="json", by_alias=True), ensure_ascii=False)
    with system_session(engine) as session:
        session.execute(
            text(
                "INSERT INTO dossiers (id, week, security_id, stage, json) "
                "VALUES (:id, :week, :security_id, :stage, CAST(:payload AS jsonb)) "
                "ON CONFLICT (id) DO UPDATE SET week = EXCLUDED.week, "
                "security_id = EXCLUDED.security_id, stage = EXCLUDED.stage, "
                "json = EXCLUDED.json, updated_at = now()"
            ),
            {
                "id": dossier.dossier_id,
                "week": dossier.dossier_id[:8],
                "security_id": security_id,
                "stage": dossier.stage.value,
                "payload": payload,
            },
        )


def load_dossier(engine: Engine, dossier_id: str) -> Dossier | None:
    with system_session(engine) as session:
        payload = session.scalar(
            text("SELECT json FROM dossiers WHERE id = :dossier_id"), {"dossier_id": dossier_id}
        )
    if payload is None:
        return None
    if isinstance(payload, str):
        return Dossier.model_validate_json(payload)
    return Dossier.model_validate(payload)
