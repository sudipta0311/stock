from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy import text
from tests.db.conftest import MigratedDatabase

from wealth.research.dossier import Dossier, Handoff, Stage
from wealth.research.router import ModelRouter, RunBudget, RunContext
from wealth.research.store import create_run, get_run, load_dossier, save_dossier

pytestmark = pytest.mark.db


class _Reply(BaseModel):
    value: str


def _completion() -> dict[str, Any]:
    return {
        "id": "synthetic-audit-call",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-6-luna",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"value":"ready"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def test_owner_store_round_trips_runs_and_dossiers(migrated_database: MigratedDatabase) -> None:
    run_id = uuid4()
    create_run(
        migrated_database.owner_engine,
        run_id=run_id,
        week="2026-W45",
        mode="weekly",
        budget_usd=Decimal("5"),
    )
    dossier = Dossier(
        dossier_id="2026-W45/IN/SYNTHETIC",
        security={"id": "IN:SYNTHETIC", "market": "IN", "type": "EQUITY"},
        stage=Stage.SCREENED,
        handoff=Handoff(
            from_agent="screen_agent",
            to_agent="analysis_agent",
            reason="Screen completed",
            budget_remaining_tokens=100,
        ),
    )
    save_dossier(migrated_database.owner_engine, dossier)

    stored_run = get_run(migrated_database.owner_engine, run_id)
    stored_dossier = load_dossier(migrated_database.owner_engine, dossier.dossier_id)

    assert stored_run is not None
    assert stored_run["status"] == "running"
    assert stored_run["budget_usd"] == Decimal("5.00000000")
    assert stored_dossier == dossier


def test_model_call_audit_never_stores_the_environment_key(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid4()
    create_run(
        migrated_database.owner_engine,
        run_id=run_id,
        week="2026-W45",
        mode="weekly",
        budget_usd=Decimal("1"),
    )
    monkeypatch.setenv("FOUNDRY_ENDPOINT", "https://foundry.example.test/openai/v1/")
    monkeypatch.setenv("FOUNDRY_API_KEY", "synthetic-test-key")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=_completion()))
    )
    router = ModelRouter(http_client=client, owner_engine=migrated_database.owner_engine)
    with client:
        result = router.complete(
            "nano",
            [{"role": "user", "content": "Review synthetic market data."}],
            schema=_Reply,
            max_output_tokens=32,
            run_ctx=RunContext(
                budget=RunBudget(max_usd=Decimal("1"), max_calls=10),
                run_id=run_id,
                owner_engine=migrated_database.owner_engine,
            ),
        )
    with migrated_database.owner_engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT request_json::text, response_json::text, usage::text "
                "FROM llm_calls WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        ).one()

    serialized = json.dumps(list(row))
    assert result.parsed == _Reply(value="ready")
    assert os.environ["FOUNDRY_API_KEY"] not in serialized
    assert "api-key" not in serialized
    assert "headers" not in serialized
