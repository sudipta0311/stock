from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import Security
from wealth.research import run
from wealth.research.dossier import Claim, Dossier, Evidence, Stage
from wealth.research.graph import (
    DECIDED_DOSSIER_STAGES,
    build_week_pack_content,
    merge_week_decisions,
)
from wealth.research.run import load_shortlist_candidates
from wealth.research.store import load_research_pack, save_dossier, save_research_pack

pytestmark = pytest.mark.db


def _persist_decision(
    database: MigratedDatabase,
    *,
    week: str,
    market: str,
    symbol: str,
    status: str,
    composite: str,
    price: str,
) -> Dossier:
    with Session(database.owner_engine) as session, session.begin():
        security = session.scalar(
            select(Security).where(Security.market == market, Security.symbol == symbol)
        )
        if security is None:
            security = Security(
                market=market,
                symbol=symbol,
                name=f"Synthetic {market} equity {symbol}",
                security_type="equity",
                currency="INR" if market == "IN" else "USD",
                sleeve="equity_india" if market == "IN" else "equity_global",
                sector="Technology",
                industry="Software",
            )
            session.add(security)
            session.flush()
        security_id = security.id

    stage = Stage(status)
    tool_call_id = uuid4()
    dossier_id = f"{week}/{market}/{symbol}"
    dossier = Dossier(
        dossier_id=dossier_id,
        security={"id": str(security_id), "market": market, "type": "EQUITY"},
        stage=stage,
        status=status,
        confidence=Decimal("0.9") if status == "APPROVED" else Decimal("0.2"),
        drivers=["A supported synthetic reason."],
        claims=[
            Claim(
                text="Synthetic screen evidence.",
                kind="FACT",
                evidence=[Evidence(tool_call_id=str(tool_call_id))],
            )
        ],
    )
    save_dossier(database.owner_engine, dossier, security_id=security_id)
    snapshot = {
        "as_of": "2026-10-30",
        "price_used_for_value": price,
        "sector": "Technology",
        "decile": 7,
        "composite": composite,
        "raw_factors": {"dist_from_200dma": "0.12"},
    }
    snapshot_json = json.dumps(snapshot)
    with database.owner_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tool_calls "
                "(tool_call_id, run_id, dossier_id, tool, args_json, result_json, "
                "result_sha256, source_url, retrieved_at, status, error_code) VALUES "
                "(:tool_call_id, NULL, :dossier_id, 'screener_snapshot', '{}'::jsonb, "
                "CAST(:result_json AS jsonb), :result_sha256, :source_url, :retrieved_at, "
                "'success', NULL)"
            ),
            {
                "tool_call_id": tool_call_id,
                "dossier_id": dossier_id,
                "result_json": snapshot_json,
                "result_sha256": hashlib.sha256(snapshot_json.encode()).hexdigest(),
                "source_url": "https://example.test/synthetic-snapshot",
                "retrieved_at": datetime.now(UTC),
            },
        )
    return dossier


def _stored_decisions(database: MigratedDatabase, week: str) -> list[dict]:
    return load_shortlist_candidates(
        database.owner_engine,
        week,
        stages=DECIDED_DOSSIER_STAGES,
        allow_incomplete=True,
    )


def test_week_pack_uses_latest_decisions_and_orders_every_market(
    migrated_database: MigratedDatabase,
) -> None:
    week = "2026-W44"
    suffix = uuid4().hex[:6].upper()
    old_in = _persist_decision(
        migrated_database,
        week=week,
        market="IN",
        symbol=f"OLD{suffix}",
        status="APPROVED",
        composite="0.60",
        price="123.45",
    )
    changed = _persist_decision(
        migrated_database,
        week=week,
        market="IN",
        symbol=f"REV{suffix}",
        status="APPROVED",
        composite="0.99",
        price="130.00",
    )
    old_us = _persist_decision(
        migrated_database,
        week=week,
        market="US",
        symbol=f"US{suffix}",
        status="APPROVED",
        composite="0.95",
        price="45.67",
    )

    stored_before_current_run = _stored_decisions(migrated_database, week)
    assert {candidate["dossier_id"] for candidate in stored_before_current_run} == {
        old_in.dossier_id,
        changed.dossier_id,
        old_us.dossier_id,
    }

    new_in = _persist_decision(
        migrated_database,
        week=week,
        market="IN",
        symbol=f"NEW{suffix}",
        status="APPROVED",
        composite="0.75",
        price="125.00",
    )
    vetoed = _persist_decision(
        migrated_database,
        week=week,
        market="IN",
        symbol=changed.dossier_id.rsplit("/", 1)[-1],
        status="VETOED",
        composite="0.99",
        price="130.00",
    )
    stored = _stored_decisions(migrated_database, week)
    current_ids = {new_in.dossier_id, vetoed.dossier_id}
    current_candidates = {
        candidate["dossier_id"]: candidate
        for candidate in stored
        if candidate["dossier_id"] in current_ids
    }
    current_dossiers = [
        Dossier.model_validate(candidate["dossier"]) for candidate in current_candidates.values()
    ]
    merged_dossiers, merged_candidates = merge_week_decisions(
        current_dossiers,
        current_candidates,
        stored,
    )
    content = build_week_pack_content(
        merged_dossiers,
        merged_candidates,
        index_percentiles={"IN": "60", "US": "40"},
        index_history={
            "IN": {"history_label": "long_history"},
            "US": {"history_label": "long_history"},
        },
    )

    expected_ids = [new_in.dossier_id, old_in.dossier_id, old_us.dossier_id]
    assert [idea["symbol"] for idea in content["ideas"]] == [
        new_in.dossier_id.rsplit("/", 1)[-1],
        old_in.dossier_id.rsplit("/", 1)[-1],
        old_us.dossier_id.rsplit("/", 1)[-1],
    ]
    assert len({idea["security_id"] for idea in content["ideas"]}) == len(expected_ids)
    assert content["markets"]["IN"]["idea_count"] == 2
    assert content["markets"]["US"]["idea_count"] == 1
    assert all("why" in idea and "entry_hint" in idea for idea in content["ideas"])
    assert (
        next(
            outcome
            for outcome in content["idea_outcomes"]
            if outcome["dossier_id"] == vetoed.dossier_id
        )["status"]
        == "VETOED"
    )
    assert vetoed.security.id not in {idea["security_id"] for idea in content["ideas"]}


def test_repack_rebuilds_ideas_without_models_and_keeps_pack_context(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    week = "2026-W43"
    suffix = uuid4().hex[:6].upper()
    _persist_decision(
        migrated_database,
        week=week,
        market="IN",
        symbol=f"IN{suffix}",
        status="APPROVED",
        composite="0.80",
        price="123.45",
    )
    _persist_decision(
        migrated_database,
        week=week,
        market="US",
        symbol=f"US{suffix}",
        status="APPROVED",
        composite="0.70",
        price="45.67",
    )
    previous = {
        "ideas": [{"symbol": "OUTDATED"}],
        "idea_outcomes": [],
        "markets": {
            "IN": {"idea_count": 99, "status": "complete"},
            "US": {"idea_count": 99, "status": "reviewing"},
        },
        "summary": "Saved synthetic summary.",
        "cio_agenda": {"synthetic": "agenda"},
        "macro": {"synthetic": "macro"},
        "run_id": "synthetic-run-id",
    }
    save_research_pack(migrated_database.owner_engine, week, previous)

    def unused_model_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("repack must not construct a model router")

    monkeypatch.setattr(run, "_owner_url", lambda: "test-only-database")
    monkeypatch.setattr(run, "make_engine", lambda _url: migrated_database.owner_engine)
    monkeypatch.setattr(migrated_database.owner_engine, "dispose", lambda: None)
    monkeypatch.setattr(run, "ModelRouter", unused_model_call)
    with migrated_database.owner_engine.connect() as connection:
        runs_before = connection.scalar(text("SELECT count(*) FROM research_runs"))
        calls_before = connection.scalar(text("SELECT count(*) FROM llm_calls"))

    exit_code = run._run_repack(Namespace(week=week))
    output = json.loads(capsys.readouterr().out)
    rebuilt = load_research_pack(migrated_database.owner_engine, week)

    assert exit_code == 0
    assert output["idea_count"] == {"IN": 1, "US": 1}
    assert rebuilt is not None
    assert rebuilt["summary"] == previous["summary"]
    assert rebuilt["cio_agenda"] == previous["cio_agenda"]
    assert rebuilt["macro"] == previous["macro"]
    assert rebuilt["run_id"] == previous["run_id"]
    assert [idea["symbol"] for idea in rebuilt["ideas"]] == [
        f"IN{suffix}",
        f"US{suffix}",
    ]
    assert all("why" in idea and "entry_hint" in idea for idea in rebuilt["ideas"])
    with migrated_database.owner_engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM research_runs")) == runs_before
        assert connection.scalar(text("SELECT count(*) FROM llm_calls")) == calls_before


def test_repack_reports_an_empty_week(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(run, "_owner_url", lambda: "test-only-database")
    monkeypatch.setattr(run, "make_engine", lambda _url: migrated_database.owner_engine)
    monkeypatch.setattr(migrated_database.owner_engine, "dispose", lambda: None)

    exit_code = run._run_repack(Namespace(week="2026-W42"))
    output = json.loads(capsys.readouterr().out)

    assert exit_code != 0
    assert output["error"] == "empty_week"
    assert "No pack or approved dossiers" in output["message"]
