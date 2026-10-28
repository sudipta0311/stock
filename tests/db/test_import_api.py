"""CAS staging and confirmation against the RLS-scoped API database login."""

from __future__ import annotations

import asyncio
import builtins
import logging
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import httpx
import jwt
import psutil
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from tests.fixtures.cas.make_pdf import build_encrypted_pdf

from conftest import MigratedDatabase
from wealth.api.main import create_app
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import AuditLog, HoldingLot, ImportJob, Security, User
from wealth.importer.models import ParsedHolding, ParsedStatement

pytestmark = pytest.mark.db

_PASSWORD = "synthetic-cas-upload-password"
_STATEMENT_MARKER = "Synthetic Tools Limited"


@pytest.fixture
def import_api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str]]:
    secret = secrets.token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    application = create_app(settings)
    with TestClient(application) as client:
        yield client, secret


def _headers(secret: str, label: str) -> tuple[dict[str, str], dict[str, str]]:
    claims = {
        "sub": f"synthetic-cas-{label}-{uuid4()}",
        "email": f"{label}-{uuid4()}@example.test",
        "name": "Synthetic CAS User",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    token = jwt.encode(claims, secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}, claims


def _fixture_lines(name: str = "nsdl.txt") -> list[str]:
    from pathlib import Path

    return (
        (Path(__file__).parents[1] / "fixtures" / "cas" / name)
        .read_text(encoding="utf-8")
        .splitlines()
    )


def _upload(
    client: TestClient,
    headers: dict[str, str],
    *,
    lines: list[str] | None = None,
    password: str = _PASSWORD,
) -> Any:
    pdf = build_encrypted_pdf(lines or _fixture_lines(), _PASSWORD)
    return client.post(
        "/v1/imports",
        headers=headers,
        files={"file": ("synthetic-cas.pdf", pdf, "application/pdf")},
        data={"password": password, "doc_type": "cas"},
    )


def test_upload_review_confirm_refreshes_holdings_and_networth(
    import_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = import_api_client
    headers, claims = _headers(secret, "roundtrip")
    assert (
        client.put(
            "/v1/me/checklist",
            headers=headers,
            json={"categories": ["cas_upload", "us_investments"]},
        ).status_code
        == 200
    )

    uploaded = _upload(client, headers)
    assert uploaded.status_code == 201, uploaded.text
    body = uploaded.json()
    assert body["status"] == "ready"
    assert body["source"] == "nsdl"
    assert body["statement_date"] == "2026-06-30"
    assert body["counts"] == {"new": 7, "changed": 0, "duplicate": 0, "unknown": 0}
    job_id = body["job_id"]

    details = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert len(details["items"]) == 7
    equity_item = next(item for item in details["items"] if item["kind"] == "equity")
    assert equity_item["sleeve"] == "equity_india"
    assert equity_item["match"] == {"lot_id": None, "security_id": None}
    assert "items" not in client.get("/v1/imports", headers=headers).json()[0]

    confirmed = client.post(f"/v1/imports/{job_id}/confirm", headers=headers)
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "confirmed"
    assert confirmed.json()["net_worth"] == "52410.00"
    assert client.post(f"/v1/imports/{job_id}/confirm", headers=headers).status_code == 409

    networth = client.get("/v1/networth", headers=headers).json()
    assert networth["assets_total"] == "52410.00"
    assert networth["investable_total"] == "52410.00"
    assert networth["liquidity_ladder"]["instant"] == "4410.00"
    assert networth["liquidity_ladder"]["within_1y"] == "48000.00"
    assert {row["sleeve"]: row["amount"] for row in networth["by_sleeve"]}[
        "equity_india"
    ] == "4410.00"
    assert networth["unvalued_assets"] == []
    assert networth["completeness"] == {
        "ticked": ["cas_upload", "us_investments"],
        "done": ["cas_upload"],
        "pct": "50.00",
    }

    with migrated_database.owner_engine.begin() as connection:
        user_id = connection.scalar(select(User.id).where(User.auth_subject == claims["sub"]))
        stored_job = connection.execute(
            select(ImportJob.file_deleted_at, ImportJob.parse_report).where(
                ImportJob.id == UUID(job_id), ImportJob.user_id == user_id
            )
        ).one()
        assert (
            connection.scalar(
                select(func.count()).select_from(HoldingLot).where(HoldingLot.user_id == user_id)
            )
            == 7
        )
        audit = connection.execute(
            select(AuditLog.detail).where(
                AuditLog.user_id == user_id,
                AuditLog.action == "cas_import_confirmed",
            )
        ).scalar_one()
        assert audit == {"new": 7, "changed": 0, "accepted": 0, "skipped": 0}
    assert stored_job.file_deleted_at is not None
    assert stored_job.parse_report == {"warnings": []}

    duplicate = _upload(client, headers)
    assert duplicate.status_code == 201
    assert duplicate.json()["counts"] == {
        "new": 0,
        "changed": 0,
        "duplicate": 7,
        "unknown": 0,
    }

    with migrated_database.owner_engine.begin() as connection:
        user_id = connection.scalar(select(User.id).where(User.auth_subject == claims["sub"]))
        security_id = connection.scalar(select(Security.id).where(Security.isin == "INE000A01012"))
        broker_id = uuid4()
        manual_id = uuid4()
        for lot_id, source in ((broker_id, "broker"), (manual_id, "manual")):
            connection.execute(
                HoldingLot.__table__.insert().values(
                    id=lot_id,
                    user_id=user_id,
                    security_id=security_id,
                    quantity=Decimal("3"),
                    source=source,
                    account_ref="XXXX5678",
                )
            )

    changed_lines = _fixture_lines()
    original_line = "Synthetic Tools Limited 10.00 20 100.00 2000.00"
    changed_line = "Synthetic Tools Limited 10.00 21 100.00 2100.00"
    changed_lines = [line.replace(original_line, changed_line) for line in changed_lines]
    changed = _upload(client, headers, lines=changed_lines)
    assert changed.status_code == 201
    assert changed.json()["counts"] == {
        "new": 0,
        "changed": 1,
        "duplicate": 6,
        "unknown": 0,
    }
    changed_detail = client.get(f"/v1/imports/{changed.json()['job_id']}", headers=headers).json()
    changed_item = next(item for item in changed_detail["items"] if item["status"] == "changed")
    assert changed_item["diff"]["quantity"] == {"old": "20.000000", "new": "21"}
    assert (
        client.post(f"/v1/imports/{changed.json()['job_id']}/confirm", headers=headers).status_code
        == 200
    )

    with migrated_database.owner_engine.connect() as connection:
        cas_quantity = connection.scalar(
            select(func.sum(HoldingLot.quantity)).where(
                HoldingLot.user_id == user_id,
                HoldingLot.security_id == security_id,
                HoldingLot.source == "cas",
                HoldingLot.account_ref == "XXXX5678",
            )
        )
        broker_count = connection.scalar(
            select(func.count())
            .select_from(HoldingLot)
            .where(HoldingLot.id == broker_id, HoldingLot.source == "broker")
        )
        manual_count = connection.scalar(
            select(func.count())
            .select_from(HoldingLot)
            .where(HoldingLot.id == manual_id, HoldingLot.source == "manual")
        )
    assert cas_quantity == Decimal("21.000000")
    assert broker_count == 1
    assert manual_count == 1


def test_import_privacy_and_failed_job_metadata(
    import_api_client: tuple[TestClient, str],
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, secret = import_api_client
    headers, claims = _headers(secret, "privacy")
    original_open = builtins.open

    def no_write_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            raise AssertionError("uploads must not open a file for writing")
        return original_open(file, mode, *args, **kwargs)

    def fail_temp(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("uploads must not create a filesystem temporary file")

    monkeypatch.setattr(builtins, "open", no_write_open)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_temp)
    monkeypatch.setattr(tempfile, "mkstemp", fail_temp)
    caplog.set_level(logging.DEBUG)

    successful = _upload(client, headers)
    assert successful.status_code == 201
    assert _PASSWORD not in successful.text
    assert _STATEMENT_MARKER not in successful.text

    failed = _upload(client, headers, password="incorrect-synthetic-password")
    assert failed.status_code == 422
    assert failed.headers["content-type"].startswith("application/problem+json")
    assert failed.json()["code"] == "wrong_password"
    assert "incorrect-synthetic-password" not in failed.text
    assert _PASSWORD not in failed.text
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert _PASSWORD not in log_text
    assert "incorrect-synthetic-password" not in log_text
    assert _STATEMENT_MARKER not in log_text

    with migrated_database.owner_engine.connect() as connection:
        user_id = connection.scalar(select(User.id).where(User.auth_subject == claims["sub"]))
        failed_job = connection.execute(
            select(ImportJob.error_code, ImportJob.parse_report, ImportJob.file_deleted_at).where(
                ImportJob.user_id == user_id, ImportJob.status == "failed"
            )
        ).one()
    assert failed_job.error_code == "wrong_password"
    assert failed_job.parse_report is None
    assert failed_job.file_deleted_at is not None


def test_upload_rejects_size_type_and_unsupported_statements(
    import_api_client: tuple[TestClient, str],
) -> None:
    client, secret = import_api_client
    headers, _claims = _headers(secret, "errors")

    too_large = client.post(
        "/v1/imports",
        headers=headers,
        files={"file": ("large.pdf", b"%PDF" + b"x" * (10 * 1024 * 1024), "application/pdf")},
    )
    assert too_large.status_code == 413
    assert too_large.headers["content-type"].startswith("application/problem+json")
    assert too_large.json()["code"] == "too_large"

    not_pdf = client.post(
        "/v1/imports",
        headers=headers,
        files={"file": ("wrong.txt", b"not a PDF", "text/plain")},
    )
    assert not_pdf.status_code == 415
    assert not_pdf.json()["code"] == "not_a_pdf"

    unsupported = _upload(client, headers, lines=["Synthetic statement with unsupported layout"])
    assert unsupported.status_code == 422
    assert unsupported.json()["code"] == "unsupported_format"


def test_unknown_items_can_be_classified_and_confirmed(
    import_api_client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from wealth.api import imports as import_routes

    client, secret = import_api_client
    headers, _claims = _headers(secret, "override")
    monkeypatch.setattr(
        import_routes,
        "parse_cas",
        lambda _data, _password, **_kwargs: ParsedStatement(
            source="nsdl",
            statement_date=date(2026, 6, 30),
            holdings=[
                ParsedHolding(
                    kind="other",
                    isin="IN0000000026",
                    name="Synthetic Unclassified Holding",
                    quantity="2",
                    market_value="200",
                    account_ref="XXXX1234",
                )
            ],
            transactions=[],
        ),
    )
    response = client.post(
        "/v1/imports",
        headers=headers,
        files={"file": ("synthetic.pdf", b"%PDFsynthetic", "application/pdf")},
    )
    assert response.status_code == 201
    assert response.json()["counts"]["unknown"] == 1
    job_id = response.json()["job_id"]
    item = client.get(f"/v1/imports/{job_id}", headers=headers).json()["items"][0]
    patched = client.patch(
        f"/v1/imports/{job_id}/items/{item['id']}",
        headers=headers,
        json={"override": {"kind": "sgb", "sleeve": "gold"}},
    )
    assert patched.status_code == 200
    assert patched.json()["status"] == "accepted"
    assert patched.json()["sleeve"] == "gold"
    assert client.post(f"/v1/imports/{job_id}/confirm", headers=headers).status_code == 200


def test_cams_confirmation_keeps_buy_lots_and_nets_sells_fifo(
    import_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = import_api_client
    headers, claims = _headers(secret, "cams-transactions")
    uploaded = _upload(client, headers, lines=_fixture_lines("cams_kfin.txt"))
    assert uploaded.status_code == 201
    item = client.get(f"/v1/imports/{uploaded.json()['job_id']}", headers=headers).json()["items"][
        0
    ]
    assert (
        client.post(f"/v1/imports/{uploaded.json()['job_id']}/confirm", headers=headers).status_code
        == 200
    )

    with migrated_database.owner_engine.connect() as connection:
        user_id = connection.scalar(select(User.id).where(User.auth_subject == claims["sub"]))
        lots = list(
            connection.execute(
                select(HoldingLot.quantity, HoldingLot.cost_per_unit, HoldingLot.acquired_on)
                .join(Security, Security.id == HoldingLot.security_id)
                .where(
                    HoldingLot.user_id == user_id,
                    Security.isin == item["isin"],
                    HoldingLot.source == "cas",
                )
            )
        )
    assert sum((row.quantity for row in lots), Decimal(0)) == Decimal(item["quantity"])
    assert any(row.acquired_on is not None and row.cost_per_unit is not None for row in lots)


def test_security_upsert_function_is_the_api_write_path(
    migrated_database: MigratedDatabase,
) -> None:
    parameters = {
        "market": "IN",
        "isin": "INE000A01012",
        "symbol": "INE000A01012",
        "name": "Synthetic Upsert Security",
        "kind": "equity",
        "currency": "INR",
        "sleeve": "equity_india",
        "category": "equity:india",
    }
    with migrated_database.api_engine.begin() as connection:
        first_id = connection.scalar(
            text(
                "SELECT app_upsert_security(:market, :isin, :symbol, :name, :kind, "
                ":currency, :sleeve, :category)"
            ),
            parameters,
        )
        second_id = connection.scalar(
            text(
                "SELECT app_upsert_security(:market, :isin, :symbol, :name, :kind, "
                ":currency, NULL, NULL)"
            ),
            parameters,
        )
        security = connection.execute(
            text("SELECT sleeve, category FROM securities WHERE isin = :isin"), parameters
        ).one()
    assert first_id == second_id
    assert security == ("equity_india", "equity:india")

    with migrated_database.owner_engine.connect() as connection:
        function = connection.execute(
            text(
                """
                SELECT pg_catalog.pg_get_userbyid(procedure.proowner) AS function_owner,
                       procedure.pronargs,
                       procedure.proconfig,
                       EXISTS (
                           SELECT 1
                           FROM pg_catalog.aclexplode(procedure.proacl) AS acl
                           WHERE acl.grantee = 0 AND acl.privilege_type = 'EXECUTE'
                       ) AS public_can_execute,
                       pg_catalog.has_function_privilege(
                           'wealth_app', procedure.oid, 'EXECUTE'
                       ) AS app_can_execute
                FROM pg_catalog.pg_proc AS procedure
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = procedure.pronamespace
                WHERE namespace.nspname = :schema_name
                  AND procedure.proname = 'app_upsert_security'
                """
            ),
            {"schema_name": migrated_database.schema_name},
        ).one()
    assert function.function_owner == migrated_database.owner_role
    assert function.pronargs == 8
    assert any(
        setting.startswith("search_path=pg_catalog,") and migrated_database.schema_name in setting
        for setting in function.proconfig
    )
    assert not function.public_can_execute
    assert function.app_can_execute

    with pytest.raises(DBAPIError), migrated_database.api_engine.begin() as connection:
        connection.execute(
            text(
                "SELECT app_upsert_security(:market, :isin, :symbol, :name, :kind, "
                ":currency, :sleeve, :category)"
            ),
            {**parameters, "market": "XX", "isin": "XX0000000001"},
        )
    with pytest.raises(DBAPIError), migrated_database.api_engine.begin() as connection:
        connection.execute(
            Security.__table__.insert().values(
                id=uuid4(),
                market="IN",
                symbol="SYNTHETIC-DIRECT",
                isin="IN0000000026",
                name="Synthetic Forbidden Security",
                security_type="equity",
                currency="INR",
            )
        )


def test_import_jobs_and_items_are_hidden_across_tenants(
    import_api_client: tuple[TestClient, str],
) -> None:
    client, secret = import_api_client
    headers_a, _claims = _headers(secret, "tenant-a")
    headers_b, _claims_b = _headers(secret, "tenant-b")
    response = _upload(client, headers_a)
    job_id = response.json()["job_id"]
    item_id = client.get(f"/v1/imports/{job_id}", headers=headers_a).json()["items"][0]["id"]

    assert client.get(f"/v1/imports/{job_id}", headers=headers_b).status_code == 404
    assert (
        client.patch(
            f"/v1/imports/{job_id}/items/{item_id}",
            headers=headers_b,
            json={"status": "accepted"},
        ).status_code
        == 404
    )
    assert client.post(f"/v1/imports/{job_id}/confirm", headers=headers_b).status_code == 404


def test_import_can_be_discarded_and_items_are_removed(
    import_api_client: tuple[TestClient, str],
) -> None:
    client, secret = import_api_client
    headers, _claims = _headers(secret, "discard")
    uploaded = _upload(client, headers)
    job_id = uploaded.json()["job_id"]
    item_id = client.get(f"/v1/imports/{job_id}", headers=headers).json()["items"][0]["id"]
    rejected = client.patch(
        f"/v1/imports/{job_id}/items/{item_id}",
        headers=headers,
        json={"status": "rejected"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"

    assert client.delete(f"/v1/imports/{job_id}", headers=headers).status_code == 204
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert detail["status"] == "discarded"
    assert detail["items"] == []


def test_upload_parse_timeout_and_global_semaphore(
    import_api_client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from wealth.api import imports as import_routes

    client, secret = import_api_client
    same_user_headers = _headers(secret, "concurrency")[0]
    assert client.get("/v1/imports", headers=same_user_headers).status_code == 200
    parsed_fixture = ParsedStatement(
        source="nsdl",
        statement_date=date(2026, 6, 30),
        holdings=[
            ParsedHolding(
                kind="equity",
                isin="INE000A01012",
                name="Synthetic Concurrent Holding",
                quantity="1",
                market_value="10",
            )
        ],
        transactions=[],
    )
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def slow_parse(_data: bytes, _password: str | None, **_kwargs: Any) -> ParsedStatement:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.15)
        with lock:
            active -= 1
        return parsed_fixture

    monkeypatch.setattr(import_routes, "parse_cas", slow_parse)

    async def upload_twice() -> list[int]:
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            responses = await asyncio.gather(
                *(
                    async_client.post(
                        "/v1/imports",
                        headers=same_user_headers,
                        files={"file": ("synthetic.pdf", b"%PDFsynthetic", "application/pdf")},
                    )
                    for _ in range(2)
                )
            )
        return [response.status_code for response in responses]

    statuses = asyncio.run(upload_twice())
    assert statuses == [201, 201]
    assert maximum_active == 1

    monkeypatch.setattr(import_routes, "PARSE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(import_routes, "parse_cas", lambda *_args, **_kwargs: time.sleep(0.1))
    timed_out = client.post(
        "/v1/imports",
        headers=same_user_headers,
        files={"file": ("synthetic.pdf", b"%PDFsynthetic", "application/pdf")},
    )
    assert timed_out.status_code == 504
    assert timed_out.json()["code"] == "parse_timeout"


def test_uvicorn_peak_rss_stays_below_300_mb_after_three_uploads(
    migrated_database: MigratedDatabase,
) -> None:
    secret = secrets.token_urlsafe(32)
    headers, _claims = _headers(secret, "rss")
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        port = server_socket.getsockname()[1]

    environment = os.environ.copy()
    environment.update(
        {
            "DATABASE_URL": migrated_database.api_engine.url.render_as_string(hide_password=False),
            "ENV": "test",
            "AUTH_MODE": "dev",
            "DEV_AUTH_SECRET": secret,
        }
    )
    server_script = f"""
import psutil
import uvicorn
from wealth.api.main import app

@app.middleware("http")
async def add_process_rss(request, call_next):
    response = await call_next(request)
    response.headers["x-process-rss-bytes"] = str(psutil.Process().memory_info().rss)
    return response

uvicorn.run(app, host="127.0.0.1", port={port}, log_level="critical", access_log=False)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", server_script],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process_handle = psutil.Process(process.pid)
    stop_sampling = threading.Event()
    peak_rss = [process_handle.memory_info().rss]

    def sample_memory() -> None:
        while not stop_sampling.wait(0.01):
            try:
                peak_rss[0] = max(peak_rss[0], process_handle.memory_info().rss)
            except psutil.NoSuchProcess:
                return

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=10) as client:
            for _attempt in range(100):
                try:
                    ready = client.get("/healthz?deep=1")
                    if ready.status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            else:
                raise AssertionError("Uvicorn did not become ready")

            pdf = build_encrypted_pdf(_fixture_lines(), _PASSWORD)
            for _ in range(3):
                response = client.post(
                    "/v1/imports",
                    headers=headers,
                    files={"file": ("synthetic-cas.pdf", pdf, "application/pdf")},
                    data={"password": _PASSWORD},
                )
                assert response.status_code == 201
                peak_rss[0] = max(peak_rss[0], int(response.headers["x-process-rss-bytes"]))
                peak_rss[0] = max(peak_rss[0], process_handle.memory_info().rss)
    finally:
        stop_sampling.set()
        sampler.join(timeout=1)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    assert process.poll() is not None
    print(f"Uvicorn peak RSS after three uploads: {peak_rss[0] / (1024 * 1024):.1f} MiB")
    assert peak_rss[0] < 300 * 1024 * 1024
