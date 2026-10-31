"""Gmail statement upload keys, inbox processing, and password privacy checks."""

from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from tests.fixtures.cas.make_pdf import build_encrypted_pdf

from conftest import MigratedDatabase
from wealth.api.auth_rate_limit import AuthRateLimiter
from wealth.api.main import create_app
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import (
    AuditLog,
    HoldingLot,
    HouseholdAccess,
    HouseholdMember,
    StatementInboxItem,
    StatementInboxKey,
    StatementSavedPassword,
    User,
)
from wealth.db.session import system_session, user_session

pytestmark = pytest.mark.db

_PDF_PASSWORD = "synthetic-statement-password"
_MAX_MB = 10


@pytest.fixture
def inbox_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, str, MigratedDatabase]]:
    auth_secret = secrets.token_urlsafe(32)
    password_key = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=auth_secret,
        statement_password_key=password_key,
        max_upload_mb=_MAX_MB,
        database_url="postgresql://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    with TestClient(create_app(settings)) as client:
        yield client, auth_secret, password_key, migrated_database


def _headers(secret: str, label: str) -> tuple[dict[str, str], dict[str, str]]:
    claims = {
        "sub": f"synthetic-statement-{label}-{uuid4()}",
        "email": f"{label}-{uuid4()}@example.test",
        "name": "Synthetic Test Member",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    token = jwt.encode(claims, secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}, claims


def _pdf(password: str = _PDF_PASSWORD) -> bytes:
    fixture = Path(__file__).parents[1] / "fixtures" / "cas" / "nsdl.txt"
    return build_encrypted_pdf(fixture.read_text(encoding="utf-8").splitlines(), password)


def _create_key(client: TestClient, headers: dict[str, str], label: str = "Synthetic Gmail") -> str:
    response = client.post("/v1/statement-inbox/keys", headers=headers, json={"label": label})
    assert response.status_code == 201, response.text
    key = response.json()["key"]
    assert key.startswith("wsk_")
    return key


def _upload(
    client: TestClient,
    key: str | None,
    *,
    pdf: bytes | None = None,
    message_id: str = "synthetic-message-1",
    received_at: datetime | None = None,
    sender: str = "nsdl@example.test",
    subject: str = "Monthly statement",
) -> Any:
    headers = {"Authorization": f"StatementKey {key}"} if key is not None else {}
    form = {
        "sender": sender,
        "subject": subject,
        "received_at": (received_at or datetime.now(UTC)).isoformat(),
        "message_id": message_id,
    }
    return client.post(
        "/v1/statement-inbox/uploads",
        headers=headers,
        files={"file": ("synthetic-statement.pdf", pdf or _pdf(), "application/pdf")},
        data=form,
    )


def _household_id(database: MigratedDatabase, subject: str) -> UUID:
    with system_session(database.owner_engine) as session:
        return session.scalar(select(User.id).where(User.auth_subject == subject))


def _add_shared_login(
    database: MigratedDatabase,
    owner_id: UUID,
    grantee_id: UUID,
    role: str,
) -> None:
    with system_session(database.owner_engine) as session:
        member = HouseholdMember(
            user_id=owner_id,
            display_name=f"Synthetic {role.title()}",
            relationship="other",
            is_self=False,
        )
        session.add(member)
        session.flush()
        session.add(
            HouseholdAccess(
                owner_user_id=owner_id,
                grantee_user_id=grantee_id,
                member_id=member.id,
                role=role,
            )
        )


def test_key_is_shown_once_and_revocation_stops_uploads(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, database = inbox_client
    headers, claims = _headers(secret, "key-lifecycle")
    key = _create_key(client, headers)

    listed = client.get("/v1/statement-inbox/keys", headers=headers)
    assert listed.status_code == 200
    assert "key" not in listed.json()[0]
    assert listed.json()[0]["label"] == "Synthetic Gmail"
    assert listed.json()[0]["last_used_at"] is None

    key_id = listed.json()[0]["id"]
    assert client.delete(f"/v1/statement-inbox/keys/{key_id}", headers=headers).status_code == 204
    assert _upload(client, key, message_id="revoked-message").status_code == 401

    owner_id = _household_id(database, claims["sub"])
    with system_session(database.owner_engine) as session:
        stored = session.scalar(
            select(StatementInboxKey).where(
                StatementInboxKey.user_id == owner_id,
                StatementInboxKey.id == UUID(key_id),
            )
        )
        assert stored is not None and stored.key_hash == hashlib.sha256(key.encode()).hexdigest()
        assert stored.revoked_at is not None


def test_partners_can_create_keys_and_viewers_cannot(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, database = inbox_client
    owner_headers, owner_claims = _headers(secret, "owner-role")
    client.get("/v1/me", headers=owner_headers)
    owner_id = _household_id(database, owner_claims["sub"])
    partner_headers, partner_claims = _headers(secret, "partner-role")
    client.get("/v1/me", headers=partner_headers)
    partner_id = _household_id(database, partner_claims["sub"])
    _add_shared_login(database, owner_id, partner_id, "partner")

    partner_request = {**partner_headers, "X-Household": str(owner_id)}
    created = client.post(
        "/v1/statement-inbox/keys",
        headers=partner_request,
        json={"label": "Partner Gmail"},
    )
    assert created.status_code == 201

    viewer_headers, viewer_claims = _headers(secret, "viewer-role")
    client.get("/v1/me", headers=viewer_headers)
    viewer_id = _household_id(database, viewer_claims["sub"])
    _add_shared_login(database, owner_id, viewer_id, "viewer")
    viewer_request = {**viewer_headers, "X-Household": str(owner_id)}
    denied = client.post(
        "/v1/statement-inbox/keys",
        headers=viewer_request,
        json={"label": "Viewer Gmail"},
    )
    assert denied.status_code == 403


def test_upload_checks_pdf_key_duplicate_and_private_audit(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, secret, _password_key, database = inbox_client
    owner_headers, claims = _headers(secret, "upload-checks")
    key = _create_key(client, owner_headers)
    pdf = _pdf()

    assert _upload(client, None, pdf=pdf, message_id="missing-key").status_code == 401
    assert _upload(client, "wsk_invalid-key", pdf=pdf, message_id="bad-key").status_code == 401
    invalid_pdf = _upload(client, key, pdf=b"not a pdf", message_id="not-pdf")
    assert invalid_pdf.status_code == 415
    first = _upload(client, key, pdf=pdf, message_id="duplicate-message")
    duplicate = _upload(client, key, pdf=pdf, message_id="duplicate-message")
    assert first.status_code == 202 and first.json() == {"status": "received"}
    assert duplicate.status_code == 200 and duplicate.json() == {"status": "duplicate"}
    assert pdf not in first.content

    listed = client.get("/v1/statement-inbox/items", headers=owner_headers).json()
    assert len(listed["items"]) == 1
    assert listed["items"][0]["status"] == "needs_password"

    owner_id = _household_id(database, claims["sub"])
    with system_session(database.owner_engine) as session:
        item = session.scalar(
            select(StatementInboxItem).where(StatementInboxItem.user_id == owner_id)
        )
        assert item is not None and item.pdf_data == pdf
        details = list(session.scalars(select(AuditLog.detail).where(AuditLog.user_id == owner_id)))
        assert all("pdf_data" not in detail and "key_hash" not in detail for detail in details)
        assert all(pdf not in str(detail).encode() for detail in details)
    assert key not in caplog.text
    assert pdf not in caplog.text.encode()


def test_upload_rejects_oversize_and_limits_each_key_to_twenty_a_day(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, _database = inbox_client
    owner_headers, _claims = _headers(secret, "upload-limits")
    key = _create_key(client, owner_headers)
    client.app.state.settings.max_upload_mb = 1
    oversize = _upload(client, key, pdf=b"%PDF-" + b"x" * (1024 * 1024), message_id="oversize")
    assert oversize.status_code == 413
    client.app.state.settings.max_upload_mb = _MAX_MB
    pdf = _pdf()
    for index in range(20):
        accepted = _upload(client, key, pdf=pdf, message_id=f"rate-message-{index}")
        assert accepted.status_code == 202
    limited = _upload(client, key, pdf=pdf, message_id="rate-message-limit")
    assert limited.status_code == 429


def test_upload_is_limited_per_ip_before_key_validation(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, _database = inbox_client
    headers, _claims = _headers(secret, "ip-limit")
    key = _create_key(client, headers)
    client.app.state.statement_upload_ip_limiter = AuthRateLimiter(limit=1, window_seconds=60)

    accepted = _upload(client, key, message_id="ip-limit-first")
    limited = _upload(client, "wsk_invalid-key", message_id="ip-limit-second")

    assert accepted.status_code == 202
    assert limited.status_code == 429
    assert limited.json()["code"] == "rate_limited"


def test_another_households_upload_key_cannot_reach_this_household(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, _database = inbox_client
    first_headers, _first_claims = _headers(secret, "first-household")
    second_headers, _second_claims = _headers(secret, "second-household")
    first_key = _create_key(client, first_headers, "First mailbox")
    second_key = _create_key(client, second_headers, "Second mailbox")

    response = _upload(client, second_key, message_id="second-household-message")
    assert response.status_code == 202
    assert client.get("/v1/statement-inbox/items", headers=first_headers).json()["items"] == []
    assert len(client.get("/v1/statement-inbox/items", headers=second_headers).json()["items"]) == 1
    assert first_key != second_key


def test_unreadable_pdf_is_kept_for_another_try_and_points_to_manual_upload(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, _database = inbox_client
    owner_headers, _claims = _headers(secret, "unreadable")
    key = _create_key(client, owner_headers)
    uploaded = _upload(client, key, pdf=b"%PDF-synthetic-broken", message_id="unreadable-pdf")
    assert uploaded.status_code == 202
    item_id = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"][0][
        "id"
    ]

    processed = client.post(
        f"/v1/statement-inbox/items/{item_id}/process",
        headers=owner_headers,
        json={"password": "synthetic-password", "remember": False},
    )

    assert processed.status_code == 422
    assert processed.json()["code"] == "unreadable"
    assert "uploading it in Money" in processed.json()["detail"]
    item = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"][0]
    assert item["status"] == "needs_password"
    assert item["error_code"] == "unreadable"
    retried = client.post(
        f"/v1/statement-inbox/items/{item_id}/process",
        headers=owner_headers,
        json={"password": "synthetic-password", "remember": False},
    )
    assert retried.json()["code"] == "unreadable"
    assert client.get("/v1/imports", headers=owner_headers).status_code == 200
    assert client.get("/v1/imports", headers=owner_headers).json() == []


def test_inbox_tells_full_statements_from_other_fund_house_emails(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, database = inbox_client
    owner_headers, claims = _headers(secret, "statement-kinds")
    key = _create_key(client, owner_headers)
    note = build_encrypted_pdf(["Transaction confirmation", "Synthetic Fund units"], _PDF_PASSWORD)
    assert (
        _upload(
            client,
            key,
            message_id="full-statement",
            subject="Your NSDL CAS - September 2026 (NSDL ID - 00000000)",
        ).status_code
        == 202
    )
    assert (
        _upload(
            client,
            key,
            pdf=note,
            message_id="purchase-note",
            sender="donotreply@kfintech.example.test",
            subject="Transaction confirmation for your purchase",
        ).status_code
        == 202
    )
    listed = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"]
    by_subject = {item["subject"]: item for item in listed}
    note_item = by_subject["Transaction confirmation for your purchase"]

    assert by_subject["Your NSDL CAS - September 2026 (NSDL ID - 00000000)"]["full_statement"]
    assert not note_item["full_statement"]

    tried = client.post(
        f"/v1/statement-inbox/items/{note_item['id']}/process",
        headers=owner_headers,
        json={"password": _PDF_PASSWORD, "remember": False},
    )
    assert tried.status_code == 422
    assert tried.json()["code"] == "not_a_full_statement"
    relisted = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"]
    assert {item["error_code"] for item in relisted if item["id"] == note_item["id"]} == {
        "not_a_full_statement"
    }

    removed = client.delete(f"/v1/statement-inbox/items/{note_item['id']}", headers=owner_headers)
    again = client.delete(f"/v1/statement-inbox/items/{note_item['id']}", headers=owner_headers)
    remaining = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"]
    assert removed.status_code == 204
    assert again.status_code == 404
    assert [item["subject"] for item in remaining] == [
        "Your NSDL CAS - September 2026 (NSDL ID - 00000000)"
    ]

    # A statement marked unreadable keeps its PDF, so a later reader fix can still open it.
    owner_id = _household_id(database, claims["sub"])
    with system_session(database.owner_engine) as session:
        session.execute(
            text(
                "UPDATE statement_inbox_items SET status = 'unreadable', "
                "error_code = 'unreadable' WHERE user_id = :owner"
            ),
            {"owner": owner_id},
        )
    retried = client.post(
        f"/v1/statement-inbox/items/{remaining[0]['id']}/process",
        headers=owner_headers,
        json={"password": _PDF_PASSWORD, "remember": False},
    )
    assert retried.status_code == 200, retried.text
    assert retried.json()["status"] == "review"


def test_a_statement_from_before_a_password_was_saved_can_use_it(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, _database = inbox_client
    owner_headers, _claims = _headers(secret, "use-saved-password")
    key = _create_key(client, owner_headers)
    assert _upload(client, key, message_id="earlier-statement").status_code == 202
    earlier = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"][0]
    assert earlier["saved_password"] is False

    no_saved = client.post(
        f"/v1/statement-inbox/items/{earlier['id']}/process",
        headers=owner_headers,
        json={"use_saved_password": True},
    )
    assert no_saved.status_code == 422
    assert no_saved.json()["code"] == "no_saved_password"

    # A later statement is opened by hand and its password remembered.
    later_pdf = _pdf()
    assert (
        _upload(client, key, pdf=later_pdf + b"\n", message_id="later-statement").status_code == 202
    )
    later = next(
        item
        for item in client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"]
        if item["id"] != earlier["id"]
    )
    remembered = client.post(
        f"/v1/statement-inbox/items/{later['id']}/process",
        headers=owner_headers,
        json={"password": _PDF_PASSWORD, "remember": True},
    )
    assert remembered.status_code == 200, remembered.text
    assert remembered.json()["password_saved"] is True

    listed = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"]
    assert {item["id"]: item["saved_password"] for item in listed}[earlier["id"]] is True
    opened = client.post(
        f"/v1/statement-inbox/items/{earlier['id']}/process",
        headers=owner_headers,
        json={"use_saved_password": True},
    )
    assert opened.status_code == 200, opened.text
    assert opened.json()["status"] == "review"
    assert _PDF_PASSWORD not in opened.text
    # Another provider has no saved password, so its statement still waits.
    assert (
        _upload(
            client,
            key,
            pdf=later_pdf + b"\n\n",
            message_id="third",
            sender="donotreply@cams.example.test",
        ).status_code
        == 202
    )
    third = next(
        item
        for item in client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"]
        if item["status"] == "needs_password"
    )
    missing = client.post(
        f"/v1/statement-inbox/items/{third['id']}/process", headers=owner_headers, json={}
    )
    assert missing.status_code == 422
    assert missing.json()["code"] == "password_required"


def test_wrong_password_waiting_item_saved_password_and_review_flow(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, password_key, database = inbox_client
    owner_headers, claims = _headers(secret, "password-flow")
    key = _create_key(client, owner_headers)
    pdf = _pdf()
    uploaded = _upload(client, key, pdf=pdf, message_id="password-flow-1")
    assert uploaded.status_code == 202
    item_id = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"][0][
        "id"
    ]

    wrong = client.post(
        f"/v1/statement-inbox/items/{item_id}/process",
        headers=owner_headers,
        json={"password": "wrong-synthetic-password", "remember": True},
    )
    assert wrong.status_code == 422
    assert wrong.json()["code"] == "wrong_password"
    owner_id = _household_id(database, claims["sub"])
    with system_session(database.owner_engine) as session:
        waiting = session.scalar(
            select(StatementInboxItem).where(StatementInboxItem.id == UUID(item_id))
        )
        assert waiting is not None
        assert waiting.status == "needs_password"
        assert waiting.pdf_data == pdf
        assert (
            session.scalar(
                select(func.count())
                .select_from(StatementSavedPassword)
                .where(StatementSavedPassword.user_id == owner_id)
            )
            == 0
        )

    processed = client.post(
        f"/v1/statement-inbox/items/{item_id}/process",
        headers=owner_headers,
        json={"password": _PDF_PASSWORD, "remember": True},
    )
    assert processed.status_code == 200
    job_id = UUID(processed.json()["job_id"])
    assert processed.json()["password_saved"] is True
    assert client.get(f"/v1/imports/{job_id}", headers=owner_headers).status_code == 200
    with system_session(database.owner_engine) as session:
        saved = session.scalar(
            select(StatementSavedPassword).where(StatementSavedPassword.user_id == owner_id)
        )
        assert saved is not None
        assert _PDF_PASSWORD.encode() not in saved.ciphertext
        assert saved.ciphertext != _PDF_PASSWORD.encode()
        assert (
            session.scalar(
                select(func.count()).select_from(HoldingLot).where(HoldingLot.user_id == owner_id)
            )
            == 0
        )
        item = session.scalar(
            select(StatementInboxItem).where(StatementInboxItem.id == UUID(item_id))
        )
        assert item is not None and item.status == "review" and item.pdf_data is None

    automatic = _upload(client, key, pdf=pdf, message_id="password-flow-2")
    assert automatic.status_code == 202
    inbox = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"]
    auto_item = next(row for row in inbox if row["id"] != item_id)
    assert auto_item["status"] == "review"
    automatic_job_id = auto_item["import_job_id"]
    assert automatic_job_id
    assert client.get(f"/v1/imports/{automatic_job_id}", headers=owner_headers).status_code == 200
    with system_session(database.owner_engine) as session:
        assert (
            session.scalar(
                select(func.count()).select_from(HoldingLot).where(HoldingLot.user_id == owner_id)
            )
            == 0
        )

    confirmed = client.post(f"/v1/imports/{automatic_job_id}/confirm", headers=owner_headers)
    assert confirmed.status_code == 200
    with system_session(database.owner_engine) as session:
        assert (
            session.scalar(
                select(func.count()).select_from(HoldingLot).where(HoldingLot.user_id == owner_id)
            )
            > 0
        )

    forgotten = client.delete("/v1/statement-inbox/passwords", headers=owner_headers)
    assert forgotten.status_code == 204
    with system_session(database.owner_engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(StatementSavedPassword)
                .where(StatementSavedPassword.user_id == owner_id)
            )
            == 0
        )
    assert (
        Settings(
            env="test",
            auth_mode="dev",
            dev_auth_secret=secret,
            statement_password_key=password_key,
        ).statement_password_key_value
        is not None
    )


def test_missing_password_key_keeps_remembering_disabled_with_plain_notice(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = secrets.token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    with TestClient(create_app(settings)) as client:
        owner_headers, claims = _headers(secret, "no-password-key")
        key = _create_key(client, owner_headers)
        _upload(client, key, message_id="no-key-message")
        item_id = client.get("/v1/statement-inbox/items", headers=owner_headers).json()["items"][0][
            "id"
        ]
        processed = client.post(
            f"/v1/statement-inbox/items/{item_id}/process",
            headers=owner_headers,
            json={"password": _PDF_PASSWORD, "remember": True},
        )
        assert processed.status_code == 200
        assert processed.json()["password_saved"] is False
        assert "unavailable" in processed.json()["notice"]
        with system_session(migrated_database.owner_engine) as session:
            owner_id = session.scalar(select(User.id).where(User.auth_subject == claims["sub"]))
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(StatementSavedPassword)
                    .where(StatementSavedPassword.user_id == owner_id)
                )
                == 0
            )


def test_stale_waiting_items_are_deleted_with_an_inbox_note(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, database = inbox_client
    owner_headers, claims = _headers(secret, "cleanup")
    key = _create_key(client, owner_headers)
    _upload(client, key, message_id="fresh-message")
    _upload(client, key, message_id="stale-message")
    owner_id = _household_id(database, claims["sub"])
    with system_session(database.owner_engine) as session:
        stale = session.scalar(
            select(StatementInboxItem).where(
                StatementInboxItem.user_id == owner_id,
                StatementInboxItem.message_id == "stale-message",
            )
        )
        assert stale is not None
        stale.received_at = datetime.now(UTC) - timedelta(days=31)

    response = client.get("/v1/statement-inbox/items", headers=owner_headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    assert "removed" in body["cleanup_notice"]
    with system_session(database.owner_engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(StatementInboxItem)
                .where(
                    StatementInboxItem.user_id == owner_id, StatementInboxItem.pdf_data.is_not(None)
                )
            )
            == 1
        )


def test_household_merge_discards_duplicate_inbox_items_and_saved_passwords(
    inbox_client: tuple[TestClient, str, str, MigratedDatabase],
) -> None:
    client, secret, _password_key, database = inbox_client
    owner_headers, owner_claims = _headers(secret, "merge-owner")
    client.get("/v1/me", headers=owner_headers)
    owner_id = _household_id(database, owner_claims["sub"])
    partner_headers, partner_claims = _headers(secret, "merge-partner")
    client.get("/v1/me", headers=partner_headers)
    partner_id = _household_id(database, partner_claims["sub"])
    _add_shared_login(database, owner_id, partner_id, "partner")

    owner_key = StatementInboxKey(
        id=uuid4(),
        user_id=owner_id,
        created_by=owner_id,
        label="Owner inbox",
        key_hash=hashlib.sha256(b"synthetic-owner-key").hexdigest(),
    )
    partner_key = StatementInboxKey(
        id=uuid4(),
        user_id=partner_id,
        created_by=partner_id,
        label="Partner inbox",
        key_hash=hashlib.sha256(b"synthetic-partner-key").hexdigest(),
    )
    digest = hashlib.sha256(b"%PDF-synthetic").hexdigest()
    with system_session(database.owner_engine) as session:
        session.add_all([owner_key, partner_key])
        session.flush()
        session.add_all(
            [
                StatementInboxItem(
                    id=uuid4(),
                    user_id=owner_id,
                    key_id=owner_key.id,
                    sender_type="nsdl",
                    sender="nsdl@example.test",
                    subject="Synthetic statement",
                    received_at=datetime.now(UTC),
                    message_id="duplicate-merge-message",
                    file_hash=digest,
                    pdf_data=b"%PDF-synthetic",
                ),
                StatementInboxItem(
                    id=uuid4(),
                    user_id=partner_id,
                    key_id=partner_key.id,
                    sender_type="nsdl",
                    sender="nsdl@example.test",
                    subject="Synthetic statement",
                    received_at=datetime.now(UTC),
                    message_id="duplicate-merge-message",
                    file_hash=digest,
                    pdf_data=b"%PDF-synthetic",
                ),
                StatementSavedPassword(
                    id=uuid4(),
                    user_id=owner_id,
                    sender_type="nsdl",
                    nonce=b"o" * 12,
                    ciphertext=b"synthetic-owner-ciphertext",
                ),
                StatementSavedPassword(
                    id=uuid4(),
                    user_id=partner_id,
                    sender_type="nsdl",
                    nonce=b"p" * 12,
                    ciphertext=b"synthetic-partner-ciphertext",
                ),
            ]
        )

    with user_session(database.api_engine, partner_id, owner_id) as session:
        merged = session.scalar(
            text("SELECT app_merge_into_household(:owner_id)"), {"owner_id": owner_id}
        )
    assert merged["counts"]["statement_inbox_items_discarded_as_duplicate"] == 1
    assert merged["counts"]["statement_saved_passwords_discarded_as_duplicate"] == 1

    with system_session(database.owner_engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(StatementInboxItem)
                .where(StatementInboxItem.user_id == owner_id)
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(StatementSavedPassword)
                .where(StatementSavedPassword.user_id == owner_id)
            )
            == 1
        )


def test_statement_inbox_migration_can_downgrade_and_upgrade(
    migrated_database: MigratedDatabase,
) -> None:
    from alembic import command

    command.downgrade(migrated_database.config, "0019_auth_sessions")
    command.upgrade(migrated_database.config, "head")
