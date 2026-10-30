"""App-session exchange, refresh rotation, revocation, and RLS checks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from secrets import token_bytes, token_urlsafe
from typing import Any
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from conftest import MigratedDatabase
from wealth.api import google_auth
from wealth.api.auth import AuthClaims
from wealth.api.main import create_app
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import AuditLog, AuthSession
from wealth.db.session import system_session

pytestmark = pytest.mark.db

_REDIRECT_URI = "https://pwa.example.test/auth/callback"
_CODE_VERIFIER = "synthetic-verifier-value-that-is-long-enough-for-pkce-000000"


@pytest.fixture
def api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, MigratedDatabase, dict[str, tuple[str, str]]]]:
    suffix = uuid4().hex
    accounts = {
        "owner": (f"synthetic-owner-{suffix}", f"owner-{suffix}@example.test"),
        "other": (f"synthetic-other-{suffix}", f"other-{suffix}@example.test"),
    }
    session_secret = token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="oidc",
        auth_jwks_url="https://identity.example.test/keys",
        auth_issuer="https://identity.example.test/",
        auth_audience="synthetic-google-client",
        google_client_id="synthetic-google-client",
        google_client_secret="synthetic-google-secret",
        app_session_secret=session_secret,
        allowed_emails=",".join(email for _subject, email in accounts.values()),
        allowed_origins="https://pwa.example.test",
        database_url="postgresql+psycopg://api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)

    async def fake_exchange(**values: str) -> tuple[str, int]:
        return values["code"], 3600

    def identity(token: str, *_args: Any) -> AuthClaims:
        subject, email = accounts[token]
        return AuthClaims(subject=subject, email=email, name="Synthetic Member")

    monkeypatch.setattr(google_auth, "_fetch_google_token", fake_exchange)
    monkeypatch.setattr(google_auth, "verify_token", identity)
    with TestClient(create_app(settings)) as client:
        yield client, session_secret, migrated_database, accounts


def _login(client: TestClient, code: str) -> dict[str, Any]:
    response = client.post(
        "/v1/auth/google/exchange",
        json={
            "code": code,
            "code_verifier": _CODE_VERIFIER,
            "redirect_uri": _REDIRECT_URI,
        },
        headers={
            "Origin": "https://pwa.example.test",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
        },
    )
    assert response.status_code == 200
    return response.json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _refresh(client: TestClient, token: str) -> Any:
    return client.post("/v1/auth/refresh", json={"refresh_token": token})


def _session_id(access_token: str, secret: str) -> UUID:
    claims = jwt.decode(
        access_token,
        secret,
        algorithms=["HS256"],
        audience="wealth-pwa",
        issuer="wealth-api",
    )
    return UUID(claims["sid"])


def test_exchange_access_and_refresh_rotation(
    api_client: tuple[TestClient, str, MigratedDatabase, dict[str, tuple[str, str]]],
) -> None:
    client, secret, database, _accounts = api_client
    first = _login(client, "owner")
    assert first["expires_in"] == 1800
    assert first["token_type"] == "Bearer"
    assert "session_expires_at" in first
    assert "refresh_token" in first
    assert client.get("/v1/me", headers=_bearer(first["access_token"])).status_code == 200

    access_claims = jwt.decode(
        first["access_token"],
        options={"verify_signature": False},
        algorithms=["HS256"],
    )
    expired_claims = {
        **access_claims,
        "exp": datetime.now(UTC) - timedelta(minutes=5),
    }
    expired_access = jwt.encode(expired_claims, secret, algorithm="HS256")
    assert client.get("/v1/me", headers=_bearer(expired_access)).status_code == 401

    rotated_response = _refresh(client, first["refresh_token"])
    assert rotated_response.status_code == 200
    rotated = rotated_response.json()
    assert rotated["refresh_token"] != first["refresh_token"]
    assert client.get("/v1/me", headers=_bearer(rotated["access_token"])).status_code == 200

    session_id = _session_id(first["access_token"], secret)
    old_hash = hashlib.sha256(first["refresh_token"].encode("ascii")).hexdigest()
    with system_session(database.owner_engine) as session:
        record = session.get(AuthSession, session_id)
        assert record is not None
        assert record.previous_hash == old_hash
        assert (
            record.token_hash
            == hashlib.sha256(rotated["refresh_token"].encode("ascii")).hexdigest()
        )
        assert record.device_label == "iPhone"


def test_reusing_rotated_refresh_token_revokes_and_audits_session(
    api_client: tuple[TestClient, str, MigratedDatabase, dict[str, tuple[str, str]]],
) -> None:
    client, secret, database, _accounts = api_client
    first = _login(client, "owner")
    rotated = _refresh(client, first["refresh_token"]).json()
    session_id = _session_id(first["access_token"], secret)

    # Inside the grace window (another tab, or a lost response) the old token is refused
    # without revoking the session.
    assert _refresh(client, first["refresh_token"]).status_code == 409
    assert client.get("/v1/me", headers=_bearer(rotated["access_token"])).status_code == 200

    with system_session(database.owner_engine) as session:
        record = session.get(AuthSession, session_id)
        assert record is not None
        record.last_used_at = datetime.now(UTC) - timedelta(minutes=5)

    reuse = _refresh(client, first["refresh_token"])
    assert reuse.status_code == 401
    assert client.get("/v1/me", headers=_bearer(rotated["access_token"])).status_code == 401
    assert _refresh(client, rotated["refresh_token"]).status_code == 401

    with system_session(database.owner_engine) as session:
        record = session.get(AuthSession, session_id)
        assert record is not None and record.revoked_at is not None
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(
                    AuditLog.user_id == record.user_id,
                    AuditLog.action == "auth_refresh_reuse_detected",
                    AuditLog.detail["session_id"].astext == str(session_id),
                )
            )
            == 1
        )


def test_refresh_sliding_expiry_is_capped_at_absolute_expiry(
    api_client: tuple[TestClient, str, MigratedDatabase, dict[str, tuple[str, str]]],
) -> None:
    client, secret, database, _accounts = api_client
    first = _login(client, "owner")
    session_id = _session_id(first["access_token"], secret)
    capped_expiry = datetime.now(UTC) + timedelta(days=1)
    with system_session(database.owner_engine) as session:
        record = session.get(AuthSession, session_id)
        assert record is not None
        record.expires_at = datetime.now(UTC) + timedelta(hours=1)
        record.absolute_expires_at = capped_expiry

    rotated = _refresh(client, first["refresh_token"])
    assert rotated.status_code == 200
    with system_session(database.owner_engine) as session:
        record = session.get(AuthSession, session_id)
        assert record is not None
        assert record.expires_at == record.absolute_expires_at
        assert record.absolute_expires_at.date() == capped_expiry.date()


def test_expired_session_and_removed_email_are_rejected_on_refresh(
    api_client: tuple[TestClient, str, MigratedDatabase, dict[str, tuple[str, str]]],
) -> None:
    client, secret, database, accounts = api_client
    first = _login(client, "owner")
    session_id = _session_id(first["access_token"], secret)
    with system_session(database.owner_engine) as session:
        record = session.get(AuthSession, session_id)
        assert record is not None
        record.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert _refresh(client, first["refresh_token"]).status_code == 401

    second = _login(client, "owner")
    client.app.state.settings.allowed_emails = accounts["other"][1]
    assert _refresh(client, second["refresh_token"]).status_code == 403


def test_logout_and_logout_all_only_affect_this_login(
    api_client: tuple[TestClient, str, MigratedDatabase, dict[str, tuple[str, str]]],
) -> None:
    client, _secret, _database, _accounts = api_client
    owner_one = _login(client, "owner")
    owner_two = _login(client, "owner")
    other = _login(client, "other")

    sessions = client.get("/v1/auth/sessions", headers=_bearer(owner_one["access_token"]))
    assert sessions.status_code == 200
    rows = sessions.json()
    assert len(rows) == 2
    assert sum(row["current"] for row in rows) == 1
    assert all(row["device_label"] == "iPhone" for row in rows)
    assert all("created_at" in row and "last_used_at" in row for row in rows)
    other_session_id = _session_id(other["access_token"], _secret)
    assert all(UUID(row["id"]) != other_session_id for row in rows)

    owner_logout = client.post("/v1/auth/logout", headers=_bearer(owner_one["access_token"]))
    assert owner_logout.status_code == 204
    assert client.get("/v1/me", headers=_bearer(owner_one["access_token"])).status_code == 401
    assert client.get("/v1/me", headers=_bearer(owner_two["access_token"])).status_code == 200
    assert client.get("/v1/me", headers=_bearer(other["access_token"])).status_code == 200

    assert (
        client.post("/v1/auth/logout-all", headers=_bearer(owner_two["access_token"])).status_code
        == 204
    )
    assert client.get("/v1/me", headers=_bearer(owner_two["access_token"])).status_code == 401
    assert client.get("/v1/me", headers=_bearer(other["access_token"])).status_code == 200


def test_refresh_lookup_is_narrow_scoped_and_sessions_are_login_rls_rows(
    migrated_database: MigratedDatabase,
) -> None:
    database = migrated_database
    with database.owner_engine.connect() as connection:
        function = connection.execute(
            text(
                """
                SELECT procedure.pronargs, procedure.proargnames,
                       pg_catalog.pg_get_function_result(procedure.oid) AS result_type,
                       procedure.prosecdef,
                       EXISTS (
                           SELECT 1 FROM pg_catalog.aclexplode(
                               COALESCE(
                                   procedure.proacl,
                                   pg_catalog.acldefault('f', procedure.proowner)
                               )
                           ) AS acl
                           WHERE acl.grantee = 0 AND acl.privilege_type = 'EXECUTE'
                       ) AS public_execute,
                       pg_catalog.has_function_privilege('wealth_api', procedure.oid, 'EXECUTE')
                           AS api_execute
                FROM pg_catalog.pg_proc AS procedure
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = procedure.pronamespace
                WHERE namespace.nspname = :schema
                  AND procedure.proname = 'app_lookup_refresh_session'
                """
            ),
            {"schema": database.schema_name},
        ).one()
        table = connection.execute(
            text(
                """
                SELECT relation.relrowsecurity, relation.relforcerowsecurity,
                       EXISTS (
                           SELECT 1 FROM pg_catalog.pg_policy AS policy
                           WHERE policy.polrelid = relation.oid
                             AND policy.polname = 'auth_sessions_personal_isolation'
                             AND pg_catalog.pg_get_expr(policy.polqual, policy.polrelid)
                                 LIKE '%app.user_id%'
                       ) AS personal_policy,
                       EXISTS (
                           SELECT 1 FROM pg_catalog.pg_trigger AS trigger
                           WHERE trigger.tgrelid = relation.oid
                             AND trigger.tgname = 'trg_auth_sessions_household_audit'
                       ) AS audit_trigger,
                       EXISTS (
                           SELECT 1 FROM pg_catalog.aclexplode(
                               COALESCE(
                                   relation.relacl,
                                   pg_catalog.acldefault('r', relation.relowner)
                               )
                           ) AS acl
                           JOIN pg_catalog.pg_roles AS granted_role
                             ON granted_role.oid = acl.grantee
                           WHERE granted_role.rolname = 'wealth_api'
                             AND acl.privilege_type IN ('SELECT', 'INSERT', 'UPDATE', 'DELETE')
                       ) AS direct_api_table_grant
                FROM pg_catalog.pg_class AS relation
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = :schema AND relation.relname = 'auth_sessions'
                """
            ),
            {"schema": database.schema_name},
        ).one()

    assert function.pronargs == 1
    assert function.proargnames == ["p_token_hash", "session_id", "user_id"]
    assert function.result_type == "TABLE(session_id uuid, user_id uuid)"
    assert function.prosecdef
    assert not function.public_execute
    assert function.api_execute
    assert table.relrowsecurity and table.relforcerowsecurity
    assert table.personal_policy
    assert table.audit_trigger
    assert not table.direct_api_table_grant

    with database.api_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT * FROM app_lookup_refresh_session(:token_hash)"),
                {"token_hash": hashlib.sha256(token_bytes(32)).hexdigest()},
            ).first()
            is None
        )


def test_a_request_after_deleting_the_account_does_not_recreate_the_user(
    api_client: tuple[TestClient, str, MigratedDatabase, dict[str, tuple[str, str]]],
) -> None:
    client, _secret, database, accounts = api_client
    login = _login(client, "owner")
    subject, _email = accounts["owner"]

    assert client.delete("/v1/me", headers=_bearer(login["access_token"])).status_code == 204
    # A request still in flight carries the old, unexpired access token.
    late = client.get("/v1/me", headers=_bearer(login["access_token"]))

    assert late.status_code == 401
    with database.owner_engine.connect() as connection:
        remaining = connection.execute(
            text("SELECT count(*) FROM users WHERE auth_subject = :subject"),
            {"subject": subject},
        ).scalar_one()
    assert remaining == 0
