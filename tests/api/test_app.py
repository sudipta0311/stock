"""API settings, error shape, security boundary, and export checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from secrets import token_urlsafe

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from wealth.api.main import app, create_app
from wealth.api.settings import Settings


def test_settings_default_to_production_oidc_and_fail_closed_for_dev_mode() -> None:
    settings = Settings(
        auth_jwks_url="https://identity.example.test/keys",
        auth_issuer="https://identity.example.test/",
        auth_audience="wealth-api",
        google_client_id="wealth-api",
        google_client_secret="synthetic-google-secret",
        app_session_secret=token_urlsafe(32),
        allowed_emails="family@example.test",
    )
    assert settings.env == "production"
    assert settings.auth_mode == "oidc"

    with pytest.raises(ValidationError, match="ENV is dev or test"):
        Settings(auth_mode="dev", dev_auth_secret=token_urlsafe(32))
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(env="dev", auth_mode="dev", dev_auth_secret=token_urlsafe(16))
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(env="test", auth_mode="dev")


def test_settings_reject_wildcard_origins_in_production() -> None:
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        Settings(
            env="production",
            auth_mode="oidc",
            auth_jwks_url="https://identity.example.test/keys",
            auth_issuer="https://identity.example.test/",
            auth_audience="wealth-api",
            google_client_id="wealth-api",
            google_client_secret="synthetic-google-secret",
            allowed_emails="family@example.test",
            allowed_origins="https://app.example.test, *",
        )


def test_app_session_secret_requires_32_bytes_and_is_hidden_in_repr() -> None:
    with pytest.raises(ValidationError, match="APP_SESSION_SECRET must be at least 32 bytes"):
        Settings(
            env="test",
            auth_mode="dev",
            dev_auth_secret=token_urlsafe(32),
            app_session_secret="short",
        )

    secret = token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=token_urlsafe(32),
        app_session_secret=secret,
    )
    assert secret not in repr(settings)


def test_settings_validation_error_hides_database_url_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "postgresql://api_user:synthetic-only@private-db.example.test/wealth"
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.delenv("AUTH_JWKS_URL", raising=False)
    monkeypatch.delenv("AUTH_ISSUER", raising=False)
    monkeypatch.delenv("AUTH_AUDIENCE", raising=False)

    with pytest.raises(ValidationError) as error:
        Settings()

    message = str(error.value)
    assert "postgresql://" not in message
    assert "private-db.example.test" not in message
    assert "synthetic-only" not in message


def test_api_engine_connection_error_has_a_sanitized_postgres_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wealth.db import session as db_session

    class FakeLoginError(Exception):
        sqlstate = "28000"

    class FakeConnection:
        def __enter__(self) -> None:
            raise FakeLoginError(
                'role "wealth_api_test" is not permitted to log in at host=db-host.example.test '
                "password=synthetic-only; url=postgresql://api:synthetic-only@db-host.example.test/db"
            )

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeEngine:
        disposed = False

        def connect(self) -> FakeConnection:
            return FakeConnection()

        def dispose(self) -> None:
            self.disposed = True

    engine = FakeEngine()
    monkeypatch.setattr(db_session, "make_engine", lambda _url: engine)

    with pytest.raises(ValueError) as error:
        db_session.api_engine("postgresql://api:synthetic-only@db-host.example.test/wealth")

    message = str(error.value)
    assert "not permitted to log in" in message
    assert "FakeLoginError" in message
    assert "28000" in message
    assert "synthetic-only" not in message
    assert "db-host.example.test" not in message
    assert "postgresql://" not in message
    assert engine.disposed


def test_authentication_errors_use_problem_json_and_security_headers() -> None:
    settings = Settings(env="test", auth_mode="dev", dev_auth_secret=token_urlsafe(32))
    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/me")
        health = client.get("/healthz")

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json() == {
        "type": "about:blank",
        "title": "Unauthorized",
        "status": 401,
        "detail": "A valid bearer token is required.",
        "instance": "/v1/me",
    }
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["cache-control"] == "no-store"
    for headers in (response.headers, health.headers):
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["x-frame-options"] == "DENY"


def test_development_settings_leave_email_allow_list_optional() -> None:
    settings = Settings(env="dev", auth_mode="dev", dev_auth_secret=token_urlsafe(32))

    assert settings.allowed_email_list == []


def test_loan_endpoints_are_in_the_openapi_contract() -> None:
    paths = app.openapi()["paths"]

    assert "/v1/loans" in paths
    assert {"get"} <= paths["/v1/loans"].keys()
    assert {"get"} <= paths["/v1/loans/{loan_id}"].keys()
    assert {"post"} <= paths["/v1/loans/{loan_id}/events"].keys()
    assert {"delete"} <= paths["/v1/loans/{loan_id}/events/{event_id}"].keys()
    assert {"post"} <= paths["/v1/loans/{loan_id}/prepay-preview"].keys()


def test_production_settings_require_email_allow_list() -> None:
    with pytest.raises(ValidationError, match="ALLOWED_EMAILS"):
        Settings(
            env="production",
            auth_mode="oidc",
            auth_jwks_url="https://identity.example.test/keys",
            auth_issuer="https://identity.example.test/",
            auth_audience="wealth-api",
            google_client_id="wealth-api",
            google_client_secret="synthetic-google-secret",
        )


def test_deep_health_reports_database_failure_as_problem_json() -> None:
    settings = Settings(env="test", auth_mode="dev", dev_auth_secret=token_urlsafe(32))
    with TestClient(create_app(settings)) as client:
        response = client.get("/healthz?deep=1")

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["db"] == "error"


def test_api_import_does_not_load_research_and_never_references_owner_session() -> None:
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import wealth.api.main; "
        "blocked = ('wealth.research', 'langgraph', 'openai'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in blocked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    api_source = Path(__file__).resolve().parents[2] / "src" / "wealth" / "api"
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in api_source.rglob("*.py"))
    assert "system_session" not in source_text


def test_openapi_export_matches_current_app() -> None:
    root = Path(__file__).resolve().parents[2]
    exported = (root / "schemas" / "openapi.json").read_text(encoding="utf-8")
    expected = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"

    assert exported == expected
