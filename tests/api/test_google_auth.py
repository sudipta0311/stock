"""Google code exchange route tests use a fake provider response."""

from __future__ import annotations

from datetime import UTC, datetime
from secrets import token_urlsafe
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from wealth.api import google_auth
from wealth.api.auth import AuthClaims
from wealth.api.auth_sessions import AuthSessionPair
from wealth.api.google_auth import GoogleExchangeFailure, GoogleExchangeRateLimiter
from wealth.api.main import create_app
from wealth.api.settings import Settings
from wealth.db import session as db_session

_REDIRECT_URI = "https://pwa.example.test/auth/callback"
_CODE_VERIFIER = "synthetic-verifier-value-that-is-long-enough-for-pkce-000000"


@pytest.fixture
def google_settings() -> Settings:
    return Settings(
        env="test",
        auth_mode="oidc",
        auth_jwks_url="https://identity.example.test/keys",
        auth_issuer="https://identity.example.test/",
        auth_audience="synthetic-google-client",
        google_client_id="synthetic-google-client",
        google_client_secret="synthetic-google-secret",
        app_session_secret=token_urlsafe(32),
        database_url="postgresql+psycopg://api@localhost/wealth_test",
        allowed_emails="Family@example.test",
        allowed_origins="https://pwa.example.test",
    )


@pytest.fixture(autouse=True)
def fake_app_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        db_session,
        "api_engine",
        lambda _url: SimpleNamespace(dispose=lambda: None),
    )
    monkeypatch.setattr(
        google_auth,
        "create_app_session",
        lambda **_values: AuthSessionPair(
            access_token="synthetic-app-access-token",
            refresh_token="synthetic-app-refresh-token",
            expires_in=1800,
            session_expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
    )


def _allow_identity(monkeypatch: pytest.MonkeyPatch, email: str) -> None:
    monkeypatch.setattr(
        google_auth,
        "verify_token",
        lambda *_args: AuthClaims(subject="synthetic-subject", email=email, name=None),
    )


def _fake_exchange(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: tuple[str, int] = ("synthetic.id.token", 3600),
    failure: Exception | None = None,
) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []

    async def exchange(**values: str) -> tuple[str, int]:
        calls.append(values)
        if failure is not None:
            raise failure
        return result

    monkeypatch.setattr(google_auth, "_fetch_google_token", exchange)
    return calls


def _post(client: TestClient) -> object:
    return client.post(
        "/v1/auth/google/exchange",
        json={
            "code": "synthetic-authorization-code",
            "code_verifier": _CODE_VERIFIER,
            "redirect_uri": _REDIRECT_URI,
        },
        headers={"Origin": "https://pwa.example.test"},
    )


def test_google_exchange_returns_verified_id_token_and_app_session(
    monkeypatch: pytest.MonkeyPatch, google_settings: Settings
) -> None:
    calls = _fake_exchange(monkeypatch)
    _allow_identity(monkeypatch, "FAMILY@example.test")

    with TestClient(create_app(google_settings)) as client:
        response = _post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["id_token"] == "synthetic.id.token"
    assert body["access_token"] == "synthetic-app-access-token"
    assert body["refresh_token"] == "synthetic-app-refresh-token"
    assert body["expires_in"] == 1800
    assert body["session_expires_at"] == "2030-01-01T00:00:00Z"
    assert body["token_type"] == "Bearer"
    assert response.headers["cache-control"] == "no-store"
    assert calls == [
        {
            "code": "synthetic-authorization-code",
            "code_verifier": _CODE_VERIFIER,
            "redirect_uri": _REDIRECT_URI,
            "client_id": "synthetic-google-client",
            "client_secret": "synthetic-google-secret",
        }
    ]


def test_google_exchange_accepts_oidc_form_post_from_client_library(
    monkeypatch: pytest.MonkeyPatch, google_settings: Settings
) -> None:
    _fake_exchange(monkeypatch)
    _allow_identity(monkeypatch, "family@example.test")

    with TestClient(create_app(google_settings)) as client:
        response = client.post(
            "/v1/auth/google/exchange",
            data={
                "grant_type": "authorization_code",
                "client_id": "synthetic-google-client",
                "code": "synthetic-authorization-code",
                "code_verifier": _CODE_VERIFIER,
                "redirect_uri": _REDIRECT_URI,
            },
            headers={"Origin": "https://pwa.example.test"},
        )

    assert response.status_code == 200
    assert response.json()["id_token"] == "synthetic.id.token"
    assert response.json()["access_token"] == "synthetic-app-access-token"


def test_google_exchange_rejects_redirects_outside_configured_origins(
    monkeypatch: pytest.MonkeyPatch, google_settings: Settings
) -> None:
    calls = _fake_exchange(monkeypatch)
    _allow_identity(monkeypatch, "family@example.test")

    with TestClient(create_app(google_settings)) as client:
        response = client.post(
            "/v1/auth/google/exchange",
            json={
                "code": "synthetic-authorization-code",
                "code_verifier": _CODE_VERIFIER,
                "redirect_uri": "https://unlisted.example.test/auth/callback",
            },
        )

    assert response.status_code == 400
    assert calls == []


def test_google_exchange_returns_friendly_forbidden_for_unlisted_email(
    monkeypatch: pytest.MonkeyPatch, google_settings: Settings
) -> None:
    _fake_exchange(monkeypatch)
    _allow_identity(monkeypatch, "other@example.test")

    with TestClient(create_app(google_settings)) as client:
        response = _post(client)

    assert response.status_code == 403
    assert response.json() == {
        "error": "access_denied",
        "error_description": "This is a private family app.",
    }


def test_google_exchange_hides_upstream_failures(
    monkeypatch: pytest.MonkeyPatch, google_settings: Settings
) -> None:
    _fake_exchange(monkeypatch, failure=GoogleExchangeFailure("synthetic failure"))

    with TestClient(create_app(google_settings)) as client:
        response = _post(client)

    assert response.status_code == 502
    assert response.json() == {
        "error": "server_error",
        "error_description": "Google sign-in is temporarily unavailable.",
    }


def test_google_exchange_is_rate_limited_per_client(
    monkeypatch: pytest.MonkeyPatch, google_settings: Settings
) -> None:
    _fake_exchange(monkeypatch)
    _allow_identity(monkeypatch, "family@example.test")
    app = create_app(google_settings)
    app.state.auth_rate_limiter = GoogleExchangeRateLimiter(limit=1)

    with TestClient(app) as client:
        first = _post(client)
        second = _post(client)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"]


def test_google_exchange_uses_render_rightmost_forwarded_hop_for_rate_limit(
    monkeypatch: pytest.MonkeyPatch, google_settings: Settings
) -> None:
    _fake_exchange(monkeypatch)
    _allow_identity(monkeypatch, "family@example.test")
    app = create_app(google_settings)
    app.state.auth_rate_limiter = GoogleExchangeRateLimiter(limit=1)

    with TestClient(app) as client:
        first = client.post(
            "/v1/auth/google/exchange",
            json={
                "code": "synthetic-authorization-code",
                "code_verifier": _CODE_VERIFIER,
                "redirect_uri": _REDIRECT_URI,
            },
            headers={
                "Origin": "https://pwa.example.test",
                "X-Forwarded-For": "192.0.2.10, 198.51.100.20",
            },
        )
        second = client.post(
            "/v1/auth/google/exchange",
            json={
                "code": "synthetic-authorization-code",
                "code_verifier": _CODE_VERIFIER,
                "redirect_uri": _REDIRECT_URI,
            },
            headers={
                "Origin": "https://pwa.example.test",
                "X-Forwarded-For": "192.0.2.99, 198.51.100.20",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 429
