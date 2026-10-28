"""OIDC verification and development-token regression tests."""

from __future__ import annotations

import base64
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from wealth.api import auth
from wealth.api.auth import AuthenticationError, JwksCache, verify_token
from wealth.api.main import create_app
from wealth.api.settings import Settings


def _base64url(value: int) -> str:
    byte_count = (value.bit_length() + 7) // 8
    encoded = base64.urlsafe_b64encode(value.to_bytes(byte_count, "big"))
    return encoded.rstrip(b"=").decode("ascii")


@pytest.fixture
def rsa_identity() -> tuple[Any, dict[str, str]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": "synthetic-key",
        "n": _base64url(public_numbers.n),
        "e": _base64url(public_numbers.e),
    }
    return private_key, jwk


@pytest.fixture
def oidc_settings() -> Settings:
    return Settings(
        auth_mode="oidc",
        auth_jwks_url="https://identity.example.test/keys",
        auth_issuer="https://identity.example.test/",
        auth_audience="wealth-api",
    )


def _oidc_token(private_key: Any, kid: str = "synthetic-key", **overrides: Any) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": "synthetic-subject",
        "email": "investor@example.test",
        "name": "Synthetic Investor",
        "iss": "https://identity.example.test/",
        "aud": "wealth-api",
        "iat": now,
        "nbf": now - timedelta(seconds=1),
        "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": kid},
    )


def test_oidc_verifies_identity_claims_with_cached_jwks(
    monkeypatch: pytest.MonkeyPatch,
    rsa_identity: tuple[Any, dict[str, str]],
    oidc_settings: Settings,
) -> None:
    private_key, jwk = rsa_identity
    fetches: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        fetches.append(url)
        return {"keys": [jwk]}

    monkeypatch.setattr(auth, "fetch_jwks", fetch)
    cache = JwksCache()
    token = _oidc_token(private_key)

    first = verify_token(token, oidc_settings, cache)
    second = verify_token(token, oidc_settings, cache)

    assert first.subject == "synthetic-subject"
    assert first.email == "investor@example.test"
    assert first.name == "Synthetic Investor"
    assert second == first
    assert fetches == ["https://identity.example.test/keys"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": datetime.now(UTC) - timedelta(minutes=5)},
        {"iss": "https://wrong-issuer.example.test/"},
        {"aud": "different-api"},
        {"nbf": datetime.now(UTC) + timedelta(minutes=5)},
    ],
    ids=["expired", "wrong-issuer", "wrong-audience", "not-yet-valid"],
)
def test_oidc_rejects_invalid_standard_claims(
    monkeypatch: pytest.MonkeyPatch,
    rsa_identity: tuple[Any, dict[str, str]],
    oidc_settings: Settings,
    overrides: dict[str, Any],
) -> None:
    private_key, jwk = rsa_identity
    monkeypatch.setattr(auth, "fetch_jwks", lambda _url: {"keys": [jwk]})

    with pytest.raises(AuthenticationError):
        verify_token(_oidc_token(private_key, **overrides), oidc_settings, JwksCache())


def test_oidc_refetches_once_when_key_id_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
    rsa_identity: tuple[Any, dict[str, str]],
    oidc_settings: Settings,
) -> None:
    private_key, jwk = rsa_identity
    fetch_count = 0

    def fetch(_url: str) -> dict[str, Any]:
        nonlocal fetch_count
        fetch_count += 1
        return {"keys": [{**jwk, "kid": "another-key"}]}

    monkeypatch.setattr(auth, "fetch_jwks", fetch)

    with pytest.raises(AuthenticationError):
        verify_token(_oidc_token(private_key), oidc_settings, JwksCache())

    assert fetch_count == 2


def test_unknown_oidc_key_returns_http_401_after_one_refetch(
    monkeypatch: pytest.MonkeyPatch,
    rsa_identity: tuple[Any, dict[str, str]],
    oidc_settings: Settings,
) -> None:
    private_key, jwk = rsa_identity
    fetch_count = 0

    def fetch(_url: str) -> dict[str, Any]:
        nonlocal fetch_count
        fetch_count += 1
        return {"keys": [jwk]}

    monkeypatch.setattr(auth, "fetch_jwks", fetch)
    token = _oidc_token(private_key, kid="unknown-key")

    with TestClient(create_app(oidc_settings)) as client:
        response = client.get("/v1/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert fetch_count == 2


def test_oidc_rejects_hs256_tokens_without_fetching_jwks(
    monkeypatch: pytest.MonkeyPatch,
    oidc_settings: Settings,
) -> None:
    secret = secrets.token_bytes(32)
    token = jwt.encode(
        {
            "sub": "synthetic-subject",
            "email": "investor@example.test",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    monkeypatch.setattr(
        auth,
        "fetch_jwks",
        lambda _url: pytest.fail("HS256 tokens must be rejected before key fetching"),
    )

    with pytest.raises(AuthenticationError):
        verify_token(token, oidc_settings, JwksCache())


def test_dev_mode_accepts_only_hs256_with_configured_secret() -> None:
    secret = secrets.token_urlsafe(32)
    settings = Settings(env="test", auth_mode="dev", dev_auth_secret=secret)
    token = jwt.encode(
        {
            "sub": "synthetic-subject",
            "email": "investor@example.test",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )

    assert verify_token(token, settings, JwksCache()).subject == "synthetic-subject"
