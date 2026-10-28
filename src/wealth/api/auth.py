"""OIDC and local development token verification."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.request import urlopen

import jwt
from jwt import PyJWK

from wealth.api.settings import Settings

_ALLOWED_OIDC_ALGORITHMS = {"RS256", "ES256"}
_JWKS_TTL_SECONDS = 60 * 60
_CLOCK_LEEWAY_SECONDS = 60


class AuthenticationError(Exception):
    """Raised when a bearer token cannot be trusted."""


@dataclass(frozen=True)
class AuthClaims:
    subject: str
    email: str
    name: str | None


def fetch_jwks(url: str) -> dict[str, Any]:
    """Fetch a bounded JWKS response from the configured identity provider."""
    with urlopen(url, timeout=5) as response:  # noqa: S310 - URL is explicit trusted config.
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("JWKS response is too large")
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
        raise ValueError("JWKS response has an invalid shape")
    return document


class JwksCache:
    """Cache OIDC signing keys for one hour and refresh once for an unknown key id."""

    def __init__(self, ttl_seconds: int = _JWKS_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._url: str | None = None
        self._keys: list[dict[str, Any]] = []
        self._fetched_at = 0.0
        self._lock = Lock()

    def signing_key(self, url: str, kid: str, algorithm: str) -> PyJWK:
        with self._lock:
            if self._url != url or time.monotonic() - self._fetched_at >= self._ttl_seconds:
                self._refresh(url)
            jwk = self._find_key(kid)
            if jwk is None:
                self._refresh(url)
                jwk = self._find_key(kid)
            if jwk is None:
                raise AuthenticationError("Unknown signing key")
            if jwk.get("alg") not in (None, algorithm):
                raise AuthenticationError("Signing key algorithm does not match the token")
            try:
                return PyJWK.from_dict(jwk, algorithm=algorithm)
            except (jwt.PyJWTError, TypeError, ValueError) as error:
                raise AuthenticationError("Invalid signing key") from error

    def _refresh(self, url: str) -> None:
        document = fetch_jwks(url)
        keys = document.get("keys")
        if not isinstance(keys, list) or any(not isinstance(key, dict) for key in keys):
            raise AuthenticationError("Invalid JWKS document")
        self._url = url
        self._keys = keys
        self._fetched_at = time.monotonic()

    def _find_key(self, kid: str) -> dict[str, Any] | None:
        for key in self._keys:
            if key.get("kid") == kid:
                return key
        return None


def verify_token(token: str, settings: Settings, jwks_cache: JwksCache) -> AuthClaims:
    """Verify signature, time bounds, issuer, audience, and identity claims."""
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if settings.auth_mode == "dev":
            secret = settings.dev_auth_secret_value
            if not secret or algorithm != "HS256":
                raise AuthenticationError("Invalid development token")
            claims = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                leeway=_CLOCK_LEEWAY_SECONDS,
                options={"require": ["sub", "email", "exp"]},
            )
        else:
            if algorithm not in _ALLOWED_OIDC_ALGORITHMS:
                raise AuthenticationError("Unsupported token algorithm")
            kid = header.get("kid")
            if not isinstance(kid, str) or not kid:
                raise AuthenticationError("Token has no key id")
            if not settings.auth_jwks_url or not settings.auth_issuer or not settings.auth_audience:
                raise AuthenticationError("OIDC is not configured")
            key = jwks_cache.signing_key(settings.auth_jwks_url, kid, algorithm)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=[algorithm],
                audience=settings.auth_audience,
                issuer=settings.auth_issuer,
                leeway=_CLOCK_LEEWAY_SECONDS,
                options={"require": ["sub", "email", "exp"]},
            )
    except AuthenticationError:
        raise
    except (jwt.PyJWTError, OSError, TypeError, ValueError, KeyError) as error:
        raise AuthenticationError("Invalid bearer token") from error

    subject = claims.get("sub")
    email = claims.get("email")
    name = claims.get("name")
    if (
        not isinstance(subject, str)
        or not subject.strip()
        or len(subject) > 255
        or not isinstance(email, str)
        or not email.strip()
        or len(email) > 320
        or (name is not None and not isinstance(name, str))
        or (isinstance(name, str) and len(name) > 120)
    ):
        raise AuthenticationError("Invalid identity claims")
    return AuthClaims(subject=subject, email=email, name=name)
