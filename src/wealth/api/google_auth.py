"""Server-side Google authorization-code exchange for the public PWA."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wealth.api.auth import AuthenticationError, JwksCache, verify_token
from wealth.api.auth_rate_limit import AuthRateLimiter, client_address
from wealth.api.auth_sessions import AuthSessionPair, SessionConfigurationError, create_app_session
from wealth.api.settings import Settings

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

GoogleExchangeRateLimiter = AuthRateLimiter


class GoogleExchangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: str = Field(min_length=1, max_length=4096)
    code_verifier: str = Field(
        min_length=43,
        max_length=128,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )
    redirect_uri: str = Field(min_length=1, max_length=2048)


class GoogleExchangeResponse(AuthSessionPair):
    id_token: str


class GoogleExchangeFailure(Exception):
    """An upstream Google token exchange failed without safe details."""


class InvalidGoogleCode(Exception):
    """Google rejected an expired, invalid, or already used code."""


async def _fetch_google_token(
    *, code: str, code_verifier: str, redirect_uri: str, client_id: str, client_secret: str
) -> tuple[str, int]:
    """Exchange one PKCE code and discard every upstream field except the ID token and expiry."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                _GOOGLE_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "code_verifier": code_verifier,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
        if response.status_code in {400, 401}:
            raise InvalidGoogleCode
        response.raise_for_status()
        payload = response.json()
    except InvalidGoogleCode:
        raise
    except httpx.HTTPStatusError as error:
        raise GoogleExchangeFailure from error
    except (httpx.HTTPError, ValueError) as error:
        raise GoogleExchangeFailure from error

    id_token = payload.get("id_token") if isinstance(payload, dict) else None
    expires_in = payload.get("expires_in") if isinstance(payload, dict) else None
    if (
        not isinstance(id_token, str)
        or not id_token
        or len(id_token) > 16_384
        or isinstance(expires_in, bool)
        or not isinstance(expires_in, int)
        or not 1 <= expires_in <= 86_400
    ):
        raise GoogleExchangeFailure
    return id_token, expires_in


def _origin(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    hostname = parsed.hostname
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    host = hostname.casefold()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None and port not in ({80} if parsed.scheme == "http" else {443}):
        host = f"{host}:{port}"
    return f"{parsed.scheme.casefold()}://{host}"


def _redirect_origin(redirect_uri: str, settings: Settings) -> str | None:
    try:
        parsed = urlsplit(redirect_uri)
    except ValueError:
        return None
    if parsed.path != "/auth/callback" or parsed.query or parsed.fragment or not parsed.netloc:
        return None
    origin = _origin(redirect_uri)
    allowed_origins = {_origin(item) for item in settings.origin_list}
    return origin if origin is not None and origin in allowed_origins else None


def _oauth_error(status: int, error: str, description: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": error, "error_description": description},
        headers={"Cache-Control": "no-store"},
    )


async def _parse_request_body(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip()
    if content_type == "application/json":
        payload = await request.json()
        return payload if isinstance(payload, dict) else {}
    if content_type == "application/x-www-form-urlencoded":
        payload = await request.form()
        return dict(payload)
    raise HTTPException(status_code=415, detail="Use a JSON or form-encoded request.")


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/auth/google/exchange",
        response_model=GoogleExchangeResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": GoogleExchangeRequest.model_json_schema()},
                    "application/x-www-form-urlencoded": {
                        "schema": GoogleExchangeRequest.model_json_schema()
                    },
                },
            }
        },
    )
    async def exchange_google_code(
        request: Request, response: Response
    ) -> GoogleExchangeResponse | JSONResponse:
        settings: Settings = request.app.state.settings
        limiter = getattr(request.app.state, "auth_rate_limiter", None) or getattr(
            request.app.state, "google_exchange_limiter", None
        )
        retry_after = limiter.retry_after(client_address(request)) if limiter is not None else None
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Please wait before trying again.",
                headers={"Retry-After": str(retry_after)},
            )

        if settings.auth_mode != "oidc":
            raise HTTPException(status_code=404, detail="Google sign-in is not enabled.")
        if not settings.google_client_id or not settings.google_client_secret_value:
            raise HTTPException(status_code=503, detail="Google sign-in is not configured.")

        raw_payload = await _parse_request_body(request)
        payload_values = dict(raw_payload)
        grant_type = payload_values.pop("grant_type", None)
        client_id = payload_values.pop("client_id", None)
        if grant_type not in {None, "authorization_code"} or client_id not in {
            None,
            settings.google_client_id,
        }:
            raise HTTPException(status_code=400, detail="The sign-in request is invalid.")
        try:
            payload = GoogleExchangeRequest.model_validate(payload_values)
        except ValidationError as error:
            raise HTTPException(
                status_code=422, detail="The sign-in request is invalid."
            ) from error

        redirect_origin = _redirect_origin(payload.redirect_uri, settings)
        if redirect_origin is None:
            raise HTTPException(
                status_code=400, detail="The sign-in return address is not allowed."
            )
        request_origin = request.headers.get("origin")
        if request_origin is not None and _origin(request_origin) != redirect_origin:
            raise HTTPException(
                status_code=400, detail="The sign-in return address is not allowed."
            )

        try:
            id_token, _google_expires_in = await _fetch_google_token(
                code=payload.code,
                code_verifier=payload.code_verifier,
                redirect_uri=payload.redirect_uri,
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret_value,
            )
        except InvalidGoogleCode:
            return _oauth_error(401, "invalid_grant", "The sign-in code expired. Please try again.")
        except GoogleExchangeFailure:
            return _oauth_error(502, "server_error", "Google sign-in is temporarily unavailable.")

        if request.app.state.jwks_cache is None:
            request.app.state.jwks_cache = JwksCache()
        try:
            claims = verify_token(id_token, settings, request.app.state.jwks_cache)
        except AuthenticationError:
            return _oauth_error(401, "invalid_grant", "Google returned an invalid sign-in token.")
        allowed_emails = settings.allowed_email_list
        if allowed_emails and claims.email.casefold() not in allowed_emails:
            return _oauth_error(403, "access_denied", "This is a private family app.")
        engine = request.app.state.db_engine
        if engine is None:
            raise HTTPException(status_code=503, detail="The data service is unavailable.")
        try:
            pair = create_app_session(
                engine=engine,
                settings=settings,
                claims=claims,
                user_agent=request.headers.get("user-agent"),
            )
        except SessionConfigurationError as error:
            raise HTTPException(status_code=503, detail="Sign-in is not configured.") from error
        response.headers["Cache-Control"] = "no-store"
        return GoogleExchangeResponse(id_token=id_token, **pair.model_dump())

    return router
