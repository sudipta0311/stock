"""App access tokens, rotating refresh tokens, and device-session routes."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text, update
from sqlalchemy.exc import SQLAlchemyError

from wealth.api.auth import AuthClaims
from wealth.api.auth_rate_limit import client_address
from wealth.api.settings import Settings
from wealth.db.models import AuditLog, AuthSession, User
from wealth.db.session import resolve_user, user_session

ACCESS_TOKEN_TTL_SECONDS = 30 * 60
SLIDING_SESSION_DAYS = 30
ABSOLUTE_SESSION_DAYS = 90
# Another tab, or a refresh whose response was lost, can present the token rotated a
# moment ago. Inside this window that is not treated as theft.
REUSE_GRACE = timedelta(seconds=30)
_REFRESH_TOKEN_BYTES = 32


class AuthSessionPair(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    session_expires_at: datetime
    token_type: str = "Bearer"


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    refresh_token: str = Field(min_length=43, max_length=43, pattern=r"^[A-Za-z0-9_-]{43}$")


class SignedInSession(BaseModel):
    id: UUID
    device_label: str
    created_at: datetime
    last_used_at: datetime
    current: bool


class SessionConfigurationError(Exception):
    """The API cannot issue an app session with its current configuration."""


def refresh_token_hash(refresh_token: str) -> str:
    """Hash one opaque refresh token before it reaches the database."""
    return hashlib.sha256(refresh_token.encode("ascii")).hexdigest()


def _new_refresh_token() -> str:
    return secrets.token_urlsafe(_REFRESH_TOKEN_BYTES)


def _session_secret(settings: Settings) -> str:
    secret = settings.app_session_secret_value
    if secret is None or len(secret.encode("utf-8")) < 32:
        raise SessionConfigurationError
    return secret


def _access_token(
    *,
    settings: Settings,
    subject: str,
    email: str,
    session_id: UUID,
    now: datetime,
) -> str:
    expires_at = now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS)
    claims = {
        "sub": subject,
        "email": email,
        "sid": str(session_id),
        "typ": "access",
        "iss": "wealth-api",
        "aud": "wealth-pwa",
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(claims, _session_secret(settings), algorithm="HS256")


def _device_label(user_agent: str | None) -> str:
    """Return a short device-family label without retaining the full user agent."""
    value = (user_agent or "").casefold()
    if "ipad" in value:
        return "iPad"
    if "iphone" in value or "ipod" in value:
        return "iPhone"
    if "android" in value:
        return "Android device"
    if "windows" in value:
        return "Windows PC"
    if "macintosh" in value or "mac os" in value:
        return "Mac"
    if "linux" in value:
        return "Linux device"
    return "Unknown device"


def create_app_session(
    *,
    engine: Any,
    settings: Settings,
    claims: AuthClaims,
    user_agent: str | None,
) -> AuthSessionPair:
    """Create a per-login session and return its access and one-time refresh tokens."""
    _session_secret(settings)
    user_id = resolve_user(engine, claims.subject, claims.email, claims.name)
    now = datetime.now(UTC)
    session_id = uuid4()
    refresh_token = _new_refresh_token()
    absolute_expires_at = now + timedelta(days=ABSOLUTE_SESSION_DAYS)
    expires_at = min(now + timedelta(days=SLIDING_SESSION_DAYS), absolute_expires_at)
    with user_session(engine, user_id) as session:
        session.add(
            AuthSession(
                id=session_id,
                user_id=user_id,
                token_hash=refresh_token_hash(refresh_token),
                created_at=now,
                last_used_at=now,
                expires_at=expires_at,
                absolute_expires_at=absolute_expires_at,
                device_label=_device_label(user_agent),
            )
        )
        session.flush()
    return AuthSessionPair(
        access_token=_access_token(
            settings=settings,
            subject=claims.subject,
            email=claims.email,
            session_id=session_id,
            now=now,
        ),
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_TTL_SECONDS,
        session_expires_at=absolute_expires_at,
    )


def _app_limiter(request: Request) -> Any:
    return getattr(request.app.state, "auth_rate_limiter", None) or getattr(
        request.app.state, "google_exchange_limiter", None
    )


def _apply_rate_limit(request: Request) -> None:
    limiter = _app_limiter(request)
    if limiter is None:
        return
    retry_after = limiter.retry_after(client_address(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Please wait before trying again.",
            headers={"Retry-After": str(retry_after)},
        )


def _lookup_refresh_session(engine: Any, token_hash: str) -> tuple[UUID, UUID] | None:
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT session_id, user_id FROM app_lookup_refresh_session(:token_hash)"),
            {"token_hash": token_hash},
        ).one_or_none()
    if row is None:
        return None
    return UUID(str(row.session_id)), UUID(str(row.user_id))


def _is_email_allowed(settings: Settings, email: str) -> bool:
    allowed = settings.allowed_email_list
    return not allowed or email.casefold() in allowed


def build_router(authenticated_user: Any) -> APIRouter:
    router = APIRouter()
    user_dependency = Depends(authenticated_user)

    @router.post(
        "/v1/auth/refresh",
        response_model=AuthSessionPair,
        responses={
            401: {"description": "The refresh token is invalid or expired."},
            409: {"description": "The token was renewed moments ago; use the newer one."},
        },
    )
    def refresh_session(
        payload: RefreshRequest,
        request: Request,
        response: Response,
    ) -> AuthSessionPair:
        _apply_rate_limit(request)
        settings: Settings = request.app.state.settings
        engine = request.app.state.db_engine
        if engine is None:
            raise HTTPException(status_code=503, detail="The data service is unavailable.")

        presented_hash = refresh_token_hash(payload.refresh_token)
        try:
            lookup = _lookup_refresh_session(engine, presented_hash)
            if lookup is None:
                raise HTTPException(status_code=401, detail="The sign-in session has expired.")
            session_id, user_id = lookup
            now = datetime.now(UTC)
            result: AuthSessionPair | None = None
            rejection_status: int | None = None
            with user_session(engine, user_id) as session:
                record = session.scalar(
                    select(AuthSession)
                    .where(AuthSession.id == session_id, AuthSession.user_id == user_id)
                    .with_for_update()
                )
                user = session.get(User, user_id)
                if record is None or user is None:
                    rejection_status = 401
                elif (
                    record.previous_hash == presented_hash
                    and record.revoked_at is None
                    and now - record.last_used_at <= REUSE_GRACE
                ):
                    rejection_status = 409
                elif record.previous_hash == presented_hash:
                    if record.revoked_at is None:
                        record.revoked_at = now
                        session.add(
                            AuditLog(
                                user_id=user_id,
                                actor_user_id=user_id,
                                actor="auth",
                                action="auth_refresh_reuse_detected",
                                detail={"session_id": str(record.id)},
                            )
                        )
                    rejection_status = 401
                elif record.token_hash != presented_hash or (
                    record.revoked_at is not None
                    or record.expires_at <= now
                    or record.absolute_expires_at <= now
                ):
                    rejection_status = 401
                elif not _is_email_allowed(settings, user.email):
                    record.revoked_at = now
                    rejection_status = 403
                else:
                    new_refresh_token = _new_refresh_token()
                    next_expiry = min(
                        now + timedelta(days=SLIDING_SESSION_DAYS), record.absolute_expires_at
                    )
                    record.previous_hash = record.token_hash
                    record.token_hash = refresh_token_hash(new_refresh_token)
                    record.last_used_at = now
                    record.expires_at = next_expiry
                    session.flush()
                    result = AuthSessionPair(
                        access_token=_access_token(
                            settings=settings,
                            subject=user.auth_subject,
                            email=user.email,
                            session_id=record.id,
                            now=now,
                        ),
                        refresh_token=new_refresh_token,
                        expires_in=ACCESS_TOKEN_TTL_SECONDS,
                        session_expires_at=record.absolute_expires_at,
                    )
        except HTTPException:
            raise
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="The data service is unavailable."
            ) from error

        if rejection_status is not None:
            detail = {
                403: "This is a private family app.",
                409: "The sign-in session was just renewed.",
            }.get(rejection_status, "The sign-in session has expired.")
            raise HTTPException(status_code=rejection_status, detail=detail)
        if result is None:
            raise HTTPException(status_code=401, detail="The sign-in session has expired.")
        response.headers["Cache-Control"] = "no-store"
        return result

    @router.post("/v1/auth/logout", status_code=204, response_class=Response)
    def logout_session(user: Any = user_dependency) -> Response:
        if user.session_id is not None:
            record = user.session.get(AuthSession, user.session_id)
            if record is not None and record.user_id == user.user_id and record.revoked_at is None:
                record.revoked_at = datetime.now(UTC)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @router.post("/v1/auth/logout-all", status_code=204, response_class=Response)
    def logout_all_sessions(user: Any = user_dependency) -> Response:
        user.session.execute(
            update(AuthSession)
            .where(AuthSession.user_id == user.user_id, AuthSession.revoked_at.is_(None))
            .values(revoked_at=datetime.now(UTC))
        )
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @router.get("/v1/auth/sessions", response_model=list[SignedInSession])
    def list_sessions(
        user: Any = user_dependency,
    ) -> list[SignedInSession]:
        now = datetime.now(UTC)
        records = user.session.scalars(
            select(AuthSession)
            .where(
                AuthSession.user_id == user.user_id,
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > now,
                AuthSession.absolute_expires_at > now,
            )
            .order_by(AuthSession.last_used_at.desc(), AuthSession.created_at.desc())
        )
        return [
            SignedInSession(
                id=record.id,
                device_label=record.device_label,
                created_at=record.created_at,
                last_used_at=record.last_used_at,
                current=record.id == user.session_id,
            )
            for record in records
        ]

    return router
