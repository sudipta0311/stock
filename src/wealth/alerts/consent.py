"""Purpose-specific consent state and signed one-click unsubscribe links."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from wealth.db.models import Consent

CONSENT_PURPOSES = ("weekly_email", "reminder_email", "push", "urgent_alerts")
UNSUBSCRIBE_LIFETIME = timedelta(days=60)


@dataclass(frozen=True)
class UnsubscribeClaims:
    user_id: UUID
    kind: str
    expires_at: datetime


def consent_is_enabled(session: Session, user_id: UUID, purpose: str) -> bool:
    """Missing consent rows are opt-out by default."""
    if purpose not in CONSENT_PURPOSES:
        return False
    consent = session.scalar(
        select(Consent)
        .where(Consent.user_id == user_id, Consent.purpose == purpose)
        .order_by(Consent.granted_at.desc(), Consent.id.desc())
        .limit(1)
    )
    return consent is not None and consent.withdrawn_at is None


def set_consent(
    session: Session, user_id: UUID, purpose: str, enabled: bool, *, now: datetime | None = None
) -> None:
    """Store an explicit consent transition for one supported purpose."""
    if purpose not in CONSENT_PURPOSES:
        raise ValueError("unsupported consent purpose")
    if not isinstance(enabled, bool):
        raise ValueError("consent state must be boolean")
    changed_at = now or datetime.now(UTC)
    consent = session.scalar(
        select(Consent)
        .where(Consent.user_id == user_id, Consent.purpose == purpose)
        .order_by(Consent.granted_at.desc(), Consent.id.desc())
        .limit(1)
    )
    if consent is None:
        if not enabled:
            return
        consent = Consent(user_id=user_id, purpose=purpose, granted_at=changed_at)
        session.add(consent)
    elif enabled:
        if consent.withdrawn_at is not None:
            consent.granted_at = changed_at
            consent.withdrawn_at = None
    else:
        consent.withdrawn_at = changed_at
    session.flush()


def consent_document(session: Session, user_id: UUID) -> dict[str, bool]:
    return {purpose: consent_is_enabled(session, user_id, purpose) for purpose in CONSENT_PURPOSES}


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_unsubscribe_token(
    user_id: UUID,
    kind: str,
    secret: str,
    *,
    now: datetime | None = None,
) -> str:
    """Create an HMAC-SHA256 link token with a sixty-day lifetime."""
    if kind not in {"weekly_email", "reminder_email", "urgent_alerts"}:
        raise ValueError("unsupported unsubscribe kind")
    if not secret:
        raise ValueError("EMAIL_LINK_SECRET is required")
    issued_at = now or datetime.now(UTC)
    expires_at = int((issued_at + UNSUBSCRIBE_LIFETIME).timestamp())
    payload = json.dumps(
        {"user_id": str(user_id), "kind": kind, "exp": expires_at},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = _encode(payload)
    signature = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_encode(signature)}"


def verify_unsubscribe_token(
    token: str, secret: str, *, now: datetime | None = None
) -> UnsubscribeClaims | None:
    """Verify signature, expiry, user id, and purpose without exposing user data."""
    if not secret:
        return None
    try:
        encoded, signature_text = token.split(".", maxsplit=1)
        provided_signature = _decode(signature_text)
        expected_signature = hmac.new(
            secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(provided_signature, expected_signature):
            return None
        payload = json.loads(_decode(encoded))
        user_id = UUID(payload["user_id"])
        kind = str(payload["kind"])
        expiry = datetime.fromtimestamp(int(payload["exp"]), UTC)
        if kind not in {"weekly_email", "reminder_email", "urgent_alerts"}:
            return None
        current = now or datetime.now(UTC)
        if expiry <= current:
            return None
        return UnsubscribeClaims(user_id=user_id, kind=kind, expires_at=expiry)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def withdraw_consent(
    session: Session, user_id: UUID, purpose: str, *, now: datetime | None = None
) -> None:
    """Withdraw a previously granted purpose without revealing the user's address."""
    if purpose not in {"weekly_email", "reminder_email", "urgent_alerts"}:
        raise ValueError("unsupported unsubscribe purpose")
    set_consent(session, user_id, purpose, False, now=now)
