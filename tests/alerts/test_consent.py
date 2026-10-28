"""HMAC unsubscribe-link behaviour and supported consent purposes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from wealth.alerts.consent import (
    CONSENT_PURPOSES,
    create_unsubscribe_token,
    verify_unsubscribe_token,
)

_USER_ID = UUID("00000000-0000-0000-0000-000000000017")
_NOW = datetime(2026, 10, 28, tzinfo=UTC)
_SECRET = "synthetic-test-link-secret"


def test_unsubscribe_token_is_valid_for_sixty_days() -> None:
    token = create_unsubscribe_token(_USER_ID, "weekly_email", _SECRET, now=_NOW)
    claims = verify_unsubscribe_token(token, _SECRET, now=_NOW + timedelta(days=59))
    assert claims is not None
    assert claims.user_id == _USER_ID
    assert claims.kind == "weekly_email"
    assert claims.expires_at == _NOW + timedelta(days=60)


def test_unsubscribe_token_rejects_expiry_and_tampering() -> None:
    token = create_unsubscribe_token(_USER_ID, "reminder_email", _SECRET, now=_NOW)
    assert verify_unsubscribe_token(token, _SECRET, now=_NOW + timedelta(days=60)) is None
    encoded, signature = token.split(".")
    changed = f"{encoded[:-1]}{'A' if encoded[-1] != 'A' else 'B'}.{signature}"
    assert verify_unsubscribe_token(changed, _SECRET, now=_NOW) is None


def test_unsubscribe_token_rejects_unknown_purpose() -> None:
    with pytest.raises(ValueError):
        create_unsubscribe_token(_USER_ID, "push", _SECRET, now=_NOW)
    assert CONSENT_PURPOSES == ("weekly_email", "reminder_email", "push", "urgent_alerts")
