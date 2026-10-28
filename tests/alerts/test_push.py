"""Host allowlist and gone-subscription behaviour for Web Push."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wealth.alerts.push import (
    PushDeliveryGone,
    PushSubscriptionInput,
    send_push,
    validate_push_endpoint,
)
from wealth.alerts.settings import AlertSettings


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://fcm.googleapis.com/fcm/send/synthetic-key",
        "https://updates.push.services.mozilla.com/wpush/v2/synthetic-key",
        "https://web.push.apple.com/synthetic-key",
    ),
)
def test_known_push_service_hosts_are_accepted(endpoint: str) -> None:
    validate_push_endpoint(endpoint)
    subscription = PushSubscriptionInput.model_validate(
        {"endpoint": endpoint, "keys": {"p256dh": "synthetic-p256dh", "auth": "synthetic-auth"}}
    )
    assert str(subscription.endpoint).startswith("https://")


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://fcm.googleapis.com/fcm/send/key",
        "https://fcm.googleapis.com.attacker.example/send/key",
        "https://attacker.example/push/key",
    ),
)
def test_unknown_or_insecure_push_hosts_are_rejected(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_push_endpoint(endpoint)


def test_push_service_410_is_signalled_for_subscription_cleanup() -> None:
    settings = AlertSettings(
        email_provider="resend",
        vapid_public_key_value="synthetic-public-key",
        vapid_private_key_value="synthetic-private-key",
    )

    class Expired(Exception):
        response = SimpleNamespace(status_code=410)

    def expired(**_kwargs):
        raise Expired("gone")

    with pytest.raises(PushDeliveryGone):
        send_push(
            {
                "endpoint": "https://fcm.googleapis.com/fcm/send/synthetic-key",
                "keys": {"p256dh": "synthetic-p256dh", "auth": "synthetic-auth"},
            },
            {"title": "Synthetic", "body": "Synthetic body"},
            settings,
            webpush=expired,
        )
