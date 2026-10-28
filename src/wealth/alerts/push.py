"""Web Push subscription validation and VAPID delivery helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from pydantic import AnyUrl, BaseModel, ConfigDict, field_validator

_EXACT_PUSH_HOSTS = frozenset({"fcm.googleapis.com", "web.push.apple.com"})
_PUSH_HOST_SUFFIXES = (".push.services.mozilla.com",)


class PushSubscriptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    endpoint: AnyUrl
    keys: dict[str, str]

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: AnyUrl) -> AnyUrl:
        validate_push_endpoint(str(value))
        return value

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, value: dict[str, str]) -> dict[str, str]:
        if not value.get("p256dh") or not value.get("auth"):
            raise ValueError("push keys must include p256dh and auth")
        if set(value) - {"p256dh", "auth"}:
            raise ValueError("push keys contain unsupported fields")
        return value


class PushDeliveryGone(Exception):
    """The push service has permanently expired a subscription."""


def validate_push_endpoint(endpoint: str) -> None:
    """Accept HTTPS endpoints hosted by known browser push services only."""
    try:
        parsed = urlsplit(endpoint)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError as error:
        raise ValueError("push endpoint must be a valid HTTPS push service URL") from error
    known_host = hostname in _EXACT_PUSH_HOSTS or any(
        hostname.endswith(suffix) for suffix in _PUSH_HOST_SUFFIXES
    )
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or not known_host
        or port not in {None, 443}
    ):
        raise ValueError("push endpoint host is not a supported push service")


def _default_webpush(
    *,
    subscription_info: Mapping[str, Any],
    data: str,
    vapid_private_key: str,
    vapid_claims: dict[str, str],
) -> Any:
    from pywebpush import webpush

    return webpush(
        subscription_info=dict(subscription_info),
        data=data,
        vapid_private_key=vapid_private_key,
        vapid_claims=vapid_claims,
    )


def send_push(
    subscription: Mapping[str, Any],
    payload: Mapping[str, Any],
    settings: object,
    *,
    webpush: Callable[..., Any] | None = None,
) -> str:
    """Send one Web Push payload and surface 404/410 as a cleanup signal."""
    public_key = getattr(settings, "vapid_public_key_value", None)
    private_key = getattr(settings, "vapid_private_key_value", None)
    if not public_key or not private_key:
        raise ValueError("VAPID keys are required for push delivery")
    endpoint = str(subscription.get("endpoint", ""))
    validate_push_endpoint(endpoint)
    sender = webpush or _default_webpush
    try:
        result = sender(
            subscription_info={
                "endpoint": endpoint,
                "keys": dict(subscription.get("keys", {})),
            },
            data=json.dumps(payload, separators=(",", ":")),
            vapid_private_key=private_key,
            vapid_claims={
                "sub": str(getattr(settings, "vapid_subject", "mailto:alerts@example.invalid"))
            },
        )
    except Exception as error:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {404, 410}:
            raise PushDeliveryGone from error
        raise
    return str(getattr(result, "status_code", "push-accepted"))
