"""Environment-backed alert delivery settings without unrelated API auth config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AlertSettings:
    deployment_mode: Literal["personal", "public"] = "personal"
    email_provider: Literal["brevo", "resend", "dry_run"] = "dry_run"
    email_from: str = "alerts@example.invalid"
    email_api_key_value: str | None = None
    email_link_secret_value: str | None = None
    email_outbox_dir: str = "out/outbox"
    email_public_base_url: str = "https://app.example"
    email_manage_alerts_url: str = "https://app.example/settings/alerts"
    email_postal_address: str = "Postal address configured by sender"
    email_disclaimer: str = (
        "Research and education only, not personalised investment advice; "
        "we're not SEBI-registered."
    )
    vapid_public_key_value: str | None = None
    vapid_private_key_value: str | None = None
    vapid_subject: str = "mailto:alerts@example.invalid"

    @classmethod
    def from_env(cls) -> AlertSettings:
        provider = os.getenv("EMAIL_PROVIDER", "dry_run").lower()
        if provider not in {"brevo", "resend", "dry_run"}:
            raise ValueError("EMAIL_PROVIDER must be brevo, resend, or dry_run")
        return cls(
            deployment_mode=os.getenv("DEPLOYMENT_MODE", "personal"),  # type: ignore[arg-type]
            email_provider=provider,  # type: ignore[arg-type]
            email_from=os.getenv("EMAIL_FROM", "alerts@example.invalid"),
            email_api_key_value=os.getenv("EMAIL_API_KEY"),
            email_link_secret_value=os.getenv("EMAIL_LINK_SECRET"),
            email_outbox_dir=os.getenv("EMAIL_OUTBOX_DIR", "out/outbox"),
            email_public_base_url=os.getenv("EMAIL_PUBLIC_BASE_URL", "https://app.example"),
            email_manage_alerts_url=os.getenv(
                "EMAIL_MANAGE_ALERTS_URL", "https://app.example/settings/alerts"
            ),
            email_postal_address=os.getenv(
                "EMAIL_POSTAL_ADDRESS", "Postal address configured by sender"
            ),
            email_disclaimer=os.getenv(
                "EMAIL_DISCLAIMER",
                "Research and education only, not personalised investment advice; "
                "we're not SEBI-registered.",
            ),
            vapid_public_key_value=os.getenv("VAPID_PUBLIC_KEY"),
            vapid_private_key_value=os.getenv("VAPID_PRIVATE_KEY"),
            vapid_subject=os.getenv("VAPID_SUBJECT", "mailto:alerts@example.invalid"),
        )
