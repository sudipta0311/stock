"""Shared helpers for privacy-preserving email delivery and audit records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote
from uuid import UUID

from jinja2 import Environment, select_autoescape

from wealth.alerts.consent import create_unsubscribe_token
from wealth.alerts.email import DeliveryResult, OutgoingEmail, rendered_body_hash, send_email
from wealth.alerts.settings import AlertSettings

_HTML_ENV = Environment(autoescape=select_autoescape(default=True))
_NOTIFICATION_TEMPLATE = _HTML_ENV.from_string(
    '<!doctype html><html lang="en"><body style="font-family:Arial,sans-serif;color:#18212b">'
    '<main style="max-width:600px;margin:auto;padding:24px"><h1>{{ subject }}</h1>'
    '<p>{{ body }}</p><p><a href="{{ manage_url }}">Manage alerts</a> · '
    '<a href="{{ unsubscribe_url }}">Unsubscribe</a></p>'
    "<p>{{ disclaimer }}</p><p>{{ postal_address }}</p></main></body></html>"
)


@dataclass(frozen=True)
class EmailAudit:
    subject: str
    message_id: str
    body_sha256: str
    outbox_path: str | None


def unsubscribe_url(
    settings: AlertSettings,
    user_id: UUID,
    kind: str,
    *,
    now: datetime | None = None,
) -> str:
    secret = settings.email_link_secret_value
    if not secret:
        raise ValueError("EMAIL_LINK_SECRET is required to send email")
    token = create_unsubscribe_token(user_id, kind, secret, now=now)
    return f"{settings.email_public_base_url.rstrip('/')}/v1/email/unsubscribe?token={quote(token)}"


def send_content(
    *,
    settings: AlertSettings,
    user_id: UUID,
    to: str,
    subject: str,
    text: str,
    html: str,
    unsubscribe_kind: str,
    dedupe_key: str,
    dry_run: bool = False,
    now: datetime | None = None,
    unsubscribe_url_value: str | None = None,
) -> EmailAudit:
    link = unsubscribe_url_value or unsubscribe_url(settings, user_id, unsubscribe_kind, now=now)
    outgoing = OutgoingEmail(
        to=to,
        subject=subject,
        html=html,
        text=text,
        from_address=settings.email_from,
        unsubscribe_url=link,
        dedupe_key=dedupe_key,
    )
    result: DeliveryResult = send_email(outgoing, settings, dry_run=dry_run)
    return EmailAudit(
        subject=subject,
        message_id=result.message_id,
        body_sha256=rendered_body_hash(html),
        outbox_path=str(result.outbox_path) if result.outbox_path is not None else None,
    )


def render_notification(
    *,
    settings: AlertSettings,
    subject: str,
    body: str,
    manage_url: str | None = None,
    unsubscribe_url_value: str = "https://app.example/unsubscribe",
) -> tuple[str, str]:
    """Render escaped compact HTML and a matching plain-text notification."""
    html = _NOTIFICATION_TEMPLATE.render(
        subject=subject,
        body=body,
        manage_url=manage_url or settings.email_manage_alerts_url,
        unsubscribe_url=unsubscribe_url_value,
        disclaimer=settings.email_disclaimer,
        postal_address=settings.email_postal_address,
    )
    text = "\n".join(
        (
            subject,
            "",
            body,
            "",
            f"Manage alerts: {manage_url or settings.email_manage_alerts_url}",
            settings.email_disclaimer,
            settings.email_postal_address,
        )
    )
    return html, text
