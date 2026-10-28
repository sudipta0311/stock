"""HTTP email providers and a local, reviewable ``.eml`` dry-run adapter."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import httpx


@dataclass(frozen=True)
class OutgoingEmail:
    to: str
    subject: str
    html: str
    text: str
    from_address: str
    unsubscribe_url: str
    dedupe_key: str = "message"


@dataclass(frozen=True)
class DeliveryResult:
    message_id: str
    outbox_path: Path | None = None


class EmailProvider(Protocol):
    def send(self, message: OutgoingEmail) -> DeliveryResult: ...


def rendered_body_hash(html: str) -> str:
    """Return the audit hash of the exact HTML body sent to the provider."""
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return cleaned[:80] or "message"


class DryRunEmailProvider:
    """Write RFC 5322 messages into a local outbox without sending them."""

    def __init__(self, outbox: Path | str = Path("out/outbox")) -> None:
        self.outbox = Path(outbox)

    def send(self, message: OutgoingEmail) -> DeliveryResult:
        self.outbox.mkdir(parents=True, exist_ok=True)
        email = EmailMessage()
        email["From"] = message.from_address
        email["To"] = message.to
        email["Subject"] = message.subject
        email["Message-ID"] = f"<{uuid4().hex}@wealth-copilot.local>"
        email["List-Unsubscribe"] = f"<{message.unsubscribe_url}>"
        email["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
        email.set_content(message.text)
        email.add_alternative(message.html, subtype="html")
        path = self.outbox / f"{_safe_filename(message.dedupe_key)}_{uuid4().hex[:8]}.eml"
        path.write_bytes(email.as_bytes())
        return DeliveryResult(message_id=email["Message-ID"], outbox_path=path)


class _HttpEmailProvider:
    endpoint: str

    def __init__(
        self, api_key: str, *, client: httpx.Client | None = None, attempts: int = 3
    ) -> None:
        self.api_key = api_key
        self.client = client
        self.attempts = attempts

    def _headers(self, message: OutgoingEmail) -> dict[str, str]:
        raise NotImplementedError

    def _payload(self, message: OutgoingEmail) -> dict[str, object]:
        raise NotImplementedError

    def _message_id(self, response: httpx.Response) -> str:
        raise NotImplementedError

    def send(self, message: OutgoingEmail) -> DeliveryResult:
        owned_client = self.client is None
        client = self.client or httpx.Client(timeout=20)
        try:
            for attempt in range(self.attempts):
                try:
                    response = client.post(
                        self.endpoint,
                        headers=self._headers(message),
                        json=self._payload(message),
                    )
                except httpx.HTTPError as error:
                    if attempt + 1 >= self.attempts:
                        raise RuntimeError("Email provider request failed") from error
                    time.sleep(min(2**attempt, 4))
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 >= self.attempts:
                        raise RuntimeError(
                            f"Email provider request failed with status {response.status_code}"
                        )
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = min(max(float(retry_after or 2**attempt), 0), 4)
                    except ValueError:
                        delay = min(2**attempt, 4)
                    time.sleep(delay)
                    continue
                if response.is_error:
                    raise RuntimeError(
                        f"Email provider request failed with status {response.status_code}"
                    )
                return DeliveryResult(message_id=self._message_id(response))
            raise RuntimeError("Email provider request failed")
        finally:
            if owned_client:
                client.close()


class BrevoEmailProvider(_HttpEmailProvider):
    endpoint = "https://api.brevo.com/v3/smtp/email"

    def _headers(self, message: OutgoingEmail) -> dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _payload(self, message: OutgoingEmail) -> dict[str, object]:
        name, address = parseaddr(message.from_address)
        sender: dict[str, str] = {"email": address}
        if name:
            sender["name"] = name
        return {
            "sender": sender,
            "to": [{"email": message.to}],
            "subject": message.subject,
            "htmlContent": message.html,
            "textContent": message.text,
            "headers": {
                "List-Unsubscribe": f"<{message.unsubscribe_url}>",
                "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
            },
        }

    def _message_id(self, response: httpx.Response) -> str:
        payload = response.json()
        return str(payload.get("messageId") or "brevo-accepted")


class ResendEmailProvider(_HttpEmailProvider):
    endpoint = "https://api.resend.com/emails"

    def _headers(self, message: OutgoingEmail) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, message: OutgoingEmail) -> dict[str, object]:
        return {
            "from": message.from_address,
            "to": [message.to],
            "subject": message.subject,
            "html": message.html,
            "text": message.text,
            "headers": {
                "List-Unsubscribe": f"<{message.unsubscribe_url}>",
                "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
            },
        }

    def _message_id(self, response: httpx.Response) -> str:
        payload = response.json()
        return str(payload.get("id") or "resend-accepted")


def email_provider(
    name: str,
    api_key: str | None = None,
    *,
    outbox: Path | str = Path("out/outbox"),
    client: httpx.Client | None = None,
) -> EmailProvider:
    """Build the selected adapter from validated environment settings."""
    if name == "dry_run":
        return DryRunEmailProvider(outbox)
    if not api_key:
        raise ValueError("EMAIL_API_KEY is required for the selected provider")
    if name == "brevo":
        return BrevoEmailProvider(api_key, client=client)
    if name == "resend":
        return ResendEmailProvider(api_key, client=client)
    raise ValueError("EMAIL_PROVIDER must be brevo, resend, or dry_run")


def send_email(
    message: OutgoingEmail,
    settings: object,
    *,
    dry_run: bool = False,
    provider: EmailProvider | None = None,
) -> DeliveryResult:
    """Send or locally preview one email, always using the one-click headers."""
    selected = provider or email_provider(
        "dry_run" if dry_run else str(getattr(settings, "email_provider", "dry_run")),
        getattr(settings, "email_api_key_value", None),
        outbox=getattr(settings, "email_outbox_dir", Path("out/outbox")),
    )
    return selected.send(message)
