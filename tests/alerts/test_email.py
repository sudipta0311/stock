"""Dry-run outbox and retry behaviour for the HTTP email adapters."""

from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path

import httpx

from wealth.alerts.email import (
    BrevoEmailProvider,
    DryRunEmailProvider,
    OutgoingEmail,
    ResendEmailProvider,
)


def _message() -> OutgoingEmail:
    return OutgoingEmail(
        to="synthetic@example.test",
        subject="Synthetic weekly brief",
        html="<html><body>Brief body</body></html>",
        text="Brief body",
        from_address="alerts@example.invalid",
        unsubscribe_url="https://app.example/v1/email/unsubscribe?token=synthetic-token",
        dedupe_key="weekly_2026-W44",
    )


def test_dry_run_writes_eml_with_unsubscribe_headers_and_both_body_parts(tmp_path: Path) -> None:
    result = DryRunEmailProvider(tmp_path / "outbox").send(_message())
    assert result.outbox_path is not None
    parsed = BytesParser(policy=policy.default).parsebytes(result.outbox_path.read_bytes())

    assert parsed["To"] == "synthetic@example.test"
    assert parsed["List-Unsubscribe"] == (
        "<https://app.example/v1/email/unsubscribe?token=synthetic-token>"
    )
    assert parsed["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    assert parsed.get_content_type() == "multipart/alternative"
    assert parsed.get_body(preferencelist=("plain",)).get_content().strip() == "Brief body"
    assert "Brief body" in parsed.get_body(preferencelist=("html",)).get_content()


def test_http_providers_retry_rate_limit_and_return_provider_ids(monkeypatch) -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}, json={"message": "retry"}),
        httpx.Response(200, json={"messageId": "synthetic-brevo-id"}),
    ]
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.pop(0)

    monkeypatch.setattr("wealth.alerts.email.time.sleep", lambda _seconds: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = BrevoEmailProvider("synthetic-api-key", client=client).send(_message())
    finally:
        client.close()

    assert result.message_id == "synthetic-brevo-id"
    assert len(calls) == 2
    assert calls[0].url == "https://api.brevo.com/v3/smtp/email"


def test_resend_uses_http_api_and_unsubscribe_headers() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"id": "synthetic-resend-id"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = ResendEmailProvider("synthetic-api-key", client=client).send(_message())
    finally:
        client.close()

    assert result.message_id == "synthetic-resend-id"
    assert captured[0].url == "https://api.resend.com/emails"
    assert captured[0].headers["Authorization"] == "Bearer synthetic-api-key"
    assert captured[0].content.find(b"List-Unsubscribe") >= 0
