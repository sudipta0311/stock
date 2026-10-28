"""Tenant isolation and scheduled delivery behaviour for the alerts card."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import jwt
import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session
from tests.policy.test_personalise import _plan_json

from conftest import MigratedDatabase
from wealth.alerts.consent import create_unsubscribe_token, set_consent
from wealth.alerts.daily import run_daily
from wealth.alerts.push import PushDeliveryGone
from wealth.alerts.settings import AlertSettings
from wealth.alerts.weekly import run_weekly
from wealth.api.main import create_app
from wealth.api.money_map import get_today
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import (
    AlertLog,
    HoldingLot,
    ManualAsset,
    NetworthSnapshot,
    PushSubscription,
    Security,
    SentinelEvent,
    User,
    WeeklyPlan,
)
from wealth.db.session import user_session

pytestmark = pytest.mark.db
_TODAY = date(2026, 10, 28)
_NOW = datetime(2026, 10, 28, 6, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
_WEEK = "2026-W44"


@pytest.fixture
def api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, object]]:
    secret = secrets.token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql://wealth_api@localhost/wealth_test",
        email_link_secret="synthetic-email-link-secret",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    application = create_app(settings)
    application.dependency_overrides[get_today] = lambda: _TODAY
    with TestClient(application) as client:
        yield client, secret, application


def _auth(secret: str, label: str) -> tuple[dict[str, str], str, str]:
    subject = f"synthetic-alert-{label}-{uuid4()}"
    email = f"{label}-{uuid4()}@example.test"
    token = jwt.encode(
        {
            "sub": subject,
            "email": email,
            "name": "Synthetic Alerts User",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}, subject, email


def _create_user(
    database: MigratedDatabase,
    *,
    label: str,
    email: str | None = None,
) -> UUID:
    user_id = uuid4()
    with Session(database.owner_engine) as session, session.begin():
        session.add(
            User(
                id=user_id,
                email=email or f"{label}-{user_id.hex}@example.test",
                display_name="Synthetic Alerts User",
                auth_subject=f"synthetic-alert-{label}-{user_id}",
            )
        )
    return user_id


def _weekly_plan(week: str = _WEEK) -> dict:
    plan = _plan_json()
    plan["week"] = week
    return plan


def _alert_settings(outbox: str, *, provider: str = "dry_run") -> AlertSettings:
    return AlertSettings(
        email_provider=provider,
        email_from="alerts@example.invalid",
        email_api_key_value="synthetic-provider-key" if provider != "dry_run" else None,
        email_link_secret_value="synthetic-email-link-secret",
        email_outbox_dir=outbox,
        vapid_public_key_value="synthetic-vapid-public",
        vapid_private_key_value="synthetic-vapid-private",
    )


def test_consent_defaults_off_and_checklist_is_available_on_me(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers_a, _subject_a, _email_a = _auth(secret, "consent-a")
    headers_b, _subject_b, _email_b = _auth(secret, "consent-b")

    assert client.get("/v1/me/consents", headers=headers_a).json() == {
        "weekly_email": False,
        "reminder_email": False,
        "push": False,
        "urgent_alerts": False,
    }
    saved = client.put(
        "/v1/me/consents",
        headers=headers_a,
        json={"weekly_email": True, "reminder_email": False, "push": True, "urgent_alerts": True},
    )
    assert saved.status_code == 200
    assert saved.json()["weekly_email"] is True
    assert client.get("/v1/me/consents", headers=headers_b).json()["weekly_email"] is False

    put_checklist = client.put(
        "/v1/me/checklist", headers=headers_a, json={"categories": ["epf", "insurance"]}
    )
    assert put_checklist.status_code == 200
    assert client.get("/v1/me/checklist", headers=headers_a).json() == {
        "categories": ["epf", "insurance"]
    }
    assert client.get("/v1/me/checklist", headers=headers_b).json() == {"categories": []}
    assert client.get("/v1/me", headers=headers_a).json()["checklist"] == ["epf", "insurance"]


def test_unsubscribe_endpoint_withdraws_consent_without_revealing_address(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers, _subject, email = _auth(secret, "unsubscribe")
    assert (
        client.put(
            "/v1/me/consents",
            headers=headers,
            json={
                "weekly_email": True,
                "reminder_email": False,
                "push": False,
                "urgent_alerts": False,
            },
        ).status_code
        == 200
    )
    user = client.get("/v1/me", headers=headers).json()
    token = create_unsubscribe_token(
        UUID(user["id"]),
        "weekly_email",
        "synthetic-email-link-secret",
    )

    response = client.get("/v1/email/unsubscribe", params={"token": token})

    assert response.status_code == 200
    assert "Alerts updated" in response.text
    assert email not in response.text
    assert client.get("/v1/me/consents", headers=headers).json()["weekly_email"] is False


def test_push_subscription_host_validation_and_tenant_delete(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers_a, _subject_a, _email_a = _auth(secret, "push-a")
    headers_b, _subject_b, _email_b = _auth(secret, "push-b")
    response = client.post(
        "/v1/push/subscriptions",
        headers=headers_a,
        json={
            "endpoint": "https://updates.push.services.mozilla.com/wpush/v2/synthetic-key",
            "keys": {"p256dh": "synthetic-p256dh", "auth": "synthetic-auth"},
        },
    )
    assert response.status_code == 201
    subscription_id = response.json()["id"]
    invalid = client.post(
        "/v1/push/subscriptions",
        headers=headers_a,
        json={
            "endpoint": "https://attacker.example/push/synthetic-key",
            "keys": {"p256dh": "synthetic-p256dh", "auth": "synthetic-auth"},
        },
    )
    assert invalid.status_code == 422
    hidden = client.delete(f"/v1/push/subscriptions/{subscription_id}", headers=headers_b)
    assert hidden.status_code == 404
    assert (
        client.delete(f"/v1/push/subscriptions/{subscription_id}", headers=headers_a).status_code
        == 204
    )


def test_weekly_delivery_is_consent_gated_and_idempotent(
    migrated_database: MigratedDatabase,
    tmp_path,
) -> None:
    user_id = _create_user(migrated_database, label="weekly")
    with user_session(migrated_database.owner_engine, user_id) as session:
        session.add(WeeklyPlan(user_id=user_id, week=_WEEK, plan=_weekly_plan()))
        set_consent(session, user_id, "weekly_email", True, now=_NOW)

    settings = _alert_settings(str(tmp_path / "outbox"))
    first = run_weekly(
        migrated_database.owner_engine,
        _WEEK,
        settings=settings,
        dry_run=True,
        now=_NOW,
    )
    second = run_weekly(
        migrated_database.owner_engine,
        _WEEK,
        settings=settings,
        dry_run=True,
        now=_NOW,
    )

    assert first.sent == 1
    assert second.sent == 0
    assert second.skipped >= 1
    assert len(list((tmp_path / "outbox").glob("*.eml"))) == 1
    with Session(migrated_database.owner_engine) as session:
        logs = session.scalar(
            select(func.count())
            .select_from(AlertLog)
            .where(AlertLog.user_id == user_id, AlertLog.kind == "weekly")
        )
    assert logs == 1


def test_daily_reminder_dedupes_after_send(
    migrated_database: MigratedDatabase,
    tmp_path,
) -> None:
    user_id = _create_user(migrated_database, label="reminder")
    asset_id = uuid4()
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(
            ManualAsset(
                id=asset_id,
                user_id=user_id,
                asset_type="fd",
                details={"asset_type": "fd", "bank": "Synthetic Bank"},
                status="active",
            )
        )
        session.add(
            NetworthSnapshot(
                user_id=user_id,
                on_date=_TODAY,
                totals={
                    "net_worth": "100000.00",
                    "maturity_calendar": [
                        {
                            "date": _TODAY.isoformat(),
                            "asset_id": str(asset_id),
                            "kind": "maturity",
                            "amount": "100000.00",
                            "label": "Synthetic deposit maturity",
                        }
                    ],
                },
            )
        )
    with user_session(migrated_database.owner_engine, user_id) as session:
        set_consent(session, user_id, "reminder_email", True, now=_NOW)

    settings = _alert_settings(str(tmp_path / "outbox"))
    first = run_daily(
        migrated_database.owner_engine,
        today=_TODAY,
        now=_NOW,
        settings=settings,
        dry_run=True,
    )
    second = run_daily(
        migrated_database.owner_engine,
        today=_TODAY,
        now=_NOW,
        settings=settings,
        dry_run=True,
    )

    assert first.reminders_sent == 1
    assert second.reminders_sent == 0
    assert len(list((tmp_path / "outbox").glob("*.eml"))) == 1
    with Session(migrated_database.owner_engine) as session:
        reminder = session.scalar(
            select(func.count())
            .select_from(AlertLog)
            .where(AlertLog.user_id == user_id, AlertLog.kind == "reminder")
        )
    assert reminder == 1


def test_urgent_alerts_are_holder_only_and_capped_at_one_per_day(
    migrated_database: MigratedDatabase,
    tmp_path,
) -> None:
    holder_id = _create_user(migrated_database, label="holder")
    nonholder_id = _create_user(migrated_database, label="nonholder")
    held_security_id = uuid4()
    unheld_security_id = uuid4()
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add_all(
            [
                Security(
                    id=held_security_id,
                    market="IN",
                    symbol="SYNTH-HOLDER",
                    name="Synthetic Holder Security",
                    security_type="equity",
                    currency="INR",
                    sleeve="equity_india",
                ),
                Security(
                    id=unheld_security_id,
                    market="IN",
                    symbol="SYNTH-UNHELD",
                    name="Synthetic Unheld Security",
                    security_type="equity",
                    currency="INR",
                    sleeve="equity_india",
                ),
            ]
        )
        session.flush()
        session.add(
            HoldingLot(
                user_id=holder_id,
                security_id=held_security_id,
                quantity=Decimal("1"),
                source="manual",
            )
        )
        for security_id, event_code in (
            (held_security_id, "AUDITOR_RESIGNATION"),
            (unheld_security_id, "RESTATEMENT"),
        ):
            session.add(
                SentinelEvent(
                    security_id=security_id,
                    event_code=event_code,
                    source=f"synthetic-event-{event_code.lower()}",
                    detected_at=_NOW - timedelta(hours=1),
                    method="rule",
                    urgent=True,
                )
            )
    for user_id in (holder_id, nonholder_id):
        with user_session(migrated_database.owner_engine, user_id) as session:
            set_consent(session, user_id, "urgent_alerts", True, now=_NOW)

    settings = _alert_settings(str(tmp_path / "outbox"))
    first = run_daily(
        migrated_database.owner_engine,
        today=_TODAY,
        now=_NOW,
        settings=settings,
        dry_run=True,
    )
    second = run_daily(
        migrated_database.owner_engine,
        today=_TODAY,
        now=_NOW,
        settings=settings,
        dry_run=True,
    )

    assert first.urgent_sent == 1
    assert second.urgent_sent == 0
    with Session(migrated_database.owner_engine) as session:
        urgent_logs = list(session.scalars(select(AlertLog).where(AlertLog.kind == "urgent")))
    assert len(urgent_logs) == 1
    assert urgent_logs[0].user_id == holder_id


def test_push_410_removes_expired_subscription(
    migrated_database: MigratedDatabase,
) -> None:
    user_id = _create_user(migrated_database, label="push-expired")
    asset_id = uuid4()
    subscription_id = uuid4()
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(
            ManualAsset(
                id=asset_id,
                user_id=user_id,
                asset_type="fd",
                details={"asset_type": "fd", "bank": "Synthetic Bank"},
                status="active",
            )
        )
        session.add(
            NetworthSnapshot(
                user_id=user_id,
                on_date=_TODAY,
                totals={
                    "maturity_calendar": [
                        {
                            "date": _TODAY.isoformat(),
                            "asset_id": str(asset_id),
                            "kind": "maturity",
                            "amount": "100000.00",
                            "label": "Synthetic maturity",
                        }
                    ]
                },
            )
        )
        session.add(
            PushSubscription(
                id=subscription_id,
                user_id=user_id,
                endpoint="https://fcm.googleapis.com/fcm/send/synthetic-key",
                keys={"p256dh": "synthetic-p256dh", "auth": "synthetic-auth"},
            )
        )
    with user_session(migrated_database.owner_engine, user_id) as session:
        set_consent(session, user_id, "push", True, now=_NOW)

    def expired(*_args, **_kwargs):
        raise PushDeliveryGone

    run_daily(
        migrated_database.owner_engine,
        today=_TODAY,
        now=_NOW,
        settings=_alert_settings("unused", provider="resend"),
        push_sender=expired,
    )

    with Session(migrated_database.owner_engine) as session:
        assert session.get(PushSubscription, subscription_id) is None


def test_inbox_and_brief_are_scoped_to_authenticated_user(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, _application = api_client
    headers_a, subject_a, _email_a = _auth(secret, "inbox-a")
    headers_b, subject_b, _email_b = _auth(secret, "inbox-b")
    assert client.get("/v1/me", headers=headers_a).status_code == 200
    assert client.get("/v1/me", headers=headers_b).status_code == 200
    with Session(migrated_database.owner_engine) as session, session.begin():
        user_a = session.scalar(select(User).where(User.auth_subject == subject_a))
        user_b = session.scalar(select(User).where(User.auth_subject == subject_b))
        assert user_a is not None and user_b is not None
        session.add_all(
            [
                WeeklyPlan(user_id=user_a.id, week="2026-W44", plan=_weekly_plan("2026-W44")),
                WeeklyPlan(user_id=user_b.id, week="2026-W45", plan=_weekly_plan("2026-W45")),
            ]
        )

    inbox_a = client.get("/v1/inbox", headers=headers_a).json()["items"]
    inbox_b = client.get("/v1/inbox", headers=headers_b).json()["items"]
    assert [item["week"] for item in inbox_a] == ["2026-W44"]
    assert [item["week"] for item in inbox_b] == ["2026-W45"]
    assert client.get("/v1/brief/2026-W44", headers=headers_a).status_code == 200
    assert client.get("/v1/brief/2026-W44", headers=headers_b).status_code == 404


def test_consent_off_sends_nothing(
    migrated_database: MigratedDatabase,
    tmp_path,
) -> None:
    user_id = _create_user(migrated_database, label="consent-off")
    with user_session(migrated_database.owner_engine, user_id) as session:
        session.add(WeeklyPlan(user_id=user_id, week=_WEEK, plan=_weekly_plan()))

    counts = run_weekly(
        migrated_database.owner_engine,
        _WEEK,
        settings=_alert_settings(str(tmp_path / "outbox")),
        dry_run=True,
        now=_NOW,
    )

    assert counts.sent == 0
    assert counts.failed == 0
    assert not list((tmp_path / "outbox").glob("*.eml"))


def test_z_alert_migration_can_downgrade_and_upgrade(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0007_personalisation")
    inspector = inspect(migrated_database.owner_engine)
    assert "alerts_log" not in inspector.get_table_names()
    assert "sentinel_events" not in inspector.get_table_names()
    reminder_columns = {column["name"] for column in inspector.get_columns("reminders")}
    assert "notify_on" not in reminder_columns

    command.upgrade(migrated_database.config, "head")
    inspector = inspect(migrated_database.owner_engine)
    assert {"alerts_log", "sentinel_events"} <= set(inspector.get_table_names())
    reminder_columns = {column["name"] for column in inspector.get_columns("reminders")}
    assert {"notify_on", "payload"} <= reminder_columns
