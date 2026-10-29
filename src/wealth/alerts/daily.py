"""Send dated Money Map reminders and holder-only urgent notifications."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wealth.alerts.consent import consent_is_enabled
from wealth.alerts.copy import load_copy
from wealth.alerts.delivery import (
    render_notification,
    send_content,
    unsubscribe_url,
)
from wealth.alerts.email import rendered_body_hash
from wealth.alerts.push import PushDeliveryGone, send_push
from wealth.alerts.settings import AlertSettings
from wealth.core.money import format_inr
from wealth.db.models import (
    AlertLog,
    HoldingLot,
    NetworthSnapshot,
    PushSubscription,
    Reminder,
    SentinelEvent,
    User,
)
from wealth.db.session import make_engine, system_session, user_session

_LOGGER = logging.getLogger("wealth.alerts.daily")
_INDIA = ZoneInfo("Asia/Kolkata")
_REMINDER_OFFSETS = (30, 7, 0)


@dataclass(frozen=True)
class DailyJobCounts:
    users: int
    reminders_sent: int
    urgent_sent: int
    failed: int


def reminders_due(calendar: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    """Return the maturity-calendar reminders due today at −30, −7, or 0 days."""
    scheduled: list[dict[str, Any]] = []
    for raw_event in calendar:
        event = dict(raw_event)
        try:
            event_date = date.fromisoformat(str(event["date"]))
        except (KeyError, ValueError):
            continue
        days_until = (event_date - today).days
        kind = str(event.get("kind", "event"))
        allowed_offsets = (
            {3}
            if kind == "loan_payment"
            else {0}
            if kind == "loan_rate_review"
            else set(_REMINDER_OFFSETS)
        )
        if days_until not in allowed_offsets:
            continue
        scheduled.append(
            {
                **event,
                "event_on": event_date.isoformat(),
                "notify_on": today.isoformat(),
                "offset_days": days_until,
                "reminder_kind": f"{kind}_{days_until}",
            }
        )
    return sorted(
        scheduled,
        key=lambda item: (item["event_on"], str(item.get("asset_id", "")), item["reminder_kind"]),
    )


def _snapshot_calendar(session: Session, user_id: Any) -> list[dict[str, Any]]:
    snapshot = session.scalar(
        select(NetworthSnapshot)
        .where(NetworthSnapshot.user_id == user_id)
        .order_by(NetworthSnapshot.on_date.desc())
        .limit(1)
    )
    if snapshot is None:
        return []
    return list(snapshot.totals.get("maturity_calendar", []))


def _upsert_reminder(session: Session, user_id: Any, event: dict[str, Any]) -> Reminder:
    event_on = date.fromisoformat(event["event_on"])
    reminder = session.scalar(
        select(Reminder).where(
            Reminder.user_id == user_id,
            Reminder.asset_id == event.get("asset_id"),
            Reminder.due_on == event_on,
            Reminder.kind == event["reminder_kind"],
        )
    )
    if reminder is None:
        reminder = Reminder(
            user_id=user_id,
            asset_id=event.get("asset_id"),
            due_on=event_on,
            notify_on=date.fromisoformat(event["notify_on"]),
            kind=event["reminder_kind"],
            payload={
                "event_kind": event.get("kind"),
                "label": event.get("label", "Money Map date"),
                "amount": event.get("amount"),
                "offset_days": event["offset_days"],
            },
        )
        session.add(reminder)
        session.flush()
    return reminder


def _reminder_content(
    event: dict[str, Any],
    settings: AlertSettings,
    user_id: Any,
    now: datetime,
    *,
    email_enabled: bool,
) -> tuple[str, str, str, str]:
    copy = load_copy()["email"]
    label = str(event.get("label", "Money Map date"))
    event_date = date.fromisoformat(event["event_on"])
    date_label = f"{event_date.day} {event_date.strftime('%b %Y')}"
    amount = event.get("amount")
    amount_line = f" Amount: {format_inr(Decimal(str(amount)), style='compact')}." if amount else ""
    subject = copy["reminder_subject"]
    body = copy["reminder_body"].format(label=label, date=date_label, amount_line=amount_line)
    unsubscribe = (
        unsubscribe_url(settings, user_id, "reminder_email", now=now)
        if email_enabled
        else settings.email_manage_alerts_url
    )
    html, text = render_notification(
        settings=settings,
        subject=subject,
        body=body,
        unsubscribe_url_value=unsubscribe,
    )
    return subject, body, html, text


def _send_reminders_for_user(
    session: Session,
    user: User,
    today: date,
    now: datetime,
    settings: AlertSettings,
    *,
    dry_run: bool,
    push_sender: Callable[..., str],
) -> int:
    email_enabled = consent_is_enabled(session, user.id, "reminder_email")
    push_enabled = consent_is_enabled(session, user.id, "push")
    if not email_enabled and not push_enabled:
        return 0

    events = reminders_due(_snapshot_calendar(session, user.id), today)
    for event in events:
        _upsert_reminder(session, user.id, event)
    due_rows = list(
        session.scalars(
            select(Reminder)
            .where(
                Reminder.user_id == user.id,
                Reminder.notify_on == today,
                Reminder.sent_at.is_(None),
            )
            .order_by(Reminder.due_on, Reminder.id)
        )
    )
    sent_count = 0
    subscriptions = (
        list(
            session.scalars(
                select(PushSubscription)
                .where(PushSubscription.user_id == user.id)
                .order_by(PushSubscription.id)
            )
        )
        if push_enabled and not dry_run
        else []
    )
    for reminder in due_rows:
        dedupe_key = f"reminder:{reminder.id}"
        already_sent = session.scalar(
            select(AlertLog.id).where(
                AlertLog.user_id == user.id,
                AlertLog.dedupe_key == dedupe_key,
            )
        )
        if already_sent is not None:
            reminder.sent_at = reminder.sent_at or now
            continue
        event = {
            "event_on": reminder.due_on.isoformat(),
            "label": reminder.payload.get("label", "Money Map date"),
            "amount": reminder.payload.get("amount"),
        }
        subject, body, html, text_body = _reminder_content(
            event, settings, user.id, now, email_enabled=email_enabled
        )
        message_ids: list[str] = []
        if email_enabled:
            audit = send_content(
                settings=settings,
                user_id=user.id,
                to=user.email,
                subject=subject,
                text=text_body,
                html=html,
                unsubscribe_kind="reminder_email",
                dedupe_key=f"reminder_{reminder.id.hex[:12]}",
                dry_run=dry_run,
                now=now,
            )
            message_ids.append(audit.message_id)
        if push_enabled and not dry_run:
            push_payload = {"title": subject, "body": body, "url": settings.email_public_base_url}
            for subscription in subscriptions:
                try:
                    message_ids.append(
                        push_sender(
                            {
                                "endpoint": subscription.endpoint,
                                "keys": subscription.keys,
                            },
                            push_payload,
                            settings,
                        )
                    )
                except PushDeliveryGone:
                    session.delete(subscription)
        if not message_ids:
            continue
        reminder.sent_at = now
        session.add(
            AlertLog(
                user_id=user.id,
                kind="reminder",
                dedupe_key=dedupe_key,
                subject=subject,
                provider_message_id=", ".join(message_ids),
                body_sha256=rendered_body_hash(html),
                sent_at=now,
            )
        )
        session.flush()
        sent_count += 1
    return sent_count


def _urgent_events(
    session: Session, user_id: Any, since: datetime, until: datetime
) -> list[SentinelEvent]:
    held_security_ids = (
        select(HoldingLot.security_id).where(HoldingLot.user_id == user_id).distinct()
    )
    statement = (
        select(SentinelEvent)
        .where(
            SentinelEvent.security_id.in_(held_security_ids),
            SentinelEvent.urgent.is_(True),
            SentinelEvent.detected_at >= since,
            SentinelEvent.detected_at <= until,
        )
        .order_by(SentinelEvent.detected_at.desc(), SentinelEvent.id)
    )
    return list(session.scalars(statement))


def _send_urgent_for_user(
    session: Session,
    user: User,
    now: datetime,
    settings: AlertSettings,
    *,
    dry_run: bool,
    push_sender: Callable[..., str],
) -> bool:
    if not consent_is_enabled(session, user.id, "urgent_alerts"):
        return False
    push_enabled = consent_is_enabled(session, user.id, "push")
    dedupe_key = f"urgent:{now.date().isoformat()}"
    existing = session.scalar(
        select(AlertLog.id).where(
            AlertLog.user_id == user.id,
            AlertLog.kind == "urgent",
            AlertLog.dedupe_key == dedupe_key,
        )
    )
    if existing is not None:
        return False
    events = _urgent_events(session, user.id, now - timedelta(hours=24), now)
    if not events:
        return False

    event_codes = ", ".join(dict.fromkeys(event.event_code for event in events))
    copy = load_copy()["email"]
    subject = copy["urgent_subject"]
    body = copy["urgent_body"].format(event_code=event_codes)
    unsubscribe = unsubscribe_url(settings, user.id, "urgent_alerts", now=now)
    html, text_body = render_notification(
        settings=settings,
        subject=subject,
        body=body,
        unsubscribe_url_value=unsubscribe,
    )
    message_ids: list[str] = []
    audit = send_content(
        settings=settings,
        user_id=user.id,
        to=user.email,
        subject=subject,
        text=text_body,
        html=html,
        unsubscribe_kind="urgent_alerts",
        dedupe_key=f"urgent_{now.date().isoformat()}_{user.id.hex[:8]}",
        dry_run=dry_run,
        now=now,
        unsubscribe_url_value=unsubscribe,
    )
    message_ids.append(audit.message_id)
    if push_enabled and not dry_run:
        push_payload = {
            "title": subject,
            "body": "An important update affects a holding you own.",
            "url": settings.email_public_base_url,
        }
        for subscription in session.scalars(
            select(PushSubscription).where(PushSubscription.user_id == user.id)
        ):
            try:
                message_ids.append(
                    push_sender(
                        {"endpoint": subscription.endpoint, "keys": subscription.keys},
                        push_payload,
                        settings,
                    )
                )
            except PushDeliveryGone:
                session.delete(subscription)
    if not message_ids:
        return False
    session.add(
        AlertLog(
            user_id=user.id,
            kind="urgent",
            dedupe_key=dedupe_key,
            subject=subject,
            provider_message_id=", ".join(message_ids),
            body_sha256=rendered_body_hash(html),
            sent_at=now,
        )
    )
    session.flush()
    return True


def run_daily(
    engine: Engine,
    *,
    today: date,
    settings: AlertSettings | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    push_sender: Callable[..., str] = send_push,
) -> DailyJobCounts:
    """Deliver date reminders and at most one held-security urgent alert per day."""
    configured = settings or AlertSettings.from_env()
    sent_at = now or datetime.now(_INDIA)
    dry = dry_run or configured.email_provider == "dry_run"
    with system_session(engine) as session:
        user_ids = list(
            session.scalars(select(User.id).where(User.deleted_at.is_(None)).order_by(User.id))
        )

    reminders_sent = urgent_sent = failed = 0
    for user_id in user_ids:
        try:
            with user_session(engine, user_id) as session:
                user = session.get(User, user_id)
                if user is None:
                    continue
                reminders_sent += _send_reminders_for_user(
                    session,
                    user,
                    today,
                    sent_at,
                    configured,
                    dry_run=dry,
                    push_sender=push_sender,
                )
                if _send_urgent_for_user(
                    session,
                    user,
                    sent_at,
                    configured,
                    dry_run=dry,
                    push_sender=push_sender,
                ):
                    urgent_sent += 1
        except Exception:
            failed += 1
            _LOGGER.error("daily alerts failed for user id %s", user_id)
    return DailyJobCounts(
        users=len(user_ids),
        reminders_sent=reminders_sent,
        urgent_sent=urgent_sent,
        failed=failed,
    )


def _database_url() -> str:
    value = os.getenv("OWNER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("Set OWNER_DATABASE_URL or DATABASE_URL")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send dated reminders and urgent alerts")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    now = datetime.now(_INDIA)
    engine = make_engine(_database_url())
    try:
        counts = run_daily(engine, today=now.date(), now=now, dry_run=arguments.dry_run)
    finally:
        engine.dispose()
    print(json.dumps(asdict(counts), sort_keys=True))
    return 2 if counts.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
