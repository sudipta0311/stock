"""Deliver one stored week's deterministic briefs to consenting users."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wealth.alerts.brief import (
    Brief,
    build_brief,
    personalize_plan_for_viewer,
    render_html,
    render_text,
)
from wealth.alerts.consent import consent_is_enabled
from wealth.alerts.delivery import send_content, unsubscribe_url
from wealth.alerts.email import rendered_body_hash
from wealth.alerts.settings import AlertSettings
from wealth.db.models import (
    AlertLog,
    HouseholdAccess,
    HouseholdMember,
    NetworthSnapshot,
    Profile,
    User,
)
from wealth.db.models import WeeklyPlan as WeeklyPlanRow
from wealth.db.session import make_engine, system_session, user_session

_LOGGER = logging.getLogger("wealth.alerts.weekly")
_INDIA = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class WeeklyJobCounts:
    week: str
    users: int
    eligible: int
    sent: int
    skipped: int
    failed: int
    outbox_paths: tuple[str, ...] = ()


def validate_week(week: str) -> None:
    try:
        year_text, week_text = week.split("-W")
        date.fromisocalendar(int(year_text), int(week_text), 1)
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError(
            "week must be a valid ISO week such as 2026-W45"
        ) from error


def _latest_networth(session: Session, user_id: UUID) -> dict:
    row = session.scalar(
        select(NetworthSnapshot)
        .where(NetworthSnapshot.user_id == user_id)
        .order_by(NetworthSnapshot.on_date.desc())
        .limit(1)
    )
    return row.totals if row is not None else {"net_worth": None, "maturity_calendar": []}


def _brief_context(
    settings: AlertSettings,
    user_id: UUID,
    week: str,
    now: datetime,
    owner_name: str | None,
) -> dict[str, str]:
    return {
        "app_base_url": settings.email_public_base_url,
        "manage_alerts_url": settings.email_manage_alerts_url,
        "unsubscribe_url": unsubscribe_url(settings, user_id, "weekly_email", now=now),
        "disclaimer": settings.email_disclaimer,
        "postal_address": settings.email_postal_address,
        "why_received": "You received this because you opted in to weekly email.",
        "week": week,
        "household_header": f"Family plan · {owner_name or 'Your'}'s household",
    }


def _login_owns_household_data(session: Session, user_id: UUID) -> bool:
    return bool(
        session.scalar(
            text(
                """
                SELECT EXISTS (SELECT 1 FROM profiles WHERE user_id = :user_id)
                    OR EXISTS (SELECT 1 FROM goals WHERE user_id = :user_id)
                    OR EXISTS (SELECT 1 FROM accounts WHERE user_id = :user_id)
                    OR EXISTS (SELECT 1 FROM manual_assets WHERE user_id = :user_id)
                    OR EXISTS (SELECT 1 FROM holdings_lots WHERE user_id = :user_id)
                    OR EXISTS (SELECT 1 FROM import_jobs WHERE user_id = :user_id)
                """
            ),
            {"user_id": user_id},
        )
    )


def _viewer_plan(
    session: Session,
    plan: dict,
    actor_user_id: UUID,
    owner_id: UUID,
    owner_name: str | None,
) -> dict:
    members = list(
        session.scalars(select(HouseholdMember).where(HouseholdMember.user_id == owner_id))
    )
    owner_self = next((member for member in members if member.is_self), None)
    if actor_user_id == owner_id:
        me_member = owner_self
    else:
        access = session.get(HouseholdAccess, (owner_id, actor_user_id))
        me_member = next(
            (member for member in members if access is not None and member.id == access.member_id),
            None,
        )
    return personalize_plan_for_viewer(
        plan,
        {str(member.id): (member.display_name, member.relationship) for member in members},
        me_member_id=str(me_member.id) if me_member is not None else None,
        owner_self_member_id=str(owner_self.id) if owner_self is not None else None,
        owner_name=owner_name or "Household owner",
    )


def run_weekly(
    engine: Engine,
    week: str,
    *,
    settings: AlertSettings | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    local_hour: int | None = None,
) -> WeeklyJobCounts:
    """List users as the owner and process each user's plan under user RLS."""
    validate_week(week)
    if local_hour is not None and not 0 <= local_hour <= 23:
        raise ValueError("local_hour must be between 0 and 23")
    configured = settings or AlertSettings.from_env()
    sent_at = now or datetime.now(UTC)
    with system_session(engine) as session:
        logins = list(
            session.scalars(select(User).where(User.deleted_at.is_(None)).order_by(User.id))
        )
        recipients: list[tuple[UUID, UUID, str, str | None]] = []
        for login in logins:
            owner_id = login.id
            if not _login_owns_household_data(session, login.id):
                shared_owner = session.scalar(
                    select(HouseholdAccess.owner_user_id)
                    .where(
                        HouseholdAccess.grantee_user_id == login.id,
                        HouseholdAccess.role == "partner",
                        HouseholdAccess.revoked_at.is_(None),
                    )
                    .order_by(HouseholdAccess.created_at, HouseholdAccess.owner_user_id)
                    .limit(1)
                )
                if shared_owner is not None:
                    owner_id = shared_owner
            owner_name = session.scalar(select(User.display_name).where(User.id == owner_id))
            recipients.append((login.id, owner_id, login.email, owner_name))

    eligible = sent = skipped = failed = 0
    outbox_paths: list[str] = []
    for user_id, owner_id, email, owner_name in recipients:
        try:
            with user_session(engine, user_id, owner_id) as session:
                user = session.get(User, user_id)
                plan_row = session.get(WeeklyPlanRow, (owner_id, week))
                if local_hour is not None:
                    profile = session.get(Profile, user_id)
                    if profile is None or not _is_local_hour(sent_at, profile.timezone, local_hour):
                        skipped += 1
                        continue
                if (
                    user is None
                    or plan_row is None
                    or not consent_is_enabled(session, user_id, "weekly_email")
                ):
                    skipped += 1
                    continue
                eligible += 1
                dedupe_key = f"weekly:{week}:{user_id}"
                existing = session.scalar(
                    select(AlertLog.id).where(
                        AlertLog.user_id == owner_id,
                        AlertLog.dedupe_key == dedupe_key,
                    )
                )
                if existing is not None:
                    skipped += 1
                    continue

                totals = _latest_networth(session, owner_id)
                viewer_plan = _viewer_plan(session, plan_row.plan, user_id, owner_id, owner_name)
                brief: Brief = build_brief(
                    viewer_plan,
                    totals,
                    totals.get("maturity_calendar", []),
                    None,
                    _brief_context(configured, user_id, week, sent_at, owner_name),
                    deployment_mode=configured.deployment_mode,
                )
                html = render_html(brief)
                text_body = render_text(brief)
                audit = send_content(
                    settings=configured,
                    user_id=user_id,
                    to=email,
                    subject=brief.subject,
                    text=text_body,
                    html=html,
                    unsubscribe_kind="weekly_email",
                    dedupe_key=f"weekly_{week}_{user_id.hex[:8]}",
                    dry_run=dry_run,
                    now=sent_at,
                    unsubscribe_url_value=brief.unsubscribe_url,
                )
                session.add(
                    AlertLog(
                        user_id=owner_id,
                        kind="weekly",
                        week=week,
                        dedupe_key=dedupe_key,
                        subject=brief.subject,
                        provider_message_id=audit.message_id,
                        body_sha256=rendered_body_hash(html),
                        sent_at=sent_at,
                    )
                )
                session.flush()
                sent += 1
                if audit.outbox_path:
                    outbox_paths.append(audit.outbox_path)
        except Exception:
            failed += 1
            _LOGGER.error("weekly alert failed for user id %s", user_id)

    return WeeklyJobCounts(
        week=week,
        users=len(recipients),
        eligible=eligible,
        sent=sent,
        skipped=skipped,
        failed=failed,
        outbox_paths=tuple(outbox_paths),
    )


def _is_local_hour(sent_at: datetime, timezone: str, local_hour: int) -> bool:
    try:
        return sent_at.astimezone(ZoneInfo(timezone)).hour == local_hour
    except (ValueError, ZoneInfoNotFoundError):
        return False


def _sample_brief(status: str, week: str, settings: AlertSettings) -> tuple[Brief, str]:
    root = Path(__file__).resolve().parents[3]
    plan = json.loads((root / "tests" / "golden" / "weekly_plan_user1.json").read_text("utf-8"))
    plan["week"] = week
    if status == "none":
        plan["actions"] = []
        plan["exit_cards"] = []
    current_value = sum(
        (Decimal(values["current"]) for values in plan["allocation"].values()), Decimal(0)
    )
    sample_user_id = UUID("00000000-0000-0000-0000-000000000017")
    now = datetime.now(UTC)
    context = {
        "app_base_url": settings.email_public_base_url,
        "manage_alerts_url": settings.email_manage_alerts_url,
        "unsubscribe_url": unsubscribe_url(settings, sample_user_id, "weekly_email", now=now),
        "disclaimer": settings.email_disclaimer,
        "postal_address": settings.email_postal_address,
        "why_received": "You received this because the sample user opted in to weekly email.",
    }
    brief = build_brief(
        plan,
        {"net_worth": format(current_value, "f"), "maturity_calendar": []},
        [],
        None,
        context,
        deployment_mode=settings.deployment_mode,
    )
    return brief, "golden-user@example.test"


def _run_sample(status: str, week: str, settings: AlertSettings) -> WeeklyJobCounts:
    brief, recipient = _sample_brief(status, week, settings)
    html = render_html(brief)
    text_body = render_text(brief)
    sample_user_id = UUID("00000000-0000-0000-0000-000000000017")
    audit = send_content(
        settings=settings,
        user_id=sample_user_id,
        to=recipient,
        subject=brief.subject,
        text=text_body,
        html=html,
        unsubscribe_kind="weekly_email",
        dedupe_key=f"sample_{status}_{week}",
        dry_run=True,
        unsubscribe_url_value=brief.unsubscribe_url,
    )
    return WeeklyJobCounts(
        week=week,
        users=1,
        eligible=1,
        sent=1,
        skipped=0,
        failed=0,
        outbox_paths=(audit.outbox_path,) if audit.outbox_path else (),
    )


def _database_url() -> str:
    value = os.getenv("OWNER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("Set OWNER_DATABASE_URL or DATABASE_URL")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send one week's approved email briefs")
    parser.add_argument("--week", required=True)
    parser.add_argument("--local-hour", type=int, choices=range(24))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", choices=("actions", "none"))
    arguments = parser.parse_args(argv)
    try:
        validate_week(arguments.week)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    settings = AlertSettings.from_env()
    if arguments.sample:
        if not arguments.dry_run:
            parser.error("--sample requires --dry-run")
        if not settings.email_link_secret_value:
            settings = AlertSettings(
                **{
                    **asdict(settings),
                    "email_link_secret_value": secrets.token_urlsafe(32),
                }
            )
        counts = _run_sample(arguments.sample, arguments.week, settings)
    else:
        engine = make_engine(_database_url())
        try:
            counts = run_weekly(
                engine,
                arguments.week,
                settings=settings,
                dry_run=arguments.dry_run,
                local_hour=arguments.local_hour,
            )
        finally:
            engine.dispose()
    print(json.dumps(asdict(counts), sort_keys=True))
    return 2 if counts.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
