"""Run one deterministic Personaliser pass for every active user."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.engine import Engine

from wealth.db.models import HouseholdAccess, User
from wealth.db.session import make_engine, system_session, user_session
from wealth.policy.data import load_policy_inputs
from wealth.policy.personalise import personalise
from wealth.policy.store import save_weekly_plan

_LOGGER = logging.getLogger("wealth.policy.run_weekly")
_INDIA = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class JobCounts:
    week: str
    users: int
    plans: int
    actions: int
    failed: int


def _validate_week(week: str) -> None:
    try:
        year_text, week_text = week.split("-W")
        date.fromisocalendar(int(year_text), int(week_text), 1)
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError(
            "week must be a valid ISO week such as 2026-W45"
        ) from error


def run_weekly(engine: Engine, week: str, as_of: date) -> JobCounts:
    """Compute one plan per household owner, then persist under user RLS."""
    _validate_week(week)
    with system_session(engine) as session:
        users = list(
            session.scalars(select(User).where(User.deleted_at.is_(None)).order_by(User.id))
        )
        active_partners = set(
            session.scalars(
                select(HouseholdAccess.grantee_user_id).where(
                    HouseholdAccess.role == "partner", HouseholdAccess.revoked_at.is_(None)
                )
            )
        )
        user_ids = []
        for user in users:
            if user.id not in active_partners:
                user_ids.append(user.id)
                continue
            owns_data = session.scalar(
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
                {"user_id": user.id},
            )
            if owns_data:
                user_ids.append(user.id)

    failed = 0
    action_count = 0
    for user_id in user_ids:
        try:
            with user_session(engine, user_id) as session:
                user_data, market, research = load_policy_inputs(
                    session,
                    user_id,
                    week,
                    as_of,
                    deployment_mode=os.getenv("DEPLOYMENT_MODE", "personal"),  # type: ignore[arg-type]
                )
                plan = personalise(user_data, market, research, week, as_of)
                save_weekly_plan(session, user_id, plan)
                user_action_count = len(plan.actions)
            action_count += user_action_count
        except Exception:
            failed += 1
            _LOGGER.error("weekly plan failed for user id %s", user_id)
    return JobCounts(
        week=week,
        users=len(user_ids),
        plans=len(user_ids) - failed,
        actions=action_count,
        failed=failed,
    )


def _database_url() -> str:
    value = (
        os.getenv("OWNER_DATABASE_URL")
        or os.getenv("TEST_DATABASE_URL")
        or os.getenv("DATABASE_URL")
    )
    if not value:
        raise RuntimeError("Set OWNER_DATABASE_URL, TEST_DATABASE_URL, or DATABASE_URL")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute deterministic per-user weekly plans")
    parser.add_argument("--week", required=True, type=str)
    arguments = parser.parse_args(argv)
    try:
        _validate_week(arguments.week)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    as_of = datetime.now(_INDIA).date()
    engine = make_engine(_database_url())
    try:
        counts = run_weekly(engine, arguments.week, as_of)
    finally:
        engine.dispose()
    print(json.dumps(asdict(counts), sort_keys=True))
    return 2 if counts.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
