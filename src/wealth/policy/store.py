"""Tenant-scoped persistence for computed weekly plans and proposed actions."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from wealth.db.models import Action as ActionRow
from wealth.db.models import WeeklyPlan as WeeklyPlanRow
from wealth.policy.personalise import WeeklyPlan


def save_weekly_plan(
    session: Session, user_id: UUID, plan: WeeklyPlan, *, replace_proposed_actions: bool = True
) -> None:
    """Upsert one plan and replace only its still-proposed action rows."""
    document = plan.model_dump(mode="json")
    statement = pg_insert(WeeklyPlanRow).values(
        user_id=user_id,
        week=plan.week,
        plan=document,
    )
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[WeeklyPlanRow.user_id, WeeklyPlanRow.week],
            set_={"plan": statement.excluded.plan, "created_at": text("now()")},
        )
    )

    if not replace_proposed_actions:
        return
    session.execute(
        delete(ActionRow).where(
            ActionRow.user_id == user_id,
            ActionRow.week == plan.week,
            ActionRow.status == "proposed",
        )
    )
    for action in plan.actions:
        session.add(
            ActionRow(
                user_id=user_id,
                week=plan.week,
                rank=int(action["rank"]),
                kind=str(action["kind"]),
                payload=action,
                status="proposed",
            )
        )
    session.flush()


def latest_plan(session: Session, user_id: UUID) -> WeeklyPlanRow | None:
    """Return the user's most recent stored plan."""
    return session.scalar(
        select(WeeklyPlanRow)
        .where(WeeklyPlanRow.user_id == user_id)
        .order_by(WeeklyPlanRow.week.desc())
        .limit(1)
    )


def actions_for_week(session: Session, user_id: UUID, week: str) -> list[ActionRow]:
    """Return this user's actions in stable rank and creation order."""
    statement = (
        select(ActionRow)
        .where(ActionRow.user_id == user_id, ActionRow.week == week)
        .order_by(ActionRow.rank, ActionRow.created_at, ActionRow.id)
    )
    return list(session.scalars(statement))
