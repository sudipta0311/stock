"""Tenant-scoped persistence for computed weekly plans and proposed actions."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from wealth.db.models import Action as ActionRow
from wealth.db.models import WeeklyPlan as WeeklyPlanRow
from wealth.policy.personalise import WeeklyPlan


def save_weekly_plan(
    session: Session, user_id: UUID, plan: WeeklyPlan, *, replace_proposed_actions: bool = True
) -> None:
    """Upsert a plan while retaining ids and status for unchanged actions."""
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
    existing = list(
        session.scalars(
            select(ActionRow)
            .where(
                ActionRow.user_id == user_id,
                ActionRow.week == plan.week,
            )
            .order_by(ActionRow.rank, ActionRow.id)
        )
    )
    for action in plan.actions:
        kind = str(action["kind"])
        member_id = action.get("member_id")
        identity_payload = {
            key: value for key, value in action.items() if key not in {"rank", "kind"}
        }
        match_index = next(
            (
                index
                for index, row in enumerate(existing)
                if row.kind == kind
                and row.payload.get("member_id") == member_id
                and {
                    key: value for key, value in row.payload.items() if key not in {"rank", "kind"}
                }
                == identity_payload
            ),
            None,
        )
        if match_index is None:
            session.add(
                ActionRow(
                    user_id=user_id,
                    week=plan.week,
                    rank=int(action["rank"]),
                    kind=kind,
                    payload=action,
                    status="proposed",
                )
            )
            continue

        matched = existing.pop(match_index)
        matched.rank = int(action["rank"])
        matched.payload = action

    for row in existing:
        if row.status == "proposed":
            session.delete(row)
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
