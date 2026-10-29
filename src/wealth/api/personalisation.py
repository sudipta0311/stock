"""Personalisation preferences, weekly plan, actions, and stress endpoints."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from wealth.db.models import Action as ActionRow
from wealth.db.models import AuditLog, Profile, ResearchPackRecord
from wealth.policy.actions import SatelliteIdea
from wealth.policy.data import load_policy_inputs
from wealth.policy.personalise import WeeklyPlan, personalise
from wealth.policy.store import actions_for_week, latest_plan, save_weekly_plan
from wealth.schemas.common import Rupees
from wealth.schemas.profile import InvestingPrefs, Residency, ResidencyInput, residency_from_stored


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WeeklyPlanResponse(WeeklyPlan):
    fresh: bool
    me_member_id: UUID


class ActionResponse(ApiModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, from_attributes=True)

    id: UUID
    week: str
    rank: int
    kind: str
    payload: dict[str, Any]
    status: Literal["proposed", "done", "dismissed"]
    done_by: UUID | None = None
    created_at: datetime


class ActionStatusUpdate(ApiModel):
    status: Literal["done", "dismissed"]


class StressResponse(ApiModel):
    week: str
    as_of: date
    stress: dict[str, Any]


class ResearchIdeaSummary(ApiModel):
    security_id: str
    symbol: str
    market: Literal["IN", "US"]
    sector: str
    price: Rupees
    quant_decile: int
    dossier_status: str
    valuation_percentile: str
    confidence: str


class ResearchIdeasResponse(ApiModel):
    week: str
    ideas: list[ResearchIdeaSummary]


def current_week(on: date) -> str:
    year, week, _weekday = on.isocalendar()
    return f"{year:04d}-W{week:02d}"


def _valid_week(value: str) -> bool:
    try:
        year_text, week_text = value.split("-W")
        date.fromisocalendar(int(year_text), int(week_text), 1)
    except (ValueError, AttributeError):
        return False
    return True


def _compute_if_stale(user: Any, today: date) -> WeeklyPlanResponse:
    week = current_week(today)
    household_owner_id = user.household_owner_id
    stored = latest_plan(user.session, household_owner_id)
    user_data, market, research = load_policy_inputs(
        user.session,
        household_owner_id,
        week,
        today,
        deployment_mode=getattr(user, "deployment_mode", "personal"),
    )
    computed = personalise(user_data, market, research, week, today)
    from wealth.api.fund_check import add_weekly_fund_action

    computed = computed.model_copy(
        update={
            "actions": add_weekly_fund_action(
                user.session,
                household_owner_id,
                today,
                computed.actions,
                can_write=getattr(user, "role", "owner") != "viewer",
            )
        }
    )
    from wealth.db.repo import acting_household_member

    me_member_id = acting_household_member(user.session, user.user_id, household_owner_id).id
    if stored is not None and stored.week == week:
        saved = WeeklyPlan.model_validate(stored.plan)
        if saved.inputs_digest == computed.inputs_digest and saved.actions == computed.actions:
            return WeeklyPlanResponse(
                **saved.model_dump(mode="python"), fresh=False, me_member_id=me_member_id
            )
    if getattr(user, "role", "owner") == "viewer":
        return WeeklyPlanResponse(
            **computed.model_dump(mode="python"), fresh=False, me_member_id=me_member_id
        )
    save_weekly_plan(user.session, household_owner_id, computed)
    return WeeklyPlanResponse(
        **computed.model_dump(mode="python"), fresh=True, me_member_id=me_member_id
    )


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    router = APIRouter()
    from wealth.api.money_map import get_today

    user_dependency = Depends(authenticated_user)
    today_dependency = Depends(get_today)

    @router.get("/v1/me/residency", response_model=Residency)
    def get_residency(user: Any = user_dependency) -> Residency:
        from wealth.db.repo import acting_household_member

        member = acting_household_member(user.session, user.user_id, user.household_owner_id)
        return residency_from_stored(member.residency)

    @router.put("/v1/me/residency", response_model=Residency)
    def put_residency(payload: ResidencyInput, user: Any = user_dependency) -> Residency:
        from wealth.db.repo import acting_household_member

        member = acting_household_member(user.session, user.user_id, user.household_owner_id)
        before = (
            dict(member.residency or {})
            if member.residency is not None
            else Residency().model_dump(mode="json")
        )
        member.residency = payload.model_dump(mode="json")
        if user.user_id == user.household_owner_id:
            profile = user.session.scalar(
                select(Profile).where(Profile.user_id == user.household_owner_id)
            )
            if profile is None:
                profile = Profile(
                    user_id=user.household_owner_id,
                    residency=payload.model_dump(mode="json"),
                )
                user.session.add(profile)
            else:
                profile.residency = payload.model_dump(mode="json")
        user.session.add(
            AuditLog(
                user_id=user.household_owner_id,
                actor=str(user.user_id),
                actor_user_id=user.user_id,
                action="residency_changed",
                detail={
                    "member_id": str(member.id),
                    "from": before,
                    "to": payload.model_dump(mode="json"),
                },
            )
        )
        user.session.flush()
        return Residency.model_validate(payload.model_dump())

    @router.get("/v1/me/investing", response_model=InvestingPrefs)
    def get_investing(user: Any = user_dependency) -> InvestingPrefs:
        from wealth.db.repo import acting_household_member

        member = acting_household_member(user.session, user.user_id, user.household_owner_id)
        if member.investing:
            return InvestingPrefs.model_validate(member.investing)
        if user.user_id != user.household_owner_id:
            return InvestingPrefs()
        profile = user.session.scalar(
            select(Profile).where(Profile.user_id == user.household_owner_id)
        )
        return (
            InvestingPrefs.model_validate(profile.investing or {}) if profile else InvestingPrefs()
        )

    @router.put("/v1/me/investing", response_model=InvestingPrefs)
    def put_investing(payload: InvestingPrefs, user: Any = user_dependency) -> InvestingPrefs:
        from wealth.db.repo import acting_household_member

        document = payload.model_dump(mode="json")
        if user.user_id == user.household_owner_id:
            statement = pg_insert(Profile).values(
                id=uuid4(),
                user_id=user.household_owner_id,
                investing=document,
            )
            user.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[Profile.user_id],
                    set_={"investing": statement.excluded.investing, "updated_at": text("now()")},
                )
            )
        member = acting_household_member(user.session, user.user_id, user.household_owner_id)
        member.investing = document
        return payload

    @router.get("/v1/plan", response_model=WeeklyPlanResponse)
    def get_plan(user: Any = user_dependency, today: date = today_dependency) -> WeeklyPlanResponse:
        return _compute_if_stale(user, today)

    @router.get("/v1/actions", response_model=list[ActionResponse])
    def get_actions(
        user: Any = user_dependency,
        today: date = today_dependency,
        week: Annotated[str | None, Query(pattern=r"^\d{4}-W\d{2}$")] = None,
    ) -> list[ActionRow]:
        selected_week = week or current_week(today)
        if not _valid_week(selected_week):
            raise HTTPException(status_code=422, detail="The requested ISO week is invalid.")
        return actions_for_week(user.session, user.household_owner_id, selected_week)

    @router.patch("/v1/actions/{action_id}", response_model=ActionResponse)
    def patch_action(
        action_id: UUID,
        payload: ActionStatusUpdate,
        user: Any = user_dependency,
    ) -> ActionRow:
        action = user.session.scalar(
            select(ActionRow).where(
                ActionRow.user_id == user.household_owner_id,
                ActionRow.id == action_id,
            )
        )
        if action is None:
            raise HTTPException(status_code=404, detail="The action was not found.")
        previous_status = action.status
        action.status = payload.status
        action.done_by = user.user_id if payload.status == "done" else None
        user.session.add(
            AuditLog(
                user_id=user.household_owner_id,
                actor=str(user.user_id),
                actor_user_id=user.user_id,
                action="action_status_changed",
                detail={
                    "action_id": str(action.id),
                    "week": action.week,
                    "from": previous_status,
                    "to": payload.status,
                },
            )
        )
        user.session.flush()
        return action

    @router.get("/v1/stress", response_model=StressResponse)
    def get_stress(user: Any = user_dependency, today: date = today_dependency) -> StressResponse:
        plan = _compute_if_stale(user, today)
        return StressResponse(week=plan.week, as_of=plan.as_of, stress=plan.stress)

    @router.get("/v1/research/ideas", response_model=ResearchIdeasResponse)
    def get_research_ideas(
        user: Any = user_dependency,
        today: date = today_dependency,
        week: Annotated[str | None, Query(pattern=r"^\d{4}-W\d{2}$")] = None,
    ) -> ResearchIdeasResponse:
        selected_week = week or current_week(today)
        if not _valid_week(selected_week):
            raise HTTPException(status_code=422, detail="The requested ISO week is invalid.")
        record = user.session.get(ResearchPackRecord, selected_week)
        if record is None:
            return ResearchIdeasResponse(week=selected_week, ideas=[])
        ideas: list[ResearchIdeaSummary] = []
        for raw in record.pack.get("ideas", []):
            idea = SatelliteIdea.model_validate(raw)
            ideas.append(
                ResearchIdeaSummary(
                    security_id=idea.security_id,
                    symbol=idea.symbol,
                    market=idea.market,
                    sector=idea.sector,
                    price=idea.price,
                    quant_decile=idea.quant_decile,
                    dossier_status=idea.dossier_status,
                    valuation_percentile=format(idea.valuation_percentile, "f"),
                    confidence=idea.confidence,
                )
            )
        return ResearchIdeasResponse(week=selected_week, ideas=ideas)

    return router
