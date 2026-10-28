"""Consent, push-subscription, unsubscribe, and tenant-scoped inbox endpoints."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, text

from wealth.alerts.brief import Brief, build_brief, personalize_plan_for_viewer
from wealth.alerts.consent import (
    CONSENT_PURPOSES,
    consent_document,
    set_consent,
    verify_unsubscribe_token,
    withdraw_consent,
)
from wealth.alerts.push import PushSubscriptionInput
from wealth.db.models import (
    AlertLog,
    HouseholdAccess,
    HouseholdMember,
    NetworthSnapshot,
    PushSubscription,
    User,
)
from wealth.db.models import (
    WeeklyPlan as WeeklyPlanRow,
)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ConsentValues(ApiModel):
    weekly_email: bool = False
    reminder_email: bool = False
    push: bool = False
    urgent_alerts: bool = False


class PushSubscriptionResponse(ApiModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, from_attributes=True)

    id: UUID
    endpoint: str
    created_at: datetime


class InboxItem(ApiModel):
    id: str
    kind: str
    subject: str
    week: str | None = None
    status: str | None = None
    sent_at: datetime | None = None


class InboxResponse(ApiModel):
    items: list[InboxItem]


def _week_is_valid(week: str) -> bool:
    try:
        year_text, week_text = week.split("-W")
        date.fromisocalendar(int(year_text), int(week_text), 1)
    except (ValueError, AttributeError):
        return False
    return True


def _settings(request: Request) -> Any:
    return request.app.state.settings


def _manage_url(settings: Any) -> str:
    return str(getattr(settings, "email_manage_alerts_url", "https://app.example/settings/alerts"))


def _brief_user(settings: Any) -> dict[str, str]:
    manage = _manage_url(settings)
    return {
        "app_base_url": str(getattr(settings, "email_public_base_url", "https://app.example")),
        "manage_alerts_url": manage,
        "unsubscribe_url": manage,
        "disclaimer": str(getattr(settings, "email_disclaimer", "Research and education only.")),
        "postal_address": str(
            getattr(settings, "email_postal_address", "Postal address configured by sender")
        ),
    }


def _latest_networth(session: Any, user_id: UUID) -> dict[str, Any]:
    snapshot = session.scalar(
        select(NetworthSnapshot)
        .where(NetworthSnapshot.user_id == user_id)
        .order_by(NetworthSnapshot.on_date.desc())
        .limit(1)
    )
    return snapshot.totals if snapshot is not None else {"net_worth": None, "maturity_calendar": []}


def _brief_for_row(
    session: Any, row: WeeklyPlanRow, settings: Any, *, actor_user_id: UUID
) -> Brief:
    networth = _latest_networth(session, row.user_id)
    owner_id = row.user_id
    members = {
        member.id: member
        for member in session.scalars(
            select(HouseholdMember).where(HouseholdMember.user_id == owner_id)
        )
    }
    own_member_id = next((member.id for member in members.values() if member.is_self), None)
    if actor_user_id == owner_id:
        me_member_id = own_member_id
    else:
        access = session.get(HouseholdAccess, (owner_id, actor_user_id))
        me_member_id = access.member_id if access is not None else None
    if actor_user_id != owner_id:
        owner_name = session.scalar(
            text(
                "SELECT display_name FROM household_people "
                "WHERE owner_id = :owner_id AND user_id = :owner_id"
            ),
            {"owner_id": owner_id},
        )
    else:
        owner = session.get(User, owner_id)
        owner_name = owner.display_name if owner is not None else None
    plan = personalize_plan_for_viewer(
        row.plan,
        {str(member.id): (member.display_name, member.relationship) for member in members.values()},
        me_member_id=str(me_member_id) if me_member_id is not None else None,
        owner_self_member_id=str(own_member_id) if own_member_id is not None else None,
        owner_name=owner_name or "Household owner",
    )
    context = _brief_user(settings)
    context["household_header"] = f"Family plan · {owner_name or 'Your'}'s household"
    return build_brief(
        plan,
        networth,
        networth.get("maturity_calendar", []),
        None,
        context,
    )


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    router = APIRouter()
    user_dependency = Depends(authenticated_user)

    @router.get("/v1/me/consents", response_model=ConsentValues)
    def get_consents(user: Any = user_dependency) -> dict[str, bool]:
        return consent_document(user.session, user.user_id)

    @router.put("/v1/me/consents", response_model=ConsentValues)
    def put_consents(payload: ConsentValues, user: Any = user_dependency) -> ConsentValues:
        values = payload.model_dump()
        for purpose in CONSENT_PURPOSES:
            set_consent(user.session, user.user_id, purpose, values[purpose])
        return payload

    @router.post("/v1/push/subscriptions", response_model=PushSubscriptionResponse, status_code=201)
    def post_push_subscription(
        payload: PushSubscriptionInput, user: Any = user_dependency
    ) -> PushSubscription:
        endpoint = str(payload.endpoint)
        subscription = user.session.scalar(
            select(PushSubscription).where(
                PushSubscription.user_id == user.user_id,
                PushSubscription.endpoint == endpoint,
            )
        )
        if subscription is None:
            subscription = PushSubscription(
                user_id=user.user_id,
                endpoint=endpoint,
                keys=payload.keys,
            )
            user.session.add(subscription)
        else:
            subscription.keys = payload.keys
        user.session.flush()
        return subscription

    @router.delete(
        "/v1/push/subscriptions/{subscription_id}", status_code=204, response_class=Response
    )
    def delete_push_subscription(subscription_id: UUID, user: Any = user_dependency) -> Response:
        subscription = user.session.scalar(
            select(PushSubscription).where(
                PushSubscription.user_id == user.user_id,
                PushSubscription.id == subscription_id,
            )
        )
        if subscription is None:
            raise HTTPException(status_code=404, detail="The push subscription was not found.")
        user.session.delete(subscription)
        return Response(status_code=204)

    def unsubscribe_response(request: Request, token: str | None) -> HTMLResponse:
        settings = _settings(request)
        secret = getattr(settings, "email_link_secret_value", None)
        claims = verify_unsubscribe_token(token or "", secret or "")
        if claims is not None:
            engine = request.app.state.db_engine
            if engine is not None:
                from wealth.db.session import user_session

                with user_session(engine, claims.user_id) as session:
                    withdraw_consent(session, claims.user_id, claims.kind)
            content = (
                "<!doctype html><html><body><main><h1>Alerts updated</h1>"
                "<p>This alert preference has been turned off.</p></main></body></html>"
            )
            return HTMLResponse(content=content, status_code=200)
        return HTMLResponse(
            content=(
                "<!doctype html><html><body><main><h1>Link unavailable</h1>"
                "<p>This link is invalid or expired.</p></main></body></html>"
            ),
            status_code=400,
        )

    @router.get("/v1/email/unsubscribe", response_class=HTMLResponse)
    def get_unsubscribe(
        request: Request, token: Annotated[str | None, Query()] = None
    ) -> HTMLResponse:
        return unsubscribe_response(request, token)

    @router.post("/v1/email/unsubscribe", response_class=HTMLResponse)
    def post_unsubscribe(
        request: Request, token: Annotated[str | None, Query()] = None
    ) -> HTMLResponse:
        return unsubscribe_response(request, token)

    @router.get("/v1/inbox", response_model=InboxResponse)
    def get_inbox(
        request: Request,
        user: Any = user_dependency,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> InboxResponse:
        settings = _settings(request)
        plans = list(
            user.session.scalars(
                select(WeeklyPlanRow)
                .where(WeeklyPlanRow.user_id == user.household_owner_id)
                .order_by(WeeklyPlanRow.week.desc())
                .limit(limit)
            )
        )
        logs = list(
            user.session.scalars(
                select(AlertLog)
                .where(AlertLog.user_id == user.household_owner_id)
                .order_by(AlertLog.sent_at.desc(), AlertLog.id)
                .limit(limit)
            )
        )
        items: list[InboxItem] = []
        plan_weeks: set[str] = set()
        for log in logs:
            plan = next((item for item in plans if item.week == log.week), None)
            status = None
            if log.kind == "weekly" and log.week:
                plan_weeks.add(log.week)
                if plan is not None:
                    status = "ACTIONS" if plan.plan.get("actions") else "NONE"
            items.append(
                InboxItem(
                    id=str(log.id),
                    kind=log.kind,
                    subject=log.subject,
                    week=log.week,
                    status=status,
                    sent_at=log.sent_at,
                )
            )
        for plan in plans:
            if plan.week in plan_weeks:
                continue
            brief = _brief_for_row(user.session, plan, settings, actor_user_id=user.user_id)
            items.append(
                InboxItem(
                    id=f"brief:{plan.week}",
                    kind="weekly",
                    subject=brief.subject,
                    week=plan.week,
                    status=brief.status,
                    sent_at=plan.created_at,
                )
            )
        items.sort(key=lambda item: item.sent_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return InboxResponse(items=items[:limit])

    @router.get("/v1/brief/{week}", response_model=Brief)
    def get_brief(
        week: str,
        request: Request,
        user: Any = user_dependency,
    ) -> Brief:
        if not _week_is_valid(week):
            raise HTTPException(status_code=422, detail="The requested ISO week is invalid.")
        row = user.session.get(WeeklyPlanRow, (user.household_owner_id, week))
        if row is None:
            raise HTTPException(status_code=404, detail="The weekly brief was not found.")
        return _brief_for_row(user.session, row, _settings(request), actor_user_id=user.user_id)

    return router
