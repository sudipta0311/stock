"""Household invitations and shared access endpoints."""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError

from wealth.db.models import AuditLog, HouseholdAccess, HouseholdInvite, HouseholdMember, User


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InviteCreate(ApiModel):
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    role: Literal["partner", "viewer"] = "partner"
    member_id: UUID | None = None
    send_email: bool = False


class InviteJoin(ApiModel):
    token: str = Field(min_length=43, max_length=256)
    bring_my_data: bool = False


class InviteResponse(ApiModel):
    id: UUID
    email: str
    role: Literal["partner", "viewer"]
    member_id: UUID | None
    expires_at: datetime
    accepted_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime


class InviteCreatedResponse(InviteResponse):
    share_link: str
    email_sent: bool = False


class HouseholdAccessPatch(ApiModel):
    role: Literal["partner", "viewer"]


class HouseholdAccessResponse(ApiModel):
    user_id: UUID
    display_name: str | None
    role: Literal["partner", "viewer"]
    member_id: UUID
    created_at: datetime
    revoked_at: datetime | None


class HouseholdSummary(ApiModel):
    owner_id: UUID
    owner_name: str | None
    role: Literal["owner", "partner", "viewer"]
    member_count: int


class HouseholdJoinResponse(ApiModel):
    owner_id: UUID
    merge_counts: dict[str, int] | None = None
    overwritten_fields: list[str] = Field(default_factory=list)


def _require_owner(user: Any) -> None:
    if user.role != "owner":
        raise HTTPException(status_code=403, detail="Only the household owner can manage access.")


def _audit(
    user: Any,
    action: str,
    detail: dict[str, Any],
    *,
    owner_id: UUID | None = None,
) -> None:
    user.session.add(
        AuditLog(
            user_id=owner_id or user.household_owner_id,
            actor=str(user.user_id),
            actor_user_id=user.user_id,
            action=action,
            detail=detail,
        )
    )


def _send_invite_email(request: Request, invitee: str, link: str, invite_id: UUID) -> bool:
    if not os.getenv("EMAIL_PROVIDER"):
        return False
    from wealth.alerts.delivery import render_notification
    from wealth.alerts.email import OutgoingEmail, send_email
    from wealth.alerts.settings import AlertSettings

    settings = AlertSettings.from_env()
    subject = "You have a household invitation"
    body = f"Open this link to join the household: {link}"
    html, text_body = render_notification(
        settings=settings,
        subject=subject,
        body=body,
        unsubscribe_url_value=settings.email_manage_alerts_url,
    )
    message = OutgoingEmail(
        to=invitee,
        subject=subject,
        html=html,
        text=text_body,
        from_address=settings.email_from,
        unsubscribe_url=settings.email_manage_alerts_url,
        dedupe_key=f"household-invite-{invite_id.hex}",
    )
    send_email(
        message,
        settings,
        dry_run=getattr(request.app.state.settings, "env", "production") == "test",
    )
    return True


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    router = APIRouter()
    user_dependency = Depends(authenticated_user)

    @router.get("/v1/households", response_model=list[HouseholdSummary])
    def list_households(user: Any = user_dependency) -> list[HouseholdSummary]:
        accesses = list(
            user.session.scalars(
                select(HouseholdAccess).where(
                    HouseholdAccess.grantee_user_id == user.user_id,
                    HouseholdAccess.revoked_at.is_(None),
                )
            )
        )
        households = [(user.user_id, "owner", user.session.get(User, user.user_id))]
        households.extend((access.owner_user_id, access.role, None) for access in accesses)
        original_owner = user.household_owner_id
        summaries: list[HouseholdSummary] = []
        for owner_id, role, owner_user in households:
            user.session.execute(
                text("SELECT set_config('app.household_owner', :owner_id, true)"),
                {"owner_id": str(owner_id)},
            )
            owner_name = (
                owner_user.display_name
                if owner_user is not None
                else user.session.scalar(
                    text(
                        "SELECT display_name FROM household_people "
                        "WHERE owner_id = :owner_id AND user_id = :owner_id"
                    ),
                    {"owner_id": owner_id},
                )
            )
            count = user.session.scalar(
                select(func.count())
                .select_from(HouseholdMember)
                .where(HouseholdMember.user_id == owner_id)
            )
            summaries.append(
                HouseholdSummary(
                    owner_id=owner_id,
                    owner_name=owner_name,
                    role=role,
                    member_count=int(count or 0),
                )
            )
        user.session.execute(
            text("SELECT set_config('app.household_owner', :owner_id, true)"),
            {"owner_id": str(original_owner)},
        )
        return summaries

    @router.post("/v1/households/invites", response_model=InviteCreatedResponse, status_code=201)
    def create_invite(
        payload: InviteCreate,
        request: Request,
        user: Any = user_dependency,
    ) -> InviteCreatedResponse:
        _require_owner(user)
        if payload.member_id is not None:
            member = user.session.scalar(
                select(HouseholdMember)
                .where(
                    HouseholdMember.user_id == user.household_owner_id,
                    HouseholdMember.id == payload.member_id,
                )
                .with_for_update()
            )
            if member is None:
                raise HTTPException(status_code=404, detail="The household member was not found.")
            if member.is_self:
                raise HTTPException(status_code=409, detail="member_is_owner")
            active_grantee = user.session.scalar(
                select(HouseholdAccess.grantee_user_id)
                .where(
                    HouseholdAccess.owner_user_id == user.household_owner_id,
                    HouseholdAccess.member_id == payload.member_id,
                    HouseholdAccess.revoked_at.is_(None),
                )
                .limit(1)
            )
            pending_invite = user.session.scalar(
                select(HouseholdInvite.id)
                .where(
                    HouseholdInvite.owner_user_id == user.household_owner_id,
                    HouseholdInvite.member_id == payload.member_id,
                    HouseholdInvite.email_lower != str(payload.email).lower(),
                    HouseholdInvite.accepted_at.is_(None),
                    HouseholdInvite.revoked_at.is_(None),
                    HouseholdInvite.expires_at > datetime.now(UTC),
                )
                .limit(1)
            )
            if active_grantee is not None or pending_invite is not None:
                raise HTTPException(status_code=409, detail="member_already_linked")

        token = secrets.token_urlsafe(32)
        invite = HouseholdInvite(
            owner_user_id=user.household_owner_id,
            email_lower=str(payload.email).lower(),
            role=payload.role,
            member_id=payload.member_id,
            token_sha256=hashlib.sha256(token.encode("utf-8")).hexdigest(),
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )
        user.session.add(invite)
        user.session.flush()
        _audit(
            user,
            "household_invite_created",
            {
                "invite_id": str(invite.id),
                "role": payload.role,
                "member_id": str(payload.member_id) if payload.member_id is not None else None,
            },
        )
        base_url = request.app.state.settings.email_public_base_url.rstrip("/")
        share_link = f"{base_url}/join?token={quote(token)}"
        email_sent = (
            _send_invite_email(request, str(payload.email), share_link, invite.id)
            if payload.send_email
            else False
        )
        return InviteCreatedResponse(
            id=invite.id,
            email=invite.email_lower,
            role=invite.role,
            member_id=invite.member_id,
            expires_at=invite.expires_at,
            accepted_at=invite.accepted_at,
            revoked_at=invite.revoked_at,
            created_at=invite.created_at,
            share_link=share_link,
            email_sent=email_sent,
        )

    @router.get("/v1/households/invites", response_model=list[InviteResponse])
    def list_invites(user: Any = user_dependency) -> list[InviteResponse]:
        _require_owner(user)
        invites = list(
            user.session.scalars(
                select(HouseholdInvite)
                .where(HouseholdInvite.owner_user_id == user.household_owner_id)
                .order_by(HouseholdInvite.created_at.desc(), HouseholdInvite.id)
            )
        )
        return [
            InviteResponse(
                id=invite.id,
                email=invite.email_lower,
                role=invite.role,
                member_id=invite.member_id,
                expires_at=invite.expires_at,
                accepted_at=invite.accepted_at,
                revoked_at=invite.revoked_at,
                created_at=invite.created_at,
            )
            for invite in invites
        ]

    @router.delete("/v1/households/invites/{invite_id}", status_code=204)
    def revoke_invite(invite_id: UUID, user: Any = user_dependency) -> None:
        _require_owner(user)
        invite = user.session.scalar(
            select(HouseholdInvite).where(
                HouseholdInvite.id == invite_id,
                HouseholdInvite.owner_user_id == user.household_owner_id,
            )
        )
        if invite is None:
            raise HTTPException(status_code=404, detail="The invite was not found.")
        invite.revoked_at = datetime.now(UTC)
        _audit(user, "household_invite_revoked", {"invite_id": str(invite.id)})

    @router.post("/v1/households/join", response_model=HouseholdJoinResponse)
    def join_household(payload: InviteJoin, user: Any = user_dependency) -> HouseholdJoinResponse:
        try:
            with user.session.begin_nested():
                owner_id = user.session.scalar(
                    text("SELECT app_accept_invite(:token)"), {"token": payload.token}
                )
        except SQLAlchemyError as error:
            diagnostic = getattr(getattr(error, "orig", None), "diag", None)
            if getattr(diagnostic, "message_primary", None) == "invalid_invite":
                raise HTTPException(status_code=400, detail="invalid_invite") from error
            raise
        if owner_id is None:
            raise HTTPException(status_code=400, detail="invalid_invite")
        owner_id = owner_id if isinstance(owner_id, UUID) else UUID(str(owner_id))
        if not payload.bring_my_data:
            return HouseholdJoinResponse(owner_id=owner_id)
        user.session.execute(
            text("SELECT set_config('app.household_owner', :owner_id, true)"),
            {"owner_id": str(owner_id)},
        )
        try:
            result = user.session.scalar(
                text("SELECT app_merge_into_household(:owner_id)"), {"owner_id": owner_id}
            )
        except SQLAlchemyError as error:
            diagnostic = getattr(getattr(error, "orig", None), "diag", None)
            message = getattr(diagnostic, "message_primary", None)
            if message in {"merge_not_allowed", "revoke_outgoing_access_first"}:
                raise HTTPException(
                    status_code=409, detail="The household data could not be merged."
                ) from error
            raise
        result = result or {}
        return HouseholdJoinResponse(
            owner_id=owner_id,
            merge_counts=result.get("counts", {}),
            overwritten_fields=result.get("overwritten_fields", []),
        )

    @router.get("/v1/households/access", response_model=list[HouseholdAccessResponse])
    def list_access(user: Any = user_dependency) -> list[HouseholdAccessResponse]:
        _require_owner(user)
        accesses = list(
            user.session.scalars(
                select(HouseholdAccess)
                .where(HouseholdAccess.owner_user_id == user.household_owner_id)
                .order_by(HouseholdAccess.created_at, HouseholdAccess.grantee_user_id)
            )
        )
        people = {
            row.user_id: row.display_name
            for row in user.session.execute(
                text(
                    "SELECT user_id, display_name FROM household_people WHERE owner_id = :owner_id"
                ),
                {"owner_id": user.household_owner_id},
            )
        }
        return [
            HouseholdAccessResponse(
                user_id=access.grantee_user_id,
                display_name=people.get(access.grantee_user_id),
                role=access.role,
                member_id=access.member_id,
                created_at=access.created_at,
                revoked_at=access.revoked_at,
            )
            for access in accesses
        ]

    @router.patch("/v1/households/access/{grantee_user_id}", response_model=HouseholdAccessResponse)
    def update_access(
        grantee_user_id: UUID,
        payload: HouseholdAccessPatch,
        user: Any = user_dependency,
    ) -> HouseholdAccessResponse:
        _require_owner(user)
        access = user.session.get(HouseholdAccess, (user.household_owner_id, grantee_user_id))
        if access is None or access.revoked_at is not None:
            raise HTTPException(status_code=404, detail="The household access was not found.")
        previous_role = access.role
        access.role = payload.role
        user.session.flush()
        _audit(
            user,
            "household_access_role_changed",
            {"grantee_user_id": str(grantee_user_id), "from": previous_role, "to": payload.role},
        )
        name = user.session.scalar(
            text(
                "SELECT display_name FROM household_people "
                "WHERE owner_id = :owner_id AND user_id = :user_id"
            ),
            {"owner_id": user.household_owner_id, "user_id": grantee_user_id},
        )
        return HouseholdAccessResponse(
            user_id=access.grantee_user_id,
            display_name=name,
            role=access.role,
            member_id=access.member_id,
            created_at=access.created_at,
            revoked_at=access.revoked_at,
        )

    @router.delete("/v1/households/access/{grantee_user_id}", status_code=204)
    def revoke_access(grantee_user_id: UUID, user: Any = user_dependency) -> None:
        _require_owner(user)
        access = user.session.get(HouseholdAccess, (user.household_owner_id, grantee_user_id))
        if access is None or access.revoked_at is not None:
            raise HTTPException(status_code=404, detail="The household access was not found.")
        access.revoked_at = datetime.now(UTC)
        _audit(user, "household_access_revoked", {"grantee_user_id": str(grantee_user_id)})

    @router.delete("/v1/households/leave", status_code=204)
    def leave_household(user: Any = user_dependency) -> None:
        try:
            user.session.scalar(
                text("SELECT app_leave_household(:owner_id)"),
                {"owner_id": user.household_owner_id},
            )
        except SQLAlchemyError as error:
            diagnostic = getattr(getattr(error, "orig", None), "diag", None)
            if getattr(diagnostic, "message_primary", None) == "not_found":
                raise HTTPException(
                    status_code=404, detail="The household was not found."
                ) from error
            raise

    return router
