"""Tenant-scoped persistence helpers for manual assets."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from wealth.db.models import HouseholdAccess, HouseholdMember, ManualAsset, Profile, User
from wealth.schemas.assets import AnyAsset, parse_asset


def add_manual_asset(
    session: Session,
    user_id: UUID,
    asset: AnyAsset,
    *,
    account_id: UUID | None = None,
    value_cached: Decimal | None = None,
    value_as_of: date | None = None,
    status: Literal["active", "closed"] = "active",
) -> ManualAsset:
    """Validate and add one C02 asset model to the user's manual asset rows."""
    validated = parse_asset(asset.model_dump(mode="json"))
    member_id = (
        UUID(validated.member_id)
        if validated.member_id is not None
        else ensure_self_household_member(session, user_id).id
    )
    record = ManualAsset(
        user_id=user_id,
        member_id=member_id,
        joint=validated.joint,
        account_id=account_id,
        asset_type=validated.asset_type,
        details=validated.model_dump(mode="json"),
        value_cached=value_cached,
        value_as_of=value_as_of,
        status=status,
    )
    session.add(record)
    session.flush()
    return record


def ensure_self_household_member(
    session: Session, user_id: UUID, *, display_name: str | None = None
) -> HouseholdMember:
    """Return the user's self member, creating a compatible row if necessary."""
    member = session.scalar(
        select(HouseholdMember).where(
            HouseholdMember.user_id == user_id, HouseholdMember.is_self.is_(True)
        )
    )
    if member is not None:
        return member

    user = session.get(User, user_id)
    profile = session.scalar(select(Profile).where(Profile.user_id == user_id))
    profile_residency = (
        dict(profile.residency or {}) if profile is not None else {"status": "RI", "country": "IN"}
    )
    profile_tax: dict[str, object] = {}
    if profile is not None:
        financial = dict(profile.financial_context or {})
        investing = dict(profile.investing or {})
        profile_tax = {
            key: value
            for key, value in {
                "tax_regime": profile.tax_regime,
                "tax_slab_pct": profile.tax_slab_pct,
                "birth_year": investing.get("birth_year"),
                "annual_income": financial.get("annual_income"),
                "employment_type": financial.get("employment_type"),
            }.items()
            if value is not None
        }
    member = HouseholdMember(
        user_id=user_id,
        display_name=display_name or (user.display_name if user is not None else None) or "You",
        relationship="self",
        is_self=True,
        residency=profile_residency,
        tax=profile_tax,
        investing=dict(profile.investing or {}) if profile is not None else {},
    )
    session.add(member)
    session.flush()
    return member


def acting_household_member(
    session: Session,
    actor_user_id: UUID,
    household_owner_id: UUID,
    *,
    display_name: str = "You",
) -> HouseholdMember:
    """Return the logged-in person's member row in the selected household."""
    if actor_user_id == household_owner_id:
        return ensure_self_household_member(session, household_owner_id, display_name=display_name)

    access = session.get(HouseholdAccess, (household_owner_id, actor_user_id))
    if access is None or access.revoked_at is not None:
        raise LookupError("The household member link is not active")
    member = session.get(HouseholdMember, access.member_id)
    if member is None or member.user_id != household_owner_id:
        raise LookupError("The household member link is invalid")
    return member


def get_manual_asset(session: Session, user_id: UUID, asset_id: UUID) -> ManualAsset | None:
    """Return one manual asset only when it belongs to the requested user."""
    return session.scalar(
        select(ManualAsset).where(ManualAsset.user_id == user_id, ManualAsset.id == asset_id)
    )


def list_manual_assets(session: Session, user_id: UUID) -> list[ManualAsset]:
    """Return a user's manual assets in stable creation order."""
    statement = (
        select(ManualAsset)
        .where(ManualAsset.user_id == user_id)
        .order_by(ManualAsset.created_at, ManualAsset.id)
    )
    return list(session.scalars(statement))
