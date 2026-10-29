"""Household-scoped Indian mutual-fund checks."""

from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from wealth.core.dates import fy_start
from wealth.db.models import (
    FundLaggingCheck,
    FundTransaction,
    HoldingLot,
    HouseholdMember,
    NavDaily,
    NavMonthly,
    Profile,
    RealisedTrade,
    Security,
)
from wealth.db.models import (
    WeeklyPlan as WeeklyPlanRow,
)
from wealth.policy.fund_check import (
    FundCheckResult,
    FundHolding,
    FundMember,
    FundReference,
    FundTransactionInput,
    check_funds,
)
from wealth.policy.tax import Lot, TaxContext
from wealth.schemas.profile import MemberTax, residency_from_stored


class FundCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    summary_key: str
    checks: list[dict[str, Any]] = Field(max_length=3)
    early_warning: dict[str, Any] | None
    details: dict[str, Any]


class FundSipOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    member_id: UUID
    monthly_amount_inr: Decimal = Field(ge=0, max_digits=14, decimal_places=2)


def _month_end(value: date) -> date:
    return date(value.year, value.month, monthrange(value.year, value.month)[1])


def _previous_month_end(value: date) -> date:
    if value.month == 1:
        year, month = value.year - 1, 12
    else:
        year, month = value.year, value.month - 1
    return date(year, month, monthrange(year, month)[1])


def _tax_lots(
    lots: list[HoldingLot], security: Security, price: Decimal | None
) -> tuple[tuple[Lot, ...], bool]:
    tax_class = None
    if security.category == "mf:equity_global":
        tax_class = "intl_fof"
    elif security.category == "mf:equity_india" and security.sub_category not in {
        "Aggressive Hybrid",
        "Balanced Hybrid",
        "Conservative Hybrid",
        "Index Funds/ETFs",
    }:
        tax_class = "equity_mf"
    elif security.category and security.category.startswith("mf:debt"):
        tax_class = "debt_mf_post_2023"

    result: list[Lot] = []
    complete = bool(lots and price is not None and tax_class is not None)
    for row in lots:
        if row.acquired_on is None or row.cost_per_unit is None or tax_class is None:
            complete = False
            continue
        if tax_class == "debt_mf_post_2023" and row.acquired_on < date(2023, 4, 1):
            complete = False
            continue
        try:
            result.append(
                Lot(
                    lot_id=str(row.id),
                    tax_class=tax_class,
                    quantity=row.quantity,
                    cost_per_unit=row.cost_per_unit,
                    acquired_on=row.acquired_on,
                    market_price=price,
                )
            )
        except ValueError:
            complete = False
    if len(result) != len(lots):
        complete = False
    return tuple(result), complete


def _member_rows(
    session: Session,
    owner_id: UUID,
    profile: Profile | None,
    plan: Mapping[str, Any],
    today: date | None = None,
) -> tuple[list[FundMember], dict[str, HouseholdMember]]:
    today = today or date.today()
    rows = list(
        session.scalars(
            select(HouseholdMember)
            .where(HouseholdMember.user_id == owner_id)
            .order_by(
                HouseholdMember.is_self.desc(), HouseholdMember.created_at, HouseholdMember.id
            )
        )
    )
    planned = {
        str(item.get("id")): item
        for item in plan.get("members", [])
        if isinstance(item, dict) and item.get("id")
    }
    profiles_by_id = {str(row.id): row for row in rows}
    result: list[FundMember] = []
    for row in rows:
        plan_member = planned.get(str(row.id), {})
        # C27a carries the same member residency in the current plan; the stored profile is
        # authoritative when an old plan has not yet been refreshed.
        plan_residency = plan_member.get("residency")
        residency = residency_from_stored(
            plan_residency if isinstance(plan_residency, dict) else row.residency
        )
        tax_values = dict(row.tax or {})
        if row.is_self and profile is not None:
            tax_values.setdefault("tax_slab_pct", profile.tax_slab_pct)
            tax_values.setdefault("surcharge_pct", "0")
        tax = MemberTax.model_validate(tax_values)
        realised_rows = session.execute(
            select(RealisedTrade.holding_period, func.sum(RealisedTrade.gain))
            .where(
                RealisedTrade.user_id == owner_id,
                RealisedTrade.member_id == row.id,
                RealisedTrade.trade_date >= fy_start(today),
                RealisedTrade.trade_date <= today,
            )
            .group_by(RealisedTrade.holding_period)
        )
        realised = {
            period: max(Decimal(0), amount or Decimal(0)) for period, amount in realised_rows
        }
        tax = tax.model_copy(
            update={
                "fy_realised_equity_ltcg": max(
                    tax.fy_realised_equity_ltcg, realised.get("LT", Decimal(0))
                ),
                "fy_realised_equity_stcg": max(
                    tax.fy_realised_equity_stcg, realised.get("ST", Decimal(0))
                ),
            }
        )
        spouse = next(
            (
                candidate
                for candidate in rows
                if candidate.relationship == "spouse"
                and residency_from_stored(
                    planned.get(str(candidate.id), {}).get("residency")
                    if isinstance(planned.get(str(candidate.id), {}).get("residency"), dict)
                    else candidate.residency
                ).country
                == "IN"
                and residency_from_stored(
                    planned.get(str(candidate.id), {}).get("residency")
                    if isinstance(planned.get(str(candidate.id), {}).get("residency"), dict)
                    else candidate.residency
                ).status
                == "RI"
            ),
            None,
        )
        overrides = (row.investing or {}).get("fund_sip_overrides", {})
        sip_overrides: dict[str, Decimal] = {}
        if isinstance(overrides, dict):
            for security_id, amount in overrides.items():
                try:
                    value = Decimal(str(amount))
                except InvalidOperation:
                    continue
                if value.is_finite() and value >= 0:
                    sip_overrides[str(security_id)] = value
        member_tax = TaxContext(
            slab_pct=tax.tax_slab_pct or Decimal("30"),
            surcharge_pct=tax.surcharge_pct,
            fy_realised_equity_ltcg=tax.fy_realised_equity_ltcg,
            fy_realised_equity_stcg=tax.fy_realised_equity_stcg,
            residency_status=tax.residency_status
            if hasattr(tax, "residency_status")
            else residency.status,
            us_person=residency.us_person,
        )
        result.append(
            FundMember(
                member_id=str(row.id),
                residency_country=residency.country,
                residency_status=residency.status,
                tax_context=member_tax,
                redirect_member_id=str(spouse.id) if residency.country == "DK" and spouse else None,
                redirect_member_name=spouse.display_name
                if residency.country == "DK" and spouse
                else None,
                clubbing_note=(
                    "The RI spouse would hold redirected monthly investments; confirm "
                    "tax treatment."
                    if residency.country == "DK" and spouse
                    else None
                ),
                sip_overrides=sip_overrides,
            )
        )
    return result, profiles_by_id


def _build_inputs(
    session: Session,
    owner_id: UUID,
    today: date,
) -> tuple[
    list[FundHolding],
    list[FundReference],
    dict[str, list[tuple[date, Decimal]]],
    list[FundTransactionInput],
    list[FundMember],
    Decimal | None,
    dict[str, str] | None,
    set[tuple[str, str]],
]:
    profile = session.scalar(select(Profile).where(Profile.user_id == owner_id))
    plan_row = session.scalar(
        select(WeeklyPlanRow)
        .where(WeeklyPlanRow.user_id == owner_id)
        .order_by(WeeklyPlanRow.week.desc())
        .limit(1)
    )
    plan = plan_row.plan if plan_row is not None and isinstance(plan_row.plan, dict) else {}
    member_rows, _ = _member_rows(session, owner_id, profile, plan, today)
    lots = list(
        session.scalars(
            select(HoldingLot)
            .where(HoldingLot.user_id == owner_id, HoldingLot.quantity > 0)
            .order_by(
                HoldingLot.member_id, HoldingLot.security_id, HoldingLot.acquired_on, HoldingLot.id
            )
        )
    )
    lots_by_security: dict[tuple[str, str], list[HoldingLot]] = defaultdict(list)
    security_ids = {row.security_id for row in lots}
    for row in lots:
        lots_by_security[(str(row.member_id), str(row.security_id))].append(row)
    held_securities = (
        list(session.scalars(select(Security).where(Security.id.in_(security_ids))))
        if security_ids
        else []
    )
    held_by_id = {row.id: row for row in held_securities}
    references_query = select(Security).where(
        Security.market == "IN",
        Security.security_type == "mf",
        Security.plan_type == "direct",
        Security.option == "growth",
        Security.category.in_(("mf:equity_india", "mf:equity_global")),
    )
    references = [*held_securities, *session.scalars(references_query)]
    unique_references = {row.id: row for row in references}
    reference_rows = list(unique_references.values())
    all_ids = set(unique_references)
    history_rows = (
        session.execute(
            select(NavMonthly.security_id, NavMonthly.month_end, NavMonthly.nav)
            .where(
                NavMonthly.security_id.in_(all_ids),
                NavMonthly.month_end <= today,
            )
            .order_by(NavMonthly.security_id, NavMonthly.month_end)
        ).all()
        if all_ids
        else []
    )
    histories: dict[str, list[tuple[date, Decimal]]] = defaultdict(list)
    for security_id, month_end, nav in history_rows:
        histories[str(security_id)].append((month_end, nav))

    latest_daily: dict[UUID, tuple[date, Decimal]] = {}
    daily_rows = (
        session.execute(
            select(NavDaily.security_id, NavDaily.as_of, NavDaily.nav)
            .where(NavDaily.security_id.in_(security_ids), NavDaily.as_of <= today)
            .order_by(NavDaily.security_id, NavDaily.as_of.desc())
        ).all()
        if security_ids
        else []
    )
    for security_id, as_of, nav in daily_rows:
        latest_daily.setdefault(security_id, (as_of, nav))
    monthly_latest: dict[UUID, Decimal] = {}
    for security_id, _month_end, nav in reversed(history_rows):
        monthly_latest.setdefault(security_id, nav)

    holdings: list[FundHolding] = []
    for (member_id, security_id), security_lots in lots_by_security.items():
        security = held_by_id.get(UUID(security_id))
        if security is None or security.security_type != "mf":
            continue
        price_row = latest_daily.get(security.id)
        price = price_row[1] if price_row else monthly_latest.get(security.id)
        quantity = sum((row.quantity for row in security_lots), Decimal(0))
        value = (
            price * quantity
            if price is not None
            else sum((row.statement_value or Decimal(0) for row in security_lots), Decimal(0))
        )
        tax_lots, complete = _tax_lots(security_lots, security, price)
        holdings.append(
            FundHolding(
                security_id=security_id,
                name=security.name,
                category=security.category,
                sub_category=security.sub_category,
                plan_type=security.plan_type,
                option_type=security.option,
                ter=security.ter,
                member_id=member_id,
                value=value,
                quantity=quantity,
                price=price,
                tax_lots=tax_lots,
                tax_lots_complete=complete,
            )
        )
    reference_data = [
        FundReference(
            security_id=str(row.id),
            name=row.name,
            category=row.category,
            sub_category=row.sub_category,
            plan_type=row.plan_type,
            option_type=row.option,
            ter=row.ter,
        )
        for row in reference_rows
    ]
    transactions = [
        FundTransactionInput(
            member_id=str(row.member_id),
            security_id=str(row.security_id),
            transaction_date=row.transaction_date,
            kind=row.kind,
            amount=row.amount,
            units=row.units,
        )
        for row in session.scalars(
            select(FundTransaction).where(FundTransaction.user_id == owner_id)
        )
    ]
    previous_month = _previous_month_end(today)
    previous_rows = list(
        session.scalars(
            select(FundLaggingCheck).where(
                FundLaggingCheck.user_id == owner_id,
                FundLaggingCheck.month_end == previous_month,
                FundLaggingCheck.bottom_quartile.is_(True),
            )
        )
    )
    previous_lagging = {(str(row.member_id), str(row.security_id)) for row in previous_rows}
    allocation = plan.get("allocation", {})
    india_target_value = (
        allocation.get("equity_india", {}).get("target") if isinstance(allocation, dict) else None
    )
    try:
        india_target = Decimal(str(india_target_value)) if india_target_value is not None else None
    except InvalidOperation:
        india_target = None
    debt_picks = plan.get("debt_picks") or []
    debt_pick = debt_picks[0] if debt_picks and isinstance(debt_picks[0], dict) else None
    best_debt_option = (
        {
            "instrument": str(debt_pick.get("instrument", "")),
            "after_tax_yield_pct": str(debt_pick.get("after_tax_yield_pct", "")),
        }
        if debt_pick
        else None
    )
    return (
        holdings,
        reference_data,
        dict(histories),
        transactions,
        member_rows,
        india_target,
        best_debt_option,
        previous_lagging,
    )


def evaluate_fund_check(
    session: Session,
    owner_id: UUID,
    today: date,
    *,
    selected_member_id: str | None = None,
    persist_monthly_state: bool = True,
    can_write: bool = True,
) -> FundCheckResult:
    """Build a household-scoped check from saved holdings and current AMFI data."""
    profile = session.scalar(select(Profile).where(Profile.user_id == owner_id))
    (
        holdings,
        references,
        histories,
        transactions,
        members,
        india_target,
        best_debt_option,
        previous_lagging,
    ) = _build_inputs(session, owner_id, today)
    result = check_funds(
        holdings=holdings,
        references=references,
        histories=histories,
        transactions=transactions,
        members=members,
        as_of=today,
        risk_profile=profile.risk_profile if profile is not None else None,
        previous_lagging=previous_lagging,
        selected_member_id=selected_member_id,
        indian_sleeve_target=india_target,
        best_debt_option=best_debt_option,
    )
    if persist_monthly_state and can_write:
        month_end = _month_end(today)
        candidates = {
            (member_id, security_id): bottom
            for member_id, security_id, bottom in result.lagging_candidates
        }
        existing = list(
            session.scalars(
                select(FundLaggingCheck).where(
                    FundLaggingCheck.user_id == owner_id,
                    FundLaggingCheck.month_end == month_end,
                )
            )
        )
        by_key = {(str(row.member_id), str(row.security_id)): row for row in existing}
        for (member_id, security_id), bottom in candidates.items():
            row = by_key.get((member_id, security_id))
            if row is None:
                session.add(
                    FundLaggingCheck(
                        user_id=owner_id,
                        member_id=UUID(member_id),
                        security_id=UUID(security_id),
                        month_end=month_end,
                        bottom_quartile=bottom,
                    )
                )
            else:
                row.bottom_quartile = bottom
    return result


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    router = APIRouter()
    from wealth.api.money_map import get_today

    user_dependency = Depends(authenticated_user)
    today_dependency = Depends(get_today)

    @router.get("/v1/fund-check", response_model=FundCheckResponse)
    def get_fund_check(
        member: str = Query(default="family"),
        user: Any = user_dependency,
        today: date = today_dependency,
    ) -> dict[str, Any]:
        selected_member = None
        if member != "family":
            try:
                selected_member = str(UUID(member))
            except ValueError as error:
                raise HTTPException(
                    status_code=422, detail="Choose a household member or family."
                ) from error
            if (
                user.session.scalar(
                    select(HouseholdMember.id).where(
                        HouseholdMember.id == UUID(selected_member),
                        HouseholdMember.user_id == user.household_owner_id,
                    )
                )
                is None
            ):
                raise HTTPException(status_code=404, detail="The household member was not found.")

        result = evaluate_fund_check(
            user.session,
            user.household_owner_id,
            today,
            selected_member_id=selected_member,
            persist_monthly_state=selected_member is None,
            can_write=user.role != "viewer",
        )
        _, _, _, _, members, _, _, _ = _build_inputs(user.session, user.household_owner_id, today)
        per_member_warning = {}
        if selected_member is None:
            for household_member in members:
                member_result = evaluate_fund_check(
                    user.session,
                    user.household_owner_id,
                    today,
                    selected_member_id=household_member.member_id,
                    persist_monthly_state=False,
                    can_write=False,
                )
                if member_result.early_warning is not None:
                    per_member_warning[household_member.member_id] = member_result.early_warning
        warning = {
            "family": result.early_warning if selected_member is None else None,
            "members": per_member_warning,
        }
        details = dict(result.details)
        details["member_id"] = selected_member
        return {
            "status": result.status,
            "summary_key": result.summary_key,
            "checks": list(result.checks),
            "early_warning": warning if selected_member is None else result.early_warning,
            "details": details,
        }

    @router.put("/v1/fund-check/sips/{security_id}")
    def put_sip_override(
        security_id: UUID,
        payload: FundSipOverride,
        user: Any = user_dependency,
    ) -> dict[str, str]:
        member = user.session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user.household_owner_id,
                HouseholdMember.id == payload.member_id,
            )
        )
        security = user.session.scalar(
            select(Security).where(
                Security.id == security_id,
                Security.security_type == "mf",
                Security.market == "IN",
            )
        )
        if member is None or security is None:
            raise HTTPException(
                status_code=404, detail="The fund or household member was not found."
            )
        investing = dict(member.investing or {})
        overrides = dict(investing.get("fund_sip_overrides", {}))
        overrides[str(security_id)] = format(payload.monthly_amount_inr, "f")
        investing["fund_sip_overrides"] = overrides
        member.investing = investing
        return {
            "security_id": str(security_id),
            "monthly_amount_inr": format(payload.monthly_amount_inr, "f"),
        }

    return router


def add_weekly_fund_action(
    session: Session,
    owner_id: UUID,
    today: date,
    actions: list[dict[str, Any]],
    *,
    can_write: bool = True,
) -> list[dict[str, Any]]:
    """Add at most one top-ranked fund action to the shared weekly action list."""
    result = evaluate_fund_check(
        session,
        owner_id,
        today,
        persist_monthly_state=True,
        can_write=can_write,
    )
    if not result.actions:
        return actions
    selected = result.actions[0]
    matching_check = next(
        (
            check
            for check in result.checks
            if check["reason_code"] == selected["reason_code"]
            and check.get("member_id")
            == selected.get("source_member_id", selected.get("member_id"))
        ),
        None,
    )
    money_at_stake = (
        Decimal(str(matching_check["money_at_stake"])) if matching_check else Decimal(0)
    )
    candidate = {
        **selected,
        "sleeve": "equity_india",
        "instrument_category": "mutual_fund",
        "rank": 0,
        "money_at_stake": format(money_at_stake, "f"),
    }
    candidates = [*actions, candidate]

    def amount_key(action: Mapping[str, Any]) -> tuple[Decimal, int]:
        try:
            stake = Decimal(str(action.get("money_at_stake", action.get("amount", "0"))))
        except InvalidOperation:
            stake = Decimal(0)
        return stake, 1 if action is candidate else 0

    selected_actions = sorted(candidates, key=amount_key, reverse=True)[:3]
    return [{**action, "rank": rank} for rank, action in enumerate(selected_actions, start=1)]
