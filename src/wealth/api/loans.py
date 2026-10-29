"""Household loan positions, event history, and deterministic repayment advice."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import yaml
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from wealth.core.dates import MonthYear
from wealth.db.models import LoanEventRecord, ManualAsset, Profile
from wealth.marketdata.store import latest_fx_rate
from wealth.policy.debt_yields import load_debt_yields
from wealth.policy.foundation import foundation_check
from wealth.policy.loans import (
    LoanCandidate,
    LoanPlanContext,
    load_loan_policy_rules,
    load_tax_relief_rules,
    pay_extra_or_invest,
)
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import LoanAsset, parse_asset
from wealth.schemas.common import CurrencyAmount, Rupees
from wealth.schemas.loans import LoanEventWrite, PrepayPreviewWrite
from wealth.valuation.loans import LoanEvent, loan_position, prepay_options


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoanSummaryItem(ApiModel):
    id: UUID
    name: str
    lender: str | None
    currency: str
    outstanding: CurrencyAmount
    outstanding_inr: Rupees | None
    rate_pct: Decimal
    emi: CurrencyAmount | None
    emi_inr: Rupees | None
    months_left: int | None
    payoff_month: MonthYear | None
    percent_repaid: Decimal | None
    never_pays_off: bool
    fx_missing: bool


class LoanListResponse(ApiModel):
    loans: list[LoanSummaryItem]
    remaining_inr: Rupees
    monthly_emis_inr: Rupees
    debt_load_pct: Decimal | None
    debt_load_level: str
    advice: str
    advice_kind: str
    suggested_amount_inr: Rupees
    reason_codes: list[str]


class LoanEventResponse(ApiModel):
    id: UUID
    kind: str
    on_date: date
    amount: CurrencyAmount | None
    rate_pct: Decimal | None
    emi: CurrencyAmount | None
    note: str | None


class LoanPaymentResponse(ApiModel):
    on_date: date
    amount: CurrencyAmount
    principal: CurrencyAmount
    interest: CurrencyAmount
    outstanding: CurrencyAmount


class LoanBalancePointResponse(ApiModel):
    month: MonthYear
    outstanding: CurrencyAmount


class LoanDetailResponse(LoanSummaryItem):
    rate_type: str | None
    tax_relief: str
    principal_paid: CurrencyAmount
    interest_paid: CurrencyAmount
    interest_to_pay: CurrencyAmount | None
    next_emi_date: date | None
    yearly_balances: list[LoanBalancePointResponse]
    scheduled_payments: list[LoanPaymentResponse]
    events: list[LoanEventResponse]
    advice: str
    advice_kind: str
    suggested_amount_inr: Rupees
    reason_codes: list[str]
    tax_source_urls: list[str]
    tax_relief_needs_review: bool


class PrepayPreviewResponse(ApiModel):
    keep_emi_months_saved: int
    keep_emi_interest_saved: CurrencyAmount
    keep_end_date_new_emi: CurrencyAmount
    keep_end_date_interest_saved: CurrencyAmount


_RESIDENCY_PATH = Path(__file__).resolve().parents[3] / "rules" / "residency.yaml"


def _read_loan(record: ManualAsset) -> LoanAsset:
    details = dict(record.details)
    if "outstanding_as_of" not in details:
        saved_at = record.updated_at.date() if record.updated_at is not None else date.today()
        details["outstanding_as_of"] = str(MonthYear(saved_at.year, saved_at.month))
    try:
        asset = parse_asset(details)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="The saved loan is invalid.") from error
    if not isinstance(asset, LoanAsset):
        raise HTTPException(status_code=404, detail="The loan was not found.")
    return asset


def _event_values(row: LoanEventRecord) -> LoanEvent:
    return LoanEvent(
        kind=row.kind,
        on_date=row.on_date,
        amount=row.amount,
        rate_pct=row.rate_pct,
        emi=row.emi,
        note=row.note,
    )


def _loan_events(session: Any, owner_id: UUID, loan_id: UUID) -> list[LoanEventRecord]:
    return list(
        session.scalars(
            select(LoanEventRecord)
            .where(
                LoanEventRecord.user_id == owner_id,
                LoanEventRecord.loan_id == loan_id,
            )
            .order_by(LoanEventRecord.on_date, LoanEventRecord.created_at, LoanEventRecord.id)
        )
    )


def _fx_rate(session: Any, currency: str, as_of: date) -> Decimal | None:
    if currency == "INR":
        return Decimal(1)
    row = latest_fx_rate(session, f"{currency}/INR", as_of)
    if row is None or (as_of - row[1]).days > 10:
        return None
    return row[0]


def _profile_advice_inputs(
    session: Any, owner_id: UUID, as_of: date
) -> tuple[Decimal, Decimal | None, Decimal | None, str | None, str]:
    profile = session.scalar(select(Profile).where(Profile.user_id == owner_id))
    context = dict(profile.financial_context or {}) if profile is not None else {}
    expenses = Decimal(str(context.get("monthly_expenses", "0")))
    income = (
        Decimal(str(context["annual_income"])) / Decimal(12)
        if context.get("annual_income")
        else None
    )
    tax_slab = profile.tax_slab_pct if profile is not None else None
    tax_regime = profile.tax_regime if profile is not None else None
    country = (profile.residency or {}).get("country", "IN") if profile is not None else "IN"
    return expenses, income, tax_slab, tax_regime, country


def _advice(
    session: Any,
    owner_id: UUID,
    loans: list[tuple[ManualAsset, LoanAsset, Any, list[LoanEventRecord]]],
    as_of: date,
) -> tuple[Any, list[LoanCandidate]]:
    from wealth.marketdata.provider import DbPriceProvider
    from wealth.policy.data import _financial_context, _manual_assets
    from wealth.policy.rules import decimal_rule, resolve_rules

    profile = session.scalar(select(Profile).where(Profile.user_id == owner_id))
    expenses, monthly_income, tax_slab, tax_regime, country = _profile_advice_inputs(
        session, owner_id, as_of
    )
    all_assets, _notes = _manual_assets(session, owner_id, as_of, DbPriceProvider(session))
    snapshot = WealthSnapshot.from_assets(all_assets)
    context = _financial_context(profile)
    foundation = foundation_check(snapshot, context)
    emergency_gap = foundation["emergency_fund"].gap or Decimal(0)
    foundation_rules = resolve_rules()["foundation"]
    threshold = decimal_rule(
        foundation_rules["high_interest_debt_rate_pct"], "foundation.high_interest_debt_rate_pct"
    )
    yields, _yield_date, _verified = load_debt_yields(as_of)
    slab = tax_slab or Decimal(0)
    safe_return = yields.get("bank_fd", Decimal(0)) * (Decimal(100) - slab) / Decimal(100)
    residency = yaml.safe_load(_RESIDENCY_PATH.read_text(encoding="utf-8"))
    dk_rule = residency.get("denmark", {}).get("interest_deduction", {})
    candidates: list[LoanCandidate] = []
    for record, asset, position, _events in loans:
        rate = _fx_rate(session, asset.currency, as_of)
        if rate is None:
            continue
        candidates.append(
            LoanCandidate(
                id=str(record.id),
                loan=asset,
                outstanding_inr=position.outstanding * rate,
                emi_inr=asset.emi * rate if asset.emi is not None else None,
                tax_regime=tax_regime if tax_regime in {"old", "new"} else None,
                tax_slab_pct=tax_slab,
                country="DK" if country == "DK" else "IN",
                member_id=str(record.member_id),
                never_pays_off=position.never_pays_off,
            )
        )
    total_emi = sum((item.emi_inr or Decimal(0) for item in candidates), Decimal(0))
    spare = (
        max(Decimal(0), monthly_income - expenses - total_emi)
        if monthly_income is not None
        else Decimal(0)
    )
    decision = pay_extra_or_invest(
        candidates,
        LoanPlanContext(
            spare_money_inr=spare,
            emergency_fund_gap_inr=emergency_gap,
            monthly_income_inr=monthly_income,
            expected_safe_return_pct=safe_return,
            high_interest_threshold_pct=threshold,
            tax_rules=load_tax_relief_rules(),
            debt_load_rules=load_loan_policy_rules(),
            danish_interest_rule=dk_rule,
        ),
    )
    return decision, candidates


def _tax_sources(asset: LoanAsset, regime: str | None, country: str) -> list[str]:
    if asset.tax_relief == "none":
        return []
    if asset.tax_relief == "dk_interest_deduction":
        residency = yaml.safe_load(_RESIDENCY_PATH.read_text(encoding="utf-8"))
        rule = residency.get("denmark", {}).get("interest_deduction", {})
        source_url = rule.get("source_url") if rule.get("verified") is False else None
        return [source_url] if country == "DK" and isinstance(source_url, str) else []
    if country != "IN":
        return []
    rules = load_tax_relief_rules().get("rules", {}).get("india", {})
    by_regime = rules.get(asset.tax_relief, {}) if isinstance(rules, dict) else {}
    regimes = (regime,) if regime in {"old", "new"} else ("old", "new")
    sources: list[str] = []
    for selected_regime in regimes:
        rule = by_regime.get(selected_regime, {}) if isinstance(by_regime, dict) else {}
        source_url = rule.get("source_url") if rule.get("verified") is False else None
        if isinstance(source_url, str) and source_url not in sources:
            sources.append(source_url)
    return sources


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    """Build loan routes with the app's authenticated household session."""
    from wealth.api.money_map import get_today

    router = APIRouter()
    AuthUser = Annotated[Any, Depends(authenticated_user)]
    Today = Annotated[date, Depends(get_today)]

    def get_loan(session: Any, owner_id: UUID, loan_id: UUID) -> tuple[ManualAsset, LoanAsset]:
        record = session.scalar(
            select(ManualAsset).where(
                ManualAsset.user_id == owner_id,
                ManualAsset.id == loan_id,
                ManualAsset.asset_type == "loan",
                ManualAsset.status == "active",
            )
        )
        if record is None:
            raise HTTPException(status_code=404, detail="The loan was not found.")
        return record, _read_loan(record)

    def summary_rows(user: Any, as_of: date) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
        records = list(
            user.session.scalars(
                select(ManualAsset)
                .where(
                    ManualAsset.user_id == user.household_owner_id,
                    ManualAsset.asset_type == "loan",
                    ManualAsset.status == "active",
                )
                .order_by(ManualAsset.created_at, ManualAsset.id)
            )
        )
        result: list[dict[str, Any]] = []
        calculations: list[tuple[Any, ...]] = []
        for record in records:
            asset = _read_loan(record)
            rows = _loan_events(user.session, user.household_owner_id, record.id)
            position = loan_position(asset, [_event_values(row) for row in rows], as_of)
            rate = _fx_rate(user.session, asset.currency, as_of)
            outstanding_inr = position.outstanding * rate if rate is not None else None
            emi_inr = asset.emi * rate if rate is not None and asset.emi is not None else None
            result.append(
                {
                    "id": record.id,
                    "name": asset.label
                    or asset.lender
                    or f"{asset.loan_kind.replace('_', ' ').title()} loan",
                    "lender": asset.lender,
                    "currency": asset.currency,
                    "outstanding": position.outstanding,
                    "outstanding_inr": outstanding_inr,
                    "rate_pct": asset.rate_pct,
                    "emi": asset.emi,
                    "emi_inr": emi_inr,
                    "months_left": position.months_left,
                    "payoff_month": position.payoff_month,
                    "percent_repaid": position.percent_repaid,
                    "never_pays_off": position.never_pays_off,
                    "fx_missing": rate is None,
                }
            )
            calculations.append((record, asset, position, rows))
        return result, calculations

    @router.get("/v1/loans", response_model=LoanListResponse)
    def list_loans(user: AuthUser, today: Today) -> dict[str, Any]:
        summaries, calculations = summary_rows(user, today)
        decision, _candidates = _advice(user.session, user.household_owner_id, calculations, today)
        return {
            "loans": summaries,
            "remaining_inr": sum(
                (row["outstanding_inr"] or Decimal(0) for row in summaries), Decimal(0)
            ),
            "monthly_emis_inr": sum(
                (row["emi_inr"] or Decimal(0) for row in summaries), Decimal(0)
            ),
            "debt_load_pct": decision.debt_load_pct,
            "debt_load_level": decision.debt_load_level,
            "advice": decision.answer,
            "advice_kind": decision.action,
            "suggested_amount_inr": decision.suggested_amount_inr,
            "reason_codes": list(decision.reason_codes),
        }

    @router.get("/v1/loans/{loan_id}", response_model=LoanDetailResponse)
    def get_loan_detail(loan_id: UUID, user: AuthUser, today: Today) -> dict[str, Any]:
        _record, asset = get_loan(user.session, user.household_owner_id, loan_id)
        rows = _loan_events(user.session, user.household_owner_id, loan_id)
        events = [_event_values(row) for row in rows]
        position = loan_position(asset, events, today)
        summaries, calculations = summary_rows(user, today)
        decision, candidates = _advice(user.session, user.household_owner_id, calculations, today)
        own_decision = decision
        if decision.loan_id != str(loan_id):
            from wealth.policy.loans import LoanPlanContext, pay_extra_or_invest

            selected = [item for item in candidates if item.id == str(loan_id)]
            own_decision = pay_extra_or_invest(
                selected,
                LoanPlanContext(
                    spare_money_inr=Decimal(0),
                    emergency_fund_gap_inr=Decimal(0),
                    monthly_income_inr=None,
                    expected_safe_return_pct=decision.safe_return_pct,
                    high_interest_threshold_pct=Decimal(100),
                    tax_rules=load_tax_relief_rules(),
                    debt_load_rules=load_loan_policy_rules(),
                    danish_interest_rule=None,
                ),
            )
        summary = next((item for item in summaries if item["id"] == loan_id), None)
        if summary is None:
            raise HTTPException(status_code=404, detail="The loan was not found.")
        _expenses, _income, _slab, tax_regime, country = _profile_advice_inputs(
            user.session, user.household_owner_id, today
        )
        tax_sources = _tax_sources(asset, tax_regime, country)
        return {
            **summary,
            "rate_type": asset.rate_type,
            "tax_relief": asset.tax_relief,
            "principal_paid": position.principal_paid,
            "interest_paid": position.interest_paid,
            "interest_to_pay": position.interest_to_pay,
            "next_emi_date": position.next_emi_date,
            "yearly_balances": [
                {"month": point.month, "outstanding": point.outstanding}
                for point in position.yearly_balances
            ],
            "scheduled_payments": [
                {
                    "on_date": payment.on_date,
                    "amount": payment.amount,
                    "principal": payment.principal,
                    "interest": payment.interest,
                    "outstanding": payment.outstanding,
                }
                for payment in position.scheduled_payments
            ],
            "events": [
                {
                    "id": row.id,
                    "kind": row.kind,
                    "on_date": row.on_date,
                    "amount": row.amount,
                    "rate_pct": row.rate_pct,
                    "emi": row.emi,
                    "note": row.note,
                }
                for row in rows
            ],
            "advice": own_decision.answer,
            "advice_kind": own_decision.action,
            "suggested_amount_inr": own_decision.suggested_amount_inr,
            "reason_codes": list(own_decision.reason_codes),
            "tax_source_urls": tax_sources,
            "tax_relief_needs_review": bool(tax_sources),
        }

    @router.post("/v1/loans/{loan_id}/events", response_model=LoanEventResponse, status_code=201)
    def add_event(loan_id: UUID, payload: LoanEventWrite, user: AuthUser) -> Any:
        record, _asset = get_loan(user.session, user.household_owner_id, loan_id)
        row = LoanEventRecord(
            user_id=user.household_owner_id,
            member_id=record.member_id,
            loan_id=record.id,
            **payload.model_dump(),
        )
        user.session.add(row)
        user.session.flush()
        return row

    @router.delete(
        "/v1/loans/{loan_id}/events/{event_id}",
        status_code=204,
        response_class=Response,
        response_model=None,
    )
    def delete_event(loan_id: UUID, event_id: UUID, user: AuthUser) -> Response:
        get_loan(user.session, user.household_owner_id, loan_id)
        row = user.session.scalar(
            select(LoanEventRecord).where(
                LoanEventRecord.user_id == user.household_owner_id,
                LoanEventRecord.loan_id == loan_id,
                LoanEventRecord.id == event_id,
            )
        )
        if row is None:
            raise HTTPException(status_code=404, detail="The loan event was not found.")
        user.session.delete(row)
        return Response(status_code=204)

    @router.post("/v1/loans/{loan_id}/prepay-preview", response_model=PrepayPreviewResponse)
    def preview_prepayment(
        loan_id: UUID, payload: PrepayPreviewWrite, user: AuthUser, today: Today
    ) -> dict[str, Any]:
        _record, asset = get_loan(user.session, user.household_owner_id, loan_id)
        events = [
            _event_values(row)
            for row in _loan_events(user.session, user.household_owner_id, loan_id)
        ]
        try:
            result = prepay_options(asset, events, payload.amount, today)
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail="This loan has no reliable payoff estimate yet."
            ) from error
        return result.__dict__

    return router
