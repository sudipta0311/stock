"""Tenant-scoped Money Map endpoints and deterministic snapshot aggregation."""

import re
from collections.abc import Callable
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Self
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from sqlalchemy import func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from wealth.core.dates import MonthYear, add_months
from wealth.importer.quick_add import QuickAddNotUnderstood, parse_quick_add
from wealth.policy.snapshot import WealthSnapshot
from wealth.rates.reference import RateUnavailable
from wealth.schemas.assets import AnyAsset, GoldAsset, LoanAsset, SavingsAsset, parse_asset
from wealth.schemas.common import LiquidityClass, Rupees, Sleeve, ValuationMethod
from wealth.schemas.profile import Goal as GoalInput
from wealth.schemas.profile import (
    InvestingPrefs,
    MemberTax,
    Residency,
    ResidencyInput,
)
from wealth.valuation import PriceUnavailable, Valuation, value_asset
from wealth.valuation.prices import PriceProvider

_CHECKLIST_CATEGORIES = (
    "epf",
    "ppf",
    "fd",
    "post_office",
    "nps",
    "insurance",
    "gold",
    "bonds",
    "savings",
    "loans",
    "property",
    "cas_upload",
    "us_investments",
)
_CATEGORY_ASSET_TYPES: dict[str, frozenset[str]] = {
    "epf": frozenset({"epf"}),
    "ppf": frozenset({"ppf"}),
    "fd": frozenset({"fd"}),
    "post_office": frozenset({"po_scheme"}),
    "nps": frozenset({"nps"}),
    "insurance": frozenset({"term_insurance", "health_insurance", "traditional_policy", "ulip"}),
    "gold": frozenset({"gold"}),
    "bonds": frozenset({"bond"}),
    "savings": frozenset({"savings"}),
    "loans": frozenset({"loan"}),
    "property": frozenset({"property"}),
    "cas_upload": frozenset(),
    "us_investments": frozenset(),
}
_ZERO_VALUATION = Valuation(
    value_today=Decimal("0.00"),
    maturity_value=None,
    maturity_date=None,
    accrued_interest=Decimal("0.00"),
    principal_so_far=Decimal("0.00"),
    method=ValuationMethod.USER,
    notes=(),
)


class MoneyMapModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, from_attributes=True)


class MemberWrite(MoneyMapModel):
    display_name: str = Field(min_length=1, max_length=40)
    relationship: Literal["self", "spouse", "child", "parent", "other"]
    is_self: bool = False
    residency: ResidencyInput = Field(default_factory=ResidencyInput)
    tax: MemberTax = Field(default_factory=MemberTax)
    investing: InvestingPrefs = Field(default_factory=InvestingPrefs)

    @model_validator(mode="after")
    def member_is_not_self(self) -> Self:
        if self.is_self or self.relationship == "self":
            raise ValueError("The self member is created automatically.")
        return self


class MemberPatch(MoneyMapModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=40)
    relationship: Literal["self", "spouse", "child", "parent", "other"] | None = None
    is_self: bool | None = None
    residency: ResidencyInput | None = None
    tax: MemberTax | None = None
    investing: InvestingPrefs | None = None


class MemberResponse(MoneyMapModel):
    id: UUID
    display_name: str
    relationship: str
    is_self: bool
    residency: Residency
    tax: MemberTax
    investing: InvestingPrefs
    created_at: datetime


class AccountWrite(MoneyMapModel):
    institution: str = Field(min_length=1, max_length=120)
    account_type: str = Field(min_length=1, max_length=32)
    masked_number: str | None = Field(default=None, max_length=32)
    tax_wrapper: Literal["ask", "ordinary", "pension"] | None = None
    member_id: UUID | None = None
    joint: bool = False

    @field_validator("masked_number")
    @classmethod
    def validate_masked_number(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"X*[0-9A-Za-z]{0,4}", value):
            raise ValueError("masked_number must contain at most four visible trailing characters")
        return value


class AccountPatch(MoneyMapModel):
    institution: str | None = Field(default=None, min_length=1, max_length=120)
    account_type: str | None = Field(default=None, min_length=1, max_length=32)
    masked_number: str | None = Field(default=None, max_length=32)
    tax_wrapper: Literal["ask", "ordinary", "pension"] | None = None
    member_id: UUID | None = None
    joint: bool | None = None

    @field_validator("masked_number")
    @classmethod
    def validate_masked_number(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"X*[0-9A-Za-z]{0,4}", value):
            raise ValueError("masked_number must contain at most four visible trailing characters")
        return value


class AccountResponse(MoneyMapModel):
    id: UUID
    institution: str
    account_type: str
    masked_number: str | None
    member_id: UUID | None
    joint: bool
    tax_wrapper: Literal["ask", "ordinary", "pension"] | None = None


class GoalResponse(MoneyMapModel):
    id: UUID
    name: str
    goal_type: str
    target_amount: Rupees
    target_year: int
    currency: Literal["INR", "DKK", "EUR", "USD"]
    priority: int
    member_id: UUID | None


class AssetResponse(MoneyMapModel):
    id: UUID
    asset_type: str
    member_id: UUID
    joint: bool
    account_id: UUID | None
    details: dict[str, Any]
    valuation: dict[str, Any] | None
    valuation_note: str | None
    valuation_status: Literal[
        "valued", "price_needed", "rate_needed", "fx_needed", "not_applicable"
    ]
    value_cached: Rupees | None
    value_as_of: date | None
    status: Literal["active", "closed"]
    created_at: datetime
    updated_at: datetime


class AssetHistoryResponse(MoneyMapModel):
    id: UUID
    changed_fields: dict[str, dict[str, str]]
    changed_by: str
    changed_at: datetime


class AssetDetailResponse(AssetResponse):
    history: list[AssetHistoryResponse]


class QuickAddResponse(MoneyMapModel):
    asset_type: str
    draft: dict[str, Any]
    missing: list[str]
    confidence: Literal["high", "medium", "low"]


class ChecklistWrite(MoneyMapModel):
    categories: list[str]

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, values: list[str]) -> list[str]:
        invalid = sorted(set(values) - set(_CHECKLIST_CATEGORIES))
        if invalid:
            raise ValueError("categories contains an unsupported Money Map tile")
        return list(dict.fromkeys(values))


class SleeveSummary(MoneyMapModel):
    sleeve: str
    amount: Rupees
    pct: str


class LiquiditySummary(MoneyMapModel):
    instant: Rupees
    within_1y: Rupees
    locked: Rupees


class DebtInsight(MoneyMapModel):
    debt_pct_including_locked: str
    locked_debt_amount: Rupees
    message_key: Literal["debt_share"]


class MaturityEvent(MoneyMapModel):
    date: date
    asset_id: UUID
    kind: str
    amount: Rupees | None
    label: str


class CompletenessResponse(MoneyMapModel):
    ticked: list[str]
    done: list[str]
    pct: str


class UnvaluedAssetResponse(MoneyMapModel):
    id: UUID
    asset_type: str
    label: str
    reason: Literal["price_needed", "rate_needed", "fx_needed"]


class HoldingValuationResponse(MoneyMapModel):
    security_id: UUID
    symbol: str
    name: str
    security_type: str
    quantity: str
    value: Rupees | None
    value_usd: str | None
    currency: str
    price_as_of: date | None
    fx_as_of: date | None
    fx_missing: bool
    price_source: str | None
    status: Literal["priced", "statement_value_used", "price_unavailable", "fx_unavailable"]


class NetworthResponse(MoneyMapModel):
    net_worth: str
    assets_total: Rupees
    liabilities_total: Rupees
    investable_total: Rupees
    by_sleeve: list[SleeveSummary]
    liquidity_ladder: LiquiditySummary
    debt_insight: DebtInsight
    maturity_calendar: list[MaturityEvent]
    unvalued_assets: list[UnvaluedAssetResponse]
    completeness: CompletenessResponse
    as_of: date
    market_data_as_of: date | None
    holdings: list[HoldingValuationResponse]


def get_today() -> date:
    """Return the current valuation date; tests can override this dependency."""
    return date.today()


def _problem(
    request: Request,
    status: int,
    title: str,
    detail: str,
    extensions: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
    }
    if extensions:
        body.update(extensions)
    return JSONResponse(status_code=status, media_type="application/problem+json", content=body)


def _uuid_or_422(value: str | UUID | None, field: str) -> UUID | None:
    if value is None:
        return None
    try:
        return value if isinstance(value, UUID) else UUID(value)
    except (ValueError, TypeError, AttributeError) as error:
        raise HTTPException(status_code=422, detail=f"{field} must be a valid UUID.") from error


def _require_member(session: Session, user_id: UUID, member_id: UUID | None) -> None:
    if member_id is None:
        return
    from wealth.db.models import HouseholdMember

    member = session.scalar(
        select(HouseholdMember.id).where(
            HouseholdMember.user_id == user_id, HouseholdMember.id == member_id
        )
    )
    if member is None:
        raise HTTPException(status_code=404, detail="The member was not found.")


def _money(value: Decimal | None) -> str | None:
    return None if value is None else format(value, ".2f")


_HOLDING_STATUS_ORDER = ("priced", "statement_value_used", "fx_unavailable", "price_unavailable")


def _merge_lots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per security: statements list each purchase as its own lot."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = merged.get(row["security_id"])
        if current is None:
            merged[row["security_id"]] = dict(row)
            continue
        current["quantity"] = format(Decimal(current["quantity"]) + Decimal(row["quantity"]), ".6f")
        # Sum what is known, as net worth does; the worst status still flags a missing price.
        for key in ("value", "value_usd"):
            known = [Decimal(item) for item in (current[key], row[key]) if item is not None]
            current[key] = format(sum(known, Decimal(0)), ".2f") if known else None
        current["fx_missing"] = current["fx_missing"] or row["fx_missing"]
        current["status"] = max(current["status"], row["status"], key=_HOLDING_STATUS_ORDER.index)
    return list(merged.values())


def _valuation_json(valuation: Valuation) -> dict[str, Any]:
    return {
        "value_today": _money(valuation.value_today),
        "maturity_value": _money(valuation.maturity_value),
        "maturity_date": valuation.maturity_date.isoformat() if valuation.maturity_date else None,
        "accrued_interest": _money(valuation.accrued_interest),
        "principal_so_far": _money(valuation.principal_so_far),
        "method": valuation.method.value,
        "notes": list(valuation.notes),
    }


def _valuation_from_json(value: dict[str, Any]) -> Valuation:
    return Valuation(
        value_today=Decimal(value["value_today"]),
        maturity_value=(Decimal(value["maturity_value"]) if value.get("maturity_value") else None),
        maturity_date=(
            date.fromisoformat(value["maturity_date"]) if value.get("maturity_date") else None
        ),
        accrued_interest=Decimal(value["accrued_interest"]),
        principal_so_far=Decimal(value["principal_so_far"]),
        method=ValuationMethod(value["method"]),
        notes=tuple(value.get("notes", ())),
    )


def _valuation_note(record: Any) -> str | None:
    status = _valuation_status(record)
    return status if status in {"fx_needed", "price_needed", "rate_needed"} else None


def _valuation_status(record: Any) -> str:
    if record.asset_type in {"term_insurance", "health_insurance", "loan"}:
        return "not_applicable"
    if record.valuation is not None or record.value_cached is not None:
        return "valued"
    if record.asset_type == "po_scheme":
        return "rate_needed"
    if record.asset_type == "savings" and record.details.get("currency", "INR") != "INR":
        return "fx_needed"
    return "price_needed"


def _asset_response(record: Any) -> dict[str, Any]:
    details = dict(record.details)
    if record.asset_type == "savings":
        details.setdefault("currency", "INR")
    return {
        "id": record.id,
        "asset_type": record.asset_type,
        "member_id": record.member_id,
        "joint": record.joint,
        "account_id": record.account_id,
        "details": details,
        "valuation": record.valuation,
        "valuation_note": _valuation_note(record),
        "valuation_status": _valuation_status(record),
        "value_cached": record.value_cached,
        "value_as_of": record.value_as_of,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _validate_asset(data: dict[str, Any]) -> AnyAsset:
    try:
        asset = parse_asset(data)
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail="The asset did not match the selected asset type.",
        ) from error
    return asset


def _evaluate_asset(
    asset: AnyAsset,
    as_of: date,
    *,
    session: Session,
    prices: PriceProvider | None = None,
) -> tuple[Valuation | None, str | None]:
    if isinstance(asset, LoanAsset):
        return None, None
    if isinstance(asset, SavingsAsset) and asset.currency != "INR":
        from wealth.marketdata.store import fresh_fx_rate_to_inr

        rate = fresh_fx_rate_to_inr(session, asset.currency, as_of)
        if rate is None:
            return None, "fx_needed"
        asset = asset.model_copy(update={"balance": asset.balance * rate, "currency": "INR"})
    try:
        return value_asset(asset, as_of, assumed_start=as_of, prices=prices), None
    except PriceUnavailable:
        return None, "price_needed" if isinstance(asset, GoldAsset) else "price_needed"
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail="The asset details are not valid for deterministic valuation.",
        ) from error


def _rate_needed_problem(request: Request) -> JSONResponse:
    return _problem(
        request,
        422,
        "Unprocessable Content",
        "A certificate rate is needed to value this asset.",
        {"code": "rate_needed", "field": "rate_pct"},
    )


def _validated_detail_member(asset: AnyAsset) -> UUID | None:
    member_id = _uuid_or_422(asset.member_id, "member_id")
    return member_id


def _history_value(value: Any) -> str:
    return "None" if value is None else str(value)


def _changed_fields(old: dict[str, Any], new: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for field in sorted(old.keys() | new.keys()):
        if old.get(field) != new.get(field):
            result[field] = {
                "old": _history_value(old.get(field)),
                "new": _history_value(new.get(field)),
            }
    return result


def _percentage(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        return Decimal("0.00")
    return (numerator * Decimal(100) / denominator).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _percentage_text(numerator: Decimal, denominator: Decimal) -> str:
    return format(_percentage(numerator, denominator), ".2f")


def _in_window(event_date: date, as_of: date) -> bool:
    return as_of <= event_date <= add_months(as_of, 12)


def _calendar_events(
    records: list[Any], valuations: dict[UUID, Valuation], as_of: date
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def add_event(
        event_date: date | None,
        record: Any,
        kind: str,
        label: str,
        amount: Decimal | None = None,
    ) -> None:
        if event_date is not None and _in_window(event_date, as_of):
            events.append(
                {
                    "date": event_date.isoformat(),
                    "asset_id": str(record.id),
                    "kind": kind,
                    "amount": _money(amount),
                    "label": label,
                }
            )

    for record in records:
        details = record.details
        asset_type = record.asset_type
        valuation = valuations.get(record.id)
        if asset_type in {"fd", "rd", "po_scheme", "bond"}:
            maturity_date = valuation.maturity_date if valuation else None
            maturity_amount = valuation.maturity_value if valuation else None
            if maturity_date is None:
                month = details.get("maturity")
                if month:
                    parsed = MonthYear(int(month[:4]), int(month[5:7]))
                    maturity_date = parsed.first_day()
            if maturity_date is None and asset_type == "po_scheme":
                purchase = details.get("purchase")
                if purchase:
                    parsed = MonthYear(int(purchase[:4]), int(purchase[5:7]))
                    scheme = details.get("scheme")
                    months = {
                        "nsc": 60,
                        "scss": 60,
                        "mis": 60,
                        "kvp": None,
                        "td": int(details.get("td_years", 0)) * 12,
                    }.get(scheme)
                    if months:
                        maturity_date = add_months(parsed.first_day(), months)
            add_event(
                maturity_date,
                record,
                "maturity",
                f"{asset_type.replace('_', ' ')} maturity",
                maturity_amount,
            )
        elif asset_type == "gold" and details.get("kind") == "sgb" and details.get("issue"):
            issue = details["issue"]
            add_event(
                add_months(MonthYear(int(issue[:4]), int(issue[5:7])).first_day(), 96),
                record,
                "maturity",
                "SGB maturity",
                None,
            )
        elif asset_type == "traditional_policy":
            start_year = int(details["start_year"])
            term_years = int(details["term_years"])
            maturity_value = valuation.maturity_value if valuation else None
            if maturity_value is None and details.get("maturity_illustration") is not None:
                maturity_value = Decimal(details["maturity_illustration"])
            add_event(
                date(start_year + term_years, 4, 1),
                record,
                "maturity",
                "Policy maturity",
                maturity_value,
            )
        elif asset_type == "health_insurance":
            renewal_month = int(details["renewal_month"])
            renewal_year = as_of.year if renewal_month > as_of.month else as_of.year + 1
            renewal = date(renewal_year, renewal_month, 1)
            add_event(renewal, record, "health_renewal", "Health insurance renewal")
        elif asset_type == "ppf":
            april_year = as_of.year if (as_of.month, as_of.day) <= (4, 5) else as_of.year + 1
            add_event(date(april_year, 4, 5), record, "ppf_deposit", "PPF deposit before the 5th")

    events.sort(key=lambda event: (event["date"], event["asset_id"], event["kind"]))
    return events


def _completeness(
    ticked: list[str],
    records: list[Any],
    *,
    has_confirmed_cas_upload: bool = False,
    has_us_market_holdings: bool = False,
) -> dict[str, Any]:
    present = {record.asset_type for record in records if record.status == "active"}
    completed = {category for category in ticked if _CATEGORY_ASSET_TYPES[category] & present}
    if has_confirmed_cas_upload and "cas_upload" in ticked:
        completed.add("cas_upload")
    if has_us_market_holdings and "us_investments" in ticked:
        completed.add("us_investments")
    done = [category for category in ticked if category in completed]
    pct = _percentage_text(Decimal(len(done)), Decimal(len(ticked))) if ticked else "0.00"
    return {"ticked": ticked, "done": done, "pct": pct}


def refresh_networth_snapshot(
    session: Session,
    user_id: UUID,
    today: date,
    member_id: UUID | None = None,
) -> dict[str, Any]:
    """Calculate and persist today's net worth, including imported market holdings."""
    from wealth.core.money import round_paise
    from wealth.db.models import (
        FxRate,
        HoldingLot,
        ImportJob,
        LoanEventRecord,
        ManualAsset,
        NetworthSnapshot,
        Profile,
        Security,
    )
    from wealth.marketdata.provider import DbPriceProvider
    from wealth.marketdata.store import fresh_fx_rate_to_inr
    from wealth.valuation.prices import PriceQuote

    statement = select(ManualAsset).where(
        ManualAsset.user_id == user_id, ManualAsset.status == "active"
    )
    if member_id is not None:
        statement = statement.where(ManualAsset.member_id == member_id)
    records = list(session.scalars(statement.order_by(ManualAsset.created_at, ManualAsset.id)))
    valued_assets: list[tuple[AnyAsset, Valuation]] = []
    valuation_by_id: dict[UUID, Valuation] = {}
    unvalued_assets: list[dict[str, Any]] = []
    loan_positions: dict[UUID, tuple[Any, Decimal]] = {}
    prices = DbPriceProvider(session)

    for record in records:
        asset = _validate_asset(dict(record.details))
        stored = record.valuation
        needs_refresh = record.value_as_of is None or record.value_as_of < today
        if isinstance(asset, LoanAsset):
            from wealth.valuation.loans import LoanEvent, loan_position

            if "outstanding_as_of" not in record.details:
                saved_at = record.updated_at.date() if record.updated_at is not None else today
                asset = asset.model_copy(
                    update={"outstanding_as_of": MonthYear(saved_at.year, saved_at.month)}
                )
            event_rows = session.scalars(
                select(LoanEventRecord)
                .where(
                    LoanEventRecord.user_id == user_id,
                    LoanEventRecord.loan_id == record.id,
                )
                .order_by(LoanEventRecord.on_date, LoanEventRecord.created_at, LoanEventRecord.id)
            )
            position = loan_position(
                asset,
                [
                    LoanEvent(
                        kind=row.kind,
                        on_date=row.on_date,
                        amount=row.amount,
                        rate_pct=row.rate_pct,
                        emi=row.emi,
                        note=row.note,
                    )
                    for row in event_rows
                ],
                today,
            )
            fx_rate = fresh_fx_rate_to_inr(session, asset.currency, today)
            if fx_rate is None:
                unvalued_assets.append(
                    {
                        "id": str(record.id),
                        "asset_type": record.asset_type,
                        "label": record.details.get("label") or record.asset_type,
                        "reason": "fx_needed",
                    }
                )
                continue
            loan_positions[record.id] = (position, fx_rate)
            asset = asset.model_copy(
                update={
                    "outstanding": position.outstanding * fx_rate,
                    "emi": asset.emi * fx_rate if asset.emi is not None else None,
                    "currency": "INR",
                }
            )
            valuation = _ZERO_VALUATION
        elif isinstance(asset, SavingsAsset) and asset.currency != "INR":
            fx_rate = fresh_fx_rate_to_inr(session, asset.currency, today)
            if fx_rate is None:
                record.valuation = None
                record.value_cached = None
                record.value_as_of = None
                unvalued_assets.append(
                    {
                        "id": str(record.id),
                        "asset_type": record.asset_type,
                        "label": record.details.get("label") or record.asset_type,
                        "reason": "fx_needed",
                    }
                )
                continue
            asset = asset.model_copy(update={"balance": asset.balance * fx_rate, "currency": "INR"})
            try:
                valuation = value_asset(asset, today, assumed_start=today, prices=prices)
            except PriceUnavailable:
                unvalued_assets.append(
                    {
                        "id": str(record.id),
                        "asset_type": record.asset_type,
                        "label": record.details.get("label") or record.asset_type,
                        "reason": "price_needed",
                    }
                )
                continue
            record.valuation = _valuation_json(valuation)
            record.value_cached = valuation.value_today
            record.value_as_of = today
        elif stored is not None and not needs_refresh:
            valuation = _valuation_from_json(stored)
        else:
            try:
                valuation = value_asset(asset, today, assumed_start=today, prices=prices)
            except PriceUnavailable:
                unvalued_assets.append(
                    {
                        "id": str(record.id),
                        "asset_type": record.asset_type,
                        "label": record.details.get("label") or record.asset_type,
                        "reason": "price_needed",
                    }
                )
                continue
            except RateUnavailable:
                unvalued_assets.append(
                    {
                        "id": str(record.id),
                        "asset_type": record.asset_type,
                        "label": record.details.get("label") or record.asset_type,
                        "reason": "rate_needed",
                    }
                )
                continue
            record.valuation = _valuation_json(valuation)
            record.value_cached = valuation.value_today
            record.value_as_of = today
        valued_assets.append((asset, valuation))
        valuation_by_id[record.id] = valuation

    market_statement = (
        select(HoldingLot, Security)
        .join(Security, Security.id == HoldingLot.security_id)
        .where(HoldingLot.user_id == user_id)
    )
    if member_id is not None:
        market_statement = market_statement.where(HoldingLot.member_id == member_id)
    market_rows = list(session.execute(market_statement))
    market_holdings: list[tuple[Sleeve | str, Decimal, LiquidityClass | str]] = []
    holding_values: list[dict[str, Any]] = []
    for lot, security in market_rows:
        try:
            quote = (
                prices.nav(security.isin or security.amfi_scheme_code or security.symbol, today)
                if security.security_type == "mf"
                else prices.price(security.id, today)
            )
        except PriceUnavailable:
            quote = None
        if isinstance(quote, PriceQuote):
            native_value = lot.quantity * quote.value
            price_as_of = quote.as_of
            price_source = quote.source
            status = "priced"
        else:
            native_value = lot.statement_value
            price_as_of = lot.statement_date if native_value is not None else None
            price_source = "statement" if native_value is not None else None
            status = "statement_value_used" if native_value is not None else "price_unavailable"

        fx_rate: Decimal | None = Decimal(1) if security.currency == "INR" else None
        fx_as_of: date | None = None
        if security.currency != "INR":
            fx_row = session.execute(
                select(FxRate.rate, FxRate.as_of)
                .where(
                    FxRate.pair == f"{security.currency}/INR",
                    FxRate.as_of <= today,
                )
                .order_by(FxRate.as_of.desc())
                .limit(1)
            ).one_or_none()
            if fx_row is not None:
                fx_rate, fx_as_of = fx_row

        fx_missing = native_value is not None and fx_rate is None
        if native_value is None:
            value = None
        elif fx_rate is None:
            value = None
            status = "fx_unavailable"
        else:
            value = round_paise(native_value * fx_rate)
        value_usd = (
            format(round_paise(native_value), "f")
            if security.currency == "USD" and native_value is not None
            else None
        )
        holding_values.append(
            {
                "security_id": str(security.id),
                "symbol": security.nse_symbol or security.symbol or security.isin or "",
                "name": security.name,
                "security_type": security.security_type,
                "quantity": format(lot.quantity, ".6f"),
                "value": _money(value),
                "value_usd": value_usd,
                "currency": security.currency,
                "price_as_of": price_as_of.isoformat() if price_as_of else None,
                "fx_as_of": fx_as_of.isoformat() if fx_as_of else None,
                "fx_missing": fx_missing,
                "price_source": price_source,
                "status": status,
            }
        )
        if value is not None and security.sleeve is not None:
            liquidity = (
                LiquidityClass.WITHIN_1Y
                if security.security_type in {"sgb", "bond"}
                else LiquidityClass.INSTANT
            )
            market_holdings.append((security.sleeve, value, liquidity))

    snapshot = WealthSnapshot.from_assets(valued_assets, market_holdings)
    by_sleeve = [
        {
            "sleeve": sleeve.value,
            "amount": _money(snapshot.totals_by_sleeve[sleeve]),
            "pct": _percentage_text(snapshot.totals_by_sleeve[sleeve], snapshot.all_assets_total),
        }
        for sleeve in Sleeve
    ]
    ladder = {
        liquidity.value: _money(snapshot.totals_by_liquidity[liquidity])
        for liquidity in LiquidityClass
    }
    debt_total = snapshot.totals_by_sleeve[Sleeve.DEBT]
    locked_debt = snapshot.totals_by_sleeve_and_liquidity[(Sleeve.DEBT, LiquidityClass.LOCKED)]
    profile = session.scalar(select(Profile).where(Profile.user_id == user_id))
    ticked = [
        item for item in (profile.checklist if profile else []) if item in _CHECKLIST_CATEGORIES
    ]
    has_confirmed_cas_upload = (
        session.scalar(
            select(ImportJob.id)
            .where(
                ImportJob.user_id == user_id,
                ImportJob.status == "confirmed",
                ImportJob.source.in_(("nsdl", "cdsl", "cams_kfin")),
            )
            .limit(1)
        )
        is not None
    )
    completeness = _completeness(
        ticked,
        records,
        has_confirmed_cas_upload=has_confirmed_cas_upload,
        has_us_market_holdings=any(
            security.market == "US" or security.currency == "USD" for _lot, security in market_rows
        ),
    )
    events = _calendar_events(records, valuation_by_id, today)
    for record in records:
        calculated = loan_positions.get(record.id)
        if calculated is None:
            continue
        position, fx_rate = calculated
        loan = _validate_asset(dict(record.details))
        next_payment = next(
            (
                payment.amount
                for payment in position.scheduled_payments
                if payment.on_date == position.next_emi_date
            ),
            loan.emi if isinstance(loan, LoanAsset) else None,
        )
        for event_date, kind, label, amount in (
            (
                position.next_emi_date,
                "loan_payment",
                "Loan monthly payment",
                next_payment,
            ),
            (
                position.payoff_month.first_day() if position.payoff_month else None,
                "loan_payoff",
                "Loan payoff",
                position.outstanding,
            ),
        ):
            if event_date is not None and _in_window(event_date, today):
                events.append(
                    {
                        "date": event_date.isoformat(),
                        "asset_id": str(record.id),
                        "kind": kind,
                        "amount": _money(amount * fx_rate) if amount is not None else None,
                        "label": label,
                    }
                )
        if isinstance(loan, LoanAsset) and loan.rate_type == "floating":
            review_month = loan.start or loan.outstanding_as_of
            review_date = date(today.year, review_month.month, 1)
            if review_date <= today:
                review_date = date(today.year + 1, review_month.month, 1)
            if _in_window(review_date, today):
                events.append(
                    {
                        "date": review_date.isoformat(),
                        "asset_id": str(record.id),
                        "kind": "loan_rate_review",
                        "amount": None,
                        "label": "Check your loan rate",
                    }
                )
    events.sort(key=lambda event: (event["date"], event["asset_id"], event["kind"]))

    totals = {
        "net_worth": _money(snapshot.net_worth),
        "assets_total": _money(snapshot.all_assets_total),
        "liabilities_total": _money(snapshot.liabilities_total),
        "investable_total": _money(snapshot.investable_total),
        "by_sleeve": by_sleeve,
        "liquidity_ladder": ladder,
        "debt_insight": {
            "debt_pct_including_locked": _percentage_text(debt_total, snapshot.all_assets_total),
            "locked_debt_amount": _money(locked_debt),
            "message_key": "debt_share",
        },
        "maturity_calendar": events,
        "completeness": completeness,
        "unvalued_assets": unvalued_assets,
        "as_of": today.isoformat(),
        "market_data_as_of": (
            prices.market_data_as_of(today).isoformat()
            if prices.market_data_as_of(today) is not None
            else None
        ),
        "holdings": _merge_lots(holding_values),
    }
    insert = pg_insert(NetworthSnapshot).values(
        id=uuid4(), user_id=user_id, on_date=today, totals=totals
    )
    insert = insert.on_conflict_do_update(
        index_elements=[NetworthSnapshot.user_id, NetworthSnapshot.on_date],
        set_={"totals": insert.excluded.totals},
    )
    session.execute(insert)
    return totals


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    """Build Money Map endpoints around the app's authenticated user dependency."""
    from wealth.db.models import (
        Account,
        AssetHistory,
        Goal,
        HouseholdMember,
        ManualAsset,
        Profile,
    )

    router = APIRouter()
    AuthUser = Annotated[Any, Depends(authenticated_user)]
    Today = Annotated[date, Depends(get_today)]

    @router.get("/v1/members", response_model=list[MemberResponse])
    def list_members(user: AuthUser) -> list[Any]:
        from wealth.db.repo import acting_household_member

        acting_household_member(user.session, user.user_id, user.household_owner_id)
        statement = (
            select(HouseholdMember)
            .where(HouseholdMember.user_id == user.household_owner_id)
            .order_by(HouseholdMember.created_at, HouseholdMember.id)
        )
        return list(user.session.scalars(statement))

    @router.post("/v1/members", response_model=MemberResponse, status_code=201)
    def create_member(payload: MemberWrite, user: AuthUser) -> Any:
        member = HouseholdMember(
            user_id=user.household_owner_id,
            display_name=payload.display_name,
            relationship=payload.relationship,
            is_self=False,
            residency=payload.residency.model_dump(mode="json"),
            tax=payload.tax.model_dump(mode="json", exclude_none=True),
            investing=payload.investing.model_dump(mode="json"),
        )
        user.session.add(member)
        user.session.flush()
        return member

    @router.patch("/v1/members/{member_id}", response_model=MemberResponse)
    def patch_member(member_id: UUID, payload: MemberPatch, user: AuthUser) -> Any:
        from wealth.db.models import Profile

        member = user.session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user.household_owner_id, HouseholdMember.id == member_id
            )
        )
        if member is None:
            raise HTTPException(status_code=404, detail="The member was not found.")
        changes = payload.model_dump(exclude_unset=True, mode="python")
        if "is_self" in changes and changes["is_self"] != member.is_self:
            raise HTTPException(status_code=422, detail="The self member cannot be changed.")
        if member.is_self and changes.get("relationship", "self") != "self":
            raise HTTPException(status_code=422, detail="The self member relationship is fixed.")
        if not member.is_self and changes.get("relationship") == "self":
            raise HTTPException(status_code=422, detail="Only one self member is allowed.")
        changes.pop("is_self", None)
        for key, value in changes.items():
            if key == "residency" and value is not None:
                value = payload.residency.model_dump(mode="json")
            elif key == "tax" and value is not None:
                merged_tax = dict(member.tax or {})
                merged_tax.update(payload.tax.model_dump(mode="json", exclude_unset=True))
                value = MemberTax.model_validate(merged_tax).model_dump(
                    mode="json", exclude_none=True
                )
            elif key == "investing" and value is not None:
                merged_investing = dict(member.investing or {})
                merged_investing.update(
                    payload.investing.model_dump(mode="json", exclude_unset=True)
                )
                value = InvestingPrefs.model_validate(merged_investing).model_dump(mode="json")
            setattr(member, key, value)
        if member.is_self and (payload.residency is not None or payload.tax is not None):
            profile = user.session.scalar(
                select(Profile).where(Profile.user_id == user.household_owner_id)
            )
            if profile is None:
                profile = Profile(
                    user_id=user.household_owner_id,
                    residency=member.residency,
                    tax_regime=(member.tax or {}).get("tax_regime"),
                    tax_slab_pct=(member.tax or {}).get("tax_slab_pct"),
                )
                user.session.add(profile)
            else:
                if payload.residency is not None:
                    profile.residency = dict(member.residency)
                if payload.tax is not None:
                    profile.tax_regime = (member.tax or {}).get("tax_regime")
                    profile.tax_slab_pct = (member.tax or {}).get("tax_slab_pct")
                    financial = dict(profile.financial_context or {})
                    investing = dict(profile.investing or {})
                    tax = dict(member.tax or {})
                    for key in ("annual_income", "employment_type"):
                        if key in tax:
                            financial[key] = tax[key]
                    if "birth_year" in tax:
                        investing["birth_year"] = tax["birth_year"]
                    profile.financial_context = financial
                    profile.investing = investing
        user.session.flush()
        return member

    @router.delete(
        "/v1/members/{member_id}",
        status_code=204,
        response_class=Response,
        response_model=None,
    )
    def delete_member(member_id: UUID, user: AuthUser) -> Response:
        member = user.session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user.household_owner_id, HouseholdMember.id == member_id
            )
        )
        if member is None:
            raise HTTPException(status_code=404, detail="The member was not found.")
        if member.is_self:
            raise HTTPException(status_code=409, detail="The self member cannot be deleted.")
        member_has_login = user.session.scalar(
            text("SELECT app_member_has_active_household_grant(:owner_id, :member_id)"),
            {"owner_id": user.household_owner_id, "member_id": member_id},
        )
        if member_has_login:
            raise HTTPException(status_code=409, detail="member_linked_to_login")
        linked_assets = (
            user.session.scalar(
                select(func.count())
                .select_from(ManualAsset)
                .outerjoin(Account, Account.id == ManualAsset.account_id)
                .where(
                    ManualAsset.user_id == user.household_owner_id,
                    or_(ManualAsset.member_id == member_id, Account.member_id == member_id),
                )
            )
            or 0
        )
        if linked_assets:
            raise HTTPException(
                status_code=409,
                detail=f"This member still has {linked_assets} linked assets.",
            )
        linked_accounts = (
            user.session.scalar(
                select(func.count())
                .select_from(Account)
                .where(
                    Account.user_id == user.household_owner_id,
                    Account.member_id == member_id,
                )
            )
            or 0
        )
        if linked_accounts:
            raise HTTPException(
                status_code=409,
                detail=f"This member still has {linked_accounts} linked accounts.",
            )
        from wealth.db.models import Goal, HoldingLot

        linked_rows = (
            user.session.scalar(
                select(func.count())
                .select_from(HoldingLot)
                .where(
                    HoldingLot.user_id == user.household_owner_id, HoldingLot.member_id == member_id
                )
            )
            or 0
        ) + (
            user.session.scalar(
                select(func.count())
                .select_from(Goal)
                .where(Goal.user_id == user.household_owner_id, Goal.member_id == member_id)
            )
            or 0
        )
        if linked_rows:
            raise HTTPException(
                status_code=409,
                detail=f"This member still has {linked_rows} linked records.",
            )
        user.session.delete(member)
        try:
            user.session.flush()
        except SQLAlchemyError as error:
            diagnostic = getattr(getattr(error, "orig", None), "diag", None)
            if getattr(diagnostic, "message_primary", None) == "member_linked_to_login":
                raise HTTPException(status_code=409, detail="member_linked_to_login") from error
            raise
        return Response(status_code=204)

    @router.get("/v1/accounts", response_model=list[AccountResponse])
    def list_accounts(user: AuthUser) -> list[Any]:
        statement = (
            select(Account)
            .where(Account.user_id == user.household_owner_id)
            .order_by(Account.institution, Account.id)
        )
        return list(user.session.scalars(statement))

    @router.post("/v1/accounts", response_model=AccountResponse, status_code=201)
    def create_account(payload: AccountWrite, user: AuthUser) -> Any:
        from wealth.db.repo import acting_household_member

        member_id = (
            payload.member_id
            or acting_household_member(user.session, user.user_id, user.household_owner_id).id
        )
        _require_member(user.session, user.household_owner_id, member_id)
        account = Account(
            user_id=user.household_owner_id,
            **{**payload.model_dump(exclude={"member_id"}), "member_id": member_id},
        )
        user.session.add(account)
        user.session.flush()
        return account

    @router.patch("/v1/accounts/{account_id}", response_model=AccountResponse)
    def patch_account(account_id: UUID, payload: AccountPatch, user: AuthUser) -> Any:
        account = user.session.scalar(
            select(Account).where(
                Account.user_id == user.household_owner_id, Account.id == account_id
            )
        )
        if account is None:
            raise HTTPException(status_code=404, detail="The account was not found.")
        merged = {
            "institution": account.institution,
            "account_type": account.account_type,
            "masked_number": account.masked_number,
            "member_id": account.member_id,
            "joint": account.joint,
            "tax_wrapper": account.tax_wrapper,
        }
        merged.update(payload.model_dump(exclude_unset=True))
        try:
            validated = AccountWrite.model_validate(merged)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="The account is invalid.") from error
        from wealth.db.repo import acting_household_member

        member_id = (
            validated.member_id
            or acting_household_member(user.session, user.user_id, user.household_owner_id).id
        )
        _require_member(user.session, user.household_owner_id, member_id)
        values = {**validated.model_dump(), "member_id": member_id}
        for key, value in values.items():
            setattr(account, key, value)
        user.session.flush()
        return account

    @router.delete(
        "/v1/accounts/{account_id}",
        status_code=204,
        response_class=Response,
        response_model=None,
    )
    def delete_account(account_id: UUID, user: AuthUser) -> Response:
        account = user.session.scalar(
            select(Account).where(
                Account.user_id == user.household_owner_id, Account.id == account_id
            )
        )
        if account is None:
            raise HTTPException(status_code=404, detail="The account was not found.")
        user.session.delete(account)
        return Response(status_code=204)

    @router.get("/v1/assets", response_model=list[AssetResponse])
    def list_assets(
        user: AuthUser,
        asset_type: Annotated[str | None, Query()] = None,
        member_id: Annotated[UUID | None, Query()] = None,
        status: Annotated[Literal["active", "closed"], Query()] = "active",
    ) -> list[dict[str, Any]]:
        if member_id is not None:
            _require_member(user.session, user.household_owner_id, member_id)
        statement = select(ManualAsset).where(
            ManualAsset.user_id == user.household_owner_id, ManualAsset.status == status
        )
        if asset_type:
            statement = statement.where(ManualAsset.asset_type == asset_type)
        if member_id is not None:
            statement = statement.where(ManualAsset.member_id == member_id)
        statement = statement.order_by(ManualAsset.created_at, ManualAsset.id)
        return [_asset_response(record) for record in user.session.scalars(statement)]

    @router.post("/v1/assets", response_model=AssetResponse, status_code=201)
    def create_asset(
        payload: AnyAsset,
        request: Request,
        user: AuthUser,
        today: Today,
    ) -> Any:
        from wealth.db.repo import acting_household_member, add_manual_asset

        member_id = (
            _validated_detail_member(payload)
            or acting_household_member(user.session, user.user_id, user.household_owner_id).id
        )
        _require_member(user.session, user.household_owner_id, member_id)
        payload = payload.model_copy(update={"member_id": str(member_id)})
        try:
            from wealth.marketdata.provider import DbPriceProvider

            valuation, note = _evaluate_asset(
                payload, today, session=user.session, prices=DbPriceProvider(user.session)
            )
        except RateUnavailable:
            return _rate_needed_problem(request)
        record = add_manual_asset(
            user.session,
            user.household_owner_id,
            payload,
            value_cached=valuation.value_today if valuation else None,
            value_as_of=today if valuation else None,
        )
        record.valuation = _valuation_json(valuation) if valuation else None
        user.session.flush()
        return _asset_response(record)

    @router.get("/v1/assets/{asset_id}", response_model=AssetDetailResponse)
    def get_asset(asset_id: UUID, user: AuthUser) -> dict[str, Any]:
        record = user.session.scalar(
            select(ManualAsset).where(
                ManualAsset.user_id == user.household_owner_id, ManualAsset.id == asset_id
            )
        )
        if record is None:
            raise HTTPException(status_code=404, detail="The asset was not found.")
        history = user.session.scalars(
            select(AssetHistory)
            .where(
                AssetHistory.user_id == user.household_owner_id, AssetHistory.asset_id == asset_id
            )
            .order_by(AssetHistory.changed_at.desc(), AssetHistory.id.desc())
            .limit(20)
        )
        return {**_asset_response(record), "history": list(history)}

    @router.patch("/v1/assets/{asset_id}", response_model=AssetResponse)
    def patch_asset(
        asset_id: UUID,
        payload: dict[str, Any],
        request: Request,
        user: AuthUser,
        today: Today,
    ) -> Any:
        record = user.session.scalar(
            select(ManualAsset).where(
                ManualAsset.user_id == user.household_owner_id, ManualAsset.id == asset_id
            )
        )
        if record is None:
            raise HTTPException(status_code=404, detail="The asset was not found.")
        merged = {
            "member_id": str(record.member_id),
            "joint": record.joint,
            **record.details,
            **payload,
        }
        validated = _validate_asset(merged)
        member_id = _validated_detail_member(validated) or record.member_id
        _require_member(user.session, user.household_owner_id, member_id)
        try:
            from wealth.marketdata.provider import DbPriceProvider

            valuation, _note = _evaluate_asset(
                validated,
                today,
                session=user.session,
                prices=DbPriceProvider(user.session),
            )
        except RateUnavailable:
            return _rate_needed_problem(request)
        old_details = dict(record.details)
        new_details = validated.model_dump(mode="json")
        changed = _changed_fields(old_details, new_details)
        if changed:
            record.details = new_details
            record.member_id = member_id
            record.joint = validated.joint
            record.valuation = _valuation_json(valuation) if valuation else None
            record.value_cached = valuation.value_today if valuation else None
            record.value_as_of = today if valuation else None
            user.session.add(
                AssetHistory(
                    user_id=user.household_owner_id,
                    asset_id=record.id,
                    changed_fields=changed,
                    changed_by="user",
                )
            )
            user.session.flush()
        return _asset_response(record)

    @router.delete(
        "/v1/assets/{asset_id}",
        status_code=204,
        response_class=Response,
        response_model=None,
    )
    def delete_asset(
        asset_id: UUID,
        user: AuthUser,
        hard: bool = Query(default=False),
    ) -> Response:
        record = user.session.scalar(
            select(ManualAsset).where(
                ManualAsset.user_id == user.household_owner_id, ManualAsset.id == asset_id
            )
        )
        if record is None:
            raise HTTPException(status_code=404, detail="The asset was not found.")
        if hard:
            user.session.delete(record)
        else:
            record.status = "closed"
            user.session.flush()
        return Response(status_code=204)

    @router.post("/v1/assets/quick-add", response_model=QuickAddResponse)
    def quick_add(payload: dict[str, str], request: Request, _user: AuthUser) -> Any:
        text = payload.get("text")
        if not isinstance(text, str):
            return _problem(
                request,
                422,
                "Unprocessable Content",
                "The request must include text to parse.",
                {"code": "not_understood"},
            )
        try:
            result = parse_quick_add(text)
        except QuickAddNotUnderstood:
            return _problem(
                request,
                422,
                "Unprocessable Content",
                "The text could not be matched to a supported asset type.",
                {"code": "not_understood"},
            )
        return {
            "asset_type": result.asset_type,
            "draft": result.draft,
            "missing": list(result.missing),
            "confidence": result.confidence,
        }

    @router.get("/v1/goals", response_model=list[GoalResponse])
    def list_goals(user: AuthUser) -> list[Any]:
        return list(
            user.session.scalars(
                select(Goal)
                .where(Goal.user_id == user.household_owner_id)
                .order_by(Goal.priority, Goal.id)
            )
        )

    @router.get("/v1/goals/{goal_id}", response_model=GoalResponse)
    def get_goal(goal_id: UUID, user: AuthUser) -> Any:
        goal = user.session.scalar(
            select(Goal).where(Goal.user_id == user.household_owner_id, Goal.id == goal_id)
        )
        if goal is None:
            raise HTTPException(status_code=404, detail="The goal was not found.")
        return goal

    @router.post("/v1/goals", response_model=GoalResponse, status_code=201)
    def create_goal(payload: GoalInput, user: AuthUser) -> Any:
        values = payload.model_dump()
        values["goal_type"] = payload.goal_type.value
        _require_member(user.session, user.household_owner_id, payload.member_id)
        goal = Goal(user_id=user.household_owner_id, **values)
        user.session.add(goal)
        user.session.flush()
        return goal

    @router.patch("/v1/goals/{goal_id}", response_model=GoalResponse)
    def patch_goal(goal_id: UUID, payload: dict[str, Any], user: AuthUser) -> Any:
        goal = user.session.scalar(
            select(Goal).where(Goal.user_id == user.household_owner_id, Goal.id == goal_id)
        )
        if goal is None:
            raise HTTPException(status_code=404, detail="The goal was not found.")
        merged = {
            "name": goal.name,
            "goal_type": goal.goal_type,
            "target_amount": goal.target_amount,
            "target_year": goal.target_year,
            "currency": goal.currency,
            "priority": goal.priority,
            "member_id": goal.member_id,
        }
        merged.update(payload)
        try:
            validated = GoalInput.model_validate(merged)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="The goal is invalid.") from error
        _require_member(user.session, user.household_owner_id, validated.member_id)
        for key, value in validated.model_dump().items():
            setattr(goal, key, value.value if key == "goal_type" else value)
        user.session.flush()
        return goal

    @router.delete(
        "/v1/goals/{goal_id}", status_code=204, response_class=Response, response_model=None
    )
    def delete_goal(goal_id: UUID, user: AuthUser) -> Response:
        goal = user.session.scalar(
            select(Goal).where(Goal.user_id == user.household_owner_id, Goal.id == goal_id)
        )
        if goal is None:
            raise HTTPException(status_code=404, detail="The goal was not found.")
        user.session.delete(goal)
        return Response(status_code=204)

    @router.put("/v1/me/checklist", response_model=ChecklistWrite)
    def put_checklist(payload: ChecklistWrite, user: AuthUser) -> dict[str, Any]:
        statement = pg_insert(Profile).values(
            id=uuid4(), user_id=user.household_owner_id, checklist=payload.categories
        )
        statement = statement.on_conflict_do_update(
            index_elements=[Profile.user_id],
            set_={"checklist": statement.excluded.checklist, "updated_at": func.now()},
        )
        user.session.execute(statement)
        return {"categories": payload.categories}

    @router.get("/v1/me/checklist", response_model=ChecklistWrite)
    def get_checklist(user: AuthUser) -> dict[str, Any]:
        profile = user.session.scalar(
            select(Profile).where(Profile.user_id == user.household_owner_id)
        )
        return {"categories": list(profile.checklist) if profile is not None else []}

    @router.get("/v1/networth", response_model=NetworthResponse)
    def get_networth(
        user: AuthUser,
        today: Today,
        member_id: Annotated[UUID | None, Query()] = None,
    ) -> Any:
        if member_id is not None:
            _require_member(user.session, user.household_owner_id, member_id)
        return refresh_networth_snapshot(user.session, user.household_owner_id, today, member_id)

    return router
