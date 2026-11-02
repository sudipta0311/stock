"""Record, edit, and list buys entered from weekly actions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from wealth.api.personalisation import ActionResponse
from wealth.core.money import round_paise
from wealth.db.models import (
    Account,
    AuditLog,
    HouseholdMember,
    RecordedBuy,
    Security,
)
from wealth.db.models import (
    Action as ActionRow,
)
from wealth.marketdata import store
from wealth.marketdata.provider import DbPriceProvider
from wealth.policy.actions import load_buying_rules
from wealth.valuation import PriceUnavailable

_PLAIN_FUND_NAMES = {
    "in_core_index": "Indian index fund",
    "in_international_index_fund": "International index fund in rupees",
    "in_listed_international_etf": "India-listed international fund",
    "us_listed_india_etf": "US-listed India fund",
    "ucits_global_etf": "Ireland-listed international fund",
    "us_broad_market_etf": "Broad US share fund",
    "mutual_fund": "Indian fund",
    "abroad_etf": "European fund",
    "gold_etf": "Gold fund",
}
_SECURITY_MARKETS = frozenset({"IN", "US", "DK", "EU", "GB"})


class BuyApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class BuyLineInput(BuyApiModel):
    line: int = Field(ge=0)
    amount: str
    currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    units: str | None = None


class RecordBuyRequest(BuyApiModel):
    bought_on: date | None = None
    lines: list[BuyLineInput] = Field(min_length=1)


class RecordedBuyPatch(BuyApiModel):
    bought_on: date | None = None
    amount: str | None = None
    units: str | None = None


class RecordedBuyResponse(BuyApiModel):
    id: UUID
    action_id: UUID | None
    member_id: UUID
    line: int
    bought_on: date
    amount: str
    currency: str
    amount_inr: str | None
    units: str | None
    units_entered: bool
    security_id: UUID | None
    security_type: str
    security_symbol: str | None
    security_isin: str | None
    name: str
    sleeve: str
    instrument_category: str
    account_type: str | None
    status: Literal["waiting", "matched", "removed"]


class ActionBoughtResponse(BuyApiModel):
    action: ActionResponse
    recorded_buys: list[RecordedBuyResponse]


def _decimal_string(value: str, field: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"{field} must be a decimal string.") from error
    if not result.is_finite():
        raise HTTPException(status_code=422, detail=f"{field} must be finite.")
    return result


def _details(action: ActionRow) -> dict[str, Any]:
    payload = action.payload if isinstance(action.payload, dict) else {}
    nested = payload.get("action")
    return nested if isinstance(nested, dict) else payload


def _buy_lines(action: ActionRow) -> list[dict[str, Any]]:
    details = _details(action)
    params = details.get("params")
    params = params if isinstance(params, dict) else {}
    raw_lines = params.get("abroad_buy_list", details.get("abroad_buy_list", []))
    return (
        [line for line in raw_lines if isinstance(line, dict)]
        if isinstance(raw_lines, list)
        else []
    )


def _plain_category_name(category: str) -> str:
    known = _PLAIN_FUND_NAMES.get(category)
    if known is not None:
        return known
    return category.replace("_", " ").strip().title() or "Fund"


def _buy_date(action: ActionRow, bought_on: date, today: date) -> None:
    if bought_on > today:
        raise HTTPException(status_code=422, detail="The buy date cannot be in the future.")
    try:
        year_text, week_text = action.week.split("-W")
        first_day = date.fromisocalendar(int(year_text), int(week_text), 1)
    except (ValueError, AttributeError) as error:
        raise HTTPException(status_code=422, detail="The action week is invalid.") from error
    if bought_on < first_day - timedelta(days=14):
        raise HTTPException(status_code=422, detail="The buy date is too early for this step.")


def _fx_to_inr(session: Session, currency: str, on: date) -> Decimal | None:
    if currency == "INR":
        return Decimal(1)
    row = store.latest_fx_rate(session, f"{currency}/INR", on)
    return row[0] if row is not None and row[0] > 0 else None


def _security_amount_in_currency(
    amount_inr: Decimal | None, security: Security | None, session: Session, on: date
) -> Decimal | None:
    if amount_inr is None or security is None:
        return None
    rate = _fx_to_inr(session, security.currency, on)
    return amount_inr / rate if rate is not None and rate > 0 else None


def _resolve_units_and_cost(
    session: Session,
    security: Security | None,
    bought_on: date,
    amount_inr: Decimal | None,
    units: Decimal | None,
    *,
    units_entered: bool,
) -> tuple[Decimal | None, Decimal | None, Decimal | None, bool]:
    quote = None
    if security is not None and units is None and amount_inr is not None:
        try:
            provider = DbPriceProvider(session)
            quote = (
                provider.nav(
                    security.isin or security.amfi_scheme_code or security.symbol,
                    bought_on,
                )
                if security.security_type == "mf"
                else provider.price(security.id, bought_on)
            )
        except PriceUnavailable:
            quote = None
        if quote is not None and (bought_on - quote.as_of).days <= 5:
            security_amount = _security_amount_in_currency(amount_inr, security, session, bought_on)
            if security_amount is not None and quote.value > 0:
                units = security_amount / quote.value
                units_entered = False

    security_amount = _security_amount_in_currency(amount_inr, security, session, bought_on)
    cost_per_unit = (
        security_amount / units
        if security_amount is not None and units is not None and units > 0
        else None
    )
    cost_per_unit_inr = (
        amount_inr / units if amount_inr is not None and units is not None and units > 0 else None
    )
    return units, cost_per_unit, cost_per_unit_inr, units_entered


def _response(row: RecordedBuy) -> RecordedBuyResponse:
    return RecordedBuyResponse(
        id=row.id,
        action_id=row.action_id,
        member_id=row.member_id,
        line=row.line,
        bought_on=row.bought_on,
        amount=format(row.amount, "f"),
        currency=row.currency,
        amount_inr=format(row.amount_inr, ".2f") if row.amount_inr is not None else None,
        units=format(row.units, "f") if row.units is not None else None,
        units_entered=row.units_entered,
        security_id=row.security_id,
        security_type=row.security_type,
        security_symbol=row.security_symbol,
        security_isin=row.security_isin,
        name=row.name,
        sleeve=row.sleeve,
        instrument_category=row.instrument_category,
        account_type=row.account_type,
        status=row.status,
    )


def _member_for_action(
    session: Session, user_id: UUID, details: dict[str, Any], line: dict[str, Any] | None
) -> HouseholdMember:
    selected = line.get("member_id") if line is not None else None
    selected = selected or details.get("member_id")
    member: HouseholdMember | None = None
    if selected is not None:
        try:
            member_id = UUID(str(selected))
        except (ValueError, TypeError, AttributeError):
            member_id = None
        if member_id is not None:
            member = session.scalar(
                select(HouseholdMember).where(
                    HouseholdMember.user_id == user_id,
                    HouseholdMember.id == member_id,
                )
            )
    if member is None and selected in (None, "self"):
        member = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user_id,
                HouseholdMember.is_self.is_(True),
            )
        )
    if member is None:
        raise HTTPException(status_code=422, detail="The step's household member was not found.")
    return member


def _find_security(
    session: Session,
    details: dict[str, Any],
    line: dict[str, Any] | None,
) -> Security | None:
    params = details.get("params") if isinstance(details.get("params"), dict) else {}
    security_id = details.get("security_id") or params.get("security_id")
    if line is not None:
        security_id = line.get("security_id") or security_id
    if security_id is not None:
        try:
            found = session.get(Security, UUID(str(security_id)))
        except (ValueError, TypeError, AttributeError):
            found = None
        if found is not None:
            return found
    instrument = line.get("instrument") if line is not None else None
    instrument = instrument if isinstance(instrument, dict) else {}
    isin = str(instrument.get("isin") or "").strip().upper()
    ticker = (
        str(
            instrument.get("ticker")
            or instrument.get("symbol")
            or (line or {}).get("ticker")
            or (line or {}).get("symbol")
            or details.get("ticker")
            or details.get("symbol")
            or params.get("ticker")
            or params.get("symbol")
            or ""
        )
        .strip()
        .upper()
    )
    if isin:
        found = session.scalar(select(Security).where(Security.isin == isin))
        if found is not None:
            return found
    if not ticker:
        return None

    reason_codes = (line or {}).get("reason_codes")
    expected_market: str | None = None
    expected_exchange: str | None = None
    if isinstance(reason_codes, list) and "approved_us_stock_idea" in reason_codes:
        expected_market = "US"
    elif line is not None:
        raw_market = line.get("market") or instrument.get("market")
        raw_exchange = line.get("exchange") or instrument.get("exchange")
        market_value = str(raw_market or "").strip().upper()
        if market_value in _SECURITY_MARKETS:
            expected_market = market_value
        expected_exchange = str(raw_exchange or (raw_market if not expected_market else "") or "")
        expected_exchange = expected_exchange.strip().upper() or None
    else:
        raw_market = details.get("market") or params.get("market")
        market_value = str(raw_market or "").strip().upper()
        expected_market = market_value if market_value in _SECURITY_MARKETS else "IN"

    if expected_market is None and expected_exchange is None:
        return None

    statement = select(Security).where(
        or_(Security.symbol == ticker, Security.nse_symbol == ticker)
    )
    if expected_market is not None:
        statement = statement.where(Security.market == expected_market)
    if expected_exchange is not None:
        statement = statement.where(func.upper(Security.exchange) == expected_exchange)
    matches = session.scalars(statement).all()
    return matches[0] if len(matches) == 1 else None


def _account_id(
    session: Session, user_id: UUID, member_id: UUID, account_type: str | None
) -> UUID | None:
    if account_type not in {"ask", "ordinary"}:
        return None
    return session.scalar(
        select(Account.id)
        .where(
            Account.user_id == user_id,
            Account.member_id == member_id,
            Account.tax_wrapper == account_type,
        )
        .order_by(Account.id)
        .limit(1)
    )


def _create_buy_row(
    session: Session,
    user_id: UUID,
    action: ActionRow,
    details: dict[str, Any],
    line_input: BuyLineInput,
    *,
    bought_on: date,
    buy_line: dict[str, Any] | None,
) -> RecordedBuy:
    amount = _decimal_string(line_input.amount, "amount")
    units = _decimal_string(line_input.units, "units") if line_input.units is not None else None
    if amount <= 0 or (units is not None and units <= 0):
        raise HTTPException(status_code=422, detail="Amount and units must be greater than zero.")

    abroad = action.kind == "BUY_ABROAD"
    if abroad:
        if buy_line is None:
            raise HTTPException(status_code=422, detail="The buy line was not found.")
        expected_currency = (
            "DKK" if "amount_dkk" in buy_line else str(buy_line.get("currency") or "DKK").upper()
        )
        if line_input.currency.upper() != expected_currency:
            raise HTTPException(status_code=422, detail="The currency does not match this line.")
    else:
        expected_currency = str(details.get("currency") or "INR").upper()
        if line_input.currency.upper() != expected_currency:
            raise HTTPException(status_code=422, detail="The currency does not match this step.")

    member = _member_for_action(session, user_id, details, buy_line)
    security = _find_security(session, details, buy_line)
    if action.kind == "BUY_STOCK_TRANCHE" and units is None:
        raw_quantity = details.get("quantity")
        if raw_quantity is not None:
            units = _decimal_string(str(raw_quantity), "units")
            if units <= 0:
                raise HTTPException(status_code=422, detail="Units must be greater than zero.")
    amount_inr_rate = _fx_to_inr(session, line_input.currency.upper(), bought_on)
    amount_inr = round_paise(amount * amount_inr_rate) if amount_inr_rate is not None else None
    units, cost_per_unit, cost_per_unit_inr, units_entered = _resolve_units_and_cost(
        session,
        security,
        bought_on,
        amount_inr,
        units,
        units_entered=line_input.units is not None or action.kind == "BUY_STOCK_TRANCHE",
    )
    instrument = buy_line.get("instrument") if buy_line is not None else None
    instrument = instrument if isinstance(instrument, dict) else {}
    account_type_raw = str(buy_line.get("account") or "").lower() if buy_line else ""
    account_type = account_type_raw if account_type_raw in {"ask", "ordinary"} else None
    reason_codes = buy_line.get("reason_codes") if buy_line else None
    is_stock_line = (
        isinstance(reason_codes, list) and "approved_us_stock_idea" in reason_codes
    ) or str(instrument.get("type") or instrument.get("security_type") or "").lower() in {
        "equity",
        "stock",
    }
    category = str(
        (buy_line or {}).get("instrument_category")
        or (buy_line or {}).get("category")
        or details.get("instrument_category")
        or "in_core_index"
    )
    sleeve = str((buy_line or {}).get("sleeve") or details.get("sleeve") or "equity_india")
    name = (
        security.name
        if security is not None
        else str(
            instrument.get("name")
            or instrument.get("ticker")
            or instrument.get("symbol")
            or _plain_category_name(category)
        )[:160]
    )
    return RecordedBuy(
        user_id=user_id,
        member_id=member.id,
        action_id=action.id,
        account_id=_account_id(session, user_id, member.id, account_type),
        line=line_input.line,
        bought_on=bought_on,
        amount=round_paise(amount),
        currency=line_input.currency.upper(),
        amount_inr=amount_inr,
        units=units,
        units_entered=units_entered,
        cost_per_unit=cost_per_unit,
        cost_per_unit_inr=cost_per_unit_inr,
        security_id=security.id if security is not None else None,
        security_type=(
            security.security_type if security is not None else "equity" if is_stock_line else "mf"
        ),
        security_symbol=(
            security.nse_symbol or security.symbol
            if security is not None
            else str(instrument.get("ticker") or instrument.get("symbol") or "") or None
        ),
        security_isin=(
            security.isin if security is not None else str(instrument.get("isin") or "") or None
        ),
        name=name,
        sleeve=sleeve,
        instrument_category=category,
        account_type=account_type,
        status="waiting",
    )


def _can_write(user: Any) -> None:
    if getattr(user, "role", "owner") == "viewer":
        raise HTTPException(status_code=403, detail="Viewers cannot record buys.")


def _audit_buy(
    session: Session,
    owner_id: UUID,
    actor_id: UUID,
    row: RecordedBuy,
    action: str,
) -> None:
    session.add(
        AuditLog(
            user_id=owner_id,
            actor=str(actor_id),
            actor_user_id=actor_id,
            action=action,
            detail={
                "recorded_buy_id": str(row.id),
                "action_id": str(row.action_id) if row.action_id is not None else None,
                "member_id": str(row.member_id),
                "status": row.status,
            },
        )
    )


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    from wealth.api.money_map import get_today

    router = APIRouter()
    user_dependency = Depends(authenticated_user)
    today_dependency = Depends(get_today)

    @router.post(
        "/v1/actions/{action_id}/bought",
        response_model=ActionBoughtResponse,
    )
    def record_bought(
        action_id: UUID,
        payload: RecordBuyRequest,
        user: Any = user_dependency,
        today: date = today_dependency,
    ) -> ActionBoughtResponse:
        _can_write(user)
        action = user.session.scalar(
            select(ActionRow)
            .where(
                ActionRow.user_id == user.household_owner_id,
                ActionRow.id == action_id,
            )
            .with_for_update()
        )
        if action is None:
            raise HTTPException(status_code=404, detail="The action was not found.")
        if action.kind not in {"BUY_LUMPSUM", "BUY_STP", "BUY_ABROAD", "BUY_STOCK_TRANCHE"}:
            raise HTTPException(status_code=422, detail="This step cannot record a buy.")
        if action.status != "proposed":
            raise HTTPException(status_code=422, detail="This step is no longer proposed.")
        bought_on = payload.bought_on or today
        _buy_date(action, bought_on, today)
        details = _details(action)
        abroad_lines = _buy_lines(action) if action.kind == "BUY_ABROAD" else []
        request_lines: list[tuple[BuyLineInput, dict[str, Any] | None]] = []
        seen_lines: set[int] = set()
        for line_input in payload.lines:
            if line_input.line in seen_lines:
                raise HTTPException(status_code=422, detail="A buy line can only be recorded once.")
            seen_lines.add(line_input.line)
            if action.kind == "BUY_ABROAD":
                if line_input.line >= len(abroad_lines):
                    raise HTTPException(status_code=422, detail="The buy line was not found.")
                request_lines.append((line_input, abroad_lines[line_input.line]))
            else:
                if line_input.line != 0 or len(payload.lines) != 1:
                    raise HTTPException(status_code=422, detail="This step has one buy line.")
                request_lines.append((line_input, None))

        rows: list[RecordedBuy] = []
        for line_input, buy_line in request_lines:
            row = _create_buy_row(
                user.session,
                user.household_owner_id,
                action,
                details,
                line_input,
                bought_on=bought_on,
                buy_line=buy_line,
            )
            rows.append(row)

        previous_status = action.status
        action.status = "done"
        action.done_by = user.user_id
        user.session.add_all(rows)
        for row in rows:
            _audit_buy(
                user.session,
                user.household_owner_id,
                user.user_id,
                row,
                "recorded_buy_created",
            )
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
                    "to": "done",
                    "recorded_buys": len(rows),
                },
            )
        )
        user.session.flush()
        action_response = ActionResponse.model_validate(action).model_copy(
            update={"has_recorded_buys": True}
        )
        return ActionBoughtResponse(
            action=action_response,
            recorded_buys=[_response(row) for row in rows],
        )

    @router.get("/v1/recorded-buys", response_model=list[RecordedBuyResponse])
    def list_recorded_buys(
        user: Any = user_dependency,
        status: Annotated[Literal["waiting", "matched", "removed"] | None, Query()] = None,
        member_id: Annotated[UUID | None, Query()] = None,
    ) -> list[RecordedBuyResponse]:
        statement = select(RecordedBuy).where(RecordedBuy.user_id == user.household_owner_id)
        if status is not None:
            statement = statement.where(RecordedBuy.status == status)
        if member_id is not None:
            statement = statement.where(RecordedBuy.member_id == member_id)
        statement = statement.order_by(RecordedBuy.bought_on.desc(), RecordedBuy.created_at.desc())
        return [_response(row) for row in user.session.scalars(statement)]

    @router.patch("/v1/recorded-buys/{recorded_buy_id}", response_model=RecordedBuyResponse)
    def patch_recorded_buy(
        recorded_buy_id: Annotated[UUID, Path()],
        payload: RecordedBuyPatch,
        user: Any = user_dependency,
        today: date = today_dependency,
    ) -> RecordedBuyResponse:
        _can_write(user)
        row = user.session.scalar(
            select(RecordedBuy)
            .where(
                RecordedBuy.user_id == user.household_owner_id,
                RecordedBuy.id == recorded_buy_id,
            )
            .with_for_update()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="The buy was not found.")
        if row.status != "waiting":
            raise HTTPException(status_code=422, detail="Only waiting buys can be edited.")
        changes = payload.model_dump(exclude_unset=True)
        action = (
            user.session.scalar(
                select(ActionRow).where(
                    ActionRow.user_id == row.user_id,
                    ActionRow.id == row.action_id,
                )
            )
            if row.action_id is not None
            else None
        )
        bought_on = changes.get("bought_on", row.bought_on)
        if not isinstance(bought_on, date) or bought_on > today:
            raise HTTPException(status_code=422, detail="The buy date cannot be in the future.")
        if action is not None:
            _buy_date(action, bought_on, today)
        amount = row.amount
        if "amount" in changes:
            amount = _decimal_string(str(changes["amount"]), "amount")
            if amount <= 0:
                raise HTTPException(status_code=422, detail="Amount must be greater than zero.")
        units = row.units
        units_entered = row.units_entered
        if "units" in changes:
            raw_units = changes["units"]
            units = _decimal_string(str(raw_units), "units") if raw_units is not None else None
            if units is not None and units <= 0:
                raise HTTPException(status_code=422, detail="Units must be greater than zero.")
            units_entered = units is not None

        amount_inr_rate = _fx_to_inr(user.session, row.currency, bought_on)
        amount_inr = round_paise(amount * amount_inr_rate) if amount_inr_rate is not None else None
        security = (
            user.session.get(Security, row.security_id) if row.security_id is not None else None
        )
        if (
            "units" not in changes
            and not row.units_entered
            and ("amount" in changes or "bought_on" in changes)
        ):
            units = None
        units, cost_per_unit, cost_per_unit_inr, units_entered = _resolve_units_and_cost(
            user.session,
            security,
            bought_on,
            amount_inr,
            units,
            units_entered=units_entered,
        )
        row.bought_on = bought_on
        row.amount = round_paise(amount)
        row.amount_inr = amount_inr
        row.units = units
        row.units_entered = units_entered
        row.cost_per_unit = cost_per_unit
        row.cost_per_unit_inr = cost_per_unit_inr
        _audit_buy(
            user.session,
            user.household_owner_id,
            user.user_id,
            row,
            "recorded_buy_updated",
        )
        user.session.flush()
        return _response(row)

    @router.delete("/v1/recorded-buys/{recorded_buy_id}", response_model=RecordedBuyResponse)
    def remove_recorded_buy(
        recorded_buy_id: Annotated[UUID, Path()],
        user: Any = user_dependency,
    ) -> RecordedBuyResponse:
        _can_write(user)
        row = user.session.scalar(
            select(RecordedBuy)
            .where(
                RecordedBuy.user_id == user.household_owner_id,
                RecordedBuy.id == recorded_buy_id,
            )
            .with_for_update()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="The buy was not found.")
        if row.status != "waiting":
            raise HTTPException(status_code=422, detail="Only waiting buys can be removed.")
        row.status = "removed"
        _audit_buy(
            user.session,
            user.household_owner_id,
            user.user_id,
            row,
            "recorded_buy_removed",
        )
        if row.action_id is not None:
            action = user.session.scalar(
                select(ActionRow)
                .where(ActionRow.user_id == row.user_id, ActionRow.id == row.action_id)
                .with_for_update()
            )
            remaining = user.session.scalar(
                select(RecordedBuy.id).where(
                    RecordedBuy.user_id == row.user_id,
                    RecordedBuy.action_id == row.action_id,
                    RecordedBuy.id != row.id,
                    RecordedBuy.status.in_(("waiting", "matched")),
                )
            )
            if action is not None and remaining is None:
                previous_status = action.status
                action.status = "proposed"
                action.done_by = None
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
                            "to": "proposed",
                        },
                    )
                )
        user.session.flush()
        return _response(row)

    return router


def match_recorded_buys_for_import(
    session: Session,
    user_id: UUID,
    member_id: UUID,
    source: str | None,
    statement_date: date | None,
    confirmed_on: date,
    security_ids: set[UUID],
) -> int:
    """Stop temporary values only for the importing member and confirmed securities."""
    settlement_days = int(load_buying_rules()["recorded_buys"]["statement_settlement_days"])
    effective_date = statement_date or confirmed_on
    rows = list(
        session.scalars(
            select(RecordedBuy).where(
                RecordedBuy.user_id == user_id,
                RecordedBuy.member_id == member_id,
                RecordedBuy.status == "waiting",
            )
        )
    )
    count = 0
    confirmed_securities = (
        {
            row.id: row
            for row in session.scalars(select(Security).where(Security.id.in_(security_ids)))
        }
        if security_ids
        else {}
    )
    for row in rows:
        if source == "cams_kfin" and row.security_type == "equity":
            continue
        matches_security = row.security_id in security_ids
        if not matches_security and row.security_id is None and row.security_isin:
            matches_security = any(
                security.isin is not None and security.isin.upper() == row.security_isin.upper()
                for security in confirmed_securities.values()
            )
        if not matches_security and row.security_id is None and row.security_symbol:
            matches_security = any(
                row.security_symbol.upper()
                in {value.upper() for value in (security.symbol, security.nse_symbol) if value}
                for security in confirmed_securities.values()
            )
        if matches_security and effective_date >= row.bought_on:
            row.status = "matched"
            count += 1
            continue
        if (
            row.security_id is None
            and not row.security_isin
            and not row.security_symbol
            and row.security_type != "equity"
            and source in {"nsdl", "cdsl", "cams_kfin"}
            and effective_date >= row.bought_on + timedelta(days=settlement_days)
        ):
            row.status = "matched"
            count += 1
    session.flush()
    return count
