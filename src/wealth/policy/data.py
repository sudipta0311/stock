"""Tenant-scoped loading of normalized inputs for the pure Personaliser."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import UUID

import yaml
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from wealth.core.dates import fy_start
from wealth.core.money import round_paise
from wealth.db.models import (
    Account,
    BrokerCashFlowRecord,
    HoldingLot,
    HouseholdMember,
    LoanEventRecord,
    ManualAsset,
    Profile,
    RealisedTrade,
    RecordedBuy,
    ResearchPackRecord,
    Security,
)
from wealth.db.models import (
    Action as ActionRow,
)
from wealth.db.models import (
    Goal as GoalRow,
)
from wealth.marketdata import store
from wealth.marketdata.provider import DbPriceProvider
from wealth.policy.actions import SatelliteIdea, satellite_idea_from_pack
from wealth.policy.debt_yields import load_debt_yields
from wealth.policy.global_split import GeoTilts
from wealth.policy.household import MemberProfile
from wealth.policy.loans import LoanCandidate
from wealth.policy.personalise import (
    HoldingInput,
    HoldingLotInput,
    MarketContext,
    PolicyProfile,
    RealisedTradesThisFy,
    RecordedBuyInput,
    ResearchPack,
    UserData,
    WeeklySpendInput,
)
from wealth.policy.residency import load_residency_rules
from wealth.rates.reference import RateUnavailable
from wealth.schemas.assets import AnyAsset, LoanAsset, SavingsAsset, parse_asset
from wealth.schemas.common import ValuationMethod
from wealth.schemas.profile import (
    FinancialContext,
    Goal,
    InvestingPrefs,
    MemberTax,
    residency_from_stored,
)
from wealth.valuation import PriceUnavailable, Valuation, value_asset
from wealth.valuation.loans import LoanEvent, loan_position

_MANUAL_PRICES_PATH = Path(__file__).resolve().parents[3] / "rules" / "manual_prices.yaml"


def _financial_context(profile: Profile | None) -> FinancialContext:
    stored = dict(profile.financial_context or {}) if profile is not None else {}
    # Skipped onboarding questions are saved as nulls, so default on None as well as absence.
    defaults: dict[str, object] = {
        "monthly_expenses": "0",
        "dependants": 0,
        "employment_type": "salaried",
        "income_stability": "medium",
    }
    for key, default in defaults.items():
        if stored.get(key) is None:
            stored[key] = default
    if stored.get("city_tier") is None and profile is not None:
        stored["city_tier"] = profile.city_tier
    if stored.get("tax_regime") is None and profile is not None:
        stored["tax_regime"] = profile.tax_regime
    return FinancialContext.model_validate(stored)


def _goals(session: Session, user_id: UUID) -> tuple[Goal, ...]:
    rows = session.scalars(
        select(GoalRow).where(GoalRow.user_id == user_id).order_by(GoalRow.priority, GoalRow.id)
    )
    return tuple(
        Goal(
            name=row.name,
            goal_type=row.goal_type,
            target_amount=row.target_amount,
            target_year=row.target_year,
            currency=row.currency,
            priority=row.priority,
            member_id=row.member_id,
        )
        for row in rows
    )


def _recorded_buy_inputs(rows: tuple[RecordedBuy, ...]) -> tuple[RecordedBuyInput, ...]:
    return tuple(
        RecordedBuyInput(
            id=str(row.id),
            action_id=str(row.action_id) if row.action_id is not None else None,
            member_id=str(row.member_id),
            bought_on=row.bought_on,
            amount=row.amount,
            currency=row.currency,
            amount_inr=row.amount_inr,
            units=row.units,
            units_entered=row.units_entered,
            cost_per_unit=row.cost_per_unit,
            cost_per_unit_inr=row.cost_per_unit_inr,
            security_id=str(row.security_id) if row.security_id is not None else None,
            security_type=row.security_type,
            security_symbol=row.security_symbol,
            security_isin=row.security_isin,
            name=row.name,
            sleeve=row.sleeve,
            instrument_category=row.instrument_category,
            account_type=row.account_type,
            status=row.status,
        )
        for row in rows
    )


def _amount_inr_for_recorded_buy(session: Session, buy: RecordedBuy) -> Decimal | None:
    if buy.amount_inr is not None:
        return buy.amount_inr
    if buy.currency == "INR":
        return round_paise(buy.amount)
    fx_row = store.latest_fx_rate(session, f"{buy.currency}/INR", buy.bought_on)
    return round_paise(buy.amount * fx_row[0]) if fx_row is not None and fx_row[0] > 0 else None


def _payload_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def _weekly_spend_for_recorded_buy(
    buy: RecordedBuy,
    action: ActionRow,
    recorded_amount_inr: Decimal | None,
) -> Decimal | None:
    payload = action.payload if isinstance(action.payload, dict) else {}
    details = payload.get("action") if isinstance(payload.get("action"), dict) else payload
    kind = str(details.get("kind") or action.kind)
    if kind == "BUY_ABROAD":
        params = details.get("params")
        lines = (
            params.get("abroad_buy_list", details.get("abroad_buy_list"))
            if isinstance(params, dict)
            else details.get("abroad_buy_list")
        )
        if not isinstance(lines, list) or not 0 <= buy.line < len(lines):
            return recorded_amount_inr
        line = lines[buy.line]
        if not isinstance(line, dict):
            return recorded_amount_inr
        if "amount_dkk" in line:
            display_currency = "DKK"
            displayed_amount = _payload_decimal(line.get("amount_dkk"))
        else:
            display_currency = str(line.get("currency") or "DKK").upper()
            displayed_amount = _payload_decimal(line.get("amount"))
        planned_amount_inr = _payload_decimal(line.get("amount_inr"))
        if (
            buy.currency == display_currency
            and displayed_amount is not None
            and buy.amount >= displayed_amount
            and planned_amount_inr is not None
        ):
            return max(recorded_amount_inr or Decimal(0), planned_amount_inr)
        return recorded_amount_inr

    if kind in {"BUY_LUMPSUM", "BUY_STP", "BUY_STOCK_TRANCHE"}:
        step_amount = _payload_decimal(details.get("amount"))
        if buy.currency == "INR" and step_amount is not None and buy.amount >= step_amount:
            return max(recorded_amount_inr or Decimal(0), step_amount)
    return recorded_amount_inr


def _weekly_spending_for_week(
    session: Session,
    user_id: UUID,
    week: str,
    recorded_buys: tuple[RecordedBuy, ...],
) -> tuple[WeeklySpendInput, ...]:
    buy_step_kinds = {"BUY_LUMPSUM", "BUY_STP", "BUY_ABROAD", "BUY_STOCK_TRANCHE"}
    actions = tuple(
        session.scalars(
            select(ActionRow).where(ActionRow.user_id == user_id, ActionRow.week == week)
        )
    )
    actions_by_id = {row.id: row for row in actions}
    buys_by_action: dict[UUID, list[RecordedBuy]] = {}
    for buy in recorded_buys:
        if buy.status in {"waiting", "matched"} and buy.action_id in actions_by_id:
            assert buy.action_id is not None
            buys_by_action.setdefault(buy.action_id, []).append(buy)

    spent_by_member_and_sleeve: dict[tuple[str, str], Decimal] = {}

    def add_spending(member_id: object, sleeve: object, amount: Decimal | None) -> None:
        if not isinstance(member_id, str) or not member_id:
            return
        if not isinstance(sleeve, str) or not sleeve or amount is None or amount <= 0:
            return
        key = (member_id, sleeve)
        spent_by_member_and_sleeve[key] = spent_by_member_and_sleeve.get(key, Decimal(0)) + amount

    for action_id, buys in buys_by_action.items():
        action = actions_by_id[action_id]
        for buy in buys:
            recorded_amount_inr = _amount_inr_for_recorded_buy(session, buy)
            amount = _weekly_spend_for_recorded_buy(buy, action, recorded_amount_inr)
            add_spending(str(buy.member_id), buy.sleeve, amount)

    for action in actions:
        if action.status != "done" or action.id in buys_by_action:
            continue
        payload = action.payload if isinstance(action.payload, dict) else {}
        details = payload.get("action") if isinstance(payload.get("action"), dict) else payload
        kind = str(details.get("kind") or action.kind)
        if kind not in buy_step_kinds:
            continue

        if kind == "BUY_ABROAD":
            params = details.get("params")
            lines = (
                params.get("abroad_buy_list", details.get("abroad_buy_list"))
                if isinstance(params, dict)
                else details.get("abroad_buy_list")
            )
            if isinstance(lines, list) and lines:
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    add_spending(
                        line.get("member_id") or details.get("member_id"),
                        line.get("sleeve") or details.get("sleeve"),
                        _payload_decimal(line.get("amount_inr")),
                    )
                continue

        amount = _payload_decimal(details.get("amount"))
        add_spending(details.get("member_id"), details.get("sleeve"), amount)

    return tuple(
        WeeklySpendInput(member_id=member_id, sleeve=sleeve, amount_inr=amount)
        for (member_id, sleeve), amount in sorted(spent_by_member_and_sleeve.items())
    )


def _manual_assets(
    session: Session, user_id: UUID, as_of: date, prices: DbPriceProvider
) -> tuple[tuple[tuple[AnyAsset, Valuation], ...], list[str]]:
    rows = session.scalars(
        select(ManualAsset)
        .where(ManualAsset.user_id == user_id, ManualAsset.status == "active")
        .order_by(ManualAsset.created_at, ManualAsset.id)
    )
    valued: list[tuple[AnyAsset, Valuation]] = []
    notes: list[str] = []
    for row in rows:
        details = dict(row.details)
        if row.asset_type == "loan" and "outstanding_as_of" not in details:
            saved_at = row.updated_at.date() if row.updated_at is not None else as_of
            details["outstanding_as_of"] = f"{saved_at.year:04d}-{saved_at.month:02d}"
        asset = parse_asset(details).model_copy(
            update={"member_id": str(row.member_id), "joint": row.joint}
        )
        if isinstance(asset, LoanAsset):
            event_rows = session.scalars(
                select(LoanEventRecord)
                .where(
                    LoanEventRecord.user_id == user_id,
                    LoanEventRecord.loan_id == row.id,
                )
                .order_by(LoanEventRecord.on_date, LoanEventRecord.created_at, LoanEventRecord.id)
            )
            events = tuple(
                LoanEvent(
                    kind=event.kind,
                    on_date=event.on_date,
                    amount=event.amount,
                    rate_pct=event.rate_pct,
                    emi=event.emi,
                    note=event.note,
                )
                for event in event_rows
            )
            position = loan_position(asset, events, as_of)
            rate = store.fresh_fx_rate_to_inr(session, asset.currency, as_of)
            if rate is None:
                notes.append(f"loan_fx_missing:{asset.currency}")
                continue
            asset = asset.model_copy(
                update={
                    "outstanding": position.outstanding * rate,
                    "emi": asset.emi * rate if asset.emi is not None else None,
                    "currency": "INR",
                }
            )
            valuation = Valuation(
                value_today=Decimal("0.00"),
                maturity_value=None,
                maturity_date=None,
                accrued_interest=Decimal("0.00"),
                principal_so_far=Decimal("0.00"),
                method=ValuationMethod.USER,
                notes=(),
            )
        elif isinstance(asset, SavingsAsset) and asset.currency != "INR":
            rate = store.fresh_fx_rate_to_inr(session, asset.currency, as_of)
            if rate is None:
                notes.append(f"savings_fx_missing:{asset.currency}")
                continue
            asset = asset.model_copy(update={"balance": asset.balance * rate, "currency": "INR"})
            valuation = value_asset(asset, as_of, assumed_start=as_of, prices=prices)
        else:
            try:
                valuation = value_asset(asset, as_of, assumed_start=as_of, prices=prices)
            except PriceUnavailable:
                notes.append(f"manual_asset_price_missing:{row.asset_type}")
                continue
            except RateUnavailable:
                notes.append(f"manual_asset_rate_missing:{row.asset_type}")
                continue
        valued.append((asset, valuation))
    return tuple(valued), notes


def _loan_candidates(
    session: Session, user_id: UUID, as_of: date, profile: Profile | None
) -> tuple[LoanCandidate, ...]:
    rows = session.scalars(
        select(ManualAsset)
        .where(
            ManualAsset.user_id == user_id,
            ManualAsset.asset_type == "loan",
            ManualAsset.status == "active",
        )
        .order_by(ManualAsset.created_at, ManualAsset.id)
    )
    candidates: list[LoanCandidate] = []
    for row in rows:
        details = dict(row.details)
        if "outstanding_as_of" not in details:
            saved_at = row.updated_at.date() if row.updated_at is not None else as_of
            details["outstanding_as_of"] = f"{saved_at.year:04d}-{saved_at.month:02d}"
        asset = parse_asset(details)
        if not isinstance(asset, LoanAsset):
            continue
        events = tuple(
            LoanEvent(
                kind=event.kind,
                on_date=event.on_date,
                amount=event.amount,
                rate_pct=event.rate_pct,
                emi=event.emi,
                note=event.note,
            )
            for event in session.scalars(
                select(LoanEventRecord)
                .where(
                    LoanEventRecord.user_id == user_id,
                    LoanEventRecord.loan_id == row.id,
                )
                .order_by(LoanEventRecord.on_date, LoanEventRecord.created_at, LoanEventRecord.id)
            )
        )
        position = loan_position(asset, events, as_of)
        fx_rate = store.fresh_fx_rate_to_inr(session, asset.currency, as_of)
        if fx_rate is None:
            continue
        member = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user_id,
                HouseholdMember.id == row.member_id,
            )
        )
        member_tax = dict(member.tax or {}) if member is not None else {}
        member_residency = dict(member.residency or {}) if member is not None else {}
        regime = member_tax.get("tax_regime") or (profile.tax_regime if profile else None)
        slab = member_tax.get("tax_slab_pct") or (profile.tax_slab_pct if profile else None)
        country = member_residency.get("country") or (
            (profile.residency or {}).get("country", "IN") if profile else "IN"
        )
        candidates.append(
            LoanCandidate(
                id=str(row.id),
                loan=asset,
                outstanding_inr=position.outstanding * fx_rate,
                emi_inr=asset.emi * fx_rate if asset.emi is not None else None,
                tax_regime=regime if regime in {"old", "new"} else None,
                tax_slab_pct=Decimal(str(slab)) if slab is not None else None,
                country="DK" if country == "DK" else "IN",
                member_id=str(row.member_id),
                never_pays_off=position.never_pays_off,
            )
        )
    return tuple(candidates)


def _holdings(
    session: Session,
    user_id: UUID,
    as_of: date,
    prices: DbPriceProvider,
    recorded_buys: tuple[RecordedBuy, ...] = (),
) -> tuple[HoldingInput, ...]:
    statement = (
        select(HoldingLot, Security)
        .join(Security, Security.id == HoldingLot.security_id)
        .where(HoldingLot.user_id == user_id)
        .order_by(HoldingLot.security_id, HoldingLot.acquired_on, HoldingLot.id)
    )
    grouped: dict[tuple[UUID, UUID], dict[str, object]] = {}
    for lot, security in session.execute(statement):
        owner_key = (security.id, lot.member_id)
        record = grouped.setdefault(
            owner_key,
            {
                "security": security,
                "quantity": Decimal(0),
                "lots": [],
                "price": None,
                "price_as_of": None,
                "price_source": None,
                "statement_value": Decimal(0),
                "statement_quantity": Decimal(0),
                "statement_as_of": None,
                "joint": lot.joint,
                "cost_value_inr": Decimal(0),
                "fallback_value_inr": Decimal(0),
            },
        )
        record["quantity"] = record["quantity"] + lot.quantity  # type: ignore[operator]
        if lot.statement_value is not None:
            record["statement_value"] = (
                record["statement_value"] + lot.statement_value  # type: ignore[operator]
            )
            record["statement_quantity"] = (
                record["statement_quantity"] + lot.quantity  # type: ignore[operator]
            )
            if lot.statement_date is not None and (
                record["statement_as_of"] is None or lot.statement_date > record["statement_as_of"]
            ):
                record["statement_as_of"] = lot.statement_date
        record["lots"].append(  # type: ignore[union-attr]
            HoldingLotInput(
                lot_id=str(lot.id),
                quantity=lot.quantity,
                cost_per_unit=lot.cost_per_unit,
                acquired_on=lot.acquired_on,
                member_id=str(lot.member_id),
                currency=lot.currency,
                cost_per_unit_inr=lot.cost_per_unit_inr,
                fx_missing=lot.fx_missing,
            )
        )

    for buy in recorded_buys:
        if buy.status != "waiting" or buy.security_id is None:
            continue
        security = session.get(Security, buy.security_id)
        if security is None:
            continue
        owner_key = (security.id, buy.member_id)
        record = grouped.setdefault(
            owner_key,
            {
                "security": security,
                "quantity": Decimal(0),
                "lots": [],
                "price": None,
                "price_as_of": None,
                "price_source": None,
                "statement_value": Decimal(0),
                "statement_quantity": Decimal(0),
                "statement_as_of": None,
                "joint": False,
                "cost_value_inr": Decimal(0),
                "fallback_value_inr": Decimal(0),
            },
        )
        if buy.units is None:
            if buy.amount_inr is not None:
                record["cost_value_inr"] = record["cost_value_inr"] + buy.amount_inr  # type: ignore[operator]
        else:
            record["quantity"] = record["quantity"] + buy.units  # type: ignore[operator]
            if buy.amount_inr is not None:
                record["fallback_value_inr"] = (
                    record["fallback_value_inr"] + buy.amount_inr  # type: ignore[operator]
                )
            record["lots"].append(  # type: ignore[union-attr]
                HoldingLotInput(
                    lot_id=f"recorded:{buy.id}",
                    quantity=buy.units,
                    cost_per_unit=buy.cost_per_unit,
                    acquired_on=buy.bought_on,
                    member_id=str(buy.member_id),
                    currency=security.currency,
                    cost_per_unit_inr=buy.cost_per_unit_inr,
                    fx_missing=buy.cost_per_unit_inr is None,
                )
            )
            if buy.cost_per_unit is not None:
                record["statement_value"] = (
                    record["statement_value"] + buy.cost_per_unit * buy.units  # type: ignore[operator]
                )
                record["statement_quantity"] = (
                    record["statement_quantity"] + buy.units  # type: ignore[operator]
                )
                current_statement_date = record["statement_as_of"]
                if current_statement_date is None or buy.bought_on > current_statement_date:
                    record["statement_as_of"] = buy.bought_on

    for record in grouped.values():
        if record["price"] is not None:
            continue
        security = record["security"]
        assert isinstance(security, Security)
        try:
            quote = (
                prices.nav(
                    security.isin or security.amfi_scheme_code or security.symbol,
                    as_of,
                )
                if security.security_type == "mf"
                else prices.price(security.id, as_of)
            )
        except PriceUnavailable:
            continue
        record["price"] = quote.value
        record["price_as_of"] = quote.as_of
        record["price_source"] = quote.source

    holdings: list[HoldingInput] = []
    for (security_id, member_id), record in grouped.items():
        security = record["security"]
        assert isinstance(security, Security)
        if record["price"] is None:
            quantity = record["quantity"]
            statement_value = record["statement_value"]
            statement_quantity = record["statement_quantity"]
            if (
                isinstance(quantity, Decimal)
                and quantity > 0
                and isinstance(statement_quantity, Decimal)
                and statement_quantity == quantity
                and isinstance(statement_value, Decimal)
                and statement_value > 0
                and isinstance(record["statement_as_of"], date)
            ):
                record["price"] = statement_value / quantity
                record["price_as_of"] = record["statement_as_of"]
                record["price_source"] = "broker_statement"
            else:
                manual = _manual_abroad_price(security, as_of)
                if manual is not None:
                    record["price"], record["price_as_of"], record["price_source"] = manual
        holdings.append(
            HoldingInput(
                security_id=str(security_id),
                symbol=security.nse_symbol or security.symbol or security.isin or "",
                market=security.market,
                currency=security.currency,
                domicile=security.domicile,
                security_type=security.security_type,
                sleeve=security.sleeve,
                category=security.category,
                quantity=record["quantity"],  # type: ignore[arg-type]
                price=record["price"],  # type: ignore[arg-type]
                statement_value=record["statement_value"],  # type: ignore[arg-type]
                lots=tuple(record["lots"]),  # type: ignore[arg-type]
                price_as_of=record["price_as_of"],  # type: ignore[arg-type]
                price_source=record["price_source"],  # type: ignore[arg-type]
                sector=security.category,
                member_id=str(member_id),
                joint=bool(record["joint"]),
                cost_value_inr=record["cost_value_inr"],  # type: ignore[arg-type]
                fallback_value_inr=record["fallback_value_inr"],  # type: ignore[arg-type]
            )
        )
    return tuple(holdings)


@lru_cache(maxsize=1)
def _manual_prices_document() -> dict[str, object]:
    document = yaml.safe_load(_MANUAL_PRICES_PATH.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


def _manual_abroad_price(security: Security, as_of: date) -> tuple[Decimal, date, str] | None:
    rows = _manual_prices_document().get("abroad_securities", {})
    if not isinstance(rows, dict):
        return None
    keys = [security.isin, f"{security.market}:{security.symbol}", security.symbol]
    for key in keys:
        row = rows.get(key) if isinstance(key, str) else None
        if not isinstance(row, dict) or row.get("source") != "manual":
            continue
        try:
            value = Decimal(str(row["value"]))
            price_date = date.fromisoformat(str(row["as_of"]))
        except (KeyError, ValueError):
            continue
        if value > 0 and price_date <= as_of and (as_of - price_date).days <= 60:
            return value, price_date, "manual"
    return None


def _previous_iso_week(week: str) -> str:
    year_text, week_text = week.split("-W")
    monday = date.fromisocalendar(int(year_text), int(week_text), 1)
    year, number, _weekday = (monday - timedelta(days=7)).isocalendar()
    return f"{year:04d}-W{number:02d}"


def research_pack_row(session: Session, week: str) -> ResearchPackRecord | None:
    """Return the research pack that serves `week`.

    The Saturday run stores its pack under that ISO week, and the pack keeps serving the
    following Monday to Friday until the next run, so fall back one week.
    """
    return session.get(ResearchPackRecord, week) or session.get(
        ResearchPackRecord, _previous_iso_week(week)
    )


def _research_pack(session: Session, week: str) -> ResearchPack | None:
    row = research_pack_row(session, week)
    if row is None:
        return None
    document = row.pack
    ideas_list: list[SatelliteIdea] = []
    for item in document.get("ideas", []):
        if not isinstance(item, dict):
            continue
        idea = satellite_idea_from_pack(item)
        if idea.market == "US":
            try:
                security = session.get(Security, UUID(idea.security_id))
            except ValueError:
                security = None
            if (
                security is not None
                and security.market == "US"
                and security.security_type == "equity"
            ):
                idea = idea.model_copy(
                    update={
                        "symbol": security.symbol or idea.symbol,
                        "name": security.name,
                        "exchange": security.exchange,
                        "isin": security.isin,
                        "currency": security.currency,
                    }
                )
        ideas_list.append(idea)
    ideas = tuple(ideas_list)
    raw_tilts = document.get("geo_tilts")
    tilts = GeoTilts.model_validate(raw_tilts) if raw_tilts is not None else None
    return ResearchPack(week=week, ideas=ideas, geo_tilts=tilts)


def _research_index_percentiles(session: Session, week: str) -> dict[str, Decimal]:
    """Read stored official index P/E percentiles for deterministic policy rules."""
    row = research_pack_row(session, week)
    if row is None or not isinstance(row.pack, dict):
        return {}
    raw_values = row.pack.get("index_valuation_percentiles")
    if not isinstance(raw_values, dict):
        return {}
    percentiles: dict[str, Decimal] = {}
    for market in ("IN", "US"):
        raw = raw_values.get(market)
        if raw is None:
            continue
        try:
            value = Decimal(str(raw))
        except (InvalidOperation, ValueError, TypeError):
            continue
        if value.is_finite() and Decimal(0) <= value <= Decimal(100):
            percentiles[market] = value
    return percentiles


def _member_profiles(
    session: Session,
    user_id: UUID,
    profile: Profile | None,
    financial_context: FinancialContext,
    fallback_investing: InvestingPrefs,
    as_of: date,
) -> tuple[MemberProfile, ...]:
    from wealth.db.repo import ensure_self_household_member

    ensure_self_household_member(session, user_id)
    rows = session.scalars(
        select(HouseholdMember)
        .where(HouseholdMember.user_id == user_id)
        .order_by(HouseholdMember.is_self.desc(), HouseholdMember.created_at, HouseholdMember.id)
    )
    members: list[MemberProfile] = []
    for row in rows:
        stored_tax = dict(row.tax or {})
        stored_investing = dict(row.investing or {})
        if row.is_self and profile is not None:
            stored_tax.setdefault("tax_regime", profile.tax_regime)
            stored_tax.setdefault("tax_slab_pct", profile.tax_slab_pct)
            stored_tax.setdefault("birth_year", fallback_investing.birth_year)
            stored_tax.setdefault("annual_income", financial_context.annual_income)
            stored_tax.setdefault("employment_type", financial_context.employment_type)
            if not stored_investing:
                stored_investing = fallback_investing.model_dump(mode="json")
        tax = MemberTax.model_validate(stored_tax)
        realised_ltcg, realised_stcg = _realised_equity_gains(session, user_id, row.id, as_of)
        tax = tax.model_copy(
            update={
                "fy_realised_equity_ltcg": max(tax.fy_realised_equity_ltcg, realised_ltcg),
                "fy_realised_equity_stcg": max(tax.fy_realised_equity_stcg, realised_stcg),
            }
        )
        ask_used_dkk, ask_fx_missing = _ask_deposits_dkk(session, user_id, row.id, as_of)
        has_prior_ask_activity = _has_prior_ask_activity(session, user_id, row.id, as_of)
        members.append(
            MemberProfile(
                id=str(row.id),
                name=row.display_name,
                relationship=row.relationship,  # type: ignore[arg-type]
                is_self=row.is_self,
                residency=residency_from_stored(row.residency),
                tax=tax,
                investing=InvestingPrefs.model_validate(stored_investing),
                realised_equity_ltcg=tax.fy_realised_equity_ltcg,
                realised_equity_stcg=tax.fy_realised_equity_stcg,
                ask_used_dkk=ask_used_dkk,
                ask_deposit_fx_missing=ask_fx_missing,
                has_prior_ask_activity=has_prior_ask_activity,
            )
        )
    return tuple(members)


def _ask_deposits_dkk(
    session: Session, user_id: UUID, member_id: UUID, as_of: date
) -> tuple[Decimal | None, bool]:
    """Return this year's imported contributions plus uncovered waiting purchases."""
    flows = session.execute(
        select(
            BrokerCashFlowRecord.amount,
            BrokerCashFlowRecord.currency,
            BrokerCashFlowRecord.transaction_date,
            BrokerCashFlowRecord.kind,
            BrokerCashFlowRecord.account_id,
        )
        .join(
            Account,
            and_(
                Account.id == BrokerCashFlowRecord.account_id,
                Account.user_id == BrokerCashFlowRecord.user_id,
                Account.member_id == BrokerCashFlowRecord.member_id,
                Account.tax_wrapper == "ask",
            ),
        )
        .where(
            BrokerCashFlowRecord.user_id == user_id,
            BrokerCashFlowRecord.member_id == member_id,
            BrokerCashFlowRecord.transaction_date >= date(as_of.year, 1, 1),
            BrokerCashFlowRecord.transaction_date <= as_of,
        )
        .order_by(BrokerCashFlowRecord.transaction_date, BrokerCashFlowRecord.id)
    ).all()
    buys = list(
        session.scalars(
            select(RecordedBuy).where(
                RecordedBuy.user_id == user_id,
                RecordedBuy.member_id == member_id,
                RecordedBuy.status == "waiting",
                RecordedBuy.account_type == "ask",
                RecordedBuy.bought_on >= date(as_of.year, 1, 1),
                RecordedBuy.bought_on <= as_of,
            )
        )
    )
    if not flows and not buys:
        return None, False
    deposits = Decimal(0)
    withdrawals = Decimal(0)
    for amount, currency, transaction_date, kind, _account_id in flows:
        if currency == "DKK":
            converted = amount
        else:
            currency_rate = store.latest_fx_rate(session, f"{currency}/INR", transaction_date)
            dkk_rate = store.latest_fx_rate(session, "DKK/INR", transaction_date)
            if (
                currency_rate is None
                or dkk_rate is None
                or currency_rate[0] <= 0
                or dkk_rate[0] <= 0
                or (transaction_date - currency_rate[1]).days > 10
                or (transaction_date - dkk_rate[1]).days > 10
            ):
                return None, True
            converted = amount * currency_rate[0] / dkk_rate[0]
        if kind == "deposit":
            deposits += converted
        else:
            withdrawals += converted
    used = deposits - withdrawals
    for buy in buys:
        cash_flow_covers_buy = any(
            transaction_date >= buy.bought_on
            and (buy.account_id is None or account_id == buy.account_id)
            for _amount, _currency, transaction_date, _kind, account_id in flows
        )
        if cash_flow_covers_buy:
            continue
        if buy.currency == "DKK":
            converted_buy = buy.amount
        else:
            currency_rate = store.latest_fx_rate(session, f"{buy.currency}/INR", buy.bought_on)
            dkk_rate = store.latest_fx_rate(session, "DKK/INR", buy.bought_on)
            if (
                currency_rate is None
                or dkk_rate is None
                or currency_rate[0] <= 0
                or dkk_rate[0] <= 0
                or (buy.bought_on - currency_rate[1]).days > 10
                or (buy.bought_on - dkk_rate[1]).days > 10
            ):
                return None, True
            converted_buy = buy.amount * currency_rate[0] / dkk_rate[0]
        used += converted_buy
    return used, False


def _has_prior_ask_activity(session: Session, user_id: UUID, member_id: UUID, as_of: date) -> bool:
    """Detect ASK holdings or imported ASK activity dated before this calendar year."""
    year_start = date(as_of.year, 1, 1)
    historical_lots = session.scalar(
        select(func.count(HoldingLot.id))
        .select_from(HoldingLot)
        .join(
            Account,
            and_(
                Account.id == HoldingLot.account_id,
                Account.user_id == HoldingLot.user_id,
                Account.member_id == HoldingLot.member_id,
                Account.tax_wrapper == "ask",
            ),
        )
        .where(
            HoldingLot.user_id == user_id,
            HoldingLot.member_id == member_id,
            or_(HoldingLot.acquired_on.is_(None), HoldingLot.acquired_on < year_start),
        )
    )
    if historical_lots:
        return True
    historical_flows = session.scalar(
        select(func.count(BrokerCashFlowRecord.id))
        .select_from(BrokerCashFlowRecord)
        .join(
            Account,
            and_(
                Account.id == BrokerCashFlowRecord.account_id,
                Account.user_id == BrokerCashFlowRecord.user_id,
                Account.member_id == BrokerCashFlowRecord.member_id,
                Account.tax_wrapper == "ask",
            ),
        )
        .where(
            BrokerCashFlowRecord.user_id == user_id,
            BrokerCashFlowRecord.member_id == member_id,
            BrokerCashFlowRecord.transaction_date < year_start,
        )
    )
    return bool(historical_flows)


def _realised_equity_gains(
    session: Session, user_id: UUID, member_id: UUID, as_of: date
) -> tuple[Decimal, Decimal]:
    rows = session.execute(
        select(RealisedTrade.holding_period, func.sum(RealisedTrade.gain))
        .where(
            RealisedTrade.user_id == user_id,
            RealisedTrade.member_id == member_id,
            RealisedTrade.trade_date >= fy_start(as_of),
            RealisedTrade.trade_date <= as_of,
        )
        .group_by(RealisedTrade.holding_period)
    )
    totals = {period: max(Decimal(0), amount or Decimal(0)) for period, amount in rows}
    return totals.get("LT", Decimal(0)), totals.get("ST", Decimal(0))


def required_fx_currencies(
    goals: tuple[Goal, ...], members: tuple[MemberProfile, ...]
) -> tuple[str, ...]:
    """List foreign currencies needed for goals, member income, and local rules."""
    currencies = {goal.currency for goal in goals if goal.currency != "INR"}
    currencies.add("USD")
    rules = load_residency_rules()
    country_rules = rules.get("country_rules", {})
    for member in members:
        income_currency = member.tax.annual_income_currency
        if income_currency != "INR":
            currencies.add(income_currency)
        country_rule = country_rules.get(member.residency.country, {})
        local_currency = country_rule.get("local_currency")
        if isinstance(local_currency, str) and local_currency != "INR":
            currencies.add(local_currency)
        if member.residency.country == "DK":
            local_currencies = rules["denmark"]["local_deposit"].get("currencies", [])
            currencies.update(
                currency
                for currency in local_currencies
                if isinstance(currency, str) and currency != "INR"
            )
    return tuple(sorted(currencies))


def load_policy_inputs(
    session: Session,
    user_id: UUID,
    week: str,
    as_of: date,
    *,
    deployment_mode: Literal["personal", "public"] = "personal",
) -> tuple[UserData, MarketContext, ResearchPack | None]:
    """Load a user's plan inputs using only their active RLS-scoped session."""
    profile = session.scalar(select(Profile).where(Profile.user_id == user_id))
    context = _financial_context(profile)
    investing = (
        InvestingPrefs.model_validate(profile.investing or {})
        if profile is not None
        else InvestingPrefs()
    )
    recorded_buys = tuple(
        session.scalars(
            select(RecordedBuy)
            .where(RecordedBuy.user_id == user_id)
            .order_by(RecordedBuy.created_at, RecordedBuy.id)
        )
    )
    members = _member_profiles(session, user_id, profile, context, investing, as_of)
    weekly_spending = _weekly_spending_for_week(session, user_id, week, recorded_buys)
    self_member = next(member for member in members if member.is_self)
    prices = DbPriceProvider(session)
    valued_assets, notes = _manual_assets(session, user_id, as_of, prices)
    holdings = _holdings(session, user_id, as_of, prices, recorded_buys)

    goals = _goals(session, user_id)
    fx_rates: dict[str, Decimal] = {}
    for currency in required_fx_currencies(goals, members):
        pair = f"{currency}/INR"
        fx_row = store.latest_fx_rate(session, pair, as_of)
        if fx_row is not None and (as_of - fx_row[1]).days <= 10:
            fx_rates[pair] = fx_row[0]
        else:
            notes.append(f"fx_{currency.lower()}_inr_missing_or_stale")
    try:
        gold = prices.gold_24k_per_gram(as_of).value
    except PriceUnavailable:
        gold = None
        notes.append("gold_price_missing_or_stale")

    yields, yield_as_of, yields_verified = load_debt_yields(as_of)
    if not yields_verified:
        notes.append(f"debt_yields_unverified_as_of:{yield_as_of.isoformat()}")
    user_data = UserData(
        profile=PolicyProfile(
            risk_profile=profile.risk_profile if profile is not None else None,
            tax_slab_pct=profile.tax_slab_pct if profile is not None else None,
            tax_regime=profile.tax_regime if profile is not None else None,
            city_tier=profile.city_tier if profile is not None else None,
            residency=self_member.residency,
        ),
        financial_context=context,
        goals=goals,
        valued_assets=valued_assets,
        holdings=holdings,
        investing=self_member.investing,
        realised_trades_this_fy=RealisedTradesThisFy(
            equity_ltcg=self_member.realised_equity_ltcg,
            equity_stcg=self_member.realised_equity_stcg,
        ),
        members=members,
        loan_candidates=_loan_candidates(session, user_id, as_of, profile),
        deployment_mode=deployment_mode,
        notes=tuple(notes),
        recorded_buys=_recorded_buy_inputs(recorded_buys),
        weekly_spending=weekly_spending,
    )
    market = MarketContext(
        fx_rates=fx_rates,
        debt_yields=yields,
        index_valuation_percentiles=_research_index_percentiles(session, week),
        gold_price_per_gram=gold,
        yield_table_as_of=yield_as_of,
        yield_table_verified=yields_verified,
        notes=(),
    )
    return user_data, market, _research_pack(session, week)
