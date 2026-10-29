"""Tenant-scoped loading of normalized inputs for the pure Personaliser."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import UUID

import yaml
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from wealth.core.dates import fy_start
from wealth.db.models import (
    Account,
    BrokerCashFlowRecord,
    HoldingLot,
    HouseholdMember,
    LoanEventRecord,
    ManualAsset,
    Profile,
    RealisedTrade,
    ResearchPackRecord,
    Security,
)
from wealth.db.models import (
    Goal as GoalRow,
)
from wealth.marketdata import store
from wealth.marketdata.provider import DbPriceProvider
from wealth.policy.actions import SatelliteIdea
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
    ResearchPack,
    UserData,
)
from wealth.policy.residency import load_residency_rules
from wealth.rates.reference import RateUnavailable
from wealth.schemas.assets import AnyAsset, LoanAsset, parse_asset
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
            rate = Decimal(1)
            if asset.currency != "INR":
                fx_row = store.latest_fx_rate(session, f"{asset.currency}/INR", as_of)
                if fx_row is None or (as_of - fx_row[1]).days > 10:
                    notes.append(f"loan_fx_missing:{asset.currency}")
                    continue
                rate = fx_row[0]
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
        fx_rate = Decimal(1)
        if asset.currency != "INR":
            fx_row = store.latest_fx_rate(session, f"{asset.currency}/INR", as_of)
            if fx_row is None or (as_of - fx_row[1]).days > 10:
                continue
            fx_rate = fx_row[0]
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
    session: Session, user_id: UUID, as_of: date, prices: DbPriceProvider
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
        if record["price"] is None:
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


def _research_pack(session: Session, week: str) -> ResearchPack | None:
    row = session.get(ResearchPackRecord, week)
    if row is None:
        return None
    document = row.pack
    ideas_list: list[SatelliteIdea] = []
    for item in document.get("ideas", []):
        idea = SatelliteIdea.model_validate(item)
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
    """Return this calendar year's net ASK deposits, converted on each transaction date."""
    flows = session.execute(
        select(
            BrokerCashFlowRecord.amount,
            BrokerCashFlowRecord.currency,
            BrokerCashFlowRecord.transaction_date,
            BrokerCashFlowRecord.kind,
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
    if not flows:
        return None, False
    deposits = Decimal(0)
    withdrawals = Decimal(0)
    for amount, currency, transaction_date, kind in flows:
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
    return deposits - withdrawals, False


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
    members = _member_profiles(session, user_id, profile, context, investing, as_of)
    self_member = next(member for member in members if member.is_self)
    prices = DbPriceProvider(session)
    valued_assets, notes = _manual_assets(session, user_id, as_of, prices)
    holdings = _holdings(session, user_id, as_of, prices)

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
    )
    market = MarketContext(
        fx_rates=fx_rates,
        debt_yields=yields,
        index_valuation_percentiles={},
        gold_price_per_gram=gold,
        yield_table_as_of=yield_as_of,
        yield_table_verified=yields_verified,
        notes=(),
    )
    return user_data, market, _research_pack(session, week)
