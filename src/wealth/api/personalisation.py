"""Personalisation preferences, weekly plan, actions, and stress endpoints."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable, Mapping
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path as FilePath
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import yaml
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from wealth.core.money import round_paise
from wealth.db.models import Action as ActionRow
from wealth.db.models import (
    AuditLog,
    HighlightDismissal,
    HouseholdMember,
    Profile,
    RecordedBuy,
    ResearchPackRecord,
    Security,
)
from wealth.marketdata import store
from wealth.policy.actions import (
    SatelliteIdea,
    india_stock_room,
    load_buying_rules,
    satellite_idea_from_pack,
)
from wealth.policy.data import load_policy_inputs, research_pack_row
from wealth.policy.denmark import dkk_from_inr
from wealth.policy.denmark_picks import global_fund_ideas, load_eu_etfs, us_stock_room
from wealth.policy.highlights import (
    Highlight,
    HighlightContext,
    HighlightScreen,
    highlight_fingerprint,
    highlights,
)
from wealth.policy.household import MemberContext, build_household_snapshot, build_member_contexts
from wealth.policy.lookthrough import (
    load_lookthrough_rules,
    same_company,
)
from wealth.policy.market_timing import map_us_market_timing
from wealth.policy.personalise import (
    MarketContext,
    UserData,
    WeeklyPlan,
    _holding_value,
    apply_weekly_spending_to_plan,
    personalise,
)
from wealth.policy.store import actions_for_week, latest_plan, save_weekly_plan
from wealth.schemas.common import LiquidityClass, Rupees, Sleeve
from wealth.schemas.profile import InvestingPrefs, Residency, ResidencyInput, residency_from_stored

_LOG = logging.getLogger(__name__)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WeeklyPlanResponse(WeeklyPlan):
    fresh: bool
    me_member_id: UUID
    highlights_by_screen: dict[HighlightScreen, list[Highlight]] = Field(default_factory=dict)


class HighlightDismissalResponse(ApiModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, from_attributes=True)

    key: str
    fingerprint: str
    dismissed_at: datetime


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
    has_recorded_buys: bool = False


class ActionStatusUpdate(ApiModel):
    status: Literal["done", "dismissed"]


class StressResponse(ApiModel):
    week: str
    as_of: date
    stress: dict[str, Any]


class EntryHintSummary(ApiModel):
    code: Literal["buy_in_parts", "wait"]
    parts: int | None
    reasons: list[str]


class ResearchIdeaWhy(ApiModel):
    reasons: list[str] = Field(default_factory=list)
    risk: str


class IdeaOwnershipFund(ApiModel):
    name: str
    amount_inr: str
    amount: str
    estimated: bool = False


class IdeaOwnership(ApiModel):
    direct_inr: str
    through_funds_inr: str
    direct: str
    through_funds: str
    currency: Literal["INR", "DKK"]
    status: Literal["none", "some", "enough"]
    funds: list[IdeaOwnershipFund] = Field(default_factory=list)


class ResearchIdeaSummary(ApiModel):
    security_id: str
    symbol: str
    name: str | None = None
    market: Literal["IN", "US"]
    sector: str
    price: Rupees
    quant_decile: int
    dossier_status: str
    valuation_percentile: str
    valuation_label: Literal["cheap", "fair", "expensive"]
    confidence: str
    entry_hint: EntryHintSummary | None = None
    why: ResearchIdeaWhy | None = None
    owned: IdeaOwnership | None = None


class ResearchIndexFundInstrument(ApiModel):
    id: str
    name: str
    isin: str
    listed_name: str
    accumulating: bool
    currency: str
    brokers: list[str]
    ask_eligible: bool


class ResearchIndexFund(ApiModel):
    exposure: Literal["global_all_world", "global_sp500", "global_nasdaq100"]
    role: Literal["plan_main", "option"]
    instrument: ResearchIndexFundInstrument
    market: Literal["US"] | None


class UsMarketSummary(ApiModel):
    percentile: Decimal | None
    history_label: Literal["long_history", "short_history"]
    label: Literal["cheap", "fair", "expensive"] | None
    extra_money_parts: int


class ResearchIdeasResponse(ApiModel):
    week: str
    ideas: list[ResearchIdeaSummary]
    core_picks: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    track_record: dict[str, Any] | None = None
    amc_international_suspended: bool | None = None
    index_funds: list[ResearchIndexFund] = Field(default_factory=list)
    us_market: UsMarketSummary | None = None


_RESEARCH_RULES_PATH = FilePath(__file__).resolve().parents[3] / "rules" / "research.yaml"


@lru_cache(maxsize=1)
def _valuation_percentile_thresholds() -> tuple[Decimal, Decimal]:
    raw = yaml.safe_load(_RESEARCH_RULES_PATH.read_text(encoding="utf-8"))
    valuation = raw["valuation"]
    return (
        Decimal(str(valuation["cheap_pe_percentile_max"])),
        Decimal(str(valuation["expensive_pe_percentile_min"])),
    )


def valuation_label(percentile: Decimal) -> Literal["cheap", "fair", "expensive"]:
    cheap_max, expensive_min = _valuation_percentile_thresholds()
    from wealth.policy.market_timing import valuation_label as classify_valuation

    return classify_valuation(
        percentile,
        cheap_pe_percentile_max=cheap_max,
        expensive_pe_percentile_min=expensive_min,
    )


def _denmark_fund_member(session: Any, household_owner_id: UUID) -> tuple[str, str] | None:
    profile = session.scalar(select(Profile).where(Profile.user_id == household_owner_id))
    members = list(
        session.scalars(
            select(HouseholdMember)
            .where(HouseholdMember.user_id == household_owner_id)
            .order_by(
                HouseholdMember.is_self.desc(),
                HouseholdMember.created_at,
                HouseholdMember.id,
            )
        ).all()
    )
    for member in members:
        residency = dict(member.residency or {})
        if residency.get("country") != "DK":
            continue
        investing = dict(member.investing or {})
        if member.is_self and profile is not None and not investing:
            investing = dict(profile.investing or {})
        return "DK", InvestingPrefs.model_validate(investing).broker
    if not members and profile is not None:
        residency = dict(profile.residency or {})
        if residency.get("country") == "DK":
            return "DK", InvestingPrefs.model_validate(profile.investing or {}).broker
    return None


def _index_funds_for_household(session: Any, household_owner_id: UUID) -> list[ResearchIndexFund]:
    member = _denmark_fund_member(session, household_owner_id)
    if member is None:
        return []
    country, broker = member
    return [
        ResearchIndexFund(
            exposure=idea.exposure,
            role=idea.role,
            instrument=ResearchIndexFundInstrument(
                id=idea.id,
                name=idea.name,
                isin=idea.isin,
                listed_name=idea.listed_name,
                accumulating=idea.accumulating,
                currency=idea.currency,
                brokers=list(idea.brokers),
                ask_eligible=idea.ask_eligible,
            ),
            market=idea.market,
        )
        for idea in global_fund_ideas(country, broker)
    ]


def _us_market_summary(
    valuation: store.IndexValuationPercentile | None,
) -> UsMarketSummary | None:
    if valuation is None:
        return None
    cheap_max, expensive_min = _valuation_percentile_thresholds()
    lumpsum = load_buying_rules()["lumpsum"]
    timing = map_us_market_timing(
        valuation.percentile,
        valuation.history_label,
        cheap_pe_percentile_max=cheap_max,
        expensive_pe_percentile_min=expensive_min,
        expensive_valuation_percentile=Decimal(str(lumpsum["expensive_valuation_percentile"])),
        stp_tranches_normal=int(lumpsum["stp_tranches_normal"]),
        stp_tranches_expensive=int(lumpsum["stp_tranches_expensive"]),
    )
    return UsMarketSummary(
        percentile=timing.percentile,
        history_label=timing.history_label,  # type: ignore[arg-type]
        label=timing.label,
        extra_money_parts=timing.extra_money_parts,
    )


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


def _member_contexts_for_ownership(
    user_data: UserData,
    market: MarketContext,
    as_of: date,
) -> dict[str, MemberContext]:
    self_member = next(member for member in user_data.members if member.is_self)
    market_holdings: list[tuple[str, Decimal, LiquidityClass]] = []
    member_market_holdings: list[tuple[str, str, Decimal, LiquidityClass]] = []
    for holding in user_data.holdings:
        value = _holding_value(holding, market.fx_rates)
        member_id = holding.member_id or self_member.id
        if value is None or holding.sleeve is None:
            continue
        liquidity = (
            LiquidityClass.WITHIN_1Y
            if holding.security_type in {"bond", "sgb"}
            else LiquidityClass.INSTANT
        )
        market_holdings.append((holding.sleeve, value, liquidity))
        member_market_holdings.append((member_id, holding.sleeve, value, liquidity))
    for buy in user_data.recorded_buys:
        if (
            buy.status != "waiting"
            or buy.security_id is not None
            or buy.amount_inr is None
            or buy.amount_inr <= 0
        ):
            continue
        try:
            Sleeve(buy.sleeve)
        except ValueError:
            continue
        market_holdings.append((buy.sleeve, buy.amount_inr, LiquidityClass.INSTANT))
        member_market_holdings.append(
            (buy.member_id, buy.sleeve, buy.amount_inr, LiquidityClass.INSTANT)
        )
    snapshot = build_household_snapshot(
        user_data.valued_assets,
        market_holdings,
        member_market_holdings,
        user_data.members,
        self_member_id=self_member.id,
    )
    has_indian_funds_by_member = {
        member.id: any(
            (holding.member_id or self_member.id) == member.id
            and holding.market == "IN"
            and holding.security_type in {"mf", "etf"}
            for holding in user_data.holdings
        )
        for member in user_data.members
    }
    contexts = build_member_contexts(
        user_data.members,
        snapshot,
        user_data.valued_assets,
        as_of_year=as_of.year,
        fx_rates=market.fx_rates,
        fallback_tax_slab_pct=user_data.profile.tax_slab_pct,
        fallback_tax_regime=user_data.profile.tax_regime,
        fallback_city_tier=user_data.profile.city_tier,
        has_indian_funds_by_member=has_indian_funds_by_member,
    )
    return {context.member_id: context for context in contexts}


def _idea_ownership(
    idea: SatelliteIdea,
    user_data: UserData,
    market: MarketContext,
    plan: WeeklyPlan,
    member_contexts: Mapping[str, MemberContext],
) -> IdeaOwnership:
    direct_by_member: dict[str, Decimal] = {}
    direct_total = Decimal(0)
    self_member_id = next(member.id for member in user_data.members if member.is_self)
    rules = load_lookthrough_rules()
    for holding in user_data.holdings:
        if holding.security_type != "equity" or holding.market != idea.market:
            continue
        is_same_company = same_company(
            holding.market,
            holding.isin,
            holding.symbol,
            idea.market,
            idea.isin,
            idea.symbol,
            user_data.lookthrough_company_keys,
            rules=rules,
        )
        if holding.security_id != idea.security_id and (not is_same_company):
            continue
        value = _holding_value(holding, market.fx_rates) or Decimal(0)
        direct_total += value
        member_id = holding.member_id or self_member_id
        direct_by_member[member_id] = direct_by_member.get(member_id, Decimal(0)) + value

    through_total = user_data.through_fund_inr.get(idea.security_id, Decimal(0))
    owned_total = direct_total + through_total
    enough = False
    inr_per_dkk = market.fx_rates.get("DKK/INR")
    if idea.market == "IN":
        buying_rules = load_buying_rules()
        satellite = buying_rules["satellite"]
        india_target = Decimal(plan.allocation.get("equity_india", {}).get("target", "0"))
        global_target = Decimal(plan.allocation.get("equity_global", {}).get("target", "0"))
        total_equity = india_target + global_target
        satellite_target = india_target * user_data.investing.satellite_pct / Decimal(100)
        held_satellite = Decimal(0)
        sector_value = Decimal(0)
        for holding in user_data.holdings:
            if holding.market != "IN" or holding.security_type != "equity":
                continue
            value = _holding_value(holding, market.fx_rates) or Decimal(0)
            held_satellite += value
            if (holding.sector or "unknown") == idea.sector:
                sector_value += value
        stock_cap = (
            total_equity * Decimal(str(satellite["max_stock_pct_of_total_equity"])) / Decimal(100)
        )
        sector_cap = (
            satellite_target * Decimal(str(satellite["max_sector_pct_of_satellite"])) / Decimal(100)
        )
        room = india_stock_room(
            max_stock=stock_cap,
            direct_value=direct_total,
            through_funds_value=through_total,
            sector_room=sector_cap - sector_value,
            satellite_gap=satellite_target - held_satellite,
            price=idea.price,
            min_action=Decimal(str(buying_rules["min_action_inr"])),
        )
        enough = room == 0
    else:
        settings = load_eu_etfs().get("selection", {}).get("stock_picks", {})
        minimum_buy = Decimal(str(settings.get("minimum_buy_dkk", "2000")))
        maximum_pct = Decimal(str(settings.get("per_stock_max_pct_of_investable_wealth", "5")))
        eligible = [
            context
            for context in member_contexts.values()
            if context.residency.country == "DK" and context.member.investing.us_stock_picks_opt_in
        ]
        has_room = False
        if inr_per_dkk is not None and inr_per_dkk > 0:
            for context in eligible:
                held_inr = direct_by_member.get(context.member_id, Decimal(0))
                held_inr += user_data.through_fund_by_member_security.get(
                    (context.member_id, idea.security_id), Decimal(0)
                )
                held_dkk = dkk_from_inr(held_inr, inr_per_dkk)
                cap_dkk = (
                    context.snapshot.investable_total / inr_per_dkk * maximum_pct / Decimal(100)
                )
                if (
                    us_stock_room(
                        per_stock_cap=cap_dkk,
                        held_dkk=held_dkk,
                        available_dkk=cap_dkk,
                        minimum_buy_dkk=minimum_buy,
                    )
                    > 0
                ):
                    has_room = True
                    break
        enough = bool(eligible) and not has_room

    status: Literal["none", "some", "enough"] = (
        "none" if owned_total <= 0 else "enough" if enough else "some"
    )
    eligible_denmark_member = any(
        context.residency.country == "DK" and context.member.investing.us_stock_picks_opt_in
        for context in member_contexts.values()
    )
    ownership_currency: Literal["INR", "DKK"] = (
        "DKK"
        if idea.market == "US"
        and eligible_denmark_member
        and inr_per_dkk is not None
        and inr_per_dkk > 0
        else "INR"
    )
    direct_in_currency = (
        dkk_from_inr(direct_total, inr_per_dkk)
        if ownership_currency == "DKK" and inr_per_dkk is not None
        else direct_total
    )
    through_in_currency = (
        dkk_from_inr(through_total, inr_per_dkk)
        if ownership_currency == "DKK" and inr_per_dkk is not None
        else through_total
    )
    funds = []
    for part in user_data.through_fund_parts.get(idea.security_id, ())[:3]:
        amount = (
            dkk_from_inr(part.amount_inr, inr_per_dkk)
            if ownership_currency == "DKK" and inr_per_dkk is not None
            else part.amount_inr
        )
        funds.append(
            IdeaOwnershipFund(
                name=part.name,
                amount_inr=format(round_paise(part.amount_inr), ".2f"),
                amount=format(round_paise(amount), ".2f"),
                estimated=part.proxy,
            )
        )
    return IdeaOwnership(
        direct_inr=format(round_paise(direct_total), ".2f"),
        through_funds_inr=format(round_paise(through_total), ".2f"),
        direct=format(round_paise(direct_in_currency), ".2f"),
        through_funds=format(round_paise(through_in_currency), ".2f"),
        currency=ownership_currency,
        status=status,
        funds=funds,
    )


def _error_location(error: BaseException) -> str:
    """Name an error's type and code lines only: no message, so no values leave the server."""
    frames = traceback.extract_tb(error.__traceback__)
    ours = [frame for frame in frames if "/wealth/" in frame.filename.replace("\\", "/")]
    parts = [type(error).__name__]
    for frame in (ours[-1] if ours else None, frames[-1] if frames else None):
        if frame is not None:
            parts.append(f"{FilePath(frame.filename).name}:{frame.lineno}")
    if isinstance(error, ValidationError):
        # Field names and error types only; keys such as ids become "*".
        for item in error.errors(include_url=False, include_input=False)[:2]:
            loc = ".".join(
                str(part) if isinstance(part, str) and part.isidentifier() else "*"
                for part in item["loc"]
            )
            parts.append(f"{loc}={item['type']}")
    return "@".join(dict.fromkeys(parts))


def _safe_idea_ownership(
    idea: SatelliteIdea,
    inputs: tuple[UserData, MarketContext, WeeklyPlan, Any] | None,
    errors: list[str] | None = None,
) -> IdeaOwnership | None:
    if inputs is None:
        return None
    user_data, market, plan, member_contexts = inputs
    try:
        return _idea_ownership(idea, user_data, market, plan, member_contexts)
    except Exception as error:
        _LOG.exception("research_idea_ownership_failed")
        if errors is not None:
            errors.append(f"idea:{_error_location(error)}")
        return None


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
    computed = personalise(user_data, market, research, week, today, apply_weekly_spending=False)
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
    computed = apply_weekly_spending_to_plan(computed, user_data, market.fx_rates)
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
        plan = _compute_if_stale(user, today)
        saved_dismissals = user.session.scalars(
            select(HighlightDismissal).where(
                HighlightDismissal.login_user_id == user.user_id,
                HighlightDismissal.household_owner_id == user.household_owner_id,
            )
        ).all()
        dismissed_fingerprints = {row.highlight_key: row.fingerprint for row in saved_dismissals}
        from wealth.alerts.copy import load_copy

        copy = load_copy()
        plan_data = WeeklyPlan.model_validate(
            plan.model_dump(mode="python", include=set(WeeklyPlan.model_fields))
        )
        per_screen = {
            screen: highlights(
                plan_data,
                HighlightContext(
                    screen=screen,
                    copy=copy,
                    dismissed_fingerprints=dismissed_fingerprints,
                ),
            )
            for screen in ("home", "plan", "ideas", "money")
        }
        return plan.model_copy(update={"highlights_by_screen": per_screen})

    @router.get("/v1/highlights/dismissals", response_model=list[HighlightDismissalResponse])
    def get_highlight_dismissals(user: Any = user_dependency) -> list[HighlightDismissal]:
        rows = list(
            user.session.scalars(
                select(HighlightDismissal)
                .where(
                    HighlightDismissal.login_user_id == user.user_id,
                    HighlightDismissal.household_owner_id == user.household_owner_id,
                )
                .order_by(HighlightDismissal.highlight_key)
            ).all()
        )
        return [
            HighlightDismissalResponse(
                key=row.highlight_key,
                fingerprint=row.fingerprint,
                dismissed_at=row.dismissed_at,
            )
            for row in rows
        ]

    @router.post("/v1/highlights/{key}/dismiss", status_code=204, response_class=Response)
    def dismiss_highlight(
        key: Annotated[str, Path(pattern=r"^[a-z0-9_]{1,100}$")],
        user: Any = user_dependency,
        today: date = today_dependency,
    ) -> Response:
        plan = _compute_if_stale(user, today)
        from wealth.alerts.copy import load_copy

        fingerprint = highlight_fingerprint(plan, key, load_copy())
        if fingerprint is None:
            raise HTTPException(status_code=404, detail="The highlight was not found.")
        statement = pg_insert(HighlightDismissal).values(
            login_user_id=user.user_id,
            household_owner_id=user.household_owner_id,
            highlight_key=key,
            fingerprint=fingerprint,
        )
        user.session.execute(
            statement.on_conflict_do_update(
                index_elements=[
                    HighlightDismissal.login_user_id,
                    HighlightDismissal.household_owner_id,
                    HighlightDismissal.highlight_key,
                ],
                set_={"fingerprint": statement.excluded.fingerprint, "dismissed_at": text("now()")},
            )
        )
        return Response(status_code=204)

    @router.get("/v1/actions", response_model=list[ActionResponse])
    def get_actions(
        user: Any = user_dependency,
        today: date = today_dependency,
        week: Annotated[str | None, Query(pattern=r"^\d{4}-W\d{2}$")] = None,
    ) -> list[ActionResponse]:
        selected_week = week or current_week(today)
        if not _valid_week(selected_week):
            raise HTTPException(status_code=422, detail="The requested ISO week is invalid.")
        actions = actions_for_week(user.session, user.household_owner_id, selected_week)
        action_ids = [row.id for row in actions]
        actions_with_buys = (
            set(
                user.session.scalars(
                    select(RecordedBuy.action_id).where(
                        RecordedBuy.user_id == user.household_owner_id,
                        RecordedBuy.action_id.in_(action_ids),
                        RecordedBuy.status.in_(("waiting", "matched")),
                    )
                )
            )
            if action_ids
            else set()
        )
        return [
            ActionResponse.model_validate(action).model_copy(
                update={"has_recorded_buys": action.id in actions_with_buys}
            )
            for action in actions
        ]

    @router.patch("/v1/actions/{action_id}", response_model=ActionResponse)
    def patch_action(
        action_id: UUID,
        payload: ActionStatusUpdate,
        user: Any = user_dependency,
    ) -> ActionResponse:
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
        has_recorded_buys = (
            user.session.scalar(
                select(RecordedBuy.id)
                .where(
                    RecordedBuy.user_id == user.household_owner_id,
                    RecordedBuy.action_id == action.id,
                    RecordedBuy.status.in_(("waiting", "matched")),
                )
                .limit(1)
            )
            is not None
        )
        return ActionResponse.model_validate(action).model_copy(
            update={"has_recorded_buys": has_recorded_buys}
        )

    @router.get("/v1/stress", response_model=StressResponse)
    def get_stress(user: Any = user_dependency, today: date = today_dependency) -> StressResponse:
        plan = _compute_if_stale(user, today)
        return StressResponse(week=plan.week, as_of=plan.as_of, stress=plan.stress)

    @router.get("/v1/research/ideas", response_model=ResearchIdeasResponse)
    def get_research_ideas(
        response: Response,
        user: Any = user_dependency,
        today: date = today_dependency,
        week: Annotated[str | None, Query(pattern=r"^\d{4}-W\d{2}$")] = None,
    ) -> ResearchIdeasResponse:
        selected_week = week or current_week(today)
        if not _valid_week(selected_week):
            raise HTTPException(status_code=422, detail="The requested ISO week is invalid.")
        index_funds = _index_funds_for_household(user.session, user.household_owner_id)
        us_market = _us_market_summary(
            store.latest_index_valuation_percentile(user.session, "S&P 500", today)
        )
        record = (
            user.session.get(ResearchPackRecord, selected_week)
            if week is not None
            else research_pack_row(user.session, selected_week)
        )
        if record is None:
            return ResearchIdeasResponse(
                week=selected_week,
                ideas=[],
                index_funds=index_funds,
                us_market=us_market,
            )
        # "Already own" lines are extra: if they fail, Ideas still lists the ideas.
        ownership_inputs: tuple[UserData, MarketContext, WeeklyPlan, Any] | None
        ownership_errors: list[str] = []
        try:
            # A savepoint keeps the request's transaction usable if a query fails here.
            with user.session.begin_nested():
                user_data, market, research = load_policy_inputs(
                    user.session,
                    user.household_owner_id,
                    selected_week,
                    today,
                    deployment_mode=getattr(user, "deployment_mode", "personal"),
                )
                current_plan = personalise(
                    user_data,
                    market,
                    research,
                    selected_week,
                    today,
                    apply_weekly_spending=False,
                )
                member_contexts = _member_contexts_for_ownership(user_data, market, today)
            ownership_inputs = (user_data, market, current_plan, member_contexts)
        except Exception as error:
            _LOG.exception("research_ideas_ownership_inputs_failed")
            ownership_errors.append(f"inputs:{_error_location(error)}")
            ownership_inputs = None
        prepared_ideas: list[tuple[SatelliteIdea, Any, Any, UUID | None]] = []
        for raw in record.pack.get("ideas", []):
            if not isinstance(raw, dict):
                continue
            raw_hint = raw.get("entry_hint")
            idea = satellite_idea_from_pack(raw)
            try:
                security_id = UUID(idea.security_id)
            except ValueError:
                security_id = None
            prepared_ideas.append((idea, raw_hint, raw.get("why"), security_id))
        security_ids = {security_id for _, _, _, security_id in prepared_ideas if security_id}
        names_by_id = (
            {
                str(security_id): name
                for security_id, name in user.session.execute(
                    select(Security.id, Security.name).where(Security.id.in_(security_ids))
                ).all()
            }
            if security_ids
            else {}
        )
        ideas: list[ResearchIdeaSummary] = []
        for idea, raw_hint, raw_why, security_id in prepared_ideas:
            ideas.append(
                ResearchIdeaSummary(
                    security_id=idea.security_id,
                    symbol=idea.symbol,
                    name=names_by_id.get(str(security_id)) if security_id else None,
                    market=idea.market,
                    sector=idea.sector,
                    price=idea.price,
                    quant_decile=idea.quant_decile,
                    dossier_status=idea.dossier_status,
                    valuation_percentile=format(idea.valuation_percentile, "f"),
                    valuation_label=valuation_label(idea.valuation_percentile),
                    confidence=idea.confidence,
                    entry_hint=(
                        EntryHintSummary.model_validate(raw_hint)
                        if isinstance(raw_hint, dict)
                        else None
                    ),
                    why=(
                        ResearchIdeaWhy.model_validate(raw_why)
                        if isinstance(raw_why, dict)
                        else None
                    ),
                    owned=_safe_idea_ownership(idea, ownership_inputs, ownership_errors),
                )
            )
        raw_picks = record.pack.get("core_picks", {})
        core_picks = (
            {
                str(category): [item for item in items if isinstance(item, dict)]
                for category, items in raw_picks.items()
                if isinstance(category, str) and isinstance(items, list)
            }
            if isinstance(raw_picks, dict)
            else {}
        )
        raw_track_record = record.pack.get("track_record")
        track_record = raw_track_record if isinstance(raw_track_record, dict) else None
        suspended = record.pack.get("amc_international_suspended")
        if ownership_errors:
            # Temporary: where the "already own" lines fail in production (types and lines only).
            first_errors = dict.fromkeys(ownership_errors[:3])
            response.headers["X-Ownership-Error"] = ";".join(first_errors)[:300]
        return ResearchIdeasResponse(
            week=selected_week,
            ideas=ideas,
            core_picks=core_picks,
            track_record=track_record,
            amc_international_suspended=suspended if isinstance(suspended, bool) else None,
            index_funds=index_funds,
            us_market=us_market,
        )

    return router
