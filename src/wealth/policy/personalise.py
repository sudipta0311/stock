"""Pure weekly plan orchestration for a single user's Money Map."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wealth.core.money import round_paise
from wealth.policy.actions import (
    Action,
    ActionKind,
    ActionList,
    SatelliteHolding,
    SatelliteIdea,
    Sip,
    WeekInput,
    build_action_list,
)
from wealth.policy.allocation import AllocationPlan, build_targets
from wealth.policy.debt_answer import debt_answer
from wealth.policy.debt_picker import (
    DebtNeed,
    DebtPick,
    UnrankedDebtOption,
    pick_debt_with_unranked,
)
from wealth.policy.denmark import account_lines, dkk_from_inr, inr_from_dkk, round_dkk_for_display
from wealth.policy.denmark_picks import (
    approved_us_stock_buys,
    load_eu_etfs,
    pick_india_vehicle,
    plain_fund_name,
    ranked_etfs,
)
from wealth.policy.exits import ExitCard, HoldingFacts, build_exit_card, load_exit_rules
from wealth.policy.foundation import FoundationItem, FoundationReport, foundation_check
from wealth.policy.global_split import GeoTilts, load_global_rules, split_equity
from wealth.policy.household import (
    MemberContext,
    MemberProfile,
    build_household_snapshot,
    build_member_contexts,
)
from wealth.policy.insurance_review import review_policy
from wealth.policy.loans import (
    LoanPlanContext,
    load_loan_policy_rules,
    load_tax_relief_rules,
    pay_extra_or_invest,
)
from wealth.policy.placement import place_actions
from wealth.policy.rebalance import band_breaches
from wealth.policy.residency import (
    deployment_disclaimer,
    india_core_vehicle,
    load_residency_rules,
    residency_note_codes,
)
from wealth.policy.routes import RouteContext, RouteDecision, select_route
from wealth.policy.rules import decimal_rule, resolve_rules
from wealth.policy.snapshot import WealthSnapshot
from wealth.policy.stress import load_stress_rules, stress_compare, stress_test
from wealth.policy.tax import Lot, load_tax_rules, ltcg_harvest
from wealth.policy.us_situs import us_situs_value_usd
from wealth.schemas.assets import AnyAsset, TraditionalPolicyAsset
from wealth.schemas.common import LiquidityClass, Sleeve
from wealth.schemas.profile import FinancialContext, Goal, InvestingPrefs, MemberTax, Residency
from wealth.valuation.base import Valuation


@dataclass(frozen=True)
class PolicyProfile:
    """Stored policy inputs needed from the user's profile row."""

    risk_profile: str | None = None
    tax_slab_pct: Decimal | None = None
    tax_regime: str | None = None
    city_tier: int | None = None
    residency: Residency = field(default_factory=Residency)


@dataclass(frozen=True)
class RealisedTradesThisFy:
    """Realised listed-equity gains already booked in the current Indian FY."""

    equity_ltcg: Decimal = Decimal("0")
    equity_stcg: Decimal = Decimal("0")


@dataclass(frozen=True)
class HoldingLotInput:
    """One stored holding lot before price and tax-class normalization."""

    lot_id: str
    quantity: Decimal
    cost_per_unit: Decimal | None
    acquired_on: date | None
    fmv_2018_01_31: Decimal | None = None
    member_id: str | None = None
    currency: str | None = None
    cost_per_unit_inr: Decimal | None = None
    fx_missing: bool = False


@dataclass(frozen=True)
class HoldingInput:
    """A security position and its source lots, with optional point-in-time pricing."""

    security_id: str
    symbol: str
    market: str
    currency: str
    security_type: str
    sleeve: str | None
    category: str | None
    quantity: Decimal
    price: Decimal | None
    domicile: str | None = None
    statement_value: Decimal | None = None
    lots: tuple[HoldingLotInput, ...] = ()
    price_as_of: date | None = None
    price_source: str | None = None
    sector: str | None = None
    member_id: str | None = None
    joint: bool = False


@dataclass(frozen=True)
class UserData:
    """Normalized, already loaded user information consumed by the pure engine."""

    profile: PolicyProfile
    financial_context: FinancialContext
    goals: tuple[Goal, ...]
    valued_assets: tuple[tuple[AnyAsset, Valuation], ...]
    holdings: tuple[HoldingInput, ...]
    investing: InvestingPrefs
    realised_trades_this_fy: RealisedTradesThisFy = RealisedTradesThisFy()
    notes: tuple[str, ...] = ()
    members: tuple[MemberProfile, ...] = ()
    loan_candidates: tuple[Any, ...] = ()
    deployment_mode: Literal["personal", "public"] = "personal"


@dataclass(frozen=True)
class MarketContext:
    """Point-in-time prices, FX, debt yields, and market-rule inputs."""

    fx_rates: Mapping[str, Decimal]
    debt_yields: Mapping[str, Decimal]
    index_valuation_percentiles: Mapping[str, Decimal]
    gold_price_per_gram: Decimal | None = None
    yield_table_as_of: date | None = None
    yield_table_verified: bool = False
    amc_international_suspended: bool = False
    etf_inav_premium_pct: Decimal | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchPack:
    """Plain research output accepted as bounded policy input; it contains no code."""

    week: str
    ideas: tuple[SatelliteIdea, ...] = ()
    geo_tilts: GeoTilts | None = None


class PlanMember(BaseModel):
    """Typed household-member details exposed to the PWA and API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    relationship: Literal["self", "spouse", "child", "parent", "other"]
    net_worth: str = "0.00"
    sleeves: dict[str, str] = Field(default_factory=dict)
    liquidity: dict[str, str] = Field(default_factory=dict)
    locked: dict[str, str] = Field(default_factory=dict)
    residency: Residency = Field(default_factory=Residency)
    eligibility: dict[str, bool] = Field(default_factory=dict)
    eligibility_status: dict[str, Literal["allowed", "conditional", "not_allowed"]] = Field(
        default_factory=dict
    )
    eligibility_conditions: dict[str, str] = Field(default_factory=dict)
    limits_room: dict[str, str | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    ltcg_harvest: dict[str, Any] = Field(default_factory=dict)
    foundation: dict[str, Any] = Field(default_factory=dict)


class WeeklyPlan(BaseModel):
    """Stable JSON-facing weekly plan built entirely by deterministic policy code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    week: str
    as_of: date
    plan_health: int = Field(default=100, ge=0, le=100)
    carry_forward: str = "0.00"
    debt_picks: list[dict[str, Any]] = Field(default_factory=list)
    unranked_options: list[dict[str, Any]] = Field(default_factory=list)
    insurance_reviews: list[dict[str, Any]] = Field(default_factory=list)
    allocation: dict[str, dict[str, str]]
    band_breaches: list[dict[str, str]]
    debt_answer: dict[str, Any]
    foundation: list[dict[str, Any]]
    route: dict[str, Any]
    actions: list[dict[str, Any]]
    abroad_buy_list: list[dict[str, Any]] = Field(default_factory=list)
    exit_cards: list[dict[str, Any]]
    stress: dict[str, Any]
    notes: list[str]
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    residency: Residency = Field(default_factory=Residency)
    residency_notes: list[str] = Field(default_factory=list)
    members: list[PlanMember] = Field(default_factory=list)
    placement_notes: list[str] = Field(default_factory=list)
    deployment_mode: Literal["personal", "public"] = "personal"
    disclaimer: str = ""
    inputs_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_iso_week(self) -> WeeklyPlan:
        year_text, week_text = self.week.split("-W")
        try:
            date.fromisocalendar(int(year_text), int(week_text), 1)
        except ValueError as error:
            raise ValueError("week must be a valid ISO week") from error
        return self


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(child)
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(child) for child in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported value in policy JSON: {type(value).__name__}")


def _digest_inputs(
    user_data: UserData,
    market: MarketContext,
    research: ResearchPack | None,
    week: str,
    as_of: date,
) -> str:
    document = {
        "engine": "c27a-v1",
        "week": week,
        "as_of": as_of,
        "user_data": user_data,
        "market": market,
        "research": research,
    }
    canonical = json.dumps(
        _jsonable(document), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _holding_value(holding: HoldingInput, fx_rates: Mapping[str, Decimal]) -> Decimal | None:
    if holding.price is None:
        return None
    fx = _currency_fx_to_inr(holding.currency, fx_rates)
    if fx is None:
        return None
    return round_paise(holding.quantity * holding.price * fx)


def _currency_fx_to_inr(currency: str, fx_rates: Mapping[str, Decimal]) -> Decimal | None:
    return Decimal(1) if currency == "INR" else fx_rates.get(f"{currency}/INR")


def _money(value: Decimal | None) -> str | None:
    return format(round_paise(value), ".2f") if value is not None else None


_EXIT_ACTION_KINDS: dict[str, ActionKind] = {
    "EXIT": "SELL_EXIT",
    "TRIM_TO_TARGET": "SELL_TRIM",
    "REVIEW_THEN_EXIT": "REVIEW_EXIT",
    "SWAP": "SWAP",
}
_FOUNDATION_ACTION_KINDS = {
    "PAY_DOWN_DEBT",
    "PAY_EXTRA_ON_LOAN",
    "BUILD_EMERGENCY_FUND",
}
_BUY_ACTION_KINDS = {"BUY_LUMPSUM", "BUY_STP", "BUY_STOCK_TRANCHE"}


def _exit_action(card: ExitCard) -> Action | None:
    """Project an actionable, policy-sized exit card into the weekly action schema."""
    kind = _EXIT_ACTION_KINDS.get(card.action)
    if kind is None or card.quantity <= 0 or not card.lots_to_sell:
        return None
    triggered = any(trigger.status == "triggered" for trigger in card.triggers.values())
    if card.action == "EXIT":
        if card.urgency not in {"CRITICAL", "HIGH"}:
            return None
    elif card.action != "SWAP" and not triggered:
        return None
    return Action(
        rank=1,
        kind=kind,
        sleeve="equity",
        instrument_category="equity_holding",
        security_id=card.security_id,
        quantity=card.quantity,
        amount=card.tax_estimate.gross_proceeds,
        reason_code=card.reason_codes[0] if card.reason_codes else "target_gap",
        params=_jsonable(card.params),
        symbol=card.symbol,
        lots_to_sell=[
            {"lot": _jsonable(lot), "quantity": _jsonable(quantity)}
            for lot, quantity in card.lots_to_sell
        ],
        tax_estimate=_jsonable(card.tax_estimate),
        urgency=card.urgency,
        member_id=card.member_id,
        member_name=card.member_name,
        member_relationship=card.member_relationship,
    )


def _weekly_action_key(action: Action, position: int) -> tuple[int, int, int, int]:
    if action.kind == "SELL_EXIT" and action.urgency in {"CRITICAL", "HIGH"}:
        urgency = 0 if action.urgency == "CRITICAL" else 1
        return (0, urgency, 0, position)
    if action.kind in _FOUNDATION_ACTION_KINDS:
        return (1, 0, 0, position)
    if action.kind == "REBALANCE_SIPS":
        return (2, 0, 0, position)
    other_exit_order = {"SELL_TRIM": 0, "REVIEW_EXIT": 1, "SWAP": 2}
    if action.kind in other_exit_order:
        return (3, other_exit_order[action.kind], 0, position)
    return (4, 0, -int(action.amount * 100), position)


def _merge_weekly_actions(
    buy_list: ActionList, exit_cards: Sequence[ExitCard]
) -> tuple[list[Action], Decimal]:
    """Order buys and actionable exit cards together, cap the week, and carry omitted buys."""
    sell_actions = [action for card in exit_cards if (action := _exit_action(card)) is not None]
    candidates = [*buy_list.actions, *sell_actions]
    ordered = [
        action
        for _position, action in sorted(
            enumerate(candidates), key=lambda pair: _weekly_action_key(pair[1], pair[0])
        )
    ]
    selected = ordered[:3]
    selected_ids = {id(action) for action in selected}
    omitted_buy_amount = sum(
        (
            action.amount
            for action in buy_list.actions
            if id(action) not in selected_ids
            and action.kind in _FOUNDATION_ACTION_KINDS | _BUY_ACTION_KINDS
        ),
        Decimal(0),
    )
    carried = round_paise(buy_list.carry_forward + omitted_buy_amount)
    ranked = [action.model_copy(update={"rank": rank}) for rank, action in enumerate(selected, 1)]
    return ranked, carried


def _abroad_exposure(action: Action) -> str | None:
    if action.sleeve == "equity_india":
        return "india_equity"
    if action.sleeve != "equity_global":
        return None
    category = action.instrument_category.casefold()
    for exposure in (
        "global_all_world",
        "global_sp500",
        "global_nasdaq100",
        "global_developed",
        "global_em_ex_india",
    ):
        if exposure in category:
            return exposure
    return "global_all_world"


def _abroad_action_rows(
    actions: Sequence[Action],
    members: Sequence[MemberProfile],
    contexts: Sequence[MemberContext],
    holdings: Sequence[HoldingInput],
    market: MarketContext,
    research: ResearchPack | None,
) -> tuple[list[Action], list[dict[str, Any]], Decimal, list[str]]:
    """Replace Danish foreign buys with sourced instruments and code-sized DKK lines."""
    profiles = {member.id: member for member in members}
    context_by_id = {context.member.id: context for context in contexts}
    output: list[Action] = []
    buy_list: list[dict[str, Any]] = []
    notes: list[str] = []
    carried = Decimal(0)
    inr_per_dkk = market.fx_rates.get("DKK/INR")
    running_ask_room: dict[str, Decimal] = {}

    for action in actions:
        member = profiles.get(action.member_id)
        if (
            member is None
            or member.residency.country != "DK"
            or action.kind not in {"BUY_LUMPSUM", "BUY_STP"}
        ):
            output.append(action)
            continue
        exposure = _abroad_exposure(action)
        if exposure is None:
            output.append(action)
            continue
        context = context_by_id.get(member.id)
        if context is None:
            output.append(action)
            continue
        direct_india = (
            exposure == "india_equity"
            and pick_india_vehicle(
                has_indian_nri_share_account=member.investing.has_indian_nri_share_account,
                broker=member.investing.broker,
            )
            == "in_equity_direct_core"
        )
        if direct_india:
            output.append(action)
            continue
        if inr_per_dkk is None or inr_per_dkk <= 0:
            carried += action.amount
            notes.append("denmark_fx_missing")
            continue

        if member.id not in running_ask_room:
            running_ask_room[member.id] = max(
                Decimal(0), context.limits_room.get("ask_deposit_dkk") or Decimal(0)
            )
        requested_account = str(action.params.get("account_type", "ASK")).upper()
        ask_room = running_ask_room[member.id] if requested_account != "ORDINARY" else Decimal(0)
        total_dkk = dkk_from_inr(action.amount, inr_per_dkk)
        ask_only = ask_room > 0
        picks = ranked_etfs(exposure, member.investing.broker, ask_only=ask_only)
        if not picks and ask_only:
            picks = ranked_etfs(exposure, member.investing.broker)
        broker_availability_unchecked = False
        if not picks:
            broker_availability_unchecked = True
            picks = ranked_etfs(
                exposure,
                member.investing.broker,
                ask_only=ask_only,
                broker_filter=False,
            )
            if not picks and ask_only:
                picks = ranked_etfs(
                    exposure,
                    member.investing.broker,
                    broker_filter=False,
                )
        if not picks:
            carried += action.amount
            notes.append("abroad_etf_unavailable")
            continue
        pick = picks[0]
        details: dict[str, Any] = {
            "alternatives": [
                {"name": plain_fund_name(str(row.instrument["name"]))} for row in picks[1:4]
            ],
            "held_us_etf_notes": [
                {
                    "symbol": holding.symbol,
                    "eu_twin": plain_fund_name(str(pick.instrument["name"])),
                }
                for holding in holdings
                if holding.member_id == member.id
                and holding.market == "US"
                and holding.security_type == "etf"
                and holding.sleeve == "equity_global"
            ],
            "pension_note": member.investing.has_danish_pension_account,
            "tax_verified": False,
        }
        stock_lines: list[dict[str, Any]] = []
        stock_buys: list[tuple[SatelliteIdea, Decimal]] = []
        if exposure != "india_equity" and member.investing.us_stock_picks_opt_in:
            investable_inr = max(Decimal(0), context.snapshot.investable_total)
            investable_dkk = dkk_from_inr(investable_inr, inr_per_dkk)
            stock_budget = total_dkk * member.investing.us_stock_picks_pct / Decimal(100)
            selected_ideas = research.ideas if research is not None else ()
            stock_buys = approved_us_stock_buys(
                selected_ideas,
                investable_wealth_dkk=investable_dkk,
                available_dkk=stock_budget,
                satellite_pct=member.investing.us_stock_picks_pct,
                minimum_buy_dkk=Decimal(
                    str(
                        load_eu_etfs()
                        .get("selection", {})
                        .get("stock_picks", {})
                        .get("minimum_buy_dkk", "2000")
                    )
                ),
            )
            stock_total = sum((amount for _idea, amount in stock_buys), Decimal(0))
            stock_tax_rules = load_eu_etfs().get("selection", {}).get("us_stock_tax", {})
            stock_tax_sources = stock_tax_rules.get("sources", {})
            current_situs_usd = us_situs_value_usd(
                holding for holding in holdings if holding.member_id == member.id
            )
            usd_per_inr = market.fx_rates.get("USD/INR")
            projected_situs_usd = (
                current_situs_usd + stock_total * inr_per_dkk / usd_per_inr
                if usd_per_inr is not None and usd_per_inr > 0
                else None
            )
            stock_tax_details = {
                "dividend_withholding_pct": str(
                    stock_tax_rules.get("dividend_withholding_pct", "15")
                ),
                "current_us_situs_usd": format(current_situs_usd, ".2f"),
                "projected_us_situs_usd": (
                    format(projected_situs_usd, ".2f") if projected_situs_usd is not None else None
                ),
                "estate_filing_threshold_usd": str(
                    stock_tax_rules.get("estate_filing_threshold_usd", "60000")
                ),
                "verified": False,
                "sources": [
                    {"label": key, "url": url}
                    for key, url in stock_tax_sources.items()
                    if isinstance(url, str)
                ],
            }
            for idea, amount_dkk in stock_buys:
                if not isinstance(idea, SatelliteIdea):
                    continue
                stock_lines.append(
                    {
                        "member_id": member.id,
                        "account": "ordinary",
                        "broker": member.investing.broker,
                        "instrument": {
                            "name": idea.name,
                            "ticker": idea.symbol,
                            "exchange": idea.exchange,
                            "isin": idea.isin,
                            "currency": idea.currency,
                        },
                        "amount_dkk": format(round_dkk_for_display(amount_dkk), "f"),
                        "amount_inr": format(inr_from_dkk(amount_dkk, inr_per_dkk), ".2f"),
                        "reason_codes": ["approved_us_stock_idea"],
                        "how_to_key": f"how_to.{member.investing.broker}",
                        "details": {
                            "tax_verified": False,
                            "us_stock_tax": stock_tax_details,
                        },
                    }
                )
        else:
            stock_total = Decimal(0)
        etf_amount = max(Decimal(0), total_dkk - stock_total)
        lines = account_lines(
            pick,
            member_id=member.id,
            broker=member.investing.broker,
            amount_dkk=etf_amount,
            ask_room_dkk=ask_room,
            inr_per_dkk=inr_per_dkk,
            details=details,
            broker_availability_unchecked=broker_availability_unchecked,
        )
        if exposure == "india_equity":
            for line in lines:
                line["reason_codes"] = ["india_fund_from_denmark", *line["reason_codes"]]
        lines.extend(stock_lines)
        already = sum(1 for line in buy_list if line["member_id"] == member.id)
        remaining = max(0, 3 - already)
        accepted = lines[:remaining]
        omitted_inr = Decimal(0)
        if len(accepted) < len(lines):
            omitted_inr = round_paise(
                sum(
                    (Decimal(str(line["amount_inr"])) for line in lines[len(accepted) :]),
                    Decimal(0),
                )
            )
            carried += omitted_inr
        if not accepted:
            carried += action.amount
            continue
        for line in accepted:
            unrounded_dkk = line.pop("_amount_dkk_unrounded", Decimal(0))
            if line["account"] == "ask":
                running_ask_room[member.id] = max(
                    Decimal(0), running_ask_room[member.id] - unrounded_dkk
                )
        for line in lines[len(accepted) :]:
            line.pop("_amount_dkk_unrounded", None)
        buy_list.extend(accepted)
        params = {
            **action.params,
            "broker": member.investing.broker,
            "abroad_buy_list": accepted,
            "account_type": accepted[0]["account"].upper(),
        }
        output.append(
            action.model_copy(
                update={
                    "kind": "BUY_ABROAD",
                    "amount": round_paise(action.amount - omitted_inr),
                    "instrument_category": "abroad_etf",
                    "reason_code": "buy_abroad",
                    "params": params,
                }
            )
        )
    return output, buy_list, round_paise(carried), list(dict.fromkeys(notes))


def _member_sip_actions(week_input: WeekInput, members: Sequence[MemberProfile]) -> list[Action]:
    """Build SIP redirections independently from each member's own inputs."""
    actions: list[Action] = []
    for member in members:
        own_sips = [
            Sip(
                sleeve=sip.sleeve,
                instrument_category=sip.instrument_category,
                monthly_amount=sip.monthly_amount,
            )
            for sip in member.investing.existing_sips
        ]
        member_input = week_input.model_copy(
            update={
                "new_money": Decimal(0),
                "monthly_sip_capacity": member.investing.monthly_sip_capacity,
                "existing_sips": own_sips,
            }
        )
        member_actions = build_action_list(member_input, apply_weekly_cap=False)
        actions.extend(
            action.model_copy(
                update={
                    "member_id": member.id,
                    "member_name": member.name,
                    "member_relationship": member.relationship,
                }
            )
            for action in member_actions.actions
            if action.kind == "REBALANCE_SIPS"
        )
    return actions


def _decimal(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _foundation_json(report: FoundationReport) -> list[dict[str, Any]]:
    return [
        {
            "key": item.key,
            "status": item.status,
            "required": _money(item.required),
            "have": _money(item.have),
            "gap": _money(item.gap),
        }
        for item in report.items
    ]


def _household_foundation(
    snapshot: WealthSnapshot,
    financial_context: FinancialContext,
    member_contexts: Sequence[MemberContext],
    *,
    emergency_months_override: int | None,
) -> tuple[FoundationReport, dict[str, dict[str, str | None]]]:
    """Keep shared checks at household level and derive life cover per earner."""
    shared = foundation_check(
        snapshot,
        financial_context,
        emergency_months_override=emergency_months_override,
    )
    if financial_context.dependants == 0:
        return shared, {
            member.member_id: {
                "required": "0.00",
                "have": "0.00",
                "gap": "0.00",
                "status": "ok",
            }
            for member in member_contexts
        }

    term_by_member: dict[str, dict[str, str | None]] = {}
    required = Decimal(0)
    covered = Decimal(0)
    gaps = Decimal(0)
    unknown = False
    for member in member_contexts:
        annual_income = member.financial_context.annual_income
        if annual_income is not None and annual_income <= 0:
            term_by_member[member.member_id] = {
                "required": "0.00",
                "have": "0.00",
                "gap": "0.00",
                "status": "not_earning",
            }
            continue
        if annual_income is None and member.financial_context.employment_type in {
            "retired",
            "other",
        }:
            term_by_member[member.member_id] = {
                "required": "0.00",
                "have": "0.00",
                "gap": "0.00",
                "status": "not_earning",
            }
            continue
        context = member.financial_context.model_copy(
            update={"dependants": financial_context.dependants}
        )
        item = foundation_check(member.snapshot, context)["term_cover"]
        term_by_member[member.member_id] = {
            "required": _money(item.required),
            "have": _money(item.have),
            "gap": _money(item.gap),
            "status": item.status,
        }
        if item.status == "unknown":
            unknown = True
            continue
        required += item.required or Decimal(0)
        covered += item.have or Decimal(0)
        gaps += item.gap or Decimal(0)

    if unknown:
        term_item = FoundationItem(
            key="term_cover",
            status="unknown",
            required=None,
            have=round_paise(covered),
            gap=None,
        )
    else:
        term_item = FoundationItem(
            key="term_cover",
            status="ok" if gaps == 0 else "gap",
            required=round_paise(required),
            have=round_paise(covered),
            gap=round_paise(gaps),
        )
    items = tuple(term_item if item.key == "term_cover" else item for item in shared.items)
    return FoundationReport(items), term_by_member


def plan_health_score(
    allocation: Mapping[str, Mapping[str, str]], foundation: Sequence[Mapping[str, Any]]
) -> int:
    """Score allocation fit and foundation status on a 0–100 scale.

    Each sleeve's absolute percentage-point gap is weighted by its target share.
    Each unresolved foundation item costs 10 points. The final score is rounded
    half-up to an integer and clamped to the 0–100 range.
    """
    allocation_penalty = sum(
        (
            abs(Decimal(item["current_pct"]) - Decimal(item["target_pct"]))
            * Decimal(item["target_pct"])
            / Decimal(100)
            for item in allocation.values()
        ),
        Decimal(0),
    )
    foundation_penalty = Decimal(10) * sum(
        1
        for item in foundation
        if item.get("status") != "ok" and Decimal(str(item.get("gap") or "0")) > 0
    )
    score = (Decimal(100) - allocation_penalty - foundation_penalty).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return int(max(Decimal(0), min(Decimal(100), score)))


def _route_json(route: RouteDecision) -> dict[str, Any]:
    return _jsonable(route)


def _debt_answer_plan(plan: AllocationPlan) -> AllocationPlan:
    """Restore the aggregate equity key expected by C12's debt-answer contract."""
    target = dict(plan.target_rupees)
    current = dict(plan.current_rupees)
    target_pct = dict(plan.target_percent)
    current_pct = dict(plan.current_percent)
    gap = dict(plan.gap)
    target["equity"] = target["equity_india"] + target["equity_global"]
    current["equity"] = current["equity_india"] + current["equity_global"]
    target_pct["equity"] = target_pct["equity_india"] + target_pct["equity_global"]
    current_pct["equity"] = current_pct["equity_india"] + current_pct["equity_global"]
    gap["equity"] = target["equity"] - current["equity"]
    return replace(
        plan,
        target_rupees=MappingProxyType(target),
        target_percent=MappingProxyType(target_pct),
        current_rupees=MappingProxyType(current),
        current_percent=MappingProxyType(current_pct),
        gap=MappingProxyType(gap),
    )


def _debt_answer_json(answer: Any) -> dict[str, Any]:
    return {
        "target_debt": _money(answer.target_debt),
        "current_debt": _money(answer.current_debt),
        "locked_debt": _money(answer.locked_debt),
        "gap": _money(answer.gap),
        "status": answer.status,
        "new_money_order": list(answer.new_money_order),
        "message_key": answer.message_key,
        "params": dict(answer.params),
    }


def _tax_class(holding: HoldingInput) -> str:
    if holding.market == "US":
        return "foreign_equity"
    if holding.security_type == "etf" and holding.sleeve == "equity_global":
        return "intl_etf_listed_in"
    if holding.security_type == "mf" and holding.category and "international" in holding.category:
        return "intl_fof"
    return "listed_equity"


def _exit_cards(
    holdings: Sequence[HoldingInput],
    member_contexts: Sequence[MemberContext],
    market: MarketContext,
    as_of: date,
    notes: list[str],
) -> list[ExitCard]:
    context_by_id = {member.member_id: member for member in member_contexts}
    self_member = next((member for member in member_contexts if member.member.is_self), None)
    if self_member is None:
        return []
    exit_rules = load_exit_rules()
    target_weight = Decimal(
        str(exit_rules["satellite"]["drift"]["max_weight_of_equity_pct"])  # type: ignore[index]
    )
    cards: list[ExitCard] = []
    for holding in holdings:
        if holding.security_type != "equity":
            continue
        value = _holding_value(holding, market.fx_rates)
        if value is None:
            if holding.price is None:
                notes.append(f"holding_price_missing:{holding.symbol}")
            else:
                notes.append(f"holding_fx_missing:{holding.symbol}")
        holding_fx = _currency_fx_to_inr(holding.currency, market.fx_rates)
        price_inr = (
            holding.price * holding_fx
            if holding.price is not None and holding_fx is not None
            else Decimal("0")
        )
        lots = _normalised_lots(holding, market.fx_rates, notes)
        owner = context_by_id.get(holding.member_id or self_member.member_id, self_member)
        equity_total = owner.snapshot.amount_for(Sleeve.EQUITY_INDIA) + owner.snapshot.amount_for(
            Sleeve.EQUITY_GLOBAL
        )
        weight_pct = (
            value * Decimal(100) / equity_total if value is not None and equity_total else None
        )
        facts = HoldingFacts(
            security_id=holding.security_id,
            symbol=holding.symbol,
            sector=holding.sector or "unknown",
            is_core=False,
            lots=lots,
            price=price_inr,
            weight_pct_of_equity=weight_pct,
            target_weight_pct=target_weight,
        )
        card = build_exit_card(facts, as_of, owner.tax_context)
        tax_notes = list(card.tax_estimate.notes)
        tax_notes.extend(_acquisition_fx_notes(holding))
        if tax_notes != list(card.tax_estimate.notes):
            card = replace(
                card,
                tax_estimate=replace(card.tax_estimate, notes=tuple(dict.fromkeys(tax_notes))),
            )
        cards.append(
            replace(
                card,
                member_id=owner.member_id,
                member_name=owner.name,
                member_relationship=owner.relationship,
            )
        )
    return cards


def _acquisition_fx_notes(holding: HoldingInput) -> tuple[str, ...]:
    foreign_lots = [lot for lot in holding.lots if (lot.currency or holding.currency) != "INR"]
    notes: list[str] = []
    if any(lot.fx_missing or lot.cost_per_unit_inr is None for lot in foreign_lots):
        notes.append("acquisition_fx_missing")
    if any(not lot.fx_missing and lot.cost_per_unit_inr is not None for lot in foreign_lots):
        notes.append("acquisition_fx_approximation")
    return tuple(notes)


def _normalised_lots(
    holding: HoldingInput,
    fx_rates: Mapping[str, Decimal],
    notes: list[str],
) -> list[Lot]:
    lots: list[Lot] = []
    for raw_lot in holding.lots:
        lot_currency = raw_lot.currency or holding.currency
        if raw_lot.acquired_on is None:
            notes.append(f"holding_lot_tax_facts_missing:{holding.symbol}")
            continue
        if lot_currency != "INR":
            if raw_lot.fx_missing or raw_lot.cost_per_unit_inr is None:
                notes.append(f"holding_lot_fx_missing:{holding.symbol}")
                continue
            cost = raw_lot.cost_per_unit_inr
        elif raw_lot.cost_per_unit is not None:
            cost = raw_lot.cost_per_unit
        else:
            notes.append(f"holding_lot_tax_facts_missing:{holding.symbol}")
            continue
        price_fx = _currency_fx_to_inr(holding.currency, fx_rates)
        price_inr = (
            holding.price * price_fx if holding.price is not None and price_fx is not None else None
        )
        lots.append(
            Lot(
                lot_id=raw_lot.lot_id,
                tax_class=_tax_class(holding),
                quantity=raw_lot.quantity,
                cost_per_unit=cost,
                acquired_on=raw_lot.acquired_on,
                fmv_2018_01_31=raw_lot.fmv_2018_01_31,
                market_price=price_inr,
            )
        )
    return lots


def _member_ltcg_harvest(
    holdings: Sequence[HoldingInput],
    member_contexts: Sequence[MemberContext],
    market: MarketContext,
    as_of: date,
    notes: list[str],
) -> dict[str, dict[str, Any]]:
    """Calculate available harvest room separately for every Indian-tax member."""
    output: dict[str, dict[str, Any]] = {}
    tax_rules = load_tax_rules()
    exemption_rule = tax_rules["equity_ltcg_exemption"]
    assert isinstance(exemption_rule, Mapping)
    exemption_amount = Decimal(str(exemption_rule["amount_inr"]))
    self_member = next((member for member in member_contexts if member.member.is_self), None)
    fallback_member_id = self_member.member_id if self_member is not None else "self"
    for member in member_contexts:
        member_lots: list[Lot] = []
        prices: dict[str, Decimal] = {}
        for holding in holdings:
            if (holding.member_id or fallback_member_id) != member.member_id:
                continue
            lots = _normalised_lots(holding, market.fx_rates, notes)
            member_lots.extend(lots)
            for lot in lots:
                if lot.market_price is not None:
                    prices[lot.lot_id] = lot.market_price
        suggestion = None
        if member.residency.status in {"RI", "NRI", "RNOR"}:
            suggestion = ltcg_harvest(
                member_lots,
                prices,
                as_of,
                member.tax_context,
            )
        harvest: dict[str, Any] = {
            "available": suggestion is not None,
            "exemption_inr": format(exemption_amount, "f"),
            "needs_review": member.residency.status == "NRI"
            and exemption_rule.get("needs_review") is True,
            "gains_booked": format(suggestion.gains_booked, "f") if suggestion else "0.00",
            "tax_saved_before_cess": (
                format(suggestion.tax_saved_before_cess, "f") if suggestion else "0.00"
            ),
            "lots": [
                {"lot_id": lot.lot_id, "quantity": format(quantity, "f")}
                for lot, quantity in suggestion.lots
            ]
            if suggestion
            else [],
        }
        output[member.member_id] = harvest
    return output


def _stress_json(current: Mapping[str, Decimal], target: Mapping[str, Decimal]) -> dict[str, Any]:
    scenarios = load_stress_rules()["scenarios"]
    result: dict[str, Any] = {}
    for name, scenario in scenarios.items():
        current_impact = stress_test(current, scenario)
        target_impact = stress_test(target, scenario)
        comparison = stress_compare(current, target, scenario)
        result[name] = {
            "current": _jsonable(current_impact),
            "target": _jsonable(target_impact),
            "difference": _jsonable(comparison),
        }
    return result


def _valid_week(week: str) -> None:
    try:
        year_text, week_text = week.split("-W")
        date.fromisocalendar(int(year_text), int(week_text), 1)
    except (ValueError, AttributeError) as error:
        raise ValueError("week must be a valid ISO week") from error


def personalise(
    user_data: UserData,
    market: MarketContext,
    research: ResearchPack | None,
    week: str,
    as_of: date,
) -> WeeklyPlan:
    """Run the deterministic Policy Engine chain and return one immutable weekly plan."""
    if not isinstance(user_data, UserData):
        raise ValueError("user_data must be UserData")
    if not isinstance(market, MarketContext):
        raise ValueError("market must be MarketContext")
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    _valid_week(week)

    notes = list(user_data.notes) + list(market.notes)
    selected_research = research if research is not None and research.week == week else None
    if research is not None and selected_research is None:
        notes.append("research_pack_week_mismatch")
    profile_band = (user_data.profile.risk_profile or "moderate").lower()
    residency = user_data.profile.residency
    if user_data.profile.risk_profile is None:
        notes.append("risk_profile_missing_default_moderate")

    member_profiles = user_data.members or (
        MemberProfile(
            id="self",
            name="You",
            relationship="self",
            is_self=True,
            residency=user_data.profile.residency,
            tax=MemberTax(
                tax_regime=user_data.profile.tax_regime or user_data.financial_context.tax_regime,
                tax_slab_pct=user_data.profile.tax_slab_pct,
                birth_year=user_data.investing.birth_year,
                annual_income=user_data.financial_context.annual_income,
                employment_type=user_data.financial_context.employment_type,
                fy_realised_equity_ltcg=user_data.realised_trades_this_fy.equity_ltcg,
                fy_realised_equity_stcg=user_data.realised_trades_this_fy.equity_stcg,
            ),
            investing=user_data.investing,
            realised_equity_ltcg=user_data.realised_trades_this_fy.equity_ltcg,
            realised_equity_stcg=user_data.realised_trades_this_fy.equity_stcg,
        ),
    )
    self_member = next((member for member in member_profiles if member.is_self), member_profiles[0])
    residency = self_member.residency
    residency_rule = load_global_rules().get("residency", {})
    emergency_months_override = None
    if any(
        member.residency.status == "NRI"
        and member.residency.country != "IN"
        and member.tax.employment_type == "salaried"
        for member in member_profiles
    ):
        emergency_months_override = int(residency_rule.get("nri_salaried_emergency_months", 9))
    market_holdings: list[tuple[str, Decimal, LiquidityClass]] = []
    member_market_holdings: list[tuple[str, str, Decimal, LiquidityClass]] = []
    holding_values: dict[tuple[str, str], Decimal | None] = {}
    for holding in user_data.holdings:
        value = _holding_value(holding, market.fx_rates)
        member_id = holding.member_id or self_member.id
        holding_values[(holding.security_id, member_id)] = value
        if value is None:
            missing_key = "holding_price_missing" if holding.price is None else "holding_fx_missing"
            notes.append(f"{missing_key}:{holding.symbol}")
            continue
        if holding.sleeve is None:
            notes.append(f"holding_sleeve_missing:{holding.symbol}")
            continue
        liquidity = (
            LiquidityClass.WITHIN_1Y
            if holding.security_type in {"bond", "sgb"}
            else LiquidityClass.INSTANT
        )
        market_holdings.append((holding.sleeve, value, liquidity))
        member_market_holdings.append((member_id, holding.sleeve, value, liquidity))

    household_snapshot = build_household_snapshot(
        user_data.valued_assets,
        market_holdings,
        member_market_holdings,
        member_profiles,
        self_member_id=self_member.id,
    )
    snapshot = household_snapshot.wealth
    has_indian_funds_by_member = {
        member.id: any(
            (holding.member_id or self_member.id) == member.id
            and holding.market == "IN"
            and holding.security_type in {"mf", "etf"}
            for holding in user_data.holdings
        )
        for member in member_profiles
    }
    member_contexts = build_member_contexts(
        member_profiles,
        household_snapshot,
        user_data.valued_assets,
        as_of_year=as_of.year,
        fx_rates=market.fx_rates,
        fallback_tax_slab_pct=user_data.profile.tax_slab_pct,
        fallback_tax_regime=user_data.profile.tax_regime,
        fallback_city_tier=user_data.profile.city_tier,
        has_indian_funds_by_member=has_indian_funds_by_member,
    )
    known_annual_incomes = [
        member.financial_context.annual_income
        for member in member_contexts
        if member.financial_context.annual_income is not None
    ]
    household_financial_context = (
        user_data.financial_context.model_copy(
            update={"annual_income": round_paise(sum(known_annual_incomes, Decimal(0)))}
        )
        if known_annual_incomes
        else user_data.financial_context
    )
    fx_usd_inr = market.fx_rates.get("USD/INR")
    base_allocation = build_targets(
        snapshot,
        user_data.financial_context,
        user_data.goals,
        profile_band,
        as_of,
        fx_usd_inr=fx_usd_inr,
        fx_rates=market.fx_rates,
        emergency_months_override=emergency_months_override,
    )
    tilts = selected_research.geo_tilts if selected_research is not None else None
    allocation = split_equity(
        base_allocation,
        user_data.goals,
        profile_band,
        tilts,
        residency=residency,
        fx_rates=market.fx_rates,
        as_of=as_of,
    )
    foundation, per_member_term_cover = _household_foundation(
        snapshot,
        household_financial_context,
        member_contexts,
        emergency_months_override=emergency_months_override,
    )
    answer = debt_answer(_debt_answer_plan(allocation), snapshot)

    equity_target = (
        allocation.target_rupees["equity_india"] + allocation.target_rupees["equity_global"]
    )
    global_share = (
        allocation.target_rupees["equity_global"] / equity_target
        if equity_target > 0
        else Decimal(0)
    )
    household_monthly_sip = round_paise(
        sum((member.monthly_sip_capacity for member in member_contexts), Decimal(0))
    )
    household_new_money = round_paise(
        sum((member.new_money for member in member_contexts), Decimal(0))
    )
    monthly_global = round_paise(household_monthly_sip * global_share)
    annual_global = round_paise(monthly_global * 12 + household_new_money * global_share)
    from wealth.policy.us_situs import us_situs_value_usd

    us_situs = us_situs_value_usd(user_data.holdings)
    route = select_route(
        RouteContext(
            monthly_global_new_money=monthly_global,
            annual_global_new_money=annual_global,
            has_lrs_broker=user_data.investing.has_lrs_broker,
            prefers_no_paperwork=user_data.investing.prefers_no_paperwork,
            us_situs_holdings_usd=us_situs,
            lrs_used_this_fy=user_data.investing.lrs_used_this_fy,
            amc_international_suspended=market.amc_international_suspended,
            etf_inav_premium_pct=market.etf_inav_premium_pct,
            residency=residency,
        )
    )
    route_candidates: list[tuple[Decimal, bool, MemberContext, RouteDecision]] = []
    for member in member_contexts:
        member_monthly_global = round_paise(member.monthly_sip_capacity * global_share)
        member_annual_global = round_paise(
            member_monthly_global * 12 + member.new_money * global_share
        )
        member_route = select_route(
            RouteContext(
                monthly_global_new_money=member_monthly_global,
                annual_global_new_money=member_annual_global,
                has_lrs_broker=member.member.investing.has_lrs_broker,
                prefers_no_paperwork=member.member.investing.prefers_no_paperwork,
                us_situs_holdings_usd=us_situs,
                lrs_used_this_fy=member.member.investing.lrs_used_this_fy,
                amc_international_suspended=market.amc_international_suspended,
                etf_inav_premium_pct=market.etf_inav_premium_pct,
                residency=member.residency,
            )
        )
        route_candidates.append((member.new_money, member.member.is_self, member, member_route))
    route_candidates.sort(key=lambda item: (item[0] <= 0, -item[0], not item[1], item[2].member_id))
    eligible_routes = [item[3] for item in route_candidates if item[3].vehicle != "no_open_route"]
    if eligible_routes:
        route = eligible_routes[0]

    self_birth_year = self_member.tax.birth_year or self_member.investing.birth_year
    age = max(0, as_of.year - self_birth_year) if self_birth_year is not None else 0
    if self_birth_year is None:
        notes.append("birth_year_missing_age_gated_debt_products_excluded")
    tax_slab = self_member.tax.tax_slab_pct or user_data.profile.tax_slab_pct or Decimal("30")
    main_debt_bucket = max(
        (
            bucket
            for bucket in allocation.buckets
            if bucket.target_by_sleeve.get("debt", Decimal(0)) > 0
        ),
        key=lambda bucket: bucket.target_by_sleeve.get("debt", Decimal(0)),
        default=None,
    )
    debt_picks: list[DebtPick] = []
    main_debt_picks: list[DebtPick] = []
    seen_instruments: set[str] = set()
    unranked_options: list[UnrankedDebtOption] = []
    seen_unranked_instruments: set[str] = set()
    for bucket in allocation.buckets:
        amount = bucket.target_by_sleeve.get("debt", Decimal(0))
        if amount <= 0:
            continue
        if bucket.kind == "emergency":
            horizon = Decimal(0)
            liquidity_need = LiquidityClass.INSTANT
        else:
            horizon = Decimal(max(bucket.horizon_years or 1, 1))
            liquidity_need = LiquidityClass.LOCKED
        result = pick_debt_with_unranked(
            DebtNeed(amount=amount, horizon_years=horizon, liquidity_need=liquidity_need),
            household_financial_context,
            tax_slab,
            age,
            market.debt_yields,
            snapshot,
            residency=residency,
        )
        for candidate in result.ranked_picks:
            if candidate.instrument not in seen_instruments:
                seen_instruments.add(candidate.instrument)
                debt_picks.append(candidate)
        for option in result.unranked_options:
            if option.instrument not in seen_unranked_instruments:
                seen_unranked_instruments.add(option.instrument)
                unranked_options.append(option)
        if bucket is main_debt_bucket:
            main_debt_picks = list(result.ranked_picks)

    index_percentiles: dict[Literal["IN", "US"], Decimal] = {}
    for market_code in ("IN", "US"):
        percentile = market.index_valuation_percentiles.get(market_code)
        if percentile is None:
            percentile = Decimal("50")
            notes.append(f"index_valuation_percentile_missing:{market_code}")
        index_percentiles[market_code] = percentile

    satellite_holdings: list[SatelliteHolding] = []
    for holding in user_data.holdings:
        value = holding_values[(holding.security_id, holding.member_id or self_member.id)]
        if holding.market == "IN" and holding.security_type == "equity" and value is not None:
            satellite_holdings.append(
                SatelliteHolding(
                    security_id=holding.security_id,
                    sector=holding.sector or "unknown",
                    value=value,
                )
            )
    ideas = list(selected_research.ideas) if selected_research is not None else []
    existing_sips = [sip for member in member_profiles for sip in member.investing.existing_sips]
    week_input = WeekInput(
        plan=allocation,
        foundation=foundation,
        new_money=household_new_money,
        monthly_sip_capacity=household_monthly_sip,
        existing_sips=[
            Sip(
                sleeve=sip.sleeve,
                instrument_category=sip.instrument_category,
                monthly_amount=sip.monthly_amount,
            )
            for sip in existing_sips
        ],
        route=route,
        debt_picks=debt_picks,
        satellite_pct=user_data.investing.satellite_pct,
        satellite_holdings=satellite_holdings,
        ideas=ideas,
        index_valuation_percentile=index_percentiles,
        week=week,
        india_core_vehicle=india_core_vehicle(residency),
    )
    loan_warnings: list[dict[str, Any]] = []
    loan_action: Action | None = None
    if user_data.loan_candidates:
        foundation_rules = resolve_rules()["foundation"]
        high_interest_threshold = decimal_rule(
            foundation_rules["high_interest_debt_rate_pct"],
            "foundation.high_interest_debt_rate_pct",
        )
        tax_slab = user_data.profile.tax_slab_pct or Decimal(0)
        safe_return = (
            market.debt_yields.get("bank_fd", Decimal(0)) * (Decimal(100) - tax_slab) / Decimal(100)
        )
        monthly_income = (
            household_financial_context.annual_income / Decimal(12)
            if household_financial_context.annual_income is not None
            else None
        )
        residency_rules = load_residency_rules()
        loan_decision = pay_extra_or_invest(
            user_data.loan_candidates,
            LoanPlanContext(
                spare_money_inr=household_new_money,
                emergency_fund_gap_inr=foundation["emergency_fund"].gap or Decimal(0),
                monthly_income_inr=monthly_income,
                expected_safe_return_pct=safe_return,
                high_interest_threshold_pct=high_interest_threshold,
                tax_rules=load_tax_relief_rules(),
                debt_load_rules=load_loan_policy_rules(),
                danish_interest_rule=residency_rules.get("denmark", {}).get("interest_deduction"),
            ),
        )
        if loan_decision.debt_load_level in {"attention", "high"}:
            loan_warnings.append(
                {
                    "key": f"loan_debt_load_{loan_decision.debt_load_level}",
                    "rate_pct": loan_decision.debt_load_pct,
                }
            )
        high_card = max(
            (
                candidate
                for candidate in user_data.loan_candidates
                if candidate.loan.loan_kind == "credit_card"
                and candidate.loan.rate_pct >= high_interest_threshold
            ),
            key=lambda item: (item.loan.rate_pct, item.id),
            default=None,
        )
        if high_card is not None:
            loan_warnings.append(
                {"key": "loan_high_interest_card", "rate_pct": high_card.loan.rate_pct}
            )
        if any(candidate.never_pays_off for candidate in user_data.loan_candidates):
            loan_warnings.append({"key": "loan_never_pays_off"})
        selected_loan = next(
            (
                candidate
                for candidate in user_data.loan_candidates
                if candidate.id == loan_decision.loan_id
            ),
            None,
        )
        if (
            loan_decision.action == "PAY_EXTRA"
            and loan_decision.suggested_amount_inr > 0
            and selected_loan is not None
        ):
            loan_action = Action(
                rank=1,
                kind="PAY_EXTRA_ON_LOAN",
                sleeve="debt",
                instrument_category="loan",
                amount=loan_decision.suggested_amount_inr,
                reason_code=loan_decision.reason_codes[0],
                params={
                    "loan_id": selected_loan.id,
                    "loan_kind": selected_loan.loan.loan_kind,
                    "effective_rate_pct": format(loan_decision.effective_rate_pct or 0, "f"),
                    "safe_return_pct": format(loan_decision.safe_return_pct, "f"),
                    "advice": loan_decision.answer,
                },
                member_id=selected_loan.member_id or self_member.id,
            )
            week_input = week_input.model_copy(
                update={
                    "new_money": max(
                        Decimal(0), household_new_money - loan_decision.suggested_amount_inr
                    )
                }
            )
    action_list = build_action_list(week_input, apply_weekly_cap=False)
    member_sips = _member_sip_actions(week_input, member_profiles)
    loan_carry = Decimal(0)
    if loan_action is not None:
        candidate = next(
            (
                item
                for item in user_data.loan_candidates
                if item.id == loan_action.params.get("loan_id")
            ),
            None,
        )
        existing_debt_amount = sum(
            (action.amount for action in action_list.actions if action.kind == "PAY_DOWN_DEBT"),
            Decimal(0),
        )
        combined_amount = loan_action.amount + existing_debt_amount
        if candidate is not None:
            applied_amount = min(combined_amount, candidate.outstanding_inr)
            loan_carry = round_paise(combined_amount - applied_amount)
            loan_action = loan_action.model_copy(update={"amount": round_paise(applied_amount)})
    action_list = action_list.model_copy(
        update={
            "actions": [
                *(
                    action
                    for action in action_list.actions
                    if action.kind != "REBALANCE_SIPS"
                    and (loan_action is None or action.kind != "PAY_DOWN_DEBT")
                ),
                *member_sips,
                *([loan_action] if loan_action is not None else []),
            ],
            "status": "ACTIONS"
            if any(action.kind != "REBALANCE_SIPS" for action in action_list.actions)
            or member_sips
            or loan_action is not None
            else "NONE",
            "carry_forward": round_paise(action_list.carry_forward + loan_carry),
        }
    )
    placement = place_actions(
        action_list.actions,
        member_contexts,
        debt_yields=market.debt_yields,
        fx_rates=market.fx_rates,
        amc_international_suspended=market.amc_international_suspended,
        etf_inav_premium_pct=market.etf_inav_premium_pct,
        us_situs_holdings_usd=us_situs,
        as_of_year=as_of.year,
    )
    action_list = action_list.model_copy(
        update={
            "actions": list(placement.actions),
            "carry_forward": round_paise(action_list.carry_forward + placement.carry_forward),
            "notes": list(dict.fromkeys([*action_list.notes, *placement.notes])),
        }
    )
    breaches = band_breaches(allocation)
    exits = _exit_cards(user_data.holdings, member_contexts, market, as_of, notes)
    ltcg_by_member = _member_ltcg_harvest(
        user_data.holdings,
        member_contexts,
        market,
        as_of,
        notes,
    )
    weekly_actions, carry_forward = _merge_weekly_actions(action_list, exits)
    weekly_actions, abroad_buy_list, abroad_carry, abroad_notes = _abroad_action_rows(
        weekly_actions,
        member_profiles,
        member_contexts,
        user_data.holdings,
        market,
        selected_research,
    )
    carry_forward = round_paise(carry_forward + abroad_carry)
    notes.extend(abroad_notes)
    if market.yield_table_as_of is not None and not market.yield_table_verified:
        notes.append(f"debt_yields_unverified_as_of:{market.yield_table_as_of.isoformat()}")
    notes.extend(action_list.notes)
    notes.extend(allocation.notes)
    has_indian_funds = has_indian_funds_by_member.get(self_member.id, False)
    selected_residency_notes = residency_note_codes(
        residency,
        has_indian_funds=has_indian_funds,
        us_situs_exposure=(us_situs > 0 or route.vehicle in {"us_etf", "us_broad_market_etf"}),
    )
    notes.extend(selected_residency_notes)
    if residency.status != "RI":
        notes.extend(route.warnings)

    current_stress = {
        "equity_india": allocation.current_rupees["equity_india"],
        "equity_global": allocation.current_rupees["equity_global"],
        "debt": allocation.current_rupees["debt"],
        "gold": allocation.current_rupees["gold"],
        "cash": allocation.current_rupees["cash"],
    }
    target_stress = {
        "equity_india": allocation.target_rupees["equity_india"],
        "equity_global": allocation.target_rupees["equity_global"],
        "debt": allocation.target_rupees["debt"],
        "gold": allocation.target_rupees["gold"],
        "cash": allocation.target_rupees["cash"],
    }
    allocation_json = {
        sleeve: {
            "target": _money(allocation.target_rupees[sleeve]),
            "current": _money(allocation.current_rupees[sleeve]),
            "gap": _money(allocation.gap[sleeve]),
            "target_pct": _decimal(allocation.target_percent[sleeve]),
            "current_pct": _decimal(allocation.current_percent[sleeve]),
        }
        for sleeve in allocation.target_rupees
    }
    foundation_json = _foundation_json(foundation)
    debt_picks_json = [
        {
            "instrument": item.instrument,
            "after_tax_yield_pct": _decimal(item.after_tax_yield_pct),
            "suggested_tenure": _jsonable(item.reason_params.get("suggested_tenure")),
            "reason_codes": item.reason_codes,
            "reason_params": _jsonable(item.reason_params),
        }
        for item in main_debt_picks[:3]
    ]
    unranked_options_json = [_jsonable(option) for option in unranked_options]
    ppf_rate = market.debt_yields.get("ppf", Decimal(0))
    fd_rate = market.debt_yields.get("bank_fd", market.debt_yields.get("sweep_fd", Decimal(0)))
    insurance_reviews_json: list[dict[str, Any]] = []
    for asset, _valuation in user_data.valued_assets:
        if not isinstance(asset, TraditionalPolicyAsset):
            continue
        review = review_policy(asset, as_of, {"ppf": ppf_rate, "fd": fd_rate})
        insurance_reviews_json.append(
            {
                "plan": asset.plan,
                "policy_kind": asset.policy_kind,
                **_jsonable(review),
            }
        )
    context_by_id = {member.member_id: member for member in member_contexts}
    member_rows: list[dict[str, Any]] = []
    for breakdown in household_snapshot.members:
        context = context_by_id[breakdown.member_id]
        has_joint_assets = any(
            asset.joint and (asset.member_id or self_member.id) == context.member_id
            for asset, _valuation in user_data.valued_assets
        ) or any(
            holding.joint and (holding.member_id or self_member.id) == context.member_id
            for holding in user_data.holdings
        )
        member_notes = list(context.notes)
        if has_joint_assets:
            member_notes.append("joint_asset_tax_owner")
        member_rows.append(
            {
                **breakdown.to_json(),
                "residency": context.residency.model_dump(mode="json"),
                "eligibility": dict(context.eligibility),
                "eligibility_status": dict(context.eligibility_status),
                "eligibility_conditions": dict(context.eligibility_conditions),
                "limits_room": {key: _decimal(value) for key, value in context.limits_room.items()},
                "notes": list(dict.fromkeys(member_notes)),
                "ltcg_harvest": ltcg_by_member[context.member_id],
                "foundation": {"term_cover": per_member_term_cover.get(context.member_id)},
            }
        )

    deployment_mode = user_data.deployment_mode
    if deployment_mode not in {"personal", "public"}:
        raise ValueError("deployment_mode must be personal or public")
    return WeeklyPlan(
        week=week,
        as_of=as_of,
        plan_health=plan_health_score(allocation_json, foundation_json),
        carry_forward=_money(carry_forward) or "0.00",
        debt_picks=debt_picks_json,
        unranked_options=unranked_options_json,
        insurance_reviews=insurance_reviews_json,
        allocation=allocation_json,
        band_breaches=[
            {
                "sleeve": item.sleeve,
                "actual_pct": _decimal(item.actual_pct),
                "target_pct": _decimal(item.target_pct),
                "direction": item.direction,
                "excess_amount": _money(item.excess_amount),
            }
            for item in breaches
        ],
        debt_answer=_debt_answer_json(answer),
        foundation=foundation_json,
        route=_route_json(route),
        actions=[action.model_dump(mode="json") for action in weekly_actions],
        abroad_buy_list=abroad_buy_list,
        exit_cards=[_jsonable(card) for card in exits],
        stress=_stress_json(current_stress, target_stress),
        notes=list(dict.fromkeys(notes)),
        residency=residency,
        residency_notes=selected_residency_notes,
        members=member_rows,
        placement_notes=list(placement.notes),
        deployment_mode=deployment_mode,
        disclaimer=deployment_disclaimer(deployment_mode, residency),
        inputs_digest=_digest_inputs(user_data, market, selected_research, week, as_of),
    )
