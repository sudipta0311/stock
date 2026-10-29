"""Build a bounded, deterministic list of this week's investment actions."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from wealth.core.money import round_paise
from wealth.policy.allocation import AllocationPlan
from wealth.policy.debt_picker import DebtPick, load_debt_rules
from wealth.policy.foundation import FoundationReport
from wealth.policy.routes import RouteDecision
from wealth.policy.rules import decimal_rule, integer_rule
from wealth.schemas.common import Rupees

PlanActionSleeve = Literal["equity_india", "equity_global", "debt", "gold", "cash"]
ActionKind = Literal[
    "PAY_DOWN_DEBT",
    "BUILD_EMERGENCY_FUND",
    "BUY_LUMPSUM",
    "BUY_STP",
    "BUY_STOCK_TRANCHE",
    "REBALANCE_SIPS",
    "SELL_EXIT",
    "SELL_TRIM",
    "REVIEW_EXIT",
    "SWAP",
    "SWITCH_TO_DIRECT",
    "REDIRECT_SIP",
    "CONSOLIDATE_FUNDS",
    "REVIEW_FUND",
]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
DossierStatus = Literal["APPROVED", "PENDING", "VETOED", "UNRESOLVED"]
Market = Literal["IN", "US"]
BuyingRules = Mapping[str, Any]
_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "buying.yaml"
_SLEEVES: tuple[PlanActionSleeve, ...] = (
    "equity_india",
    "equity_global",
    "debt",
    "gold",
    "cash",
)


def _strict_decimal(value: object) -> Decimal:
    result = decimal_rule(value, "decimal")
    return Decimal(format(result, "f"))


def _fixed_decimal(value: Decimal) -> str:
    return format(value, "f")


def _money_string(value: Decimal) -> str:
    return format(round_paise(value), ".2f")


_INPUT_DECIMAL = Annotated[Decimal, BeforeValidator(_strict_decimal)]
_PERCENTILE = Annotated[_INPUT_DECIMAL, Field(ge=Decimal(0), le=Decimal(100))]
_POSITIVE_PRICE = Annotated[Rupees, Field(gt=Decimal(0))]
_OUTPUT_DECIMAL = Annotated[
    Decimal,
    BeforeValidator(_strict_decimal),
    PlainSerializer(_fixed_decimal, return_type=str, when_used="json"),
]
ParamsValue = str | list[dict[str, str]]


class Sip(BaseModel):
    """One existing monthly SIP commitment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sleeve: PlanActionSleeve
    instrument_category: str = Field(min_length=1)
    monthly_amount: Rupees


class SatelliteHolding(BaseModel):
    """Current value of one holding in the India satellite sleeve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    value: Rupees


class SatelliteIdea(BaseModel):
    """A researched candidate that deterministic rules may admit or reject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    market: Market
    sector: str = Field(min_length=1)
    price: _POSITIVE_PRICE
    quant_decile: int = Field(ge=1, le=10)
    dossier_status: DossierStatus
    valuation_percentile: _PERCENTILE
    confidence: Confidence


class WeekInput(BaseModel):
    """Inputs from the allocation, foundation, research, and route engines."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    plan: AllocationPlan
    foundation: FoundationReport
    new_money: Rupees
    monthly_sip_capacity: Rupees
    existing_sips: list[Sip] = Field(default_factory=list)
    route: RouteDecision
    debt_picks: list[DebtPick] = Field(default_factory=list)
    satellite_pct: Annotated[_INPUT_DECIMAL, Field(ge=Decimal(0), le=Decimal(25))]
    satellite_holdings: list[SatelliteHolding] = Field(default_factory=list)
    ideas: list[SatelliteIdea] = Field(default_factory=list)
    index_valuation_percentile: dict[Market, _PERCENTILE] = Field(default_factory=dict)
    week: str = Field(pattern=r"^\d{4}-W(?:0[1-9]|[1-4][0-9]|5[0-3])$")
    india_core_vehicle: str = "in_core_index"

    def model_post_init(self, __context: Any) -> None:
        year_text, week_text = self.week.split("-W")
        try:
            date.fromisocalendar(int(year_text), int(week_text), 1)
        except ValueError as error:
            raise ValueError("week must be a valid ISO week") from error


class Action(BaseModel):
    """One concrete policy action with machine-readable parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    kind: ActionKind
    sleeve: str = Field(min_length=1)
    instrument_category: str = Field(min_length=1)
    route: str | None = None
    security_id: str | None = None
    quantity: _OUTPUT_DECIMAL | None = None
    amount: Rupees
    tranche: dict[str, str] | None = None
    reason_code: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    symbol: str | None = None
    lots_to_sell: list[dict[str, Any]] | None = None
    tax_estimate: dict[str, Any] | None = None
    urgency: str | None = None
    member_id: str = "self"
    member_name: str = "You"
    member_relationship: str = "self"
    clubbing_check: str | None = None


class ActionList(BaseModel):
    """The final policy output for one ISO week."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    week: str
    status: Literal["ACTIONS", "NONE"]
    actions: list[Action]
    carry_forward: Rupees
    notes: list[str] = Field(default_factory=list)


def load_buying_rules() -> dict[str, Any]:
    """Read and validate the checked-in buying thresholds."""
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        document = yaml.safe_load(rules_file)
    if not isinstance(document, dict):
        raise ValueError("buying rules must contain a YAML mapping")
    validate_buying_rules(document)
    return document


def validate_buying_rules(rules: BuyingRules) -> None:
    """Validate the versioned allocation, satellite, and STP thresholds."""
    if integer_rule(rules.get("version"), "version") != 1:
        raise ValueError("unsupported buying rules version")
    increment = decimal_rule(rules.get("round_to_inr"), "round_to_inr")
    if increment <= 0 or increment != increment.to_integral_value():
        raise ValueError("round_to_inr must be a positive whole number")
    if decimal_rule(rules.get("min_action_inr"), "min_action_inr") <= 0:
        raise ValueError("min_action_inr must be positive")
    maximum_actions = integer_rule(rules.get("max_actions"), "max_actions")
    if not 1 <= maximum_actions <= 3:
        raise ValueError("max_actions must be from 1 to 3")

    satellite = rules.get("satellite")
    if not isinstance(satellite, Mapping):
        raise ValueError("satellite must be a mapping")
    default_pct = decimal_rule(
        satellite.get("default_pct_of_india_equity"), "satellite.default_pct_of_india_equity"
    )
    if not Decimal(0) <= default_pct <= Decimal(25):
        raise ValueError("satellite default percentage must be from 0 to 25")
    for key in ("max_stock_pct_of_total_equity", "max_sector_pct_of_satellite"):
        value = decimal_rule(satellite.get(key), f"satellite.{key}")
        if not Decimal(0) <= value <= Decimal(100):
            raise ValueError(f"satellite.{key} must be from 0 to 100")
    if integer_rule(satellite.get("tranches"), "satellite.tranches") <= 0:
        raise ValueError("satellite.tranches must be positive")
    min_decile = integer_rule(satellite.get("min_quant_decile"), "satellite.min_quant_decile")
    if not 1 <= min_decile <= 10:
        raise ValueError("satellite.min_quant_decile must be from 1 to 10")
    max_valuation = decimal_rule(
        satellite.get("max_valuation_percentile"), "satellite.max_valuation_percentile"
    )
    if not Decimal(0) <= max_valuation <= Decimal(100):
        raise ValueError("satellite.max_valuation_percentile must be from 0 to 100")

    global_satellite = rules.get("global_satellite")
    if not isinstance(global_satellite, Mapping):
        raise ValueError("global_satellite must be a mapping")
    if global_satellite.get("requires_route") != "C":
        raise ValueError("global_satellite.requires_route must be C")
    if (
        decimal_rule(
            global_satellite.get("min_global_equity_inr"),
            "global_satellite.min_global_equity_inr",
        )
        < 0
    ):
        raise ValueError("global_satellite.min_global_equity_inr must be non-negative")
    global_satellite_pct = decimal_rule(
        global_satellite.get("max_pct_of_global_equity"),
        "global_satellite.max_pct_of_global_equity",
    )
    if not Decimal(0) <= global_satellite_pct <= Decimal(100):
        raise ValueError("global satellite percentage must be from 0 to 100")

    lumpsum = rules.get("lumpsum")
    if not isinstance(lumpsum, Mapping):
        raise ValueError("lumpsum must be a mapping")
    if (
        decimal_rule(
            lumpsum.get("stp_threshold_multiple_of_monthly_sip"),
            "lumpsum.stp_threshold_multiple_of_monthly_sip",
        )
        < 0
    ):
        raise ValueError("STP threshold multiple must be non-negative")
    for key in ("stp_tranches_normal", "stp_tranches_expensive"):
        if integer_rule(lumpsum.get(key), f"lumpsum.{key}") <= 0:
            raise ValueError(f"lumpsum.{key} must be positive")
    expensive_percentile = decimal_rule(
        lumpsum.get("expensive_valuation_percentile"),
        "lumpsum.expensive_valuation_percentile",
    )
    if not Decimal(0) <= expensive_percentile <= Decimal(100):
        raise ValueError("expensive valuation percentile must be from 0 to 100")


def build_action_list(
    week_input: WeekInput,
    rules: BuyingRules | None = None,
    *,
    apply_weekly_cap: bool = True,
) -> ActionList:
    """Use foundation gaps, portfolio gaps, and SIP capacity to build up to three actions."""
    selected_rules = load_buying_rules() if rules is None else rules
    validate_buying_rules(selected_rules)
    increment = decimal_rule(selected_rules["round_to_inr"], "round_to_inr")
    min_action = decimal_rule(selected_rules["min_action_inr"], "min_action_inr")

    remaining = week_input.new_money
    carry_forward = Decimal("0.00")
    foundation_actions: list[Action] = []
    emergency_pick_unavailable = False

    debt_gap = _gap_value(week_input.foundation["high_interest_debt"].gap)
    debt_amount = min(remaining, debt_gap)
    if debt_amount > 0:
        foundation_actions.append(
            _action(
                kind="PAY_DOWN_DEBT",
                sleeve="debt",
                instrument_category="high_interest_debt",
                amount=debt_amount,
                reason_code="high_interest_debt_gap",
            )
        )
        remaining = round_paise(remaining - debt_amount)

    emergency_gap = _gap_value(week_input.foundation["emergency_fund"].gap)
    emergency_amount = min(remaining, emergency_gap)
    if emergency_amount > 0:
        instant_pick = _top_instant_debt_pick(week_input.debt_picks)
        if instant_pick is not None:
            foundation_actions.append(
                _action(
                    kind="BUILD_EMERGENCY_FUND",
                    sleeve="debt",
                    instrument_category=instant_pick.instrument,
                    amount=emergency_amount,
                    reason_code="emergency_fund_gap",
                )
            )
            remaining = round_paise(remaining - emergency_amount)
        else:
            carry_forward += remaining
            remaining = Decimal("0.00")
            emergency_pick_unavailable = True

    positive_gaps = {
        sleeve: max(Decimal(0), _gap_value(week_input.plan.gap.get(sleeve))) for sleeve in _SLEEVES
    }
    if remaining > 0 and any(gap > 0 for gap in positive_gaps.values()):
        shares, unapplied = _proportional_shares(
            remaining, positive_gaps, increment, min_action=min_action
        )
        carry_forward += unapplied
        remaining = Decimal("0.00")
    else:
        shares = {sleeve: Decimal("0.00") for sleeve in _SLEEVES}
        carry_forward += remaining
        remaining = Decimal("0.00")

    investment_actions: list[Action] = []
    for sleeve in _SLEEVES:
        amount = shares[sleeve]
        if amount <= 0:
            continue
        if sleeve == "equity_india":
            india_actions, unused = _india_actions(
                amount, week_input, selected_rules, increment, min_action
            )
            investment_actions.extend(india_actions)
            carry_forward += unused
            continue
        action = _sleeve_action(sleeve, amount, week_input, selected_rules, increment, min_action)
        if action is None:
            carry_forward += amount
        else:
            investment_actions.append(action)

    sip_action = _sip_action(week_input, positive_gaps, selected_rules, increment, min_action)
    maximum_actions = min(3, integer_rule(selected_rules["max_actions"], "max_actions"))
    if apply_weekly_cap:
        actions, dropped = _cap_actions(
            foundation_actions, sip_action, investment_actions, maximum_actions
        )
    else:
        actions = [*foundation_actions]
        if sip_action is not None:
            actions.append(sip_action)
        actions.extend(sorted(investment_actions, key=lambda action: action.amount, reverse=True))
        dropped = Decimal("0.00")
    carry_forward += dropped
    notes: list[str] = []
    if week_input.plan.new_money_only:
        notes.append("tilt_applied_via_new_money")
    if emergency_pick_unavailable:
        notes.append("emergency_pick_unavailable")
    if not actions:
        notes.append("on_plan")
    ranked = [action.model_copy(update={"rank": rank}) for rank, action in enumerate(actions, 1)]
    return ActionList(
        week=week_input.week,
        status="ACTIONS" if ranked else "NONE",
        actions=ranked,
        carry_forward=round_paise(carry_forward),
        notes=notes,
    )


def _action(
    *,
    kind: ActionKind,
    sleeve: str,
    instrument_category: str,
    amount: Decimal,
    reason_code: str,
    route: str | None = None,
    security_id: str | None = None,
    quantity: Decimal | None = None,
    tranche: dict[str, str] | None = None,
    params: dict[str, ParamsValue] | None = None,
) -> Action:
    return Action(
        rank=1,
        kind=kind,
        sleeve=sleeve,
        instrument_category=instrument_category,
        route=route,
        security_id=security_id,
        quantity=quantity,
        amount=round_paise(amount),
        tranche=tranche,
        reason_code=reason_code,
        params={} if params is None else params,
    )


def _gap_value(value: Decimal | None) -> Decimal:
    if value is None:
        return Decimal(0)
    return max(Decimal(0), round_paise(value))


def _top_instant_debt_pick(debt_picks: list[DebtPick]) -> DebtPick | None:
    if not debt_picks:
        return None
    liquidity_by_key = {item["key"]: item["liquidity"] for item in load_debt_rules()["instruments"]}
    return next(
        (pick for pick in debt_picks if liquidity_by_key.get(pick.instrument) == "instant"), None
    )


def _proportional_shares(
    amount: Decimal,
    gaps: Mapping[PlanActionSleeve, Decimal],
    increment: Decimal,
    *,
    min_action: Decimal | None = None,
) -> tuple[dict[PlanActionSleeve, Decimal], Decimal]:
    active = {sleeve: gap for sleeve, gap in gaps.items() if gap > 0}
    shares = {sleeve: Decimal("0.00") for sleeve in _SLEEVES}
    if amount <= 0 or not active:
        return shares, round_paise(amount)
    total_gap = sum(active.values(), Decimal(0))
    largest = max(_SLEEVES, key=lambda sleeve: active.get(sleeve, Decimal(0)))
    for sleeve, gap in active.items():
        shares[sleeve] = _floor_to_increment(amount * gap / total_gap, increment)
    shares[largest] += amount - sum(shares.values(), Decimal(0))
    if min_action is not None:
        for sleeve in _SLEEVES:
            if sleeve != largest and Decimal(0) < shares[sleeve] < min_action:
                shares[largest] += shares[sleeve]
                shares[sleeve] = Decimal("0.00")
        if Decimal(0) < shares[largest] < min_action:
            unapplied = shares[largest]
            shares[largest] = Decimal("0.00")
            return shares, round_paise(unapplied)
    return shares, Decimal("0.00")


def _floor_to_increment(amount: Decimal, increment: Decimal) -> Decimal:
    units = (amount / increment).to_integral_value(rounding=ROUND_FLOOR)
    return units * increment


def _sleeve_action(
    sleeve: PlanActionSleeve,
    amount: Decimal,
    week_input: WeekInput,
    rules: BuyingRules,
    increment: Decimal,
    min_action: Decimal,
) -> Action | None:
    if amount < min_action:
        return None
    if sleeve == "equity_global":
        if week_input.route.route == "WAIT" or week_input.route.vehicle == "no_open_route":
            return None
        category = week_input.route.vehicle
        route = week_input.route.route
        market: Market | None = "US"
    elif sleeve == "debt":
        if not week_input.debt_picks:
            return None
        category = week_input.debt_picks[0].instrument
        route = None
        market = None
    elif sleeve == "gold":
        category = "gold_etf"
        route = None
        market = None
    else:
        category = "sweep_fd"
        route = None
        market = None
    return _buy_action(
        sleeve,
        category,
        amount,
        week_input,
        rules,
        increment,
        market=market,
        route=route,
    )


def _buy_action(
    sleeve: PlanActionSleeve,
    category: str,
    amount: Decimal,
    week_input: WeekInput,
    rules: BuyingRules,
    increment: Decimal,
    *,
    market: Market | None,
    route: str | None = None,
) -> Action:
    lumpsum_rules = rules["lumpsum"]
    threshold_multiple = decimal_rule(
        lumpsum_rules["stp_threshold_multiple_of_monthly_sip"],
        "lumpsum.stp_threshold_multiple_of_monthly_sip",
    )
    threshold = threshold_multiple * week_input.monthly_sip_capacity
    valuation = (
        week_input.index_valuation_percentile.get(market, Decimal(0))
        if market is not None
        else Decimal(0)
    )
    tranches = integer_rule(lumpsum_rules["stp_tranches_normal"], "lumpsum.stp_tranches_normal")
    if valuation >= decimal_rule(
        lumpsum_rules["expensive_valuation_percentile"],
        "lumpsum.expensive_valuation_percentile",
    ):
        tranches = integer_rule(
            lumpsum_rules["stp_tranches_expensive"], "lumpsum.stp_tranches_expensive"
        )
    if amount > threshold:
        tranche_amount = _floor_to_increment(amount / Decimal(tranches), increment)
        if tranche_amount > 0:
            return _action(
                kind="BUY_STP",
                sleeve=sleeve,
                instrument_category=category,
                amount=amount,
                route=route,
                reason_code="target_gap",
                tranche={"n": "1", "of": str(tranches)},
                params={
                    "tranches": str(tranches),
                    "tranche_amount": _money_string(tranche_amount),
                    "valuation_percentile": _fixed_decimal(valuation),
                },
            )
    return _action(
        kind="BUY_LUMPSUM",
        sleeve=sleeve,
        instrument_category=category,
        amount=amount,
        route=route,
        reason_code="target_gap",
        params={} if market is None else {"market": market},
    )


def _india_actions(
    india_alloc: Decimal,
    week_input: WeekInput,
    rules: BuyingRules,
    increment: Decimal,
    min_action: Decimal,
) -> tuple[list[Action], Decimal]:
    satellite = rules["satellite"]
    india_target = week_input.plan.target_rupees.get("equity_india", Decimal(0))
    global_target = week_input.plan.target_rupees.get("equity_global", Decimal(0))
    total_equity_target = india_target + global_target
    satellite_target = india_target * week_input.satellite_pct / Decimal(100)
    held_satellite = sum((item.value for item in week_input.satellite_holdings), Decimal(0))
    satellite_gap = max(Decimal(0), satellite_target - held_satellite)
    max_stock = (
        total_equity_target
        * decimal_rule(
            satellite["max_stock_pct_of_total_equity"],
            "satellite.max_stock_pct_of_total_equity",
        )
        / Decimal(100)
    )
    tranche_amount = round_paise(
        max_stock / integer_rule(satellite["tranches"], "satellite.tranches")
    )
    proposed = min(india_alloc, tranche_amount, satellite_gap)
    max_sector = (
        satellite_target
        * decimal_rule(
            satellite["max_sector_pct_of_satellite"], "satellite.max_sector_pct_of_satellite"
        )
        / Decimal(100)
    )
    security_values: dict[str, Decimal] = {}
    sector_values: dict[str, Decimal] = {}
    for holding in week_input.satellite_holdings:
        security_values[holding.security_id] = (
            security_values.get(holding.security_id, Decimal(0)) + holding.value
        )
        sector_values[holding.sector] = (
            sector_values.get(holding.sector, Decimal(0)) + holding.value
        )

    confidence_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    eligible: list[tuple[tuple[int, int, Decimal, str], SatelliteIdea, Decimal, Decimal]] = []
    if proposed > 0 and total_equity_target > 0:
        for idea in week_input.ideas:
            if (
                idea.market != "IN"
                or idea.dossier_status != "APPROVED"
                or idea.quant_decile
                < integer_rule(satellite["min_quant_decile"], "satellite.min_quant_decile")
                or idea.valuation_percentile
                >= decimal_rule(
                    satellite["max_valuation_percentile"],
                    "satellite.max_valuation_percentile",
                )
            ):
                continue
            stock_room = max_stock - security_values.get(idea.security_id, Decimal(0))
            sector_room = max_sector - sector_values.get(idea.sector, Decimal(0))
            allowed = min(proposed, max(Decimal(0), stock_room), max(Decimal(0), sector_room))
            quantity = (allowed / idea.price).to_integral_value(rounding=ROUND_FLOOR)
            amount = round_paise(quantity * idea.price)
            if amount <= 0:
                continue
            key = (
                -idea.quant_decile,
                confidence_rank[idea.confidence],
                idea.valuation_percentile,
                idea.security_id,
            )
            eligible.append((key, idea, quantity, amount))

    actions: list[Action] = []
    stock_amount = Decimal(0)
    if eligible:
        _, idea, quantity, stock_amount = min(eligible, key=lambda item: item[0])
        actions.append(
            _action(
                kind="BUY_STOCK_TRANCHE",
                sleeve="equity_india",
                instrument_category="stock",
                amount=stock_amount,
                route=None,
                security_id=idea.security_id,
                quantity=quantity,
                tranche={
                    "n": "1",
                    "of": str(integer_rule(satellite["tranches"], "satellite.tranches")),
                },
                reason_code="approved_satellite_idea",
                params={"price": _money_string(idea.price)},
            )
        )

    core_before_rounding = india_alloc - stock_amount
    core_amount = _floor_to_increment(core_before_rounding, increment)
    unused = round_paise(india_alloc - stock_amount - core_amount)
    if core_amount >= min_action:
        actions.append(
            _buy_action(
                "equity_india",
                week_input.india_core_vehicle,
                core_amount,
                week_input,
                rules,
                increment,
                market="IN",
            )
        )
    else:
        unused += core_amount
    return actions, round_paise(unused)


def _sip_action(
    week_input: WeekInput,
    gaps: Mapping[PlanActionSleeve, Decimal],
    rules: BuyingRules,
    increment: Decimal,
    min_action: Decimal,
) -> Action | None:
    monthly_by_sleeve = {sleeve: Decimal(0) for sleeve in _SLEEVES}
    for sip in week_input.existing_sips:
        monthly_by_sleeve[sip.sleeve] += sip.monthly_amount
    total_existing = sum(monthly_by_sleeve.values(), Decimal(0))
    overweight_total = sum(
        (amount for sleeve, amount in monthly_by_sleeve.items() if gaps[sleeve] <= 0),
        Decimal(0),
    )
    pool = max(
        Decimal(0),
        round_paise(overweight_total + week_input.monthly_sip_capacity - total_existing),
    )
    underweight = {sleeve: gap for sleeve, gap in gaps.items() if gap > 0}
    if pool < min_action or not underweight:
        return None
    allocations, _ = _proportional_shares(pool, underweight, increment)
    changes: list[dict[str, str]] = []
    change_order = [sleeve for sleeve in _SLEEVES if gaps[sleeve] <= 0] + [
        sleeve for sleeve in _SLEEVES if gaps[sleeve] > 0
    ]
    for sleeve in change_order:
        old_amount = monthly_by_sleeve[sleeve]
        new_amount = Decimal(0) if gaps[sleeve] <= 0 else old_amount + allocations[sleeve]
        if old_amount != new_amount:
            changes.append(
                {
                    "sleeve": sleeve,
                    "from": _money_string(old_amount),
                    "to": _money_string(new_amount),
                }
            )
    if not changes:
        return None
    return _action(
        kind="REBALANCE_SIPS",
        sleeve="multiple",
        instrument_category="monthly_sip",
        amount=pool,
        reason_code="sip_capacity_redirected",
        params={"changes": changes},
    )


def _cap_actions(
    foundation_actions: list[Action],
    sip_action: Action | None,
    investment_actions: list[Action],
    maximum_actions: int,
) -> tuple[list[Action], Decimal]:
    selected: list[Action] = []
    selected.extend(foundation_actions[:maximum_actions])
    if sip_action is not None and len(selected) < maximum_actions:
        selected.append(sip_action)
    remaining_slots = maximum_actions - len(selected)
    ordered_buys = sorted(investment_actions, key=lambda action: action.amount, reverse=True)
    selected.extend(ordered_buys[:remaining_slots])
    selected_ids = {id(action) for action in selected}
    dropped = sum(
        (
            action.amount
            for action in [*foundation_actions, *investment_actions]
            if id(action) not in selected_ids
        ),
        Decimal(0),
    )
    return selected, round_paise(dropped)
