"""Deterministic satellite exit cards and tax-aware sale sizing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_FLOOR, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import yaml

from wealth.core.money import round_paise
from wealth.policy.tax import (
    Lot,
    TaxContext,
    TaxEstimate,
    days_to_long_term,
    estimate_sale_tax,
    select_lots,
)

TriggerState = Literal["ok", "watch", "triggered", "unknown"]
ExitAction = Literal[
    "KEEP",
    "TRIM_TO_TARGET",
    "EXIT",
    "REVIEW_THEN_EXIT",
    "SWAP",
    "WAIT_FOR_LONG_TERM",
    "GLIDE_TO_DEBT",
]
Urgency = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ThesisStatus = Literal["on_track", "watch", "broken", "unknown"]

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "exits.yaml"
_ZERO = Decimal("0")


class BetterCandidate(TypedDict):
    """A same-sector candidate evaluated by the deterministic research pipeline."""

    security_id: str
    z_advantage: Decimal
    weeks_ahead: int
    expected_edge_pct: Decimal


@dataclass(frozen=True)
class HoldingFacts:
    """Inputs available for one holding's exit rules; missing metrics stay explicit."""

    security_id: str
    symbol: str
    sector: str
    is_core: bool
    lots: list[Lot]
    price: Decimal
    weight_pct_of_equity: Decimal | None
    target_weight_pct: Decimal | None
    revenue_yoy_pct_last_quarters: list[Decimal] = field(default_factory=list)
    roce_pct_last_quarters: list[Decimal] = field(default_factory=list)
    guidance_cut_pct: Decimal | None = None
    governance_events: list[str] = field(default_factory=list)
    pledge_pct: Decimal | None = None
    pledge_pct_prev_quarter: Decimal | None = None
    pe_own_band_percentile: Decimal | None = None
    pe_vs_sector_multiple: Decimal | None = None
    below_200dma: bool | None = None
    thesis_status: ThesisStatus = "unknown"
    geo_revenue_hit_pct: Decimal | None = None
    geo_evidence_ids: list[str] = field(default_factory=list)
    better_candidate: BetterCandidate | None = None
    goal_within_years: int | None = None

    def __post_init__(self) -> None:
        for name in ("security_id", "symbol", "sector"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty text")
        if not isinstance(self.is_core, bool):
            raise ValueError("is_core must be a bool")
        if not isinstance(self.lots, list) or any(not isinstance(lot, Lot) for lot in self.lots):
            raise ValueError("lots must be a list of Lot values")
        _decimal(self.price, "price", minimum=_ZERO)
        for name in (
            "weight_pct_of_equity",
            "target_weight_pct",
            "guidance_cut_pct",
            "pledge_pct",
            "pledge_pct_prev_quarter",
            "pe_own_band_percentile",
            "pe_vs_sector_multiple",
            "geo_revenue_hit_pct",
        ):
            value = getattr(self, name)
            if value is not None:
                _decimal(value, name, minimum=_ZERO)
        for name in ("revenue_yoy_pct_last_quarters", "roce_pct_last_quarters"):
            values = getattr(self, name)
            if not isinstance(values, list):
                raise ValueError(f"{name} must be a list")
            for value in values:
                _decimal(value, name)
        if not isinstance(self.governance_events, list) or any(
            not isinstance(event, str) or not event for event in self.governance_events
        ):
            raise ValueError("governance_events must contain non-empty event codes")
        if not isinstance(self.geo_evidence_ids, list) or any(
            not isinstance(evidence_id, str) or not evidence_id
            for evidence_id in self.geo_evidence_ids
        ):
            raise ValueError("geo_evidence_ids must contain non-empty ids")
        if self.thesis_status not in {"on_track", "watch", "broken", "unknown"}:
            raise ValueError("unsupported thesis_status")
        if self.goal_within_years is not None and (
            isinstance(self.goal_within_years, bool)
            or not isinstance(self.goal_within_years, int)
            or self.goal_within_years < 0
        ):
            raise ValueError("goal_within_years must be a non-negative integer or None")
        if self.better_candidate is not None:
            candidate = self.better_candidate
            if not isinstance(candidate, dict):
                raise ValueError("better_candidate must be a mapping")
            if not isinstance(candidate.get("security_id"), str) or not candidate["security_id"]:
                raise ValueError("better_candidate.security_id must be non-empty text")
            _decimal(candidate.get("z_advantage"), "better_candidate.z_advantage")
            _decimal(candidate.get("expected_edge_pct"), "better_candidate.expected_edge_pct")
            weeks = candidate.get("weeks_ahead")
            if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 0:
                raise ValueError("better_candidate.weeks_ahead must be a non-negative integer")


@dataclass(frozen=True)
class TriggerStatus:
    key: str
    status: TriggerState
    value: str | None
    threshold: str


@dataclass(frozen=True)
class ExitCard:
    security_id: str
    symbol: str
    action: ExitAction
    urgency: Urgency
    triggers: dict[str, TriggerStatus]
    quantity: Decimal
    lots_to_sell: list[tuple[Lot, Decimal]]
    tax_estimate: TaxEstimate
    reason_codes: tuple[str, ...]
    params: dict[str, date | Decimal | str] = field(default_factory=dict)
    member_id: str = "self"
    member_name: str = "You"
    member_relationship: str = "self"

    @property
    def trigger_statuses(self) -> dict[str, TriggerStatus]:
        """Alias for clients that prefer the longer response field name."""
        return self.triggers


def _decimal(value: object, name: str, *, minimum: Decimal | None = None) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@lru_cache(maxsize=1)
def _exit_rules() -> dict[str, object]:
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        loaded = yaml.safe_load(rules_file)
    if not isinstance(loaded, dict) or loaded.get("version") != 1:
        raise ValueError("unsupported exit rules")
    return loaded


def load_exit_rules() -> dict[str, object]:
    """Return a caller-owned copy of the checked-in exit rulebook."""
    return deepcopy(_exit_rules())


def _decimal_rule(rules: dict[str, object], path: tuple[str, ...]) -> Decimal:
    value: object = rules
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing exit rule: {'.'.join(path)}")
        value = value[part]
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"invalid exit rule: {'.'.join(path)}") from error
    if not result.is_finite():
        raise ValueError(f"invalid exit rule: {'.'.join(path)}")
    return result


def _integer_rule(rules: dict[str, object], path: tuple[str, ...]) -> int:
    value: object = rules
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing exit rule: {'.'.join(path)}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid exit rule: {'.'.join(path)}")
    return value


def _text(value: Decimal | int | None) -> str | None:
    return None if value is None else format(value, "f")


def _trigger(key: str, state: TriggerState, value: str | None, threshold: str) -> TriggerStatus:
    return TriggerStatus(key=key, status=state, value=value, threshold=threshold)


def _recent_negative_quarters(values: list[Decimal]) -> int:
    count = 0
    for value in values:
        if value >= 0:
            break
        count += 1
    return count


def _thesis_trigger(facts: HoldingFacts, rules: dict[str, object]) -> TriggerStatus:
    base = ("satellite", "thesis_break")
    revenue_needed = _integer_rule(rules, (*base, "revenue_yoy_negative_quarters"))
    roce_floor = _decimal_rule(rules, (*base, "roce_floor_pct"))
    roce_needed = _integer_rule(rules, (*base, "roce_floor_quarters"))
    guidance_limit = _decimal_rule(rules, (*base, "guidance_cut_pct"))
    watch = _decimal_rule(rules, ("watch_fraction",))
    evidence: list[str] = []
    triggered = facts.thesis_status == "broken"
    watching = facts.thesis_status == "watch"
    known = facts.thesis_status in {"on_track", "watch", "broken"}

    if facts.revenue_yoy_pct_last_quarters:
        known = True
        count = _recent_negative_quarters(facts.revenue_yoy_pct_last_quarters)
        evidence.append(f"revenue negative streak {count}")
        triggered = triggered or count >= revenue_needed
        watching = watching or 0 < count < revenue_needed
    if facts.roce_pct_last_quarters:
        known = True
        low_streak = 0
        for value in facts.roce_pct_last_quarters:
            if value < roce_floor:
                low_streak += 1
            else:
                break
        evidence.append(f"ROCE below floor for {low_streak} recent quarters")
        triggered = triggered or low_streak >= roce_needed
        watching = watching or (
            0 < low_streak < roce_needed
            and (
                low_streak >= (roce_needed * watch).to_integral_value(rounding=ROUND_FLOOR)
                or facts.roce_pct_last_quarters[0] <= roce_floor * watch
            )
        )
    if facts.guidance_cut_pct is not None:
        known = True
        evidence.append(f"guidance cut {facts.guidance_cut_pct}%")
        triggered = triggered or facts.guidance_cut_pct > guidance_limit
        watching = watching or facts.guidance_cut_pct >= guidance_limit * watch

    threshold = (
        f"{revenue_needed} recent revenue declines; ROCE <{roce_floor}% for "
        f"{roce_needed} quarters; guidance cut >{guidance_limit}%"
    )
    value = "; ".join(evidence) or facts.thesis_status
    if triggered:
        return _trigger("thesis_break", "triggered", value, threshold)
    if watching:
        return _trigger("thesis_break", "watch", value, threshold)
    return _trigger(
        "thesis_break", "ok" if known else "unknown", value if known else None, threshold
    )


def _governance_trigger(facts: HoldingFacts, rules: dict[str, object]) -> TriggerStatus:
    base = ("satellite", "governance_break")
    event_rule = rules["satellite"]["governance_break"]["events"]  # type: ignore[index]
    if not isinstance(event_rule, list):
        raise ValueError("governance event rule must be a list")
    jump_limit = _decimal_rule(rules, (*base, "pledge_jump_pp"))
    absolute_limit = _decimal_rule(rules, (*base, "pledge_abs_pct"))
    watch = _decimal_rule(rules, ("watch_fraction",))
    events = [event for event in facts.governance_events if event in event_rule]
    jump = (
        facts.pledge_pct - facts.pledge_pct_prev_quarter
        if facts.pledge_pct is not None and facts.pledge_pct_prev_quarter is not None
        else None
    )
    known = bool(facts.governance_events) or facts.pledge_pct is not None
    triggered = (
        bool(events)
        or (jump is not None and jump >= jump_limit)
        or (facts.pledge_pct is not None and facts.pledge_pct > absolute_limit)
    )
    watching = (jump is not None and jump >= jump_limit * watch) or (
        facts.pledge_pct is not None and facts.pledge_pct >= absolute_limit * watch
    )
    values = [f"event {event}" for event in events]
    if facts.pledge_pct is not None:
        values.append(f"pledge {facts.pledge_pct}%")
    if jump is not None:
        values.append(f"pledge change {jump} pp")
    threshold = (
        f"event in {', '.join(str(event) for event in event_rule)}; pledge increase "
        f">={jump_limit} pp or pledge >{absolute_limit}%"
    )
    value = "; ".join(values) or None
    if triggered:
        return _trigger("governance_break", "triggered", value, threshold)
    if watching:
        return _trigger("governance_break", "watch", value, threshold)
    return _trigger("governance_break", "ok" if known else "unknown", value, threshold)


def _geopolitical_trigger(facts: HoldingFacts, rules: dict[str, object]) -> TriggerStatus:
    base = ("satellite", "geopolitical_break")
    limit = _decimal_rule(rules, (*base, "addressable_revenue_hit_pct"))
    source_count = _integer_rule(rules, (*base, "min_independent_sources"))
    watch = _decimal_rule(rules, ("watch_fraction",))
    evidence_count = len(set(facts.geo_evidence_ids))
    known = facts.geo_revenue_hit_pct is not None
    value = (
        f"{facts.geo_revenue_hit_pct}% hit; {evidence_count} independent evidence ids"
        if known
        else None
    )
    threshold = f"revenue hit >{limit}% with at least {source_count} independent evidence ids"
    if not known:
        state: TriggerState = "unknown"
    elif facts.geo_revenue_hit_pct > limit and evidence_count >= source_count:
        state = "triggered"
    elif evidence_count == 0:
        state = "unknown"
    elif facts.geo_revenue_hit_pct >= limit * watch:
        state = "watch"
    else:
        state = "ok"
    return _trigger("geopolitical_break", state, value, threshold)


def _valuation_trigger(facts: HoldingFacts, rules: dict[str, object]) -> TriggerStatus:
    base = ("satellite", "valuation_stretch")
    percentile_limit = _decimal_rule(rules, (*base, "own_band_percentile"))
    multiple_limit = _decimal_rule(rules, (*base, "sector_multiple"))
    watch = _decimal_rule(rules, ("watch_fraction",))
    known = facts.pe_own_band_percentile is not None and facts.pe_vs_sector_multiple is not None
    value = (
        f"{facts.pe_own_band_percentile} percentile; {facts.pe_vs_sector_multiple}x sector"
        if known
        else None
    )
    threshold = f"own PE percentile >{percentile_limit} and sector multiple >{multiple_limit}x"
    if not known:
        state: TriggerState = "unknown"
    elif (
        facts.pe_own_band_percentile > percentile_limit
        and facts.pe_vs_sector_multiple > multiple_limit
    ):
        state = "triggered"
    elif (
        facts.pe_own_band_percentile >= percentile_limit * watch
        and facts.pe_vs_sector_multiple >= multiple_limit * watch
    ):
        state = "watch"
    else:
        state = "ok"
    return _trigger("valuation_stretch", state, value, threshold)


def _drift_trigger(facts: HoldingFacts, rules: dict[str, object]) -> TriggerStatus:
    base = ("satellite", "drift")
    weight_limit = _decimal_rule(rules, (*base, "max_weight_of_equity_pct"))
    ratio_limit = _decimal_rule(rules, (*base, "max_vs_target_ratio"))
    watch = _decimal_rule(rules, ("watch_fraction",))
    weight = facts.weight_pct_of_equity
    target = facts.target_weight_pct
    ratio = weight / target if weight is not None and target is not None and target > 0 else None
    exceeds_weight = weight is not None and weight > weight_limit
    exceeds_ratio = ratio is not None and ratio > ratio_limit
    reaches_watch = (weight is not None and weight >= weight_limit * watch) or (
        ratio is not None and ratio >= ratio_limit * watch
    )
    if exceeds_weight or exceeds_ratio:
        state: TriggerState = "triggered"
    elif reaches_watch:
        state = "watch"
    elif weight is None or target is None:
        state = "unknown"
    else:
        state = "ok"
    value_parts = []
    if weight is not None:
        value_parts.append(f"{weight}% of equity")
    if ratio is not None:
        value_parts.append(f"{ratio}x target")
    threshold = f"weight >{weight_limit}% of equity or >{ratio_limit}x target"
    return _trigger("drift", state, "; ".join(value_parts) or None, threshold)


def _position_drawdown(facts: HoldingFacts) -> Decimal | None:
    quantity = sum((lot.quantity for lot in facts.lots), _ZERO)
    basis_total = sum((lot.quantity * lot.cost_per_unit for lot in facts.lots), _ZERO)
    if quantity <= 0 or basis_total <= 0:
        return None
    average_cost = basis_total / quantity
    return (facts.price - average_cost) / average_cost * Decimal(100)


def _risk_trigger(facts: HoldingFacts, rules: dict[str, object]) -> TriggerStatus:
    base = ("satellite", "risk_stop")
    drawdown_limit = _decimal_rule(rules, (*base, "drawdown_from_cost_pct"))
    dma_period = _integer_rule(rules, (*base, "below_dma"))
    required_thesis = str(rules["satellite"]["risk_stop"]["requires_thesis"]).lower()  # type: ignore[index]
    drawdown = _position_drawdown(facts)
    known = (
        drawdown is not None and facts.below_200dma is not None and facts.thesis_status != "unknown"
    )
    value = (
        f"{drawdown}% from cost; below {dma_period}-DMA={facts.below_200dma}; "
        f"thesis={facts.thesis_status}"
        if drawdown is not None
        else None
    )
    threshold = (
        f"drawdown <=-{drawdown_limit}% and below {dma_period}-DMA and thesis {required_thesis}"
    )
    if not known:
        state: TriggerState = "unknown"
    elif (
        drawdown <= -drawdown_limit
        and facts.below_200dma
        and facts.thesis_status == required_thesis
    ):
        state = "triggered"
    elif (
        drawdown <= -(drawdown_limit * _decimal_rule(rules, ("watch_fraction",)))
        and facts.below_200dma
        and facts.thesis_status == required_thesis
    ):
        state = "watch"
    else:
        state = "ok"
    return _trigger("risk_stop", state, value, threshold)


def _goal_trigger(facts: HoldingFacts, rules: dict[str, object]) -> TriggerStatus:
    years = _integer_rule(rules, ("satellite", "goal_approach", "within_years"))
    value = _text(facts.goal_within_years)
    threshold = f"goal within {years} years"
    if facts.goal_within_years is None:
        state: TriggerState = "unknown"
    elif facts.goal_within_years <= years:
        state = "triggered"
    elif facts.goal_within_years <= years / _decimal_rule(rules, ("watch_fraction",)):
        state = "watch"
    else:
        state = "ok"
    return _trigger("goal_approach", state, value, threshold)


def _tax_wait_status(
    facts: HoldingFacts,
    rules: dict[str, object],
    *,
    days: int | None = None,
    saved: Decimal | None = None,
) -> TriggerStatus:
    max_days = _integer_rule(rules, ("satellite", "tax_wait", "max_days_to_long_term"))
    threshold = f"within {max_days} days of long-term with positive tax savings"
    if days is None:
        if not facts.lots:
            return _trigger("tax_wait", "unknown", None, threshold)
        return _trigger("tax_wait", "ok", None, threshold)
    value = f"{days} days; tax saved {saved}" if saved is not None else f"{days} days"
    if saved is not None and saved > 0:
        return _trigger("tax_wait", "triggered", value, threshold)
    return _trigger("tax_wait", "ok", value, threshold)


def _empty_tax_estimate(today: date, tax_ctx: TaxContext) -> TaxEstimate:
    return estimate_sale_tax([], today, tax_ctx)


def _sale_size(facts: HoldingFacts, action: ExitAction) -> Decimal:
    total_quantity = sum((lot.quantity for lot in facts.lots), _ZERO)
    if action in {"EXIT", "REVIEW_THEN_EXIT", "SWAP"}:
        return total_quantity
    if action != "TRIM_TO_TARGET":
        return _ZERO
    weight = facts.weight_pct_of_equity
    target = facts.target_weight_pct
    position_value = round_paise(total_quantity * facts.price)
    if weight is None or target is None or weight <= 0 or target >= weight or facts.price <= 0:
        return _ZERO
    value_to_sell = position_value * (weight - target) / weight
    return (value_to_sell / facts.price).to_integral_value(rounding=ROUND_FLOOR)


def _sell_lots(
    facts: HoldingFacts, quantity: Decimal, today: date, tax_ctx: TaxContext
) -> list[tuple[Lot, Decimal]]:
    if quantity <= 0 or not facts.lots:
        return []
    available = sum((lot.quantity for lot in facts.lots), _ZERO)
    if quantity > available:
        quantity = available
    return select_lots(facts.lots, quantity, facts.price, today, tax_ctx)


def _estimate_selected(
    selected: list[tuple[Lot, Decimal]], sale_date: date, price: Decimal, tax_ctx: TaxContext
) -> TaxEstimate:
    sales = [(lot, quantity, price) for lot, quantity in selected]
    return estimate_sale_tax(sales, sale_date, tax_ctx)


def _wait_opportunity(
    selected: list[tuple[Lot, Decimal]],
    today: date,
    price: Decimal,
    tax_ctx: TaxContext,
    max_days: int,
) -> tuple[date, Decimal, TaxEstimate] | None:
    if not selected:
        return None
    current_tax = _estimate_selected(selected, today, price, tax_ctx)
    candidate_days = sorted(
        {days for lot, _ in selected if 0 < (days := days_to_long_term(lot, today)) <= max_days}
    )
    for days in candidate_days:
        sale_date = today + timedelta(days=days)
        future_tax = _estimate_selected(selected, sale_date, price, tax_ctx)
        tax_saved = round_paise(current_tax.total_tax - future_tax.total_tax)
        if tax_saved > 0:
            return sale_date, tax_saved, future_tax
    return None


def _swap_qualified(
    facts: HoldingFacts,
    value: Decimal,
    tax_cost: Decimal,
) -> tuple[bool, bool]:
    candidate = facts.better_candidate
    if candidate is None:
        return False, False
    min_z = Decimal("1.0")
    min_weeks = 4
    threshold_met = candidate["z_advantage"] > min_z and candidate["weeks_ahead"] >= min_weeks
    edge_value = round_paise(value * candidate["expected_edge_pct"] / Decimal(100))
    if not threshold_met:
        return False, False
    if edge_value > tax_cost:
        return True, False
    return False, True


def build_exit_card(
    facts: HoldingFacts, today: date, tax_ctx: TaxContext, rules: dict[str, object] | None = None
) -> ExitCard:
    """Evaluate one holding's triggers, choose the highest-priority action, and size its sale."""
    if not isinstance(facts, HoldingFacts):
        raise ValueError("facts must be HoldingFacts")
    if not isinstance(today, date):
        raise ValueError("today must be a date")
    if not isinstance(tax_ctx, TaxContext):
        raise ValueError("tax_ctx must be a TaxContext")
    selected_rules = load_exit_rules() if rules is None else deepcopy(rules)
    if not isinstance(selected_rules, dict):
        raise ValueError("rules must be an exit rules mapping")
    if facts.is_core:
        triggers = {
            "thesis_break": _trigger("thesis_break", "ok", None, "not applicable to core"),
            "governance_break": _trigger("governance_break", "ok", None, "not applicable to core"),
            "geopolitical_break": _trigger(
                "geopolitical_break", "ok", None, "not applicable to core"
            ),
            "valuation_stretch": _trigger(
                "valuation_stretch", "ok", None, "not applicable to core"
            ),
            "drift": _drift_trigger(facts, selected_rules),
            "risk_stop": _trigger("risk_stop", "ok", None, "no stop-loss for core"),
            "tax_wait": _trigger("tax_wait", "ok", None, "no core exit timing trigger"),
            "goal_approach": _trigger("goal_approach", "ok", None, "handled by whole-wealth plan"),
        }
    else:
        triggers = {
            "thesis_break": _thesis_trigger(facts, selected_rules),
            "governance_break": _governance_trigger(facts, selected_rules),
            "geopolitical_break": _geopolitical_trigger(facts, selected_rules),
            "valuation_stretch": _valuation_trigger(facts, selected_rules),
            "drift": _drift_trigger(facts, selected_rules),
            "risk_stop": _risk_trigger(facts, selected_rules),
            "tax_wait": _tax_wait_status(facts, selected_rules),
            "goal_approach": _goal_trigger(facts, selected_rules),
        }

    governance_fired = triggers["governance_break"].status == "triggered"
    thesis_fired = triggers["thesis_break"].status == "triggered"
    geopolitical_fired = triggers["geopolitical_break"].status == "triggered"
    risk_fired = triggers["risk_stop"].status == "triggered"
    valuation_fired = triggers["valuation_stretch"].status == "triggered"
    drift_fired = triggers["drift"].status == "triggered"
    reasons: list[str] = []
    if governance_fired:
        action: ExitAction = "EXIT"
        urgency: Urgency = "CRITICAL"
        reasons.append("governance_break")
    elif thesis_fired:
        action = "EXIT"
        urgency = "HIGH"
        reasons.append("thesis_break")
    elif geopolitical_fired:
        action = "REVIEW_THEN_EXIT"
        urgency = "MEDIUM"
        reasons.append("geopolitical_break")
    elif risk_fired:
        action = "REVIEW_THEN_EXIT"
        urgency = "MEDIUM"
        reasons.append("risk_stop")
    elif valuation_fired:
        action = "TRIM_TO_TARGET"
        urgency = "LOW"
        reasons.append("valuation_stretch")
    elif drift_fired:
        action = "TRIM_TO_TARGET"
        urgency = "LOW"
        reasons.append("drift")
    else:
        action = "KEEP"
        urgency = "LOW"

    quantity = _sale_size(facts, action)
    lots_to_sell = _sell_lots(facts, quantity, today, tax_ctx)
    tax_estimate = (
        _estimate_selected(lots_to_sell, today, facts.price, tax_ctx)
        if lots_to_sell
        else _empty_tax_estimate(today, tax_ctx)
    )
    params: dict[str, date | Decimal | str] = {}

    if action == "KEEP" and not facts.is_core and facts.better_candidate is not None:
        value = round_paise(sum((lot.quantity for lot in facts.lots), _ZERO) * facts.price)
        candidate_quantity = _sale_size(facts, "SWAP")
        candidate_lots = _sell_lots(facts, candidate_quantity, today, tax_ctx)
        candidate_tax = (
            _estimate_selected(candidate_lots, today, facts.price, tax_ctx)
            if candidate_lots
            else _empty_tax_estimate(today, tax_ctx)
        )
        should_swap, not_worth_tax = _swap_qualified(facts, value, candidate_tax.total_tax)
        if should_swap:
            action = "SWAP"
            urgency = "MEDIUM"
            reasons.append("opportunity_cost_swap")
            quantity = candidate_quantity
            lots_to_sell = candidate_lots
            tax_estimate = candidate_tax
        elif not_worth_tax:
            reasons.append("swap_not_worth_tax")
            tax_estimate = candidate_tax
        else:
            candidate = facts.better_candidate
            if candidate["z_advantage"] <= Decimal("1.0") or candidate["weeks_ahead"] < 4:
                reasons.append("swap_threshold_not_met")

    if action == "KEEP" and not facts.is_core and triggers["goal_approach"].status == "triggered":
        action = "GLIDE_TO_DEBT"
        urgency = "LOW"
        reasons.append("goal_approach")

    wait_rules = selected_rules["satellite"]["tax_wait"]  # type: ignore[index]
    wait_families = wait_rules["applies_to"]
    can_wait = (
        not facts.is_core
        and action in {"TRIM_TO_TARGET", "SWAP"}
        and any(
            reason in {"valuation_stretch", "drift", "opportunity_cost_swap"} for reason in reasons
        )
        and isinstance(wait_families, list)
    ) or (
        facts.is_core
        and action == "TRIM_TO_TARGET"
        and drift_fired
        and isinstance(wait_families, list)
    )
    if can_wait:
        max_days = _integer_rule(selected_rules, ("satellite", "tax_wait", "max_days_to_long_term"))
        opportunity = _wait_opportunity(lots_to_sell, today, facts.price, tax_ctx, max_days)
        if opportunity is not None:
            wait_until, tax_saved, future_estimate = opportunity
            days = (wait_until - today).days
            action = "WAIT_FOR_LONG_TERM"
            urgency = "LOW"
            tax_estimate = future_estimate
            params.update({"wait_until": wait_until, "tax_saved": tax_saved})
            reasons.append("tax_wait")
            triggers["tax_wait"] = _tax_wait_status(
                facts, selected_rules, days=days, saved=tax_saved
            )

    if action == "KEEP" and triggers["goal_approach"].status == "triggered" and facts.is_core:
        reasons.append("core_goal_glide_deferred")

    return ExitCard(
        security_id=facts.security_id,
        symbol=facts.symbol,
        action=action,
        urgency=urgency,
        triggers=triggers,
        quantity=quantity,
        lots_to_sell=lots_to_sell,
        tax_estimate=tax_estimate,
        reason_codes=tuple(reasons),
        params=params,
    )
