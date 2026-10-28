"""Allocation bands and an ordered cash, SIP, and sell rebalancing plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from wealth.core.money import round_paise
from wealth.policy.actions import Action, WeekInput, build_action_list
from wealth.policy.tax import Lot, TaxContext, TaxEstimate, estimate_sale_tax, select_lots

Direction = Literal["over", "under"]
_ABSOLUTE_BAND_PP = Decimal("5")
_RELATIVE_BAND_PCT = Decimal("25")


@dataclass(frozen=True)
class BandRules:
    """The absolute percentage-point and relative-percent trigger thresholds."""

    absolute_pp: Decimal = _ABSOLUTE_BAND_PP
    relative_pct: Decimal = _RELATIVE_BAND_PCT

    def __post_init__(self) -> None:
        for name, value in (
            ("absolute_pp", self.absolute_pp),
            ("relative_pct", self.relative_pct),
        ):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite non-negative Decimal")


@dataclass(frozen=True)
class Breach:
    """One sleeve outside its permitted allocation band."""

    sleeve: str
    actual_pct: Decimal
    target_pct: Decimal
    direction: Direction
    excess_amount: Decimal


@dataclass(frozen=True)
class PricedLot:
    """A tax lot paired with its caller-supplied current sale price."""

    lot: Lot
    sale_price: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.lot, Lot):
            raise ValueError("lot must be a Lot")
        if (
            not isinstance(self.sale_price, Decimal)
            or not self.sale_price.is_finite()
            or self.sale_price < 0
        ):
            raise ValueError("sale_price must be a finite non-negative Decimal")


@dataclass(frozen=True)
class RebalanceSale:
    """One overweight sleeve sale with its selected lots and attached tax estimate."""

    sleeve: str
    lot_sales: tuple[tuple[Lot, Decimal, Decimal], ...]
    gross_proceeds: Decimal
    tax_estimate: TaxEstimate


@dataclass(frozen=True)
class RebalancePlan:
    """Ordered operations: C12b cash actions, one SIP redirect, then capped sells."""

    cash_actions: tuple[Action, ...]
    sip_redirect: Action | None
    sells: tuple[RebalanceSale, ...]
    cash_carry_forward: Decimal
    tax_estimate: TaxEstimate | None = None
    notes: tuple[str, ...] = ()

    @property
    def sales(self) -> tuple[RebalanceSale, ...]:
        """Alias for the sell phase."""
        return self.sells


def _selected_rules(rules: BandRules | Mapping[str, Decimal] | None) -> BandRules:
    if rules is None:
        return BandRules()
    if isinstance(rules, BandRules):
        return rules
    if not isinstance(rules, Mapping):
        raise ValueError("rules must be BandRules or a mapping")
    absolute = rules.get("absolute_pp", rules.get("absolute_band_pp", _ABSOLUTE_BAND_PP))
    relative = rules.get("relative_pct", rules.get("relative_band_pct", _RELATIVE_BAND_PCT))
    return BandRules(absolute_pp=absolute, relative_pct=relative)


def _band_width(target_pct: Decimal, rules: BandRules) -> Decimal:
    if target_pct > 0:
        return min(rules.absolute_pp, target_pct * rules.relative_pct / Decimal(100))
    return rules.absolute_pp


def _plan_percentage(
    supplied: Mapping[str, Decimal], amounts: Mapping[str, Decimal], sleeve: str, total: Decimal
) -> Decimal:
    if sleeve in supplied:
        value = supplied[sleeve]
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(f"allocation percentage for {sleeve} must be a finite Decimal")
        return value
    amount = amounts.get(sleeve, Decimal(0))
    return amount * Decimal(100) / total if total > 0 else Decimal(0)


def band_breaches(
    plan: Any, rules: BandRules | Mapping[str, Decimal] | None = None
) -> list[Breach]:
    """Return sleeves outside the 5/25 band, with rupees beyond the nearest band edge."""
    selected_rules = _selected_rules(rules)
    total = plan.investable_total
    if not isinstance(total, Decimal) or not total.is_finite() or total < 0:
        raise ValueError("plan.investable_total must be a finite non-negative Decimal")
    targets = plan.target_percent
    actuals = plan.current_percent
    target_amounts = plan.target_rupees
    actual_amounts = plan.current_rupees
    sleeves = tuple(dict.fromkeys((*targets.keys(), *actuals.keys())))
    breaches: list[Breach] = []
    for sleeve in sleeves:
        target_pct = _plan_percentage(targets, target_amounts, sleeve, total)
        actual_pct = _plan_percentage(actuals, actual_amounts, sleeve, total)
        deviation = actual_pct - target_pct
        width = _band_width(target_pct, selected_rules)
        if abs(deviation) < width:
            continue
        direction: Direction = "over" if deviation > 0 else "under"
        excess_amount = round_paise((abs(deviation) - width) * total / Decimal(100))
        breaches.append(
            Breach(
                sleeve=sleeve,
                actual_pct=actual_pct,
                target_pct=target_pct,
                direction=direction,
                excess_amount=max(Decimal("0.00"), excess_amount),
            )
        )
    return breaches


def _gap(item: object) -> Decimal:
    value = getattr(item, "gap", None)
    if not isinstance(value, Decimal) or not value.is_finite():
        return Decimal(0)
    return max(Decimal(0), value)


def _priced_lots(value: object) -> tuple[list[PricedLot], bool]:
    """Accept explicit PricedLot/pairs or Lot.market_price supplied by the caller."""
    if isinstance(value, Mapping) and "lots" in value and "sale_price" in value:
        lots = value["lots"]
        price = value["sale_price"]
        if isinstance(lots, Sequence) and isinstance(price, Decimal):
            return [PricedLot(lot, price) for lot in lots], False
        return [], True
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], True
    result: list[PricedLot] = []
    missing_price = False
    for item in value:
        if isinstance(item, PricedLot):
            result.append(item)
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], Lot)
            and isinstance(item[1], Decimal)
        ):
            result.append(PricedLot(item[0], item[1]))
        elif isinstance(item, Lot) and item.market_price is not None:
            result.append(PricedLot(item, item.market_price))
        elif isinstance(item, Lot):
            missing_price = True
        else:
            raise ValueError("lots_by_sleeve values must contain priced Lot values")
    return result, missing_price


def _sale_priority(
    priced: PricedLot, sale_date: date, tax_ctx: TaxContext
) -> tuple[int, Decimal, str]:
    sample_quantity = min(Decimal(1), priced.lot.quantity)
    line = estimate_sale_tax(
        [(priced.lot, sample_quantity, priced.sale_price)], sale_date, tax_ctx
    ).lines[0]
    gain_per_unit = line.gain / sample_quantity
    if gain_per_unit < 0:
        return (0, gain_per_unit, priced.lot.lot_id)
    if line.holding_period == "LT":
        return (1, gain_per_unit, priced.lot.lot_id)
    return (2, gain_per_unit, priced.lot.lot_id)


def _sale_for_budget(
    sleeve: str,
    priced_lots: Sequence[PricedLot],
    budget: Decimal,
    sale_date: date,
    tax_ctx: TaxContext,
) -> RebalanceSale | None:
    remaining = round_paise(budget)
    selections: list[tuple[Lot, Decimal, Decimal]] = []
    ordered = sorted(priced_lots, key=lambda item: _sale_priority(item, sale_date, tax_ctx))
    for priced in ordered:
        if remaining <= 0:
            break
        quantity_to_sell = (
            min(priced.lot.quantity, remaining / priced.sale_price)
            if priced.sale_price
            else Decimal(0)
        )
        if quantity_to_sell <= 0:
            continue
        selected = select_lots(
            [priced.lot], quantity_to_sell, priced.sale_price, sale_date, tax_ctx
        )
        for lot, quantity in selected:
            proceeds = round_paise(quantity * priced.sale_price)
            if proceeds > remaining:
                continue
            selections.append((lot, quantity, priced.sale_price))
            remaining = round_paise(remaining - proceeds)

    if not selections:
        return None
    sales = [(lot, quantity, price) for lot, quantity, price in selections]
    estimate = estimate_sale_tax(sales, sale_date, tax_ctx)
    return RebalanceSale(
        sleeve=sleeve,
        lot_sales=tuple(selections),
        gross_proceeds=estimate.gross_proceeds,
        tax_estimate=estimate,
    )


def rebalance_plan(
    breaches: Sequence[Breach],
    week_input: WeekInput,
    lots_by_sleeve: Mapping[str, object],
    sale_date: date,
    tax_ctx: TaxContext,
) -> RebalancePlan:
    """Spend cash into breached underweights, redirect SIPs, then sell only remaining overweights.

    ``lots_by_sleeve`` accepts ``PricedLot`` entries, ``(Lot, sale_price)`` pairs, or bare
    lots with ``market_price`` set. Sale prices are caller-provided; this module never fetches them.
    """
    if not isinstance(week_input, WeekInput):
        raise ValueError("week_input must be a WeekInput")
    if not isinstance(tax_ctx, TaxContext):
        raise ValueError("tax_ctx must be a TaxContext")
    if not isinstance(sale_date, date):
        raise ValueError("sale_date must be a date")

    if any(
        not isinstance(breach.excess_amount, Decimal)
        or not breach.excess_amount.is_finite()
        or breach.excess_amount < 0
        for breach in breaches
    ):
        raise ValueError("breach excess_amount values must be finite non-negative Decimals")
    underweight_rooms = {
        breach.sleeve: breach.excess_amount
        for breach in breaches
        if breach.direction == "under" and breach.excess_amount > 0
    }
    overweights = [breach for breach in breaches if breach.direction == "over"]

    gaps = {sleeve: Decimal(0) for sleeve in week_input.plan.gap}
    for sleeve, room in underweight_rooms.items():
        if sleeve in gaps:
            gaps[sleeve] = room
    selected_plan = replace(week_input.plan, gap=gaps)
    foundation_need = _gap(week_input.foundation["high_interest_debt"]) + _gap(
        week_input.foundation["emergency_fund"]
    )
    cash_room = sum(underweight_rooms.values(), Decimal(0)) + foundation_need
    cash_budget = min(week_input.new_money, cash_room)
    selected_week = week_input.model_copy(update={"plan": selected_plan, "new_money": cash_budget})
    allocation = build_action_list(selected_week)
    cash_actions = tuple(action for action in allocation.actions if action.kind != "REBALANCE_SIPS")
    sip_redirect = next(
        (action for action in allocation.actions if action.kind == "REBALANCE_SIPS"), None
    )

    purchases: dict[str, Decimal] = {}
    portfolio_addition = Decimal(0)
    for action in cash_actions:
        if action.kind == "PAY_DOWN_DEBT":
            continue
        purchases[action.sleeve] = purchases.get(action.sleeve, Decimal(0)) + action.amount
        portfolio_addition += action.amount
    updated_total = round_paise(week_input.plan.investable_total + portfolio_addition)

    sales: list[RebalanceSale] = []
    notes = list(allocation.notes)
    for breach in overweights:
        target = breach.target_pct
        current_amount = week_input.plan.current_rupees.get(breach.sleeve, Decimal(0))
        updated_amount = current_amount + purchases.get(breach.sleeve, Decimal(0))
        if updated_total <= 0:
            continue
        actual_pct = updated_amount * Decimal(100) / updated_total
        width = _band_width(target, BandRules())
        if actual_pct - target < width:
            continue
        edge_amount = round_paise(updated_amount - (target + width) * updated_total / Decimal(100))
        if edge_amount <= 0:
            continue
        priced, missing_price = _priced_lots(lots_by_sleeve.get(breach.sleeve, ()))
        if not priced:
            if missing_price:
                notes.append(f"sale_price_missing:{breach.sleeve}")
            continue
        sale = _sale_for_budget(breach.sleeve, priced, edge_amount, sale_date, tax_ctx)
        if sale is not None:
            sales.append(sale)

    if sales:
        notes = [note for note in notes if note != "on_plan"]

    aggregate_tax: TaxEstimate | None = None
    if sales:
        all_sale_lines = [
            (lot, quantity, price) for sale in sales for lot, quantity, price in sale.lot_sales
        ]
        aggregate_tax = estimate_sale_tax(all_sale_lines, sale_date, tax_ctx)
        sales = [replace(sale, tax_estimate=aggregate_tax) for sale in sales]

    return RebalancePlan(
        cash_actions=cash_actions,
        sip_redirect=sip_redirect,
        sells=tuple(sales),
        cash_carry_forward=allocation.carry_forward,
        tax_estimate=aggregate_tax,
        notes=tuple(dict.fromkeys(notes)),
    )
