"""Pure verdict rules for a direct share already in the household."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from wealth.policy.exits import ExitCard, HoldingFacts, load_exit_rules

HoldingVerdict = Literal["sell", "review", "sell_some", "add_more", "keep", "not_checked"]
_ZERO = Decimal(0)
_REASON_ORDER = (
    "governance_break",
    "thesis_break",
    "geopolitical_break",
    "risk_stop",
    "valuation_stretch",
    "drift",
    "goal_approach",
)


@dataclass(frozen=True)
class StockCheckFacts:
    security_id: str
    week: str
    as_of: date
    market: str
    quant_decile: int | None = None
    factor_coverage: str | None = None
    pe_percentile: Decimal | None = None
    pe_vs_sector: Decimal | None = None
    revenue_yoy_pct: tuple[Decimal, ...] = ()
    price: Decimal | None = None
    dma200: Decimal | None = None
    below_dma200: bool | None = None
    events_watched: bool = False


@dataclass(frozen=True)
class HoldingReviewDecision:
    verdict: HoldingVerdict
    reasons: tuple[dict[str, Any], ...]
    checks: tuple[dict[str, str], ...]


def thesis_status_from_revenue(values: tuple[Decimal, ...] | list[Decimal]) -> str:
    """Map recent revenue evidence to the exit engine's code-owned thesis status."""
    if not values:
        return "unknown"
    negative_quarters = 0
    for value in values:
        if value >= 0:
            break
        negative_quarters += 1
    required = int(load_exit_rules()["satellite"]["thesis_break"]["revenue_yoy_negative_quarters"])
    if negative_quarters >= required:
        return "broken"
    if negative_quarters:
        return "watch"
    return "on_track"


def price_is_reasonable(check: StockCheckFacts) -> bool:
    """Use own-history valuation when present, otherwise the sector comparison."""
    accumulation = load_exit_rules()["accumulate"]
    maximum_pe = Decimal(str(accumulation["max_pe_percentile"]))
    maximum_sector_multiple = Decimal(str(accumulation["max_pe_vs_sector"]))
    if check.pe_percentile is not None:
        return check.pe_percentile < maximum_pe
    if check.pe_vs_sector is not None:
        return check.pe_vs_sector <= maximum_sector_multiple
    return False


def holding_review_decision(
    *,
    market: str,
    as_of: date,
    check: StockCheckFacts | None,
    card: ExitCard | None,
    facts: HoldingFacts | None,
    add_more_room: Decimal | None,
    minimum_buy: Decimal,
    position_size_known: bool = False,
) -> HoldingReviewDecision:
    """Choose the plain verdict and bounded reason/check rows from deterministic facts."""
    if market not in {"IN", "US"} or check is None or card is None or facts is None:
        return HoldingReviewDecision(
            "not_checked", (), _price_only_checks(check, position_size_known)
        )
    maximum_age = int(load_exit_rules()["checks_max_age_days"])
    age = (as_of - check.as_of).days
    if age < 0 or age > maximum_age:
        return HoldingReviewDecision(
            "not_checked", (), _price_only_checks(check, position_size_known)
        )

    triggered = any(item.status == "triggered" for item in card.triggers.values())
    watching = any(item.status == "watch" for item in card.triggers.values())
    trim_triggered = any(
        card.triggers[key].status == "triggered" for key in ("valuation_stretch", "drift")
    )
    accumulation = load_exit_rules()["accumulate"]
    minimum_decile = int(accumulation["min_quant_decile"])
    weak_decile_max = int(accumulation["weak_quant_decile_max"])
    fresh = age <= maximum_age
    enough_room = add_more_room is not None and add_more_room >= minimum_buy
    can_add_more = (
        not triggered
        and not watching
        and check.quant_decile is not None
        and check.quant_decile >= minimum_decile
        and price_is_reasonable(check)
        and fresh
        and enough_room
    )

    if card.action == "EXIT":
        verdict: HoldingVerdict = "sell"
    elif card.action == "REVIEW_THEN_EXIT":
        verdict = "review"
    elif (
        card.action == "TRIM_TO_TARGET"
        and trim_triggered
        or card.action == "WAIT_FOR_LONG_TERM"
        and trim_triggered
    ):
        verdict = "sell_some"
    elif can_add_more:
        verdict = "add_more"
    else:
        verdict = "keep"

    reasons = _reason_rows(verdict, card, facts, check)
    checks = tuple(
        {"key": key, "status": card.triggers[trigger].status}
        for key, trigger in (
            ("company_results", "thesis_break"),
            ("company_events", "governance_break"),
            ("price_history", "valuation_stretch"),
            ("position_size", "drift"),
            ("price_and_cost", "risk_stop"),
        )
    )
    research_status = (
        "unknown"
        if check.quant_decile is None
        else "ok"
        if check.quant_decile >= minimum_decile or check.quant_decile > weak_decile_max
        else "watch"
    )
    checks += ({"key": "company_strength", "status": research_status},)
    return HoldingReviewDecision(verdict, reasons, checks)


def _price_only_checks(
    check: StockCheckFacts | None, position_size_known: bool
) -> tuple[dict[str, str], ...]:
    return (
        {
            "key": "latest_price",
            "status": "ok" if check is not None and check.price is not None else "unknown",
        },
        {"key": "position_size", "status": "ok" if position_size_known else "unknown"},
    )


def _reason_rows(
    verdict: HoldingVerdict,
    card: ExitCard,
    facts: HoldingFacts,
    check: StockCheckFacts,
) -> tuple[dict[str, Any], ...]:
    if verdict == "add_more":
        return ({"code": "add_more", "params": {}},)
    weak_decile_max = int(load_exit_rules()["accumulate"]["weak_quant_decile_max"])
    statuses = {key: card.triggers[key].status for key in _REASON_ORDER}
    selected = [key for key in _REASON_ORDER if statuses[key] in {"triggered", "watch"}]
    weak_company = check.quant_decile is not None and check.quant_decile <= weak_decile_max
    if not selected and verdict == "keep" and not weak_company:
        return ()
    rows: list[dict[str, Any]] = []
    for trigger in selected:
        params: dict[str, str | int | bool] = {}
        if trigger == "thesis_break":
            negative = 0
            for value in check.revenue_yoy_pct:
                if value >= 0:
                    break
                negative += 1
            code = "revenue_down_quarters" if negative else trigger
            if negative:
                params["count"] = negative
        elif (
            trigger == "valuation_stretch"
            and check.pe_percentile is None
            and check.pe_vs_sector is not None
        ):
            code = "valuation_vs_sector"
        else:
            code = trigger
        if trigger == "drift" and facts.weight_pct_of_equity is not None:
            params["share_of_equity_pct"] = _plain_percent(facts.weight_pct_of_equity)
        if trigger == "valuation_stretch":
            if check.pe_percentile is not None:
                params["pe_percentile"] = format(check.pe_percentile, "f")
            if check.pe_vs_sector is not None:
                params["pe_vs_sector"] = format(check.pe_vs_sector, "f")
        if trigger == "risk_stop":
            drawdown = _drawdown_pct(facts)
            if drawdown is not None:
                params["drawdown_pct"] = _plain_percent(drawdown)
            if facts.below_200dma is not None:
                params["below_dma200"] = facts.below_200dma
        rows.append({"code": code, "params": params})
        if len(rows) == 3:
            break

    if weak_company and len(rows) < 3:
        rows.append({"code": "company_strength_weak", "params": {}})

    if card.params.get("wait_until") is not None and len(rows) < 3:
        rows.append(
            {
                "code": "tax_wait",
                "params": {"long_term_on": card.params["wait_until"].isoformat()},
            }
        )
    if not rows and card.reason_codes:
        rows.append({"code": card.reason_codes[0], "params": {}})
    return tuple(rows[:3])


def _drawdown_pct(facts: HoldingFacts) -> Decimal | None:
    quantity = sum((lot.quantity for lot in facts.lots), _ZERO)
    basis = sum((lot.quantity * lot.cost_per_unit for lot in facts.lots), _ZERO)
    if quantity <= 0 or basis <= 0:
        return None
    average_cost = basis / quantity
    return (facts.price - average_cost) / average_cost * Decimal(100)


def _plain_percent(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), "f")
