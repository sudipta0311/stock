"""Summarise debt targets and rank where additional money is underweight."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Literal

from wealth.policy.allocation import AllocationPlan
from wealth.policy.rules import decimal_rule, integer_rule, resolve_rules
from wealth.policy.snapshot import POLICY_SLEEVES, PolicySleeve, WealthSnapshot
from wealth.schemas.common import Sleeve


@dataclass(frozen=True)
class DebtAnswer:
    """Structured debt target answer for presentation without generated prose."""

    target_debt: Decimal
    current_debt: Decimal
    locked_debt: Decimal
    gap: Decimal
    status: Literal["enough", "add"]
    new_money_order: tuple[PolicySleeve, ...]
    message_key: str
    params: Mapping[str, str]


def debt_answer(plan: AllocationPlan, snapshot: WealthSnapshot) -> DebtAnswer:
    """Explain the debt target, existing debt share, and underweight order."""
    target_debt = plan.target_rupees["debt"]
    current_debt = plan.current_rupees["debt"]
    locked_debt = snapshot.locked_by_sleeve[Sleeve.DEBT]
    gap = target_debt - current_debt
    enough_ratio = decimal_rule(
        resolve_rules()["debt_answer"]["enough_ratio"], "debt_answer.enough_ratio"
    )
    status = "enough" if current_debt >= target_debt * enough_ratio else "add"
    order = tuple(
        sorted(
            (sleeve for sleeve in POLICY_SLEEVES if plan.gap[sleeve] > 0),
            key=lambda sleeve: (-plan.gap[sleeve], POLICY_SLEEVES.index(sleeve)),
        )
    )
    rules = resolve_rules()
    display_places = integer_rule(
        rules["debt_answer"]["percent_display_places"],
        "debt_answer.percent_display_places",
    )
    params = MappingProxyType(
        {
            "current_pct": _format_percent(plan.current_percent["debt"], display_places),
            "target_pct": _format_percent(plan.target_percent["debt"], display_places),
        }
    )
    return DebtAnswer(
        target_debt=target_debt,
        current_debt=current_debt,
        locked_debt=locked_debt,
        gap=gap,
        status=status,
        new_money_order=order,
        message_key="debt_target_met" if status == "enough" else "debt_target_gap",
        params=params,
    )


def _format_percent(value: Decimal, places: int) -> str:
    quantum = Decimal(1).scaleb(-places)
    return format(value.quantize(quantum, rounding=ROUND_HALF_UP), f".{places}f")
