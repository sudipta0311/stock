"""Deterministic rupee impacts for labelled portfolio stress assumptions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import yaml

from wealth.core.money import round_paise

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "stress.yaml"
_SLEEVES = ("equity_india", "equity_global", "debt", "gold", "cash")
_ZERO = Decimal("0")


@lru_cache(maxsize=1)
def _stress_rules() -> dict[str, object]:
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        loaded = yaml.safe_load(rules_file)
    if not isinstance(loaded, dict) or loaded.get("version") != 1:
        raise ValueError("unsupported stress rules")
    return loaded


def load_stress_rules() -> dict[str, object]:
    """Return a caller-owned copy of the checked-in stress scenario assumptions."""
    return deepcopy(_stress_rules())


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    return value


def _scenario_rules(scenario: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(scenario, str):
        scenarios = _stress_rules().get("scenarios")
        if not isinstance(scenarios, dict) or scenario not in scenarios:
            raise ValueError(f"unknown stress scenario: {scenario}")
        selected = scenarios[scenario]
        if not isinstance(selected, dict):
            raise ValueError("stress scenario must be a mapping")
        return selected
    if not isinstance(scenario, Mapping):
        raise ValueError("scenario must be a key or a mapping")
    return scenario


def _scenario_decimal(scenario: Mapping[str, object], key: str) -> Decimal:
    raw = scenario.get(key)
    try:
        value = Decimal(str(raw))
    except Exception as error:
        raise ValueError(f"scenario.{key} must be a decimal fraction") from error
    if not value.is_finite():
        raise ValueError(f"scenario.{key} must be finite")
    return value


def _amounts(amounts_by_sleeve: Mapping[str, Decimal]) -> dict[str, Decimal]:
    if not isinstance(amounts_by_sleeve, Mapping):
        raise ValueError("amounts_by_sleeve must be a mapping")
    result = {sleeve: _ZERO for sleeve in _SLEEVES}
    for key, raw in amounts_by_sleeve.items():
        normalized = "equity_global" if key == "equity_global_usd" else key
        if normalized not in _SLEEVES:
            raise ValueError(f"unsupported sleeve: {key}")
        value = _decimal(raw, f"amounts_by_sleeve.{key}")
        if value < 0:
            raise ValueError(f"amounts_by_sleeve.{key} cannot be negative")
        result[normalized] = value
    return result


def stress_test(
    amounts_by_sleeve: Mapping[str, Decimal], scenario: str | Mapping[str, object]
) -> dict[str, object]:
    """Return per-sleeve and total gain/loss under one assumed scenario."""
    amounts = _amounts(amounts_by_sleeve)
    selected = _scenario_rules(scenario)
    india_return = _scenario_decimal(selected, "equity_india")
    global_usd_return = _scenario_decimal(selected, "equity_global_usd")
    fx_change = _scenario_decimal(selected, "fx")
    global_return = (Decimal(1) + global_usd_return) * (Decimal(1) + fx_change) - Decimal(1)
    rates = {
        "equity_india": india_return,
        "equity_global": global_return,
        "debt": _scenario_decimal(selected, "debt"),
        "gold": _scenario_decimal(selected, "gold"),
        "cash": _ZERO,
    }
    by_sleeve = {sleeve: round_paise(amounts[sleeve] * rates[sleeve]) for sleeve in _SLEEVES}
    total = round_paise(sum(by_sleeve.values(), _ZERO))
    return {"by_sleeve": by_sleeve, "total": total}


def stress_compare(
    current: Mapping[str, Decimal],
    target: Mapping[str, Decimal],
    scenario: str | Mapping[str, object],
) -> dict[str, Decimal]:
    """Compare current and target plan impacts; positive means the target loses less."""
    current_total = stress_test(current, scenario)["total"]
    target_total = stress_test(target, scenario)["total"]
    assert isinstance(current_total, Decimal) and isinstance(target_total, Decimal)
    return {
        "current_total": current_total,
        "target_total": target_total,
        "difference": round_paise(target_total - current_total),
    }
