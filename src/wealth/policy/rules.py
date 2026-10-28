"""Load and read the versioned deterministic allocation rules."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

AllocationRules = Mapping[str, Any]
_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "allocation.yaml"


def load_allocation_rules() -> dict[str, Any]:
    """Read the checked-in allocation and foundation rules."""
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        document = yaml.safe_load(rules_file)
    if not isinstance(document, dict):
        raise ValueError("allocation rules must contain a YAML mapping")
    validate_allocation_rules(document)
    return document


def resolve_rules(rules: AllocationRules | None = None) -> AllocationRules:
    """Use the supplied rule mapping or load the checked-in version."""
    if rules is None:
        return load_allocation_rules()
    if not isinstance(rules, Mapping):
        raise TypeError("rules must be a mapping")
    return rules


def decimal_rule(value: object, name: str) -> Decimal:
    """Parse a rule number without accepting binary floating-point values."""
    if isinstance(value, (float, bool)):
        raise ValueError(f"{name} must be an integer, string, or Decimal")
    try:
        result = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid decimal number") from error
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def integer_rule(value: object, name: str) -> int:
    """Read a whole-number rule and reject booleans."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def validate_allocation_rules(rules: Mapping[str, Any]) -> None:
    """Check the core shape and ensure every target mix totals 100%."""
    if integer_rule(rules.get("version"), "version") != 1:
        raise ValueError("unsupported allocation rules version")
    scale = decimal_rule(rules.get("percent_scale"), "percent_scale")
    if scale <= 0:
        raise ValueError("percent_scale must be positive")

    profiles = rules.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("profiles must be a mapping")
    for name, mix in profiles.items():
        _validate_mix(mix, f"profiles.{name}", scale)

    horizon = rules.get("horizon")
    if not isinstance(horizon, Mapping):
        raise ValueError("horizon must be a mapping")
    _validate_mix(horizon.get("short_mix"), "horizon.short_mix", scale)

    allocation = rules.get("allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError("allocation must be a mapping")
    _validate_mix(allocation.get("emergency_mix"), "allocation.emergency_mix", scale)

    risk = rules.get("risk")
    if not isinstance(risk, Mapping) or not isinstance(risk.get("questions"), list):
        raise ValueError("risk.questions must be a list")
    question_count = integer_rule(risk.get("question_count"), "risk.question_count")
    options_per_question = integer_rule(
        risk.get("options_per_question"), "risk.options_per_question"
    )
    allowed_scores = risk.get("allowed_option_scores")
    if not isinstance(allowed_scores, list) or not allowed_scores:
        raise ValueError("risk.allowed_option_scores must be a non-empty list")
    allowed_scores = [
        integer_rule(score, "risk.allowed_option_scores[]") for score in allowed_scores
    ]
    if len(risk["questions"]) != question_count:
        raise ValueError(f"risk.questions must contain {question_count} questions")
    bands = risk.get("bands")
    if not isinstance(bands, Mapping) or not bands:
        raise ValueError("risk.bands must be a non-empty mapping")
    question_ids: set[str] = set()
    for question in risk["questions"]:
        if not isinstance(question, Mapping):
            raise ValueError("each risk question must be a mapping")
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id or question_id in question_ids:
            raise ValueError("risk question ids must be non-empty and unique")
        question_ids.add(question_id)
        options = question.get("options")
        if not isinstance(options, list) or len(options) != options_per_question:
            raise ValueError(
                f"risk question {question_id!r} must have {options_per_question} options"
            )
        option_ids: set[str] = set()
        for option in options:
            if not isinstance(option, Mapping):
                raise ValueError("each risk option must be a mapping")
            option_id = option.get("id")
            if not isinstance(option_id, str) or not option_id or option_id in option_ids:
                raise ValueError(f"risk option ids for {question_id!r} must be unique")
            option_ids.add(option_id)
            score = integer_rule(option.get("score"), f"{question_id}.{option_id}.score")
            if score not in allowed_scores:
                raise ValueError("risk option score is not in risk.allowed_option_scores")


def _validate_mix(value: object, name: str, scale: Decimal) -> None:
    sleeves = ("equity", "debt", "gold", "cash")
    if not isinstance(value, Mapping) or set(value) != set(sleeves):
        raise ValueError(f"{name} must define equity, debt, gold, and cash")
    total = sum((decimal_rule(value[key], f"{name}.{key}") for key in sleeves), Decimal(0))
    if total != scale:
        raise ValueError(f"{name} must sum to {scale}")
