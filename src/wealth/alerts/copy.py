"""Load and format the repo-wide English copy map used by email and the PWA."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from wealth.core.money import format_inr

_COPY_PATH = Path(__file__).resolve().parents[3] / "copy" / "en.yaml"
_FALLBACK = "This follows the rules in your plan."


@lru_cache(maxsize=1)
def load_copy() -> dict[str, Any]:
    """Read the single checked-in English copy map."""
    with _COPY_PATH.open(encoding="utf-8") as copy_file:
        document = yaml.safe_load(copy_file)
    if not isinstance(document, dict):
        raise ValueError("copy/en.yaml must contain a mapping")
    for section in ("reasons", "messages", "labels", "email"):
        if not isinstance(document.get(section), dict):
            raise ValueError(f"copy/en.yaml is missing the {section} mapping")
    return document


def _parameter_value(name: str, value: Any) -> str:
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, (int, str)) and not isinstance(value, bool):
        try:
            decimal_value = Decimal(value)
        except InvalidOperation:
            return str(value)
    else:
        return str(value)

    if name in {"amount", "gap", "value", "current", "target", "difference"} or name.endswith(
        ("_amount", "_inr")
    ):
        return format_inr(decimal_value, style="compact")
    return format(decimal_value, "f")


def format_reason(code: str, params: Mapping[str, Any] | None = None) -> str:
    """Render a reason code using the shared map, with no raw-code fallback."""
    template = load_copy()["reasons"].get(code)
    if not isinstance(template, str):
        return _FALLBACK
    values = {name: _parameter_value(name, value) for name, value in (params or {}).items()}
    try:
        return template.format_map(values)
    except (KeyError, ValueError):
        return _FALLBACK


def format_message(code: str, params: Mapping[str, Any] | None = None) -> str:
    """Render a policy message key from the shared map."""
    template = load_copy()["messages"].get(code)
    if not isinstance(template, str):
        return _FALLBACK
    values = {name: _parameter_value(name, value) for name, value in (params or {}).items()}
    try:
        return template.format_map(values)
    except (KeyError, ValueError):
        return _FALLBACK
