"""Choose a few useful, deterministic highlights for each app screen."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from wealth.core.money import format_inr

HighlightScreen = Literal["home", "plan", "ideas", "money"]
HighlightSeverity = Literal["info", "attention"]

_SCREEN_LIMITS: dict[HighlightScreen, int] = {
    "home": 1,
    "plan": 3,
    "ideas": 2,
    "money": 3,
}
_ALL_SCREENS = ("home", "plan", "ideas", "money")


class Highlight(BaseModel):
    """Small, plain-language item suitable for a screen summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    screen: HighlightScreen
    title: str
    body: str
    learn_more_key: str
    severity: HighlightSeverity
    context: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class HighlightContext:
    """Screen-specific copy and the login's current dismissal fingerprints."""

    screen: HighlightScreen
    copy: Mapping[str, Any]
    dismissed_fingerprints: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Candidate:
    key: str
    copy_key: str
    amount: Decimal | None
    position: int
    rate_pct: Decimal | None = None
    instrument_code: str | None = None


def _document(plan: Any) -> Mapping[str, Any]:
    if isinstance(plan, BaseModel):
        value = plan.model_dump(mode="python")
    elif isinstance(plan, Mapping):
        value = plan
    else:
        raise TypeError("plan must be a mapping or Pydantic model")
    return value


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value)
        except Exception:
            return None
    return None


def _positive_amount(value: Any) -> Decimal | None:
    amount = _decimal(value)
    return amount if amount is not None and amount > 0 else None


def _safe_key(value: Any) -> str:
    raw = str(value or "other_detail").strip().casefold()
    normalized = "".join(character if character.isalnum() else "_" for character in raw)
    normalized = "_".join(part for part in normalized.split("_") if part)
    if normalized and len(normalized) <= 100:
        return normalized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"detail_{digest}"


def _instrument_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.casefold()
    return code if re.fullmatch(r"[a-z][a-z0-9_]{0,59}", code) else None


def _note_parts(value: Any) -> tuple[str, Decimal | None, Decimal | None] | None:
    if isinstance(value, str):
        return _safe_key(value), None, None
    if not isinstance(value, Mapping):
        return None
    key = value.get("key", value.get("code", value.get("reason")))
    if not isinstance(key, str):
        return None
    amount = next(
        (_positive_amount(value.get(field)) for field in ("amount", "gap", "value")),
        None,
    )
    rate_pct = _decimal(value.get("rate_pct"))
    return _safe_key(key), amount, rate_pct


def _candidate_rows(plan: Mapping[str, Any]) -> list[_Candidate]:
    rows: dict[str, _Candidate] = {}
    position = 0

    def add(
        key: str,
        copy_key: str | None = None,
        amount: Decimal | None = None,
        rate_pct: Decimal | None = None,
        instrument_code: str | None = None,
    ) -> None:
        nonlocal position
        candidate_key = _safe_key(key)
        existing = rows.get(candidate_key)
        resolved_copy_key = _safe_key(copy_key or key)
        if existing is None:
            rows[candidate_key] = _Candidate(
                candidate_key, resolved_copy_key, amount, position, rate_pct, instrument_code
            )
            position += 1
        elif (
            (existing.amount is None and amount is not None)
            or (existing.rate_pct is None and rate_pct is not None)
            or (existing.instrument_code is None and instrument_code is not None)
        ):
            rows[candidate_key] = _Candidate(
                candidate_key,
                existing.copy_key,
                amount if amount is not None else existing.amount,
                existing.position,
                rate_pct if rate_pct is not None else existing.rate_pct,
                instrument_code if existing.instrument_code is None else existing.instrument_code,
            )

    for field_name in ("notes", "residency_notes", "placement_notes"):
        raw_values = plan.get(field_name, ())
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
            continue
        for raw_value in raw_values:
            parsed = _note_parts(raw_value)
            if parsed is not None:
                key, amount, rate_pct = parsed
                add(key, amount=amount, rate_pct=rate_pct)

    for raw_item in plan.get("foundation", ()):
        if not isinstance(raw_item, Mapping):
            continue
        amount = _positive_amount(raw_item.get("gap"))
        item_key = raw_item.get("key")
        if amount is not None and isinstance(item_key, str):
            code = _safe_key(item_key)
            add(f"{code}_gap", amount=amount)

    route = plan.get("route", {})
    warning_groups = [plan.get("warnings", ()), plan.get("band_breaches", ())]
    if isinstance(route, Mapping):
        warning_groups.append(route.get("warnings", ()))
    for raw_group in warning_groups:
        if not isinstance(raw_group, Sequence) or isinstance(raw_group, str):
            continue
        for raw_warning in raw_group:
            parsed = _note_parts(raw_warning)
            if parsed is None:
                continue
            key, amount, rate_pct = parsed
            add(key, amount=amount, rate_pct=rate_pct)

    for raw_option in plan.get("unranked_options", ()):
        if not isinstance(raw_option, Mapping):
            continue
        reason = raw_option.get("reason")
        if isinstance(reason, str) and reason in {"yield_unavailable", "missing_yield"}:
            add(
                "missing_yield",
                amount=_positive_amount(raw_option.get("amount")),
                instrument_code=_instrument_code(raw_option.get("instrument")),
            )

    return list(rows.values())


def _rounded_fingerprint_amount(amount: Decimal | None) -> str:
    if amount is None:
        return "none"
    quantum = Decimal("1E2") if abs(amount) >= Decimal("10000") else Decimal("1E-2")
    rounded = amount.quantize(quantum, rounding=ROUND_HALF_UP)
    return format(rounded, "f")


def _fingerprint(candidate: _Candidate) -> str:
    payload = f"{candidate.key}\0{_rounded_fingerprint_amount(candidate.amount)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def highlight_fingerprint(plan: Any, key: str, copy: Mapping[str, Any]) -> str | None:
    """Return the current value fingerprint for a known highlight key."""
    candidates = {candidate.key: candidate for candidate in _candidate_rows(_document(plan))}
    candidate = candidates.get(_safe_key(key))
    if candidate is None:
        return None
    catalog = copy.get("highlights", {}).get("catalog", {})
    if not isinstance(catalog, Mapping) or not catalog:
        return None
    if candidate.copy_key not in catalog and candidate.key not in catalog:
        return None
    return _fingerprint(candidate)


def highlights(plan: Any, context: HighlightContext) -> list[Highlight]:
    """Rank all plan notes for one screen and return no more than its display limit."""
    document = _document(plan)
    catalog = context.copy.get("highlights", {}).get("catalog", {})
    if not isinstance(catalog, Mapping):
        return []

    ranked: list[tuple[tuple[int, Decimal, int, int], Highlight]] = []
    for candidate in _candidate_rows(document):
        definition = catalog.get(candidate.copy_key, catalog.get(candidate.key))
        # A note without plain copy (e.g. an internal data-quality flag) stays in the
        # details views; a vague generic card would only add noise ("when unsure, show less").
        if not isinstance(definition, Mapping):
            continue
        fingerprint = _fingerprint(candidate)
        if context.dismissed_fingerprints.get(candidate.key) == fingerprint:
            continue

        screens = definition.get("screens", _ALL_SCREENS)
        if not isinstance(screens, Sequence) or isinstance(screens, str):
            screens = _ALL_SCREENS
        relevant = context.screen in screens
        title = str(definition.get("title", "Review a plan detail"))
        body = str(definition.get("body", "One detail may change what to do next."))
        amount_text = format_inr(candidate.amount, style="compact") if candidate.amount else ""
        rate_text = (
            f"{format(candidate.rate_pct.normalize(), 'f')}%"
            if candidate.rate_pct is not None
            else ""
        )
        title = title.format(amount=amount_text, rate=rate_text)
        body = body.format(amount=amount_text, rate=rate_text)
        severity = definition.get("severity", "info")
        if severity not in {"info", "attention"}:
            severity = "info"
        item = Highlight(
            key=candidate.key,
            screen=context.screen,
            title=title,
            body=body,
            learn_more_key=str(definition.get("learn_more_key", candidate.copy_key)),
            severity=severity,
            context=(
                {"instrument_code": candidate.instrument_code}
                if candidate.instrument_code is not None
                else {}
            ),
        )
        score = (
            int(bool(definition.get("needs_action_now", False))),
            abs(candidate.amount or Decimal(0)),
            int(relevant),
            -candidate.position,
        )
        ranked.append((score, item))

    ranked.sort(key=lambda row: row[0], reverse=True)
    return [item for _score, item in ranked[: _SCREEN_LIMITS[context.screen]]]
