"""Typed research handovers and deterministic grounding checks."""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wealth.research.errors import BudgetExceeded


class Stage(StrEnum):
    QUEUED = "QUEUED"
    SCREENED = "SCREENED"
    ANALYSED = "ANALYSED"
    VALUED = "VALUED"
    RED_TEAM_DONE = "RED_TEAM_DONE"
    COMPLIANCE_DONE = "COMPLIANCE_DONE"
    APPROVED = "APPROVED"
    VETOED = "VETOED"
    UNRESOLVED = "UNRESOLVED"


class DossierSecurity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    market: Literal["IN", "US", "DK", "EU", "GB"]
    type: Literal["EQUITY", "ETF", "MF", "BOND", "SGB"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    source_url: str | None = None
    retrieved_at: datetime | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    kind: Literal["FACT", "INFERENCE", "OPINION"]
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def fact_requires_tool_evidence(self) -> Claim:
        if self.kind == "FACT" and not any(item.tool_call_id.strip() for item in self.evidence):
            raise ValueError("FACT claims require evidence with a tool_call_id")
        return self


class Handoff(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_agent: str = Field(alias="from")
    to_agent: str = Field(alias="to")
    reason: str
    budget_remaining_tokens: int = Field(ge=0)


class CitedReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=300)
    evidence: list[str] = Field(default_factory=list)


class RedTeamVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strongest_bear_case: str = ""
    falsifier: str = ""
    downgrade: Literal["NONE", "WATCH", "VETO"] = "NONE"
    reasons: list[CitedReason] = Field(default_factory=list)


class Verdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fundamental: str | None = None
    valuation: str | None = None
    red_team: RedTeamVerdict | None = None
    compliance: str = "PENDING"


class GeoExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    us_revenue_pct: Decimal | None = Field(default=None, ge=0, le=100)
    tariff_sensitive: bool | None = None
    evidence: list[str] = Field(default_factory=list)


class DossierBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_usd: Decimal = Field(default=Decimal("0.25"), ge=0)
    max_calls: int = Field(default=20, ge=0)
    spent_usd: Decimal = Field(default=Decimal("0"), ge=0)
    calls: int = Field(default=0, ge=0)

    def ensure_call(self, estimated_cost: Decimal) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded("Dossier call limit reached")
        if self.spent_usd + estimated_cost > self.max_usd:
            raise BudgetExceeded("Estimated model cost exceeds the remaining dossier budget")


class Dossier(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dossier_id: str = Field(pattern=r"^\d{4}-W\d{2}/(?:IN|US|DK|EU|GB)/[A-Za-z0-9._-]+$")
    schema_version: str = "1.0"
    security: DossierSecurity
    stage: Stage = Stage.QUEUED
    handoff: Handoff | None = None
    claims: list[Claim] = Field(default_factory=list)
    analyst_claim_ids: list[str] = Field(default_factory=list)
    valuation_claim_ids: list[str] = Field(default_factory=list)
    drivers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    geo_exposure: GeoExposure | None = None
    verdicts: Verdicts = Field(default_factory=Verdicts)
    valuation_stats: dict[str, Any] | None = None
    valuation_explanation: str | None = None
    valuation_evidence_ids: list[str] = Field(default_factory=list)
    valuation_percentile: Decimal | None = Field(default=None, ge=0, le=100)
    compliance_failures: list[str] = Field(default_factory=list)
    red_team_flags: list[str] = Field(default_factory=list)
    status: Literal["APPROVED", "PENDING", "VETOED", "UNRESOLVED"] = "PENDING"
    status_reason: str | None = None
    open_questions: list[str] = Field(default_factory=list)
    confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    trace_id: str | None = None
    budget: DossierBudget = Field(default_factory=DossierBudget)
    handback_count: int = Field(default=0, ge=0, le=2)


_STAGE_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.QUEUED: {Stage.SCREENED},
    Stage.SCREENED: {Stage.ANALYSED},
    Stage.ANALYSED: {Stage.VALUED},
    Stage.VALUED: {Stage.RED_TEAM_DONE},
    Stage.RED_TEAM_DONE: {Stage.COMPLIANCE_DONE},
    Stage.COMPLIANCE_DONE: {Stage.APPROVED, Stage.VETOED, Stage.UNRESOLVED},
    Stage.APPROVED: set(),
    Stage.VETOED: set(),
    Stage.UNRESOLVED: set(),
}

_NUMBER_PATTERN = re.compile(r"(?<![\w])[-+]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?%?(?![\w])")


def transition_stage(dossier: Dossier, next_stage: Stage) -> Dossier:
    """Return a dossier at the next legal stage, rejecting skipped or reversed stages."""
    next_stage = Stage(next_stage)
    if next_stage not in _STAGE_TRANSITIONS[dossier.stage]:
        raise ValueError(f"Illegal dossier transition: {dossier.stage} -> {next_stage}")
    return dossier.model_copy(update={"stage": next_stage})


def hand_back_to_analysis(
    dossier: Dossier,
    *,
    reason: str,
    budget_remaining_tokens: int = 0,
    from_agent: str = "compliance_agent",
    to_agent: str = "analysis_agent",
) -> Dossier:
    """Apply one compliance hand-back, or close the dossier after two hand-backs."""
    if dossier.stage != Stage.COMPLIANCE_DONE:
        raise ValueError("Only a compliance-complete dossier can be handed back")
    if dossier.handback_count >= 2:
        return dossier.model_copy(update={"stage": Stage.UNRESOLVED})
    handoff = Handoff(
        **{
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "budget_remaining_tokens": budget_remaining_tokens,
        }
    )
    return dossier.model_copy(
        update={
            "stage": Stage.ANALYSED,
            "handback_count": dossier.handback_count + 1,
            "handoff": handoff,
        }
    )


def _normalise_number(value: str) -> str:
    cleaned = value.replace(",", "").removesuffix("%")
    number = Decimal(cleaned).normalize()
    return format(number, "f")


# Claims round and rescale cited values ("$1.573B", "7.0%" for 0.0700..., "Rs 1,234 crore").
_CLAIM_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?P<number>[-+]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)"
    r"(?:(?P<percent>%)|\s?(?P<scale>billion|bn|million|mn|thousand|crores?|cr|lakhs?|[BMK])"
    r"(?![A-Za-z]))?(?![\w])"
)
_SCALE_POWERS = {
    "billion": 9,
    "bn": 9,
    "b": 9,
    "million": 6,
    "mn": 6,
    "m": 6,
    "thousand": 3,
    "k": 3,
    "crore": 7,
    "crores": 7,
    "cr": 7,
    "lakh": 5,
    "lakhs": 5,
}


def _rounds_to(value: str, target: Decimal, quantum: Decimal) -> bool:
    # Words often carry the sign ("3.1% below its average"), so magnitudes are compared.
    try:
        return abs(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP)) == abs(target)
    except InvalidOperation:
        return False


def _supported(match: re.Match[str], evidence_numbers: set[str]) -> bool:
    """A claim may quote a cited value at its own precision and scale.

    Tool results carry full Decimal precision, so exact matching rejected every rounded
    figure ("18.9%" for 18.8810...), including the screener's own four-place factor summary,
    and every value a reader would rescale ("$221.2M", "7.0%" for 0.0700...).
    """
    raw = match.group("number")
    digits = raw.replace(",", "")
    places = len(digits.split(".", 1)[1]) if "." in digits else 0
    number = Decimal(digits)
    power = _SCALE_POWERS.get((match.group("scale") or "").casefold(), 0)
    target = number.scaleb(power)
    quantum = Decimal(1).scaleb(power - places)
    if any(_rounds_to(value, target, quantum) for value in evidence_numbers):
        return True
    if match.group("percent"):
        # A ratio stored as a fraction is quoted as a percentage.
        fraction_quantum = Decimal(1).scaleb(-places - 2)
        return any(
            _rounds_to(value, number.scaleb(-2), fraction_quantum) for value in evidence_numbers
        )
    return False


def _numbers(value: str) -> set[str]:
    return {_normalise_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(value)}


def _tool_call_map(tool_calls: Any) -> dict[str, Any]:
    if isinstance(tool_calls, dict):
        return {str(key): value for key, value in tool_calls.items()}
    result: dict[str, Any] = {}
    for item in tool_calls:
        identifier = getattr(item, "tool_call_id", None) or getattr(item, "id", None)
        if identifier is not None:
            result[str(identifier)] = item
    return result


def _result_payload(tool_call: Any) -> Any:
    if hasattr(tool_call, "result"):
        return tool_call.result
    if isinstance(tool_call, dict) and "result" in tool_call:
        return tool_call["result"]
    if isinstance(tool_call, dict) and "result_json" in tool_call:
        return tool_call["result_json"]
    return tool_call


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def validate_grounding(dossier: Dossier, tool_calls: Any) -> list[str]:
    """Find FACT claims with missing citations or unsupported numeric values."""
    calls = _tool_call_map(tool_calls)
    failures: list[str] = []
    for claim in dossier.claims:
        if claim.kind != "FACT":
            continue
        cited_ids = [item.tool_call_id for item in claim.evidence]
        missing_ids = [identifier for identifier in cited_ids if identifier not in calls]
        if missing_ids:
            failures.append(f"{claim.id}: unknown tool_call_id {', '.join(missing_ids)}")
            continue
        evidence_numbers: set[str] = set()
        for identifier in cited_ids:
            evidence_text = _json_text(_result_payload(calls[identifier]))
            evidence_numbers.update(_numbers(evidence_text))
            # Field names carry periods ("momentum_12_1_pct", "volatility_1y_pct",
            # "dist_from_200dma") that the model rightly writes as "12-1 momentum",
            # "1-year volatility" or "200-day average".
            evidence_numbers.update(
                _numbers(re.sub(r"_|(?<=\d)(?=(?:dma|[dwmy])\b)", " ", evidence_text))
            )
        unsupported = sorted(
            {
                _normalise_number(match.group("number"))
                for match in _CLAIM_NUMBER_PATTERN.finditer(claim.text)
                if not _supported(match, evidence_numbers)
            },
            key=Decimal,
        )
        if unsupported:
            failures.append(f"{claim.id}: unsupported numbers {', '.join(unsupported)}")
    return failures
