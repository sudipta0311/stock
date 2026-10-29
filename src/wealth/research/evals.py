"""Golden-set evaluation runner for the research graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from wealth.research.dossier import Dossier, validate_grounding
from wealth.research.errors import BudgetExceeded, SchemaError
from wealth.research.graph import build_research_graph
from wealth.research.router import (
    FakeRouter,
    ModelRouter,
    RouterResult,
    RunBudget,
    RunContext,
)
from wealth.research.rules import ResearchRules
from wealth.research.tools import ToolRegistry

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CASES = _REPOSITORY_ROOT / "evals" / "golden"
_DEFAULT_THRESHOLDS = _REPOSITORY_ROOT / "evals" / "thresholds.yaml"
_CASE_ID = re.compile(r"^[a-z0-9_]+$")
_AGENT_NAMES = {
    "fundamental_analyst",
    "valuation_agent",
    "red_team_agent",
    "compliance_agent",
    "cio_agent",
}
_DOWNGRADE_RANK = {"NONE": 0, "WATCH": 1, "VETO": 2}


class EvalExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["APPROVED", "VETOED", "UNRESOLVED"]
    red_team_downgrade: Literal["NONE", "WATCH", "VETO"] | None = None
    minimum_red_team_downgrade: Literal["NONE", "WATCH", "VETO"] | None = None
    red_flag: bool = False
    compliance_catch: bool = False
    open_questions: bool = False
    injection_equivalent_to: str | None = None
    rejected_tool: str | None = None
    valuation_calls: int | None = Field(default=None, ge=1, le=2)
    valuation_explanation: str | None = None


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: Literal[
        "positive",
        "red_flag",
        "grounding_trap",
        "injection",
        "prohibited_language",
        "data_sparse",
    ]
    input: dict[str, Any]
    expected: EvalExpectation


@dataclass
class CaseResult:
    case_id: str
    category: str
    expected_status: str
    status: str
    status_reason: str | None
    red_team_downgrade: str | None
    passed: bool
    reasons: list[str]
    schema_valid: bool
    eligible_fact_claims: int
    grounded_fact_claims: int
    red_flags_planted: int
    red_flags_caught: int
    compliance_required: bool
    compliance_caught: bool
    injection_case: bool
    injection_unchanged: bool | None
    injection_pair: str | None
    rejected_tool_checked: bool
    rejected_tool_seen: bool
    outcome_signature: tuple[str, ...]
    cost_usd: Decimal
    latency_ms: list[int]


@dataclass(frozen=True)
class EvalMetrics:
    groundedness_rate: Decimal
    schema_valid_rate: Decimal
    redteam_recall: Decimal
    injection_resistance: Decimal
    compliance_catch_rate: Decimal
    cost_usd: Decimal
    cost_per_dossier: Decimal
    latency_p50_ms: int
    latency_p95_ms: int


class MissingRecording(ValueError):
    """A fake-router agent response is missing from the committed recording."""


class RecordingFakeRouter(FakeRouter):
    """Replay recorded outputs through FakeRouter's budget and schema handling."""

    def __init__(self, recordings_dir: Path, *, cost_per_call: Decimal) -> None:
        super().__init__(cost_per_call=cost_per_call)
        self._responses: dict[str, list[Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.recording_errors: list[str] = []
        for path in sorted(recordings_dir.glob("*.json")):
            if path.stem not in _AGENT_NAMES:
                raise ValueError("Unknown agent recording")
            responses = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(responses, list):
                raise ValueError("Agent recordings must be JSON arrays")
            self._responses[path.stem] = responses

    def complete(
        self,
        tier: str,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        schema: type[BaseModel] | None,
        max_output_tokens: int,
        run_ctx: RunContext,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> RouterResult:
        queue = self._responses.get(run_ctx.agent, [])
        if not queue:
            self.recording_errors.append(run_ctx.agent)
            raise MissingRecording(f"No recorded output for {run_ctx.agent}")
        response = queue.pop(0)
        self.fixtures[tier] = response
        event: dict[str, Any] = {
            "agent": run_ctx.agent,
            "tier": tier,
            "schema": schema.__name__ if schema is not None else None,
            "allowed_tools": _tool_names(tools),
            "messages": [dict(message) for message in messages],
            "response": response,
            "schema_valid": None,
            "cost_usd": Decimal("0"),
            "latency_ms": 0,
        }
        previous_calls = self.calls
        try:
            result = super().complete(
                tier,
                messages,
                schema=schema,
                max_output_tokens=max_output_tokens,
                run_ctx=run_ctx,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as error:
            event["error"] = type(error).__name__
            completed = self.calls > previous_calls
            event["schema_valid"] = False if schema is not None and completed else None
            if completed:
                event["cost_usd"] = self.cost_per_call
            self.events.append(event)
            raise
        event["schema_valid"] = True if schema is not None else None
        event["requested_tools"] = [
            str(call.get("function", {}).get("name", "")) for call in result.tool_calls
        ]
        event["cost_usd"] = result.cost_usd
        event["latency_ms"] = result.latency_ms
        self.events.append(event)
        return result

    def unused_responses(self) -> int:
        return sum(len(responses) for responses in self._responses.values())


class RecordingLiveRouter:
    """Record safe per-call metadata while delegating every request to ModelRouter."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self.events: list[dict[str, Any]] = []
        self._responses_by_agent: dict[str, list[Any]] = defaultdict(list)

    def complete(
        self,
        tier: str,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        schema: type[BaseModel] | None,
        max_output_tokens: int,
        run_ctx: RunContext,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> RouterResult:
        event: dict[str, Any] = {
            "agent": run_ctx.agent,
            "tier": tier,
            "schema": schema.__name__ if schema is not None else None,
            "allowed_tools": _tool_names(tools),
            "messages": [dict(message) for message in messages],
            "schema_valid": None,
            "cost_usd": Decimal("0"),
            "latency_ms": 0,
        }
        try:
            result = self.router.complete(
                tier,
                messages,
                schema=schema,
                max_output_tokens=max_output_tokens,
                run_ctx=run_ctx,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as error:
            event["error"] = type(error).__name__
            event["schema_valid"] = (
                False if schema is not None and isinstance(error, SchemaError) else None
            )
            self.events.append(event)
            raise
        recorded_response: Any
        if result.tool_calls:
            recorded_response = {
                "tool_calls": list(result.tool_calls),
                "content": result.raw_text,
            }
        elif isinstance(result.parsed, BaseModel):
            recorded_response = result.parsed.model_dump(mode="json")
        else:
            recorded_response = result.parsed
        self._responses_by_agent[run_ctx.agent].append(recorded_response)
        event["response"] = recorded_response
        event["requested_tools"] = [
            str(call.get("function", {}).get("name", "")) for call in result.tool_calls
        ]
        event["schema_valid"] = True if schema is not None else None
        event["cost_usd"] = result.cost_usd
        event["latency_ms"] = result.latency_ms
        self.events.append(event)
        return result

    def write_recordings(self, root: Path, case_id: str) -> None:
        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for agent, responses in sorted(self._responses_by_agent.items()):
            path = case_dir / f"{agent}.json"
            path.write_text(
                json.dumps(responses, ensure_ascii=False, indent=2, default=_json_default) + "\n",
                encoding="utf-8",
            )


def _tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    return [str(tool.get("function", {}).get("name", "")) for tool in (tools or [])]


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def load_cases(cases_dir: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for path in sorted(cases_dir.glob("*.json")):
        cases.append(GoldenCase.model_validate_json(path.read_text(encoding="utf-8")))
    if not cases:
        raise ValueError("No golden case JSON files were found")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Golden case IDs must be unique")
    return cases


def _make_tool_registry(case: GoldenCase) -> ToolRegistry:
    inputs = case.input
    valuation = dict(inputs["valuation_stats"])
    factor_result = dict(inputs.get("factor_snapshot", {}))
    registry = ToolRegistry()

    def factor_snapshot(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        output = {**factor_result, "security_id": str(args["security_id"])}
        return output, "https://evidence.example.test/factor"

    def valuation_stats(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        output = {**valuation, "security_id": str(args["security_id"])}
        return output, "https://evidence.example.test/valuation"

    registry.register("factor_snapshot", factor_snapshot)
    registry.register("valuation_stats", valuation_stats)
    return registry


def _outcome_signature(dossiers: list[Dossier], pack: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for dossier in dossiers:
        red_team = dossier.verdicts.red_team
        values.extend(
            [
                dossier.status,
                str(dossier.verdicts.fundamental),
                str(dossier.verdicts.valuation),
                red_team.downgrade if red_team is not None else "NONE",
                ",".join(sorted(dossier.open_questions)),
            ]
        )
    values.append(str(len(pack.get("ideas", []))))
    return tuple(values)


def _tool_misuse_was_rejected(events: list[dict[str, Any]], tool_name: str) -> bool:
    attempted = any(
        tool_name in event.get("requested_tools", [])
        and tool_name not in event.get("allowed_tools", [])
        for event in events
    )
    rejected = any(
        message.get("role") == "tool" and "tool_request_rejected" in str(message.get("content", ""))
        for event in events
        for message in event.get("messages", [])
    )
    return attempted and rejected


def _run_case(
    case: GoldenCase,
    *,
    router_mode: Literal["fake", "live"],
    budget: RunBudget,
    live_router: RecordingLiveRouter | None,
    fake_cost_per_call: Decimal,
) -> tuple[CaseResult, RecordingLiveRouter | RecordingFakeRouter]:
    candidate = dict(case.input["candidate"])
    context = RunContext(budget=budget, mode="weekly")
    if router_mode == "fake":
        router: RecordingLiveRouter | RecordingFakeRouter = RecordingFakeRouter(
            _REPOSITORY_ROOT / "evals" / "recordings" / case.case_id,
            cost_per_call=fake_cost_per_call,
        )
    else:
        if live_router is None:
            raise ValueError("A live router is required for live evaluations")
        live_router.events = []
        live_router._responses_by_agent.clear()
        router = live_router

    registry = _make_tool_registry(case)
    try:
        graph = build_research_graph(
            router,
            run_ctx=context,
            tools=registry,
            shortlist=[candidate],
            rules=ResearchRules.load(),
            interrupt_after_screen=False,
            interrupt_after_red_team=False,
        )
        week = str(candidate["dossier_id"]).split("/", 1)[0]
        state = graph.invoke(
            {
                "run_id": f"eval-{case.case_id}",
                "week": week,
                "shortlist": [candidate],
                "shortlist_ids": [],
                "dossiers": [],
                "tool_results_by_dossier": {},
                "stage": "QUEUED",
                "status": "running",
            }
        )
    except Exception as error:
        failed = _failed_case(case, type(error).__name__)
        failed.cost_usd = sum(
            (Decimal(str(item.get("cost_usd", "0"))) for item in router.events),
            Decimal("0"),
        )
        failed.latency_ms = [int(item.get("latency_ms", 0)) for item in router.events]
        return failed, router
    finally:
        registry.close()

    schema_valid = True
    dossiers: list[Dossier] = []
    for raw_dossier in state.get("dossiers", []):
        try:
            dossiers.append(Dossier.model_validate(raw_dossier))
        except (TypeError, ValueError):
            schema_valid = False
    if len(dossiers) != len(state.get("dossiers", [])) or not dossiers:
        schema_valid = False
    dossier = dossiers[0] if dossiers else None
    if dossier is None:
        return _failed_case(case, "missing_dossier"), router

    schema_valid = schema_valid and all(
        event.get("schema_valid") is not False
        for event in router.events
        if event.get("schema") is not None
    )
    analyst_schema_error = dossier.status_reason in {"analyst_schema", "handback_schema"}
    analyst_output_events = [
        event
        for event in router.events
        if event.get("agent") == "fundamental_analyst"
        and event.get("schema") is None
        and "error" not in event
        and not event.get("requested_tools")
    ]
    if analyst_output_events and analyst_schema_error:
        schema_valid = False

    tool_results = state.get("tool_results_by_dossier", {}).get(dossier.dossier_id, {})
    status = dossier.status
    downgrade = (
        dossier.verdicts.red_team.downgrade if dossier.verdicts.red_team is not None else None
    )
    compliance_caught = dossier.status_reason == "compliance_failed"
    eligible_facts = (
        [claim for claim in dossier.claims if claim.kind == "FACT"] if status == "APPROVED" else []
    )
    grounding_failures = validate_grounding(dossier, tool_results) if eligible_facts else []
    failed_claim_ids = {item.split(":", 1)[0] for item in grounding_failures if ":" in item}
    grounded_facts = sum(claim.id not in failed_claim_ids for claim in eligible_facts)
    red_flags_caught = int(
        case.expected.red_flag
        and downgrade is not None
        and case.expected.minimum_red_team_downgrade is not None
        and _DOWNGRADE_RANK[downgrade] >= _DOWNGRADE_RANK[case.expected.minimum_red_team_downgrade]
    )
    reasons: list[str] = []
    if status != case.expected.status:
        reasons.append(f"expected {case.expected.status}, got {status}")
    if (
        case.expected.red_team_downgrade is not None
        and downgrade != case.expected.red_team_downgrade
    ):
        reasons.append(
            f"expected red-team downgrade {case.expected.red_team_downgrade}, got {downgrade}"
        )
    if (
        case.expected.minimum_red_team_downgrade is not None
        and downgrade is not None
        and _DOWNGRADE_RANK[downgrade] < _DOWNGRADE_RANK[case.expected.minimum_red_team_downgrade]
    ):
        reasons.append("red-team downgrade was below the expected minimum")
    if case.expected.compliance_catch != compliance_caught:
        reasons.append("deterministic compliance catch did not match the expected outcome")
    if case.expected.open_questions and not dossier.open_questions:
        reasons.append("expected unresolved open questions")
    valuation_calls = sum(event.get("agent") == "valuation_agent" for event in router.events)
    if (
        case.expected.valuation_calls is not None
        and valuation_calls != case.expected.valuation_calls
    ):
        reasons.append("valuation wording did not follow the expected rewrite path")
    if (
        case.expected.valuation_explanation is not None
        and dossier.valuation_explanation != case.expected.valuation_explanation
    ):
        reasons.append("valuation wording did not match the expected safe replacement")
    if not schema_valid:
        reasons.append("final dossier failed schema validation")
    if isinstance(router, RecordingFakeRouter):
        if router.recording_errors:
            reasons.append("a committed model recording was missing")
        if router.unused_responses():
            reasons.append("committed model recordings were unused")
    tool_checked = case.expected.rejected_tool is not None
    tool_seen = (
        _tool_misuse_was_rejected(router.events, case.expected.rejected_tool)
        if tool_checked
        else False
    )
    if tool_checked and not tool_seen:
        reasons.append("the injected tool request was not observed and rejected")

    event_cost = sum(
        (Decimal(str(item.get("cost_usd", "0"))) for item in router.events), Decimal("0")
    )
    latencies = [int(item.get("latency_ms", 0)) for item in router.events]
    result = CaseResult(
        case_id=case.case_id,
        category=case.category,
        expected_status=case.expected.status,
        status=status,
        status_reason=dossier.status_reason,
        red_team_downgrade=downgrade,
        passed=not reasons,
        reasons=reasons,
        schema_valid=schema_valid,
        eligible_fact_claims=len(eligible_facts),
        grounded_fact_claims=grounded_facts,
        red_flags_planted=int(case.expected.red_flag),
        red_flags_caught=red_flags_caught,
        compliance_required=case.expected.compliance_catch,
        compliance_caught=compliance_caught,
        injection_case=case.category == "injection",
        injection_unchanged=None,
        injection_pair=case.expected.injection_equivalent_to,
        rejected_tool_checked=tool_checked,
        rejected_tool_seen=tool_seen,
        outcome_signature=_outcome_signature(dossiers, state.get("pack", {})),
        cost_usd=event_cost,
        latency_ms=latencies,
    )
    return result, router


def _failed_case(case: GoldenCase, error_type: str) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        expected_status=case.expected.status,
        status="UNRESOLVED",
        status_reason=error_type,
        red_team_downgrade=None,
        passed=False,
        reasons=[f"graph failed with {error_type}"],
        schema_valid=False,
        eligible_fact_claims=0,
        grounded_fact_claims=0,
        red_flags_planted=int(case.expected.red_flag),
        red_flags_caught=0,
        compliance_required=case.expected.compliance_catch,
        compliance_caught=False,
        injection_case=case.category == "injection",
        injection_unchanged=None,
        injection_pair=case.expected.injection_equivalent_to,
        rejected_tool_checked=case.expected.rejected_tool is not None,
        rejected_tool_seen=False,
        outcome_signature=("UNRESOLVED", error_type),
        cost_usd=Decimal("0"),
        latency_ms=[],
    )


def _apply_injection_checks(cases: list[GoldenCase], results: list[CaseResult]) -> None:
    by_id = {result.case_id: result for result in results}
    for case, result in zip(cases, results, strict=True):
        if case.category != "injection":
            continue
        paired = by_id.get(case.expected.injection_equivalent_to or "")
        result.injection_unchanged = bool(
            paired is not None and result.outcome_signature == paired.outcome_signature
        )
        if not result.injection_unchanged:
            result.reasons.append("injected evidence changed the research outcome")
            result.passed = False


def compute_metrics(results: list[CaseResult]) -> EvalMetrics:
    fact_total = sum(item.eligible_fact_claims for item in results)
    fact_grounded = sum(item.grounded_fact_claims for item in results)
    case_total = len(results)
    planted = sum(item.red_flags_planted for item in results)
    caught = sum(item.red_flags_caught for item in results)
    injections = [item for item in results if item.injection_case]
    compliance_cases = [item for item in results if item.compliance_required]
    latencies = sorted(value for result in results for value in result.latency_ms)
    total_cost = sum((item.cost_usd for item in results), Decimal("0"))
    return EvalMetrics(
        groundedness_rate=_rate(fact_grounded, fact_total),
        schema_valid_rate=_rate(sum(item.schema_valid for item in results), case_total),
        redteam_recall=_rate(caught, planted),
        injection_resistance=_rate(
            sum(item.injection_unchanged is True for item in injections), len(injections)
        ),
        compliance_catch_rate=_rate(
            sum(item.compliance_caught for item in compliance_cases), len(compliance_cases)
        ),
        cost_usd=total_cost,
        cost_per_dossier=(total_cost / Decimal(case_total) if case_total else Decimal("0")),
        latency_p50_ms=int(median(latencies)) if latencies else 0,
        latency_p95_ms=_percentile_95(latencies),
    )


def _rate(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else Decimal("1")


def _percentile_95(values: list[int]) -> int:
    if not values:
        return 0
    return values[max(0, math.ceil(Decimal("0.95") * len(values)) - 1)]


def _load_thresholds(path: Path) -> dict[str, Decimal]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Thresholds must be a YAML mapping")
    expected = {
        "groundedness_rate",
        "schema_valid_rate",
        "redteam_recall",
        "injection_resistance",
        "compliance_catch_rate",
    }
    if set(raw) != expected:
        raise ValueError("Thresholds must define every required eval metric")
    try:
        return {key: Decimal(str(value)) for key, value in raw.items()}
    except InvalidOperation as error:
        raise ValueError("Threshold values must be decimal numbers") from error


def threshold_failures(metrics: EvalMetrics, thresholds: dict[str, Decimal]) -> list[str]:
    failures: list[str] = []
    for name in (
        "groundedness_rate",
        "schema_valid_rate",
        "redteam_recall",
        "compliance_catch_rate",
    ):
        actual = getattr(metrics, name)
        if actual < thresholds[name]:
            failures.append(f"{name} below threshold")
    if metrics.injection_resistance != thresholds["injection_resistance"]:
        failures.append("injection_resistance did not equal its threshold")
    return failures


def threshold_exit_code(metrics: EvalMetrics, thresholds: dict[str, Decimal]) -> int:
    return 1 if threshold_failures(metrics, thresholds) else 0


def _metric_payload(metrics: EvalMetrics) -> dict[str, str | int]:
    return {
        "groundedness_rate": format(metrics.groundedness_rate, ".6f"),
        "schema_valid_rate": format(metrics.schema_valid_rate, ".6f"),
        "redteam_recall": format(metrics.redteam_recall, ".6f"),
        "injection_resistance": format(metrics.injection_resistance, ".6f"),
        "compliance_catch_rate": format(metrics.compliance_catch_rate, ".6f"),
        "cost_usd": format(metrics.cost_usd, "f"),
        "cost_per_dossier": format(metrics.cost_per_dossier, "f"),
        "latency_p50_ms": metrics.latency_p50_ms,
        "latency_p95_ms": metrics.latency_p95_ms,
    }


def _case_payload(result: CaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "category": result.category,
        "expected_status": result.expected_status,
        "status": result.status,
        "status_reason": result.status_reason,
        "red_team_downgrade": result.red_team_downgrade,
        "passed": result.passed,
        "reasons": list(result.reasons),
        "schema_valid": result.schema_valid,
        "eligible_fact_claims": result.eligible_fact_claims,
        "grounded_fact_claims": result.grounded_fact_claims,
        "red_flags_planted": result.red_flags_planted,
        "red_flags_caught": result.red_flags_caught,
        "compliance_required": result.compliance_required,
        "compliance_caught": result.compliance_caught,
        "injection_unchanged": result.injection_unchanged,
        "cost_usd": format(result.cost_usd, "f"),
        "latency_ms": result.latency_ms,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Research golden-set evaluation",
        "",
        f"Router: `{report['router']}`  ",
        f"Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## Metrics",
        "",
        "| Metric | Result | Threshold |",
        "|---|---:|---:|",
    ]
    for name, value in report["metrics"].items():
        threshold = report["thresholds"].get(name, "—")
        lines.append(f"| {name} | {value} | {threshold} |")
    lines.extend(
        [
            "",
            "Groundedness counts FACT claims in approved dossiers. Compliance catches are measured "
            "on rejected cases.",
            "",
            "## Cases",
            "",
            "| Case | Category | Expected | Actual | Result | Reasons |",
            "|---|---|---|---|---|---|",
        ]
    )
    for case in report["cases"]:
        reasons = "; ".join(case["reasons"]) or "—"
        lines.append(
            f"| {case['case_id']} | {case['category']} | {case['expected_status']} | "
            f"{case['status']} | {'PASS' if case['passed'] else 'FAIL'} | {reasons} |"
        )
    lines.append("")
    if report["threshold_failures"]:
        lines.extend(["## Threshold failures", ""])
        lines.extend(f"- {failure}" for failure in report["threshold_failures"])
        lines.append("")
    return "\n".join(lines)


def _resolve_output(path_text: str) -> Path:
    path = Path(path_text)
    resolved = path.resolve() if path.is_absolute() else (_REPOSITORY_ROOT / path).resolve()
    output_root = (_REPOSITORY_ROOT / "out").resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as error:
        raise ValueError("Reports must be written under the repository out/ directory") from error
    return resolved


def _live_router() -> RecordingLiveRouter:
    missing = [name for name in ("FOUNDRY_ENDPOINT", "FOUNDRY_API_KEY") if not os.getenv(name)]
    if missing:
        raise ValueError(
            "--router live requires configured environment variables: " + ", ".join(missing)
        )
    return RecordingLiveRouter(ModelRouter())


def _run(args: argparse.Namespace) -> int:
    try:
        budget_usd = Decimal(args.budget_usd)
    except InvalidOperation:
        raise ValueError("--budget-usd must be a decimal number") from None
    if not budget_usd.is_finite() or budget_usd < 0:
        raise ValueError("--budget-usd must be finite and non-negative")
    cases_dir = Path(args.cases)
    if not cases_dir.is_absolute():
        cases_dir = (_REPOSITORY_ROOT / cases_dir).resolve()
    cases = load_cases(cases_dir)
    thresholds = _load_thresholds(Path(args.thresholds))
    live_router = _live_router() if args.router == "live" else None
    if args.record and args.router != "live":
        raise ValueError("--record is available only with --router live")
    shared_budget = RunBudget(max_usd=budget_usd, max_calls=100_000)
    results: list[CaseResult] = []
    for case in cases:
        result, router = _run_case(
            case,
            router_mode=args.router,
            budget=shared_budget,
            live_router=live_router,
            fake_cost_per_call=Decimal("0.001"),
        )
        results.append(result)
        if args.record and isinstance(router, RecordingLiveRouter):
            router.write_recordings(_resolve_output(args.out) / "recordings", case.case_id)
    _apply_injection_checks(cases, results)
    metrics = compute_metrics(results)
    threshold_misses = threshold_failures(metrics, thresholds)
    failed_cases = [result for result in results if not result.passed]
    output_dir = _resolve_output(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_payload = {key: format(value, "f") for key, value in thresholds.items()}
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "router": args.router,
        "budget_usd": format(budget_usd, "f"),
        "passed": not threshold_misses and not failed_cases,
        "metrics": _metric_payload(metrics),
        "thresholds": threshold_payload,
        "threshold_failures": threshold_misses,
        "failed_cases": [result.case_id for result in failed_cases],
        "groundedness_scope": "FACT claims in APPROVED dossiers",
        "cases": [_case_payload(result) for result in results],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
    print(
        f"Research eval {'passed' if report['passed'] else 'failed'}: "
        f"{len(results)} cases, {metrics.cost_usd} USD; reports in {output_dir}"
    )
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m wealth.research.evals")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--router", choices=("fake", "live"), required=True)
    run.add_argument("--cases", default=str(_DEFAULT_CASES))
    run.add_argument("--out", required=True)
    run.add_argument("--budget-usd", default="0.50")
    run.add_argument("--thresholds", default=str(_DEFAULT_THRESHOLDS))
    run.add_argument("--record", action="store_true")
    run.set_defaults(handler=_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, yaml.YAMLError, BudgetExceeded) as error:
        print(f"Research eval failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
