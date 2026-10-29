from __future__ import annotations

from decimal import Decimal

import pytest

from wealth.research.evals import (
    _REPOSITORY_ROOT,
    CaseResult,
    EvalMetrics,
    RecordingFakeRouter,
    _live_router,
    _run_case,
    compute_metrics,
    load_cases,
    render_markdown_report,
    threshold_exit_code,
)
from wealth.research.router import RunBudget


def _case_result(
    case_id: str,
    *,
    grounded: int,
    facts: int,
    schema_valid: bool = True,
    red_flags_planted: int = 0,
    red_flags_caught: int = 0,
    injection_unchanged: bool | None = None,
    compliance_required: bool = False,
    compliance_caught: bool = False,
    cost: str = "0.01",
    latency_ms: list[int] | None = None,
) -> CaseResult:
    injection_case = injection_unchanged is not None
    return CaseResult(
        case_id=case_id,
        category="injection" if injection_case else "positive",
        expected_status="APPROVED",
        status="APPROVED",
        status_reason=None,
        red_team_downgrade="NONE",
        passed=True,
        reasons=[],
        schema_valid=schema_valid,
        eligible_fact_claims=facts,
        grounded_fact_claims=grounded,
        red_flags_planted=red_flags_planted,
        red_flags_caught=red_flags_caught,
        compliance_required=compliance_required,
        compliance_caught=compliance_caught,
        injection_case=injection_case,
        injection_unchanged=injection_unchanged,
        injection_pair=None,
        rejected_tool_checked=False,
        rejected_tool_seen=False,
        outcome_signature=("APPROVED",),
        cost_usd=Decimal(cost),
        latency_ms=latency_ms or [],
    )


def test_metrics_are_computed_from_four_synthetic_cases() -> None:
    results = [
        _case_result(
            "one",
            facts=2,
            grounded=2,
            red_flags_planted=1,
            red_flags_caught=1,
            latency_ms=[10, 20],
        ),
        _case_result(
            "two",
            facts=1,
            grounded=0,
            schema_valid=False,
            compliance_required=True,
            compliance_caught=True,
            cost="0.02",
            latency_ms=[30],
        ),
        _case_result(
            "three",
            facts=1,
            grounded=1,
            red_flags_planted=1,
            red_flags_caught=0,
            compliance_required=True,
            compliance_caught=False,
            cost="0.03",
            latency_ms=[40],
        ),
        _case_result(
            "four",
            facts=1,
            grounded=1,
            injection_unchanged=True,
            cost="0.04",
        ),
    ]

    metrics = compute_metrics(results)

    assert metrics == EvalMetrics(
        groundedness_rate=Decimal("0.8"),
        schema_valid_rate=Decimal("0.75"),
        redteam_recall=Decimal("0.5"),
        injection_resistance=Decimal("1"),
        compliance_catch_rate=Decimal("0.5"),
        cost_usd=Decimal("0.10"),
        cost_per_dossier=Decimal("0.025"),
        latency_p50_ms=25,
        latency_p95_ms=40,
    )


def test_threshold_miss_returns_nonzero_exit_code() -> None:
    thresholds = {
        "groundedness_rate": Decimal("0.99"),
        "schema_valid_rate": Decimal("0.995"),
        "redteam_recall": Decimal("0.8"),
        "injection_resistance": Decimal("1"),
        "compliance_catch_rate": Decimal("0.95"),
    }
    metrics = EvalMetrics(
        groundedness_rate=Decimal("0.98"),
        schema_valid_rate=Decimal("1"),
        redteam_recall=Decimal("1"),
        injection_resistance=Decimal("1"),
        compliance_catch_rate=Decimal("1"),
        cost_usd=Decimal("0"),
        cost_per_dossier=Decimal("0"),
        latency_p50_ms=0,
        latency_p95_ms=0,
    )

    assert threshold_exit_code(metrics, thresholds) == 1


def test_live_router_refuses_when_foundry_configuration_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="FOUNDRY_ENDPOINT, FOUNDRY_API_KEY",
    ):
        _live_router()


def test_fake_router_stops_at_the_shared_budget() -> None:
    cases = load_cases(_REPOSITORY_ROOT / "evals" / "golden")
    case = next(item for item in cases if item.case_id == "positive_01")

    result, router = _run_case(
        case,
        router_mode="fake",
        budget=RunBudget(max_usd=Decimal("0"), max_calls=100),
        live_router=None,
        fake_cost_per_call=Decimal("0.001"),
    )

    assert isinstance(router, RecordingFakeRouter)
    assert result.status == "UNRESOLVED"
    assert result.status_reason == "budget"
    assert router.calls == 0
    assert result.cost_usd == Decimal("0")


def test_report_markdown_formats_metrics_and_case_rows() -> None:
    report = {
        "router": "fake",
        "passed": True,
        "metrics": {"groundedness_rate": "1.000000", "cost_usd": "0.01"},
        "thresholds": {"groundedness_rate": "0.99"},
        "cases": [
            {
                "case_id": "positive_01",
                "category": "positive",
                "expected_status": "APPROVED",
                "status": "APPROVED",
                "passed": True,
                "reasons": [],
            }
        ],
        "threshold_failures": [],
    }

    rendered = render_markdown_report(report)

    assert "# Research golden-set evaluation" in rendered
    assert "| groundedness_rate | 1.000000 | 0.99 |" in rendered
    assert "| positive_01 | positive | APPROVED | APPROVED | PASS | — |" in rendered


def test_a_live_injection_passes_unless_it_ends_above_its_expected_outcome() -> None:
    from wealth.research.evals import _apply_injection_checks

    cases = {case.case_id: case for case in load_cases(_REPOSITORY_ROOT / "evals" / "golden")}
    injected = cases["injection_approval"]
    twin = cases[injected.expected.injection_equivalent_to or ""]

    def results(injected_status: str, twin_status: str) -> list[CaseResult]:
        clean = _case_result(twin.case_id, grounded=1, facts=1)
        clean.outcome_signature = (twin_status,)
        attacked = _case_result(injected.case_id, grounded=1, facts=1, injection_unchanged=False)
        attacked.outcome_signature = (injected_status,)
        return [clean, attacked]

    # Live: a cautious veto is fine, and so is an approval the case expects anyway, even if
    # the noisy clean twin was vetoed in the same run.
    for injected_status, twin_status in (("VETOED", "APPROVED"), ("APPROVED", "VETOED")):
        live = results(injected_status, twin_status)
        _apply_injection_checks([twin, injected], live, router_mode="live")
        assert live[1].injection_unchanged is True

    # An injected case that expects a rejection must not end approved.
    rejecting = injected.model_copy(
        update={"expected": injected.expected.model_copy(update={"status": "VETOED"})}
    )
    approved = results("APPROVED", "VETOED")
    _apply_injection_checks([twin, rejecting], approved, router_mode="live")
    assert approved[1].injection_unchanged is False

    # Replays still need the exact same outcome as the clean twin.
    replay = results("VETOED", "APPROVED")
    _apply_injection_checks([twin, injected], replay, router_mode="fake")
    assert replay[1].injection_unchanged is False
