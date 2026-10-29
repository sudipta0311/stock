from __future__ import annotations

from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import MemorySaver

from wealth.research.graph import build_research_graph
from wealth.research.router import BudgetExceeded, FakeRouter, RunBudget, RunContext


def _context(*, max_usd: str = "1") -> RunContext:
    return RunContext(budget=RunBudget(max_usd=Decimal(max_usd), max_calls=100))


def test_graph_waits_after_screen_then_resumes_to_completion() -> None:
    router = FakeRouter()
    graph = build_research_graph(router, run_ctx=_context(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "synthetic-research-run"}}
    initial = {
        "run_id": "synthetic-research-run",
        "dossier_ids": [],
        "stage": "QUEUED",
        "status": "running",
        "handback_count": 0,
    }

    waiting = graph.invoke(initial, config)
    assert waiting["stage"] == "SCREENED"
    assert waiting["status"] == "waiting"
    assert router.calls == 2

    completed = graph.invoke(None, config)
    assert completed["stage"] == "APPROVED"
    assert completed["status"] == "completed"
    assert router.calls == 7


def test_compliance_can_hand_back_twice_then_marks_unresolved() -> None:
    router = FakeRouter()
    router.register("nano", lambda call: {"handoff_to": "analyse"} if call >= 3 else {})
    graph = build_research_graph(router, run_ctx=_context(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "synthetic-handback-run"}}
    graph.invoke(
        {
            "run_id": "synthetic-handback-run",
            "dossier_ids": [],
            "stage": "QUEUED",
            "status": "running",
            "handback_count": 0,
        },
        config,
    )

    completed = graph.invoke(None, config)
    assert completed["stage"] == "UNRESOLVED"
    assert completed["handback_count"] == 2
    assert completed["status"] == "completed"


def test_budget_exhaustion_stops_before_a_fake_model_call() -> None:
    router = FakeRouter(cost_per_call=Decimal("0.02"))
    graph = build_research_graph(router, run_ctx=_context(max_usd="0.01"))
    with pytest.raises(BudgetExceeded):
        graph.invoke(
            {
                "run_id": "synthetic-budget-run",
                "dossier_ids": [],
                "stage": "QUEUED",
                "status": "running",
                "handback_count": 0,
            }
        )
    assert router.calls == 0
