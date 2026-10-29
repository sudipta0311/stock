"""Resumable, checkpointed research stage graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from wealth.research.router import FakeRouter, ModelRouter, RunBudget, RunContext
from wealth.research.tools import EVIDENCE_SYSTEM_PROMPT


class ResearchState(TypedDict, total=False):
    run_id: str
    dossier_ids: list[str]
    budget: dict[str, Any]
    stage: str
    status: str
    handback_count: int


class StageDecision(BaseModel):
    proceed: bool = True
    handoff_to: Literal["analyse", "signoff"] | None = None
    verdict: Literal["APPROVED", "VETOED", "UNRESOLVED"] = "APPROVED"


_NODE_SPECS: dict[str, tuple[str, str, str]] = {
    "plan": ("nano", "plan_agent", "Choose the next research stage."),
    "screen": ("nano", "screen_agent", "Record a qualitative screen decision."),
    "analyse": ("mini", "analysis_agent", "Review cited evidence."),
    "value": ("mini", "valuation_agent", "Check evidence completeness; do not invent amounts."),
    "red_team": ("redteam", "red_team_agent", "Find the strongest counterargument."),
    "compliance": ("nano", "compliance_agent", "Check evidence and policy compliance."),
    "signoff": ("strong", "cio_agent", "Return a qualitative sign-off verdict."),
}


def _budget_snapshot(budget: RunBudget) -> dict[str, Any]:
    return {
        "max_usd": format(budget.max_usd, "f"),
        "spent_usd": format(budget.spent_usd, "f"),
        "max_calls": budget.max_calls,
        "calls": budget.calls,
    }


def build_research_graph(
    router: ModelRouter | FakeRouter,
    *,
    run_ctx: RunContext | None = None,
    checkpointer: Any | None = None,
    interrupt_after_screen: bool = True,
) -> Any:
    """Compile the stage pipeline, with an optional Batch/Flex wait after screening."""
    context = run_ctx or RunContext(
        budget=RunBudget(max_usd=100, max_calls=10_000),
        mode="weekly",
    )
    builder: StateGraph[ResearchState] = StateGraph(ResearchState)

    def make_node(node_name: str):
        tier, agent, instruction = _NODE_SPECS[node_name]

        def run_node(state: ResearchState) -> Command:
            context.agent = agent
            call = router.complete(
                tier,
                [
                    {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                schema=StageDecision,
                max_output_tokens=64,
                run_ctx=context,
            )
            state_budget = _budget_snapshot(context.budget)
            if node_name == "plan":
                return Command(goto="screen", update={"stage": "QUEUED", "budget": state_budget})
            if node_name == "screen":
                return Command(
                    goto="analyse",
                    update={"stage": "SCREENED", "status": "waiting", "budget": state_budget},
                )
            if node_name == "analyse":
                return Command(goto="value", update={"stage": "ANALYSED", "budget": state_budget})
            if node_name == "value":
                return Command(goto="red_team", update={"stage": "VALUED", "budget": state_budget})
            if node_name == "red_team":
                return Command(
                    goto="compliance",
                    update={"stage": "RED_TEAM_DONE", "budget": state_budget},
                )
            if node_name == "compliance":
                decision = call.parsed
                handbacks = int(state.get("handback_count", 0))
                if decision.handoff_to == "analyse":
                    if handbacks >= 2:
                        return Command(
                            goto="signoff",
                            update={
                                "stage": "UNRESOLVED",
                                "status": "running",
                                "handback_count": handbacks,
                                "budget": _budget_snapshot(context.budget),
                            },
                        )
                    return Command(
                        goto="analyse",
                        update={
                            "stage": "ANALYSED",
                            "status": "running",
                            "handback_count": handbacks + 1,
                            "budget": _budget_snapshot(context.budget),
                        },
                    )
                return Command(
                    goto="signoff",
                    update={"stage": "COMPLIANCE_DONE", "budget": state_budget},
                )
            if node_name == "signoff":
                stage = state.get("stage")
                final_stage = "UNRESOLVED" if stage == "UNRESOLVED" else call.parsed.verdict
                return Command(
                    goto=END,
                    update={
                        "stage": final_stage,
                        "status": "completed",
                        "budget": state_budget,
                    },
                )
            raise RuntimeError(f"Unknown research node: {node_name}")

        return run_node

    for node_name in _NODE_SPECS:
        builder.add_node(node_name, make_node(node_name))
    builder.set_entry_point("plan")
    interrupt_after = ["screen"] if interrupt_after_screen else []
    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)
