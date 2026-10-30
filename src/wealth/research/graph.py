"""Resumable weekly agent graph with bounded dossiers and pack sign-off."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, TypedDict
from uuid import UUID

from langgraph.graph import END, StateGraph

from wealth.policy.actions import SatelliteIdea
from wealth.research.agents import (
    ResearchTeam,
    dossier_compliance_failures,
)
from wealth.research.dossier import (
    Dossier,
    DossierBudget,
    DossierSecurity,
    Stage,
    hand_back_to_analysis,
)
from wealth.research.errors import BudgetExceeded, SchemaError
from wealth.research.router import FakeRouter, ModelRouter, RunBudget, RunContext
from wealth.research.rules import ResearchRules
from wealth.research.store import (
    load_prior_geo_tilts,
    save_dossier,
    save_macro_geo_pack,
    save_research_pack,
)
from wealth.research.tools import ToolRegistry


class ResearchState(TypedDict, total=False):
    run_id: str
    week: str
    shortlist: list[dict[str, Any]]
    shortlist_ids: list[str]
    dossiers: list[dict[str, Any]]
    tool_results_by_dossier: dict[str, dict[str, dict[str, Any]]]
    pack: dict[str, Any]
    summary: str
    budget: dict[str, Any]
    stage: str
    status: str


def _budget_snapshot(budget: RunBudget) -> dict[str, Any]:
    return {
        "max_usd": format(budget.max_usd, "f"),
        "spent_usd": format(budget.spent_usd, "f"),
        "max_calls": budget.max_calls,
        "calls": budget.calls,
    }


def _serialise_dossier(dossier: Dossier) -> dict[str, Any]:
    return dossier.model_dump(mode="json", by_alias=True)


def _load_dossiers(state: ResearchState) -> list[Dossier]:
    return [Dossier.model_validate(item) for item in state.get("dossiers", [])]


def _candidate_map(state: ResearchState) -> dict[str, dict[str, Any]]:
    return {str(item["dossier_id"]): item for item in state.get("shortlist", [])}


def _budget_for(dossier: Dossier, rules: ResearchRules, context: RunContext) -> None:
    dossier.budget.max_usd = rules.max_usd_per_dossier
    dossier.budget.max_calls = max(dossier.budget.max_calls, 100)
    context.dossier_budget = dossier.budget
    context.dossier_id = dossier.dossier_id


def _with_budget(dossier: Dossier, context: RunContext) -> Dossier:
    confidence = dossier.confidence
    if dossier.stage == Stage.VALUED and dossier.verdicts.red_team is None:
        confidence = _base_confidence(dossier)
    elif dossier.verdicts.red_team is not None:
        confidence = min(confidence, _confidence_for(dossier))
    return dossier.model_copy(
        update={
            "budget": context.dossier_budget or dossier.budget,
            "confidence": confidence,
        }
    )


def _base_confidence(dossier: Dossier) -> Decimal:
    if dossier.verdicts.fundamental == "POSITIVE" and dossier.verdicts.valuation in {
        "CHEAP",
        "FAIR",
    }:
        return Decimal("0.9")
    if dossier.verdicts.fundamental == "NEUTRAL" and dossier.verdicts.valuation in {
        "CHEAP",
        "FAIR",
    }:
        return Decimal("0.6")
    return Decimal("0.2")


def _confidence_for(dossier: Dossier) -> Decimal:
    fundamental = dossier.verdicts.fundamental
    valuation = dossier.verdicts.valuation
    red_team = dossier.verdicts.red_team.downgrade if dossier.verdicts.red_team else None
    if fundamental == "POSITIVE" and valuation in {"CHEAP", "FAIR"} and red_team == "NONE":
        return Decimal("0.9")
    if (
        fundamental in {"POSITIVE", "NEUTRAL"}
        and valuation in {"CHEAP", "FAIR"}
        and red_team in {"NONE", "WATCH"}
    ):
        return Decimal("0.6")
    return Decimal("0.2")


def _save_if_configured(
    engine: Any | None,
    dossier: Dossier,
    candidate: dict[str, Any] | None,
) -> None:
    if engine is None:
        return
    security_id: UUID | None = None
    if candidate is not None:
        try:
            security_id = UUID(str(candidate["security_id"]))
        except (KeyError, ValueError):
            security_id = None
    save_dossier(engine, dossier, security_id=security_id)


def _mark_unresolved(dossier: Dossier, reason: str, context: RunContext) -> Dossier:
    budget = context.dossier_budget or dossier.budget
    return dossier.model_copy(
        update={
            "stage": Stage.UNRESOLVED,
            "status": "UNRESOLVED",
            "status_reason": reason,
            "budget": budget,
            "confidence": Decimal("0.2"),
        }
    )


def _make_dossier(candidate: dict[str, Any], rules: ResearchRules) -> Dossier:
    raw = candidate.get("dossier")
    if isinstance(raw, dict):
        dossier = Dossier.model_validate(raw)
    else:
        dossier_id = str(candidate["dossier_id"])
        dossier = Dossier(
            dossier_id=dossier_id,
            security=DossierSecurity(
                id=str(candidate["security_id"]),
                market=str(candidate["market"]),
                type="EQUITY",
            ),
            stage=Stage.SCREENED,
        )
    budget = DossierBudget(
        max_usd=rules.max_usd_per_dossier,
        max_calls=100,
        spent_usd=dossier.budget.spent_usd,
        calls=dossier.budget.calls,
    )
    return dossier.model_copy(update={"stage": Stage.SCREENED, "budget": budget})


def build_research_graph(
    router: ModelRouter | FakeRouter,
    *,
    run_ctx: RunContext | None = None,
    owner_engine: Any | None = None,
    tools: ToolRegistry | None = None,
    shortlist: list[dict[str, Any]] | None = None,
    rules: ResearchRules | None = None,
    checkpointer: Any | None = None,
    interrupt_after_screen: bool = True,
    interrupt_after_red_team: bool = True,
) -> Any:
    """Compile plan → screen → analyst gates → signed-off research pack."""
    context = run_ctx or RunContext(
        budget=RunBudget(max_usd=Decimal("5"), max_calls=10_000),
        mode="weekly",
    )
    selected_rules = rules or ResearchRules.load()
    registry = tools or ToolRegistry(owner_engine)
    team = ResearchTeam(router, registry, context, rules=selected_rules)
    source_shortlist = shortlist or []
    builder: StateGraph[ResearchState] = StateGraph(ResearchState)

    def plan_node(_state: ResearchState) -> dict[str, Any]:
        return {"stage": "QUEUED", "status": "running", "budget": _budget_snapshot(context.budget)}

    def screen_node(state: ResearchState) -> dict[str, Any]:
        incoming = state.get("shortlist", source_shortlist)
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_candidate in incoming[: selected_rules.max_dossiers_per_run]:
            candidate = dict(raw_candidate)
            dossier_id = str(candidate.get("dossier_id", ""))
            if not dossier_id or dossier_id in seen:
                continue
            seen.add(dossier_id)
            candidate["dossier_id"] = dossier_id
            dossier = _make_dossier(candidate, selected_rules)
            candidate["dossier"] = _serialise_dossier(dossier)
            selected.append(candidate)
            _save_if_configured(owner_engine, dossier, candidate)
        evidence = {
            candidate["dossier_id"]: dict(candidate.get("tool_results", {}))
            for candidate in selected
        }
        team._active_tool_results = evidence
        return {
            "shortlist": selected,
            "shortlist_ids": [candidate["dossier_id"] for candidate in selected],
            "dossiers": [candidate["dossier"] for candidate in selected],
            "tool_results_by_dossier": evidence,
            "stage": "SCREENED",
            "status": "waiting",
            "budget": _budget_snapshot(context.budget),
        }

    def analyse_node(state: ResearchState) -> dict[str, Any]:
        candidates = _candidate_map(state)
        evidence = state.get("tool_results_by_dossier", {})
        dossiers: list[Dossier] = []
        for dossier in _load_dossiers(state):
            candidate = candidates[dossier.dossier_id]
            if dossier.status in {"UNRESOLVED", "VETOED"}:
                dossiers.append(dossier)
                continue
            _budget_for(dossier, selected_rules, context)
            try:
                dossier, tool_results = team.analyse(
                    candidate,
                    dossier,
                    dict(evidence.get(dossier.dossier_id, {})),
                    feedback=dossier.compliance_failures,
                )
                evidence[dossier.dossier_id] = tool_results
                dossier = _with_budget(dossier, context)
            except BudgetExceeded:
                dossier = _mark_unresolved(dossier, "budget", context)
            except (SchemaError, ValueError):
                dossier = _mark_unresolved(dossier, "analyst_schema", context)
            _save_if_configured(owner_engine, dossier, candidate)
            dossiers.append(dossier)
        return {
            "dossiers": [_serialise_dossier(item) for item in dossiers],
            "tool_results_by_dossier": evidence,
            "stage": "ANALYSED",
            "budget": _budget_snapshot(context.budget),
        }

    def value_node(state: ResearchState) -> dict[str, Any]:
        candidates = _candidate_map(state)
        evidence = state.get("tool_results_by_dossier", {})
        dossiers: list[Dossier] = []
        for dossier in _load_dossiers(state):
            candidate = candidates[dossier.dossier_id]
            if dossier.status in {"UNRESOLVED", "VETOED"}:
                dossiers.append(dossier)
                continue
            _budget_for(dossier, selected_rules, context)
            try:
                dossier, tool_results = team.value(
                    candidate,
                    dossier,
                    dict(evidence.get(dossier.dossier_id, {})),
                )
                evidence[dossier.dossier_id] = tool_results
                dossier = _with_budget(dossier, context)
                if dossier.verdicts.valuation == "UNKNOWN":
                    dossier = dossier.model_copy(
                        update={
                            "stage": Stage.UNRESOLVED,
                            "status": "UNRESOLVED",
                            "status_reason": "valuation_unknown",
                        }
                    )
            except BudgetExceeded:
                dossier = _mark_unresolved(dossier, "budget", context)
            except (SchemaError, ValueError):
                dossier = _mark_unresolved(dossier, "valuation_schema", context)
            _save_if_configured(owner_engine, dossier, candidate)
            dossiers.append(dossier)
        return {
            "dossiers": [_serialise_dossier(item) for item in dossiers],
            "tool_results_by_dossier": evidence,
            "stage": "VALUED",
            "budget": _budget_snapshot(context.budget),
        }

    def red_team_node(state: ResearchState) -> dict[str, Any]:
        candidates = _candidate_map(state)
        evidence = state.get("tool_results_by_dossier", {})
        dossiers: list[Dossier] = []
        for dossier in _load_dossiers(state):
            candidate = candidates[dossier.dossier_id]
            if dossier.status in {"UNRESOLVED", "VETOED"}:
                dossiers.append(dossier)
                continue
            _budget_for(dossier, selected_rules, context)
            try:
                dossier, tool_results = team.red_team(
                    candidate,
                    dossier,
                    dict(evidence.get(dossier.dossier_id, {})),
                )
                evidence[dossier.dossier_id] = tool_results
                dossier = _with_budget(dossier, context)
            except BudgetExceeded:
                dossier = _mark_unresolved(dossier, "budget", context)
            except (SchemaError, ValueError):
                dossier = _mark_unresolved(dossier, "red_team_schema", context)
            _save_if_configured(owner_engine, dossier, candidate)
            dossiers.append(dossier)
        team._active_tool_results = evidence
        return {
            "dossiers": [_serialise_dossier(item) for item in dossiers],
            "tool_results_by_dossier": evidence,
            "stage": "RED_TEAM_DONE",
            "status": "waiting",
            "budget": _budget_snapshot(context.budget),
        }

    def compliance_node(state: ResearchState) -> dict[str, Any]:
        candidates = _candidate_map(state)
        evidence = state.get("tool_results_by_dossier", {})
        dossiers: list[Dossier] = []
        for initial in _load_dossiers(state):
            dossier = initial
            candidate = candidates[dossier.dossier_id]
            if dossier.status in {"UNRESOLVED", "VETOED"}:
                dossiers.append(dossier)
                continue
            handbacks = 0
            while True:
                _budget_for(dossier, selected_rules, context)
                dossier = dossier.model_copy(update={"stage": Stage.COMPLIANCE_DONE})
                tool_results = dict(evidence.get(dossier.dossier_id, {}))
                names = [str(candidate.get("name", "")), str(candidate.get("symbol", ""))]
                failures = dossier_compliance_failures(
                    dossier,
                    tool_results,
                    security_names=names,
                )
                if failures:
                    dossier = dossier.model_copy(update={"compliance_failures": failures})
                    if handbacks >= 2:
                        dossier = _mark_unresolved(dossier, "compliance_failed", context)
                        break
                    try:
                        dossier = hand_back_to_analysis(
                            dossier,
                            reason="; ".join(failures[:8]),
                            budget_remaining_tokens=0,
                        )
                        handbacks += 1
                        dossier, tool_results = team.analyse(
                            candidate,
                            dossier,
                            tool_results,
                            feedback=failures,
                        )
                        dossier, tool_results = team.value(candidate, dossier, tool_results)
                        if dossier.verdicts.valuation == "UNKNOWN":
                            dossier = _mark_unresolved(dossier, "valuation_unknown", context)
                            break
                        dossier, tool_results = team.red_team(
                            candidate,
                            dossier,
                            tool_results,
                        )
                        evidence[dossier.dossier_id] = tool_results
                        dossier = _with_budget(dossier, context)
                    except BudgetExceeded:
                        dossier = _mark_unresolved(dossier, "budget", context)
                        break
                    except (SchemaError, ValueError):
                        dossier = _mark_unresolved(dossier, "handback_schema", context)
                        break
                    continue
                try:
                    answer = team.compliance_answer(dossier)
                except BudgetExceeded:
                    dossier = _mark_unresolved(dossier, "budget", context)
                    break
                except (SchemaError, ValueError):
                    dossier = _mark_unresolved(dossier, "compliance_schema", context)
                    break
                if answer.has_personalised_advice_or_amount:
                    dossier = dossier.model_copy(
                        update={
                            "status": "VETOED",
                            "status_reason": "personalised_advice_or_amount",
                            "stage": Stage.COMPLIANCE_DONE,
                            "compliance_failures": [],
                        }
                    )
                else:
                    dossier = dossier.model_copy(
                        update={
                            "stage": Stage.COMPLIANCE_DONE,
                            "compliance_failures": [],
                            "budget": context.dossier_budget or dossier.budget,
                        }
                    )
                break
            _save_if_configured(owner_engine, dossier, candidate)
            dossiers.append(dossier)
        team._active_tool_results = evidence
        return {
            "dossiers": [_serialise_dossier(item) for item in dossiers],
            "tool_results_by_dossier": evidence,
            "stage": "COMPLIANCE_DONE",
            "budget": _budget_snapshot(context.budget),
        }

    def signoff_node(state: ResearchState) -> dict[str, Any]:
        candidates = _candidate_map(state)
        dossiers = _load_dossiers(state)
        shortlist_ids = set(state.get("shortlist_ids", []))
        team._active_tool_results = state.get("tool_results_by_dossier", {})
        try:
            output, selected_ids = team.cio_signoff(dossiers, candidates, shortlist_ids)
        except BudgetExceeded:
            output = None
            selected_ids = []
            signoff_budget_exhausted = True
        except (SchemaError, ValueError):
            output = None
            selected_ids = []
            signoff_budget_exhausted = False
        else:
            signoff_budget_exhausted = False
        selected = set(selected_ids)
        final_dossiers: list[Dossier] = []
        for dossier in dossiers:
            if dossier.status in {"UNRESOLVED", "VETOED"}:
                final_dossiers.append(dossier)
                continue
            if signoff_budget_exhausted and team._cio_eligible(dossier):
                updated = dossier.model_copy(
                    update={
                        "stage": Stage.UNRESOLVED,
                        "status": "UNRESOLVED",
                        "status_reason": "budget",
                        "confidence": Decimal("0.2"),
                    }
                )
                _save_if_configured(owner_engine, updated, candidates.get(dossier.dossier_id))
                final_dossiers.append(updated)
                continue
            if team._cio_eligible(dossier) and dossier.dossier_id in selected:
                status = "APPROVED"
                reason = None
                confidence = _confidence_for(dossier)
            elif team._cio_eligible(dossier):
                status = "VETOED"
                reason = "not_selected_by_cio"
                confidence = Decimal("0.2")
            elif (
                dossier.verdicts.fundamental == "NEGATIVE"
                or dossier.verdicts.valuation in {"EXPENSIVE", "UNKNOWN"}
                or (
                    dossier.verdicts.red_team is not None
                    and dossier.verdicts.red_team.downgrade == "VETO"
                )
            ):
                status = "VETOED"
                reason = dossier.status_reason or "code_gate"
                confidence = Decimal("0.2")
            else:
                status = "UNRESOLVED"
                reason = "signoff_rejected"
                confidence = Decimal("0.2")
            updated = dossier.model_copy(
                update={
                    "stage": (
                        Stage.APPROVED
                        if status == "APPROVED"
                        else Stage.VETOED
                        if status == "VETOED"
                        else Stage.UNRESOLVED
                    ),
                    "status": status,
                    "status_reason": reason,
                    "confidence": confidence,
                }
            )
            _save_if_configured(owner_engine, updated, candidates.get(dossier.dossier_id))
            final_dossiers.append(updated)
        summary = output.summary if output is not None else "No ideas passed the checks this week."
        return {
            "dossiers": [_serialise_dossier(item) for item in final_dossiers],
            "summary": summary,
            "stage": "SIGNED_OFF",
            "budget": _budget_snapshot(context.budget),
        }

    def pack_node(state: ResearchState) -> dict[str, Any]:
        candidates = _candidate_map(state)
        dossiers = _load_dossiers(state)
        shortlist_ids = set(state.get("shortlist_ids", []))
        approved_ids = {dossier.dossier_id for dossier in dossiers if dossier.status == "APPROVED"}
        if not approved_ids <= shortlist_ids:
            raise RuntimeError("Approved ideas must be a subset of the screener shortlist")
        ideas: list[dict[str, Any]] = []
        for dossier in dossiers:
            if dossier.dossier_id not in approved_ids:
                continue
            candidate = candidates[dossier.dossier_id]
            value = {
                "security_id": dossier.security.id,
                "symbol": str(candidate["symbol"]),
                "market": dossier.security.market,
                "sector": str(candidate["sector"]),
                "price": str(candidate["price"]),
                "quant_decile": int(candidate["quant_decile"]),
                "dossier_status": "APPROVED",
                "valuation_percentile": format(dossier.valuation_percentile or Decimal("50"), "f"),
                "confidence": (
                    "HIGH"
                    if dossier.confidence >= Decimal("0.8")
                    else "MEDIUM"
                    if dossier.confidence >= Decimal("0.5")
                    else "LOW"
                ),
            }
            idea = SatelliteIdea.model_validate(value)
            ideas.append(idea.model_dump(mode="json", exclude_unset=True))
        week = str(state.get("week", ""))
        run_id = str(state.get("run_id", ""))
        markets = {
            market: {
                "idea_count": sum(1 for idea in ideas if idea["market"] == market),
                "status": "complete",
            }
            for market in ("IN", "US")
        }
        pack = {
            "ideas": ideas,
            "summary": str(state.get("summary", "No ideas passed the checks this week.")),
            "run_id": run_id,
            "markets": markets,
        }
        if owner_engine is not None:
            save_research_pack(owner_engine, week, pack)
            from datetime import date

            from wealth.research.agents.macro_geo import (
                MacroGeoAgent,
                register_macro_geo_tools,
            )

            macro_date = (
                date.fromisocalendar(int(week[:4]), int(week[-2:]), 5)
                if "-W" in week
                else date.today()
            )
            remaining_usd = max(Decimal(0), context.budget.max_usd - context.budget.spent_usd)
            macro_budget = RunBudget(
                max_usd=min(selected_rules.macro_max_usd, remaining_usd),
                max_calls=9,
            )
            macro_context = RunContext(
                budget=macro_budget,
                run_id=context.run_id,
                mode="weekly",
                owner_engine=owner_engine,
            )
            macro_tools = ToolRegistry(owner_engine)
            try:
                macro_toolbox = register_macro_geo_tools(macro_tools, owner_engine, macro_date)
                macro_result = MacroGeoAgent(
                    router,
                    macro_tools,
                    macro_context,
                    as_of=macro_date,
                    week=week,
                    toolbox=macro_toolbox,
                ).run(previous=load_prior_geo_tilts(owner_engine, week))
            finally:
                macro_tools.close()
            context.budget.spent_usd += macro_budget.spent_usd
            context.budget.calls += macro_budget.calls
            if context.on_budget_change is not None:
                context.on_budget_change(context.budget.spent_usd)
            macro_fields = macro_result.pack_fields()
            save_macro_geo_pack(owner_engine, week, macro_fields)
            pack.update(macro_fields)
        return {
            "pack": pack,
            "dossiers": [_serialise_dossier(item) for item in dossiers],
            "stage": "PACKED",
            "status": "completed",
            "budget": _budget_snapshot(context.budget),
        }

    builder.add_node("plan", plan_node)
    builder.add_node("screen", screen_node)
    builder.add_node("analyse", analyse_node)
    builder.add_node("value", value_node)
    builder.add_node("red_team", red_team_node)
    builder.add_node("compliance", compliance_node)
    builder.add_node("signoff", signoff_node)
    builder.add_node("pack", pack_node)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "screen")
    builder.add_edge("screen", "analyse")
    builder.add_edge("analyse", "value")
    builder.add_edge("value", "red_team")
    builder.add_edge("red_team", "compliance")
    builder.add_edge("compliance", "signoff")
    builder.add_edge("signoff", "pack")
    builder.add_edge("pack", END)
    interrupt_after: list[str] = []
    if interrupt_after_screen:
        interrupt_after.append("screen")
    if interrupt_after_red_team:
        interrupt_after.append("red_team")
    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)
