"""Live smoke for the research graph: real Foundry models, synthetic companies, no database.

Runs analyse -> value -> red team -> compliance -> sign-off for a small synthetic shortlist
with synthetic tool data, then prints each dossier's status, the cost and the call count.
Keys come from FOUNDRY_* environment variables and are never printed. Not part of CI.

    python scripts/smoke_research.py --market IN --limit 2 --budget-usd 0.25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

from wealth.research.graph import build_research_graph
from wealth.research.router import ModelRouter, RunBudget, RunContext
from wealth.research.rules import ResearchRules
from wealth.research.tools import ToolRegistry

_SYMBOLS = ("ALPHA", "BETA", "GAMMA")


def _candidates(market: str, limit: int, week: str) -> list[dict[str, Any]]:
    return [
        {
            "dossier_id": f"{week}/{market}/{symbol}",
            "security_id": f"smoke-security-{symbol.lower()}",
            "symbol": symbol,
            "name": f"Sample {symbol} Industries",
            "market": market,
            "sector": "Industrials",
            "as_of": "2026-10-30",
            "price": "250.00",
            "quant_decile": 10,
            "composite": str(Decimal("2.5") - Decimal(index) / 2),
        }
        for index, symbol in enumerate(_SYMBOLS[:limit])
    ]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    def factor_snapshot(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        return (
            {
                "security_id": args["security_id"],
                "roce_pct": "21.4",
                "roe_pct": "17.9",
                "debt_to_equity": "0.32",
                "revenue_growth_3y_pct": "11.2",
                "momentum_12_1_pct": "14.0",
                "volatility_1y_pct": "24.5",
                "note": "Synthetic company used for a smoke test.",
            },
            "https://example.test/factors",
        )

    def valuation_stats(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        return (
            {
                "security_id": args["security_id"],
                "market": "IN",
                "price": "250.00",
                "pe_percentile": "40",
                "earnings_yield_pct": "5.5",
                "sector_earnings_yield_median_pct": "4.8",
                "earnings_yield_vs_sector": "1.15",
                "fcf_yield_pct": "3.9",
                "reverse_dcf_implied_growth_pct": "8.0",
                "discount_rate_pct": "12",
                "verdict": "FAIR",
            },
            "https://example.test/valuation",
        )

    registry.register("factor_snapshot", factor_snapshot)
    registry.register("valuation_stats", valuation_stats)
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="IN", choices=["IN", "US"])
    parser.add_argument("--limit", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--budget-usd", default="0.25")
    parser.add_argument("--week", default="2026-W44")
    args = parser.parse_args(argv)
    if not (os.getenv("FOUNDRY_ENDPOINT") and os.getenv("FOUNDRY_API_KEY")):
        print("Set FOUNDRY_ENDPOINT and FOUNDRY_API_KEY to run the live smoke.", file=sys.stderr)
        return 2

    candidates = _candidates(args.market, args.limit, args.week)
    budget = RunBudget(max_usd=Decimal(args.budget_usd), max_calls=200)
    run_id = str(uuid4())
    # "smoke" is not a Flex mode, so calls don't queue for minutes.
    context = RunContext(budget=budget, run_id=run_id, mode="smoke", agent="research")
    graph = build_research_graph(
        ModelRouter(),
        run_ctx=context,
        tools=_registry(),
        shortlist=candidates,
        rules=ResearchRules.load(),
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "research-smoke"}}
    state: dict[str, Any] | None = {
        "run_id": run_id,
        "week": args.week,
        "shortlist": candidates,
        "shortlist_ids": [],
        "dossiers": [],
        "tool_results_by_dossier": {},
    }
    result: dict[str, Any] = {}
    for _ in range(6):
        result = graph.invoke(state, config)
        state = None
        if result.get("stage") in {"PACKED", "DONE"} or result.get("status") == "complete":
            break

    summary = {
        "stage": result.get("stage"),
        "dossiers": [
            {
                "symbol": item.get("symbol") or item.get("dossier_id"),
                "status": item.get("status"),
                "confidence": item.get("confidence"),
                "reasons": item.get("reasons") or item.get("veto_reasons"),
            }
            for item in result.get("dossiers", [])
        ],
        "ideas": len((result.get("pack") or {}).get("ideas", [])),
        "spent_usd": str(budget.spent_usd),
        "calls": budget.calls,
    }
    if os.getenv("SMOKE_VERBOSE"):
        # Synthetic companies only, so the full dossiers are safe to show.
        summary["raw_dossiers"] = result.get("dossiers", [])
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
