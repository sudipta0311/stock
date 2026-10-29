"""Live smoke check for the two configured Foundry model families."""

from __future__ import annotations

import json
from decimal import Decimal

from pydantic import BaseModel

from wealth.research.router import ModelRouter, RunBudget, RunContext


class SmokeResponse(BaseModel):
    summary: str
    ready: bool


def main() -> int:
    router = ModelRouter()
    for tier in ("nano", "redteam"):
        context = RunContext(
            budget=RunBudget(max_usd=Decimal("1"), max_calls=2),
            mode="weekly",
        )
        result = router.complete(
            tier,
            [
                {"role": "system", "content": "Return a short structured health response."},
                {"role": "user", "content": "Confirm the model endpoint can return the schema."},
            ],
            schema=SmokeResponse,
            max_output_tokens=64,
            run_ctx=context,
        )
        print(
            json.dumps(
                {
                    "model": result.model,
                    "tier": tier,
                    "tokens": result.usage,
                    "cost_usd": format(result.cost_usd, "f"),
                    "latency_ms": result.latency_ms,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
