"""Bounded weekly CIO agenda; all counts and budgets come from code and rules."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from wealth.core.plain_language import find_banned_terms, sentence_word_counts
from wealth.research.errors import BudgetExceeded
from wealth.research.router import FakeRouter, ModelRouter, RunContext
from wealth.research.rules import ResearchRules


class AgendaCopy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    focus: list[str] = Field(default_factory=list, max_length=3)
    note: str = Field(default="", max_length=200)


def _plain_items(items: list[str]) -> list[str]:
    accepted: list[str] = []
    for value in items[:3]:
        text = value.strip()
        if not text or len(text.split()) > 20 or any(char.isdigit() for char in text):
            continue
        if find_banned_terms(text) or any(count > 20 for count in sentence_word_counts(text)):
            continue
        accepted.append(text)
    return accepted


def build_cio_agenda(
    *,
    week: str,
    rules: ResearchRules,
    run_ctx: RunContext,
    router: ModelRouter | FakeRouter | None,
    scorekeeper_checklist: list[str],
    regime_notes: list[str],
) -> dict[str, Any]:
    """Use a model only for short focus wording; code sets every bounded number."""
    markets = ("IN", "US")
    quotient, remainder = divmod(rules.max_dossiers_per_run, len(markets))
    limits: dict[str, dict[str, Any]] = {}
    for index, market in enumerate(markets):
        dossier_limit = quotient + (1 if index < remainder else 0)
        idea_limit = min(rules.max_ideas_per_market, dossier_limit)
        budget = min(
            run_ctx.budget.max_usd / Decimal(len(markets)),
            rules.max_usd_per_dossier * Decimal(dossier_limit),
        )
        limits[market] = {
            "max_dossiers": dossier_limit,
            "max_ideas": idea_limit,
            "budget_usd": format(budget, "f"),
        }

    focus = _plain_items(scorekeeper_checklist)
    note = "Weekly review uses saved evidence and bounded research limits."
    if router is not None:
        try:
            response = router.complete(
                "mini",
                [
                    {
                        "role": "system",
                        "content": (
                            "Write up to three short research focus items from the supplied "
                            "checklist and regime notes. Use no numbers. Do not set budgets, "
                            "choose securities, or decide idea status."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Checklist: {scorekeeper_checklist[:3]}\n"
                            f"Regime notes: {regime_notes[:4]}"
                        ),
                    },
                ],
                schema=AgendaCopy,
                max_output_tokens=350,
                run_ctx=run_ctx,
            )
            result = (
                response.parsed
                if isinstance(response.parsed, AgendaCopy)
                else AgendaCopy.model_validate(response.parsed)
            )
            accepted = _plain_items(result.focus)
            accepted_note = _plain_items([result.note])
            if accepted:
                focus = accepted
            if accepted_note:
                note = accepted_note[0]
        except (BudgetExceeded, ValueError, TypeError):
            pass
    return {
        "week": week,
        "market_limits": limits,
        "focus": focus,
        "note": note,
        "max_dossiers": rules.max_dossiers_per_run,
        "max_ideas_per_market": rules.max_ideas_per_market,
    }
