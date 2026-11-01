from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from wealth.db.models import ResearchPackRecord
from wealth.policy import data
from wealth.policy.actions import satellite_idea_from_pack


def _pack_idea(**extra: Any) -> dict[str, Any]:
    return {
        "security_id": str(uuid4()),
        "symbol": "SYNTH",
        "market": "IN",
        "sector": "Technology",
        "price": "123.45",
        "quant_decile": 7,
        "dossier_status": "APPROVED",
        "valuation_percentile": "40",
        "confidence": "MEDIUM",
        **extra,
    }


def test_pack_idea_ignores_display_fields_and_joined_fields() -> None:
    idea = satellite_idea_from_pack(
        _pack_idea(
            why={"reasons": ["A synthetic reason."], "risk": "A synthetic risk."},
            entry_hint={"code": "buy_in_parts", "parts": 3, "reasons": []},
            valuation_label="fair",
            name="Should come from securities, not the pack",
            future_display_field=True,
        )
    )

    assert idea.symbol == "SYNTH"
    assert idea.name is None


@dataclass
class _FakeSession:
    record: ResearchPackRecord

    def get(self, model: type, key: object) -> object | None:
        return self.record if model is ResearchPackRecord else None


def test_plan_reads_a_pack_whose_ideas_carry_why_and_hints() -> None:
    # Packs written since C37 carry `why`; the plan loader rejected them and every
    # plan rebuild for that week failed.
    record = ResearchPackRecord(
        week="2026-W44",
        pack={
            "ideas": [
                _pack_idea(
                    why={"reasons": ["A synthetic reason."], "risk": "A synthetic risk."},
                    entry_hint={"code": "buy_in_parts", "parts": 3, "reasons": []},
                )
            ]
        },
    )

    pack = data._research_pack(_FakeSession(record), "2026-W44")  # type: ignore[arg-type]

    assert pack is not None
    assert [idea.symbol for idea in pack.ideas] == ["SYNTH"]
