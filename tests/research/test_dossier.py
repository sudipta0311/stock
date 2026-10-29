from __future__ import annotations

import pytest
from pydantic import ValidationError

from wealth.research.dossier import (
    Claim,
    Dossier,
    Evidence,
    Stage,
    hand_back_to_analysis,
    transition_stage,
    validate_grounding,
)


def _dossier(*, stage: Stage = Stage.QUEUED, claims: list[Claim] | None = None) -> Dossier:
    return Dossier(
        dossier_id="2026-W45/IN/SYNTHETIC",
        security={"id": "IN:SYNTHETIC", "market": "IN", "type": "EQUITY"},
        stage=stage,
        claims=claims or [],
    )


def test_fact_claim_requires_a_tool_call_reference() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        Claim(text="Revenue increased.", kind="FACT")


def test_transition_rejects_skipped_stages() -> None:
    with pytest.raises(ValueError, match="Illegal dossier transition"):
        transition_stage(_dossier(), Stage.ANALYSED)


def test_transition_allows_the_next_stage() -> None:
    updated = transition_stage(_dossier(), Stage.SCREENED)
    assert updated.stage == Stage.SCREENED


def test_third_handback_closes_dossier_as_unresolved() -> None:
    dossier = _dossier(stage=Stage.COMPLIANCE_DONE)
    dossier = hand_back_to_analysis(dossier, reason="Missing source")
    assert dossier.stage == Stage.ANALYSED
    dossier = dossier.model_copy(update={"stage": Stage.COMPLIANCE_DONE})
    dossier = hand_back_to_analysis(dossier, reason="Needs one more check")
    assert dossier.stage == Stage.ANALYSED
    dossier = dossier.model_copy(update={"stage": Stage.COMPLIANCE_DONE})
    dossier = hand_back_to_analysis(dossier, reason="Evidence remains incomplete")
    assert dossier.stage == Stage.UNRESOLVED
    assert dossier.handback_count == 2


def _fact_claim(text: str, tool_call_id: str = "tool-1") -> Claim:
    return Claim(
        id="claim-1",
        text=text,
        kind="FACT",
        evidence=[Evidence(tool_call_id=tool_call_id)],
    )


def test_grounding_accepts_a_number_in_the_cited_result() -> None:
    dossier = _dossier(claims=[_fact_claim("ROCE improved to 16.8%")])
    failures = validate_grounding(dossier, {"tool-1": {"roce": "16.8"}})
    assert failures == []


def test_grounding_rejects_a_number_absent_from_the_cited_result() -> None:
    dossier = _dossier(claims=[_fact_claim("ROCE improved to 16.8%")])
    failures = validate_grounding(dossier, {"tool-1": {"roce": "16.5"}})
    assert len(failures) == 1
    assert "16.8" in failures[0]


def test_grounding_rejects_unknown_tool_call_ids() -> None:
    dossier = _dossier(claims=[_fact_claim("ROCE improved to 16.8%", "unknown-tool")])
    failures = validate_grounding(dossier, {})
    assert len(failures) == 1
    assert "unknown-tool" in failures[0]


def test_grounding_normalizes_indian_and_international_grouping() -> None:
    dossier = _dossier(claims=[_fact_claim("Coverage reached 1,23,456.00")])
    failures = validate_grounding(dossier, {"tool-1": {"coverage": "123,456"}})
    assert failures == []


def test_grounding_accepts_periods_named_in_the_cited_field_names() -> None:
    dossier = _dossier(
        claims=[_fact_claim("The 12-1 momentum is 14.0% and 1-year volatility is 24.5%.")]
    )
    failures = validate_grounding(
        dossier, {"tool-1": {"momentum_12_1_pct": "14.0", "volatility_1y_pct": "24.5"}}
    )
    assert failures == []


def test_grounding_still_rejects_a_value_only_present_as_a_period() -> None:
    dossier = _dossier(claims=[_fact_claim("Momentum reached 24.5% over the year.")])
    failures = validate_grounding(dossier, {"tool-1": {"momentum_12_1_pct": "14.0"}})
    assert len(failures) == 1
    assert "24.5" in failures[0]
