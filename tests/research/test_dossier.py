from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from wealth.research.dossier import (
    Claim,
    Dossier,
    Evidence,
    Stage,
    dossier_id_for,
    hand_back_to_analysis,
    symbol_from_dossier_id,
    transition_stage,
    validate_grounding,
)

_CK_DOSSIERS_ID_PATTERN = re.compile(r"^[0-9]{4}-W[0-9]{2}/(IN|US|DK|EU|GB)/[A-Za-z0-9._-]+$")


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


def test_dossier_symbol_encoding_round_trips_and_matches_database_check() -> None:
    symbols = ("M&M", "J&KBANK", "ARE&M", "BRK.B", "BF-B", "RELIANCE", "A_B", "SYN☃")

    for symbol in symbols:
        dossier_id = dossier_id_for("2026-W46", "IN", symbol)

        assert _CK_DOSSIERS_ID_PATTERN.fullmatch(dossier_id)
        assert symbol_from_dossier_id(dossier_id) == symbol

    assert dossier_id_for("2026-W46", "IN", "M&M") == "2026-W46/IN/M_26M"
    assert dossier_id_for("2026-W46", "IN", "A_B") == "2026-W46/IN/A_5FB"
    assert dossier_id_for("2026-W46", "IN", "SYN☃") == "2026-W46/IN/SYN_E2_98_83"
    assert dossier_id_for("2026-W46", "IN", "RELIANCE") == "2026-W46/IN/RELIANCE"
    assert dossier_id_for("2026-W46", "US", "BRK.B") == "2026-W46/US/BRK.B"
    assert dossier_id_for("2026-W46", "US", "BF-B") == "2026-W46/US/BF-B"


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


def test_grounding_accepts_a_cited_value_rounded_to_the_claims_precision() -> None:
    evidence = {
        "tool-1": {
            "fcf_yield_pct": "18.88104007153400805429701392",
            "momentum": "-0.3871180994932996131594742704",
            "earnings_yield_vs_sector": "2.908146417744015776562178157",
        }
    }
    accepted = _dossier(
        claims=[_fact_claim("Free-cash-flow yield is 18.9%, momentum -0.3871, about 2.9 times.")]
    )
    overstated = _dossier(claims=[_fact_claim("Free-cash-flow yield is 19.0%.")])

    assert validate_grounding(accepted, evidence) == []
    failures = validate_grounding(overstated, evidence)
    assert len(failures) == 1
    assert "unsupported numbers 19" in failures[0]


def test_grounding_accepts_rescaled_values_and_percentages_of_fractions() -> None:
    evidence = {
        "tool-1": {
            "revenue": "1572500000",
            "net_income": "221200000",
            "fcf_yield": "0.07000074464851961859764166882",
            "segment_revenue_inr": "12340000000",
            "dist_from_200dma": "-0.0312",
            "period": "2026",
        }
    }
    accepted = _dossier(
        claims=[
            _fact_claim(
                "Q3 2026 revenue was $1.573B and net income $221.2M; FCF yield 7.0%; "
                "segment revenue 1,234 crore; price 3.1% below its 200-day average."
            )
        ]
    )
    overstated = _dossier(claims=[_fact_claim("Revenue was $1.7B with a 7.5% FCF yield.")])

    assert validate_grounding(accepted, evidence) == []
    failures = validate_grounding(overstated, evidence)
    assert len(failures) == 1
    assert "unsupported numbers 1.7, 7.5" in failures[0]


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
