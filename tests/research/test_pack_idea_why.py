from wealth.research.dossier import Dossier, RedTeamVerdict, Verdicts
from wealth.research.graph import pack_idea_why


def dossier(**overrides: object) -> Dossier:
    values: dict[str, object] = {
        "dossier_id": "2026-W44/IN/SYNTH",
        "security": {"id": "security-synthetic", "market": "IN", "type": "EQUITY"},
    }
    values.update(overrides)
    return Dossier.model_validate(values)


def test_pack_idea_why_keeps_two_reasons_and_truncates_at_word_boundaries() -> None:
    first = "first reason " * 20
    second = "A second useful reason."
    result = pack_idea_why(
        dossier(
            drivers=[first, second, "A third reason."],
            risks=["A risk from the review."],
        )
    )

    assert result["reasons"] == [first[: first.rfind(" ", 0, 160)], second]
    assert all(len(reason) <= 160 for reason in result["reasons"])
    assert result["risk"] == "A risk from the review."


def test_pack_idea_why_uses_red_team_falsifier_when_risks_are_empty() -> None:
    result = pack_idea_why(
        dossier(
            drivers=["A supported reason."],
            risks=[],
            verdicts=Verdicts(
                red_team=RedTeamVerdict(falsifier="The growth may not continue."),
            ),
        )
    )

    assert result == {
        "reasons": ["A supported reason."],
        "risk": "The growth may not continue.",
    }
