from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from wealth.policy.highlights import HighlightContext, highlight_fingerprint, highlights

_COPY = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "copy" / "en.yaml").read_text(encoding="utf-8")
)


def test_nri_household_highlights_are_trimmed_to_each_screen_limit() -> None:
    plan = {
        "residency": {"status": "NRI", "country": "DK"},
        "notes": [
            {"key": f"plan_note_{index}", "amount": str(index * 1_000)} for index in range(20)
        ],
        "residency_notes": ["ask_start_value_missing", "dk_fund_tax_review"],
        "placement_notes": ["tax_residence_country_may_apply"],
        "foundation": [
            {"key": "high_interest_debt", "gap": "25000.00"},
            {"key": "emergency_fund", "gap": "15000.00"},
        ],
        "warnings": ["tcs_applies", "schedule_fa_reporting"],
    }

    limits = {"home": 1, "plan": 3, "ideas": 2, "money": 3}
    for screen, limit in limits.items():
        result = highlights(plan, HighlightContext(screen=screen, copy=_COPY))
        assert len(result) <= limit
        assert all(item.screen == screen for item in result)


def test_highlights_rank_action_and_amount_before_screen_relevance() -> None:
    copy = deepcopy(_COPY)
    copy["highlights"]["catalog"].update(
        {
            "large_amount": {
                "title": "Large amount",
                "body": "This amount is large.",
                "screens": ["ideas"],
            },
            "relevant_smaller": {
                "title": "Relevant detail",
                "body": "This applies to the plan.",
                "screens": ["plan"],
            },
        }
    )
    plan = {
        "notes": [
            {"key": "large_amount", "amount": "90000.00"},
            {"key": "relevant_smaller", "amount": "50000.00"},
            {"key": "high_interest_debt_gap", "amount": "10000.00"},
        ]
    }

    result = highlights(plan, HighlightContext(screen="plan", copy=copy))

    assert [item.key for item in result] == [
        "high_interest_debt_gap",
        "large_amount",
        "relevant_smaller",
    ]


def test_highlights_use_screen_relevance_then_first_seen_for_ties() -> None:
    copy = deepcopy(_COPY)
    copy["highlights"]["catalog"].update(
        {
            key: {
                "title": key.replace("_", " ").title(),
                "body": "This detail may help.",
                "screens": screens,
            }
            for key, screens in (
                ("unrelated_first", ["ideas"]),
                ("relevant_first", ["plan"]),
                ("relevant_second", ["plan"]),
                ("unrelated_second", ["money"]),
            )
        }
    )
    plan = {
        "notes": [
            {"key": key, "amount": "50000.00"}
            for key in (
                "unrelated_first",
                "relevant_first",
                "relevant_second",
                "unrelated_second",
            )
        ]
    }

    result = highlights(plan, HighlightContext(screen="plan", copy=copy))

    assert [item.key for item in result] == [
        "relevant_first",
        "relevant_second",
        "unrelated_first",
    ]


def test_dismissed_highlight_returns_after_its_display_value_changes() -> None:
    original = {"notes": [{"key": "target_gap", "amount": "10000.00"}]}
    same_display_value = {"notes": [{"key": "target_gap", "amount": "10040.00"}]}
    changed_display_value = {"notes": [{"key": "target_gap", "amount": "10060.00"}]}
    fingerprint = highlight_fingerprint(original, "target_gap", _COPY)
    assert fingerprint is not None
    context = HighlightContext(
        screen="home", copy=_COPY, dismissed_fingerprints={"target_gap": fingerprint}
    )

    assert highlight_fingerprint(same_display_value, "target_gap", _COPY) == fingerprint
    assert highlights(same_display_value, context) == []
    assert highlight_fingerprint(changed_display_value, "target_gap", _COPY) != fingerprint
    assert highlights(changed_display_value, context)


def test_small_household_keeps_a_matching_highlight() -> None:
    result = highlights(
        {"notes": ["ask_start_value_missing"]},
        HighlightContext(screen="ideas", copy=_COPY),
    )

    assert len(result) == 1
    assert result[0].key == "ask_start_value_missing"
    assert result[0].title == "Add your account value"


def test_highlight_amount_fingerprint_uses_display_quantum() -> None:
    fingerprint = highlight_fingerprint(
        {"notes": [{"key": "target_gap", "amount": Decimal("10040.00")}]},
        "target_gap",
        _COPY,
    )
    rounded = highlight_fingerprint(
        {"notes": [{"key": "target_gap", "amount": Decimal("10000.00")}]},
        "target_gap",
        _COPY,
    )

    assert fingerprint == rounded


def test_notes_without_plain_copy_are_not_shown_as_highlights() -> None:
    plan: dict[str, Any] = {
        "notes": [
            {"key": "unlisted_note", "amount": "0"},
            "debt_yields_unverified_as_of:2026-10-28",
        ]
    }
    result = highlights(plan, HighlightContext(screen="plan", copy=_COPY))

    assert result == []
    assert highlight_fingerprint(plan, "unlisted_note", _COPY) is None
