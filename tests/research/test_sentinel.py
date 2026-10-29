from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from wealth.research.sentinel import (
    EventCode,
    SentinelClassification,
    _nano_event,
    rule_events_for_filing,
)
from wealth.research.tools import ToolResult


class FakeRouter:
    def __init__(self, classification: dict[str, object]) -> None:
        self.classification = SentinelClassification.model_validate(classification)
        self.calls = 0
        self.messages: list[object] = []

    def complete(self, tier, messages, *, schema, max_output_tokens, run_ctx):
        self.calls += 1
        self.messages = list(messages)
        assert tier == "nano"
        assert schema is SentinelClassification
        assert max_output_tokens <= 300
        return SimpleNamespace(parsed=self.classification)


def _tool_result(text: str) -> ToolResult:
    return ToolResult(
        tool_call_id="synthetic-tool-call",
        result={"text": text},
        source_url="https://www.sec.gov/Archives/edgar/data/example.htm",
        retrieved_at=datetime(2026, 10, 29, tzinfo=UTC),
    )


def test_known_8k_items_use_rule_mapping_without_a_model() -> None:
    class NeverCallRouter:
        calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("known filing items do not use a model")

    router = NeverCallRouter()
    items = (
        "<p>ITEM 4.01 Departure of independent auditor</p>"
        "<p>ITEM 4.02 Non-reliance on previously issued financial statements</p>"
        "<p>ITEM 5.02 Departure of a director</p>"
    )

    events = rule_events_for_filing("8-K", items)

    assert events == ["AUDITOR_RESIGNATION", "RESTATEMENT", "KEY_PERSON_EXIT"]
    assert router.calls == 0
    assert rule_events_for_filing("NT 10-Q") == ["LATE_FILING"]


def test_free_text_uses_nano_and_accepts_only_an_exact_quote() -> None:
    content = "The issuer announced a material departure of its finance chief."
    router = FakeRouter(
        {
            "material": True,
            "event_code": EventCode.KEY_PERSON_EXIT,
            "confidence": "HIGH",
            "quote": "material departure of its finance chief",
        }
    )

    event_code = _nano_event(
        router,
        _tool_result(content),
        {"text": content},
        content,
        SimpleNamespace(),
    )

    assert event_code == "KEY_PERSON_EXIT"
    assert router.calls == 1
    assert '<evidence id="synthetic-tool-call">' in router.messages[1]["content"]


def test_nano_invented_quote_is_rejected() -> None:
    content = "The issuer published a routine quarterly update."
    router = FakeRouter(
        {
            "material": True,
            "event_code": "RESTATEMENT",
            "confidence": "HIGH",
            "quote": "The issuer announced a major restatement.",
        }
    )

    event_code = _nano_event(
        router,
        _tool_result(content),
        {"text": content},
        content,
        SimpleNamespace(),
    )

    assert event_code is None
    assert router.calls == 1
