from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from wealth.research import sentinel
from wealth.research.router import RunBudget, RunContext
from wealth.research.sentinel import (
    EventCode,
    SentinelClassification,
    SentinelTarget,
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


def _recorded_result(result: object) -> ToolResult:
    return ToolResult(
        tool_call_id=f"synthetic-{uuid4()}",
        result=result,
        source_url="https://example.test/synthetic-evidence",
        retrieved_at=datetime(2026, 11, 8, tzinfo=UTC),
    )


class BudgetedFakeRouter:
    def __init__(self) -> None:
        self.attempts = 0

    def complete(self, tier, messages, *, schema, max_output_tokens, run_ctx):
        self.attempts += 1
        run_ctx.budget.reserve(Decimal("0"))
        return SimpleNamespace(
            parsed=SentinelClassification(
                material=False,
                event_code=EventCode.NONE,
                confidence="LOW",
                quote="",
            )
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


def test_india_keeps_rule_checks_after_nano_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
        SentinelTarget(id=uuid4(), market="IN", symbol=f"SYNTH-IN-{index}") for index in range(1, 4)
    ]
    announcements = {
        target.symbol: [
            {
                "date": "2026-11-08",
                "category": "Routine update",
                "text": "Synthetic routine business update.",
            }
        ]
        for target in targets
    }
    announcements[targets[2].symbol][0]["category"] = "Auditor resignation"

    class FakeRegistry:
        def call(self, tool: str, args: dict[str, object], _run_ctx: RunContext) -> ToolResult:
            assert tool == "nse_announcements"
            symbol = str(args["symbol"])
            return _recorded_result({"status": "available", "announcements": announcements[symbol]})

    stored: list[dict[str, object]] = []
    monkeypatch.setattr(
        sentinel,
        "_store_event",
        lambda _engine, **event: stored.append(event) or uuid4(),
    )
    router = BudgetedFakeRouter()
    run_ctx = RunContext(budget=RunBudget(max_usd=Decimal("1"), max_calls=1))

    result = sentinel._process_india(
        object(),
        FakeRegistry(),  # type: ignore[arg-type]
        targets,
        days=7,
        as_of=datetime(2026, 11, 8, tzinfo=UTC).date(),
        now=datetime(2026, 11, 8, tzinfo=UTC),
        router=router,
        run_ctx=run_ctx,
        category_rules=[("auditor resignation", "AUDITOR_RESIGNATION")],
        governance_events=frozenset({"AUDITOR_RESIGNATION"}),
    )

    assert result["status"] == "available"
    assert result["nano_budget_exhausted"] == 1
    assert result["nano_calls"] == 1
    assert router.attempts == 2
    assert len(stored) == 1
    assert stored[0]["security_id"] == targets[2].id
    assert stored[0]["event_code"] == "AUDITOR_RESIGNATION"
    assert stored[0]["method"] == "rule"


def test_edgar_keeps_rule_checks_after_nano_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
        SentinelTarget(id=uuid4(), market="US", symbol=f"SYNTH-US-{index}") for index in range(1, 4)
    ]
    cik_by_symbol = {
        target.symbol: str(index + 1000).zfill(10) for index, target in enumerate(targets)
    }
    symbol_by_cik = {cik: symbol for symbol, cik in cik_by_symbol.items()}
    text_by_accession = {
        f"synthetic-accession-{target.symbol}": (
            "<p>ITEM 4.01 Auditor resignation</p>"
            if target == targets[2]
            else "<p>Synthetic routine business update.</p>"
        )
        for target in targets
    }

    class FakeRegistry:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def call(self, tool: str, args: dict[str, object], _run_ctx: RunContext) -> ToolResult:
            self.calls.append((tool, args))
            if tool == "edgar_company_tickers":
                return _recorded_result(
                    [
                        {
                            "ticker": target.symbol,
                            "cik": cik_by_symbol[target.symbol],
                            "name": f"Synthetic {target.symbol}",
                        }
                        for target in targets
                    ]
                )
            if tool == "edgar_filings":
                symbol = symbol_by_cik[str(args["cik"])]
                accession = f"synthetic-accession-{symbol}"
                return _recorded_result(
                    [
                        {
                            "form": "8-K",
                            "document_url": f"https://example.test/{symbol}/filing",
                            "accession": accession,
                        }
                    ]
                )
            if tool == "edgar_filing_text":
                accession = str(args["accession"])
                return _recorded_result({"text": text_by_accession[accession]})
            raise AssertionError(f"Unexpected fake tool: {tool}")

    stored: list[dict[str, object]] = []
    monkeypatch.setattr(
        sentinel,
        "_store_event",
        lambda _engine, **event: stored.append(event) or uuid4(),
    )
    router = BudgetedFakeRouter()
    run_ctx = RunContext(budget=RunBudget(max_usd=Decimal("1"), max_calls=1))
    registry = FakeRegistry()

    result = sentinel._process_edgar(
        object(),
        registry,  # type: ignore[arg-type]
        targets,
        days=7,
        as_of=datetime(2026, 11, 8, tzinfo=UTC).date(),
        now=datetime(2026, 11, 8, tzinfo=UTC),
        router=router,
        run_ctx=run_ctx,
        governance_events=frozenset({"AUDITOR_RESIGNATION"}),
    )

    assert result["status"] == "available"
    assert result["filings"] == 3, registry.calls
    assert result["nano_budget_exhausted"] == 1
    assert result["nano_calls"] == 1
    assert router.attempts == 2
    assert len(stored) == 1
    assert stored[0]["security_id"] == targets[2].id
    assert stored[0]["event_code"] == "AUDITOR_RESIGNATION"
    assert stored[0]["method"] == "rule"
