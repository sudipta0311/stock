from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any

import httpx
import pytest
from openai import APIStatusError
from pydantic import BaseModel

from wealth.research.dossier import DossierBudget
from wealth.research.router import (
    BudgetExceeded,
    ModelRouter,
    PromptPolicyError,
    RunBudget,
    RunContext,
    SchemaError,
    strict_json_schema,
)


class Reply(BaseModel):
    value: str


def _completion(
    content: str,
    *,
    prompt_tokens: int = 12,
    completion_tokens: int = 4,
) -> dict[str, Any]:
    return {
        "id": "synthetic-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-6-luna",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _context(*, max_usd: str = "1", max_calls: int = 20, mode: str = "weekly") -> RunContext:
    return RunContext(
        budget=RunBudget(max_usd=Decimal(max_usd), max_calls=max_calls),
        mode=mode,
    )


def _router(
    handler: Any,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
) -> ModelRouter:
    monkeypatch.setenv("FOUNDRY_ENDPOINT", "https://foundry.example.test/openai/v1/")
    monkeypatch.setenv("FOUNDRY_API_KEY", "synthetic-test-key")
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return ModelRouter(
        http_client=http_client, audit_writer=rows.append, sleep=lambda _seconds: None
    )


def _post_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_json_schema_success_uses_chat_completions_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_post_body(request))
        return httpx.Response(200, json=_completion('{"value":"grounded"}'))

    router = _router(handler, monkeypatch, [])
    result = router.complete(
        "nano",
        [{"role": "user", "content": "Review market data."}],
        schema=Reply,
        max_output_tokens=32,
        run_ctx=_context(),
    )

    assert result.parsed == Reply(value="grounded")
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[0]["response_format"]["json_schema"]["strict"] is True
    assert bodies[0]["model"] == "gpt-6-luna"
    assert bodies[0]["max_completion_tokens"] == 32
    assert bodies[0]["service_tier"] == "flex"


def test_rejected_json_schema_falls_back_to_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_post_body(request))
        if len(bodies) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "json_schema is unsupported for this deployment"}},
            )
        return httpx.Response(200, json=_completion('{"value":"fallback"}'))

    router = _router(handler, monkeypatch, [])
    result = router.complete(
        "redteam",
        [{"role": "user", "content": "Review market evidence."}],
        schema=Reply,
        max_output_tokens=32,
        run_ctx=_context(),
    )

    assert result.parsed == Reply(value="fallback")
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[1]["response_format"]["type"] == "json_object"
    assert "matching this schema" in bodies[1]["messages"][0]["content"]


class _Detail(BaseModel):
    default: str
    note: str | None = None


class _Nested(BaseModel):
    title: str
    detail: _Detail
    tags: list[str] = []


def test_strict_schema_closes_every_object_and_requires_all_fields() -> None:
    schema = strict_json_schema(_Nested.model_json_schema())

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["title", "detail", "tags"]
    assert "default" not in schema["properties"]["tags"]
    detail = schema["$defs"]["_Detail"]
    assert detail["additionalProperties"] is False
    # A field literally named "default" is kept; only the keyword is removed.
    assert detail["required"] == ["default", "note"]


def test_invalid_response_format_schema_falls_back_to_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_post_body(request))
        if len(bodies) == 1:
            message = "Invalid schema for response_format 'Reply': unsupported keyword."
            return httpx.Response(400, json={"error": {"message": message}})
        return httpx.Response(200, json=_completion('{"value":"fallback"}'))

    router = _router(handler, monkeypatch, [])
    result = router.complete(
        "nano",
        [{"role": "user", "content": "Review market evidence."}],
        schema=Reply,
        max_output_tokens=32,
        run_ctx=_context(),
    )

    assert result.parsed == Reply(value="fallback")
    assert bodies[1]["response_format"]["type"] == "json_object"


def test_invalid_json_gets_one_repair_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_post_body(request))
        content = "not json" if len(bodies) == 1 else '{"value":"repaired"}'
        return httpx.Response(200, json=_completion(content))

    router = _router(handler, monkeypatch, [])
    result = router.complete(
        "mini",
        [{"role": "user", "content": "Use only cited market data."}],
        schema=Reply,
        max_output_tokens=32,
        run_ctx=_context(),
    )

    assert result.parsed == Reply(value="repaired")
    assert len(bodies) == 2
    assert "Previous response: not json" in bodies[1]["messages"][-1]["content"]
    assert result.usage["input_tokens"] == 24
    assert result.usage["output_tokens"] == 8


def test_invalid_repair_raises_schema_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion('{"other":"invalid"}'))

    router = _router(handler, monkeypatch, [])
    with pytest.raises(SchemaError):
        router.complete(
            "mini",
            [{"role": "user", "content": "Review market data."}],
            schema=Reply,
            max_output_tokens=32,
            run_ctx=_context(),
        )


def test_flex_capacity_failures_fall_back_to_default_service_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(_post_body(request))
        if len(requests) <= 3:
            return httpx.Response(429, json={"error": {"message": "capacity unavailable"}})
        return httpx.Response(200, json=_completion('{"value":"ready"}'))

    router = _router(handler, monkeypatch, rows)
    result = router.complete(
        "nano",
        [{"role": "user", "content": "Review synthetic prices."}],
        schema=Reply,
        max_output_tokens=32,
        run_ctx=_context(),
    )

    assert result.parsed == Reply(value="ready")
    assert [request.get("service_tier") for request in requests] == ["flex", "flex", "flex", None]
    assert rows[-1]["service_tier"] == "default"
    assert [row["status"] for row in rows] == ["failed", "failed", "failed", "success"]


def test_unauthorized_response_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(401, json={"error": {"message": "unauthorized"}})

    router = _router(handler, monkeypatch, [])
    with pytest.raises(APIStatusError):
        router.complete(
            "nano",
            [{"role": "user", "content": "Review synthetic data."}],
            schema=Reply,
            max_output_tokens=32,
            run_ctx=_context(),
        )
    assert requests == 1


def test_budget_is_checked_before_a_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_completion('{"value":"unused"}'))

    router = _router(handler, monkeypatch, [])
    with pytest.raises(BudgetExceeded):
        router.complete(
            "mini",
            [{"role": "user", "content": "Synthetic market data."}],
            schema=Reply,
            max_output_tokens=1000,
            run_ctx=_context(max_usd="0.00000001"),
        )
    assert requests == 0


def test_dossier_budget_is_checked_before_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_completion('{"value":"unused"}'))

    router = _router(handler, monkeypatch, [])
    context = _context()
    context.dossier_budget = DossierBudget(max_usd=Decimal("0.00000001"), max_calls=2)
    with pytest.raises(BudgetExceeded):
        router.complete(
            "mini",
            [{"role": "user", "content": "Synthetic market data."}],
            schema=Reply,
            max_output_tokens=1000,
            run_ctx=context,
        )
    assert requests == 0


def test_luna_flex_cost_uses_configured_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion('{"value":"ok"}', prompt_tokens=1000, completion_tokens=500),
        )

    router = _router(handler, monkeypatch, [])
    result = router.complete(
        "nano",
        [{"role": "user", "content": "Review market data."}],
        schema=Reply,
        max_output_tokens=500,
        run_ctx=_context(),
    )

    assert result.cost_usd == Decimal("0.000175")


def test_prompt_guard_rejects_email_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router(lambda _request: httpx.Response(500), monkeypatch, [])
    with pytest.raises(PromptPolicyError):
        router.complete(
            "nano",
            [{"role": "user", "content": "Synthetic investor@example.test market note."}],
            schema=Reply,
            max_output_tokens=32,
            run_ctx=_context(),
        )


def test_prompt_guard_rejects_pan_patterns_built_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router(lambda _request: httpx.Response(500), monkeypatch, [])
    synthetic_id = "".join(["ABCDE", "1234", "F"])
    with pytest.raises(PromptPolicyError):
        router.complete(
            "nano",
            [{"role": "user", "content": f"Market note contains {synthetic_id}."}],
            schema=Reply,
            max_output_tokens=32,
            run_ctx=_context(),
        )


def test_audit_rows_never_contain_the_environment_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, Any]] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion('{"value":"ok"}'))

    router = _router(handler, monkeypatch, rows)
    router.complete(
        "nano",
        [{"role": "user", "content": "Review synthetic market data."}],
        schema=Reply,
        max_output_tokens=32,
        run_ctx=_context(),
    )

    serialized_rows = json.dumps(rows, default=str)
    assert rows
    assert os.environ["FOUNDRY_API_KEY"] not in serialized_rows
    assert "headers" not in serialized_rows
    assert "api-key" not in serialized_rows
