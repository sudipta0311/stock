from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from wealth.db.models import MarketdataRun
from wealth.marketdata import refresh
from wealth.marketdata.sources.macro_geo import FeedBatch, FeedItemObservation


class _RecordingSession:
    def __init__(self) -> None:
        self.runs: list[MarketdataRun] = []

    def add(self, value: MarketdataRun) -> None:
        self.runs.append(value)

    def flush(self) -> None:
        for run in self.runs:
            if run.id is None:
                run.id = uuid4()


def _prepare_refresh(monkeypatch: pytest.MonkeyPatch) -> _RecordingSession:
    session = _RecordingSession()
    monkeypatch.setattr(refresh, "system_session", lambda _engine: nullcontext(session))
    for name in (
        "prune_fundamentals",
        "prune_feed_items",
        "prune_index_valuations",
        "prune_macro_series",
    ):
        monkeypatch.setattr(refresh.store, name, lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(refresh.store, "upsert_feed_items", lambda _session, items: len(items))
    original_store_source = refresh._store_source

    def store_source(session: Any, source: str, records: Any, **kwargs: Any) -> Any:
        if source == "nse":
            return {"prices_eod": 1}
        return original_store_source(session, source, records, **kwargs)

    monkeypatch.setattr(refresh, "_store_source", store_source)
    return session


def _refresh_result(
    monkeypatch: pytest.MonkeyPatch,
    batch: FeedBatch,
) -> tuple[dict[str, Any], _RecordingSession]:
    session = _prepare_refresh(monkeypatch)
    result = refresh._run_refresh_on_engine(
        object(),
        date(2026, 10, 30),
        ("nse", "feeds"),
        fetchers={"nse": lambda: [], "feeds": lambda: batch},
        fetched_at=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )
    return result, session


def _feed_item() -> FeedItemObservation:
    return FeedItemObservation(
        source="rbi",
        guid="synthetic-rbi-item",
        title="Synthetic policy update",
        url="https://example.test/feed-item",
        published_at=None,
        summary="Synthetic summary",
        fetched_at=datetime(2026, 10, 30, 12, tzinfo=UTC),
    )


def _main_result(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, dict[str, Any]]:
    engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.delenv("OWNER_DATABASE_URL", raising=False)
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite://")
    monkeypatch.setattr(refresh, "make_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(refresh, "run_refresh", lambda *_args, **_kwargs: result)

    code = refresh.main(["--date", "2026-10-30", "--only", "nse"])
    return code, json.loads(capsys.readouterr().out)


def test_one_feed_failure_is_stored_as_a_warning_and_refresh_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, session = _refresh_result(
        monkeypatch,
        FeedBatch(items=[_feed_item()], errors={"pib": "NotAFeed"}),
    )

    assert result["status"] == "succeeded"
    assert result["errors"] == {}
    assert result["warnings"] == {"feeds:pib": "NotAFeed"}
    assert session.runs[0].counts["warnings"] == result["warnings"]

    exit_code, output = _main_result(monkeypatch, result, capsys)

    assert exit_code == 0
    assert output["warnings"] == {"feeds:pib": "NotAFeed"}


def test_all_feed_failures_remain_errors_and_refresh_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, _session = _refresh_result(
        monkeypatch,
        FeedBatch(
            items=[],
            errors={
                "federal_register:tariffs": "HTTPStatusError",
                "federal_register:sanctions": "HTTPStatusError",
                "pib": "NotAFeed",
                "rbi": "HTTPStatusError",
                "ustr": "ParseError",
            },
        ),
    )

    assert result["status"] == "partial"
    assert result["warnings"] == {}
    assert set(result["errors"]) == {
        "feeds:federal_register:tariffs",
        "feeds:federal_register:sanctions",
        "feeds:pib",
        "feeds:rbi",
        "feeds:ustr",
    }

    exit_code, output = _main_result(monkeypatch, result, capsys)

    assert exit_code == 2
    assert output["errors"] == result["errors"]
