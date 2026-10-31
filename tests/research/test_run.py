from __future__ import annotations

import json
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from langgraph.checkpoint.memory import MemorySaver

from wealth.research import run
from wealth.research.errors import BudgetExceeded


def test_cli_matches_scheduled_workflow_week_selectors() -> None:
    parser = run.build_parser()
    start = parser.parse_args(["start", "--week", "2026-W45", "--mode", "weekly"])
    continue_by_week = parser.parse_args(["continue", "--week", "2026-W45"])
    continue_by_run_id = parser.parse_args(
        ["continue", "--run-id", "00000000-0000-0000-0000-000000000045"]
    )
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "research-weekly.yml"
    ).read_text(encoding="utf-8")

    assert start.command == "start" and start.week == "2026-W45" and start.mode == "weekly"
    assert continue_by_week.command == "continue" and continue_by_week.week == "2026-W45"
    assert continue_by_run_id.command == "continue" and continue_by_run_id.run_id is not None
    assert 'start --week "$WEEK" --mode weekly' in workflow
    assert 'continue --week "$WEEK"' in workflow
    assert "vars.RESEARCH_ENABLED == 'true'" in workflow
    assert "github.event_name == 'workflow_dispatch' && inputs.step != 'continue'" in workflow
    assert "github.event_name == 'workflow_dispatch' && inputs.step == 'continue'" in workflow


def test_cli_returns_exit_code_two_when_the_run_budget_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeSaver(MemorySaver):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def setup(self) -> None:
            return None

    class FakePostgresSaver:
        @staticmethod
        def from_conn_string(_url: str) -> FakeSaver:
            return FakeSaver()

    class FakeEngine:
        def dispose(self) -> None:
            return None

    run_id = UUID("00000000-0000-0000-0000-000000000045")
    monkeypatch.setattr(run, "_owner_url", lambda: "postgresql+psycopg://localhost/wealth")
    monkeypatch.setattr(run, "make_engine", lambda _url: FakeEngine())
    monkeypatch.setattr(run, "create_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "get_open_run_for_week", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run,
        "get_run",
        lambda *_args, **_kwargs: {
            "run_id": run_id,
            "week": "2026-W45",
            "mode": "weekly",
            "budget_usd": Decimal("0.01"),
            "spent_usd": Decimal("0"),
            "calls": 0,
        },
    )
    monkeypatch.setattr(run, "update_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run, "PostgresSaver", FakePostgresSaver)
    monkeypatch.setattr(run, "run_screener", lambda *_args, **_kwargs: {"security_ids": []})
    monkeypatch.setattr(run, "load_shortlist_candidates", lambda *_args, **_kwargs: [])

    def exhaust_budget(**_kwargs):
        raise BudgetExceeded("synthetic budget exhaustion")

    monkeypatch.setattr(run, "_graph_run", exhaust_budget)

    exit_code = run._run_start(Namespace(week="2026-W45", mode="weekly", budget_usd="0.01"))
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 2, result
    assert result["status"] == "budget_exceeded"
