from __future__ import annotations

import os
import subprocess
import sys
from decimal import Decimal
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from tests.db.conftest import MigratedDatabase

from wealth.research.graph import build_research_graph
from wealth.research.router import FakeRouter, RunBudget, RunContext
from wealth.research.run import _checkpointer_url

pytestmark = pytest.mark.db


def test_checkpoint_resumes_after_screen_in_a_fresh_process(
    migrated_database: MigratedDatabase,
) -> None:
    database_url = migrated_database.owner_engine.url.render_as_string(hide_password=False)
    checkpoint_url = _checkpointer_url(database_url)
    run_id = f"c14-{uuid4()}"
    config = {"configurable": {"thread_id": run_id}}
    context = RunContext(budget=RunBudget(max_usd=Decimal("1"), max_calls=100))

    with PostgresSaver.from_conn_string(checkpoint_url) as checkpointer:
        checkpointer.setup()
        graph = build_research_graph(FakeRouter(), run_ctx=context, checkpointer=checkpointer)
        state = graph.invoke(
            {
                "run_id": run_id,
                "dossier_ids": [],
                "stage": "QUEUED",
                "status": "running",
                "handback_count": 0,
            },
            config,
        )
    assert state["stage"] == "SCREENED"
    assert state["status"] == "waiting"

    environment = os.environ.copy()
    environment["C14_CHECKPOINT_DATABASE_URL"] = checkpoint_url
    environment["C14_CHECKPOINT_THREAD_ID"] = run_id
    code = (
        "from decimal import Decimal\n"
        "import os\n"
        "from langgraph.checkpoint.postgres import PostgresSaver\n"
        "from wealth.research.graph import build_research_graph\n"
        "from wealth.research.router import FakeRouter, RunBudget, RunContext\n"
        "url = os.environ['C14_CHECKPOINT_DATABASE_URL']\n"
        "thread_id = os.environ['C14_CHECKPOINT_THREAD_ID']\n"
        "context = RunContext(budget=RunBudget(Decimal('1'), 100))\n"
        "with PostgresSaver.from_conn_string(url) as saver:\n"
        "    saver.setup()\n"
        "    graph = build_research_graph(FakeRouter(), run_ctx=context, checkpointer=saver)\n"
        "    state = graph.invoke(None, {'configurable': {'thread_id': thread_id}})\n"
        "assert state['stage'] == 'APPROVED'\n"
        "assert state['status'] == 'completed'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, "fresh process could not resume the Postgres checkpoint"
