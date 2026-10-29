from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import NavDaily, PriceEod, Security
from wealth.research.router import RunBudget, RunContext
from wealth.research.store import create_run
from wealth.research.tools import ToolRegistry, evidence_message

pytestmark = pytest.mark.db


def test_market_data_tools_return_and_log_source_backed_evidence(
    migrated_database: MigratedDatabase,
) -> None:
    security = Security(
        market="IN",
        symbol="C14SYN",
        isin="INF000000999",
        name="Synthetic research security",
        security_type="mf",
        currency="INR",
        sleeve="debt",
        category="mf:debt:liquid",
    )
    as_of = date.today() - timedelta(days=1)
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(security)
        session.flush()
        session.add(
            PriceEod(
                security_id=security.id,
                as_of=as_of,
                close=Decimal("101.250000"),
                source="SYNTHETIC",
                source_url="https://example.test/price",
            )
        )
        session.add(
            NavDaily(
                security_id=security.id,
                as_of=as_of,
                nav=Decimal("10.25000000"),
                source="SYNTHETIC",
            )
        )
        security_id = security.id

    run_id = uuid4()
    create_run(
        migrated_database.owner_engine,
        run_id=run_id,
        week="2026-W45",
        mode="weekly",
        budget_usd=Decimal("1"),
    )
    registry = ToolRegistry(migrated_database.owner_engine)
    context = RunContext(
        budget=RunBudget(max_usd=Decimal("1"), max_calls=10),
        run_id=run_id,
        owner_engine=migrated_database.owner_engine,
    )
    price_result = registry.call(
        "price_history", {"security_id": str(security_id), "days": 30}, context
    )
    nav_result = registry.call("latest_nav", {"isin": "INF000000999"}, context)
    with migrated_database.owner_engine.connect() as connection:
        audit_rows = connection.execute(
            text(
                "SELECT tool_call_id, result_sha256, source_url, status "
                "FROM tool_calls WHERE run_id = :run_id ORDER BY retrieved_at"
            ),
            {"run_id": run_id},
        ).all()

    assert price_result.result[0]["close"] == "101.250000"
    assert price_result.source_url == "https://example.test/price"
    assert nav_result.result["nav"] == "10.25000000"
    assert nav_result.source_url == "https://www.amfiindia.com/"
    assert len(audit_rows) == 2
    assert all(row.result_sha256 and row.status == "success" for row in audit_rows)
    assert "<evidence id=" in evidence_message(price_result)
