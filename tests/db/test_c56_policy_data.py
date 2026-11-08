from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import MarketdataRun, Security, StockCheck
from wealth.policy.data import _stock_check_inputs

pytestmark = pytest.mark.db


def _security_with_check(session: Session, market: str, week: str, as_of: date) -> Security:
    security = Security(
        market=market,
        symbol=f"SYNTH-{market}-{uuid4().hex[:8]}",
        name="Synthetic Share",
        security_type="equity",
        currency="INR" if market == "IN" else "USD",
        sleeve="equity_india" if market == "IN" else "equity_global",
        sector="Synthetic sector",
        industry="Synthetic industry",
    )
    session.add(security)
    session.flush()
    session.add(
        StockCheck(
            security_id=security.id,
            week=week,
            as_of=as_of,
            sources={},
            revenue_yoy_pct=[],
        )
    )
    return security


def test_plan_marks_only_available_market_events_as_watched(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 11, 7)
    with Session(migrated_database.owner_engine) as session, session.begin():
        india = _security_with_check(session, "IN", "2026-W45", as_of)
        us = _security_with_check(session, "US", "2026-W45", as_of)
        india_id = str(india.id)
        us_id = str(us.id)
        session.add(
            MarketdataRun(
                started_at=datetime(2026, 11, 6, 8, tzinfo=UTC),
                finished_at=datetime(2026, 11, 6, 8, tzinfo=UTC),
                status="partial",
                counts={"sentinel": {"IN": "unavailable", "US": "available", "targets": 2}},
                errors={},
            )
        )
        checks, _events, _week = _stock_check_inputs(
            session,
            (
                SimpleNamespace(security_id=india_id, security_type="equity"),
                SimpleNamespace(security_id=us_id, security_type="equity"),
            ),
            as_of,
        )

    assert checks[india_id].events_watched is False
    assert checks[us_id].events_watched is True


def test_plan_ignores_sentinel_runs_older_than_three_days(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 10, 15)
    with Session(migrated_database.owner_engine) as session, session.begin():
        india = _security_with_check(session, "IN", "2026-W42", as_of)
        india_id = str(india.id)
        session.add(
            MarketdataRun(
                started_at=datetime(2026, 10, 11, 8, tzinfo=UTC),
                finished_at=datetime(2026, 10, 11, 8, tzinfo=UTC),
                status="succeeded",
                counts={"sentinel": {"IN": "available", "US": "available", "targets": 1}},
                errors={},
            )
        )
        checks, _events, _week = _stock_check_inputs(
            session,
            (SimpleNamespace(security_id=india_id, security_type="equity"),),
            as_of,
        )

    assert checks[india_id].events_watched is False
