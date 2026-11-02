from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from conftest import MigratedDatabase
from wealth.alerts.health import load_health_facts
from wealth.db.models import (
    HouseholdMember,
    MarketdataRun,
    PriceEod,
    RecordedBuy,
    Security,
    User,
)
from wealth.marketdata import store
from wealth.marketdata.refresh import run_refresh
from wealth.marketdata.sources.alpha_vantage import SOURCE_URL
from wealth.policy.denmark_picks import load_eu_etfs

pytestmark = pytest.mark.db


def _security(
    symbol: str,
    *,
    market: str = "EU",
    isin: str | None = None,
) -> Security:
    return Security(
        market=market,
        symbol=symbol,
        isin=isin,
        name=f"Synthetic {symbol}",
        security_type="etf",
        currency="EUR" if market == "EU" else "USD",
        sleeve="equity_global",
        category="equity:global",
    )


def test_prune_keeps_curated_and_waiting_buy_history_for_health_loader(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 11, 2)
    fund_isin = load_eu_etfs()["instruments"][0]["isin"]
    curated = _security(f"C46-{uuid4().hex[:10]}", isin=fund_isin)
    bought = _security(f"C46-{uuid4().hex[:10]}")
    unrelated = _security(f"C46-{uuid4().hex[:10]}")
    owner_id = uuid4()
    owner = User(
        id=owner_id,
        email=f"c46-{uuid4()}@example.test",
        auth_subject=str(uuid4()),
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(owner)
        session.flush()
        member = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == owner_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert member is not None
        session.add_all([curated, bought, unrelated])
        session.flush()
        curated_id, bought_id, unrelated_id = curated.id, bought.id, unrelated.id
        session.add(
            RecordedBuy(
                user_id=owner_id,
                member_id=member.id,
                bought_on=on - timedelta(days=1),
                amount=Decimal("2000.00"),
                currency="DKK",
                security_id=bought.id,
                security_type="etf",
                name="Synthetic waiting purchase",
                sleeve="equity_global",
                instrument_category="abroad_etf",
                status="waiting",
            )
        )
        for security in (curated, bought):
            for offset in (2, 30):
                store.upsert_eod_price(
                    session,
                    security.id,
                    on - timedelta(days=offset),
                    Decimal("10.00"),
                    source="AlphaVantage",
                    source_url=SOURCE_URL,
                )
        store.upsert_eod_price(
            session,
            unrelated.id,
            on - timedelta(days=2),
            Decimal("10.00"),
            source="AlphaVantage",
            source_url=SOURCE_URL,
        )

        counts = store.prune(session, on)
        surviving_ids = set(
            session.scalars(
                select(PriceEod.security_id).where(
                    PriceEod.security_id.in_([curated_id, bought_id, unrelated_id])
                )
            )
        )
        curated_rows = session.scalar(
            select(func.count()).select_from(PriceEod).where(PriceEod.security_id == curated_id)
        )
        bought_rows = session.scalar(
            select(func.count()).select_from(PriceEod).where(PriceEod.security_id == bought_id)
        )

    facts = load_health_facts(
        migrated_database.owner_engine,
        datetime(2026, 11, 2, 12, tzinfo=UTC),
        email_sending_configured=False,
    )
    assert counts["prices_eod"] >= 1
    assert surviving_ids == {curated_id, bought_id}
    assert curated_rows == 2
    assert bought_rows == 2
    assert any(day.on == on - timedelta(days=2) and day.rows == 2 for day in facts.abroad_prices)


def test_run_row_is_saved_after_a_fetch_invalidates_its_connection(
    migrated_database: MigratedDatabase,
) -> None:
    def fail_during_fetch() -> list[object]:
        raise DBAPIError(
            "synthetic fetch statement",
            {},
            RuntimeError("synthetic connection invalidation"),
            connection_invalidated=True,
        )

    result = run_refresh(
        migrated_database.owner_engine,
        date(2026, 11, 2),
        only=("amfi_history",),
        fetchers={"amfi_history": fail_during_fetch},
    )

    with Session(migrated_database.owner_engine) as session:
        run = session.get(MarketdataRun, UUID(result["run_id"]))

    assert result["status"] == "failed"
    assert result["errors"] == {"amfi_history": "DBAPIError"}
    assert run is not None
    assert run.status == "failed"
    assert run.errors == {"amfi_history": "DBAPIError"}
