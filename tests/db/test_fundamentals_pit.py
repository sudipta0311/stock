from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import FundamentalPIT
from wealth.marketdata import store
from wealth.marketdata.sources.fundamentals import FundamentalObservation
from wealth.research.factors import fundamental_factors

pytestmark = pytest.mark.db


def _observation(
    metric: str,
    period_end: date,
    value: str,
    filed_on: date,
    *,
    unit: str = "USD",
    period_type: str = "Q",
) -> FundamentalObservation:
    filed_at = datetime.combine(filed_on, datetime.min.time(), tzinfo=UTC)
    return FundamentalObservation(
        metric=metric,
        period_end=period_end,
        period_type=period_type,
        value=Decimal(value),
        unit=unit,
        filed_at=filed_at,
        known_at=filed_at,
        source="SEC_EDGAR",
        source_url="https://data.sec.gov/example/companyfacts.json",
    )


def test_new_shared_tables_grant_select_only_and_disable_rls(
    migrated_database: MigratedDatabase,
) -> None:
    tables = ("fundamentals_pit", "universe_members")
    writes = ("INSERT", "UPDATE", "DELETE")
    with migrated_database.owner_engine.connect() as connection:
        for table in tables:
            qualified_name = f"{migrated_database.schema_name}.{table}"
            assert (
                connection.scalar(
                    text("SELECT has_table_privilege('wealth_app', :table_name, 'SELECT')"),
                    {"table_name": qualified_name},
                )
                is True
            )
            for privilege in writes:
                assert (
                    connection.scalar(
                        text("SELECT has_table_privilege('wealth_app', :table_name, :privilege)"),
                        {"table_name": qualified_name, "privilege": privilege},
                    )
                    is False
                )
            for privilege in ("SELECT", *writes):
                assert connection.scalar(
                    text("SELECT has_table_privilege(:role, :table_name, :privilege)"),
                    {
                        "role": "wealth_api",
                        "table_name": qualified_name,
                        "privilege": privilege,
                    },
                ) == connection.scalar(
                    text("SELECT has_table_privilege('wealth_api', :table_name, :privilege)"),
                    {
                        "table_name": f"{migrated_database.schema_name}.prices_eod",
                        "privilege": privilege,
                    },
                )
            assert (
                connection.scalar(
                    text(
                        "SELECT relrowsecurity FROM pg_catalog.pg_class "
                        "WHERE oid = pg_catalog.to_regclass(:table_name)"
                    ),
                    {"table_name": qualified_name},
                )
                is False
            )
            assert (
                connection.scalar(
                    text("SELECT has_table_privilege(:role, :table_name, 'INSERT')"),
                    {"role": migrated_database.owner_role, "table_name": qualified_name},
                )
                is True
            )


def test_fundamental_store_collapses_repeats_and_keeps_restatements(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 12, 31)
    with Session(migrated_database.owner_engine) as session, session.begin():
        security_id = store.upsert_us_equity_security(session, "SYNTHETIC", "Synthetic Public Co")
        rows = [
            _observation("net_income", date(2025, 3, 31), "10", date(2025, 4, 30)),
            _observation("net_income", date(2025, 6, 30), "10", date(2025, 7, 30)),
            _observation("net_income", date(2025, 9, 30), "10", date(2025, 10, 30)),
            _observation("net_income", date(2025, 12, 31), "10", date(2026, 1, 20)),
            _observation("net_income", date(2025, 12, 31), "10", date(2026, 1, 25)),
            _observation("net_income", date(2025, 12, 31), "20", date(2026, 2, 10)),
            _observation(
                "total_equity",
                date(2025, 12, 31),
                "100",
                date(2026, 1, 20),
                period_type="FY",
            ),
        ]
        assert (
            store.upsert_fundamentals(
                session, security_id, rows, as_of=on, fetched_at=datetime(2026, 12, 31, tzinfo=UTC)
            )
            == 6
        )
        assert (
            store.upsert_fundamentals(
                session, security_id, rows, as_of=on, fetched_at=datetime(2026, 12, 31, tzinfo=UTC)
            )
            == 0
        )
        persisted = list(
            session.scalars(
                select(FundamentalPIT)
                .where(FundamentalPIT.security_id == security_id)
                .order_by(FundamentalPIT.metric, FundamentalPIT.period_end, FundamentalPIT.known_at)
            )
        )
        revenue_period_rows = [
            (row.known_at, row.filed_at, row.is_latest_for_period, row.value)
            for row in persisted
            if row.metric == "net_income" and row.period_end == date(2025, 12, 31)
        ]
        factors_before = fundamental_factors(persisted, as_of=date(2026, 2, 1), price=None)
        factors_after = fundamental_factors(persisted, as_of=date(2026, 2, 11), price=None)
        version_count = session.scalar(
            select(func.count())
            .select_from(FundamentalPIT)
            .where(FundamentalPIT.security_id == security_id)
        )

    assert version_count == 6
    assert len(revenue_period_rows) == 2
    assert revenue_period_rows[0] == (
        datetime(2026, 1, 20, tzinfo=UTC),
        datetime(2026, 1, 20, tzinfo=UTC),
        False,
        Decimal("10.00000000"),
    )
    assert revenue_period_rows[1][2] is True
    assert revenue_period_rows[1][3] == Decimal("20.00000000")
    assert factors_before.roe == Decimal("0.4")
    assert factors_after.roe == Decimal("0.5")


def test_fundamentals_retention_prunes_periods_outside_the_storage_window(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 12, 31)
    with Session(migrated_database.owner_engine) as session, session.begin():
        security_id = store.upsert_us_equity_security(
            session, "RETENTION", "Synthetic Retention Co"
        )
        rejected = store.upsert_fundamentals(
            session,
            security_id,
            [
                _observation(
                    "revenue", date(2014, 12, 31), "100", date(2015, 2, 1), period_type="FY"
                )
            ],
            as_of=as_of,
        )
        old = FundamentalPIT(
            security_id=security_id,
            metric="net_income",
            period_end=date(2014, 12, 31),
            period_type="FY",
            source="SEC_EDGAR",
            known_at=datetime(2015, 2, 1, tzinfo=UTC),
            value=Decimal("10"),
            unit="USD",
            filed_at=datetime(2015, 2, 1, tzinfo=UTC),
            source_url="https://data.sec.gov/example/companyfacts.json",
            fetched_at=datetime(2015, 2, 1, tzinfo=UTC),
        )
        session.add(old)
        session.flush()
        pruned = store.prune_fundamentals(session, as_of)
        remains = session.scalar(
            select(func.count())
            .select_from(FundamentalPIT)
            .where(
                FundamentalPIT.security_id == security_id,
                FundamentalPIT.period_end == date(2014, 12, 31),
            )
        )

    assert rejected == 0
    assert pruned == 1
    assert remains == 0


def test_universe_membership_refresh_is_idempotent(
    migrated_database: MigratedDatabase,
) -> None:
    loaded_on = date(2026, 10, 29)
    with Session(migrated_database.owner_engine) as session, session.begin():
        first = store.upsert_us_equity_security(session, "SYNTHETIC", "Synthetic Public Co")
        second = store.upsert_us_equity_security(session, "SYNTHETIC", "Synthetic Public Co")
        assert first == second
        first_load = store.upsert_universe_members(
            session,
            "sp500",
            {first},
            loaded_on=loaded_on,
            source="SSGA_SPY_HOLDINGS",
            source_url="https://www.ssga.com/example.xlsx",
            survivorship_label="current constituents only (survivorship bias)",
        )
        second_load = store.upsert_universe_members(
            session,
            "sp500",
            {second},
            loaded_on=loaded_on,
            source="SSGA_SPY_HOLDINGS",
            source_url="https://www.ssga.com/example.xlsx",
            survivorship_label="current constituents only (survivorship bias)",
        )
        member_count = session.scalar(
            text(
                "SELECT count(*) FROM universe_members "
                "WHERE universe = 'sp500' AND removed_on IS NULL"
            )
        )

    assert first_load == {"added": 1, "removed": 0, "active": 1}
    assert second_load == {"added": 0, "removed": 0, "active": 1}
    assert member_count == 1
