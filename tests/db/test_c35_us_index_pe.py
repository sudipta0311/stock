from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import IndexValuation
from wealth.marketdata.refresh import run_refresh
from wealth.marketdata.sources.macro_geo import IndexValuationObservation
from wealth.marketdata.sources.us_index_pe import (
    BACKFILL_SOURCE,
    CURRENT_SOURCE,
    UsIndexPeBatch,
)

pytestmark = pytest.mark.db


def test_low_coverage_refresh_reports_error_and_stores_nothing(
    migrated_database: MigratedDatabase,
) -> None:
    batch = UsIndexPeBatch([], {"coverage": "CoverageBelow90Pct"})
    with Session(migrated_database.owner_engine) as session, session.begin():
        result = run_refresh(
            session,
            date(2026, 10, 31),
            only=("us_index_pe",),
            fetchers={"us_index_pe": lambda: batch},
        )
        count = session.scalar(
            select(func.count())
            .select_from(IndexValuation)
            .where(IndexValuation.index_name == "S&P 500")
        )

    assert result["status"] == "failed"
    assert result["errors"] == {"us_index_pe:coverage": "CoverageBelow90Pct"}
    assert count == 0


def test_backfill_refresh_is_idempotent_by_index_and_date(
    migrated_database: MigratedDatabase,
) -> None:
    batch = UsIndexPeBatch(
        [
            IndexValuationObservation(
                "S&P 500",
                date(2025, 10, 31),
                Decimal("23.4375"),
                None,
                None,
                BACKFILL_SOURCE,
                "https://www.ssga.com/synthetic-nav.xlsx",
            ),
            IndexValuationObservation(
                "S&P 500",
                date(2026, 10, 30),
                Decimal("46.875"),
                None,
                None,
                CURRENT_SOURCE,
                "https://www.ssga.com/synthetic-holdings.xlsx",
            ),
        ]
    )
    for _ in range(2):
        with Session(migrated_database.owner_engine) as session, session.begin():
            result = run_refresh(
                session,
                date(2026, 10, 31),
                only=("us_index_pe",),
                fetchers={"us_index_pe": lambda: batch},
            )
            assert result["status"] == "succeeded"

    with Session(migrated_database.owner_engine) as session:
        rows = list(
            session.scalars(
                select(IndexValuation)
                .where(IndexValuation.index_name == "S&P 500")
                .order_by(IndexValuation.as_of)
            )
        )

    assert len(rows) == 2
    assert [(row.as_of, row.pe, row.source) for row in rows] == [
        (date(2025, 10, 31), Decimal("23.43750000"), BACKFILL_SOURCE),
        (date(2026, 10, 30), Decimal("46.87500000"), CURRENT_SOURCE),
    ]


def test_refresh_removes_backfill_months_that_no_longer_pass_the_gate(
    migrated_database: MigratedDatabase,
) -> None:
    stale_batch = UsIndexPeBatch(
        [
            IndexValuationObservation(
                "S&P 500",
                date(2024, 8, 31),
                Decimal("21.4"),
                None,
                None,
                BACKFILL_SOURCE,
                "https://www.ssga.com/synthetic-nav.xlsx",
            )
        ]
    )
    latest_batch = UsIndexPeBatch(
        [
            IndexValuationObservation(
                "S&P 500",
                date(2025, 6, 30),
                Decimal("30.1"),
                None,
                None,
                BACKFILL_SOURCE,
                "https://www.ssga.com/synthetic-nav.xlsx",
            ),
            IndexValuationObservation(
                "S&P 500",
                date(2026, 10, 30),
                Decimal("27.1"),
                None,
                None,
                CURRENT_SOURCE,
                "https://www.ssga.com/synthetic-holdings.xlsx",
            ),
        ]
    )
    for batch in (stale_batch, latest_batch):
        with Session(migrated_database.owner_engine) as session, session.begin():
            run_refresh(
                session,
                date(2026, 10, 31),
                only=("us_index_pe",),
                fetchers={"us_index_pe": lambda batch=batch: batch},
            )

    with Session(migrated_database.owner_engine) as session:
        rows = list(
            session.scalars(
                select(IndexValuation)
                .where(IndexValuation.index_name == "S&P 500")
                .order_by(IndexValuation.as_of)
            )
        )

    assert [(row.as_of, row.source) for row in rows] == [
        (date(2025, 6, 30), BACKFILL_SOURCE),
        (date(2026, 10, 30), CURRENT_SOURCE),
    ]
