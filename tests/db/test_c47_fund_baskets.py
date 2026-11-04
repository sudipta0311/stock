"""Storage guarantees for public fund-basket reference snapshots."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from alembic import command
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from conftest import MigratedDatabase
from wealth.marketdata.refresh import SourceBatchResult, _store_source
from wealth.marketdata.sources.fund_baskets import (
    BasketSnapshot,
    BasketWeight,
    FundBasketBatch,
)

pytestmark = pytest.mark.db


def _snapshot(as_of: date, weights: tuple[BasketWeight, ...]) -> BasketSnapshot:
    return BasketSnapshot(
        id="sp500",
        label="S&P 500",
        as_of=as_of,
        source="synthetic_test",
        source_url="https://example.test/sp500",
        weights_estimated=False,
        weight_source=None,
        members_total=len(weights),
        members_found=len(weights),
        weights=weights,
    )


def test_fund_basket_migration_downgrades_and_upgrades(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0027_recorded_buys")
    tables = set(inspect(migrated_database.owner_engine).get_table_names())
    assert "fund_baskets" not in tables
    assert "fund_basket_weights" not in tables

    command.upgrade(migrated_database.config, "head")
    tables = set(inspect(migrated_database.owner_engine).get_table_names())
    assert {"fund_baskets", "fund_basket_weights"} <= tables
    with migrated_database.owner_engine.connect() as connection:
        for table in ("fund_baskets", "fund_basket_weights"):
            rls_enabled = connection.scalar(
                text("SELECT relrowsecurity FROM pg_class WHERE oid = to_regclass(:table_name)"),
                {"table_name": table},
            )
            has_select = connection.scalar(
                text("SELECT has_table_privilege('wealth_app', :table_name, 'SELECT')"),
                {"table_name": table},
            )
            assert rls_enabled is False
            assert has_select is True


def test_basket_writes_replace_atomically_and_report_partial_sources(
    migrated_database: MigratedDatabase,
) -> None:
    fetched_at = datetime(2026, 11, 4, tzinfo=UTC)
    initial = _snapshot(
        date(2026, 11, 3),
        (BasketWeight("US", None, "SYNTH", "Synthetic Company", Decimal("100")),),
    )
    failed_update = _snapshot(
        date(2026, 11, 4),
        (
            BasketWeight("US", None, "DUPLICATE", "Synthetic First", Decimal("60")),
            BasketWeight("US", None, "DUPLICATE", "Synthetic Second", Decimal("40")),
        ),
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        first = _store_source(
            session,
            "fund_baskets",
            FundBasketBatch({"sp500": initial}, {"nasdaq100": "ConnectionError"}),
            on=initial.as_of,
            fetched_at=fetched_at,
        )
        assert isinstance(first, SourceBatchResult)
        assert first.counts["baskets_stored"] == 1
        assert first.counts["baskets"]["sp500"] == {
            "rows": 1,
            "members_total": 1,
            "members_found": 1,
            "as_of": "2026-11-03",
        }
        assert first.errors == {"nasdaq100": "ConnectionError"}
        second = _store_source(
            session,
            "fund_baskets",
            FundBasketBatch({"sp500": failed_update}, {}),
            on=failed_update.as_of,
            fetched_at=fetched_at,
        )
        assert isinstance(second, SourceBatchResult)
        assert second.counts["baskets_stored"] == 0
        assert second.errors == {"sp500": "IntegrityError"}

    with Session(migrated_database.owner_engine) as session:
        from wealth.db.models import FundBasket, FundBasketWeight

        basket = session.get(FundBasket, "sp500")
        weights = list(session.query(FundBasketWeight).filter_by(basket_id="sp500"))

    assert basket is not None and basket.as_of == date(2026, 11, 3)
    assert len(weights) == 1 and weights[0].symbol == "SYNTH"
