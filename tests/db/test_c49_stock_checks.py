from datetime import date, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import HoldingLot, HouseholdMember, PriceEod, Security, StockCheck, User
from wealth.research.stock_checks import run_stock_checks

pytestmark = pytest.mark.db


def test_stock_checks_migration_can_downgrade_and_upgrade_fresh_schema(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0028_fund_baskets")
    assert "stock_checks" not in inspect(migrated_database.owner_engine).get_table_names()

    command.upgrade(migrated_database.config, "head")

    assert "stock_checks" in inspect(migrated_database.owner_engine).get_table_names()
    columns = {
        column["name"]
        for column in inspect(migrated_database.owner_engine).get_columns("stock_checks")
    }
    assert "user_id" not in columns


def test_held_share_outside_lists_gets_price_dma_and_old_weeks_are_pruned(
    migrated_database: MigratedDatabase,
) -> None:
    week = "2026-W44"
    as_of = date(2026, 10, 30)
    user_id = uuid4()
    with Session(migrated_database.owner_engine) as session, session.begin():
        user = User(
            id=user_id,
            email=f"stock-check-{user_id.hex}@example.test",
            auth_subject=f"stock-check-{user_id.hex}",
        )
        security = Security(
            market="EU",
            symbol="SYNTH-EU",
            name="Synthetic European Share",
            security_type="equity",
            currency="EUR",
            sleeve="equity_global",
            sector="Synthetic sector",
            industry="Synthetic industry",
        )
        session.add(user)
        session.flush()
        self_member_id = session.scalar(
            select(HouseholdMember.id).where(
                HouseholdMember.user_id == user_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert self_member_id is not None
        session.add(security)
        session.flush()
        session.add(
            HoldingLot(
                user_id=user_id,
                member_id=self_member_id,
                security_id=security.id,
                quantity=Decimal("2"),
                cost_per_unit=None,
                acquired_on=None,
                source="manual",
                currency="EUR",
            )
        )
        for index in range(200):
            session.add(
                PriceEod(
                    security_id=security.id,
                    as_of=as_of - timedelta(days=199 - index),
                    close=Decimal(index + 1),
                    source="SYNTHETIC",
                    source_url="https://example.test/synthetic-price",
                )
            )
        session.add(
            StockCheck(
                security_id=security.id,
                week="2026-W20",
                as_of=date(2026, 5, 15),
                sources={},
                revenue_yoy_pct=[],
            )
        )
    result = run_stock_checks(migrated_database.owner_engine, week)

    assert result["checked"] == 1
    assert result["held_checked"] == 1
    assert result["held_not_covered"] == 1
    with migrated_database.owner_engine.connect() as connection:
        rows = connection.execute(
            text("SELECT week, price, dma200, quant_decile FROM stock_checks")
        ).all()
    assert len(rows) == 1
    assert rows[0].week == week
    assert rows[0].price == Decimal("200")
    assert rows[0].dma200 == Decimal("100.5")
    assert rows[0].quant_decile is None
