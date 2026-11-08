from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import (
    FundamentalPIT,
    HoldingLot,
    HouseholdMember,
    PriceEod,
    Security,
    StockCheck,
    User,
)
from wealth.research import stock_checks
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
    monkeypatch: pytest.MonkeyPatch,
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
            market="IN",
            symbol="SYNTH-OUTSIDE",
            name="Synthetic Unlisted Share",
            security_type="equity",
            currency="INR",
            sleeve="equity_india",
            sector="Synthetic sector",
            industry="Synthetic industry",
        )
        screened_security = Security(
            market="IN",
            symbol="SYNTH-SCREENED",
            name="Synthetic Screened Share",
            security_type="equity",
            currency="INR",
            sleeve="equity_india",
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
        session.add_all([security, screened_security])
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
                currency="INR",
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
        for period_end, value, known_on in (
            (date(2025, 9, 30), Decimal("100"), datetime(2025, 10, 15, tzinfo=UTC)),
            (date(2026, 9, 30), Decimal("120"), datetime(2026, 10, 15, tzinfo=UTC)),
        ):
            session.add(
                FundamentalPIT(
                    security_id=security.id,
                    metric="revenue",
                    period_end=period_end,
                    period_type="Q",
                    source="SYNTHETIC",
                    known_at=known_on,
                    value=value,
                    unit="INR",
                    filed_at=known_on,
                    source_url="https://example.test/synthetic-fundamental",
                )
            )
        screened_security_id = screened_security.id
        held_security_id = security.id

    monkeypatch.setattr(
        stock_checks,
        "_screened_rows",
        lambda *_args: {str(screened_security_id): {"decile": 6, "coverage": "synthetic"}},
    )
    result = run_stock_checks(migrated_database.owner_engine, week)

    assert result["checked"] == 2
    assert result["held_checked"] == 1
    assert result["held_not_covered"] == 1
    assert result["coverage"]["quant_decile"] == 1
    assert result["held_coverage"] == {
        "price": 1,
        "dma200": 1,
        "revenue_yoy_pct": 1,
        "pe_vs_sector": 0,
        "pe_percentile": 0,
        "quant_decile": 0,
    }
    with migrated_database.owner_engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT security_id, week, price, dma200, quant_decile, "
                "revenue_yoy_pct, sources FROM stock_checks"
            )
        ).all()
    assert len(rows) == 2
    held_row = next(row for row in rows if row.security_id == held_security_id)
    assert held_row.week == week
    assert held_row.price == Decimal("200")
    assert held_row.dma200 == Decimal("100.5")
    assert held_row.quant_decile is None
    assert len(held_row.revenue_yoy_pct) == 1
    assert Decimal(held_row.revenue_yoy_pct[0]) == Decimal("20")
    assert held_row.sources["fundamentals"]["revenue_period_ends"]
