"""Migration and household policy checks for fund history."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError
from tests.db.conftest import MigratedDatabase

from wealth.api.imports import _persist_fund_transactions
from wealth.db.models import (
    FundLaggingCheck,
    FundTransaction,
    HouseholdAccess,
    HouseholdMember,
    NavMonthly,
    Security,
)
from wealth.db.session import resolve_user, system_session, user_session
from wealth.marketdata.refresh import run_refresh
from wealth.marketdata.sources.amfi_history import parse_nav_history
from wealth.marketdata.store import missing_amfi_history_windows

pytestmark = pytest.mark.db


def test_0012_fund_migration_downgrades_and_upgrades(
    migrated_database: MigratedDatabase,
) -> None:
    database = migrated_database

    command.downgrade(database.config, "0011_shared_household")
    with database.owner_engine.connect() as connection:
        assert connection.scalar(text("SELECT to_regclass('nav_monthly')")) is None

    command.upgrade(database.config, "head")
    with database.owner_engine.connect() as connection:
        assert connection.scalar(text("SELECT to_regclass('nav_monthly')")) is not None
        assert connection.scalar(text("SELECT to_regclass('fund_transactions')")) is not None
        assert connection.scalar(text("SELECT to_regclass('fund_lagging_checks')")) is not None


def test_fund_records_follow_household_roles_and_shared_market_data_is_read_only(
    migrated_database: MigratedDatabase,
) -> None:
    database = migrated_database
    owner_id = resolve_user(
        database.api_engine,
        f"fund-check-owner-{uuid4()}",
        "fund-owner@example.test",
        "Synthetic Fund Owner",
    )
    viewer_id = resolve_user(
        database.api_engine,
        f"fund-check-viewer-{uuid4()}",
        "fund-viewer@example.test",
        "Synthetic Fund Viewer",
    )
    with system_session(database.owner_engine) as session:
        owner_member = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == owner_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert owner_member is not None
        security = Security(
            market="IN",
            symbol="AMFI-710001-G",
            name="Synthetic Large Company Fund Direct Growth",
            security_type="mf",
            currency="INR",
            sleeve="equity_india",
            category="mf:equity_india",
            sub_category="Large Cap",
            plan_type="direct",
            option="growth",
            amfi_scheme_code="710001",
        )
        session.add(security)
        session.flush()
        session.add(
            NavMonthly(
                security_id=security.id,
                month_end=date(2020, 1, 31),
                nav=Decimal("90.00"),
                source="AMFI",
            )
        )
        session.flush()
        fixture = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "marketdata"
            / "amfi_nav_history_small.txt"
        )
        history_rows = parse_nav_history(fixture.read_text(encoding="utf-8"))
        refresh = run_refresh(
            session,
            date(2026, 10, 28),
            only=("amfi_history",),
            fetchers={"amfi_history": lambda: history_rows},
        )
        assert refresh["counts"]["amfi_history"]["nav_monthly"] == 1
        assert refresh["counts"]["amfi_history"]["nav_monthly_pruned"] == 1
        missing_windows = missing_amfi_history_windows(session, date(2026, 10, 28))
        assert missing_windows[0][0] == date(2021, 10, 1)
        assert all((end - start).days + 1 <= 90 for start, end in missing_windows)
        session.add(
            HouseholdAccess(
                owner_user_id=owner_id,
                grantee_user_id=viewer_id,
                member_id=owner_member.id,
                role="viewer",
            )
        )
        session.flush()
        security_id = security.id
        member_id = owner_member.id

    with user_session(database.api_engine, owner_id) as session:
        session.add(
            FundTransaction(
                user_id=owner_id,
                member_id=member_id,
                security_id=security_id,
                transaction_date=date(2026, 10, 5),
                kind="buy",
                amount=Decimal("5000"),
                units=Decimal("50"),
                account_ref=None,
            )
        )
        payload = {
            "transactions": [
                {
                    "date": "2026-10-06",
                    "kind": "buy",
                    "amount": "6000",
                    "units": "60",
                }
            ]
        }
        _persist_fund_transactions(session, owner_id, member_id, security_id, payload, None)
        _persist_fund_transactions(session, owner_id, member_id, security_id, payload, None)
    with user_session(database.api_engine, owner_id) as session:
        assert session.scalar(select(func.count()).select_from(FundTransaction)) == 2

    with user_session(database.api_engine, viewer_id, owner_id) as session:
        assert (
            session.scalar(text("SELECT app_household_role(:owner_id)"), {"owner_id": owner_id})
            == "viewer"
        )
        assert session.scalar(select(FundTransaction.id)) is not None
        with pytest.raises(DBAPIError), session.begin_nested():
            session.add(
                FundLaggingCheck(
                    user_id=owner_id,
                    member_id=member_id,
                    security_id=security_id,
                    month_end=date(2026, 10, 31),
                    bottom_quartile=True,
                )
            )
            session.flush()
        assert session.scalar(select(NavMonthly.nav)) == Decimal("101.25")
        with pytest.raises(DBAPIError), session.begin_nested():
            session.add(
                NavMonthly(
                    security_id=security_id,
                    month_end=date(2026, 10, 31),
                    nav=Decimal("100"),
                    source="AMFI",
                )
            )
            session.flush()
        with pytest.raises(DBAPIError), session.begin_nested():
            session.execute(
                update(Security).where(Security.id == security_id).values(sub_category="Other")
            )

    with user_session(database.api_engine, viewer_id) as session:
        assert session.scalar(select(FundTransaction.id)) is None
