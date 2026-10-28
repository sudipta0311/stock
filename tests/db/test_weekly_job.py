"""Exercise the actual weekly-job module against an isolated seeded golden user."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tests.policy.test_personalise import synthetic_user_data

from conftest import MigratedDatabase
from wealth.db.models import (
    FxRate,
    HoldingLot,
    ManualAsset,
    PriceEod,
    Profile,
    Security,
    User,
    WeeklyPlan,
)
from wealth.db.models import (
    Goal as GoalRow,
)
from wealth.policy.store import actions_for_week

pytestmark = pytest.mark.db
_INDIA = ZoneInfo("Asia/Kolkata")


def test_weekly_cli_runs_for_seeded_golden_user(migrated_database: MigratedDatabase) -> None:
    user_data = synthetic_user_data()
    as_of = datetime.now(_INDIA).date()
    year, iso_week, _weekday = as_of.isocalendar()
    week = f"{year:04d}-W{iso_week:02d}"
    user_id = uuid4()
    india_security_id = uuid4()
    us_security_id = uuid4()
    india_lot_id = uuid4()
    us_lot_id = uuid4()

    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(
            User(
                id=user_id,
                email=f"{user_id.hex}@example.test",
                display_name="Synthetic Golden User",
                auth_subject=f"synthetic-golden-{user_id}",
            )
        )
        session.flush()
        session.add(
            Profile(
                user_id=user_id,
                tax_regime=user_data.profile.tax_regime,
                tax_slab_pct=user_data.profile.tax_slab_pct,
                city_tier=user_data.profile.city_tier,
                risk_profile=user_data.profile.risk_profile,
                financial_context=user_data.financial_context.model_dump(mode="json"),
                investing=user_data.investing.model_dump(mode="json"),
            )
        )
        session.add_all(
            [
                GoalRow(
                    user_id=user_id,
                    name=goal.name,
                    goal_type=goal.goal_type.value,
                    target_amount=goal.target_amount,
                    target_year=goal.target_year,
                    currency=goal.currency,
                    priority=goal.priority,
                )
                for goal in user_data.goals
            ]
        )
        session.flush()
        session.add_all(
            [
                ManualAsset(
                    user_id=user_id,
                    asset_type=asset.asset_type,
                    details=asset.model_dump(mode="json"),
                    status="active",
                )
                for asset, _valuation in user_data.valued_assets
            ]
        )
        session.add_all(
            [
                Security(
                    id=india_security_id,
                    market="IN",
                    symbol="SYNTH-GOLDEN-IN",
                    name="Synthetic India Equity",
                    security_type="equity",
                    currency="INR",
                    sleeve="equity_india",
                    category="equity:india",
                ),
                Security(
                    id=us_security_id,
                    market="US",
                    symbol="SYNTH-GOLDEN-ETF",
                    name="Synthetic Global ETF",
                    security_type="etf",
                    currency="USD",
                    sleeve="equity_global",
                    category="etf:global",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                HoldingLot(
                    id=india_lot_id,
                    user_id=user_id,
                    security_id=india_security_id,
                    quantity=20,
                    cost_per_unit=350,
                    acquired_on=date(2025, 1, 15),
                    source="cas",
                ),
                HoldingLot(
                    id=us_lot_id,
                    user_id=user_id,
                    security_id=us_security_id,
                    quantity=2,
                    cost_per_unit=160,
                    acquired_on=date(2025, 1, 15),
                    source="cas",
                ),
            ]
        )
        session.add_all(
            [
                PriceEod(
                    security_id=india_security_id,
                    as_of=as_of,
                    close=450,
                    source="synthetic_fixture",
                    source_url="https://example.test/price",
                ),
                PriceEod(
                    security_id=us_security_id,
                    as_of=as_of,
                    close=200,
                    source="synthetic_fixture",
                    source_url="https://example.test/price",
                ),
                FxRate(
                    pair="USD/INR",
                    as_of=as_of,
                    rate=Decimal("83.50"),
                    source="synthetic_fixture",
                    source_url="https://example.test/fx",
                ),
            ]
        )

    scoped_url = migrated_database.owner_engine.url.render_as_string(hide_password=False)
    environment = os.environ.copy()
    environment["OWNER_DATABASE_URL"] = scoped_url
    environment["TEST_DATABASE_URL"] = scoped_url
    result = subprocess.run(
        [sys.executable, "-m", "wealth.policy.run_weekly", "--week", week],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "weekly job CLI failed"
    counts = json.loads(result.stdout)
    assert counts["failed"] == 0
    assert counts["plans"] == 1
    assert counts["users"] == 1
    assert counts["week"] == week
    assert counts["actions"] == 3

    with Session(migrated_database.owner_engine) as session:
        saved = session.get(WeeklyPlan, (user_id, week))
        actions = actions_for_week(session, user_id, week)
        plan_count = session.scalar(
            select(func.count()).select_from(WeeklyPlan).where(WeeklyPlan.user_id == user_id)
        )
    assert saved is not None
    assert len(saved.plan["inputs_digest"]) == 64
    assert len(saved.plan["actions"]) == counts["actions"]
    assert "SELL_TRIM" in {action["kind"] for action in saved.plan["actions"]}
    assert len(actions) == counts["actions"]
    assert plan_count == 1
