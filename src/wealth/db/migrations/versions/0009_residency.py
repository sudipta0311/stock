"""Add residency profile data and goal currency support."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB

revision = "0009_residency"
down_revision = "0008_alerts_delivery"
branch_labels = None
depends_on = None

_DEFAULT_RESIDENCY = (
    "jsonb_build_object('status', 'RI', 'country', NULL, 'us_person', false, "
    "'planned_return_year', NULL, 'settle_plan', 'undecided')"
)
_GOAL_CURRENCIES = "'INR', 'DKK', 'EUR', 'USD'"


def upgrade() -> None:
    op.add_column(
        "profiles",
        Column("residency", JSONB(), nullable=False, server_default=text(_DEFAULT_RESIDENCY)),
    )
    op.drop_constraint("ck_goals_currency", "goals", type_="check")
    op.create_check_constraint("ck_goals_currency", "goals", f"currency IN ({_GOAL_CURRENCIES})")


def downgrade() -> None:
    op.drop_constraint("ck_goals_currency", "goals", type_="check")
    op.create_check_constraint("ck_goals_currency", "goals", "currency IN ('INR', 'USD')")
    op.drop_column("profiles", "residency")
