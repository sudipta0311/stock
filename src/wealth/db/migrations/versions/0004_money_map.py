"""Persist Money Map checklist progress and asset valuations."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004_money_map"
down_revision = "0003_profile_financial_context"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profiles",
        Column("checklist", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    )
    op.add_column("manual_assets", Column("valuation", JSONB()))


def downgrade() -> None:
    op.drop_column("manual_assets", "valuation")
    op.drop_column("profiles", "checklist")
