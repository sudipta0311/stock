"""Persist the optional C02 financial context with each profile."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003_profile_financial_context"
down_revision = "0002_api_role_and_bootstrap"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("profiles", Column("financial_context", JSONB()))


def downgrade() -> None:
    op.drop_column("profiles", "financial_context")
