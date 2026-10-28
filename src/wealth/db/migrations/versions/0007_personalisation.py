"""Add investing preferences, tenant weekly plans, and frozen research packs."""

from __future__ import annotations

from alembic import op
from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, String, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007_personalisation"
down_revision = "0006_market_data"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("profiles", Column("investing", JSONB()))

    op.create_table(
        "weekly_plans",
        Column(
            "user_id",
            Uuid(as_uuid=True),
            ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("week", String(8), primary_key=True),
        Column("plan", JSONB(), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint(
            "week ~ '^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$'",
            name="ck_weekly_plans_week",
        ),
    )
    op.create_index("ix_weekly_plans_week", "weekly_plans", ["week"])
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE weekly_plans TO wealth_app")
    op.execute("ALTER TABLE weekly_plans ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY weekly_plans_user_isolation ON weekly_plans FOR ALL TO wealth_app "
        "USING (user_id = nullif(current_setting('app.user_id', true), '')::uuid) "
        "WITH CHECK (user_id = nullif(current_setting('app.user_id', true), '')::uuid)"
    )

    op.create_table(
        "research_packs",
        Column("week", String(8), primary_key=True),
        Column("pack", JSONB(), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint(
            "week ~ '^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$'",
            name="ck_research_packs_week",
        ),
    )
    op.execute("GRANT SELECT ON TABLE research_packs TO wealth_app")
    op.execute("ALTER TABLE research_packs DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_table("research_packs")
    op.drop_table("weekly_plans")
    op.drop_column("profiles", "investing")
