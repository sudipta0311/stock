"""Add shared point-in-time fundamentals and universe membership."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    Uuid,
    text,
)

revision = "0017_fundamentals_pit"
down_revision = "0016_research"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)

    op.create_table(
        "fundamentals_pit",
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("metric", String(48), primary_key=True),
        Column("period_end", Date(), primary_key=True),
        Column("period_type", String(2), primary_key=True),
        Column("source", String(24), primary_key=True),
        Column("known_at", DateTime(timezone=True), primary_key=True),
        Column("value", Numeric(30, 8), nullable=False),
        Column("unit", String(24), nullable=False),
        Column("filed_at", DateTime(timezone=True), nullable=False),
        Column("source_url", Text(), nullable=False),
        Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        Column("is_latest_for_period", Boolean(), nullable=False, server_default=text("true")),
        CheckConstraint("period_type IN ('Q', 'FY')", name="ck_fundamentals_pit_period_type"),
        CheckConstraint(
            "value > '-Infinity'::numeric AND value < 'Infinity'::numeric",
            name="ck_fundamentals_pit_value_finite",
        ),
    )
    op.create_index(
        "ix_fundamentals_pit_asof",
        "fundamentals_pit",
        ["security_id", "metric", "known_at"],
    )
    op.create_index(
        "ix_fundamentals_pit_period",
        "fundamentals_pit",
        ["security_id", "metric", "period_end", "period_type"],
    )

    op.create_table(
        "universe_members",
        Column("universe", String(24), primary_key=True),
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("added_on", Date(), primary_key=True),
        Column("removed_on", Date()),
        Column("source", String(40), nullable=False),
        Column("source_url", Text(), nullable=False),
        Column("survivorship_label", String(80), nullable=False),
        CheckConstraint(
            "removed_on IS NULL OR removed_on >= added_on",
            name="ck_universe_members_dates",
        ),
    )
    op.create_index(
        "uq_universe_members_active",
        "universe_members",
        ["universe", "security_id"],
        unique=True,
        postgresql_where=text("removed_on IS NULL"),
    )
    op.create_index("ix_universe_members_added_on", "universe_members", ["universe", "added_on"])

    shared_tables = ("fundamentals_pit", "universe_members")
    qualified_tables = ", ".join(f"{quoted_schema}.{table}" for table in shared_tables)
    op.execute(f"GRANT SELECT ON TABLE {qualified_tables} TO wealth_app")
    for table_name in shared_tables:
        op.execute(f"ALTER TABLE {quoted_schema}.{table_name} DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_index("ix_universe_members_added_on", table_name="universe_members")
    op.drop_index("uq_universe_members_active", table_name="universe_members")
    op.drop_table("universe_members")
    op.drop_index("ix_fundamentals_pit_period", table_name="fundamentals_pit")
    op.drop_index("ix_fundamentals_pit_asof", table_name="fundamentals_pit")
    op.drop_table("fundamentals_pit")
