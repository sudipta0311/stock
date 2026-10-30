"""Add selector index closes, rule proposals, and paper scorekeeping."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

revision = "0024_etf_watchers_scorekeeper"
down_revision = "0023_macro_geo"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)

    op.add_column("index_valuations", Column("close", Numeric(20, 8)))
    op.create_check_constraint(
        "ck_index_valuations_close_positive_finite",
        "index_valuations",
        "close IS NULL OR (close > 0 AND close > '-Infinity'::numeric "
        "AND close < 'Infinity'::numeric)",
    )

    op.create_table(
        "rule_change_proposals",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column("kind", String(32), nullable=False),
        Column("path", String(256), nullable=False),
        Column("current_value", JSONB),
        Column("proposed_value", JSONB),
        Column("evidence", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        Column("status", String(16), nullable=False, server_default="pending"),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        Column("decided_at", DateTime(timezone=True)),
        Column("decided_by", String(120)),
        CheckConstraint(
            "kind IN ('small_savings_rate', 'epf_rate', 'rulebook')",
            name="ck_rule_change_proposals_kind",
        ),
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_rule_change_proposals_status",
        ),
    )
    op.create_index(
        "ix_rule_change_proposals_status_created",
        "rule_change_proposals",
        ["status", "created_at"],
    )

    op.create_table(
        "paper_positions",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column("week", String(8), nullable=False),
        Column("market", String(2), nullable=False),
        Column("dossier_id", String(128), nullable=False),
        Column("security_id", Uuid(as_uuid=True), ForeignKey("securities.id", ondelete="SET NULL")),
        Column("symbol", String(32), nullable=False),
        Column("confidence", String(8), nullable=False),
        Column("currency", String(3), nullable=False),
        Column("notional", Numeric(20, 4), nullable=False),
        Column("entry_date", Date(), nullable=False),
        Column("entry_price", Numeric(20, 8), nullable=False),
        Column("benchmark_symbol", String(32)),
        Column("benchmark_status", String(32), nullable=False),
        Column("benchmark_entry_price", Numeric(20, 8)),
        Column("benchmark_entry_date", Date()),
        Column("current_price", Numeric(20, 8)),
        Column("benchmark_current_price", Numeric(20, 8)),
        Column("return_pct", Numeric(12, 6)),
        Column("benchmark_return_pct", Numeric(12, 6)),
        Column("excess_return_pct", Numeric(12, 6)),
        Column("status", String(16), nullable=False, server_default="open"),
        Column("exit_date", Date()),
        Column("exit_reason", String(24)),
        Column(
            "red_team_falsifiers",
            JSONB,
            nullable=False,
            server_default=text("'[]'::jsonb"),
        ),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        UniqueConstraint("week", "dossier_id", name="uq_paper_positions_week_dossier"),
        CheckConstraint("market IN ('IN', 'US')", name="ck_paper_positions_market"),
        CheckConstraint("currency IN ('INR', 'USD')", name="ck_paper_positions_currency"),
        CheckConstraint("status IN ('open', 'closed')", name="ck_paper_positions_status"),
        CheckConstraint(
            "benchmark_status IN ('available', 'benchmark_unavailable')",
            name="ck_paper_positions_benchmark_status",
        ),
        CheckConstraint("notional > 0", name="ck_paper_positions_notional_positive"),
        CheckConstraint("entry_price > 0", name="ck_paper_positions_entry_price_positive"),
    )
    op.create_index("ix_paper_positions_market_week", "paper_positions", ["market", "week"])
    op.create_index("ix_paper_positions_status", "paper_positions", ["status"])

    op.create_table(
        "track_record",
        Column("week", String(8), primary_key=True),
        Column("market", String(2), primary_key=True),
        Column("ideas_count", Integer, nullable=False),
        Column("ahead_count", Integer, nullable=False),
        Column("avg_excess_return_pct", Numeric(12, 6)),
        Column("details", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("market IN ('IN', 'US')", name="ck_track_record_market"),
        CheckConstraint("ideas_count >= 0", name="ck_track_record_ideas_non_negative"),
        CheckConstraint(
            "ahead_count >= 0 AND ahead_count <= ideas_count",
            name="ck_track_record_ahead_count",
        ),
        CheckConstraint(
            "avg_excess_return_pct IS NULL OR "
            "(avg_excess_return_pct > '-Infinity'::numeric "
            "AND avg_excess_return_pct < 'Infinity'::numeric)",
            name="ck_track_record_excess_finite",
        ),
    )

    system_tables = ("rule_change_proposals", "paper_positions", "track_record")
    qualified_tables = ", ".join(f"{quoted_schema}.{name}" for name in system_tables)
    op.execute(f"REVOKE ALL ON TABLE {qualified_tables} FROM PUBLIC, wealth_app, wealth_api")
    for name in system_tables:
        op.execute(f"ALTER TABLE {quoted_schema}.{name} DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_index("ix_paper_positions_status", table_name="paper_positions")
    op.drop_index("ix_paper_positions_market_week", table_name="paper_positions")
    op.drop_table("track_record")
    op.drop_table("paper_positions")
    op.drop_index("ix_rule_change_proposals_status_created", table_name="rule_change_proposals")
    op.drop_table("rule_change_proposals")
    op.drop_constraint(
        "ck_index_valuations_close_positive_finite", "index_valuations", type_="check"
    )
    op.drop_column("index_valuations", "close")
