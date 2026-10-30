"""Add shared index valuations, Treasury series, and official feed evidence."""

from __future__ import annotations

from alembic import op
from sqlalchemy import CheckConstraint, Column, Date, DateTime, Numeric, String, Text, text

revision = "0023_macro_geo"
down_revision = "0022_prompt_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)

    op.create_table(
        "index_valuations",
        Column("index_name", String(48), primary_key=True),
        Column("as_of", Date(), primary_key=True),
        Column("pe", Numeric(20, 8)),
        Column("pb", Numeric(20, 8)),
        Column("div_yield", Numeric(20, 8)),
        Column("source", String(40), nullable=False),
        Column("source_url", Text(), nullable=False),
        CheckConstraint(
            "pe IS NULL OR (pe > '-Infinity'::numeric AND pe < 'Infinity'::numeric)",
            name="ck_index_valuations_pe_finite",
        ),
        CheckConstraint(
            "pb IS NULL OR (pb > '-Infinity'::numeric AND pb < 'Infinity'::numeric)",
            name="ck_index_valuations_pb_finite",
        ),
        CheckConstraint(
            "div_yield IS NULL OR (div_yield > '-Infinity'::numeric "
            "AND div_yield < 'Infinity'::numeric)",
            name="ck_index_valuations_div_yield_finite",
        ),
    )

    op.create_table(
        "macro_series",
        Column("series", String(40), primary_key=True),
        Column("as_of", Date(), primary_key=True),
        Column("value", Numeric(20, 8), nullable=False),
        Column("source", String(40), nullable=False),
        Column("source_url", Text(), nullable=False),
        CheckConstraint(
            "value > '-Infinity'::numeric AND value < 'Infinity'::numeric",
            name="ck_macro_series_value_finite",
        ),
    )

    op.create_table(
        "feed_items",
        Column("source", String(32), primary_key=True),
        Column("guid", String(512), primary_key=True),
        Column("title", String(1024), nullable=False),
        Column("url", Text(), nullable=False),
        Column("published_at", DateTime(timezone=True)),
        Column("summary", String(1024), nullable=False),
        Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint(
            "source IN ('pib', 'rbi', 'ustr', 'federal_register')",
            name="ck_feed_items_source",
        ),
        CheckConstraint("octet_length(title) <= 1024", name="ck_feed_items_title_bytes"),
        CheckConstraint("octet_length(guid) <= 512", name="ck_feed_items_guid_bytes"),
        CheckConstraint("octet_length(url) <= 2048", name="ck_feed_items_url_bytes"),
        CheckConstraint("octet_length(summary) <= 1024", name="ck_feed_items_summary_bytes"),
    )
    op.create_index("ix_feed_items_published_at", "feed_items", ["published_at"])
    op.create_index("ix_feed_items_source_published_at", "feed_items", ["source", "published_at"])

    shared_tables = ("index_valuations", "macro_series", "feed_items")
    qualified_tables = ", ".join(f"{quoted_schema}.{table}" for table in shared_tables)
    op.execute(f"GRANT SELECT ON TABLE {qualified_tables} TO wealth_app")
    for table_name in shared_tables:
        op.execute(f"ALTER TABLE {quoted_schema}.{table_name} DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_index("ix_feed_items_source_published_at", table_name="feed_items")
    op.drop_index("ix_feed_items_published_at", table_name="feed_items")
    op.drop_table("feed_items")
    op.drop_table("macro_series")
    op.drop_table("index_valuations")
