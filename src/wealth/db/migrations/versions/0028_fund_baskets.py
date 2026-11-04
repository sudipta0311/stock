"""Store latest public fund and index basket weights."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    text,
)

revision = "0028_fund_baskets"
down_revision = "0027_recorded_buys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)

    op.create_table(
        "fund_baskets",
        Column("id", String(48), primary_key=True),
        Column("label", String(120), nullable=False),
        Column("as_of", Date(), nullable=False),
        Column("source", String(48), nullable=False),
        Column("source_url", Text(), nullable=False),
        Column("weights_estimated", Boolean(), nullable=False),
        Column("weight_source", String(48)),
        Column("members_total", Integer(), nullable=False),
        Column("members_found", Integer(), nullable=False),
        Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("members_total >= 0", name="ck_fund_baskets_members_total_non_negative"),
        CheckConstraint(
            "members_found >= 0 AND members_found <= members_total",
            name="ck_fund_baskets_members_found_range",
        ),
    )
    op.create_table(
        "fund_basket_weights",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column(
            "basket_id",
            String(48),
            ForeignKey("fund_baskets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("market", String(2)),
        Column("isin", String(12)),
        Column("symbol", String(32)),
        Column("name", String(160), nullable=False),
        Column("weight_pct", Numeric(9, 6), nullable=False),
        CheckConstraint(
            "market IS NULL OR market IN ('IN', 'US')",
            name="ck_fund_basket_weights_market",
        ),
        CheckConstraint(
            "isin IS NOT NULL OR symbol IS NOT NULL",
            name="ck_fund_basket_weights_company_key",
        ),
        CheckConstraint(
            "weight_pct >= 0 AND weight_pct > '-Infinity'::numeric "
            "AND weight_pct < 'Infinity'::numeric",
            name="ck_fund_basket_weights_non_negative_finite",
        ),
    )
    op.create_index("ix_fund_basket_weights_isin", "fund_basket_weights", ["isin"])
    op.create_index(
        "ix_fund_basket_weights_market_symbol",
        "fund_basket_weights",
        ["market", "symbol"],
    )
    op.create_index(
        "uq_fund_basket_weights_basket_isin",
        "fund_basket_weights",
        ["basket_id", "isin"],
        unique=True,
        postgresql_where=text("isin IS NOT NULL"),
    )
    op.create_index(
        "uq_fund_basket_weights_basket_symbol",
        "fund_basket_weights",
        ["basket_id", "market", "symbol"],
        unique=True,
        postgresql_where=text("symbol IS NOT NULL AND isin IS NULL"),
    )
    op.execute(
        f"GRANT SELECT ON TABLE {quoted_schema}.fund_baskets, "
        f"{quoted_schema}.fund_basket_weights TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.fund_baskets DISABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {quoted_schema}.fund_basket_weights DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_index("uq_fund_basket_weights_basket_symbol", table_name="fund_basket_weights")
    op.drop_index("uq_fund_basket_weights_basket_isin", table_name="fund_basket_weights")
    op.drop_index("ix_fund_basket_weights_market_symbol", table_name="fund_basket_weights")
    op.drop_index("ix_fund_basket_weights_isin", table_name="fund_basket_weights")
    op.drop_table("fund_basket_weights")
    op.drop_table("fund_baskets")
