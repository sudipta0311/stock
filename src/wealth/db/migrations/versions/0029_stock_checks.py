"""Store weekly public facts for held share reviews."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

revision = "0029_stock_checks"
down_revision = "0028_fund_baskets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)

    op.create_table(
        "stock_checks",
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("week", String(8), primary_key=True),
        Column("as_of", Date(), nullable=False),
        Column("quant_decile", SmallInteger()),
        Column("factor_coverage", String(32)),
        Column("pe_percentile", Numeric(8, 4)),
        Column("pe_vs_sector", Numeric(20, 8)),
        Column("revenue_yoy_pct", JSONB(), nullable=False, server_default=text("'[]'::jsonb")),
        Column("price", Numeric(20, 8)),
        Column("dma200", Numeric(20, 8)),
        Column("below_dma200", Boolean()),
        Column("sources", JSONB(), nullable=False, server_default=text("'{}'::jsonb")),
        CheckConstraint(
            "week ~ '^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$'",
            name="ck_stock_checks_week",
        ),
        CheckConstraint(
            "quant_decile IS NULL OR quant_decile BETWEEN 1 AND 10",
            name="ck_stock_checks_quant_decile",
        ),
        CheckConstraint(
            "pe_percentile IS NULL OR pe_percentile BETWEEN 0 AND 100",
            name="ck_stock_checks_pe_percentile",
        ),
        CheckConstraint("price IS NULL OR price > 0", name="ck_stock_checks_price_positive"),
        CheckConstraint("dma200 IS NULL OR dma200 > 0", name="ck_stock_checks_dma200_positive"),
    )
    op.create_index("ix_stock_checks_week", "stock_checks", ["week"])
    op.execute(f"GRANT SELECT ON TABLE {quoted_schema}.stock_checks TO wealth_app")
    op.execute(f"ALTER TABLE {quoted_schema}.stock_checks DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_index("ix_stock_checks_week", table_name="stock_checks")
    op.drop_table("stock_checks")
