"""Add shared point-in-time market data tables."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006_market_data"
down_revision = "0005_imports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("securities", Column("amfi_scheme_code", String(16)))
    op.add_column("securities", Column("nse_symbol", String(32)))
    op.add_column("securities", Column("series", String(8)))
    op.create_index("ix_securities_amfi_scheme_code", "securities", ["amfi_scheme_code"])
    op.create_index("ix_securities_nse_symbol", "securities", ["nse_symbol"])

    op.create_table(
        "prices_eod",
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("as_of", Date(), primary_key=True),
        Column("close", Numeric(20, 6), nullable=False),
        Column("source", String(40), nullable=False),
        Column("source_url", Text(), nullable=False),
        Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("close > 0", name="ck_prices_eod_close_positive"),
    )
    op.create_index("ix_prices_eod_as_of", "prices_eod", ["as_of"])
    op.create_index("ix_prices_eod_security_as_of", "prices_eod", ["security_id", "as_of"])

    op.create_table(
        "nav_daily",
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("as_of", Date(), primary_key=True),
        Column("nav", Numeric(20, 8), nullable=False),
        Column("source", String(40), nullable=False),
        Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("nav > 0", name="ck_nav_daily_nav_positive"),
    )
    op.create_index("ix_nav_daily_as_of", "nav_daily", ["as_of"])
    op.create_index("ix_nav_daily_security_as_of", "nav_daily", ["security_id", "as_of"])

    op.create_table(
        "fx_rates",
        Column("pair", String(7), primary_key=True),
        Column("as_of", Date(), primary_key=True),
        Column("rate", Numeric(20, 8), nullable=False),
        Column("source", String(40), nullable=False),
        Column("source_url", Text(), nullable=False),
        Column("publication_lag_days", SmallInteger()),
        Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("rate > 0", name="ck_fx_rates_rate_positive"),
    )
    op.create_index("ix_fx_rates_pair_as_of", "fx_rates", ["pair", "as_of"])

    op.create_table(
        "gold_prices",
        Column("as_of", Date(), primary_key=True),
        Column("inr_per_gram_24k", Numeric(20, 6), nullable=False),
        Column("source", String(40), nullable=False),
        Column("source_url", Text(), nullable=False),
        Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        CheckConstraint("inr_per_gram_24k > 0", name="ck_gold_prices_value_positive"),
    )
    op.create_index("ix_gold_prices_as_of", "gold_prices", ["as_of"])

    op.create_table(
        "marketdata_runs",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column("started_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        Column("finished_at", DateTime(timezone=True)),
        Column("status", String(16), nullable=False),
        Column("counts", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        Column("errors", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        CheckConstraint(
            "status IN ('succeeded', 'partial', 'failed')", name="ck_marketdata_runs_status"
        ),
    )
    op.create_index("ix_marketdata_runs_started_at", "marketdata_runs", ["started_at"])

    shared_tables = ("prices_eod", "nav_daily", "fx_rates", "gold_prices", "marketdata_runs")
    op.execute(f"GRANT SELECT ON TABLE {', '.join(shared_tables)} TO wealth_app")
    for table_name in shared_tables:
        op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_table("marketdata_runs")
    op.drop_table("gold_prices")
    op.drop_table("fx_rates")
    op.drop_table("nav_daily")
    op.drop_table("prices_eod")
    op.drop_index("ix_securities_nse_symbol", table_name="securities")
    op.drop_index("ix_securities_amfi_scheme_code", table_name="securities")
    op.drop_column("securities", "series")
    op.drop_column("securities", "nse_symbol")
    op.drop_column("securities", "amfi_scheme_code")
