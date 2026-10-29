"""Add sector labels and NSE trading activity to shared market data."""

from __future__ import annotations

from alembic import op
from sqlalchemy import BigInteger, Column, Numeric, String, text

revision = "0021_screener_sentinel"
down_revision = "0020_statement_inbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "securities",
        Column(
            "sector",
            String(80),
            nullable=False,
            server_default=text("'Unclassified'"),
        ),
    )
    op.add_column(
        "securities",
        Column(
            "industry",
            String(120),
            nullable=False,
            server_default=text("'Unclassified'"),
        ),
    )
    op.add_column("prices_eod", Column("volume", BigInteger(), nullable=True))
    op.add_column("prices_eod", Column("traded_value", Numeric(24, 4), nullable=True))
    op.create_check_constraint(
        "ck_prices_eod_volume_non_negative", "prices_eod", "volume IS NULL OR volume >= 0"
    )
    op.create_check_constraint(
        "ck_prices_eod_traded_value_non_negative",
        "prices_eod",
        "traded_value IS NULL OR traded_value >= 0",
    )


def downgrade() -> None:
    op.drop_constraint("ck_prices_eod_traded_value_non_negative", "prices_eod", type_="check")
    op.drop_constraint("ck_prices_eod_volume_non_negative", "prices_eod", type_="check")
    op.drop_column("prices_eod", "traded_value")
    op.drop_column("prices_eod", "volume")
    op.drop_column("securities", "industry")
    op.drop_column("securities", "sector")
