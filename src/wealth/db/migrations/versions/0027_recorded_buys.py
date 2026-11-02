"""Track buys entered from weekly actions until a statement includes them."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0027_recorded_buys"
down_revision = "0026_audit_log_hardening"
branch_labels = None
depends_on = None


def _schema() -> tuple[str, str]:
    connection = op.get_bind()
    schema = connection.execute(sa.text("SELECT current_schema()")).scalar_one()
    return schema, connection.dialect.identifier_preparer.quote_schema(schema)


def upgrade() -> None:
    schema, quoted_schema = _schema()
    op.create_table(
        "recorded_buys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(f"{schema}.users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_id", postgresql.UUID(as_uuid=True)),
        sa.Column("account_id", postgresql.UUID(as_uuid=True)),
        sa.Column("line", sa.SmallInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("bought_on", sa.Date(), nullable=False),
        sa.Column("amount", sa.Numeric(16, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("amount_inr", sa.Numeric(16, 2)),
        sa.Column("units", sa.Numeric(20, 6)),
        sa.Column("units_entered", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("cost_per_unit", sa.Numeric(20, 8)),
        sa.Column("cost_per_unit_inr", sa.Numeric(20, 8)),
        sa.Column("security_id", postgresql.UUID(as_uuid=True)),
        sa.Column("security_type", sa.String(16), nullable=False, server_default="mf"),
        sa.Column("security_symbol", sa.String(32)),
        sa.Column("security_isin", sa.String(24)),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("sleeve", sa.String(32), nullable=False),
        sa.Column("instrument_category", sa.String(80), nullable=False),
        sa.Column("account_type", sa.String(16)),
        sa.Column("status", sa.String(16), nullable=False, server_default="waiting"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("user_id", "id", name="uq_recorded_buys_user_id_id"),
        sa.ForeignKeyConstraint(
            ["user_id", "member_id"],
            [f"{schema}.household_members.user_id", f"{schema}.household_members.id"],
            name="fk_recorded_buys_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "action_id"],
            [f"{schema}.actions.user_id", f"{schema}.actions.id"],
            name="fk_recorded_buys_action_tenant",
            ondelete="SET NULL (action_id)",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id", "account_id"],
            [f"{schema}.accounts.user_id", f"{schema}.accounts.id"],
            name="fk_recorded_buys_account_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        sa.ForeignKeyConstraint(
            ["security_id"],
            [f"{schema}.securities.id"],
            name="fk_recorded_buys_security",
            ondelete="SET NULL (security_id)",
        ),
        sa.CheckConstraint("line >= 0", name="ck_recorded_buys_line_nonnegative"),
        sa.CheckConstraint("amount > 0", name="ck_recorded_buys_amount_positive"),
        sa.CheckConstraint(
            "amount_inr IS NULL OR amount_inr > 0", name="ck_recorded_buys_amount_inr_positive"
        ),
        sa.CheckConstraint("units IS NULL OR units > 0", name="ck_recorded_buys_units_positive"),
        sa.CheckConstraint(
            "status IN ('waiting', 'matched', 'removed')", name="ck_recorded_buys_status"
        ),
        sa.CheckConstraint(
            "account_type IS NULL OR account_type IN ('ask', 'ordinary')",
            name="ck_recorded_buys_account_type",
        ),
        schema=schema,
    )
    op.create_index(
        "ix_recorded_buys_household_member_status_date",
        "recorded_buys",
        ["user_id", "member_id", "status", "bought_on"],
        schema=schema,
    )
    op.create_index(
        "ix_recorded_buys_household_action_status",
        "recorded_buys",
        ["user_id", "action_id", "status"],
        schema=schema,
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {quoted_schema}.recorded_buys TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.recorded_buys ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {quoted_schema}.recorded_buys FORCE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY recorded_buys_household_access
        ON {quoted_schema}.recorded_buys FOR ALL TO wealth_app
        USING (
            user_id = nullif(current_setting('app.household_owner', true), '')::uuid
            AND {quoted_schema}.app_household_role(user_id) IS NOT NULL
        )
        WITH CHECK (
            user_id = nullif(current_setting('app.household_owner', true), '')::uuid
            AND {quoted_schema}.app_household_role(user_id) IN ('owner', 'partner')
        )
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER trg_recorded_buys_household_audit
        AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.recorded_buys
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_household_write()
        """
    )


def downgrade() -> None:
    schema, quoted_schema = _schema()
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_recorded_buys_household_audit ON {quoted_schema}.recorded_buys"
    )
    op.execute(
        f"DROP POLICY IF EXISTS recorded_buys_household_access ON {quoted_schema}.recorded_buys"
    )
    op.drop_table("recorded_buys", schema=schema)
