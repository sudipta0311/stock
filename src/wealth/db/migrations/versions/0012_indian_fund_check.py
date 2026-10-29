"""Add shared Indian-fund reference data and household check history."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
    text,
)

revision = "0012_indian_fund_check"
down_revision = "0011_shared_household"
branch_labels = None
depends_on = None


def _qualified_schema() -> tuple[str, str]:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    return schema, connection.dialect.identifier_preparer.quote_schema(schema)


def upgrade() -> None:
    _schema, quoted_schema = _qualified_schema()
    op.add_column("securities", Column("sub_category", String(80)))
    op.add_column("securities", Column("plan_type", String(12)))
    op.add_column("securities", Column("option", String(12)))
    op.add_column("securities", Column("ter", Numeric(8, 4)))
    op.add_column("securities", Column("ter_as_of", Date()))
    op.add_column("securities", Column("ter_source", String(80)))
    op.create_check_constraint(
        "ck_securities_plan_type",
        "securities",
        "plan_type IS NULL OR plan_type IN ('direct', 'regular')",
    )
    op.create_check_constraint(
        "ck_securities_option",
        "securities",
        "option IS NULL OR option IN ('growth', 'idcw')",
    )
    op.create_check_constraint(
        "ck_securities_ter_non_negative", "securities", "ter IS NULL OR ter >= 0"
    )

    op.create_table(
        "nav_monthly",
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        Column("month_end", Date(), primary_key=True),
        Column("nav", Numeric(20, 8), nullable=False),
        Column("source", String(40), nullable=False),
        CheckConstraint("nav > 0", name="ck_nav_monthly_nav_positive"),
    )
    op.create_index("ix_nav_monthly_month_end", "nav_monthly", ["month_end"])
    op.execute(f"GRANT SELECT ON TABLE {quoted_schema}.nav_monthly TO wealth_app")
    op.execute(f"ALTER TABLE {quoted_schema}.nav_monthly DISABLE ROW LEVEL SECURITY")

    op.create_table(
        "fund_lagging_checks",
        Column(
            "id", Uuid(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
        ),
        Column(
            "user_id",
            Uuid(as_uuid=True),
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("member_id", Uuid(as_uuid=True), nullable=False),
        Column("security_id", Uuid(as_uuid=True), nullable=False),
        Column("month_end", Date(), nullable=False),
        Column("bottom_quartile", Boolean(), nullable=False),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_fund_lagging_checks_member_tenant",
            ondelete="CASCADE",
            # app_merge_into_household moves rows before their members (SET CONSTRAINTS
            # ALL DEFERRED), like the tenant keys 0011 made deferrable.
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["security_id"],
            ["securities.id"],
            name="fk_fund_lagging_checks_security",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "month_end = (date_trunc('month', month_end) + interval '1 month - 1 day')::date",
            name="ck_fund_lagging_checks_month_end",
        ),
        Index(
            "ix_fund_lagging_checks_household_member_security_month",
            "user_id",
            "member_id",
            "security_id",
            "month_end",
        ),
        UniqueConstraint("user_id", "id", name="uq_fund_lagging_checks_user_id_id"),
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "
        f"{quoted_schema}.fund_lagging_checks TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.fund_lagging_checks ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY fund_lagging_checks_household_access
        ON {quoted_schema}.fund_lagging_checks FOR ALL TO wealth_app
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
        CREATE TRIGGER trg_fund_lagging_checks_household_audit
        AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.fund_lagging_checks
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_household_write()
        """
    )

    op.create_table(
        "fund_transactions",
        Column(
            "id", Uuid(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
        ),
        Column(
            "user_id",
            Uuid(as_uuid=True),
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("member_id", Uuid(as_uuid=True), nullable=False),
        Column("security_id", Uuid(as_uuid=True), nullable=False),
        Column("transaction_date", Date(), nullable=False),
        Column("kind", String(24), nullable=False),
        Column("amount", Numeric(16, 2), nullable=False),
        Column("units", Numeric(20, 6), nullable=False),
        Column("account_ref", String(16)),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_fund_transactions_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["security_id"],
            ["securities.id"],
            name="fk_fund_transactions_security",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "kind IN ('buy', 'sell', 'switch_in', 'switch_out', 'dividend_reinvest', 'other')",
            name="ck_fund_transactions_kind",
        ),
        CheckConstraint("amount >= 0 AND units >= 0", name="ck_fund_transactions_non_negative"),
        CheckConstraint(
            "account_ref IS NULL OR account_ref ~ '^XXXX[A-Z0-9]{4}$'",
            name="ck_fund_transactions_account_ref_masked",
        ),
        UniqueConstraint("user_id", "id", name="uq_fund_transactions_user_id_id"),
        Index(
            "ix_fund_transactions_household_member_security_date",
            "user_id",
            "member_id",
            "security_id",
            "transaction_date",
        ),
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "
        f"{quoted_schema}.fund_transactions TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.fund_transactions ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY fund_transactions_household_access
        ON {quoted_schema}.fund_transactions FOR ALL TO wealth_app
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
        CREATE TRIGGER trg_fund_transactions_household_audit
        AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.fund_transactions
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_household_write()
        """
    )


def downgrade() -> None:
    _schema, quoted_schema = _qualified_schema()
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_fund_lagging_checks_household_audit "
        f"ON {quoted_schema}.fund_lagging_checks"
    )
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_fund_transactions_household_audit "
        f"ON {quoted_schema}.fund_transactions"
    )
    op.drop_table("fund_transactions")
    op.drop_table("fund_lagging_checks")
    op.drop_table("nav_monthly")
    op.drop_constraint("ck_securities_ter_non_negative", "securities", type_="check")
    op.drop_constraint("ck_securities_option", "securities", type_="check")
    op.drop_constraint("ck_securities_plan_type", "securities", type_="check")
    op.drop_column("securities", "ter_source")
    op.drop_column("securities", "ter_as_of")
    op.drop_column("securities", "ter")
    op.drop_column("securities", "option")
    op.drop_column("securities", "plan_type")
    op.drop_column("securities", "sub_category")
