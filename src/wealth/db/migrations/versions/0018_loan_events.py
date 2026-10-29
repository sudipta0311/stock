"""Add tenant-scoped loan events and household access controls."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
    text,
)

revision = "0018_loan_events"
down_revision = "0017_fundamentals_pit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    preparer = connection.dialect.identifier_preparer
    quoted_schema = preparer.quote_schema(schema)
    op.create_table(
        "loan_events",
        Column("id", Uuid(as_uuid=True), primary_key=True),
        Column(
            "user_id",
            Uuid(as_uuid=True),
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("member_id", Uuid(as_uuid=True), nullable=False),
        Column("loan_id", Uuid(as_uuid=True), nullable=False),
        Column("kind", String(24), nullable=False),
        Column("on_date", Date(), nullable=False),
        Column("amount", Numeric(16, 2)),
        Column("rate_pct", Numeric(7, 4)),
        Column("emi", Numeric(16, 2)),
        Column("note", String(140)),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_loan_events_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "loan_id"],
            ["manual_assets.user_id", "manual_assets.id"],
            name="fk_loan_events_loan_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        CheckConstraint(
            "kind IN ('prepayment', 'rate_change', 'emi_change', 'balance_update', "
            "'missed_emi', 'closed')",
            name="ck_loan_events_kind",
        ),
        CheckConstraint("amount IS NULL OR amount >= 0", name="ck_loan_events_amount_nonnegative"),
        CheckConstraint(
            "rate_pct IS NULL OR rate_pct BETWEEN 0 AND 60",
            name="ck_loan_events_rate_range",
        ),
        CheckConstraint("emi IS NULL OR emi >= 0", name="ck_loan_events_emi_nonnegative"),
        UniqueConstraint("user_id", "id", name="uq_loan_events_user_id_id"),
    )
    op.create_index(
        "ix_loan_events_tenant_loan_date", "loan_events", ["user_id", "loan_id", "on_date"]
    )
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {quoted_schema}.loan_events TO wealth_app")
    op.execute(f"ALTER TABLE {quoted_schema}.loan_events ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY loan_events_household_access ON {quoted_schema}.loan_events "
        f"FOR ALL TO wealth_app "
        f"USING (user_id = nullif(current_setting('app.household_owner', true), '')::uuid "
        f"AND {quoted_schema}.app_household_role(user_id) IS NOT NULL) "
        f"WITH CHECK (user_id = nullif(current_setting('app.household_owner', true), '')::uuid "
        f"AND {quoted_schema}.app_household_role(user_id) IN ('owner', 'partner'))"
    )
    op.execute(
        f"CREATE TRIGGER trg_loan_events_household_audit "
        f"AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.loan_events "
        f"FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_household_write()"
    )


def downgrade() -> None:
    schema = op.get_bind().execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = op.get_bind().dialect.identifier_preparer.quote_schema(schema)
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_loan_events_household_audit ON {quoted_schema}.loan_events"
    )
    op.execute("DROP POLICY IF EXISTS loan_events_household_access ON loan_events")
    op.drop_index("ix_loan_events_tenant_loan_date", table_name="loan_events")
    op.drop_table("loan_events")
