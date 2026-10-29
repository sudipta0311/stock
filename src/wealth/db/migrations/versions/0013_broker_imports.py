"""Add broker imports, household templates, realised equity trades, and lot lineage."""

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
from sqlalchemy.dialects.postgresql import JSONB

revision = "0013_broker_imports"
down_revision = "0012_indian_fund_check"
branch_labels = None
depends_on = None


def _schema() -> tuple[str, str]:
    connection = op.get_bind()
    name = connection.execute(text("SELECT current_schema()")).scalar_one()
    return name, connection.dialect.identifier_preparer.quote_schema(name)


def _install_household_access(quoted_schema: str, table: str) -> None:
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {quoted_schema}.{table} TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.{table} ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY {table}_household_access ON {quoted_schema}.{table} FOR ALL TO wealth_app
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
        CREATE TRIGGER trg_{table}_household_audit
        AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.{table}
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_household_write()
        """
    )


def upgrade() -> None:
    _, quoted_schema = _schema()
    op.add_column(
        "holdings_lots",
        Column("trade_ids", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    )

    op.add_column("import_jobs", Column("account_id", Uuid(as_uuid=True)))
    op.add_column("import_jobs", Column("member_id", Uuid(as_uuid=True)))
    op.drop_constraint("ck_import_jobs_status", "import_jobs", type_="check")
    op.create_check_constraint(
        "ck_import_jobs_status",
        "import_jobs",
        "status IN ('ready', 'needs_mapping', 'confirmed', 'failed', 'discarded')",
    )
    op.drop_constraint("ck_import_jobs_source", "import_jobs", type_="check")
    op.create_check_constraint(
        "ck_import_jobs_source",
        "import_jobs",
        "source IS NULL OR source IN ('nsdl', 'cdsl', 'cams_kfin', 'broker')",
    )
    op.create_foreign_key(
        "fk_import_jobs_member_tenant",
        "import_jobs",
        "household_members",
        ["user_id", "member_id"],
        ["user_id", "id"],
        ondelete="CASCADE",
        deferrable=True,
        initially="IMMEDIATE",
    )
    op.create_foreign_key(
        "fk_import_jobs_account_tenant",
        "import_jobs",
        "accounts",
        ["user_id", "account_id"],
        ["user_id", "id"],
        ondelete="SET NULL (account_id)",
        deferrable=True,
        initially="IMMEDIATE",
    )

    op.alter_column(
        "import_items",
        "status",
        existing_type=String(16),
        type_=String(32),
        existing_nullable=False,
    )
    op.drop_constraint("ck_import_items_status", "import_items", type_="check")
    op.create_check_constraint(
        "ck_import_items_status",
        "import_items",
        "status IN ('new', 'changed', 'duplicate', 'unknown', 'accepted', 'rejected', "
        "'enrich', 'replace_with_lots', 'quantity_mismatch')",
    )

    op.create_table(
        "import_templates",
        Column(
            "id", Uuid(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
        ),
        Column(
            "user_id",
            Uuid(as_uuid=True),
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("name", String(80), nullable=False),
        Column("kind", String(16), nullable=False),
        Column("header_fingerprint", String(64), nullable=False),
        Column("header_names", JSONB, nullable=False),
        Column("columns", JSONB, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
        UniqueConstraint("user_id", "id", name="uq_import_templates_user_id_id"),
        UniqueConstraint(
            "user_id", "kind", "header_fingerprint", name="uq_import_templates_household_header"
        ),
        CheckConstraint("kind IN ('holdings', 'tradebook')", name="ck_import_templates_kind"),
        CheckConstraint(
            "header_fingerprint ~ '^[0-9a-f]{64}$'", name="ck_import_templates_fingerprint"
        ),
    )
    op.create_index("ix_import_templates_user_id", "import_templates", ["user_id"])
    _install_household_access(quoted_schema, "import_templates")

    op.create_table(
        "realised_trades",
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
        Column("account_id", Uuid(as_uuid=True)),
        Column("import_job_id", Uuid(as_uuid=True), nullable=False),
        Column(
            "security_id",
            Uuid(as_uuid=True),
            ForeignKey("securities.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        Column("trade_date", Date(), nullable=False),
        Column("acquired_on", Date(), nullable=False),
        Column("quantity", Numeric(20, 6), nullable=False),
        Column("cost_per_unit", Numeric(16, 4), nullable=False),
        Column("sale_price", Numeric(16, 4), nullable=False),
        Column("gain", Numeric(20, 2), nullable=False),
        Column("holding_period", String(2), nullable=False),
        Column("trade_id", String(80)),
        Column("source_broker", String(32), nullable=False),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_realised_trades_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "account_id"],
            ["accounts.user_id", "accounts.id"],
            name="fk_realised_trades_account_tenant",
            ondelete="SET NULL (account_id)",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "import_job_id"],
            ["import_jobs.user_id", "import_jobs.id"],
            name="fk_realised_trades_import_job_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        UniqueConstraint("user_id", "id", name="uq_realised_trades_user_id_id"),
        CheckConstraint("quantity > 0", name="ck_realised_trades_quantity_positive"),
        CheckConstraint("holding_period IN ('LT', 'ST')", name="ck_realised_trades_holding_period"),
    )
    op.create_index("ix_realised_trades_user_id", "realised_trades", ["user_id"])
    op.create_index(
        "ix_realised_trades_household_member_date_period",
        "realised_trades",
        ["user_id", "member_id", "trade_date", "holding_period"],
    )
    _install_household_access(quoted_schema, "realised_trades")

    # The generic 0011 merge moves all rows by changing user_id. Discard matching imported
    # templates first so the target household's saved choice remains authoritative.
    op.execute(
        f"ALTER FUNCTION {quoted_schema}.app_merge_into_household(uuid) "
        "RENAME TO app_merge_into_household_before_0013"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {quoted_schema}.app_merge_into_household_before_0013(uuid) "
        "FROM PUBLIC, wealth_app"
    )
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_merge_into_household(p_owner uuid)
        RETURNS jsonb
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            active_user uuid;
        BEGIN
            active_user := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF active_user IS NOT NULL
               AND {quoted_schema}.app_household_role(p_owner) = 'partner' THEN
                DELETE FROM {quoted_schema}.import_templates AS incoming
                USING {quoted_schema}.import_templates AS saved
                WHERE incoming.user_id = active_user
                  AND saved.user_id = p_owner
                  AND incoming.kind = saved.kind
                  AND incoming.header_fingerprint = saved.header_fingerprint;
            END IF;
            RETURN {quoted_schema}.app_merge_into_household_before_0013(p_owner);
        END;
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) FROM PUBLIC")
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) TO wealth_app"
    )


def downgrade() -> None:
    _schema_name, quoted_schema = _schema()
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_merge_into_household(uuid)")
    op.execute(
        f"ALTER FUNCTION {quoted_schema}.app_merge_into_household_before_0013(uuid) "
        "RENAME TO app_merge_into_household"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) TO wealth_app"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_realised_trades_household_audit ON "
        f"{quoted_schema}.realised_trades"
    )
    op.execute(
        f"DROP POLICY IF EXISTS realised_trades_household_access ON {quoted_schema}.realised_trades"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.realised_trades DISABLE ROW LEVEL SECURITY")
    op.drop_table("realised_trades")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_import_templates_household_audit ON "
        f"{quoted_schema}.import_templates"
    )
    op.execute(
        "DROP POLICY IF EXISTS import_templates_household_access ON "
        f"{quoted_schema}.import_templates"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.import_templates DISABLE ROW LEVEL SECURITY")
    op.drop_table("import_templates")

    op.drop_constraint("ck_import_items_status", "import_items", type_="check")
    op.create_check_constraint(
        "ck_import_items_status",
        "import_items",
        "status IN ('new', 'changed', 'duplicate', 'unknown', 'accepted', 'rejected')",
    )
    op.alter_column(
        "import_items",
        "status",
        existing_type=String(32),
        type_=String(16),
        existing_nullable=False,
    )
    op.drop_constraint("fk_import_jobs_account_tenant", "import_jobs", type_="foreignkey")
    op.drop_constraint("fk_import_jobs_member_tenant", "import_jobs", type_="foreignkey")
    op.drop_constraint("ck_import_jobs_source", "import_jobs", type_="check")
    op.create_check_constraint(
        "ck_import_jobs_source",
        "import_jobs",
        "source IS NULL OR source IN ('nsdl', 'cdsl', 'cams_kfin')",
    )
    op.drop_constraint("ck_import_jobs_status", "import_jobs", type_="check")
    op.create_check_constraint(
        "ck_import_jobs_status",
        "import_jobs",
        "status IN ('ready', 'confirmed', 'failed', 'discarded')",
    )
    op.drop_column("import_jobs", "member_id")
    op.drop_column("import_jobs", "account_id")
    op.drop_column("holdings_lots", "trade_ids")
