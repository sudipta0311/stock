"""Add privacy-safe CAS import staging and shared security upserts."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, Date, DateTime, Numeric, String, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005_imports"
down_revision = "0004_money_map"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema_name = connection.execute(text("SELECT current_schema()")).scalar_one()
    schema = connection.dialect.identifier_preparer.quote_schema(schema_name)

    op.add_column("import_jobs", Column("error_code", String(40)))
    op.add_column("import_jobs", Column("statement_date", Date()))
    op.add_column("import_jobs", Column("source", String(16)))
    op.add_column(
        "import_jobs",
        Column("counts", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    )
    op.add_column("import_jobs", Column("confirmed_at", DateTime(timezone=True)))
    op.execute("UPDATE import_jobs SET file_deleted_at = now() WHERE file_deleted_at IS NULL")
    op.alter_column("import_jobs", "file_deleted_at", nullable=False, server_default=text("now()"))
    op.create_check_constraint(
        "ck_import_jobs_status",
        "import_jobs",
        "status IN ('ready', 'confirmed', 'failed', 'discarded')",
    )
    op.create_check_constraint(
        "ck_import_jobs_source",
        "import_jobs",
        "source IS NULL OR source IN ('nsdl', 'cdsl', 'cams_kfin')",
    )

    op.add_column("import_items", Column("match_lot_id", Uuid(as_uuid=True)))
    op.add_column("import_items", Column("match_security_id", Uuid(as_uuid=True)))
    op.add_column("import_items", Column("diff", JSONB))
    op.add_column("import_items", Column("user_override", JSONB))
    op.create_foreign_key(
        "fk_import_items_match_lot_tenant",
        "import_items",
        "holdings_lots",
        ["user_id", "match_lot_id"],
        ["user_id", "id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_import_items_match_security",
        "import_items",
        "securities",
        ["match_security_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.add_column("holdings_lots", Column("account_ref", String(16)))
    op.add_column("holdings_lots", Column("statement_value", Numeric(16, 2)))
    op.add_column("holdings_lots", Column("statement_date", Date()))
    op.add_column("holdings_lots", Column("import_job_id", Uuid(as_uuid=True)))
    op.create_check_constraint(
        "ck_holdings_lots_account_ref_masked",
        "holdings_lots",
        "account_ref IS NULL OR account_ref ~ '^XXXX[A-Z0-9]{4}$'",
    )
    op.create_foreign_key(
        "fk_holdings_lots_import_job_tenant",
        "holdings_lots",
        "import_jobs",
        ["user_id", "import_job_id"],
        ["user_id", "id"],
        ondelete="CASCADE",
    )

    op.add_column("securities", Column("sleeve", String(32)))
    op.add_column("securities", Column("category", String(80)))
    op.create_check_constraint(
        "ck_securities_sleeve",
        "securities",
        "sleeve IS NULL OR sleeve IN ('equity_india', 'equity_global', 'debt', 'gold', "
        "'cash', 'real_estate', 'protection')",
    )

    op.execute(
        f"""
        CREATE FUNCTION {schema}.app_upsert_security(
            p_market text,
            p_isin text,
            p_symbol text,
            p_name text,
            p_type text,
            p_currency text,
            p_sleeve text,
            p_category text
        ) RETURNS uuid
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {schema}
        AS $app_upsert_security$
        DECLARE
            result_id uuid;
        BEGIN
            IF p_isin IS NULL OR p_isin !~ '^[A-Z]{{2}}[A-Z0-9]{{9}}[0-9]$' THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid ISIN';
            END IF;
            IF p_market IS NULL OR p_market NOT IN ('IN', 'US') THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid market';
            END IF;
            IF p_type IS NULL OR p_type NOT IN ('equity', 'etf', 'mf', 'bond', 'sgb') THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid security type';
            END IF;
            IF p_currency IS NULL OR p_currency NOT IN ('INR', 'USD') THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid currency';
            END IF;
            IF p_sleeve IS NOT NULL AND p_sleeve NOT IN
                ('equity_india', 'equity_global', 'debt', 'gold', 'cash', 'real_estate',
                 'protection')
            THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid sleeve';
            END IF;
            IF p_symbol IS NULL OR length(p_symbol) = 0 OR length(p_symbol) > 32
               OR p_name IS NULL OR length(p_name) = 0 OR length(p_name) > 160
               OR (p_category IS NOT NULL AND length(p_category) > 80)
            THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid security fields';
            END IF;

            INSERT INTO {schema}.securities AS target
                (id, market, symbol, isin, name, security_type, currency, sleeve, category)
            VALUES
                (pg_catalog.gen_random_uuid(), p_market, p_symbol, p_isin, p_name, p_type,
                 p_currency, p_sleeve, p_category)
            ON CONFLICT (isin) DO UPDATE SET
                market = EXCLUDED.market,
                symbol = EXCLUDED.symbol,
                name = EXCLUDED.name,
                security_type = EXCLUDED.security_type,
                currency = EXCLUDED.currency,
                sleeve = COALESCE(EXCLUDED.sleeve, target.sleeve),
                category = COALESCE(EXCLUDED.category, target.category)
            RETURNING target.id INTO result_id;
            RETURN result_id;
        END;
        $app_upsert_security$
        """
    )
    signature = "text, text, text, text, text, text, text, text"
    op.execute(f"REVOKE ALL ON FUNCTION {schema}.app_upsert_security({signature}) FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {schema}.app_upsert_security({signature}) TO wealth_app")


def downgrade() -> None:
    connection = op.get_bind()
    schema_name = connection.execute(text("SELECT current_schema()")).scalar_one()
    schema = connection.dialect.identifier_preparer.quote_schema(schema_name)
    signature = "text, text, text, text, text, text, text, text"
    op.execute(f"DROP FUNCTION {schema}.app_upsert_security({signature})")
    op.drop_constraint("ck_securities_sleeve", "securities", type_="check")
    op.drop_column("securities", "category")
    op.drop_column("securities", "sleeve")
    op.drop_constraint("fk_holdings_lots_import_job_tenant", "holdings_lots", type_="foreignkey")
    op.drop_constraint("ck_holdings_lots_account_ref_masked", "holdings_lots", type_="check")
    op.drop_column("holdings_lots", "import_job_id")
    op.drop_column("holdings_lots", "statement_date")
    op.drop_column("holdings_lots", "statement_value")
    op.drop_column("holdings_lots", "account_ref")
    op.drop_constraint("fk_import_items_match_security", "import_items", type_="foreignkey")
    op.drop_constraint("fk_import_items_match_lot_tenant", "import_items", type_="foreignkey")
    op.drop_column("import_items", "user_override")
    op.drop_column("import_items", "diff")
    op.drop_column("import_items", "match_security_id")
    op.drop_column("import_items", "match_lot_id")
    op.drop_constraint("ck_import_jobs_source", "import_jobs", type_="check")
    op.drop_constraint("ck_import_jobs_status", "import_jobs", type_="check")
    op.drop_column("import_jobs", "confirmed_at")
    op.drop_column("import_jobs", "counts")
    op.drop_column("import_jobs", "source")
    op.drop_column("import_jobs", "statement_date")
    op.drop_column("import_jobs", "error_code")
    op.alter_column("import_jobs", "file_deleted_at", nullable=True, server_default=None)
