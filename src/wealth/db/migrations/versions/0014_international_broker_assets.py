"""Add international broker securities, tax basis FX, and broker cash movements."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    ForeignKeyConstraint,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
    text,
)

revision = "0014_international_broker_assets"
down_revision = "0013_broker_imports"
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

    op.drop_constraint("uq_securities_market_symbol", "securities", type_="unique")
    op.alter_column("securities", "symbol", existing_type=String(32), nullable=True)
    op.add_column("securities", Column("exchange", String(32)))
    op.add_column("securities", Column("domicile", String(2)))
    op.execute(
        "UPDATE securities SET domicile = CASE "
        "WHEN isin IS NOT NULL THEN substr(isin, 1, 2) "
        "WHEN market = 'US' THEN 'US' ELSE NULL END"
    )
    op.drop_constraint("ck_securities_market", "securities", type_="check")
    op.create_check_constraint(
        "ck_securities_market", "securities", "market IN ('IN', 'US', 'DK', 'EU', 'GB')"
    )
    op.create_check_constraint(
        "ck_securities_currency",
        "securities",
        "currency IN ('INR', 'USD', 'EUR', 'DKK', 'GBP', 'SEK')",
    )
    op.create_check_constraint(
        "ck_securities_domicile",
        "securities",
        "domicile IS NULL OR domicile ~ '^[A-Z]{2}$'",
    )
    op.create_index(
        "uq_securities_market_symbol",
        "securities",
        ["market", "symbol"],
        unique=True,
        postgresql_where=text("symbol IS NOT NULL"),
    )

    op.add_column("accounts", Column("tax_wrapper", String(16)))
    op.create_check_constraint(
        "ck_accounts_tax_wrapper",
        "accounts",
        "tax_wrapper IS NULL OR tax_wrapper IN ('ask', 'ordinary', 'pension')",
    )

    for table in ("holdings_lots", "realised_trades"):
        op.add_column(
            table,
            Column("currency", String(3), nullable=False, server_default=text("'INR'")),
        )
        op.add_column(table, Column("fx_at_acquisition", Numeric(12, 6)))
        op.add_column(table, Column("cost_per_unit_inr", Numeric(16, 4)))
        op.add_column(
            table,
            Column("fx_missing", Boolean(), nullable=False, server_default=text("false")),
        )
    op.execute(
        "UPDATE holdings_lots SET cost_per_unit_inr = cost_per_unit WHERE cost_per_unit IS NOT NULL"
    )
    op.execute(
        "UPDATE realised_trades SET cost_per_unit_inr = cost_per_unit, "
        "fx_missing = false WHERE currency = 'INR'"
    )
    op.add_column("realised_trades", Column("fx_at_sale", Numeric(12, 6)))
    op.add_column("realised_trades", Column("sale_price_inr", Numeric(16, 4)))
    op.add_column("realised_trades", Column("gain_inr", Numeric(20, 2)))
    op.execute(
        "UPDATE realised_trades SET sale_price_inr = sale_price, gain_inr = gain "
        "WHERE currency = 'INR'"
    )

    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_upsert_security_by_symbol(
            p_market text,
            p_symbol text,
            p_exchange text,
            p_name text,
            p_type text,
            p_currency text,
            p_isin text,
            p_sleeve text,
            p_category text
        ) RETURNS uuid
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            result_id uuid;
            expanded text := '';
            digit_sum integer := 0;
            digit_value integer;
            character_value text;
            position integer;
            expanded_length integer;
            security_domicile text;
        BEGIN
            IF p_market IS NULL OR p_market NOT IN ('IN', 'US', 'DK', 'EU', 'GB') THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid market';
            END IF;
            IF p_symbol IS NULL OR length(p_symbol) NOT BETWEEN 1 AND 15
               OR p_symbol !~ '^[A-Z0-9.-]+$'
            THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid symbol';
            END IF;
            IF p_currency IS NULL OR p_currency NOT IN (
                'INR', 'USD', 'EUR', 'DKK', 'GBP', 'SEK'
            ) THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid currency';
            END IF;
            IF p_type IS NULL OR p_type NOT IN ('equity', 'etf', 'mf', 'bond', 'sgb') THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid security type';
            END IF;
            IF p_sleeve IS NOT NULL AND p_sleeve NOT IN
                ('equity_india', 'equity_global', 'debt', 'gold', 'cash', 'real_estate',
                 'protection')
            THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid sleeve';
            END IF;
            IF p_name IS NULL OR length(p_name) = 0 OR length(p_name) > 160
               OR (p_exchange IS NOT NULL AND length(p_exchange) > 32)
               OR (p_category IS NOT NULL AND length(p_category) > 80)
            THEN
                RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid security fields';
            END IF;

            IF p_isin IS NOT NULL THEN
                IF length(p_isin) <> 12
                   OR substr(p_isin, 1, 2) !~ '^[A-Z][A-Z]$'
                   OR substr(p_isin, 3, 9) !~ '^[A-Z0-9]+$'
                   OR substr(p_isin, 12, 1) !~ '^[0-9]$'
                THEN
                    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid ISIN';
                END IF;
                FOR position IN 1..11 LOOP
                    character_value := substr(p_isin, position, 1);
                    IF character_value ~ '^[A-Z]$' THEN
                        expanded := expanded || (ascii(character_value) - ascii('A') + 10)::text;
                    ELSE
                        expanded := expanded || character_value;
                    END IF;
                END LOOP;
                expanded := expanded || substr(p_isin, 12, 1);
                expanded_length := length(expanded);
                FOR position IN REVERSE expanded_length..1 LOOP
                    digit_value := substr(expanded, position, 1)::integer;
                    IF (expanded_length - position) % 2 = 1 THEN
                        digit_value := digit_value * 2;
                    END IF;
                    digit_sum := digit_sum + digit_value / 10 + digit_value % 10;
                END LOOP;
                IF digit_sum % 10 <> 0 THEN
                    RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'invalid ISIN';
                END IF;
            END IF;

            security_domicile := CASE
                WHEN p_isin IS NOT NULL THEN substr(p_isin, 1, 2)
                WHEN p_market = 'US' THEN 'US'
                ELSE NULL
            END;

            IF p_isin IS NOT NULL THEN
                SELECT id INTO result_id
                FROM {quoted_schema}.securities
                WHERE isin = p_isin
                FOR UPDATE;
                IF result_id IS NOT NULL THEN
                    UPDATE {quoted_schema}.securities
                    SET market = p_market,
                        symbol = p_symbol,
                        exchange = p_exchange,
                        domicile = security_domicile,
                        name = p_name,
                        security_type = p_type,
                        currency = p_currency,
                        sleeve = COALESCE(p_sleeve, sleeve),
                        category = COALESCE(p_category, category)
                    WHERE id = result_id;
                    RETURN result_id;
                END IF;
            END IF;

            INSERT INTO {quoted_schema}.securities AS target
                (id, market, symbol, isin, exchange, domicile, name, security_type, currency,
                 sleeve, category)
            VALUES
                (gen_random_uuid(), p_market, p_symbol, p_isin, p_exchange, security_domicile,
                 p_name, p_type, p_currency, p_sleeve, p_category)
            ON CONFLICT (market, symbol) WHERE symbol IS NOT NULL DO UPDATE SET
                isin = COALESCE(EXCLUDED.isin, target.isin),
                exchange = COALESCE(EXCLUDED.exchange, target.exchange),
                domicile = COALESCE(EXCLUDED.domicile, target.domicile),
                name = EXCLUDED.name,
                security_type = EXCLUDED.security_type,
                currency = EXCLUDED.currency,
                sleeve = COALESCE(EXCLUDED.sleeve, target.sleeve),
                category = COALESCE(EXCLUDED.category, target.category)
            RETURNING target.id INTO result_id;
            RETURN result_id;
        END;
        $function$
        """
    )
    signature = "text, text, text, text, text, text, text, text, text"
    op.execute(
        f"REVOKE ALL ON FUNCTION {quoted_schema}.app_upsert_security_by_symbol({signature}) "
        "FROM PUBLIC"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_upsert_security_by_symbol({signature}) "
        "TO wealth_app"
    )

    op.create_table(
        "broker_cash_flows",
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
        Column("transaction_date", Date(), nullable=False),
        Column("kind", String(16), nullable=False),
        Column("amount", Numeric(20, 4), nullable=False),
        Column("currency", String(3), nullable=False),
        Column("source_broker", String(32), nullable=False),
        Column("transaction_id", String(80)),
        UniqueConstraint("user_id", "id", name="uq_broker_cash_flows_user_id_id"),
        ForeignKeyConstraint(
            ["user_id", "member_id"],
            ["household_members.user_id", "household_members.id"],
            name="fk_broker_cash_flows_member_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "account_id"],
            ["accounts.user_id", "accounts.id"],
            name="fk_broker_cash_flows_account_tenant",
            ondelete="SET NULL (account_id)",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        ForeignKeyConstraint(
            ["user_id", "import_job_id"],
            ["import_jobs.user_id", "import_jobs.id"],
            name="fk_broker_cash_flows_import_job_tenant",
            ondelete="CASCADE",
            deferrable=True,
            initially="IMMEDIATE",
        ),
        CheckConstraint("kind IN ('deposit', 'withdrawal')", name="ck_broker_cash_flows_kind"),
        CheckConstraint("amount > 0", name="ck_broker_cash_flows_amount_positive"),
        CheckConstraint(
            "currency IN ('INR', 'USD', 'EUR', 'DKK', 'GBP', 'SEK')",
            name="ck_broker_cash_flows_currency",
        ),
    )
    op.create_index("ix_broker_cash_flows_user_id", "broker_cash_flows", ["user_id"])
    op.create_index(
        "ix_broker_cash_flows_household_member_date",
        "broker_cash_flows",
        ["user_id", "member_id", "transaction_date"],
    )
    _install_household_access(quoted_schema, "broker_cash_flows")

    op.execute(
        f"ALTER FUNCTION {quoted_schema}.app_merge_into_household(uuid) "
        "RENAME TO app_merge_into_household_before_0014"
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {quoted_schema}.app_merge_into_household_before_0014(uuid) "
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
        BEGIN
            RETURN {quoted_schema}.app_merge_into_household_before_0014(p_owner);
        END;
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) FROM PUBLIC")
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) TO wealth_app"
    )


def downgrade() -> None:
    _, quoted_schema = _schema()
    signature = "text, text, text, text, text, text, text, text, text"
    op.execute(f"DROP FUNCTION {quoted_schema}.app_upsert_security_by_symbol({signature})")

    op.execute(f"DROP FUNCTION {quoted_schema}.app_merge_into_household(uuid)")
    op.execute(
        f"ALTER FUNCTION {quoted_schema}.app_merge_into_household_before_0014(uuid) "
        "RENAME TO app_merge_into_household"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) TO wealth_app"
    )

    op.execute(
        f"DROP TRIGGER IF EXISTS trg_broker_cash_flows_household_audit ON "
        f"{quoted_schema}.broker_cash_flows"
    )
    op.execute(
        f"DROP POLICY IF EXISTS broker_cash_flows_household_access ON "
        f"{quoted_schema}.broker_cash_flows"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.broker_cash_flows DISABLE ROW LEVEL SECURITY")
    op.drop_table("broker_cash_flows")

    op.drop_column("realised_trades", "gain_inr")
    op.drop_column("realised_trades", "sale_price_inr")
    op.drop_column("realised_trades", "fx_at_sale")
    for table in ("holdings_lots", "realised_trades"):
        op.drop_column(table, "fx_missing")
        op.drop_column(table, "cost_per_unit_inr")
        op.drop_column(table, "fx_at_acquisition")
        op.drop_column(table, "currency")
    op.drop_constraint("ck_accounts_tax_wrapper", "accounts", type_="check")
    op.drop_column("accounts", "tax_wrapper")

    op.drop_index("uq_securities_market_symbol", table_name="securities")
    op.drop_constraint("ck_securities_domicile", "securities", type_="check")
    op.drop_constraint("ck_securities_currency", "securities", type_="check")
    op.drop_constraint("ck_securities_market", "securities", type_="check")
    op.create_check_constraint("ck_securities_market", "securities", "market IN ('IN', 'US')")
    op.alter_column("securities", "symbol", existing_type=String(32), nullable=False)
    op.create_unique_constraint("uq_securities_market_symbol", "securities", ["market", "symbol"])
    op.drop_column("securities", "domicile")
    op.drop_column("securities", "exchange")
