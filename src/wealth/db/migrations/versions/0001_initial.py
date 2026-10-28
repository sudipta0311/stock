"""Initial multi-tenant PostgreSQL schema with application RLS policies."""

from __future__ import annotations

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

USER_TABLES = (
    "users",
    "profiles",
    "household_members",
    "goals",
    "accounts",
    "manual_assets",
    "asset_history",
    "asset_cashflows",
    "holdings_lots",
    "import_jobs",
    "import_items",
    "networth_snapshots",
    "actions",
    "exit_cards",
    "reminders",
    "push_subscriptions",
    "consents",
    "audit_log",
)

RLS_POLICIES = (
    "users_isolation",
    "profiles_isolation",
    "household_members_isolation",
    "goals_isolation",
    "accounts_isolation",
    "manual_assets_isolation",
    "asset_history_isolation",
    "asset_cashflows_isolation",
    "holdings_lots_isolation",
    "import_jobs_isolation",
    "import_items_isolation",
    "networth_snapshots_isolation",
    "actions_isolation",
    "exit_cards_isolation",
    "reminders_isolation",
    "push_subscriptions_isolation",
    "consents_isolation",
    "audit_log_isolation",
)


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'wealth_app') THEN
                CREATE ROLE wealth_app NOLOGIN;
            END IF;
        END;
        $$;
        """
    )

    op.execute(
        """
        CREATE TABLE users (
            id uuid PRIMARY KEY,
            email varchar(320) NOT NULL,
            display_name varchar(120),
            auth_subject varchar(255) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            deleted_at timestamptz,
            CONSTRAINT ck_users_email_lowercase CHECK (email = lower(email)),
            CONSTRAINT uq_users_auth_subject UNIQUE (auth_subject)
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX uq_users_email_lower ON users (lower(email))")
    op.execute(
        """
        CREATE TABLE profiles (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            tax_regime varchar(16),
            tax_slab_pct numeric(7, 4),
            city_tier smallint,
            timezone varchar(64) NOT NULL DEFAULT 'Asia/Kolkata',
            language varchar(16) NOT NULL DEFAULT 'en',
            risk_answers jsonb,
            risk_profile varchar(32),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_profiles_user_id_id UNIQUE (user_id, id),
            CONSTRAINT uq_profiles_user_id UNIQUE (user_id),
            CONSTRAINT ck_profiles_city_tier CHECK (city_tier IS NULL OR city_tier BETWEEN 1 AND 3)
        )
        """
    )
    op.execute("CREATE INDEX ix_profiles_user_id ON profiles (user_id)")
    op.execute(
        """
        CREATE TABLE household_members (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            display_name varchar(40) NOT NULL,
            relationship varchar(40) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_household_members_user_id_id UNIQUE (user_id, id)
        )
        """
    )
    op.execute("CREATE INDEX ix_household_members_user_id ON household_members (user_id)")
    op.execute(
        """
        CREATE TABLE goals (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name varchar(120) NOT NULL,
            goal_type varchar(32) NOT NULL,
            target_amount numeric(16, 2) NOT NULL,
            target_year integer NOT NULL,
            currency varchar(3) NOT NULL DEFAULT 'INR',
            priority smallint NOT NULL DEFAULT 3,
            CONSTRAINT uq_goals_user_id_id UNIQUE (user_id, id),
            CONSTRAINT ck_goals_priority CHECK (priority BETWEEN 1 AND 5),
            CONSTRAINT ck_goals_currency CHECK (currency IN ('INR', 'USD'))
        )
        """
    )
    op.execute("CREATE INDEX ix_goals_user_id ON goals (user_id)")
    op.execute(
        """
        CREATE TABLE securities (
            id uuid PRIMARY KEY,
            market varchar(2) NOT NULL,
            symbol varchar(32) NOT NULL,
            isin varchar(12),
            name varchar(160) NOT NULL,
            security_type varchar(16) NOT NULL,
            currency varchar(3) NOT NULL,
            CONSTRAINT uq_securities_market_symbol UNIQUE (market, symbol),
            CONSTRAINT uq_securities_isin UNIQUE (isin),
            CONSTRAINT ck_securities_market CHECK (market IN ('IN', 'US')),
            CONSTRAINT ck_securities_security_type
                CHECK (security_type IN ('equity', 'etf', 'mf', 'bond', 'sgb'))
        )
        """
    )
    op.execute(
        """
        CREATE TABLE accounts (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            member_id uuid,
            institution varchar(120) NOT NULL,
            account_type varchar(32) NOT NULL,
            masked_number varchar(32),
            CONSTRAINT uq_accounts_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_accounts_member_tenant FOREIGN KEY (user_id, member_id)
                REFERENCES household_members(user_id, id) ON DELETE CASCADE,
            CONSTRAINT ck_accounts_masked_number
                CHECK (masked_number IS NULL OR masked_number ~ '^X*[0-9A-Za-z]{0,4}$')
        )
        """
    )
    op.execute("CREATE INDEX ix_accounts_user_id ON accounts (user_id)")
    op.execute(
        """
        CREATE TABLE manual_assets (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            member_id uuid,
            account_id uuid,
            asset_type varchar(32) NOT NULL,
            details jsonb NOT NULL,
            value_cached numeric(16, 2),
            value_as_of date,
            status varchar(16) NOT NULL DEFAULT 'active',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_manual_assets_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_manual_assets_member_tenant FOREIGN KEY (user_id, member_id)
                REFERENCES household_members(user_id, id) ON DELETE CASCADE,
            CONSTRAINT fk_manual_assets_account_tenant FOREIGN KEY (user_id, account_id)
                REFERENCES accounts(user_id, id) ON DELETE CASCADE,
            CONSTRAINT ck_manual_assets_asset_type CHECK (
                asset_type IN ('savings', 'fd', 'rd', 'ppf', 'epf', 'nps', 'po_scheme', 'bond',
                    'gold', 'term_insurance', 'health_insurance', 'traditional_policy', 'ulip',
                    'property', 'loan')
            ),
            CONSTRAINT ck_manual_assets_status CHECK (status IN ('active', 'closed'))
        )
        """
    )
    op.execute("CREATE INDEX ix_manual_assets_user_id ON manual_assets (user_id)")
    op.execute(
        """
        CREATE TABLE asset_history (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            asset_id uuid NOT NULL,
            changed_fields jsonb NOT NULL,
            changed_at timestamptz NOT NULL DEFAULT now(),
            changed_by varchar(16) NOT NULL,
            CONSTRAINT uq_asset_history_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_asset_history_asset_tenant FOREIGN KEY (user_id, asset_id)
                REFERENCES manual_assets(user_id, id) ON DELETE CASCADE,
            CONSTRAINT ck_asset_history_changed_by
                CHECK (changed_by IN ('user', 'import', 'system'))
        )
        """
    )
    op.execute("CREATE INDEX ix_asset_history_user_id ON asset_history (user_id)")
    op.execute(
        """
        CREATE TABLE asset_cashflows (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            asset_id uuid NOT NULL,
            on_date date NOT NULL,
            amount numeric(16, 2) NOT NULL,
            kind varchar(16) NOT NULL,
            planned boolean NOT NULL DEFAULT false,
            CONSTRAINT uq_asset_cashflows_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_asset_cashflows_asset_tenant FOREIGN KEY (user_id, asset_id)
                REFERENCES manual_assets(user_id, id) ON DELETE CASCADE,
            CONSTRAINT ck_asset_cashflows_kind
                CHECK (kind IN ('deposit', 'premium', 'interest', 'maturity', 'withdrawal'))
        )
        """
    )
    op.execute("CREATE INDEX ix_asset_cashflows_user_id ON asset_cashflows (user_id)")
    op.execute(
        """
        CREATE TABLE holdings_lots (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            account_id uuid,
            member_id uuid,
            security_id uuid NOT NULL,
            quantity numeric(20, 6) NOT NULL,
            cost_per_unit numeric(16, 4),
            acquired_on date,
            source varchar(16) NOT NULL,
            CONSTRAINT uq_holdings_lots_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_holdings_lots_account_tenant FOREIGN KEY (user_id, account_id)
                REFERENCES accounts(user_id, id) ON DELETE CASCADE,
            CONSTRAINT fk_holdings_lots_member_tenant FOREIGN KEY (user_id, member_id)
                REFERENCES household_members(user_id, id) ON DELETE CASCADE,
            CONSTRAINT fk_holdings_lots_security FOREIGN KEY (security_id)
                REFERENCES securities(id) ON DELETE RESTRICT,
            CONSTRAINT ck_holdings_lots_source CHECK (source IN ('cas', 'broker', 'manual'))
        )
        """
    )
    op.execute("CREATE INDEX ix_holdings_lots_user_id ON holdings_lots (user_id)")
    op.execute(
        """
        CREATE TABLE import_jobs (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            doc_type varchar(32) NOT NULL,
            status varchar(16) NOT NULL,
            parse_report jsonb,
            file_deleted_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_import_jobs_user_id_id UNIQUE (user_id, id)
        )
        """
    )
    op.execute("CREATE INDEX ix_import_jobs_user_id ON import_jobs (user_id)")
    op.execute(
        """
        CREATE TABLE import_items (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_id uuid NOT NULL,
            item_kind varchar(32) NOT NULL,
            payload jsonb NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'new',
            CONSTRAINT uq_import_items_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_import_items_job_tenant FOREIGN KEY (user_id, job_id)
                REFERENCES import_jobs(user_id, id) ON DELETE CASCADE,
            CONSTRAINT ck_import_items_status CHECK (
                status IN ('new', 'changed', 'duplicate', 'unknown', 'accepted', 'rejected')
            )
        )
        """
    )
    op.execute("CREATE INDEX ix_import_items_user_id ON import_items (user_id)")
    op.execute(
        """
        CREATE TABLE networth_snapshots (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            on_date date NOT NULL,
            totals jsonb NOT NULL,
            CONSTRAINT uq_networth_snapshots_user_id_id UNIQUE (user_id, id),
            CONSTRAINT uq_networth_snapshots_user_date UNIQUE (user_id, on_date)
        )
        """
    )
    op.execute("CREATE INDEX ix_networth_snapshots_user_id ON networth_snapshots (user_id)")
    op.execute(
        """
        CREATE TABLE actions (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            week varchar(8) NOT NULL,
            rank smallint NOT NULL,
            kind varchar(32) NOT NULL,
            payload jsonb NOT NULL,
            status varchar(16) NOT NULL DEFAULT 'proposed',
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_actions_user_id_id UNIQUE (user_id, id),
            CONSTRAINT ck_actions_week CHECK (week ~ '^[0-9]{4}-W[0-9]{2}$'),
            CONSTRAINT ck_actions_status CHECK (status IN ('proposed', 'done', 'dismissed'))
        )
        """
    )
    op.execute("CREATE INDEX ix_actions_user_id ON actions (user_id)")
    op.execute(
        """
        CREATE TABLE exit_cards (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            lot_id uuid NOT NULL,
            rules jsonb NOT NULL,
            status jsonb NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_exit_cards_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_exit_cards_lot_tenant FOREIGN KEY (user_id, lot_id)
                REFERENCES holdings_lots(user_id, id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX ix_exit_cards_user_id ON exit_cards (user_id)")
    op.execute(
        """
        CREATE TABLE reminders (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            asset_id uuid,
            due_on date NOT NULL,
            kind varchar(32) NOT NULL,
            sent_at timestamptz,
            CONSTRAINT uq_reminders_user_id_id UNIQUE (user_id, id),
            CONSTRAINT fk_reminders_asset_tenant FOREIGN KEY (user_id, asset_id)
                REFERENCES manual_assets(user_id, id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX ix_reminders_user_id ON reminders (user_id)")
    op.execute(
        """
        CREATE TABLE push_subscriptions (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            endpoint text NOT NULL UNIQUE,
            keys jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_push_subscriptions_user_id_id UNIQUE (user_id, id)
        )
        """
    )
    op.execute("CREATE INDEX ix_push_subscriptions_user_id ON push_subscriptions (user_id)")
    op.execute(
        """
        CREATE TABLE consents (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            purpose varchar(80) NOT NULL,
            granted_at timestamptz NOT NULL,
            withdrawn_at timestamptz,
            CONSTRAINT uq_consents_user_id_id UNIQUE (user_id, id)
        )
        """
    )
    op.execute("CREATE INDEX ix_consents_user_id ON consents (user_id)")
    op.execute(
        """
        CREATE TABLE audit_log (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            at timestamptz NOT NULL DEFAULT now(),
            actor varchar(80) NOT NULL,
            action varchar(80) NOT NULL,
            detail jsonb NOT NULL,
            CONSTRAINT uq_audit_log_user_id_id UNIQUE (user_id, id)
        )
        """
    )
    op.execute("CREATE INDEX ix_audit_log_user_id ON audit_log (user_id)")

    op.execute("GRANT wealth_app TO CURRENT_USER")
    op.execute(
        "DO $$ BEGIN EXECUTE format("
        "'GRANT USAGE ON SCHEMA %I TO wealth_app', current_schema()); END $$"
    )
    user_tables = ", ".join(USER_TABLES)
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {user_tables} TO wealth_app")
    op.execute("GRANT SELECT ON TABLE securities TO wealth_app")

    for table_name, policy_name in zip(USER_TABLES, RLS_POLICIES, strict=True):
        column = "id" if table_name == "users" else "user_id"
        check = f"{column} = nullif(current_setting('app.user_id', true), '')::uuid"
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {policy_name} ON {table_name} FOR ALL TO wealth_app "
            f"USING ({check}) WITH CHECK ({check})"
        )


def downgrade() -> None:
    tables_to_drop = [*reversed(USER_TABLES), "securities"]
    op.execute(f"DROP TABLE {', '.join(tables_to_drop)}")
