"""Add rotating per-login app sessions."""

from __future__ import annotations

import re

from alembic import op
from sqlalchemy import text

revision = "0019_auth_sessions"
down_revision = "0018_loan_events"
branch_labels = None
depends_on = None


def _schema() -> tuple[str, str]:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    return schema, connection.dialect.identifier_preparer.quote_schema(schema)


def _revise_household_merge(*, add_auth_sessions: bool) -> None:
    connection = op.get_bind()
    schema, quoted_schema = _schema()
    function_name = f"{quoted_schema}.app_merge_into_household_before_0013(uuid)"
    definition = connection.execute(
        text("SELECT pg_get_functiondef(CAST(:function_name AS regprocedure))"),
        {"function_name": function_name},
    ).scalar_one()
    if add_auth_sessions:
        definition, skip_count = re.subn(
            r"('users'\s*,\s*'consents'\s*,\s*'push_subscriptions'\s*,)(\s*'profiles'\s*,)",
            r"\1 'auth_sessions',\2",
            definition,
            count=1,
        )
        definition, personal_count = re.subn(
            r"IF table_row\.table_name IN \('consents',\s*'push_subscriptions'\) THEN",
            "IF table_row.table_name IN ('consents', 'push_subscriptions', 'auth_sessions') THEN",
            definition,
            count=1,
        )
        if skip_count != 1 or personal_count != 1:
            raise RuntimeError("Could not preserve per-login sessions during household merges")
    else:
        definition, skip_count = re.subn(
            r"'auth_sessions',\s*(?='profiles')",
            "",
            definition,
            count=1,
        )
        definition, personal_count = re.subn(
            r",\s*'auth_sessions'(?=\) THEN)",
            "",
            definition,
            count=1,
        )
        if skip_count != 1 or personal_count != 1:
            raise RuntimeError("Could not restore household merge behavior")
    # psycopg uses percent signs for placeholders, while this stored function
    # contains PostgreSQL format() placeholders such as %I.
    connection.exec_driver_sql(definition.replace("%", "%%"))


def upgrade() -> None:
    schema, quoted_schema = _schema()
    op.execute(
        f"""
        CREATE TABLE {quoted_schema}.auth_sessions (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            token_hash varchar(64) NOT NULL,
            previous_hash varchar(64),
            created_at timestamptz NOT NULL DEFAULT now(),
            last_used_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL,
            absolute_expires_at timestamptz NOT NULL,
            revoked_at timestamptz,
            device_label varchar(60) NOT NULL,
            CONSTRAINT uq_auth_sessions_user_id_id UNIQUE (user_id, id),
            CONSTRAINT uq_auth_sessions_token_hash UNIQUE (token_hash),
            CONSTRAINT ck_auth_sessions_token_hash
                CHECK (token_hash ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT ck_auth_sessions_previous_hash
                CHECK (previous_hash IS NULL OR previous_hash ~ '^[0-9a-f]{{64}}$')
        )
        """
    )
    op.create_index(
        "ix_auth_sessions_user_expiry",
        "auth_sessions",
        ["user_id", "expires_at"],
    )
    op.create_index("ix_auth_sessions_previous_hash", "auth_sessions", ["previous_hash"])
    op.execute(f"ALTER TABLE {quoted_schema}.auth_sessions ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {quoted_schema}.auth_sessions FORCE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY auth_sessions_personal_isolation
        ON {quoted_schema}.auth_sessions FOR ALL TO wealth_app
        USING (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
        WITH CHECK (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
        """
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {quoted_schema}.auth_sessions TO wealth_app"
    )
    op.execute(
        f"""
        CREATE TRIGGER trg_auth_sessions_household_audit
        AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.auth_sessions
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_household_write()
        """
    )
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_lookup_refresh_session(p_token_hash text)
        RETURNS TABLE(session_id uuid, user_id uuid)
        LANGUAGE sql
        STABLE
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
            SELECT sessions.id, sessions.user_id
            FROM {quoted_schema}.auth_sessions AS sessions
            WHERE sessions.token_hash = p_token_hash
               OR sessions.previous_hash = p_token_hash
            ORDER BY (sessions.token_hash = p_token_hash) DESC
            LIMIT 1
        $function$
        """
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {quoted_schema}.app_lookup_refresh_session(text) FROM PUBLIC"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_lookup_refresh_session(text) TO wealth_api"
    )
    _revise_household_merge(add_auth_sessions=True)


def downgrade() -> None:
    _schema_name, quoted_schema = _schema()
    _revise_household_merge(add_auth_sessions=False)
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_lookup_refresh_session(text)")
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_auth_sessions_household_audit ON {quoted_schema}.auth_sessions"
    )
    op.drop_table("auth_sessions")
