"""Resolve an app-session token's user without ever creating one.

`app_resolve_user` creates a user for any valid token. After "Delete my account", a
request still in flight with the old access token recreated a bare user row (email and
Google subject). App-session tokens are only issued to existing users, so they now
resolve through this lookup, which returns NULL for a deleted user or a dead session.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0025_session_user_lookup"
down_revision = "0024_etf_watchers_scorekeeper"
branch_labels = None
depends_on = None


def _schema() -> str:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    return connection.dialect.identifier_preparer.quote_schema(schema)


def upgrade() -> None:
    quoted_schema = _schema()
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_session_user(
            p_session_id uuid,
            p_auth_subject text
        ) RETURNS uuid
        LANGUAGE sql
        STABLE
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
            SELECT users.id
            FROM {quoted_schema}.auth_sessions AS sessions
            JOIN {quoted_schema}.users AS users ON users.id = sessions.user_id
            WHERE sessions.id = p_session_id
              AND users.auth_subject = p_auth_subject
              AND sessions.revoked_at IS NULL
              AND sessions.expires_at > pg_catalog.now()
              AND sessions.absolute_expires_at > pg_catalog.now()
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.app_session_user(uuid, text) FROM PUBLIC")
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_session_user(uuid, text) TO wealth_app"
    )


def downgrade() -> None:
    quoted_schema = _schema()
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_session_user(uuid, text)")
