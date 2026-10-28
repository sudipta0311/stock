"""Separate API credentials from the database owner and add identity bootstrap."""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0002_api_role_and_bootstrap"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    owner_role = connection.execute(text("SELECT current_user")).scalar_one()
    if owner_role == "wealth_api":
        raise RuntimeError("Migrations must run as the database owner, not wealth_api")

    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'wealth_api') THEN
                CREATE ROLE wealth_api NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
                    NOREPLICATION NOBYPASSRLS INHERIT;
            END IF;
        END;
        $$;
        """
    )
    # Managed Postgres (e.g. Neon) migrates as a non-superuser owner, and only superusers may
    # name SUPERUSER/REPLICATION in ALTER ROLE, even to switch them off. Set what the owner
    # may set, then verify the attributes it may not; the resulting role is the same.
    op.execute("ALTER ROLE wealth_api NOCREATEDB NOCREATEROLE NOBYPASSRLS INHERIT")
    elevated = connection.execute(
        text(
            "SELECT rolsuper OR rolreplication FROM pg_catalog.pg_roles "
            "WHERE rolname = 'wealth_api'"
        )
    ).scalar_one()
    if elevated:
        raise RuntimeError("wealth_api must not be a superuser or a replication role")

    owner_identifier = connection.dialect.identifier_preparer.quote_identifier(owner_role)
    # REVOKE needs ADMIN on the owner role even when there is no membership (Neon's owner
    # lacks it), so only revoke a membership that actually exists.
    is_direct_member = connection.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members "
            "WHERE roleid = CAST(:owner_role AS regrole) AND member = 'wealth_api'::regrole)"
        ),
        {"owner_role": owner_role},
    ).scalar_one()
    if is_direct_member:
        op.execute(f"REVOKE {owner_identifier} FROM wealth_api")
    is_owner_member = connection.execute(
        text("SELECT pg_catalog.pg_has_role('wealth_api', :owner_role, 'MEMBER')"),
        {"owner_role": owner_role},
    ).scalar_one()
    if is_owner_member:
        raise RuntimeError("wealth_api must not be a member of the migration owner role")

    api_role_owns_objects = connection.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_shdepend AS dependency
                WHERE dependency.refclassid = 'pg_catalog.pg_authid'::pg_catalog.regclass
                  AND dependency.refobjid = (
                      SELECT role.oid FROM pg_catalog.pg_roles AS role
                      WHERE role.rolname = 'wealth_api'
                  )
                  AND dependency.deptype = 'o'
            )
            """
        )
    ).scalar_one()
    if api_role_owns_objects:
        raise RuntimeError("wealth_api must not own database objects before migration")

    op.execute("REVOKE ADMIN OPTION FOR wealth_app FROM wealth_api")
    op.execute("GRANT wealth_app TO wealth_api")

    schema_name = connection.execute(text("SELECT current_schema()")).scalar_one()
    schema_identifier = connection.dialect.identifier_preparer.quote_schema(schema_name)
    op.execute(
        f"""
        CREATE FUNCTION {schema_identifier}.app_resolve_user(
            p_auth_subject text,
            p_email text,
            p_display_name text
        ) RETURNS uuid
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {schema_identifier}
        AS $app_resolve_user$
        DECLARE
            resolved_user_id uuid;
            active_user_id uuid;
        BEGIN
            active_user_id := NULLIF(
                pg_catalog.current_setting('app.user_id', true), ''
            )::uuid;
            IF active_user_id IS NOT NULL AND NOT EXISTS (
                SELECT 1
                FROM {schema_identifier}.users AS active_user
                WHERE active_user.id = active_user_id
                  AND active_user.auth_subject = p_auth_subject
            ) THEN
                RAISE EXCEPTION 'auth subject does not match the active user';
            END IF;

            INSERT INTO {schema_identifier}.users AS target
                (id, email, display_name, auth_subject)
            VALUES (
                pg_catalog.gen_random_uuid(),
                pg_catalog.lower(p_email),
                p_display_name,
                p_auth_subject
            )
            ON CONFLICT (auth_subject) DO UPDATE
                SET email = EXCLUDED.email,
                    display_name = EXCLUDED.display_name
            RETURNING target.id INTO resolved_user_id;

            RETURN resolved_user_id;
        END;
        $app_resolve_user$
        """
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {schema_identifier}.app_resolve_user(text, text, text) FROM PUBLIC"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {schema_identifier}.app_resolve_user(text, text, text) "
        "TO wealth_app"
    )


def downgrade() -> None:
    connection = op.get_bind()
    schema_name = connection.execute(text("SELECT current_schema()")).scalar_one()
    schema_identifier = connection.dialect.identifier_preparer.quote_schema(schema_name)
    op.execute(f"DROP FUNCTION {schema_identifier}.app_resolve_user(text, text, text)")
