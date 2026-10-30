"""Make audit history append-only and remove the caller-controlled cascade bypass."""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0026_audit_log_hardening"
down_revision = "0025_session_user_lookup"
branch_labels = None
depends_on = None


def _schema() -> str:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    return connection.dialect.identifier_preparer.quote_schema(schema)


def _replace_audit_function(quoted_schema: str, *, legacy_setting_bypass: bool) -> None:
    legacy_bypass = (
        """
            IF NULLIF(current_setting('app.household_deleting_user', true), '') =
                tenant_id::text THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
    """
        if legacy_setting_bypass
        else ""
    )
    # A cascade from deleting a login runs its AFTER triggers at depth 1, after the users
    # row is gone; an audit row naming that login would fail its foreign key.
    actor_gone = (
        ""
        if legacy_setting_bypass
        else f" OR NOT EXISTS (SELECT 1 FROM {quoted_schema}.users WHERE id = actor_id)"
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {quoted_schema}.audit_household_write()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            tenant_id uuid;
            row_id text;
            actor_id uuid;
            operation text;
        BEGIN
            IF pg_trigger_depth() > 1 THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                tenant_id := OLD.user_id;
                row_id := COALESCE(to_jsonb(OLD) ->> 'id', to_jsonb(OLD) ->> 'week');
            ELSE
                tenant_id := NEW.user_id;
                row_id := COALESCE(to_jsonb(NEW) ->> 'id', to_jsonb(NEW) ->> 'week');
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM {quoted_schema}.users WHERE id = tenant_id
            ) THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
            {legacy_bypass}
            actor_id := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF actor_id IS NULL OR actor_id = tenant_id{actor_gone} THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
            operation := lower(TG_OP);

            INSERT INTO {quoted_schema}.audit_log
                (id, user_id, actor_user_id, actor, action, detail)
            VALUES (
                gen_random_uuid(), tenant_id, actor_id,
                COALESCE(actor_id::text, 'system'), 'household_row_' || operation,
                jsonb_build_object('table', TG_TABLE_NAME, 'row_id', row_id)
            );
            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END;
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.audit_household_write() FROM PUBLIC")


def _replace_highlight_audit_function(quoted_schema: str, *, legacy_setting_bypass: bool) -> None:
    # The old skip compared a setting that `mark_user_deleting` set inside a function with
    # its own SET clause, so PostgreSQL reverted it on return and it never applied. A
    # partner deleting their account then failed: the cascade logged them as the actor
    # after their users row was gone. Skip once the actor's users row no longer exists.
    skip = (
        "NULLIF(current_setting('app.household_deleting_user', true), '') = actor_id::text"
        if legacy_setting_bypass
        else f"NOT EXISTS (SELECT 1 FROM {quoted_schema}.users WHERE id = actor_id)"
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {quoted_schema}.audit_highlight_dismissal_write()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            tenant_id uuid;
            actor_id uuid;
            highlight_key text;
        BEGIN
            IF TG_OP = 'DELETE' THEN
                tenant_id := OLD.household_owner_id;
                highlight_key := OLD.highlight_key;
            ELSE
                tenant_id := NEW.household_owner_id;
                highlight_key := NEW.highlight_key;
            END IF;
            actor_id := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF actor_id IS NULL OR actor_id = tenant_id OR
                {skip} OR
                NOT EXISTS (SELECT 1 FROM {quoted_schema}.users WHERE id = tenant_id) THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
            INSERT INTO {quoted_schema}.audit_log
                (id, user_id, actor_user_id, actor, action, detail)
            VALUES (
                gen_random_uuid(), tenant_id, actor_id, actor_id::text,
                'household_row_' || lower(TG_OP),
                jsonb_build_object(
                    'table', TG_TABLE_NAME,
                    'row_id', highlight_key,
                    'highlight_key', highlight_key
                )
            );
            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END;
        $function$
        """
    )


def _install_user_delete_setting_triggers(quoted_schema: str) -> None:
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.mark_user_deleting()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        BEGIN
            PERFORM set_config('app.household_deleting_user', OLD.id::text, true);
            RETURN OLD;
        END;
        $function$
        """
    )
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.clear_user_deleting()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        BEGIN
            PERFORM set_config('app.household_deleting_user', '', true);
            RETURN OLD;
        END;
        $function$
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_users_mark_household_delete BEFORE DELETE ON {quoted_schema}.users "
        f"FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.mark_user_deleting()"
    )
    op.execute(
        f"CREATE TRIGGER trg_users_clear_household_delete AFTER DELETE ON {quoted_schema}.users "
        f"FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.clear_user_deleting()"
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.mark_user_deleting() FROM PUBLIC")
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.clear_user_deleting() FROM PUBLIC")


def upgrade() -> None:
    quoted_schema = _schema()
    op.execute(f"DROP TRIGGER IF EXISTS trg_users_mark_household_delete ON {quoted_schema}.users")
    op.execute(f"DROP TRIGGER IF EXISTS trg_users_clear_household_delete ON {quoted_schema}.users")
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.mark_user_deleting()")
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.clear_user_deleting()")
    _replace_audit_function(quoted_schema, legacy_setting_bypass=False)
    _replace_highlight_audit_function(quoted_schema, legacy_setting_bypass=False)
    op.execute(
        f"REVOKE UPDATE, DELETE, TRUNCATE ON TABLE {quoted_schema}.audit_log FROM wealth_app"
    )


def downgrade() -> None:
    quoted_schema = _schema()
    op.execute(f"GRANT UPDATE, DELETE ON TABLE {quoted_schema}.audit_log TO wealth_app")
    _replace_audit_function(quoted_schema, legacy_setting_bypass=True)
    _replace_highlight_audit_function(quoted_schema, legacy_setting_bypass=True)
    _install_user_delete_setting_triggers(quoted_schema)
