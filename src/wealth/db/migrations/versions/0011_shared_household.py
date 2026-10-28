"""Add shared household grants, invites, and household-scoped row security."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Column, Uuid, text

revision = "0011_shared_household"
down_revision = "0010_household_planning"
branch_labels = None
depends_on = None


def _schema() -> tuple[str, str]:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    return schema, connection.dialect.identifier_preparer.quote_schema(schema)


def _user_tables(schema: str) -> list[str]:
    return list(
        op.get_bind()
        .execute(
            text(
                """
                SELECT table_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND column_name = 'user_id'
                  AND table_name IN (
                      SELECT table_name FROM information_schema.tables
                      WHERE table_schema = :schema AND table_type = 'BASE TABLE'
                  )
                ORDER BY table_name
                """
            ),
            {"schema": schema},
        )
        .scalars()
    )


def _drop_wealth_app_policies(schema: str, table_name: str) -> None:
    connection = op.get_bind()
    policies = connection.execute(
        text(
            """
            SELECT policy.polname
            FROM pg_catalog.pg_policy AS policy
            JOIN pg_catalog.pg_class AS relation ON relation.oid = policy.polrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = :schema
              AND relation.relname = :table_name
              AND 'wealth_app'::regrole = ANY(policy.polroles::regrole[])
            """
        ),
        {"schema": schema, "table_name": table_name},
    ).scalars()
    preparer = connection.dialect.identifier_preparer
    quoted_schema = preparer.quote_schema(schema)
    quoted_table = preparer.quote_identifier(table_name)
    for policy_name in policies:
        quoted_policy = preparer.quote_identifier(policy_name)
        op.execute(f"DROP POLICY {quoted_policy} ON {quoted_schema}.{quoted_table}")


def _install_household_policies(schema: str, quoted_schema: str) -> None:
    connection = op.get_bind()
    preparer = connection.dialect.identifier_preparer
    for table_name in _user_tables(schema):
        quoted_table = preparer.quote_identifier(table_name)
        _drop_wealth_app_policies(schema, table_name)
        if table_name in {"consents", "push_subscriptions"}:
            # These records belong to a login, even when its wealth is shared.
            policy_name = preparer.quote_identifier(f"{table_name}_personal_isolation")
            predicate = "user_id = nullif(current_setting('app.user_id', true), '')::uuid"
            op.execute(
                f"CREATE POLICY {policy_name} ON {quoted_schema}.{quoted_table} FOR ALL "
                f"TO wealth_app USING ({predicate}) WITH CHECK ({predicate})"
            )
            continue

        policy_name = preparer.quote_identifier(f"{table_name}_household_access")
        using = (
            "user_id = nullif(current_setting('app.household_owner', true), '')::uuid "
            "AND app_household_role(user_id) IS NOT NULL"
        )
        check = (
            "user_id = nullif(current_setting('app.household_owner', true), '')::uuid "
            "AND app_household_role(user_id) IN ('owner', 'partner')"
        )
        op.execute(
            f"CREATE POLICY {policy_name} ON {quoted_schema}.{quoted_table} FOR ALL "
            f"TO wealth_app USING ({using}) WITH CHECK ({check})"
        )


def _install_original_policies(schema: str, quoted_schema: str) -> None:
    connection = op.get_bind()
    preparer = connection.dialect.identifier_preparer
    for table_name in _user_tables(schema):
        quoted_table = preparer.quote_identifier(table_name)
        _drop_wealth_app_policies(schema, table_name)
        policy_name = {
            "weekly_plans": "weekly_plans_user_isolation",
            "alerts_log": "alerts_log_user_isolation",
        }.get(table_name, f"{table_name}_isolation")
        quoted_policy = preparer.quote_identifier(policy_name)
        predicate = "user_id = nullif(current_setting('app.user_id', true), '')::uuid"
        op.execute(
            f"CREATE POLICY {quoted_policy} ON {quoted_schema}.{quoted_table} FOR ALL "
            f"TO wealth_app USING ({predicate}) WITH CHECK ({predicate})"
        )


def _install_audit_triggers(schema: str, quoted_schema: str) -> None:
    preparer = op.get_bind().dialect.identifier_preparer
    function_name = preparer.quote_identifier("audit_household_write")
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
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.{function_name}()
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
            IF NULLIF(current_setting('app.household_deleting_user', true), '') =
                tenant_id::text THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
            actor_id := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF actor_id IS NULL OR actor_id = tenant_id THEN
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
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.{function_name}() FROM PUBLIC")
    for table_name in _user_tables(schema):
        if table_name == "audit_log":
            continue
        quoted_table = preparer.quote_identifier(table_name)
        trigger_name = preparer.quote_identifier(f"trg_{table_name}_household_audit")
        op.execute(
            f"CREATE TRIGGER {trigger_name} AFTER INSERT OR UPDATE OR DELETE "
            f"ON {quoted_schema}.{quoted_table} FOR EACH ROW "
            f"EXECUTE FUNCTION {quoted_schema}.{function_name}()"
        )


def _make_tenant_foreign_keys_deferrable(schema: str) -> None:
    connection = op.get_bind()
    constraints = connection.execute(
        text(
            """
            SELECT relation.relname AS table_name, con.conname AS constraint_name
            FROM pg_catalog.pg_constraint AS con
            JOIN pg_catalog.pg_class AS relation ON relation.oid = con.conrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = :schema
              AND con.contype = 'f'
              AND cardinality(con.conkey) > 1
              AND NOT con.condeferrable
            """
        ),
        {"schema": schema},
    ).all()
    preparer = connection.dialect.identifier_preparer
    quoted_schema = preparer.quote_schema(schema)
    for table_name, constraint_name in constraints:
        op.execute(
            f"ALTER TABLE {quoted_schema}.{preparer.quote_identifier(table_name)} "
            f"ALTER CONSTRAINT {preparer.quote_identifier(constraint_name)} "
            "DEFERRABLE INITIALLY IMMEDIATE"
        )


def _restore_tenant_foreign_keys(schema: str) -> None:
    connection = op.get_bind()
    constraints = connection.execute(
        text(
            """
            SELECT relation.relname AS table_name, con.conname AS constraint_name
            FROM pg_catalog.pg_constraint AS con
            JOIN pg_catalog.pg_class AS relation ON relation.oid = con.conrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = :schema
              AND con.contype = 'f'
              AND con.condeferrable
              AND cardinality(con.conkey) > 1
            """
        ),
        {"schema": schema},
    ).all()
    preparer = connection.dialect.identifier_preparer
    quoted_schema = preparer.quote_schema(schema)
    for table_name, constraint_name in constraints:
        op.execute(
            f"ALTER TABLE {quoted_schema}.{preparer.quote_identifier(table_name)} "
            f"ALTER CONSTRAINT {preparer.quote_identifier(constraint_name)} "
            "NOT DEFERRABLE INITIALLY IMMEDIATE"
        )


def upgrade() -> None:
    schema, quoted_schema = _schema()
    schema_literal = "'" + schema.replace("'", "''") + "'"
    op.add_column("audit_log", Column("actor_user_id", Uuid(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_audit_log_actor_user",
        "audit_log",
        "users",
        ["actor_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.add_column("actions", Column("done_by", Uuid(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_actions_done_by_user",
        "actions",
        "users",
        ["done_by"],
        ["id"],
        ondelete="SET NULL",
    )

    op.execute(
        f"""
        CREATE TABLE {quoted_schema}.household_access (
            owner_user_id uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            grantee_user_id uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            member_id uuid NOT NULL,
            role text NOT NULL CHECK (role IN ('partner', 'viewer')),
            created_at timestamptz NOT NULL DEFAULT now(),
            revoked_at timestamptz,
            PRIMARY KEY (owner_user_id, grantee_user_id),
            CONSTRAINT fk_household_access_member_tenant
                FOREIGN KEY (owner_user_id, member_id)
                REFERENCES {quoted_schema}.household_members(user_id, id) ON DELETE CASCADE,
            CHECK (owner_user_id <> grantee_user_id)
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE {quoted_schema}.household_invites (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            owner_user_id uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            email_lower varchar(320) NOT NULL CHECK (email_lower = lower(email_lower)),
            role text NOT NULL CHECK (role IN ('partner', 'viewer')),
            member_id uuid,
            token_sha256 char(64) NOT NULL UNIQUE CHECK (token_sha256 ~ '^[0-9a-f]{{64}}$'),
            expires_at timestamptz NOT NULL,
            accepted_at timestamptz,
            revoked_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT fk_household_invites_member_tenant
                FOREIGN KEY (owner_user_id, member_id)
                REFERENCES {quoted_schema}.household_members(user_id, id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        f"CREATE INDEX ix_household_access_grantee ON {quoted_schema}.household_access "
        "(grantee_user_id, revoked_at)"
    )
    op.execute(
        f"CREATE UNIQUE INDEX uq_household_access_owner_member_active "
        f"ON {quoted_schema}.household_access (owner_user_id, member_id) "
        "WHERE revoked_at IS NULL"
    )
    op.execute(
        f"CREATE INDEX ix_household_invites_owner_created ON {quoted_schema}.household_invites "
        "(owner_user_id, created_at DESC)"
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "
        f"{quoted_schema}.household_access, {quoted_schema}.household_invites TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.household_access ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {quoted_schema}.household_invites ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY household_access_select ON {quoted_schema}.household_access
        FOR SELECT TO wealth_app
        USING (
            owner_user_id = nullif(current_setting('app.user_id', true), '')::uuid
            OR grantee_user_id = nullif(current_setting('app.user_id', true), '')::uuid
        )
        """
    )
    for command in ("INSERT", "UPDATE", "DELETE"):
        name = f"household_access_owner_{command.lower()}"
        using = (
            " USING (owner_user_id = nullif(current_setting('app.user_id', true), '')::uuid)"
            if command != "INSERT"
            else ""
        )
        check = (
            " WITH CHECK (owner_user_id = nullif(current_setting('app.user_id', true), '')::uuid)"
            if command in {"INSERT", "UPDATE"}
            else ""
        )
        op.execute(
            f"CREATE POLICY {name} ON {quoted_schema}.household_access "
            f"FOR {command} TO wealth_app{using}{check}"
        )
    op.execute(
        f"""
        CREATE POLICY household_invites_owner_access ON {quoted_schema}.household_invites
        FOR ALL TO wealth_app
        USING (owner_user_id = nullif(current_setting('app.user_id', true), '')::uuid)
        WITH CHECK (owner_user_id = nullif(current_setting('app.user_id', true), '')::uuid)
        """
    )

    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_household_role(p_owner uuid)
        RETURNS text
        LANGUAGE plpgsql
        STABLE
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            active_user uuid;
            grant_role text;
        BEGIN
            active_user := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF active_user IS NULL OR p_owner IS NULL THEN
                RETURN NULL;
            END IF;
            IF p_owner = active_user THEN
                RETURN 'owner';
            END IF;
            SELECT access.role INTO grant_role
            FROM {quoted_schema}.household_access AS access
            WHERE access.owner_user_id = p_owner
              AND access.grantee_user_id = active_user
              AND access.revoked_at IS NULL;
            RETURN grant_role;
        END;
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.app_household_role(uuid) FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_household_role(uuid) TO wealth_app")

    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.prevent_linked_household_member_delete()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        BEGIN
            IF pg_trigger_depth() > 1 THEN
                RETURN OLD;
            END IF;
            IF EXISTS (
                SELECT 1 FROM {quoted_schema}.household_access AS access
                WHERE access.owner_user_id = OLD.user_id
                  AND access.member_id = OLD.id
                  AND access.revoked_at IS NULL
            ) THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'member_linked_to_login';
            END IF;
            RETURN OLD;
        END;
        $function$
        """
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION "
        f"{quoted_schema}.prevent_linked_household_member_delete() FROM PUBLIC"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION "
        f"{quoted_schema}.prevent_linked_household_member_delete() TO wealth_app"
    )
    op.execute(
        f"CREATE TRIGGER trg_household_member_active_access_guard "
        f"BEFORE DELETE ON {quoted_schema}.household_members "
        f"FOR EACH ROW EXECUTE FUNCTION "
        f"{quoted_schema}.prevent_linked_household_member_delete()"
    )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {quoted_schema}.default_self_member_for_owner()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            active_user uuid;
        BEGIN
            IF NEW.member_id IS NULL THEN
                active_user := NULLIF(current_setting('app.user_id', true), '')::uuid;
                IF active_user IS NULL OR active_user = NEW.user_id THEN
                    SELECT member.id INTO NEW.member_id
                    FROM {quoted_schema}.household_members AS member
                    WHERE member.user_id = NEW.user_id AND member.is_self;
                ELSE
                    SELECT access.member_id INTO NEW.member_id
                    FROM {quoted_schema}.household_access AS access
                    WHERE access.owner_user_id = NEW.user_id
                      AND access.grantee_user_id = active_user
                      AND access.revoked_at IS NULL;
                END IF;
            END IF;
            IF NEW.member_id IS NULL OR NOT EXISTS (
                SELECT 1 FROM {quoted_schema}.household_members AS member
                WHERE member.user_id = NEW.user_id AND member.id = NEW.member_id
            ) THEN
                RAISE EXCEPTION 'a member in the selected household is required';
            END IF;
            RETURN NEW;
        END;
        $function$
        """
    )

    op.execute(
        f"""
        CREATE VIEW {quoted_schema}.household_people WITH (security_barrier = true) AS
        SELECT access.owner_user_id AS owner_id,
               access.grantee_user_id AS user_id,
               grantee.display_name,
               access.role
        FROM {quoted_schema}.household_access AS access
        JOIN {quoted_schema}.users AS grantee ON grantee.id = access.grantee_user_id
        WHERE access.revoked_at IS NULL
          AND {quoted_schema}.app_household_role(access.owner_user_id) IS NOT NULL
        UNION ALL
        SELECT access.owner_user_id AS owner_id,
               access.owner_user_id AS user_id,
               owner.display_name,
               'owner'::text AS role
        FROM {quoted_schema}.household_access AS access
        JOIN {quoted_schema}.users AS owner ON owner.id = access.owner_user_id
        WHERE access.revoked_at IS NULL
          AND {quoted_schema}.app_household_role(access.owner_user_id) IS NOT NULL
        """
    )
    op.execute(f"REVOKE ALL ON {quoted_schema}.household_people FROM PUBLIC")
    op.execute(f"GRANT SELECT ON {quoted_schema}.household_people TO wealth_app")

    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_accept_invite(p_token text)
        RETURNS uuid
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            active_user uuid;
            active_email text;
            active_name text;
            invite_row {quoted_schema}.household_invites%ROWTYPE;
            linked_member uuid;
            linked_member_is_self boolean;
        BEGIN
            active_user := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF active_user IS NULL OR p_token IS NULL OR length(p_token) < 43 THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'invalid_invite';
            END IF;
            SELECT email, display_name INTO active_email, active_name
            FROM {quoted_schema}.users WHERE id = active_user;
            SELECT * INTO invite_row
            FROM {quoted_schema}.household_invites
            WHERE token_sha256 = encode(sha256(convert_to(p_token, 'UTF8')), 'hex')
              AND accepted_at IS NULL
              AND revoked_at IS NULL
              AND expires_at > now()
            FOR UPDATE;
            IF invite_row.id IS NULL
               OR lower(active_email) <> invite_row.email_lower
               OR invite_row.owner_user_id = active_user THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'invalid_invite';
            END IF;

            linked_member := invite_row.member_id;
            IF linked_member IS NULL THEN
                INSERT INTO {quoted_schema}.household_members
                    (id, user_id, display_name, relationship, is_self)
                VALUES (
                    gen_random_uuid(), invite_row.owner_user_id,
                    left(COALESCE(NULLIF(active_name, ''), 'Partner'), 40),
                    'spouse', false
                )
                RETURNING id INTO linked_member;
            ELSE
                SELECT member.is_self INTO linked_member_is_self
                FROM {quoted_schema}.household_members AS member
                WHERE member.user_id = invite_row.owner_user_id
                  AND member.id = linked_member
                FOR UPDATE;
                IF linked_member_is_self IS DISTINCT FROM false OR EXISTS (
                    SELECT 1 FROM {quoted_schema}.household_access AS access
                    WHERE access.owner_user_id = invite_row.owner_user_id
                      AND access.member_id = linked_member
                      AND access.revoked_at IS NULL
                      AND access.grantee_user_id <> active_user
                ) THEN
                    RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'invalid_invite';
                END IF;
            END IF;
            INSERT INTO {quoted_schema}.household_access AS existing
                (owner_user_id, grantee_user_id, member_id, role, created_at, revoked_at)
            VALUES (
                invite_row.owner_user_id, active_user, linked_member,
                invite_row.role, now(), NULL
            )
            ON CONFLICT (owner_user_id, grantee_user_id) DO UPDATE
            SET member_id = EXCLUDED.member_id,
                role = EXCLUDED.role,
                created_at = now(),
                revoked_at = NULL;
            UPDATE {quoted_schema}.household_invites
            SET accepted_at = now()
            WHERE id = invite_row.id;
            INSERT INTO {quoted_schema}.audit_log
                (id, user_id, actor_user_id, actor, action, detail)
            VALUES (
                gen_random_uuid(), invite_row.owner_user_id, active_user,
                active_user::text, 'household_invite_accepted',
                jsonb_build_object('role', invite_row.role, 'member_id', linked_member)
            );
            RETURN invite_row.owner_user_id;
        END;
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.app_accept_invite(text) FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_accept_invite(text) TO wealth_app")

    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_leave_household(p_owner uuid)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            active_user uuid;
        BEGIN
            active_user := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF active_user IS NULL OR p_owner IS NULL OR active_user = p_owner THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'not_found';
            END IF;
            UPDATE {quoted_schema}.household_access
            SET revoked_at = now()
            WHERE owner_user_id = p_owner
              AND grantee_user_id = active_user
              AND revoked_at IS NULL;
            IF NOT FOUND THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'not_found';
            END IF;
            INSERT INTO {quoted_schema}.audit_log
                (id, user_id, actor_user_id, actor, action, detail)
            VALUES (
                gen_random_uuid(), p_owner, active_user, active_user::text,
                'household_access_left', jsonb_build_object('grantee_user_id', active_user)
            );
        END;
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.app_leave_household(uuid) FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_leave_household(uuid) TO wealth_app")

    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_member_has_active_household_grant(
            p_owner uuid, p_member uuid
        )
        RETURNS boolean
        LANGUAGE plpgsql
        STABLE
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        BEGIN
            IF p_owner IS NULL OR p_member IS NULL
               OR {quoted_schema}.app_household_role(p_owner) IS DISTINCT FROM 'owner' THEN
                RETURN false;
            END IF;
            RETURN EXISTS (
                SELECT 1 FROM {quoted_schema}.household_access AS access
                WHERE access.owner_user_id = p_owner
                  AND access.member_id = p_member
                  AND access.revoked_at IS NULL
            );
        END;
        $function$
        """
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION "
        f"{quoted_schema}.app_member_has_active_household_grant(uuid, uuid) FROM PUBLIC"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION "
        f"{quoted_schema}.app_member_has_active_household_grant(uuid, uuid) TO wealth_app"
    )

    _make_tenant_foreign_keys_deferrable(schema)
    _install_household_policies(schema, quoted_schema)
    _install_audit_triggers(schema, quoted_schema)

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
            source_member uuid;
            target_member uuid;
            table_row record;
            table_count bigint;
            total_counts jsonb := '{{}}'::jsonb;
            overwritten jsonb := '[]'::jsonb;
            source_profile {quoted_schema}.profiles%ROWTYPE;
            target_profile {quoted_schema}.profiles%ROWTYPE;
            source_self {quoted_schema}.household_members%ROWTYPE;
            target_self {quoted_schema}.household_members%ROWTYPE;
        BEGIN
            active_user := NULLIF(current_setting('app.user_id', true), '')::uuid;
            IF active_user IS NULL OR {quoted_schema}.app_household_role(p_owner) <> 'partner' THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'merge_not_allowed';
            END IF;
            IF EXISTS (
                SELECT 1 FROM {quoted_schema}.household_access
                WHERE owner_user_id = active_user AND revoked_at IS NULL
            ) THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'revoke_outgoing_access_first';
            END IF;
            SELECT member_id INTO target_member
            FROM {quoted_schema}.household_access
            WHERE owner_user_id = p_owner AND grantee_user_id = active_user
              AND revoked_at IS NULL AND role = 'partner';
            IF target_member IS NULL THEN
                RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'merge_not_allowed';
            END IF;
            SELECT id INTO source_member
            FROM {quoted_schema}.household_members
            WHERE user_id = active_user AND is_self;
            SELECT * INTO source_self FROM {quoted_schema}.household_members
            WHERE user_id = active_user AND is_self;
            SELECT * INTO target_self FROM {quoted_schema}.household_members
            WHERE user_id = p_owner AND id = target_member;

            IF source_member IS NOT NULL AND target_member IS NOT NULL THEN
                IF source_self.residency IS DISTINCT FROM target_self.residency THEN
                    overwritten := overwritten || '["residency"]'::jsonb;
                END IF;
                IF source_self.tax IS DISTINCT FROM target_self.tax THEN
                    overwritten := overwritten || '["tax"]'::jsonb;
                END IF;
                IF source_self.investing IS DISTINCT FROM target_self.investing THEN
                    overwritten := overwritten || '["investing"]'::jsonb;
                END IF;
                UPDATE {quoted_schema}.household_members
                SET residency = source_self.residency,
                    tax = source_self.tax,
                    investing = source_self.investing
                WHERE user_id = p_owner AND id = target_member
                  AND (residency IS DISTINCT FROM source_self.residency
                       OR tax IS DISTINCT FROM source_self.tax
                       OR investing IS DISTINCT FROM source_self.investing);
            END IF;

            SELECT * INTO source_profile FROM {quoted_schema}.profiles
            WHERE user_id = active_user;
            SELECT * INTO target_profile FROM {quoted_schema}.profiles
            WHERE user_id = p_owner;
            IF source_profile.user_id IS NOT NULL THEN
                IF target_profile.user_id IS NULL THEN
                    UPDATE {quoted_schema}.profiles SET user_id = p_owner
                    WHERE user_id = active_user;
                ELSE
                    UPDATE {quoted_schema}.profiles AS target
                    SET tax_regime = COALESCE(target.tax_regime, source_profile.tax_regime),
                        tax_slab_pct = COALESCE(target.tax_slab_pct, source_profile.tax_slab_pct),
                        city_tier = COALESCE(target.city_tier, source_profile.city_tier),
                        risk_answers = COALESCE(target.risk_answers, source_profile.risk_answers),
                        risk_profile = COALESCE(target.risk_profile, source_profile.risk_profile),
                        financial_context = COALESCE(
                            target.financial_context, source_profile.financial_context
                        ),
                        checklist = CASE WHEN target.checklist = '[]'::jsonb
                                         THEN source_profile.checklist ELSE target.checklist END,
                        updated_at = now()
                    WHERE target.user_id = p_owner;
                    DELETE FROM {quoted_schema}.profiles WHERE user_id = active_user;
                END IF;
                total_counts := total_counts || jsonb_build_object('profiles', 1);
            END IF;

            SET CONSTRAINTS ALL DEFERRED;
            DELETE FROM {quoted_schema}.weekly_plans AS source
            WHERE source.user_id = active_user
              AND EXISTS (
                  SELECT 1 FROM {quoted_schema}.weekly_plans AS target
                  WHERE target.user_id = p_owner AND target.week = source.week
              );
            GET DIAGNOSTICS table_count = ROW_COUNT;
            total_counts := total_counts || jsonb_build_object(
                'weekly_plans_discarded_as_duplicate', table_count
            );

            DELETE FROM {quoted_schema}.networth_snapshots AS source
            WHERE source.user_id = active_user
              AND EXISTS (
                  SELECT 1 FROM {quoted_schema}.networth_snapshots AS target
                  WHERE target.user_id = p_owner AND target.on_date = source.on_date
              );
            GET DIAGNOSTICS table_count = ROW_COUNT;
            total_counts := total_counts || jsonb_build_object(
                'networth_snapshots_discarded_as_duplicate', table_count
            );

            FOR table_row IN
                SELECT columns.table_name,
                       bool_or(columns.column_name = 'member_id') AS has_member_id
                FROM information_schema.columns AS columns
                WHERE columns.table_schema = {schema_literal}
                  AND columns.column_name IN ('user_id', 'member_id')
                  AND columns.table_name IN (
                      SELECT tables.table_name FROM information_schema.tables AS tables
                      WHERE tables.table_schema = {schema_literal}
                        AND tables.table_type = 'BASE TABLE'
                  )
                GROUP BY columns.table_name
                HAVING bool_or(columns.column_name = 'user_id')
                ORDER BY columns.table_name
            LOOP
                IF table_row.table_name IN (
                    'users', 'consents', 'push_subscriptions', 'profiles',
                    'weekly_plans', 'networth_snapshots'
                ) THEN
                    IF table_row.table_name IN ('consents', 'push_subscriptions') THEN
                        total_counts := total_counts || jsonb_build_object(table_row.table_name, 0);
                    END IF;
                    CONTINUE;
                END IF;
                IF table_row.table_name = 'household_members' THEN
                    EXECUTE format(
                        'UPDATE %I.household_members SET user_id = $1 '
                        'WHERE user_id = $2 AND NOT is_self', {schema_literal}
                    ) USING p_owner, active_user;
                ELSIF table_row.has_member_id THEN
                    EXECUTE format(
                        'UPDATE %I.%I SET user_id = $1, '
                        'member_id = CASE WHEN member_id = $3 THEN $4 ELSE member_id END '
                        'WHERE user_id = $2', {schema_literal}, table_row.table_name
                    ) USING p_owner, active_user, source_member, target_member;
                ELSE
                    EXECUTE format(
                        'UPDATE %I.%I SET user_id = $1 WHERE user_id = $2',
                        {schema_literal}, table_row.table_name
                    ) USING p_owner, active_user;
                END IF;
                GET DIAGNOSTICS table_count = ROW_COUNT;
                total_counts := total_counts || jsonb_build_object(
                    table_row.table_name, table_count
                );
            END LOOP;

            IF jsonb_array_length(overwritten) > 0 OR EXISTS (
                SELECT 1 FROM jsonb_each_text(total_counts) AS count_row(key, value)
                WHERE count_row.value::bigint > 0
            ) THEN
                INSERT INTO {quoted_schema}.audit_log
                    (id, user_id, actor_user_id, actor, action, detail)
                VALUES (
                    gen_random_uuid(), p_owner, active_user, active_user::text,
                    'household_merge_performed',
                    jsonb_build_object('counts', total_counts, 'overwritten_fields', overwritten)
                );
            END IF;
            RETURN jsonb_build_object('counts', total_counts, 'overwritten_fields', overwritten);
        END;
        $function$
        """
    )
    op.execute(f"REVOKE ALL ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) FROM PUBLIC")
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_merge_into_household(uuid) TO wealth_app"
    )


def downgrade() -> None:
    schema, quoted_schema = _schema()
    preparer = op.get_bind().dialect.identifier_preparer
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_merge_into_household(uuid)")
    op.execute(
        f"DROP FUNCTION IF EXISTS {quoted_schema}.app_member_has_active_household_grant(uuid, uuid)"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_leave_household(uuid)")
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_household_member_active_access_guard "
        f"ON {quoted_schema}.household_members"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.prevent_linked_household_member_delete()")
    for table_name in _user_tables(schema):
        if table_name == "audit_log":
            continue
        trigger_name = preparer.quote_identifier(f"trg_{table_name}_household_audit")
        table = preparer.quote_identifier(table_name)
        op.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON {quoted_schema}.{table}")
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.audit_household_write()")
    op.execute(f"DROP TRIGGER IF EXISTS trg_users_clear_household_delete ON {quoted_schema}.users")
    op.execute(f"DROP TRIGGER IF EXISTS trg_users_mark_household_delete ON {quoted_schema}.users")
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.clear_user_deleting()")
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.mark_user_deleting()")
    op.execute(f"DROP VIEW IF EXISTS {quoted_schema}.household_people")
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_accept_invite(text)")
    _install_original_policies(schema, quoted_schema)
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_household_role(uuid)")
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {quoted_schema}.default_self_member_for_owner()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        BEGIN
            IF NEW.member_id IS NULL THEN
                SELECT member.id INTO NEW.member_id
                FROM {quoted_schema}.household_members AS member
                WHERE member.user_id = NEW.user_id AND member.is_self;
            END IF;
            IF NEW.member_id IS NULL THEN
                RAISE EXCEPTION 'self household member is required for %', TG_TABLE_NAME;
            END IF;
            RETURN NEW;
        END;
        $function$
        """
    )
    _restore_tenant_foreign_keys(schema)
    op.drop_table("household_invites")
    op.drop_table("household_access")
    op.drop_constraint("fk_actions_done_by_user", "actions", type_="foreignkey")
    op.drop_column("actions", "done_by")
    op.drop_constraint("fk_audit_log_actor_user", "audit_log", type_="foreignkey")
    op.drop_column("audit_log", "actor_user_id")
