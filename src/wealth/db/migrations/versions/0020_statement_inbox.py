"""Add household Gmail statement inbox, upload keys, and saved passwords."""

from __future__ import annotations

import re

from alembic import op
from sqlalchemy import text

revision = "0020_statement_inbox"
down_revision = "0019_auth_sessions"
branch_labels = None
depends_on = None


def _schema() -> tuple[str, str]:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    return schema, connection.dialect.identifier_preparer.quote_schema(schema)


def _install_household_access(quoted_schema: str, table_name: str) -> None:
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {quoted_schema}.{table_name} TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.{table_name} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {quoted_schema}.{table_name} FORCE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY {table_name}_household_access
        ON {quoted_schema}.{table_name} FOR ALL TO wealth_app
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
        CREATE TRIGGER trg_{table_name}_household_audit
        AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.{table_name}
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_household_write()
        """
    )


def _revise_household_merge(*, add_cleanup: bool) -> None:
    connection = op.get_bind()
    _schema_name, quoted_schema = _schema()
    function_name = f"{quoted_schema}.app_merge_into_household_before_0013(uuid)"
    definition = connection.execute(
        text("SELECT pg_get_functiondef(CAST(:function_name AS regprocedure))"),
        {"function_name": function_name},
    ).scalar_one()
    block = f"""            -- Preserve household uniqueness when inbox data is merged.
            DELETE FROM {quoted_schema}.statement_inbox_items AS pending_source
            WHERE pending_source.user_id = active_user
              AND EXISTS (
                  SELECT 1 FROM {quoted_schema}.statement_inbox_items AS pending_target
                  WHERE pending_target.user_id = p_owner
                    AND pending_target.message_id = pending_source.message_id
                    AND pending_target.file_hash = pending_source.file_hash
              );
            GET DIAGNOSTICS table_count = ROW_COUNT;
            total_counts := total_counts || jsonb_build_object(
                'statement_inbox_items_discarded_as_duplicate', table_count
            );

            DELETE FROM {quoted_schema}.statement_saved_passwords AS password_source
            WHERE password_source.user_id = active_user
              AND EXISTS (
                  SELECT 1 FROM {quoted_schema}.statement_saved_passwords AS password_target
                  WHERE password_target.user_id = p_owner
                    AND password_target.sender_type = password_source.sender_type
              );
            GET DIAGNOSTICS table_count = ROW_COUNT;
            total_counts := total_counts || jsonb_build_object(
                'statement_saved_passwords_discarded_as_duplicate', table_count
            );

"""
    marker = "            FOR table_row IN\n                SELECT columns.table_name,"
    if add_cleanup:
        if (
            marker not in definition
            or "Preserve household uniqueness when inbox data is merged." in definition
        ):
            raise RuntimeError("Could not add statement inbox household merge rules")
        definition = definition.replace(marker, block + marker, 1)
    else:
        definition, count = re.subn(
            r"            -- Preserve household uniqueness when inbox data is merged\.\n"
            r"[\s\S]+?\n\n            FOR table_row IN\n"
            r"                SELECT columns\.table_name,",
            marker,
            definition,
            count=1,
        )
        if count != 1:
            raise RuntimeError("Could not remove statement inbox household merge rules")
    connection.exec_driver_sql(definition.replace("%", "%%"))


def upgrade() -> None:
    _schema_name, quoted_schema = _schema()
    op.execute(
        f"""
        CREATE TABLE {quoted_schema}.statement_inbox_keys (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            created_by uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            label varchar(80) NOT NULL,
            key_hash varchar(64) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            last_used_at timestamptz,
            revoked_at timestamptz,
            CONSTRAINT uq_statement_inbox_keys_user_id_id UNIQUE (user_id, id),
            CONSTRAINT uq_statement_inbox_keys_key_hash UNIQUE (key_hash),
            CONSTRAINT ck_statement_inbox_keys_key_hash CHECK (key_hash ~ '^[0-9a-f]{{64}}$')
        )
        """
    )
    op.create_index(
        "ix_statement_inbox_keys_user_id",
        "statement_inbox_keys",
        ["user_id"],
    )
    op.create_index(
        "ix_statement_inbox_keys_user_created",
        "statement_inbox_keys",
        ["user_id", "created_at"],
    )
    _install_household_access(quoted_schema, "statement_inbox_keys")

    op.execute(
        f"""
        CREATE TABLE {quoted_schema}.statement_inbox_items (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            key_id uuid NOT NULL,
            sender_type varchar(16) NOT NULL,
            sender varchar(320) NOT NULL,
            subject varchar(512) NOT NULL,
            received_at timestamptz NOT NULL,
            message_id varchar(255) NOT NULL,
            file_hash varchar(64) NOT NULL,
            pdf_data bytea,
            status varchar(24) NOT NULL DEFAULT 'needs_password',
            error_code varchar(40),
            import_job_id uuid,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_statement_inbox_items_user_id_id UNIQUE (user_id, id),
            CONSTRAINT uq_statement_inbox_items_message_file
                UNIQUE (user_id, message_id, file_hash),
            CONSTRAINT fk_statement_inbox_items_key_tenant
                FOREIGN KEY (user_id, key_id)
                REFERENCES {quoted_schema}.statement_inbox_keys(user_id, id)
                ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE,
            CONSTRAINT fk_statement_inbox_items_job_tenant
                FOREIGN KEY (user_id, import_job_id)
                REFERENCES {quoted_schema}.import_jobs(user_id, id)
                ON DELETE SET NULL (import_job_id) DEFERRABLE INITIALLY IMMEDIATE,
            CONSTRAINT ck_statement_inbox_items_sender_type
                CHECK (sender_type IN ('cams', 'kfintech', 'mfcentral', 'nsdl', 'cdsl', 'other')),
            CONSTRAINT ck_statement_inbox_items_status
                CHECK (status IN ('needs_password', 'review', 'unreadable')),
            CONSTRAINT ck_statement_inbox_items_file_hash
                CHECK (file_hash ~ '^[0-9a-f]{{64}}$')
        )
        """
    )
    op.create_index(
        "ix_statement_inbox_items_user_id",
        "statement_inbox_items",
        ["user_id"],
    )
    op.create_index(
        "ix_statement_inbox_items_user_received",
        "statement_inbox_items",
        ["user_id", "received_at"],
    )
    op.create_index(
        "ix_statement_inbox_items_key_received",
        "statement_inbox_items",
        ["key_id", "received_at"],
    )
    _install_household_access(quoted_schema, "statement_inbox_items")

    op.execute(
        f"""
        CREATE TABLE {quoted_schema}.statement_saved_passwords (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES {quoted_schema}.users(id) ON DELETE CASCADE,
            sender_type varchar(16) NOT NULL,
            nonce bytea NOT NULL,
            ciphertext bytea NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            last_used_at timestamptz,
            CONSTRAINT uq_statement_saved_passwords_user_id_id UNIQUE (user_id, id),
            CONSTRAINT uq_statement_saved_passwords_sender UNIQUE (user_id, sender_type),
            CONSTRAINT ck_statement_saved_passwords_sender_type
                CHECK (sender_type IN ('cams', 'kfintech', 'mfcentral', 'nsdl', 'cdsl', 'other')),
            CONSTRAINT ck_statement_saved_passwords_nonce CHECK (octet_length(nonce) = 12)
        )
        """
    )
    _install_household_access(quoted_schema, "statement_saved_passwords")
    op.create_index(
        "ix_statement_saved_passwords_user_id",
        "statement_saved_passwords",
        ["user_id"],
    )

    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_lookup_statement_inbox_key(p_key_hash text)
        RETURNS TABLE(key_id uuid, user_id uuid, created_by uuid)
        LANGUAGE sql
        STABLE
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
            SELECT keys.id, keys.user_id, keys.created_by
            FROM {quoted_schema}.statement_inbox_keys AS keys
            WHERE keys.key_hash = p_key_hash
              AND keys.revoked_at IS NULL
            LIMIT 1
        $function$
        """
    )
    op.execute(
        f"REVOKE ALL ON FUNCTION {quoted_schema}.app_lookup_statement_inbox_key(text) FROM PUBLIC"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.app_lookup_statement_inbox_key(text) "
        "TO wealth_api"
    )
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.app_ingest_statement_inbox_item(
            p_key_hash text,
            p_item_id uuid,
            p_sender_type text,
            p_sender text,
            p_subject text,
            p_received_at timestamptz,
            p_message_id text,
            p_file_hash text,
            p_pdf_data bytea
        )
        RETURNS TABLE(status text, item_id uuid, user_id uuid, created_by uuid)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        DECLARE
            matched_key {quoted_schema}.statement_inbox_keys%ROWTYPE;
            existing_item uuid;
            recent_uploads bigint;
        BEGIN
            SELECT * INTO matched_key
            FROM {quoted_schema}.statement_inbox_keys AS keys
            WHERE keys.key_hash = p_key_hash AND keys.revoked_at IS NULL
            FOR UPDATE;
            IF matched_key.id IS NULL THEN
                RETURN QUERY SELECT 'invalid'::text, NULL::uuid, NULL::uuid, NULL::uuid;
                RETURN;
            END IF;

            SELECT pending.id INTO existing_item
            FROM {quoted_schema}.statement_inbox_items AS pending
            WHERE pending.user_id = matched_key.user_id
              AND pending.message_id = p_message_id
              AND pending.file_hash = p_file_hash
            LIMIT 1;
            IF existing_item IS NOT NULL THEN
                UPDATE {quoted_schema}.statement_inbox_keys
                SET last_used_at = clock_timestamp()
                WHERE id = matched_key.id;
                RETURN QUERY SELECT
                    'duplicate'::text, existing_item, matched_key.user_id, matched_key.created_by;
                RETURN;
            END IF;

            SELECT count(*) INTO recent_uploads
            FROM {quoted_schema}.statement_inbox_items AS pending
            WHERE pending.key_id = matched_key.id
              AND pending.created_at >= clock_timestamp() - interval '24 hours';
            IF recent_uploads >= 20 THEN
                RETURN QUERY SELECT 'rate_limited'::text, NULL::uuid,
                    matched_key.user_id, matched_key.created_by;
                RETURN;
            END IF;

            PERFORM set_config('app.user_id', matched_key.created_by::text, true);
            INSERT INTO {quoted_schema}.statement_inbox_items (
                id, user_id, key_id, sender_type, sender, subject, received_at,
                message_id, file_hash, pdf_data, status
            ) VALUES (
                p_item_id, matched_key.user_id, matched_key.id, p_sender_type,
                p_sender, p_subject, p_received_at, p_message_id, p_file_hash,
                p_pdf_data, 'needs_password'
            );
            UPDATE {quoted_schema}.statement_inbox_keys
            SET last_used_at = clock_timestamp()
            WHERE id = matched_key.id;
            RETURN QUERY SELECT 'received'::text, p_item_id,
                matched_key.user_id, matched_key.created_by;
        END;
        $function$
        """
    )
    op.execute(
        f"""REVOKE ALL ON FUNCTION {quoted_schema}.app_ingest_statement_inbox_item(
            text, uuid, text, text, text, timestamptz, text, text, bytea
        ) FROM PUBLIC"""
    )
    op.execute(
        f"""GRANT EXECUTE ON FUNCTION {quoted_schema}.app_ingest_statement_inbox_item(
            text, uuid, text, text, text, timestamptz, text, text, bytea
        ) TO wealth_api"""
    )
    _revise_household_merge(add_cleanup=True)


def downgrade() -> None:
    _schema_name, quoted_schema = _schema()
    _revise_household_merge(add_cleanup=False)
    op.execute(
        f"""DROP FUNCTION IF EXISTS {quoted_schema}.app_ingest_statement_inbox_item(
            text, uuid, text, text, text, timestamptz, text, text, bytea
        )"""
    )
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.app_lookup_statement_inbox_key(text)")
    op.drop_table("statement_saved_passwords")
    op.drop_table("statement_inbox_items")
    op.drop_table("statement_inbox_keys")
