"""Store per-login highlight dismissals within their selected household."""

from __future__ import annotations

from alembic import op
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    String,
    Uuid,
    text,
)

revision = "0015_highlight_dismissals"
down_revision = "0014_international_broker_assets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)
    op.create_table(
        "highlight_dismissals",
        Column("login_user_id", Uuid(as_uuid=True), nullable=False),
        Column("household_owner_id", Uuid(as_uuid=True), nullable=False),
        Column("highlight_key", String(100), nullable=False),
        Column("fingerprint", String(24), nullable=False),
        Column(
            "dismissed_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=text("now()"),
        ),
        ForeignKeyConstraint(
            ["login_user_id"],
            ["users.id"],
            name="fk_highlight_dismissals_login",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["household_owner_id"],
            ["users.id"],
            name="fk_highlight_dismissals_household_owner",
            ondelete="CASCADE",
        ),
        PrimaryKeyConstraint(
            "login_user_id", "household_owner_id", "highlight_key", name="pk_highlight_dismissals"
        ),
        CheckConstraint("highlight_key ~ '^[a-z0-9_]{1,100}$'", name="ck_highlight_dismissals_key"),
        CheckConstraint(
            "fingerprint ~ '^[0-9a-f]{24}$'", name="ck_highlight_dismissals_fingerprint"
        ),
    )
    op.create_index(
        "ix_highlight_dismissals_household_login",
        "highlight_dismissals",
        ["household_owner_id", "login_user_id"],
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "
        f"{quoted_schema}.highlight_dismissals TO wealth_app"
    )
    op.execute(f"ALTER TABLE {quoted_schema}.highlight_dismissals ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY highlight_dismissals_personal_household_access
        ON {quoted_schema}.highlight_dismissals FOR ALL TO wealth_app
        USING (
            login_user_id = nullif(current_setting('app.user_id', true), '')::uuid
            AND household_owner_id = nullif(current_setting('app.household_owner', true), '')::uuid
            AND {quoted_schema}.app_household_role(household_owner_id) IS NOT NULL
        )
        WITH CHECK (
            login_user_id = nullif(current_setting('app.user_id', true), '')::uuid
            AND household_owner_id = nullif(current_setting('app.household_owner', true), '')::uuid
            AND {quoted_schema}.app_household_role(household_owner_id) IS NOT NULL
        )
        """
    )
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.audit_highlight_dismissal_write()
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
                NULLIF(current_setting('app.household_deleting_user', true), '') = actor_id::text OR
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
    op.execute(
        f"REVOKE ALL ON FUNCTION {quoted_schema}.audit_highlight_dismissal_write() FROM PUBLIC"
    )
    op.execute(
        f"GRANT EXECUTE ON FUNCTION {quoted_schema}.audit_highlight_dismissal_write() TO wealth_app"
    )
    op.execute(
        f"""
        CREATE TRIGGER trg_highlight_dismissals_household_audit
        AFTER INSERT OR UPDATE OR DELETE ON {quoted_schema}.highlight_dismissals
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.audit_highlight_dismissal_write()
        """
    )


def downgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_highlight_dismissals_household_audit "
        f"ON {quoted_schema}.highlight_dismissals"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.audit_highlight_dismissal_write()")
    op.drop_table("highlight_dismissals")
