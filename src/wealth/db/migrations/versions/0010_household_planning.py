"""Move tax residence and member ownership onto household members."""

from __future__ import annotations

from alembic import op
from sqlalchemy import Boolean, Column, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB

revision = "0010_household_planning"
down_revision = "0009_residency"
branch_labels = None
depends_on = None

_DEFAULT_RESIDENCY = (
    "jsonb_build_object('status', 'RI', 'country', 'IN', 'us_person', false, "
    "'planned_return_year', null, 'settle_plan', 'undecided')"
)
_RELATIONSHIPS = "'self', 'spouse', 'child', 'parent', 'other'"


def upgrade() -> None:
    op.add_column(
        "household_members",
        Column("is_self", Boolean(), nullable=False, server_default=text("false")),
    )
    op.add_column(
        "household_members",
        Column("residency", JSONB(), nullable=False, server_default=text(_DEFAULT_RESIDENCY)),
    )
    op.add_column(
        "household_members",
        Column("tax", JSONB(), nullable=False, server_default=text("'{}'::jsonb")),
    )
    op.add_column(
        "household_members",
        Column("investing", JSONB(), nullable=False, server_default=text("'{}'::jsonb")),
    )
    # Relationship was free text (≤ 40 chars) before this card: normalise it before the
    # CHECK constraint is created, or existing rows would make the migration fail.
    op.execute(
        """
        UPDATE household_members SET relationship = CASE
            WHEN lower(btrim(relationship)) IN ('self', 'me', 'myself') THEN 'self'
            WHEN lower(btrim(relationship)) IN ('spouse', 'wife', 'husband', 'partner')
                THEN 'spouse'
            WHEN lower(btrim(relationship)) IN ('child', 'son', 'daughter', 'kid') THEN 'child'
            WHEN lower(btrim(relationship)) IN ('parent', 'mother', 'father', 'mom', 'dad')
                THEN 'parent'
            ELSE 'other'
        END
        """
    )
    op.create_check_constraint(
        "ck_household_members_relationship",
        "household_members",
        f"relationship IN ({_RELATIONSHIPS})",
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT id, row_number() OVER (
                PARTITION BY user_id ORDER BY created_at, id
            ) AS position
            FROM household_members
            WHERE relationship = 'self'
        )
        UPDATE household_members AS member
        SET is_self = ranked.position = 1,
            relationship = CASE WHEN ranked.position = 1 THEN 'self' ELSE 'other' END
        FROM ranked
        WHERE member.id = ranked.id
        """
    )
    op.execute(
        f"""
        INSERT INTO household_members
            (id, user_id, display_name, relationship, is_self, residency, tax, investing)
        SELECT gen_random_uuid(), users.id, COALESCE(NULLIF(users.display_name, ''), 'You'),
               'self', true, {_DEFAULT_RESIDENCY}, '{{}}'::jsonb, '{{}}'::jsonb
        FROM users
        WHERE NOT EXISTS (
            SELECT 1 FROM household_members AS member
            WHERE member.user_id = users.id AND member.is_self
        )
        """
    )
    op.create_index(
        "uq_household_members_one_self_per_user",
        "household_members",
        ["user_id"],
        unique=True,
        postgresql_where=text("is_self"),
    )
    op.create_check_constraint(
        "ck_household_members_self_relationship",
        "household_members",
        "is_self = (relationship = 'self')",
    )

    op.execute(
        """
        UPDATE household_members AS member
        SET residency = COALESCE(profile.residency, member.residency),
            tax = jsonb_strip_nulls(jsonb_build_object(
                'tax_regime', profile.tax_regime,
                'tax_slab_pct', profile.tax_slab_pct,
                'birth_year', profile.investing -> 'birth_year',
                'annual_income', profile.financial_context -> 'annual_income',
                'employment_type', profile.financial_context -> 'employment_type'
            )),
            investing = COALESCE(profile.investing, '{}'::jsonb)
        FROM profiles AS profile
        WHERE member.user_id = profile.user_id AND member.is_self
        """
    )

    for table in ("accounts", "manual_assets", "holdings_lots"):
        op.add_column(
            table,
            Column("joint", Boolean(), nullable=False, server_default=text("false")),
        )

    op.execute(
        """
        UPDATE accounts AS account
        SET member_id = member.id
        FROM household_members AS member
        WHERE account.user_id = member.user_id AND member.is_self AND account.member_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE manual_assets AS asset
        SET member_id = COALESCE(
                (SELECT account.member_id FROM accounts AS account
                 WHERE account.user_id = asset.user_id AND account.id = asset.account_id),
                member.id
            ),
            joint = CASE
                WHEN asset.details ->> 'joint' IN ('true', 'false')
                THEN (asset.details ->> 'joint')::boolean
                ELSE false
            END
        FROM household_members AS member
        WHERE asset.user_id = member.user_id AND member.is_self AND asset.member_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE holdings_lots AS lot
        SET member_id = COALESCE(
                (SELECT account.member_id FROM accounts AS account
                 WHERE account.user_id = lot.user_id AND account.id = lot.account_id),
                member.id
            ),
            joint = COALESCE(
                (SELECT account.joint FROM accounts AS account
                 WHERE account.user_id = lot.user_id AND account.id = lot.account_id),
                false
            )
        FROM household_members AS member
        WHERE lot.user_id = member.user_id AND member.is_self AND lot.member_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE manual_assets
        SET joint = CASE
            WHEN details ->> 'joint' IN ('true', 'false') THEN (details ->> 'joint')::boolean
            ELSE false
        END
        """
    )
    op.execute(
        """
        UPDATE holdings_lots AS lot
        SET joint = COALESCE(
            (SELECT account.joint FROM accounts AS account
             WHERE account.user_id = lot.user_id AND account.id = lot.account_id),
            false
        )
        """
    )

    for table in ("accounts", "manual_assets", "holdings_lots"):
        op.alter_column(table, "member_id", nullable=False)
        op.create_index(f"ix_{table}_member_id", table, ["user_id", "member_id"])

    op.add_column("goals", Column("member_id", Uuid(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_goals_member_tenant",
        "goals",
        "household_members",
        ["user_id", "member_id"],
        ["user_id", "id"],
        ondelete="RESTRICT",
    )
    op.create_index("ix_goals_member_id", "goals", ["user_id", "member_id"])

    _install_self_member_triggers()


def _install_self_member_triggers() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.create_self_household_member()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        BEGIN
            INSERT INTO {quoted_schema}.household_members
                (id, user_id, display_name, relationship, is_self, residency, tax, investing)
            VALUES (
                pg_catalog.gen_random_uuid(), NEW.id,
                COALESCE(NULLIF(NEW.display_name, ''), 'You'), 'self', true,
                {_DEFAULT_RESIDENCY}, '{{}}'::jsonb, '{{}}'::jsonb
            )
            ON CONFLICT (user_id) WHERE is_self DO NOTHING;
            RETURN NEW;
        END;
        $function$
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER trg_users_create_self_member
        AFTER INSERT ON {quoted_schema}.users
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.create_self_household_member()
        """
    )
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.prevent_self_member_update()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, {quoted_schema}
        AS $function$
        BEGIN
            IF OLD.is_self AND (
                NOT NEW.is_self OR NEW.relationship <> 'self' OR NEW.user_id <> OLD.user_id
            ) THEN
                RAISE EXCEPTION 'the self household member identity cannot be changed';
            END IF;
            RETURN NEW;
        END;
        $function$
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER trg_household_members_prevent_self_update
        BEFORE UPDATE ON {quoted_schema}.household_members
        FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.prevent_self_member_update()
        """
    )
    op.execute(
        f"""
        CREATE FUNCTION {quoted_schema}.default_self_member_for_owner()
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
    for table in ("accounts", "manual_assets", "holdings_lots"):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_default_self_member
            BEFORE INSERT ON {quoted_schema}.{table}
            FOR EACH ROW EXECUTE FUNCTION {quoted_schema}.default_self_member_for_owner()
            """
        )


def downgrade() -> None:
    connection = op.get_bind()
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    quoted_schema = connection.dialect.identifier_preparer.quote_schema(schema)
    op.execute(
        f"DROP TRIGGER IF EXISTS trg_household_members_prevent_self_update "
        f"ON {quoted_schema}.household_members"
    )
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.prevent_self_member_update()")
    op.execute(f"DROP TRIGGER IF EXISTS trg_users_create_self_member ON {quoted_schema}.users")
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.create_self_household_member()")
    for table in ("accounts", "manual_assets", "holdings_lots"):
        op.execute(
            f"DROP TRIGGER IF EXISTS trg_{table}_default_self_member ON {quoted_schema}.{table}"
        )
    op.execute(f"DROP FUNCTION IF EXISTS {quoted_schema}.default_self_member_for_owner()")

    op.drop_index("ix_goals_member_id", table_name="goals")
    op.drop_constraint("fk_goals_member_tenant", "goals", type_="foreignkey")
    op.drop_column("goals", "member_id")
    for table in ("accounts", "manual_assets", "holdings_lots"):
        op.drop_index(f"ix_{table}_member_id", table_name=table)
        op.alter_column(table, "member_id", nullable=True)
        op.drop_column(table, "joint")
    op.drop_constraint("ck_household_members_self_relationship", "household_members", type_="check")
    op.drop_index("uq_household_members_one_self_per_user", table_name="household_members")
    op.drop_constraint("ck_household_members_relationship", "household_members", type_="check")
    op.drop_column("household_members", "investing")
    op.drop_column("household_members", "tax")
    op.drop_column("household_members", "residency")
    op.drop_column("household_members", "is_self")
