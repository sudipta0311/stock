"""PostgreSQL API-role, identity bootstrap, and row-level security tests."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import DBAPIError

from conftest import MigratedDatabase
from wealth.db.models import Account, HouseholdMember, ManualAsset, User
from wealth.db.repo import add_manual_asset, list_manual_assets
from wealth.db.session import resolve_user, system_session, user_session
from wealth.schemas.assets import parse_asset

pytestmark = pytest.mark.db


def test_api_role_bootstrap_and_reset_role_remain_tenant_scoped(
    migrated_database: MigratedDatabase,
) -> None:
    database = migrated_database
    with database.owner_engine.connect() as connection:
        role = connection.execute(
            text(
                """
                SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolreplication,
                       rolbypassrls, rolinherit,
                       pg_catalog.pg_has_role(
                           'wealth_api'::name, CAST(:owner_role AS name), 'MEMBER'
                       ) AS is_owner_member,
                       pg_catalog.pg_has_role(
                           'wealth_api'::name, 'wealth_app'::name, 'MEMBER'
                       ) AS is_app_member
                FROM pg_catalog.pg_roles
                WHERE rolname = 'wealth_api'
                """
            ),
            {"owner_role": database.owner_role},
        ).one()
        owned_objects = connection.scalar(
            text(
                """
                SELECT count(*)
                FROM pg_catalog.pg_shdepend AS dependency
                WHERE dependency.refclassid = 'pg_catalog.pg_authid'::pg_catalog.regclass
                  AND dependency.refobjid = (
                      SELECT oid FROM pg_catalog.pg_roles WHERE rolname = 'wealth_api'
                  )
                  AND dependency.deptype = 'o'
                """
            )
        )
        function = connection.execute(
            text(
                """
                SELECT pg_catalog.pg_get_userbyid(procedure.proowner) AS function_owner,
                       pg_catalog.pg_get_function_result(procedure.oid) AS result_type,
                       procedure.pronargs,
                       procedure.proconfig,
                       EXISTS (
                           SELECT 1
                           FROM pg_catalog.aclexplode(procedure.proacl) AS acl
                           WHERE acl.grantee = 0 AND acl.privilege_type = 'EXECUTE'
                       ) AS public_can_execute,
                       pg_catalog.has_function_privilege(
                           'wealth_api', procedure.oid, 'EXECUTE'
                       ) AS api_can_execute
                FROM pg_catalog.pg_proc AS procedure
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = procedure.pronamespace
                WHERE namespace.nspname = :schema_name
                  AND procedure.proname = 'app_resolve_user'
                """
            ),
            {"schema_name": database.schema_name},
        ).one()

    assert not role.rolsuper
    assert not role.rolcreatedb
    assert not role.rolcreaterole
    assert not role.rolreplication
    assert not role.rolbypassrls
    assert role.rolinherit
    assert not role.is_owner_member
    assert role.is_app_member
    assert owned_objects == 0
    assert function.function_owner == database.owner_role
    assert function.result_type == "uuid"
    assert function.pronargs == 3
    assert any(
        setting.startswith("search_path=pg_catalog,") and database.schema_name in setting
        for setting in function.proconfig
    )
    assert not function.public_can_execute
    assert function.api_can_execute

    subject_a = f"test-subject-{uuid4()}"
    subject_b = f"test-subject-{uuid4()}"
    user_a_id = resolve_user(
        database.api_engine, subject_a, "First.User@Example.test", "Synthetic User A"
    )
    same_user_id = resolve_user(
        database.api_engine, subject_a, "Updated.User@Example.test", "Updated User A"
    )
    user_b_id = resolve_user(
        database.api_engine, subject_b, "Second.User@Example.test", "Synthetic User B"
    )
    assert user_a_id == same_user_id
    assert user_a_id != user_b_id

    with database.api_engine.begin() as connection:
        result = connection.execute(
            text(
                "SELECT app_resolve_user(:auth_subject, :email, :display_name) AS resolved_user_id"
            ),
            {
                "auth_subject": subject_a,
                "email": "Updated.User@Example.test",
                "display_name": "Updated User A",
            },
        )
        assert result.keys() == ["resolved_user_id"]
        assert result.scalar_one() == user_a_id

    with user_session(database.api_engine, user_a_id) as session:
        assert session.execute(
            select(User.email, User.display_name).where(User.id == user_a_id)
        ).one() == ("updated.user@example.test", "Updated User A")
        session.execute(text("RESET ROLE"))
        assert session.scalar(select(func.count()).select_from(User)) == 1
        assert session.scalar(select(User.id).where(User.id == user_b_id)) is None
        assert (
            session.execute(
                select(User.email, User.auth_subject).where(User.id == user_b_id)
            ).first()
            is None
        )
        with pytest.raises(DBAPIError), session.begin_nested():
            session.execute(
                text("SELECT app_resolve_user(:auth_subject, :email, :display_name)"),
                {
                    "auth_subject": subject_b,
                    "email": "Second.User@Example.test",
                    "display_name": "Synthetic User B",
                },
            )
        owner_identifier = session.connection().dialect.identifier_preparer.quote_identifier(
            database.owner_role
        )
        with pytest.raises(DBAPIError), session.begin_nested():
            session.execute(text(f"SET ROLE {owner_identifier}"))
        assert session.scalar(select(func.count()).select_from(User)) == 1

    with database.owner_engine.connect() as connection, connection.begin():
        assert connection.scalar(select(func.count()).select_from(User)) == 2

    with database.api_engine.connect() as connection, connection.begin():
        assert connection.scalar(select(func.count()).select_from(User)) == 0

    with system_session(database.owner_engine) as session:
        member_a = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user_a_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert member_a is not None
        account_a = Account(
            user_id=user_a_id,
            member_id=member_a.id,
            institution="Synthetic Bank A",
            account_type="bank",
            masked_number="X1234",
        )
        session.add(account_a)
        session.flush()
        asset_a = add_manual_asset(
            session,
            user_a_id,
            parse_asset(
                {
                    "asset_type": "savings",
                    "bank": "Synthetic Bank A",
                    "balance": "125000",
                    "member_id": str(member_a.id),
                }
            ),
            account_id=account_a.id,
            value_cached=Decimal("125000.00"),
        )
        asset_b = add_manual_asset(
            session,
            user_b_id,
            parse_asset({"asset_type": "savings", "bank": "Synthetic Bank B", "balance": "25000"}),
            value_cached=Decimal("25000.00"),
        )
        asset_a_id = asset_a.id
        asset_b_id = asset_b.id

    with pytest.raises(DBAPIError), user_session(database.api_engine, user_a_id) as session:
        session.add(
            ManualAsset(
                user_id=user_b_id,
                asset_type="savings",
                details={"asset_type": "savings", "bank": "Synthetic", "balance": "1.00"},
            )
        )
        session.flush()

    with user_session(database.api_engine, user_a_id) as session:
        assert [asset.id for asset in list_manual_assets(session, user_a_id)] == [asset_a_id]
        session.execute(text("RESET ROLE"))
        assert session.scalar(select(func.count()).select_from(User)) == 1
        assert session.scalar(select(ManualAsset.id).where(ManualAsset.id == asset_b_id)) is None
        assert (
            session.execute(
                update(ManualAsset).where(ManualAsset.id == asset_b_id).values(status="closed")
            ).rowcount
            == 0
        )
        assert (
            session.execute(delete(ManualAsset).where(ManualAsset.id == asset_b_id)).rowcount == 0
        )
        assert session.execute(select(User.email).where(User.id == user_b_id)).first() is None
        owner_identifier = session.connection().dialect.identifier_preparer.quote_identifier(
            database.owner_role
        )
        with pytest.raises(DBAPIError), session.begin_nested():
            session.execute(text(f"SET ROLE {owner_identifier}"))
        assert session.scalar(select(func.count()).select_from(ManualAsset)) == 1

    with system_session(database.owner_engine) as session:
        assert session.scalar(select(func.count()).select_from(User)) == 2
        assert session.scalar(select(func.count()).select_from(ManualAsset)) == 2
        assert session.execute(delete(User).where(User.id == user_a_id)).rowcount == 1

    with system_session(database.owner_engine) as session:
        assert session.scalar(select(func.count()).select_from(User)) == 1
        assert session.scalar(select(func.count()).select_from(ManualAsset)) == 1
        assert session.scalar(select(func.count()).select_from(Account)) == 0
        remaining_member = session.scalar(select(HouseholdMember))
        assert remaining_member is not None
        assert remaining_member.user_id == user_b_id
        assert session.scalar(select(func.count()).select_from(HouseholdMember)) == 1

    command.downgrade(database.config, "base")
    command.upgrade(database.config, "head")


def test_migrations_preserve_an_enabled_api_role_login(
    migrated_database: MigratedDatabase,
) -> None:
    database = migrated_database
    with database.owner_engine.connect() as connection:
        original_login = connection.scalar(
            text("SELECT rolcanlogin FROM pg_catalog.pg_roles WHERE rolname = 'wealth_api'")
        )

    with database.owner_engine.begin() as connection:
        connection.execute(text("ALTER ROLE wealth_api LOGIN"))
    try:
        command.downgrade(database.config, "base")
        command.upgrade(database.config, "head")
        with database.owner_engine.connect() as connection:
            assert connection.scalar(
                text("SELECT rolcanlogin FROM pg_catalog.pg_roles WHERE rolname = 'wealth_api'")
            )
    finally:
        with database.owner_engine.begin() as connection:
            if original_login:
                connection.execute(text("ALTER ROLE wealth_api LOGIN"))
            else:
                connection.execute(text("ALTER ROLE wealth_api NOLOGIN"))
