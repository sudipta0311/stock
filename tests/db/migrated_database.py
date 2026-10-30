"""Create an isolated PostgreSQL schema and API login for database tests."""

from __future__ import annotations

import os
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url

from wealth.db.session import normalize_database_url


@dataclass(frozen=True)
class MigratedDatabase:
    owner_engine: Engine
    api_engine: Engine
    config: Config
    owner_role: str
    schema_name: str
    test_login: str
    api_database_url: str = field(repr=False)


def _drop_test_login(admin_engine: Engine, login: str, schema: str) -> None:
    with admin_engine.begin() as connection:
        role_exists = connection.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": login}
        ).scalar_one_or_none()
        if role_exists is None:
            return
        schema_exists = connection.execute(
            text("SELECT 1 FROM pg_namespace WHERE nspname = :schema"),
            {"schema": schema},
        ).scalar_one_or_none()
        if schema_exists is not None:
            connection.execute(text(f'REVOKE ALL ON SCHEMA "{schema}" FROM {login}'))
        connection.execute(text(f"REVOKE wealth_api FROM {login}"))
        connection.execute(text(f"DROP ROLE IF EXISTS {login}"))


@contextmanager
def create_migrated_database(
    *, schema_prefix: str = "c05", login_prefix: str = "wealth_api_test"
) -> Iterator[MigratedDatabase]:
    """Migrate one new schema and yield owner/API connections until cleanup."""
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        raise RuntimeError("TEST_DATABASE_URL is required to create a migrated database")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,20}", schema_prefix):
        raise ValueError("Invalid isolated schema prefix")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,20}", login_prefix):
        raise ValueError("Invalid temporary login prefix")

    schema_name = f"{schema_prefix}_{secrets.token_hex(16)}"
    test_login = f"{login_prefix}_{secrets.token_hex(16)}"
    test_password = secrets.token_urlsafe(32)
    normalized_url = normalize_database_url(database_url)
    admin_engine = create_engine(normalized_url, pool_pre_ping=True)
    owner_engine: Engine | None = None
    api_engine: Engine | None = None
    schema_created = False
    test_role_created = False
    try:
        with admin_engine.begin() as connection:
            owner_role = connection.execute(text("SELECT current_user")).scalar_one()
            connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
            schema_created = True

        scoped_url = make_url(normalized_url).update_query_dict(
            {"options": f"-csearch_path={schema_name}"}
        )
        owner_engine = create_engine(scoped_url, pool_pre_ping=True)
        root = Path(__file__).resolve().parents[2]
        config = Config(str(root / "alembic.ini"))
        rendered_url = scoped_url.render_as_string(hide_password=False)
        config.set_main_option("sqlalchemy.url", rendered_url.replace("%", "%%"))
        command.upgrade(config, "head")
        try:
            with admin_engine.begin() as connection:
                connection.execute(
                    text(
                        f"CREATE ROLE {test_login} LOGIN PASSWORD '{test_password}' "
                        "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                    )
                )
                test_role_created = True
                connection.execute(text(f"GRANT wealth_api TO {test_login}"))
                connection.execute(text(f'GRANT USAGE ON SCHEMA "{schema_name}" TO {test_login}'))
        except Exception:
            raise RuntimeError("Unable to create the temporary API test login") from None

        api_url = scoped_url.set(username=test_login, password=test_password)
        api_engine = create_engine(api_url, pool_pre_ping=True)
        yield MigratedDatabase(
            owner_engine=owner_engine,
            api_engine=api_engine,
            config=config,
            owner_role=owner_role,
            schema_name=schema_name,
            test_login=test_login,
            api_database_url=api_url.render_as_string(hide_password=False),
        )
    finally:
        if api_engine is not None:
            api_engine.dispose()
        if owner_engine is not None:
            owner_engine.dispose()
        try:
            if test_role_created:
                _drop_test_login(admin_engine, test_login, schema_name)
        except Exception:
            # Dropping the schema below can clear a grant that blocked role cleanup.
            pass
        finally:
            try:
                if schema_created:
                    with admin_engine.begin() as connection:
                        connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
            finally:
                try:
                    if test_role_created:
                        _drop_test_login(admin_engine, test_login, schema_name)
                finally:
                    admin_engine.dispose()
