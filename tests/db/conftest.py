"""Isolated PostgreSQL schema and least-privilege API login for RLS tests."""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
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


@pytest.fixture(scope="module")
def migrated_database() -> Iterator[MigratedDatabase]:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set")

    schema_name = f"c05_{uuid4().hex}"
    test_login = f"wealth_api_test_{uuid4().hex}"
    test_password = secrets.token_urlsafe(32)
    normalized_url = normalize_database_url(database_url)
    admin_engine = create_engine(normalized_url, pool_pre_ping=True)
    with admin_engine.begin() as connection:
        owner_role = connection.execute(text("SELECT current_user")).scalar_one()
        connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))

    scoped_url = make_url(normalized_url).update_query_dict(
        {"options": f"-csearch_path={schema_name}"}
    )
    owner_engine = create_engine(scoped_url, pool_pre_ping=True)
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    rendered_url = scoped_url.render_as_string(hide_password=False)
    config.set_main_option("sqlalchemy.url", rendered_url.replace("%", "%%"))
    api_engine: Engine | None = None
    test_role_created = False
    try:
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
            owner_engine, api_engine, config, owner_role, schema_name, test_login
        )
    finally:
        if api_engine is not None:
            api_engine.dispose()
        owner_engine.dispose()
        with admin_engine.begin() as connection:
            if test_role_created:
                connection.execute(text(f'REVOKE ALL ON SCHEMA "{schema_name}" FROM {test_login}'))
                connection.execute(text(f"DROP ROLE {test_login}"))
            connection.execute(text(f'DROP SCHEMA "{schema_name}" CASCADE'))
        admin_engine.dispose()
