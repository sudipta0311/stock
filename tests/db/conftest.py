"""Isolated PostgreSQL schema and least-privilege API login for RLS tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from tests.db.migrated_database import MigratedDatabase, create_migrated_database


@pytest.fixture(scope="module")
def migrated_database() -> Iterator[MigratedDatabase]:
    if not os.getenv("TEST_DATABASE_URL"):
        pytest.skip("TEST_DATABASE_URL is not set")
    with create_migrated_database() as database:
        yield database
