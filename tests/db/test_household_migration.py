"""Data migration checks for 0010_household_planning."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import text

from conftest import MigratedDatabase

pytestmark = pytest.mark.db


def test_free_text_relationships_are_normalised_before_the_check_constraint(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0009_residency")
    with_members, without_members = uuid4(), uuid4()
    residency = {"status": "NRI", "country": "DK", "us_person": False}
    legacy = {"Self": "self", " Wife ": "spouse", "Mother": "parent", "son": "child"}
    legacy["Sibling"] = "other"
    with migrated_database.owner_engine.begin() as connection:
        for user_id, label in ((with_members, "a"), (without_members, "b")):
            connection.execute(
                text(
                    "INSERT INTO users (id, email, display_name, auth_subject) "
                    "VALUES (:id, :email, 'Synthetic', :sub)"
                ),
                {"id": user_id, "email": f"migration-{label}@example.test", "sub": f"mig-{label}"},
            )
        connection.execute(
            text(
                "INSERT INTO profiles (id, user_id, residency) "
                "VALUES (:id, :user_id, CAST(:residency AS jsonb))"
            ),
            {"id": uuid4(), "user_id": with_members, "residency": json.dumps(residency)},
        )
        for relationship in legacy:
            connection.execute(
                text(
                    "INSERT INTO household_members (id, user_id, display_name, relationship) "
                    "VALUES (:id, :user_id, 'Member', :relationship)"
                ),
                {"id": uuid4(), "user_id": with_members, "relationship": relationship},
            )

    command.upgrade(migrated_database.config, "head")

    with migrated_database.owner_engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT relationship, is_self, residency FROM household_members "
                "WHERE user_id = :user_id ORDER BY relationship"
            ),
            {"user_id": with_members},
        ).all()
        created_self = connection.execute(
            text(
                "SELECT count(*) FROM household_members "
                "WHERE user_id = :user_id AND is_self AND relationship = 'self'"
            ),
            {"user_id": without_members},
        ).scalar_one()

    assert sorted(row.relationship for row in rows) == sorted(legacy.values())
    self_rows = [row for row in rows if row.is_self]
    assert len(self_rows) == 1
    assert self_rows[0].residency["country"] == "DK"
    assert created_self == 1
