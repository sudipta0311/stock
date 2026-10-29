from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from tests.db.conftest import MigratedDatabase

pytestmark = pytest.mark.db


def test_only_dossiers_are_selectable_by_api_roles_and_private_tables_are_inaccessible(
    migrated_database: MigratedDatabase,
) -> None:
    private_tables = ("research_runs", "llm_calls", "tool_calls")
    privileges = ("SELECT", "INSERT", "UPDATE", "DELETE")
    with migrated_database.owner_engine.connect() as connection:
        for role in ("wealth_app", "wealth_api"):
            dossier_select = connection.scalar(
                text("SELECT has_table_privilege(:role, :table_name, 'SELECT')"),
                {"role": role, "table_name": f"{migrated_database.schema_name}.dossiers"},
            )
            assert dossier_select is True
            for privilege in privileges[1:]:
                allowed = connection.scalar(
                    text("SELECT has_table_privilege(:role, :table_name, :privilege)"),
                    {
                        "role": role,
                        "table_name": f"{migrated_database.schema_name}.dossiers",
                        "privilege": privilege,
                    },
                )
                assert allowed is False
            for table_name in private_tables:
                for privilege in privileges:
                    allowed = connection.scalar(
                        text("SELECT has_table_privilege(:role, :table_name, :privilege)"),
                        {
                            "role": role,
                            "table_name": f"{migrated_database.schema_name}.{table_name}",
                            "privilege": privilege,
                        },
                    )
                    assert allowed is False
        for table_name in (*private_tables, "dossiers"):
            row_security_enabled = connection.scalar(
                text(
                    "SELECT relrowsecurity FROM pg_catalog.pg_class "
                    "WHERE oid = pg_catalog.to_regclass(:table_name)"
                ),
                {"table_name": f"{migrated_database.schema_name}.{table_name}"},
            )
            assert row_security_enabled is False

    update_columns = {
        "research_runs": "run_id",
        "llm_calls": "id",
        "tool_calls": "tool_call_id",
    }
    for table_name, column_name in update_columns.items():
        qualified_name = f'"{migrated_database.schema_name}"."{table_name}"'
        statements = (
            f"SELECT * FROM {qualified_name} LIMIT 0",
            f"INSERT INTO {qualified_name} DEFAULT VALUES",
            f"UPDATE {qualified_name} SET {column_name} = {column_name} WHERE false",
            f"DELETE FROM {qualified_name} WHERE false",
        )
        with migrated_database.api_engine.connect() as connection:
            for statement in statements:
                with pytest.raises(DBAPIError):
                    connection.execute(text(statement))
                connection.rollback()
