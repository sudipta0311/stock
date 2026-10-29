"""Database model and migration contract tests that do not need PostgreSQL."""

from __future__ import annotations

import ast
from pathlib import Path

from wealth.db.models import Base, ManualAsset, Profile
from wealth.db.session import make_engine, normalize_database_url

_SHARED_MARKET_TABLES = {
    "prices_eod",
    "nav_daily",
    "nav_monthly",
    "fx_rates",
    "gold_prices",
    "marketdata_runs",
    "research_packs",
    "sentinel_events",
    "fundamentals_pit",
    "universe_members",
}
_POST_INITIAL_TABLES = {
    "fund_lagging_checks",
    "fund_transactions",
    "household_access",
    "household_invites",
    "weekly_plans",
    "research_packs",
    "alerts_log",
    "import_templates",
    "realised_trades",
    "broker_cash_flows",
    "highlight_dismissals",
}
_NON_USER_ID_TABLES = {
    "household_access",
    "household_invites",
    "highlight_dismissals",
}


def test_all_user_tables_have_required_tenant_keys_and_indexes() -> None:
    for table in Base.metadata.sorted_tables:
        if table.name == "securities" or table.name in _SHARED_MARKET_TABLES:
            continue
        if table.name in _NON_USER_ID_TABLES:
            continue
        if table.name == "users":
            assert table.c.id.primary_key
            continue
        assert table.c.user_id.nullable is False
        assert table.primary_key.columns.contains_column(table.c.user_id) or any(
            not index.unique and "user_id" in {column.name for column in index.columns}
            for index in table.indexes
        )


def test_initial_migration_declares_policy_for_each_user_table() -> None:
    root = Path(__file__).resolve().parents[2]
    migration_path = root / "src" / "wealth" / "db" / "migrations" / "versions" / "0001_initial.py"
    source = migration_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"USER_TABLES", "RLS_POLICIES"}
    }

    expected = set(Base.metadata.tables) - {
        "securities",
        *_SHARED_MARKET_TABLES,
        *_POST_INITIAL_TABLES,
    }
    assert set(assignments["USER_TABLES"]) == expected
    assert assignments["RLS_POLICIES"] == tuple(
        f"{table_name}_isolation" for table_name in assignments["USER_TABLES"]
    )
    assert "CREATE POLICY {policy_name}" in source
    assert "ENABLE ROW LEVEL SECURITY" in source


def test_market_data_tables_are_shared_without_tenant_keys() -> None:
    for table_name in _SHARED_MARKET_TABLES:
        table = Base.metadata.tables[table_name]
        assert "user_id" not in table.c


def test_make_engine_normalizes_supported_postgres_urls() -> None:
    assert normalize_database_url("postgres://localhost/wealth") == (
        "postgresql+psycopg://localhost/wealth"
    )
    assert normalize_database_url("postgresql://localhost/wealth") == (
        "postgresql+psycopg://localhost/wealth"
    )
    engine = make_engine("postgresql+psycopg://localhost/wealth")
    try:
        assert engine.url.drivername == "postgresql+psycopg"
    finally:
        engine.dispose()


def test_money_map_columns_are_in_the_orm_metadata() -> None:
    assert Profile.__table__.c.checklist.nullable is False
    assert str(Profile.__table__.c.checklist.server_default.arg) == "'[]'::jsonb"
    assert ManualAsset.__table__.c.valuation.nullable is True
