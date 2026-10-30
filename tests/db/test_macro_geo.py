from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import ResearchPackRecord
from wealth.db.session import resolve_user, user_session
from wealth.marketdata import store
from wealth.marketdata.sources.macro_geo import (
    FeedItemObservation,
    IndexValuationObservation,
    MacroSeriesObservation,
)
from wealth.policy.data import _research_index_percentiles, load_policy_inputs
from wealth.policy.global_split import GeoTilts
from wealth.research.graph import build_research_graph
from wealth.research.router import FakeRouter, RunBudget, RunContext
from wealth.research.store import (
    load_prior_geo_tilts,
    save_macro_geo_pack,
    save_research_pack,
)
from wealth.research.tools import ToolRegistry

pytestmark = pytest.mark.db


def test_index_percentile_uses_history_and_labels_short_history(
    migrated_database: MigratedDatabase,
) -> None:
    old_dates = [date(2022, 10, 29), date(2024, 10, 29), date(2025, 10, 29)]
    records = [
        IndexValuationObservation(
            "NIFTY 50",
            on,
            Decimal(value),
            Decimal("3.0"),
            Decimal("1.0"),
            "NSE",
            "https://nsearchives.nseindia.com/synthetic.csv",
        )
        for on, value in zip(old_dates, ("10", "20", "30"), strict=True)
    ]
    records.append(
        IndexValuationObservation(
            "NIFTY 50",
            date(2026, 10, 29),
            Decimal("25"),
            Decimal("3.0"),
            Decimal("1.0"),
            "NSE",
            "https://nsearchives.nseindia.com/synthetic.csv",
        )
    )
    records.extend(
        [
            IndexValuationObservation(
                "NIFTY 500",
                date(2026, 10, 28),
                Decimal("18"),
                None,
                None,
                "NSE",
                "https://nsearchives.nseindia.com/synthetic.csv",
            ),
            IndexValuationObservation(
                "NIFTY 500",
                date(2026, 10, 29),
                Decimal("19"),
                None,
                None,
                "NSE",
                "https://nsearchives.nseindia.com/synthetic.csv",
            ),
        ]
    )

    with Session(migrated_database.owner_engine) as session, session.begin():
        assert store.upsert_index_valuations(session, records) == len(records)
        long_history = store.latest_index_valuation_percentile(
            session, "NIFTY 50", date(2026, 10, 30)
        )
        short_history = store.latest_index_valuation_percentile(
            session, "NIFTY 500", date(2026, 10, 30)
        )

    assert long_history is not None
    assert long_history.percentile == Decimal("66.67")
    # The macro agent's index_valuation tool reports these too (it crashed without them).
    assert (long_history.pb, long_history.div_yield) == (Decimal("3.0"), Decimal("1.0"))
    assert long_history.history_label == "long_history"
    assert long_history.history_count == 4
    assert short_history is not None
    assert short_history.percentile == Decimal("100.00")
    assert short_history.history_label == "short_history"

    from wealth.research.agents.macro_geo import MacroGeoToolbox

    tool_result, source_url = MacroGeoToolbox(
        migrated_database.owner_engine, date(2026, 10, 30)
    ).index_valuation({"index": "NIFTY 50"})
    assert (tool_result["pb"], tool_result["div_yield"]) == ("3.00000000", "1.00000000")
    assert source_url == "https://nsearchives.nseindia.com/synthetic.csv"


def test_weekly_pack_merges_macro_fields_and_policy_loads_tilts_and_percentiles(
    migrated_database: MigratedDatabase,
) -> None:
    week = "2026-W44"
    as_of = date(2026, 10, 30)
    tilts = GeoTilts(
        global_pp=Decimal("2"),
        gold_pp=Decimal("1"),
        evidence_ids=["source-one", "source-two"],
        as_of=as_of,
    )
    save_research_pack(
        migrated_database.owner_engine,
        week,
        {"ideas": [], "summary": "Synthetic weekly summary.", "run_id": str(uuid4())},
    )
    save_macro_geo_pack(
        migrated_database.owner_engine,
        week,
        {
            "geo_tilts": tilts.model_dump(mode="json"),
            "macro": {"agent_status": "completed", "regime_notes": []},
            "index_valuation_percentiles": {"IN": "72.50", "US": None},
        },
    )
    user_id = resolve_user(
        migrated_database.owner_engine,
        f"synthetic-macro-{uuid4()}",
        f"synthetic-{uuid4()}@example.test",
        "Synthetic Test User",
    )

    with user_session(migrated_database.owner_engine, user_id) as session:
        user_data, market, pack = load_policy_inputs(session, user_id, week, as_of)
        assert user_data is not None
        assert pack is not None
        assert pack.geo_tilts == tilts
        assert market.index_valuation_percentiles == {"IN": Decimal("72.50")}

    assert load_prior_geo_tilts(migrated_database.owner_engine, "2026-W45") == tilts
    with Session(migrated_database.owner_engine) as session:
        row = session.get(ResearchPackRecord, week)
        assert row is not None
        assert row.pack["macro"]["agent_status"] == "completed"
        assert row.pack["index_valuation_percentiles"] == {"IN": "72.50", "US": None}
        assert "macro" not in ResearchPackRecord.__table__.columns
        assert _research_index_percentiles(session, week) == {"IN": Decimal("72.50")}


def test_macro_market_data_tables_have_shared_read_only_grants(
    migrated_database: MigratedDatabase,
) -> None:
    tables = ("index_valuations", "macro_series", "feed_items")
    with migrated_database.owner_engine.connect() as connection:
        for table in tables:
            relation = f"{migrated_database.schema_name}.{table}"
            app_privileges = connection.execute(
                text(
                    "SELECT has_table_privilege('wealth_app', :relation, 'SELECT'), "
                    "has_table_privilege('wealth_app', :relation, 'INSERT'), "
                    "has_table_privilege('wealth_app', :relation, 'UPDATE'), "
                    "has_table_privilege('wealth_app', :relation, 'DELETE')"
                ),
                {"relation": relation},
            ).one()
            owner_writes = connection.execute(
                text(
                    "SELECT has_table_privilege(current_user, :relation, 'INSERT'), "
                    "has_table_privilege(current_user, :relation, 'UPDATE'), "
                    "has_table_privilege(current_user, :relation, 'DELETE')"
                ),
                {"relation": relation},
            ).one()
            directly_granted_to_api = connection.scalar(
                text(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_class AS c "
                    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                    "CROSS JOIN LATERAL aclexplode("
                    "COALESCE(c.relacl, acldefault('r', c.relowner))) AS acl "
                    "WHERE n.nspname = :schema AND c.relname = :table "
                    "AND acl.grantee = 'wealth_api'::regrole)"
                ),
                {"schema": migrated_database.schema_name, "table": table},
            )
            row_level_security = connection.scalar(
                text("SELECT relrowsecurity FROM pg_class WHERE oid = to_regclass(:relation)"),
                {"relation": relation},
            )

            assert app_privileges == (True, False, False, False)
            assert owner_writes == (True, True, True)
            assert directly_granted_to_api is False
            assert row_level_security is False


def test_weekly_research_graph_merges_macro_agent_output_into_pack(
    migrated_database: MigratedDatabase,
) -> None:
    week = "2026-W44"
    run_id = str(uuid4())
    router = FakeRouter()
    graph = build_research_graph(
        router,
        run_ctx=RunContext(
            budget=RunBudget(max_usd=Decimal("1"), max_calls=20),
            run_id=run_id,
            owner_engine=migrated_database.owner_engine,
        ),
        owner_engine=migrated_database.owner_engine,
        tools=ToolRegistry(migrated_database.owner_engine),
        shortlist=[],
        interrupt_after_screen=False,
        interrupt_after_red_team=False,
    )

    state = graph.invoke({"run_id": run_id, "week": week, "shortlist": []})

    assert "geo_tilts" in state["pack"]
    assert "macro" in state["pack"]
    assert state["pack"]["macro"]["agent_status"] == "completed"
    with Session(migrated_database.owner_engine) as session:
        row = session.get(ResearchPackRecord, week)
        assert row is not None
        assert row.pack["geo_tilts"]["global_pp"] == "0"
        assert row.pack["macro"]["agent_status"] == "completed"


def test_macro_market_rows_round_trip_and_feed_pruning(
    migrated_database: MigratedDatabase,
) -> None:
    fetched_at = datetime(2026, 10, 30, tzinfo=UTC)
    with Session(migrated_database.owner_engine) as session, session.begin():
        assert (
            store.upsert_macro_series(
                session,
                [
                    MacroSeriesObservation(
                        "UST_10Y",
                        date(2026, 10, 29),
                        Decimal("4.25"),
                        "US Treasury",
                        "https://home.treasury.gov/synthetic-feed",
                    )
                ],
            )
            == 1
        )
        assert (
            store.upsert_feed_items(
                session,
                [
                    FeedItemObservation(
                        "pib",
                        "synthetic-old-item",
                        "Old synthetic item",
                        "https://pib.gov.in/synthetic-old",
                        None,
                        "Stored only for the pruning test.",
                        fetched_at,
                    ),
                    FeedItemObservation(
                        "rbi",
                        "synthetic-current-item",
                        "Current synthetic item",
                        "https://rbi.org.in/synthetic-current",
                        None,
                        "Short retained summary.",
                        fetched_at,
                    ),
                ],
            )
            == 2
        )
        assert store.prune_feed_items(session, date(2026, 10, 30), retention_days=120) == 0
        assert store.prune_feed_items(session, date(2027, 3, 1), retention_days=120) == 2
        history = store.macro_series_history(session, "UST_10Y", date(2026, 10, 30))
        assert history[0].value == Decimal("4.25000000")
