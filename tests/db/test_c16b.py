from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from alembic import command
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import NavDaily, PaperPosition, PriceEod, Security
from wealth.marketdata.sources.macro_geo import IndexValuationObservation
from wealth.marketdata.store import upsert_index_valuations
from wealth.research.etf_selector import CategoryRule, SelectorRules, select_core_picks
from wealth.research.scorekeeper import run_scorekeeper
from wealth.research.store import load_research_pack, save_pack_fields, save_research_pack

pytestmark = pytest.mark.db


def _selector_rules() -> SelectorRules:
    return SelectorRules(
        categories=(
            CategoryRule(
                id="in_nifty50",
                label="India large companies",
                market="IN",
                security_types=("etf", "mf"),
                index_name="NIFTY 50",
                include_any=("nifty 50",),
                exclude_any=("nifty next 50",),
            ),
        ),
        history_days=90,
        minimum_history_days=20,
        liquidity_floor_inr=Decimal("500"),
        liquidity_full_score_inr=Decimal("10000"),
        premium_disqualify_pct=Decimal("1"),
        weights={
            "liquidity": Decimal("30"),
            "premium": Decimal("25"),
            "tracking_difference": Decimal("25"),
            "expense_ratio": Decimal("15"),
            "assets": Decimal("5"),
        },
    )


def _add_etf(
    session: Session,
    *,
    symbol: str,
    as_of: date,
    traded_value: Decimal,
    premium_pct: Decimal = Decimal("0"),
) -> Security:
    security = Security(
        market="IN",
        symbol=symbol,
        name=f"Synthetic Nifty 50 Fund {symbol}",
        security_type="etf",
        currency="INR",
        nse_symbol=symbol,
    )
    session.add(security)
    session.flush()
    nav = Decimal("100")
    close = nav * (Decimal("1") + premium_pct / Decimal("100"))
    session.add(NavDaily(security_id=security.id, as_of=as_of, nav=nav, source="synthetic"))
    for day in range(20):
        session.add(
            PriceEod(
                security_id=security.id,
                as_of=as_of - timedelta(days=day),
                close=close,
                volume=1000,
                traded_value=traded_value,
                source="synthetic",
                source_url="https://example.test/prices",
            )
        )
    return security


def test_selector_returns_deterministic_top_two_and_excludes_high_premium(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 10, 30)
    with Session(migrated_database.owner_engine) as session, session.begin():
        _add_etf(session, symbol="SYN-A", as_of=as_of, traded_value=Decimal("8000"))
        _add_etf(session, symbol="SYN-B", as_of=as_of, traded_value=Decimal("9000"))
        _add_etf(session, symbol="SYN-C", as_of=as_of, traded_value=Decimal("6000"))
        _add_etf(
            session,
            symbol="SYN-D",
            as_of=as_of,
            traded_value=Decimal("20000"),
            premium_pct=Decimal("1.5"),
        )
        upsert_index_valuations(
            session,
            [
                IndexValuationObservation(
                    index_name="NIFTY 50",
                    as_of=as_of,
                    pe=Decimal("22"),
                    pb=Decimal("3"),
                    div_yield=Decimal("1"),
                    source="synthetic",
                    source_url="https://example.test/index",
                    close=Decimal("25000"),
                )
            ],
        )

    first = select_core_picks(migrated_database.owner_engine, as_of=as_of, rules=_selector_rules())
    second = select_core_picks(migrated_database.owner_engine, as_of=as_of, rules=_selector_rules())

    picks = first["core_picks"]["in_nifty50"]
    assert [item["symbol"] for item in picks] == ["SYN-B", "SYN-A"]
    assert [item["symbol"] for item in second["core_picks"]["in_nifty50"]] == [
        "SYN-B",
        "SYN-A",
    ]
    assert first["selector_meta"]["disqualified_count_by_category"]["in_nifty50"] == 1
    assert first["inav_premium_pct"]["in_nifty50"] == "0"
    assert "ter_unavailable" in picks[0]["reason_codes"]


def test_scorekeeper_closes_a_vetoed_idea_and_saves_track_record(
    migrated_database: MigratedDatabase,
) -> None:
    idea = Security(
        market="IN",
        symbol="SYN-IDEA",
        name="Synthetic Idea",
        security_type="equity",
        currency="INR",
    )
    benchmark = Security(
        market="IN",
        symbol="NIFTYBEES",
        name="Synthetic Nifty 50 Fund",
        security_type="etf",
        currency="INR",
    )
    entry_date = date(2026, 11, 2)
    exit_date = date(2026, 11, 6)
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add_all([idea, benchmark])
        session.flush()
        for security, prices in (
            (idea, {entry_date: Decimal("100"), exit_date: Decimal("90")}),
            (benchmark, {entry_date: Decimal("200"), exit_date: Decimal("210")}),
        ):
            for as_of, close in prices.items():
                session.add(
                    PriceEod(
                        security_id=security.id,
                        as_of=as_of,
                        close=close,
                        source="synthetic",
                        source_url="https://example.test/prices",
                    )
                )
        idea_id = idea.id

    opened = run_scorekeeper(
        migrated_database.owner_engine,
        week="2026-W44",
        idea_outcomes=[
            {
                "dossier_id": "synthetic/idea",
                "security_id": str(idea_id),
                "market": "IN",
                "symbol": "SYN-IDEA",
                "status": "APPROVED",
                "confidence": "HIGH",
            }
        ],
    )
    assert opened["track_record"]["markets"]["IN"]["status"] == "insufficient_data"

    closed = run_scorekeeper(
        migrated_database.owner_engine,
        week="2026-W45",
        idea_outcomes=[
            {
                "dossier_id": "synthetic/idea",
                "security_id": str(idea_id),
                "market": "IN",
                "symbol": "SYN-IDEA",
                "status": "VETOED",
            }
        ],
    )

    assert closed["track_record"]["markets"]["IN"]["ahead_count"] == 0
    assert closed["track_record"]["markets"]["IN"]["benchmark_unavailable_count"] == 0
    with Session(migrated_database.owner_engine) as session:
        position = session.scalar(
            select(PaperPosition).where(PaperPosition.dossier_id == "synthetic/idea")
        )
        saved_record = session.execute(
            text(
                "SELECT ideas_count, ahead_count FROM track_record "
                "WHERE week = :week AND market = 'IN'"
            ),
            {"week": "2026-W45"},
        ).one()
    assert position is not None
    assert position.status == "closed"
    assert position.exit_reason == "vetoed"
    assert position.exit_date == exit_date
    assert position.excess_return_pct == Decimal("-15.000000")
    assert saved_record == (1, 0)


def test_owner_only_system_tables_have_no_app_grants_or_rls(
    migrated_database: MigratedDatabase,
) -> None:
    with migrated_database.owner_engine.connect() as connection:
        for table in ("rule_change_proposals", "paper_positions", "track_record"):
            row = connection.execute(
                text(
                    "SELECT c.relrowsecurity, "
                    "has_table_privilege('wealth_app', :qualified, 'SELECT'), "
                    "has_table_privilege('wealth_app', :qualified, 'INSERT'), "
                    "has_table_privilege('wealth_api', :qualified, 'SELECT'), "
                    "has_table_privilege('wealth_api', :qualified, 'INSERT'), "
                    "has_table_privilege(current_user, :qualified, 'INSERT') "
                    "FROM pg_class AS c JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = :schema AND c.relname = :table"
                ),
                {
                    "qualified": f'"{migrated_database.schema_name}"."{table}"',
                    "schema": migrated_database.schema_name,
                    "table": table,
                },
            ).one()
            assert row == (False, False, False, False, False, True)


def test_selector_scorekeeper_and_agenda_fields_merge_into_existing_pack(
    migrated_database: MigratedDatabase,
) -> None:
    week = "2026-W44"
    original = {"ideas": [], "summary": "No new research ideas."}
    additional = {
        "core_picks": {"in_nifty50": []},
        "inav_premium_pct": {"in_nifty50": None},
        "track_record": {"status": "insufficient_data"},
        "cio_agenda": {"market_limits": {"IN": {"max_dossiers": 10}}},
    }

    save_research_pack(migrated_database.owner_engine, week, original)
    save_pack_fields(migrated_database.owner_engine, week, additional)
    saved = load_research_pack(migrated_database.owner_engine, week)

    assert saved is not None
    assert saved["ideas"] == original["ideas"]
    assert saved["summary"] == original["summary"]
    assert {key: saved[key] for key in additional} == additional


def test_c16b_migration_can_downgrade_and_upgrade_again(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0023_macro_geo")
    inspector = inspect(migrated_database.owner_engine)

    assert "rule_change_proposals" not in inspector.get_table_names()
    assert "paper_positions" not in inspector.get_table_names()
    assert "track_record" not in inspector.get_table_names()
    assert "close" not in {column["name"] for column in inspector.get_columns("index_valuations")}

    command.upgrade(migrated_database.config, "head")
    inspector = inspect(migrated_database.owner_engine)

    assert {"rule_change_proposals", "paper_positions", "track_record"} <= set(
        inspector.get_table_names()
    )
    assert "close" in {column["name"] for column in inspector.get_columns("index_valuations")}
