from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import FundamentalPIT, PriceEod, Security, UniverseMember
from wealth.research.screener import run_screener

pytestmark = pytest.mark.db


def test_weekly_run_persists_a_cited_screened_dossier(
    migrated_database: MigratedDatabase,
) -> None:
    week = "2026-W44"
    as_of = date(2026, 10, 30)
    security = Security(
        market="IN",
        symbol=f"SYNTH{uuid4().hex[:6].upper()}",
        name="Synthetic Indian Equity",
        security_type="equity",
        currency="INR",
        sleeve="equity_india",
        sector="Services",
        industry="Technology Services",
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(security)
        session.flush()
        session.add(
            UniverseMember(
                universe="nifty500",
                security_id=security.id,
                added_on=as_of,
                source="NSE_SYNTHETIC",
                source_url="https://example.test/universe.csv",
                survivorship_label="current constituents only (survivorship bias)",
            )
        )
        for index in range(301):
            on = as_of - timedelta(days=300 - index)
            session.add(
                PriceEod(
                    security_id=security.id,
                    as_of=on,
                    close=Decimal("100") + Decimal(index) / Decimal("10"),
                    volume=1_000_000,
                    traded_value=Decimal("100000000"),
                    source="NSE_SYNTHETIC",
                    source_url="https://example.test/bhavcopy.zip",
                )
            )
        quarter_ends = (
            date(2025, 12, 31),
            date(2026, 3, 31),
            date(2026, 6, 30),
            date(2026, 9, 30),
        )
        for metric, value in (
            ("net_income", Decimal("20")),
            ("eps_diluted", Decimal("2")),
        ):
            for period_end in quarter_ends:
                known_at = datetime.combine(
                    period_end + timedelta(days=20), datetime.min.time(), UTC
                )
                session.add(
                    FundamentalPIT(
                        security_id=security.id,
                        metric=metric,
                        period_end=period_end,
                        period_type="Q",
                        source="NSE_SYNTHETIC",
                        known_at=known_at,
                        value=value,
                        unit="INR",
                        filed_at=known_at,
                        source_url="https://example.test/filing.xml",
                    )
                )
                session.flush()
        session.add(
            FundamentalPIT(
                security_id=security.id,
                metric="total_equity",
                period_end=as_of - timedelta(days=30),
                period_type="Q",
                source="NSE_SYNTHETIC",
                known_at=datetime(2026, 10, 25, tzinfo=UTC),
                value=Decimal("100"),
                unit="INR",
                filed_at=datetime(2026, 10, 25, tzinfo=UTC),
                source_url="https://example.test/filing.xml",
            )
        )
        session.flush()
        security_id = security.id

    result = run_screener(migrated_database.owner_engine, week)

    assert result["shortlisted"]["IN"] == 1
    assert result["security_ids"] == [str(security_id)]
    with migrated_database.owner_engine.connect() as connection:
        dossier = connection.execute(
            text("SELECT id, stage, json FROM dossiers WHERE security_id = :security_id"),
            {"security_id": security_id},
        ).one()
        snapshot = connection.execute(
            text(
                "SELECT tool_call_id, result_json FROM tool_calls "
                "WHERE tool = 'screener_snapshot' ORDER BY retrieved_at DESC LIMIT 1"
            )
        ).one()

    assert dossier.stage == "SCREENED"
    claim = dossier.json["claims"][0]
    assert claim["kind"] == "FACT"
    assert claim["evidence"][0]["tool_call_id"] == str(snapshot.tool_call_id)
    assert snapshot.result_json["raw_factors"]["roe"] is not None
    assert snapshot.result_json["peer_inputs"]["roe"]
    assert snapshot.result_json["source_urls"]
