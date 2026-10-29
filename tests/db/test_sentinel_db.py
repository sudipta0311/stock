from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import HoldingLot, Security, SentinelEvent, User
from wealth.research.sentinel import run_sentinel
from wealth.research.tools import ToolResult

pytestmark = pytest.mark.db


class FakeRouter:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("known 8-K items must not reach the model")


class FilingRegistry:
    def __init__(self, filing_text: str) -> None:
        self.filing_text = filing_text
        self.text_call_id = str(uuid4())
        self.filings_call_id = str(uuid4())

    def call(self, name, args, _run_ctx):
        now = datetime(2026, 10, 29, 7, tzinfo=UTC)
        if name == "edgar_company_tickers":
            result = [{"ticker": "SYNTH", "cik": "0000000001", "name": "Synthetic Issuer"}]
            return ToolResult(
                str(uuid4()), result, "https://www.sec.gov/files/company_tickers.json", now
            )
        if name == "edgar_filings":
            result = [
                {
                    "cik": "0000000001",
                    "accession": "0000000001-26-000001",
                    "form": "8-K",
                    "filed_at": "2026-10-29",
                    "document_url": (
                        "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/synthetic.htm"
                    ),
                    "primary_document": "synthetic.htm",
                }
            ]
            return ToolResult(
                self.filings_call_id, result, "https://www.sec.gov/edgar/search/", now
            )
        if name == "edgar_filing_text":
            return ToolResult(
                self.text_call_id,
                {"accession": args["accession"], "text": self.filing_text},
                "https://www.sec.gov/Archives/edgar/data/1/synthetic.htm",
                now,
            )
        raise AssertionError(f"Unexpected tool: {name}")


def test_rule_events_are_idempotent_and_skip_nano(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_API_KEY", raising=False)
    no_targets = run_sentinel(
        migrated_database.owner_engine,
        now=datetime(2026, 10, 29, 7, tzinfo=UTC),
        router=FakeRouter(),
        registry=FilingRegistry(""),
    )
    assert no_targets["status"] == "no_targets"
    assert no_targets["targets"] == 0
    assert no_targets["edgar"]["status"] == "unavailable"
    user = User(
        email=f"sentinel-{uuid4().hex}@example.test",
        auth_subject=str(uuid4()),
    )
    security = Security(
        market="US",
        symbol="SYNTH",
        name="Synthetic US Equity",
        security_type="equity",
        currency="USD",
        sleeve="equity_global",
        sector="Industrials",
        industry="Unclassified",
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add_all([user, security])
        session.flush()
        session.add(
            HoldingLot(
                user_id=user.id,
                security_id=security.id,
                quantity=Decimal("1"),
                source="manual",
            )
        )
        security_id = security.id

    unavailable = run_sentinel(
        migrated_database.owner_engine,
        now=datetime(2026, 10, 29, 7, tzinfo=UTC),
        router=FakeRouter(),
        registry=FilingRegistry(""),
    )
    assert unavailable["edgar"]["status"] == "unavailable"
    assert unavailable["nse"]["status"] == "no_targets"

    monkeypatch.setenv("SEC_USER_AGENT", "Synthetic sentinel test contact=sentinel@example.test")
    registry = FilingRegistry(
        "ITEM 4.01 Auditor departure. ITEM 4.02 Restatement. ITEM 5.02 Key person departure."
    )
    router = FakeRouter()
    now = datetime(2026, 10, 29, 7, tzinfo=UTC)
    first = run_sentinel(
        migrated_database.owner_engine,
        now=now,
        registry=registry,
    )
    with Session(migrated_database.owner_engine) as session:
        first_events = list(
            session.scalars(
                select(SentinelEvent)
                .where(SentinelEvent.security_id == security_id)
                .order_by(SentinelEvent.event_code)
            )
        )
    second = run_sentinel(
        migrated_database.owner_engine,
        now=now,
        router=router,
        registry=registry,
    )
    with Session(migrated_database.owner_engine) as session:
        event_count = session.scalar(
            select(func.count())
            .select_from(SentinelEvent)
            .where(SentinelEvent.security_id == security_id)
        )
        second_ids = list(
            session.scalars(
                select(SentinelEvent.id)
                .where(SentinelEvent.security_id == security_id)
                .order_by(SentinelEvent.event_code)
            )
        )

    assert first["edgar"]["events"] == second["edgar"]["events"] == 3
    assert first["models"]["nano"] == "unavailable"
    assert event_count == 3
    assert [event.id for event in first_events] == second_ids
    assert {event.event_code for event in first_events} == {
        "AUDITOR_RESIGNATION",
        "RESTATEMENT",
        "KEY_PERSON_EXIT",
    }
    assert all(event.method == "rule" for event in first_events)
    assert all(event.evidence_tool_call_id == UUID(registry.text_call_id) for event in first_events)
    urgent_codes = {event.event_code for event in first_events if event.urgent}
    assert urgent_codes == {"AUDITOR_RESIGNATION", "RESTATEMENT"}
    assert router.calls == 0


def test_existing_sentinel_event_access_is_unchanged(
    migrated_database: MigratedDatabase,
) -> None:
    table_name = f"{migrated_database.schema_name}.sentinel_events"
    with migrated_database.owner_engine.connect() as connection:
        wealth_app_grants = connection.scalars(
            text(
                "SELECT privilege_type FROM pg_catalog.pg_class AS relation "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) AS acl "
                "WHERE relation.oid = pg_catalog.to_regclass(:table_name) "
                "AND acl.grantee = 'wealth_app'::regrole ORDER BY privilege_type"
            ),
            {"table_name": table_name},
        ).all()
        wealth_api_grants = connection.scalars(
            text(
                "SELECT privilege_type FROM pg_catalog.pg_class AS relation "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) AS acl "
                "WHERE relation.oid = pg_catalog.to_regclass(:table_name) "
                "AND acl.grantee = 'wealth_api'::regrole"
            ),
            {"table_name": table_name},
        ).all()
        assert wealth_app_grants == ["SELECT"]
        assert wealth_api_grants == []
        # wealth_api inherits wealth_app, so this effective SELECT is historical role behavior.
        assert (
            connection.scalar(
                text("SELECT has_table_privilege('wealth_api', :table_name, 'SELECT')"),
                {"table_name": table_name},
            )
            is True
        )
        for privilege in ("INSERT", "UPDATE", "DELETE"):
            assert (
                connection.scalar(
                    text("SELECT has_table_privilege('wealth_app', :table_name, :privilege)"),
                    {"table_name": table_name, "privilege": privilege},
                )
                is False
            )
        assert (
            connection.scalar(
                text(
                    "SELECT relrowsecurity FROM pg_catalog.pg_class "
                    "WHERE oid = pg_catalog.to_regclass(:table_name)"
                ),
                {"table_name": table_name},
            )
            is False
        )
        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            assert (
                connection.scalar(
                    text("SELECT has_table_privilege(:role, :table_name, :privilege)"),
                    {
                        "role": migrated_database.owner_role,
                        "table_name": table_name,
                        "privilege": privilege,
                    },
                )
                is True
            )
