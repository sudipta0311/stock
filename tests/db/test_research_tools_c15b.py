from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import FundamentalPIT, PriceEod, Security
from wealth.marketdata.sources.edgar import EdgarCompany
from wealth.marketdata.sources.edgar_filings import EdgarFiling
from wealth.research.router import RunBudget, RunContext
from wealth.research.tools import ToolRegistry

pytestmark = pytest.mark.db


class FakeSecClient:
    def company_tickers(self):
        return {"SYNTH": EdgarCompany("SYNTH", "0000000001", "Synthetic Issuer")}

    def filings(self, cik, *, forms, days, as_of):
        return [
            EdgarFiling(
                cik=cik,
                accession="0000000001-26-000001",
                form="8-K",
                filed_at=as_of,
                document_url="https://www.sec.gov/Archives/edgar/data/1/synthetic.htm",
                primary_document="synthetic.htm",
            )
        ]

    def filing_text(self, filing, *, max_chars):
        return "Synthetic filing text."[:max_chars]

    def close(self):
        return None


class ForbiddenNseClient:
    def __init__(self, status_code: int = 403) -> None:
        self.status_code = status_code

    def announcements(self, symbol, *, days, as_of):
        request = httpx.Request("GET", "https://www.nseindia.com/api/corporate-announcements")
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    def close(self):
        return None


@pytest.mark.parametrize("status_code", [401, 403])
def test_research_tools_log_source_provenance_and_nse_auth_errors_are_unavailable(
    migrated_database: MigratedDatabase,
    status_code: int,
) -> None:
    as_of = date.today()
    security = Security(
        market="IN",
        symbol=f"SYNTH{uuid4().hex[:6].upper()}",
        name="Synthetic Research Equity",
        security_type="equity",
        currency="INR",
        sleeve="equity_india",
        sector="Services",
        industry="Software Services",
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(security)
        session.flush()
        session.add(
            FundamentalPIT(
                security_id=security.id,
                metric="net_income",
                period_end=as_of,
                period_type="Q",
                source="NSE_SYNTHETIC",
                known_at=datetime.combine(as_of, datetime.min.time(), UTC),
                value=Decimal("25"),
                unit="INR",
                filed_at=datetime.combine(as_of, datetime.min.time(), UTC),
                source_url="https://example.test/financials.xml",
            )
        )
        session.add(
            PriceEod(
                security_id=security.id,
                as_of=as_of,
                close=Decimal("125"),
                volume=1000,
                traded_value=Decimal("125000"),
                source="NSE_SYNTHETIC",
                source_url="https://example.test/bhavcopy.zip",
            )
        )
        security_id = security.id

    registry = ToolRegistry(
        migrated_database.owner_engine,
        edgar_client_factory=FakeSecClient,
        nse_client_factory=lambda: ForbiddenNseClient(status_code),
    )
    context = RunContext(
        budget=RunBudget(max_usd=Decimal("1"), max_calls=10),
        owner_engine=migrated_database.owner_engine,
    )
    results = [
        registry.call(
            "fundamentals_series",
            {"security_id": str(security_id), "metrics": ["net_income"], "quarters": 1},
            context,
        ),
        registry.call("factor_snapshot", {"security_id": str(security_id)}, context),
        registry.call("price_stats", {"security_id": str(security_id)}, context),
        registry.call("sector_peers", {"security_id": str(security_id), "metric": "roe"}, context),
        registry.call("edgar_company_tickers", {"tickers": ["SYNTH"]}, context),
        registry.call(
            "edgar_filings",
            {"cik": "0000000001", "forms": ["8-K"], "days": 2, "as_of": as_of.isoformat()},
            context,
        ),
        registry.call(
            "edgar_filing_text",
            {"accession": "0000000001-26-000001", "max_chars": 100},
            context,
        ),
        registry.call("nse_announcements", {"symbol": "SYNTH", "days": 2}, context),
        # Valuation reads peers after its queries; it must not touch expired ORM rows.
        registry.call("valuation_stats", {"security_id": str(security_id)}, context),
    ]
    with migrated_database.owner_engine.connect() as connection:
        audit_rows = connection.execute(
            text(
                "SELECT tool, source_url, retrieved_at, status FROM tool_calls "
                "WHERE tool_call_id::text IN :ids ORDER BY retrieved_at"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": [result.tool_call_id for result in results]},
        ).all()

    assert results[0].result[0]["value"] == "25.00000000"
    assert results[6].result["text"] == "Synthetic filing text."
    assert results[7].result == {"status": "unavailable"}
    assert results[8].result["security_id"] == str(security_id)
    assert all(result.source_url for result in results)
    assert all(result.retrieved_at.tzinfo is not None for result in results)
    assert len(audit_rows) == len(results)
    assert all(
        row.source_url and row.retrieved_at.tzinfo and row.status == "success" for row in audit_rows
    )
