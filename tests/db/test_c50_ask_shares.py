from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import MigratedDatabase
from wealth.core.ids import isin_from_cusip
from wealth.db.models import ResearchPackRecord, Security
from wealth.marketdata.refresh import SourceBatchResult, UniverseBatch, _store_source
from wealth.marketdata.sources.universes import Sp500Holdings, UniverseSecurity
from wealth.policy.actions import Action
from wealth.policy.data import _research_pack
from wealth.policy.household import MemberProfile
from wealth.policy.personalise import MarketContext, _abroad_action_rows
from wealth.schemas.profile import InvestingPrefs, MemberTax, Residency

pytestmark = pytest.mark.db


def test_refreshed_universe_isins_are_preserved_and_drive_danish_share_lines(
    migrated_database: MigratedDatabase,
) -> None:
    valid_isin = isin_from_cusip("111111116")
    conflict_isin = isin_from_cusip("123456782")
    assert valid_isin is not None
    assert conflict_isin is not None
    imported_id: UUID
    target_id: UUID
    with Session(migrated_database.owner_engine) as session, session.begin():
        imported = Security(
            market="US",
            symbol="IMPORTED",
            isin=conflict_isin,
            domicile="US",
            name="Synthetic imported company",
            security_type="equity",
            currency="USD",
            sleeve="equity_global",
            category="equity:us",
        )
        session.add(imported)
        session.flush()
        imported_id = imported.id
        target = Security(
            market="US",
            symbol="SYNTHC",
            isin=None,
            domicile="US",
            name="Synthetic conflict company",
            security_type="equity",
            currency="USD",
            sleeve="equity_global",
            category="equity:us",
        )
        session.add(target)
        session.flush()
        target_id = target.id

        refreshed = _store_source(
            session,
            "universes",
            UniverseBatch(
                sp500=Sp500Holdings(
                    securities=[
                        UniverseSecurity(
                            symbol="SYNTHA",
                            name="Synthetic refreshed company",
                            cusip="111111116",
                        ),
                        UniverseSecurity(
                            symbol="SYNTHC",
                            name="Synthetic conflict company",
                            cusip="123456782",
                        ),
                        UniverseSecurity(
                            symbol="SYNTHX",
                            name="Synthetic CINS company",
                            cusip="G1151C101",
                        ),
                    ],
                    as_of=date(2026, 11, 4),
                ),
                nifty500=None,
                errors={},
            ),
            on=date(2026, 11, 4),
            fetched_at=datetime(2026, 11, 4, tzinfo=UTC),
        )
        assert isinstance(refreshed, SourceBatchResult)
        assert refreshed.counts["isin_filled"] == 1
        assert refreshed.counts["isin_conflicts"] == 1
        assert refreshed.counts["isin_unavailable"] == 1
        assert refreshed.errors == {}

        imported_after = session.get(Security, imported_id)
        assert imported_after is not None
        assert imported_after.symbol == "IMPORTED"
        assert imported_after.isin == conflict_isin
        target_after = session.get(Security, target_id)
        assert target_after is not None
        assert target_after.symbol == "SYNTHC"
        assert target_after.name == "Synthetic conflict company"
        assert target_after.isin is None
        refreshed_security = session.scalar(
            select(Security).where(Security.market == "US", Security.symbol == "SYNTHA")
        )
        conflict_security = session.scalar(
            select(Security).where(Security.market == "US", Security.symbol == "SYNTHC")
        )
        cins_security = session.scalar(
            select(Security).where(Security.market == "US", Security.symbol == "SYNTHX")
        )
        assert refreshed_security is not None and refreshed_security.isin == valid_isin
        assert conflict_security is not None and conflict_security.id == target_id
        assert cins_security is not None and cins_security.isin is None

        week = "2026-W45"
        session.add(
            ResearchPackRecord(
                week=week,
                pack={
                    "ideas": [
                        {
                            "security_id": str(refreshed_security.id),
                            "symbol": "SYNTHA",
                            "market": "US",
                            "sector": "Industrials",
                            "price": "100.00",
                            "quant_decile": 10,
                            "dossier_status": "APPROVED",
                            "valuation_percentile": "35",
                            "confidence": "HIGH",
                            "name": "Synthetic refreshed company",
                            "currency": "USD",
                            "exchange": "NASDAQ",
                        }
                    ]
                },
            )
        )
        session.flush()
        research = _research_pack(session, week)
        assert research is not None
        assert research.ideas[0].isin == valid_isin

        member = MemberProfile(
            id="member-dk",
            name="Synthetic member",
            relationship="self",
            is_self=True,
            residency=Residency(status="NRI", country="DK"),
            tax=MemberTax(),
            investing=InvestingPrefs(us_stock_picks_opt_in=True, us_stock_picks_pct="25"),
        )
        context = SimpleNamespace(
            member=member,
            snapshot=SimpleNamespace(
                net_worth=Decimal("1250000"), investable_total=Decimal("1250000")
            ),
            limits_room={"ask_deposit_dkk": Decimal("20000")},
        )
        market = MarketContext(
            fx_rates={"DKK/INR": Decimal("12.5"), "USD/INR": Decimal("12.5")},
            debt_yields={},
            index_valuation_percentiles={},
        )
        action = Action(
            rank=1,
            kind="BUY_LUMPSUM",
            sleeve="equity_global",
            instrument_category="global_all_world",
            amount=Decimal("250000"),
            reason_code="target_gap",
            member_id=member.id,
            params={"account_type": "ASK"},
        )

        actions, buy_lines, _carry, _notes = _abroad_action_rows(
            [action], [member], [context], [], market, research, {}
        )

        share_line = next(
            line for line in buy_lines if line["instrument"].get("ticker") == "SYNTHA"
        )
        assert share_line["instrument"]["isin"] == valid_isin
        assert share_line["account"] == "ask"
        assert share_line["details"]["ask_unverified"] is True
        assert actions[0].params["account_type"] == "ASK"
