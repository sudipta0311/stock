from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import PriceEod, Security, UniverseMember
from wealth.marketdata import refresh, store
from wealth.marketdata.refresh import UniverseBatch, run_refresh
from wealth.marketdata.sources.nse import NseClose, NseSecurity
from wealth.marketdata.sources.universes import (
    SURVIVORSHIP_LABEL,
    Sp500Holdings,
    Sp500Sector,
    UniverseSecurity,
)

pytestmark = pytest.mark.db


def test_nse_universe_sectors_and_price_turnover_are_persisted(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 10, 28)
    initial = NseSecurity(
        symbol="SYNTH",
        name="Synthetic NSE Equity",
        isin=None,
        series="EQ",
        security_type="equity",
        sector="Services",
        industry="Software Services",
    )
    exchange_refresh = NseSecurity(
        symbol="SYNTH",
        name="Synthetic NSE Equity",
        isin=None,
        series="EQ",
        security_type="equity",
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        store.upsert_nse_master(session, [initial])
        store.upsert_nse_master(session, [exchange_refresh])
        security = session.scalar(select(Security).where(Security.symbol == "SYNTH"))
        assert security is not None
        session.refresh(security)
        session.add(
            UniverseMember(
                universe="nifty500",
                security_id=security.id,
                added_on=as_of,
                source="NSE_SYNTHETIC",
                source_url="https://example.test/nifty500.csv",
                survivorship_label=SURVIVORSHIP_LABEL,
            )
        )
        stored = store.upsert_nse_closes(
            session,
            [
                NseClose(
                    "SYNTH",
                    Decimal("100.50"),
                    as_of,
                    series="EQ",
                    volume=2500,
                    traded_value=Decimal("251250"),
                )
            ],
        )
        price = session.execute(
            select(PriceEod.volume, PriceEod.traded_value).where(
                PriceEod.security_id == security.id,
                PriceEod.as_of == as_of,
            )
        ).one_or_none()
        sector, industry = security.sector, security.industry

    assert stored == 1
    assert sector == "Services"
    assert industry == "Software Services"
    assert price is not None
    assert price.volume == 2500
    assert price.traded_value == Decimal("251250.0000")


def test_sp500_refresh_stores_sector_from_issuer_workbook(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 10, 28)
    batch = UniverseBatch(
        sp500=Sp500Holdings(
            securities=[
                UniverseSecurity(
                    symbol="SYNTH",
                    name="Synthetic US Equity",
                    sector="Industrials",
                    industry="Industrial Products",
                )
            ],
            as_of=as_of,
        ),
        nifty500=None,
        errors={},
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        result = run_refresh(
            session,
            as_of,
            only=("universes",),
            fetchers={"universes": lambda: batch},
        )
        security = session.scalar(
            select(Security).where(Security.market == "US", Security.symbol == "SYNTH")
        )
        sector, industry = (
            (security.sector, security.industry) if security is not None else (None, None)
        )

    assert result["status"] == "succeeded"
    assert security is not None
    assert (sector, industry) == ("Industrials", "Industrial Products")


def test_sp500_refresh_fills_missing_sectors_and_stores_implied_closes(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 10, 28)
    batch = UniverseBatch(
        sp500=Sp500Holdings(
            securities=[UniverseSecurity(symbol="SYNTHB", name="Synthetic US Equity B")],
            as_of=as_of,
        ),
        nifty500=None,
        errors={},
        sp500_closes={"SYNTHB": Decimal("123.456789")},
        sp500_sectors={"SYNTHB": Sp500Sector("Health Care", "Life Sciences Tools")},
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        result = run_refresh(
            session,
            as_of,
            only=("universes",),
            fetchers={"universes": lambda: batch},
        )
        security = session.scalar(
            select(Security).where(Security.market == "US", Security.symbol == "SYNTHB")
        )
        assert security is not None
        classification = (security.sector, security.industry)
        price = session.execute(
            select(PriceEod.close, PriceEod.source).where(
                PriceEod.security_id == security.id, PriceEod.as_of == as_of
            )
        ).one()

    assert result["counts"]["universes"]["sp500"]["prices_eod"] == 1
    assert classification == ("Health Care", "Life Sciences Tools")
    assert price.close == Decimal("123.456789")
    assert price.source == "SSGA_SPY_IMPLIED"


def test_nse_history_loads_only_missing_weekdays_for_current_members(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    on = date(2026, 10, 30)
    loaded_day = date(2026, 10, 29)
    with Session(migrated_database.owner_engine) as session, session.begin():
        store.upsert_nse_master(
            session,
            [NseSecurity("HISTSYN", "Synthetic History Equity", "INE000H01011", "EQ", "equity")],
        )
        security = session.scalar(select(Security).where(Security.symbol == "HISTSYN"))
        assert security is not None
        session.add(
            UniverseMember(
                universe="nifty500",
                security_id=security.id,
                added_on=on,
                source="NSE_SYNTHETIC",
                source_url="https://example.test/nifty500.csv",
                survivorship_label=SURVIVORSHIP_LABEL,
            )
        )
        session.flush()
        store.upsert_nse_closes(
            session, [NseClose("HISTSYN", Decimal("10"), loaded_day, series="EQ")]
        )
        requested: list[date] = []

        def fake_bhavcopy(day: date) -> list[NseClose]:
            requested.append(day)
            if day == date(2026, 10, 27):
                raise RuntimeError("synthetic outage")
            return [
                NseClose("HISTSYN", Decimal("11"), day, series="EQ"),
                NseClose("NOTSTORED", Decimal("1"), day, series="EQ"),
            ]

        already = store.loaded_nse_sessions(session, on - timedelta(days=400), on)
        monkeypatch.setattr(refresh.nse, "fetch_bhavcopy", fake_bhavcopy)
        result = run_refresh(session, on, only=("nse_history",))
        stored_days = set(
            session.scalars(select(PriceEod.as_of).where(PriceEod.security_id == security.id))
        )

    weekdays = [
        on - timedelta(days=offset)
        for offset in range(401)
        if (on - timedelta(days=offset)).weekday() < 5
    ]
    assert loaded_day in already
    assert sorted(requested) == sorted(day for day in weekdays if day not in already)
    assert result["status"] == "partial"
    assert result["errors"] == {"nse_history:failed_sessions": "1"}
    assert stored_days == (set(requested) - {date(2026, 10, 27)}) | {loaded_day}
