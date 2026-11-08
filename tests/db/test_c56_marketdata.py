from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import (
    HoldingLot,
    HouseholdMember,
    PriceEod,
    Security,
    UniverseMember,
    User,
)
from wealth.marketdata import refresh
from wealth.marketdata.refresh import FundamentalsBatch
from wealth.marketdata.sources.alpha_vantage import DailyClose

pytestmark = pytest.mark.db


def test_fundamentals_targets_include_held_equities_once(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = uuid4()
    as_of = date(2026, 11, 6)
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(
            User(
                id=user_id,
                email=f"c56-{user_id.hex}@example.test",
                auth_subject=f"c56-{user_id.hex}",
            )
        )
        session.flush()
        member_id = session.scalar(
            select(HouseholdMember.id).where(
                HouseholdMember.user_id == user_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert member_id is not None
        securities = {
            key: Security(
                market=market,
                symbol=symbol,
                name=f"Synthetic {key} Share",
                security_type="equity",
                currency="INR" if market == "IN" else "USD",
                sleeve="equity_india" if market == "IN" else "equity_global",
                sector="Synthetic sector",
                industry="Synthetic industry",
            )
            for key, market, symbol in (
                ("in_member", "IN", "SYNTH-IN-MEMBER"),
                ("in_outside", "IN", "SYNTH-IN-OUTSIDE"),
                ("us_member", "US", "SYNTH-US-MEMBER"),
                ("us_outside", "US", "SYNTH-US-OUTSIDE"),
            )
        }
        session.add_all(securities.values())
        session.flush()
        for key, universe in (("in_member", "nifty500"), ("us_member", "sp500")):
            session.add(
                UniverseMember(
                    universe=universe,
                    security_id=securities[key].id,
                    added_on=date(2020, 1, 1),
                    source="synthetic_fixture",
                    source_url="https://example.test/synthetic-universe",
                    survivorship_label="synthetic fixture",
                )
            )
        for key in ("in_member", "in_outside", "us_member", "us_outside"):
            session.add(
                HoldingLot(
                    user_id=user_id,
                    member_id=member_id,
                    security_id=securities[key].id,
                    quantity=Decimal("1"),
                    cost_per_unit=Decimal("10"),
                    acquired_on=date(2025, 1, 1),
                    source="manual",
                    currency=securities[key].currency,
                )
            )

        nse_target_symbols: list[str] = []
        edgar_target_symbols: list[str] = []

        def capture_nse(_on, targets, _missing=None):
            nse_target_symbols.extend(target.symbol for target in targets)
            return FundamentalsBatch({}, {})

        def capture_edgar(_on, symbols):
            edgar_target_symbols.extend(symbols)
            return FundamentalsBatch({}, {}, {})

        monkeypatch.setattr(refresh, "_fetch_nse_fundamental_targets", capture_nse)
        monkeypatch.setattr(refresh, "_fetch_edgar_for_symbols", capture_edgar)

        refresh._fetch_nse_fundamentals(session, as_of, None)
        refresh._fetch_edgar(session, as_of, None)

    assert set(nse_target_symbols) == {"SYNTH-IN-MEMBER", "SYNTH-IN-OUTSIDE"}
    assert len(nse_target_symbols) == 2
    assert set(edgar_target_symbols) == {"SYNTH-US-MEMBER", "SYNTH-US-OUTSIDE"}
    assert len(edgar_target_symbols) == 2


def test_us_stock_refresh_only_rewrites_the_newest_close_on_repeat(
    migrated_database: MigratedDatabase,
) -> None:
    as_of = date(2026, 11, 6)
    fetched_at = datetime(2026, 11, 7, 8, tzinfo=UTC)
    symbol = "SYNTH-US-PRICE"
    with Session(migrated_database.owner_engine) as session, session.begin():
        security = Security(
            market="US",
            symbol=symbol,
            name="Synthetic U.S. share",
            security_type="equity",
            currency="USD",
            sleeve="equity_global",
            sector="Synthetic sector",
            industry="Synthetic industry",
        )
        session.add(security)
        session.flush()
        security_id = security.id
        records = [
            refresh.AbroadPrice(
                security_id,
                DailyClose(symbol, as_of - timedelta(days=offset), Decimal(str(10 + offset))),
            )
            for offset in (2, 1, 0)
        ]
        recorded_batch = refresh.AbroadBatch(records, 1, refresh._new_abroad_skip_counts())

        first = refresh._store_source(
            session,
            "us_stocks",
            recorded_batch,
            on=as_of,
            fetched_at=fetched_at,
        )
        second = refresh._store_source(
            session,
            "us_stocks",
            recorded_batch,
            on=as_of,
            fetched_at=fetched_at,
        )
        corrected_latest = [
            *records[:-1],
            refresh.AbroadPrice(
                security_id,
                DailyClose(symbol, as_of, Decimal("99")),
            ),
        ]
        corrected = refresh._store_source(
            session,
            "us_stocks",
            refresh.AbroadBatch(
                corrected_latest,
                1,
                refresh._new_abroad_skip_counts(),
            ),
            on=as_of,
            fetched_at=fetched_at,
        )
        stored = session.scalars(
            select(PriceEod).where(PriceEod.security_id == security_id).order_by(PriceEod.as_of)
        ).all()
        latest_close = stored[-1].close

    assert isinstance(first, refresh.SourceBatchResult)
    assert isinstance(second, refresh.SourceBatchResult)
    assert isinstance(corrected, refresh.SourceBatchResult)
    assert first.counts["prices_eod"] == 3
    assert second.counts["prices_eod"] == 1
    assert corrected.counts["prices_eod"] == 1
    assert len(stored) == 3
    assert latest_close == Decimal("99")
