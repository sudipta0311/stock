"""Market-data persistence, provider, refresh and API behavior."""

from __future__ import annotations

import csv
import io
import secrets
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.api.settings import Settings
from wealth.db.models import (
    FxRate,
    GoldPrice,
    HoldingLot,
    MarketdataRun,
    NavDaily,
    PriceEod,
    Security,
    User,
)
from wealth.marketdata import store
from wealth.marketdata.provider import DbPriceProvider
from wealth.marketdata.refresh import NseMasterBatch, run_refresh
from wealth.marketdata.sources import ecb, nse
from wealth.marketdata.sources.amfi import AmfiNav
from wealth.marketdata.sources.fred import ECB_SOURCE, FxObservation
from wealth.marketdata.sources.ibja import GoldObservation
from wealth.marketdata.sources.nse import NseClose
from wealth.valuation.prices import PriceUnavailable

pytestmark = pytest.mark.db
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _security(
    symbol: str,
    *,
    security_type: str = "equity",
    isin: str | None = None,
    sleeve: str = "equity_india",
    category: str = "equity:india",
) -> Security:
    return Security(
        market="IN",
        symbol=symbol,
        isin=isin,
        name=f"Synthetic {symbol}",
        security_type=security_type,
        currency="INR",
        sleeve=sleeve,
        category=category,
    )


def test_store_upserts_are_idempotent_and_latest_price_is_point_in_time(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 10, 28)
    security = _security("SYNTHETIC-PRICE")
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(security)
        session.flush()
        store.upsert_eod_price(session, security.id, on - timedelta(days=2), Decimal("100.00"))
        store.upsert_eod_price(session, security.id, on, Decimal("101.25"))
        store.upsert_eod_price(session, security.id, on, Decimal("102.50"))
        store.upsert_eod_price(session, security.id, on + timedelta(days=1), Decimal("999.00"))
        latest = store.latest_price(session, security.id, on)
        row_count = session.scalar(
            select(func.count())
            .select_from(PriceEod)
            .where(PriceEod.security_id == security.id, PriceEod.as_of == on)
        )

    assert latest == (Decimal("102.500000"), on, "NSE")
    assert row_count == 1


def test_master_and_bhavcopy_refresh_store_nse_etf_proxy_prices(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 10, 27)
    equities = nse.parse_equity_list(_fixture("nse_equity_list.csv"))
    etfs = nse.parse_etf_list(_fixture("nse_etfseclist.csv"))
    closes = nse.parse_bhavcopy(_fixture("nse_etf_bhavcopy.csv"))
    batch = NseMasterBatch(equities + etfs, len(etfs))

    with Session(migrated_database.owner_engine) as session, session.begin():
        result = run_refresh(
            session,
            on,
            ("master", "nse"),
            fetchers={
                "master": lambda: batch,
                "nse": lambda: closes,
            },
        )
        security_types = {
            security.symbol: security.security_type
            for security in session.scalars(
                select(Security).where(Security.symbol.in_(("NIFTYBEES", "JUNIORBEES", "GOLDBEES")))
            )
        }
        prices = {
            security.symbol: close
            for security, close in session.execute(
                select(Security, PriceEod.close)
                .join(PriceEod, PriceEod.security_id == Security.id)
                .where(Security.symbol.in_(("NIFTYBEES", "JUNIORBEES", "GOLDBEES")))
            )
        }
        store.upsert_amfi_master(
            session,
            [
                AmfiNav(
                    scheme_code="999001",
                    isin_primary="INF204KB17I5",
                    isin_reinvestment=None,
                    name="ETF NAV overlay",
                    plan="Direct Plan",
                    option="Growth",
                    nav=Decimal("116.0000"),
                    as_of=on,
                    category="mf:gold",
                    sleeve="gold",
                )
            ],
        )
        gold_after_amfi = session.scalar(select(Security).where(Security.symbol == "GOLDBEES"))
        assert gold_after_amfi is not None
        gold_identity = (
            gold_after_amfi.symbol,
            gold_after_amfi.security_type,
            gold_after_amfi.nse_symbol,
            gold_after_amfi.sleeve,
        )

    assert result["status"] == "succeeded"
    assert result["counts"]["master"] == len(equities) + len(etfs)
    assert result["counts"]["master_etf"] == len(etfs)
    assert security_types == {
        "NIFTYBEES": "etf",
        "JUNIORBEES": "etf",
        "GOLDBEES": "etf",
    }
    assert prices == {
        "NIFTYBEES": Decimal("259.490000"),
        "JUNIORBEES": Decimal("727.130000"),
        "GOLDBEES": Decimal("115.640000"),
    }
    assert gold_identity == ("GOLDBEES", "etf", "GOLDBEES", "gold")


def test_master_refresh_reports_etf_endpoint_failure_and_keeps_equities(
    migrated_database: MigratedDatabase,
) -> None:
    equities = nse.parse_equity_list(_fixture("nse_equity_list.csv"))
    batch = NseMasterBatch(equities, 0, "ConnectTimeout")

    with Session(migrated_database.owner_engine) as session, session.begin():
        result = run_refresh(
            session,
            date(2026, 10, 28),
            ("master",),
            fetchers={"master": lambda: batch},
        )
        stored_symbols = set(session.scalars(select(Security.symbol)))

    assert result["status"] == "partial"
    assert result["errors"] == {"etf_master": "ConnectTimeout"}
    assert result["counts"]["master"] == len(equities)
    assert result["counts"]["master_etf"] == 0
    assert equities[0].symbol in stored_symbols


def test_prune_keeps_recent_held_and_proxy_prices_only(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 10, 28)
    held = _security("SYNTHETIC-HELD")
    unheld = _security("SYNTHETIC-UNHELD")
    proxy = _security("NIFTYBEES", security_type="etf", category="etf:equity_india")
    user = User(email=f"marketdata-{uuid4()}@example.test", auth_subject=str(uuid4()))
    cutoff = on - timedelta(days=400)
    old = cutoff - timedelta(days=1)
    recent = on - timedelta(days=2)

    with Session(migrated_database.owner_engine) as session, session.begin():
        existing_proxy = session.scalar(select(Security).where(Security.symbol == "NIFTYBEES"))
        if existing_proxy is None:
            session.add(proxy)
        else:
            proxy = existing_proxy
        session.add_all([held, unheld, user])
        session.flush()
        session.execute(delete(PriceEod).where(PriceEod.security_id == proxy.id))
        session.execute(delete(NavDaily).where(NavDaily.security_id == proxy.id))
        session.add(
            HoldingLot(
                user_id=user.id,
                security_id=held.id,
                quantity=Decimal("2"),
                source="manual",
            )
        )
        for security in (held, unheld, proxy):
            store.upsert_eod_price(session, security.id, recent, Decimal("10.00"))
            store.upsert_nav(session, security.id, recent, Decimal("10.00"))
        store.upsert_eod_price(session, held.id, old, Decimal("9.00"))
        counts = store.prune(session, on)
        held_price_count = session.scalar(
            select(func.count()).select_from(PriceEod).where(PriceEod.security_id == held.id)
        )
        old_price_count = session.scalar(
            select(func.count())
            .select_from(PriceEod)
            .where(PriceEod.security_id == held.id, PriceEod.as_of == old)
        )
        unheld_price_count = session.scalar(
            select(func.count()).select_from(PriceEod).where(PriceEod.security_id == unheld.id)
        )
        proxy_price_count = session.scalar(
            select(func.count()).select_from(PriceEod).where(PriceEod.security_id == proxy.id)
        )

    assert counts["prices_eod"] >= 2
    assert counts["nav_daily"] >= 1
    assert held_price_count == 1
    assert old_price_count == 0
    assert unheld_price_count == 0
    assert proxy_price_count == 1


def test_db_price_provider_returns_source_and_rejects_prices_over_ten_days_old(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 10, 28)
    security = _security("SYNTHETIC-STALE", security_type="mf", isin="INF000000101")
    old_security = _security("SYNTHETIC-OLD", security_type="mf", isin="INF000000102")
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add_all([security, old_security])
        session.flush()
        store.upsert_nav(session, security.id, on - timedelta(days=3), Decimal("12.3456"))
        store.upsert_nav(session, old_security.id, on - timedelta(days=11), Decimal("10.00"))
        provider = DbPriceProvider(session)
        quote = provider.nav("INF000000101", on)
        with pytest.raises(PriceUnavailable):
            provider.nav("INF000000102", on)

    assert quote.value == Decimal("12.34560000")
    assert quote.as_of == on - timedelta(days=3)
    assert quote.source == "AMFI"
    assert quote.stale_days == 3


def test_refresh_records_partial_run_and_keeps_other_sources_when_nse_fails(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 10, 28)
    fetched_at = datetime(2026, 10, 28, 12, tzinfo=UTC)

    def fail_nse() -> list[NseClose]:
        raise RuntimeError("synthetic source outage")

    fetchers = {
        "nse": fail_nse,
        "fx": lambda: [
            FxObservation(
                "USD/INR",
                Decimal("95.1242"),
                on,
                1,
                source=ECB_SOURCE,
                source_url=ecb.ECB_SOURCE_URL,
            )
        ],
        "gold": lambda: GoldObservation(Decimal("10000.00"), on),
    }
    with Session(migrated_database.owner_engine) as session, session.begin():
        result = run_refresh(
            session,
            on,
            ("nse", "fx", "gold"),
            fetchers=fetchers,
            fetched_at=fetched_at,
        )
        fx = session.get(FxRate, ("USD/INR", on))
        gold = session.get(GoldPrice, on)
        run = session.get(MarketdataRun, result["run_id"])
        fx_values = (fx.rate, fx.publication_lag_days) if fx is not None else None
        fx_source = fx.source if fx is not None else None
        fx_source_urls = fx.source_url.splitlines() if fx is not None else None
        gold_source = gold.source if gold is not None else None
        run_errors = run.errors if run is not None else None

    assert result["status"] == "partial"
    assert result["errors"] == {"nse": "RuntimeError"}
    assert result["counts"]["fx"] == {"fx_rates": 1}
    assert result["counts"]["gold"]["gold_prices"] == 1
    assert fx_values == (Decimal("95.12420000"), 1)
    assert fx_source == ECB_SOURCE
    assert fx_source_urls == [ecb.ECB_SOURCE_URL]
    assert gold_source == "IBJA"
    assert run_errors == {"nse": "RuntimeError"}


def test_fx_refresh_stores_available_series_and_records_missing_currency(
    migrated_database: MigratedDatabase,
) -> None:
    on = date(2026, 10, 28)
    reader = csv.DictReader(io.StringIO(_fixture("ecb_multi_eur_cross.csv")))
    fieldnames = reader.fieldnames
    assert fieldnames is not None
    available_rows = [row for row in reader if row["CURRENCY"] != "SEK"]
    csv_text = io.StringIO()
    writer = csv.DictWriter(csv_text, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(available_rows)
    batch = ecb.parse_ecb_multi_fx_rates(csv_text.getvalue(), fetched_on=on)

    with Session(migrated_database.owner_engine) as session, session.begin():
        result = run_refresh(session, on, ("fx",), fetchers={"fx": lambda: batch})
        stored_pairs = set(
            session.scalars(select(FxRate.pair).where(FxRate.as_of == date(2026, 10, 27)))
        )
        run = session.get(MarketdataRun, result["run_id"])
        run_errors = run.errors if run is not None else None

    assert result["status"] == "partial"
    assert result["errors"] == {"fx:SEK": "NoObservations"}
    assert result["counts"]["fx"] == {"fx_rates": 8}
    assert {"USD/INR", "DKK/INR", "EUR/INR"}.issubset(stored_pairs)
    assert "SEK/INR" not in stored_pairs
    assert run_errors == {"fx:SEK": "NoObservations"}


@pytest.fixture
def api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, str]]:
    from wealth.api.main import create_app
    from wealth.api.money_map import get_today
    from wealth.db import session as db_session

    secret = secrets.token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    application = create_app(settings)
    application.dependency_overrides[get_today] = lambda: date(2026, 10, 28)
    auth_subject = f"marketdata-{uuid4()}"
    email = f"marketdata-{uuid4()}@example.test"
    token = jwt.encode(
        {
            "sub": auth_subject,
            "email": email,
            "name": "Synthetic Market Data User",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    with TestClient(application) as client:
        yield client, auth_subject, f"Bearer {token}"


def test_api_values_gold_and_mutual_funds_from_prices_with_statement_fallback(
    api_client: tuple[TestClient, str, str],
    migrated_database: MigratedDatabase,
) -> None:
    client, auth_subject, authorization = api_client
    headers = {"Authorization": authorization}
    on = date(2026, 10, 28)
    gold_price = GoldObservation(
        inr_per_gram_24k=Decimal("10000.00"),
        as_of=on,
        source_url="https://www.ibjarates.com/",
    )
    mutual_fund = _security(
        "SYNTHETIC-MF",
        security_type="mf",
        isin="INF000000201",
        sleeve="debt",
        category="mf:debt:liquid",
    )
    equity = _security("SYNTHETIC-NO-PRICE")
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add_all([mutual_fund, equity])
        session.flush()
        mutual_fund_id = mutual_fund.id
        equity_id = equity.id
        store.upsert_gold_price(session, gold_price)
        market_holding_user_id = session.scalar(
            select(User.id).where(User.auth_subject == auth_subject)
        )
        assert market_holding_user_id is None

    gold_response = client.post(
        "/v1/assets",
        headers=headers,
        json={"asset_type": "gold", "kind": "physical", "grams": "20", "purity": 22},
    )
    assert gold_response.status_code == 201
    assert gold_response.json()["valuation"]["value_today"] == "183333.33"
    assert "price as of 2026-10-28 (IBJA)" in gold_response.json()["valuation"]["notes"]

    with Session(migrated_database.owner_engine) as session, session.begin():
        user = session.scalar(select(User).where(User.auth_subject == auth_subject))
        assert user is not None
        session.add_all(
            [
                HoldingLot(
                    user_id=user.id,
                    security_id=mutual_fund_id,
                    quantity=Decimal("100"),
                    source="manual",
                    statement_value=Decimal("900.00"),
                ),
                HoldingLot(
                    user_id=user.id,
                    security_id=equity_id,
                    quantity=Decimal("2"),
                    source="manual",
                    statement_value=Decimal("420.00"),
                ),
            ]
        )
        store.upsert_nav(session, mutual_fund_id, on, Decimal("12.50"))

    response = client.get("/v1/networth", headers=headers)
    assert response.status_code == 200
    networth = response.json()
    holdings = {item["symbol"]: item for item in networth["holdings"]}
    assert holdings["SYNTHETIC-MF"]["value"] == "1250.00"
    assert holdings["SYNTHETIC-MF"]["price_as_of"] == on.isoformat()
    assert holdings["SYNTHETIC-MF"]["price_source"] == "AMFI"
    assert holdings["SYNTHETIC-MF"]["status"] == "priced"
    assert holdings["SYNTHETIC-NO-PRICE"]["value"] == "420.00"
    assert holdings["SYNTHETIC-NO-PRICE"]["status"] == "statement_value_used"
    assert networth["market_data_as_of"] == on.isoformat()
    assert networth["assets_total"] == "185003.33"
