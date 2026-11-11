"""Broker upload mapping, household ownership, lot replacement, and tax feeds."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import jwt
import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from conftest import MigratedDatabase
from wealth.api.fund_check import _member_rows
from wealth.api.imports import _get_item_response
from wealth.api.main import create_app
from wealth.api.money_map import get_today
from wealth.api.settings import Settings
from wealth.core.ids import isin_check_digit_ok
from wealth.db import session as db_session
from wealth.db.models import (
    Account,
    BrokerCashFlowRecord,
    FxRate,
    HoldingLot,
    HouseholdMember,
    ImportItem,
    ImportJob,
    ImportTemplate,
    Profile,
    RealisedTrade,
    Security,
)
from wealth.db.repo import ensure_self_household_member
from wealth.db.session import resolve_user, system_session, user_session
from wealth.policy.data import load_policy_inputs
from wealth.policy.fund_check import _FundSaleLedger
from wealth.policy.tax import Lot

pytestmark = pytest.mark.db
_TODAY = date(2026, 10, 28)
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "brokers"


@pytest.fixture
def broker_api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str]]:
    secret = "synthetic-broker-test-secret-0123456789"
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url=migrated_database.api_database_url,
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    application = create_app(settings)
    application.dependency_overrides[get_today] = lambda: _TODAY
    with TestClient(application) as client:
        yield client, secret


def _identity(secret: str, label: str) -> tuple[dict[str, str], dict[str, str]]:
    claims = {
        "sub": f"synthetic-broker-{label}-{uuid4()}",
        "email": f"synthetic-{label}-{uuid4()}@example.test",
        "name": "Synthetic Broker User",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    token = jwt.encode(claims, secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}, claims


def _fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def _upload(
    client: TestClient,
    headers: dict[str, str],
    filename: str,
    data: bytes,
    **fields: str,
) -> object:
    return client.post(
        "/v1/imports",
        headers=headers,
        files={"file": (filename, data, "text/csv")},
        data={"doc_type": "broker_statement", **fields},
    )


def _user_id(database: MigratedDatabase, claims: dict[str, str]) -> UUID:
    return resolve_user(
        database.api_engine,
        claims["sub"],
        claims["email"],
        claims["name"],
    )


def _valid_synthetic_isin(seed: str) -> str:
    base = f"IN{seed[:9]:0<9}"
    for check_digit in range(10):
        candidate = f"{base}{check_digit}"
        if isin_check_digit_ok(candidate):
            return candidate
    raise AssertionError("A synthetic ISIN check digit could not be generated.")


def _valid_market_isin(prefix: str) -> str:
    body_length = 11 - len(prefix)
    body = f"{uuid4().int % (10**body_length):0{body_length}d}"
    for check_digit in range(10):
        candidate = f"{prefix}{body}{check_digit}"
        if isin_check_digit_ok(candidate):
            return candidate
    raise AssertionError("A synthetic ISIN check digit could not be generated.")


def test_0013_broker_migration_downgrades_and_upgrades(
    migrated_database: MigratedDatabase,
) -> None:
    database = migrated_database
    command.downgrade(database.config, "0012_indian_fund_check")
    with database.owner_engine.connect() as connection:
        assert connection.scalar(select(func.to_regclass("import_templates"))) is None
        assert connection.scalar(select(func.to_regclass("realised_trades"))) is None

    command.upgrade(database.config, "head")
    with database.owner_engine.connect() as connection:
        assert connection.scalar(select(func.to_regclass("import_templates"))) is not None
        assert connection.scalar(select(func.to_regclass("realised_trades"))) is not None


def test_0014_international_assets_migration_downgrades_and_upgrades(
    migrated_database: MigratedDatabase,
) -> None:
    database = migrated_database
    command.downgrade(database.config, "0013_broker_imports")
    with database.owner_engine.connect() as connection:
        assert connection.scalar(select(func.to_regclass("broker_cash_flows"))) is None
        columns = connection.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'securities'"
        ).scalars()
        assert "domicile" not in set(columns)

    command.upgrade(database.config, "head")
    with database.owner_engine.connect() as connection:
        assert connection.scalar(select(func.to_regclass("broker_cash_flows"))) is not None
        columns = connection.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'securities'"
        ).scalars()
        assert {"domicile", "exchange"} <= set(columns)


def test_icici_portfolio_matches_names_searches_and_accepts_owner_choice(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "icici-portfolio")
    user_id = _user_id(migrated_database, claims)
    symbols = {
        "Example Paints Limited": "SYNTH-ICICI-PAINT",
        "Sample and Sons Limited": "SYNTH-ICICI-SONS",
        "Nandi Wall's (India) Limited": "SYNTH-ICICI-NANDI",
        "Alpha Nifty 50 ETF": "SYNTH-ICICI-NIFTY",
        "Listed Synthetic Index Fund": "SYNTH-ICICI-PICK",
    }
    with system_session(migrated_database.owner_engine) as session:
        member = ensure_self_household_member(session, user_id)
        securities = [
            Security(
                market="IN",
                symbol=symbol,
                name=name,
                security_type="etf" if "ETF" in name or "Fund" in name else "equity",
                currency="INR",
                sleeve="equity_india",
                category="equity:india",
            )
            for name, symbol in symbols.items()
        ]
        foreign_security = Security(
            market="US",
            symbol="SYNTH-ICICI-US",
            name="Synthetic US security",
            security_type="equity",
            currency="USD",
            sleeve="equity_global",
        )
        session.add_all([*securities, foreign_security])
        session.flush()
        paint = next(
            security for security in securities if security.name == "Example Paints Limited"
        )
        paint_id = paint.id
        session.add(
            HoldingLot(
                user_id=user_id,
                member_id=member.id,
                security_id=paint.id,
                quantity=Decimal("10"),
                source="cas",
            )
        )
        selected_etf_id = next(
            security.id for security in securities if security.name == "Listed Synthetic Index Fund"
        )
        foreign_security_id = foreign_security.id
        member_id = member.id

    uploaded = _upload(
        client,
        headers,
        "icici_direct_portfolio.csv",
        _fixture("icici_direct_portfolio.csv"),
        broker="nordnet",
    )
    assert uploaded.status_code == 201, uploaded.text
    job_id = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert detail["broker"] == "icici_direct"
    assert detail["market_default"] == "IN"
    assert detail["currency_default"] == "INR"
    assert detail["detected_broker"] == "icici_direct"
    by_name = {item["name"]: item for item in detail["items"]}
    assert by_name["Example Paints Ltd."]["status"] == "enrich"
    assert by_name["Example Paints Ltd."]["matched_by"] == "name"
    assert by_name["Sample & Sons Ltd"]["status"] == "new"
    assert by_name["Nandi Walls (India) Limited"]["status"] == "new"
    assert by_name["Alpha Nifty50 ETF"]["status"] == "new"
    assert by_name["Mystery Basket ETF"]["status"] == "unknown"
    assert by_name["Mystery Basket ETF"]["can_find_security"] is True
    assert by_name["Alpha Nifty50 ETF"]["symbol"] == symbols["Alpha Nifty 50 ETF"]

    zerodha_headers = b"Symbol,Qty,Average Price,Exchange\nSYNTH-C61-VESTED,3,25.00,NASDAQ\n"
    zerodha_upload = _upload(
        client,
        headers,
        "synthetic-vested-holdings.csv",
        zerodha_headers,
        broker="vested",
    )
    assert zerodha_upload.status_code == 201, zerodha_upload.text
    zerodha_detail = client.get(
        f"/v1/imports/{zerodha_upload.json()['job_id']}", headers=headers
    ).json()
    assert zerodha_detail["broker"] == "vested"
    assert zerodha_detail["market_default"] == "US"
    assert zerodha_detail["currency_default"] == "USD"
    assert zerodha_detail["detected_broker"] is None

    search = client.get(
        "/v1/securities/search",
        headers=headers,
        params={"q": symbols["Listed Synthetic Index Fund"], "market": "IN"},
    )
    assert search.status_code == 200, search.text
    assert search.json()[0] == {
        "id": str(selected_etf_id),
        "symbol": symbols["Listed Synthetic Index Fund"],
        "name": "Listed Synthetic Index Fund",
        "security_type": "etf",
    }
    short_search = client.get(
        "/v1/securities/search", headers=headers, params={"q": " a ", "market": "IN"}
    )
    assert short_search.status_code == 422

    unknown_item_id = by_name["Mystery Basket ETF"]["id"]
    wrong_market = client.patch(
        f"/v1/imports/{job_id}/items/{unknown_item_id}",
        headers=headers,
        json={"security_id": str(foreign_security_id)},
    )
    assert wrong_market.status_code == 422
    selected = client.patch(
        f"/v1/imports/{job_id}/items/{unknown_item_id}",
        headers=headers,
        json={"security_id": str(selected_etf_id)},
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["status"] == "new"
    assert selected.json()["symbol"] == symbols["Listed Synthetic Index Fund"]
    assert selected.json()["matched_by"] == "owner"
    assert selected.json()["match"]["security_id"] == str(selected_etf_id)
    assert selected.json()["can_find_security"] is False

    with system_session(migrated_database.owner_engine) as session:
        nonbroker_job = ImportJob(
            user_id=user_id,
            member_id=member_id,
            doc_type="nsdl_cas",
            status="ready",
            source="nsdl",
            statement_date=_TODAY,
            counts={},
            parse_report={},
        )
        session.add(nonbroker_job)
        session.flush()
        nonbroker_item = ImportItem(
            user_id=user_id,
            job_id=nonbroker_job.id,
            item_kind="equity",
            payload={"market": "IN", "quantity": "1", "reason": "symbol_not_found"},
            status="unknown",
        )
        session.add(nonbroker_item)
        session.flush()
        nonbroker_job_id = nonbroker_job.id
        nonbroker_item_id = nonbroker_item.id
    nonbroker_patch = client.patch(
        f"/v1/imports/{nonbroker_job_id}/items/{nonbroker_item_id}",
        headers=headers,
        json={"security_id": str(selected_etf_id)},
    )
    assert nonbroker_patch.status_code == 422

    us_unknown_upload = _upload(
        client,
        headers,
        "synthetic-vested-unknown.csv",
        (b"Name,Quantity,Average Price,Last Price\nSynthetic Foreign Holding,2,10.00,12.00\n"),
        broker="vested",
    )
    assert us_unknown_upload.status_code == 201, us_unknown_upload.text
    us_unknown_detail = client.get(
        f"/v1/imports/{us_unknown_upload.json()['job_id']}", headers=headers
    ).json()
    us_unknown = us_unknown_detail["items"][0]
    assert us_unknown["status"] == "unknown"
    assert us_unknown["market"] == "US"
    assert us_unknown["reason"] == "symbol_not_found"
    assert us_unknown["can_find_security"] is False
    us_patch = client.patch(
        f"/v1/imports/{us_unknown_upload.json()['job_id']}/items/{us_unknown['id']}",
        headers=headers,
        json={"security_id": str(foreign_security_id)},
    )
    assert us_patch.status_code == 422

    confirmed = client.post(f"/v1/imports/{job_id}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text
    with system_session(migrated_database.owner_engine) as session:
        saved_lot = session.scalar(
            select(HoldingLot).where(
                HoldingLot.user_id == user_id,
                HoldingLot.security_id == paint_id,
                HoldingLot.source == "cas",
            )
        )
        assert saved_lot is not None
        assert saved_lot.cost_per_unit == Decimal("90.0000")

    tradebook = (
        b"Company Name,Trade Date,Trade Type,Quantity,Price,Trade ID\n"
        b"Sample & Sons Ltd,2026-10-01,Buy,2,50.00,SYNTH-ICICI-TRADE\n"
        b"Unlisted Synthetic Company,2026-10-02,Buy,1,20.00,SYNTH-ICICI-UNKNOWN\n"
    )
    tradebook_upload = _upload(
        client,
        headers,
        "synthetic-icici-tradebook.csv",
        tradebook,
        broker="icici_direct",
    )
    assert tradebook_upload.status_code == 201, tradebook_upload.text
    tradebook_detail = client.get(
        f"/v1/imports/{tradebook_upload.json()['job_id']}", headers=headers
    ).json()
    trade_item = next(
        item for item in tradebook_detail["items"] if item["name"] == "Sample & Sons Ltd"
    )
    assert trade_item["matched_by"] == "name"
    assert trade_item["symbol"] == symbols["Sample and Sons Limited"]
    unknown_trade = next(
        item for item in tradebook_detail["items"] if item["name"] == "Unlisted Synthetic Company"
    )
    assert unknown_trade["status"] == "unknown"
    assert unknown_trade["reason"] == "symbol_not_found"
    assert unknown_trade["can_find_security"] is False
    trade_patch = client.patch(
        f"/v1/imports/{tradebook_upload.json()['job_id']}/items/{unknown_trade['id']}",
        headers=headers,
        json={"security_id": str(selected_etf_id)},
    )
    assert trade_patch.status_code == 422


def test_saxo_upload_cleans_symbols_and_reuses_valid_isins(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "saxo-full")
    danish_isin = _valid_market_isin("DK")
    european_isin = _valid_market_isin("IE")
    statement = (
        "Symbol,ISIN,Name,Quantity,Open Price,Last Price,Currency\n"
        f"SYNTH B:xcse,{danish_isin},Synthetic Danish share,2,20.00,25.00,DKK\n"
        f"SYNTHF:xams,{european_isin},Synthetic European fund,3,30.00,35.00,EUR\n"
    ).encode()

    uploaded = _upload(client, headers, "synthetic-saxo.csv", statement, broker="saxo")
    assert uploaded.status_code == 201, uploaded.text
    first_job_id = uploaded.json()["job_id"]
    first = client.get(f"/v1/imports/{first_job_id}", headers=headers).json()
    assert first["auto_mapped"] is True
    assert first["has_missing_cost"] is False
    assert [item["status"] for item in first["items"]] == ["new", "new"]
    first_by_symbol = {item["symbol"]: item for item in first["items"]}
    assert {symbol: item["exchange"] for symbol, item in first_by_symbol.items()} == {
        "SYNTHB": "XCSE",
        "SYNTHF": "XAMS",
    }
    assert {symbol: item["statement_value"] for symbol, item in first_by_symbol.items()} == {
        "SYNTHB": "50.00",
        "SYNTHF": "105.00",
    }
    confirmed = client.post(f"/v1/imports/{first_job_id}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text
    first_confirmed_detail = client.get(f"/v1/imports/{first_job_id}", headers=headers).json()
    first_security_ids = {
        item["isin"]: item["match"]["security_id"] for item in first_confirmed_detail["items"]
    }

    repeated_upload = _upload(client, headers, "synthetic-saxo-again.csv", statement, broker="saxo")
    assert repeated_upload.status_code == 201, repeated_upload.text
    second_job_id = repeated_upload.json()["job_id"]
    second = client.get(f"/v1/imports/{second_job_id}", headers=headers).json()
    assert {
        item["isin"]: item["match"]["security_id"] for item in second["items"]
    } == first_security_ids
    assert [item["status"] for item in second["items"]] == ["duplicate", "duplicate"]
    assert second["counts"]["duplicate"] == 2

    second_confirmed = client.post(f"/v1/imports/{second_job_id}/confirm", headers=headers)
    assert second_confirmed.status_code == 200, second_confirmed.text

    changed_statement = (
        "Symbol,ISIN,Name,Quantity,Open Price,Last Price,Currency\n"
        f"SYNTH B:xcse,{danish_isin},Synthetic Danish share,3,20.00,25.00,DKK\n"
        f"SYNTHF:xams,{european_isin},Synthetic European fund,3,30.00,35.00,EUR\n"
    ).encode()
    changed_upload = _upload(
        client, headers, "synthetic-saxo-changed.csv", changed_statement, broker="saxo"
    )
    assert changed_upload.status_code == 201, changed_upload.text
    changed_job_id = changed_upload.json()["job_id"]
    changed = client.get(f"/v1/imports/{changed_job_id}", headers=headers).json()
    changed_by_isin = {item["isin"]: item for item in changed["items"]}
    assert changed["counts"]["changed"] == 1
    assert changed["counts"]["duplicate"] == 1
    assert changed_by_isin[danish_isin]["status"] == "changed"
    assert changed_by_isin[danish_isin]["diff"] == {"quantity": {"old": "2.000000", "new": "3"}}
    assert changed_by_isin[european_isin]["status"] == "duplicate"

    changed_confirmed = client.post(f"/v1/imports/{changed_job_id}/confirm", headers=headers)
    assert changed_confirmed.status_code == 200, changed_confirmed.text
    with system_session(migrated_database.owner_engine) as session:
        saved = list(
            session.scalars(select(Security).where(Security.isin.in_([danish_isin, european_isin])))
        )
        assert len(saved) == 2
        assert {security.isin for security in saved} == {danish_isin, european_isin}
        saved_lots = list(
            session.scalars(
                select(HoldingLot).where(
                    HoldingLot.user_id == _user_id(migrated_database, _claims),
                    HoldingLot.security_id.in_([security.id for security in saved]),
                    HoldingLot.source == "broker",
                )
            )
        )
        assert len(saved_lots) == 2
        assert {lot.quantity for lot in saved_lots} == {Decimal("3")}


@pytest.mark.parametrize("with_accounts", [False, True], ids=["no-accounts", "two-accounts"])
def test_broker_holdings_replace_only_lots_from_the_same_place(
    broker_api_client: tuple[TestClient, str],
    migrated_database: MigratedDatabase,
    with_accounts: bool,
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, f"broker-place-{with_accounts}")
    user_id = _user_id(migrated_database, claims)
    isin = _valid_synthetic_isin("SYNTHPLACE")
    account_ids: dict[str, UUID] = {}
    if with_accounts:
        with system_session(migrated_database.owner_engine) as session:
            member = ensure_self_household_member(session, user_id)
            zerodha_account = Account(
                user_id=user_id,
                member_id=member.id,
                institution="Synthetic Zerodha Custody",
                account_type="broker",
                masked_number="X1101",
            )
            hdfc_account = Account(
                user_id=user_id,
                member_id=member.id,
                institution="Synthetic HDFC Custody",
                account_type="broker",
                masked_number="X2202",
            )
            session.add_all([zerodha_account, hdfc_account])
            session.flush()
            account_ids = {"zerodha": zerodha_account.id, "hdfc_securities": hdfc_account.id}

    def upload_holding(broker: str, quantity: str) -> dict[str, Any]:
        statement = (
            "Symbol,ISIN,Name,Quantity,Average Price,Last Price,Currency\n"
            f"SYNTH,{isin},Synthetic shared holding,{quantity},10.00,12.00,INR\n"
        )
        fields = {"broker": broker}
        if broker in account_ids:
            fields["account_id"] = str(account_ids[broker])
        response = _upload(
            client,
            headers,
            f"synthetic-{broker}-{quantity}.csv",
            statement.encode("ascii"),
            **fields,
        )
        assert response.status_code == 201, response.text
        return response.json()

    zerodha = upload_holding("zerodha", "2")
    zerodha_job_id = zerodha["job_id"]
    zerodha_detail = client.get(f"/v1/imports/{zerodha_job_id}", headers=headers).json()
    assert zerodha_detail["items"][0]["status"] == "new"
    assert client.post(f"/v1/imports/{zerodha_job_id}/confirm", headers=headers).status_code == 200

    hdfc = upload_holding("hdfc_securities", "5")
    hdfc_job_id = hdfc["job_id"]
    hdfc_detail = client.get(f"/v1/imports/{hdfc_job_id}", headers=headers).json()
    assert hdfc_detail["items"][0]["status"] == "new"
    assert hdfc_detail["broker"] == "hdfc_securities"
    expected_hdfc_account = account_ids.get("hdfc_securities")
    assert hdfc_detail["account_id"] == (
        str(expected_hdfc_account) if expected_hdfc_account is not None else None
    )
    assert client.post(f"/v1/imports/{hdfc_job_id}/confirm", headers=headers).status_code == 200

    money = client.get("/v1/networth", headers=headers)
    assert money.status_code == 200, money.text
    shared_rows = [row for row in money.json()["holdings"] if row["symbol"] == "SYNTH"]
    assert sum(Decimal(row["quantity"]) for row in shared_rows) == Decimal("7")

    changed = upload_holding("zerodha", "3")
    changed_job_id = changed["job_id"]
    changed_detail = client.get(f"/v1/imports/{changed_job_id}", headers=headers).json()
    assert changed_detail["items"][0]["status"] == "changed"
    assert changed_detail["items"][0]["diff"] == {"quantity": {"old": "2.000000", "new": "3.00"}}
    assert client.post(f"/v1/imports/{changed_job_id}/confirm", headers=headers).status_code == 200

    with system_session(migrated_database.owner_engine) as session:
        security = session.scalar(select(Security).where(Security.isin == isin))
        assert security is not None
        security_id = security.id
        lots = list(
            session.execute(
                select(HoldingLot, ImportJob.parse_report)
                .join(
                    ImportJob,
                    (ImportJob.id == HoldingLot.import_job_id)
                    & (ImportJob.user_id == HoldingLot.user_id),
                )
                .where(
                    HoldingLot.user_id == user_id,
                    HoldingLot.security_id == security_id,
                    HoldingLot.source == "broker",
                )
            )
        )
        actual = {(report["broker"], lot.account_id, lot.quantity) for lot, report in lots}
        expected = {
            ("zerodha", account_ids.get("zerodha"), Decimal("3")),
            ("hdfc_securities", account_ids.get("hdfc_securities"), Decimal("5")),
        }
        assert actual == expected

    tradebook = _fixture("zerodha_tradebook.csv").replace(b"IN00000000A3", isin.encode("ascii"))
    tradebook_fields = {"broker": "zerodha"}
    if with_accounts:
        tradebook_fields["account_id"] = str(account_ids["zerodha"])
    tradebook_upload = _upload(
        client,
        headers,
        "synthetic-zerodha-tradebook.csv",
        tradebook,
        **tradebook_fields,
    )
    assert tradebook_upload.status_code == 201, tradebook_upload.text
    tradebook_job_id = tradebook_upload.json()["job_id"]
    tradebook_detail = client.get(f"/v1/imports/{tradebook_job_id}", headers=headers).json()
    assert tradebook_detail["items"][0]["status"] == "replace_with_lots"
    assert (
        client.post(f"/v1/imports/{tradebook_job_id}/confirm", headers=headers).status_code == 200
    )

    with system_session(migrated_database.owner_engine) as session:
        lots = list(
            session.execute(
                select(HoldingLot, ImportJob.parse_report)
                .join(
                    ImportJob,
                    (ImportJob.id == HoldingLot.import_job_id)
                    & (ImportJob.user_id == HoldingLot.user_id),
                )
                .where(
                    HoldingLot.user_id == user_id,
                    HoldingLot.security_id == security_id,
                    HoldingLot.source == "broker",
                )
            )
        )
        by_broker = defaultdict(Decimal)
        for lot, report in lots:
            by_broker[report["broker"]] += lot.quantity
        assert by_broker == {"zerodha": Decimal("3"), "hdfc_securities": Decimal("5")}


def test_broker_rows_enrich_only_one_matching_cas_account(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "cas-account-safety")
    user_id = _user_id(migrated_database, claims)
    first_isin = _valid_synthetic_isin("SYNTHCASA")
    second_isin = _valid_synthetic_isin("SYNTHCASB")
    with system_session(migrated_database.owner_engine) as session:
        member = ensure_self_household_member(session, user_id)
        securities = [
            Security(
                market="IN",
                symbol=f"SYNTH-{uuid4().hex[:6].upper()}",
                isin=isin,
                name=f"Synthetic CAS security {index}",
                security_type="equity",
                currency="INR",
                sleeve="equity_india",
            )
            for index, isin in enumerate((first_isin, second_isin), start=1)
        ]
        session.add_all(securities)
        session.flush()
        lots = [
            HoldingLot(
                user_id=user_id,
                member_id=member.id,
                security_id=securities[0].id,
                quantity=Decimal("6"),
                cost_per_unit=None,
                source="cas",
                account_ref="XXXX1101",
            ),
            HoldingLot(
                user_id=user_id,
                member_id=member.id,
                security_id=securities[0].id,
                quantity=Decimal("4"),
                cost_per_unit=None,
                source="cas",
                account_ref="XXXX1101",
            ),
            HoldingLot(
                user_id=user_id,
                member_id=member.id,
                security_id=securities[0].id,
                quantity=Decimal("5"),
                cost_per_unit=Decimal("50"),
                source="cas",
                account_ref="XXXX2202",
            ),
            HoldingLot(
                user_id=user_id,
                member_id=member.id,
                security_id=securities[1].id,
                quantity=Decimal("10"),
                cost_per_unit=Decimal("60"),
                source="cas",
                account_ref="XXXX1101",
            ),
            HoldingLot(
                user_id=user_id,
                member_id=member.id,
                security_id=securities[1].id,
                quantity=Decimal("5"),
                cost_per_unit=Decimal("70"),
                source="cas",
                account_ref="XXXX2202",
            ),
        ]
        session.add_all(lots)
        session.flush()
        security_ids = [security.id for security in securities]

    statement = (
        "Symbol,ISIN,Name,Quantity,Average Price,Currency\n"
        f"SYNTHA,{first_isin},Synthetic CAS account A,10,12.50,INR\n"
        f"SYNTHB,{second_isin},Synthetic CAS account B,12,15.00,INR\n"
    ).encode("ascii")
    uploaded = _upload(client, headers, "synthetic-cas-accounts.csv", statement, broker="zerodha")
    assert uploaded.status_code == 201, uploaded.text
    job_id = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    items_by_isin = {item["isin"]: item for item in detail["items"]}
    assert items_by_isin[first_isin]["status"] == "enrich"
    assert items_by_isin[second_isin]["status"] == "unknown"
    assert items_by_isin[second_isin]["reason"] == "several_accounts"

    confirmed = client.post(f"/v1/imports/{job_id}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text
    with system_session(migrated_database.owner_engine) as session:
        stored = list(
            session.scalars(
                select(HoldingLot).where(
                    HoldingLot.user_id == user_id,
                    HoldingLot.security_id.in_(security_ids),
                    HoldingLot.source == "cas",
                )
            )
        )
        first_account = [lot for lot in stored if lot.security_id == security_ids[0]]
        second_account = [lot for lot in stored if lot.security_id == security_ids[1]]
        assert len(first_account) == 3
        assert [lot.cost_per_unit for lot in first_account if lot.account_ref == "XXXX1101"] == [
            Decimal("12.5000"),
            Decimal("12.5000"),
        ]
        assert [lot.cost_per_unit for lot in first_account if lot.account_ref == "XXXX2202"] == [
            Decimal("50.0000")
        ]
        assert {(lot.account_ref, lot.quantity, lot.cost_per_unit) for lot in second_account} == {
            ("XXXX1101", Decimal("10"), Decimal("60.0000")),
            ("XXXX2202", Decimal("5"), Decimal("70.0000")),
        }
        assert (
            session.scalar(
                select(func.count())
                .select_from(HoldingLot)
                .where(
                    HoldingLot.user_id == user_id,
                    HoldingLot.security_id == security_ids[1],
                    HoldingLot.source == "broker",
                )
            )
            == 0
        )


def test_same_place_broker_reimport_enriches_missing_purchase_price(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "broker-cost-enrich")
    user_id = _user_id(migrated_database, claims)
    isin = _valid_market_isin("US")
    without_cost = (
        "Symbol,ISIN,Name,Quantity,Current Price,Currency\n"
        f"SYNTHCOST,{isin},Synthetic missing cost,4,15.00,USD\n"
    ).encode("ascii")
    uploaded = _upload(client, headers, "synthetic-no-cost.csv", without_cost, broker="indmoney")
    assert uploaded.status_code == 201, uploaded.text
    first_job_id = uploaded.json()["job_id"]
    assert client.post(f"/v1/imports/{first_job_id}/confirm", headers=headers).status_code == 200

    with_cost = (
        "Symbol,ISIN,Name,Quantity,Average Price,Current Price,Currency\n"
        f"SYNTHCOST,{isin},Synthetic missing cost,4,10.00,15.00,USD\n"
    ).encode("ascii")
    repeated = _upload(client, headers, "synthetic-with-cost.csv", with_cost, broker="indmoney")
    assert repeated.status_code == 201, repeated.text
    job_id = repeated.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert detail["items"][0]["status"] == "enrich"
    assert detail["items"][0]["action"] == "enrich"
    assert client.post(f"/v1/imports/{job_id}/confirm", headers=headers).status_code == 200
    with system_session(migrated_database.owner_engine) as session:
        lot = session.scalar(
            select(HoldingLot).where(
                HoldingLot.user_id == user_id,
                HoldingLot.source == "broker",
                HoldingLot.security_id
                == session.scalar(select(Security.id).where(Security.isin == isin)),
            )
        )
        assert lot is not None
        assert lot.quantity == Decimal("4")
        assert lot.cost_per_unit == Decimal("10.0000")


def test_zerodha_prices_and_indian_etf_types_are_staged(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "zerodha-prices")
    stock_isin = _valid_synthetic_isin("SYNTHSTK")
    etf_isin = _valid_market_isin("INF")
    statement = (
        "Synthetic Zerodha export\nReport for a test household\n"
        "Symbol,ISIN,Sector,Quantity Available,Quantity Discrepant,Quantity Long Term,"
        "Quantity Pledged (Margin),Quantity Pledged (Loan),Average Price,"
        "Previous Closing Price,Unrealized P&L,Unrealized P&L Pct.\n"
        f"SYNTHSTK,{stock_isin},Equity,10,0,10,0,0,10.00,12.50,25.00,2.50\n"
        f"SYNTHETF,{etf_isin},ETF,50,0,50,0,0,100.00,105.00,250.00,5.00\n"
    ).encode()
    uploaded = _upload(client, headers, "synthetic-zerodha.csv", statement, broker="zerodha")
    assert uploaded.status_code == 201, uploaded.text
    job_id = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert [item["status"] for item in detail["items"]] == ["new", "new"]
    items_by_symbol = {item["symbol"]: item for item in detail["items"]}
    assert {symbol: item["statement_value"] for symbol, item in items_by_symbol.items()} == {
        "SYNTHSTK": "125.00",
        "SYNTHETF": "5250.00",
    }
    assert {symbol: item["security_type"] for symbol, item in items_by_symbol.items()} == {
        "SYNTHSTK": "equity",
        "SYNTHETF": "etf",
    }
    confirmed = client.post(f"/v1/imports/{job_id}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text

    with system_session(migrated_database.owner_engine) as session:
        fund_security = session.scalar(select(Security).where(Security.isin == etf_isin))
        assert fund_security is not None and fund_security.security_type == "etf"


def test_currency_mismatch_is_reported_and_bad_symbol_stays_unknown(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "currency-headers")
    foreign_isin = _valid_market_isin("IE")
    statement = (
        "Symbol,ISIN,Name,Quantity,Currency,Market Value (DKK)\n"
        f"SYNTHF,{foreign_isin},Synthetic European holding,2,EUR,100.00\n"
        "Bad symbol!,,Synthetic unknown holding,1,DKK,50.00\n"
    ).encode()
    uploaded = _upload(client, headers, "synthetic-currency.csv", statement, broker="saxo")
    assert uploaded.status_code == 201, uploaded.text
    detail = client.get(f"/v1/imports/{uploaded.json()['job_id']}", headers=headers).json()
    with system_session(migrated_database.owner_engine) as session:
        parsed_job = session.get(ImportJob, UUID(uploaded.json()["job_id"]))
        assert parsed_job is not None
        assert parsed_job.parse_report["currency_mismatch_values"] == 1
    assert detail["auto_mapped"] is True
    assert detail["columns_used"]["market_value"] == "Market Value (DKK)"
    assert detail["currency_mismatch_values"] == 1
    items_by_name = {item["name"]: item for item in detail["items"]}
    assert items_by_name["Synthetic European holding"]["statement_value"] is None
    assert items_by_name["Synthetic European holding"]["status"] == "new"
    assert items_by_name["Synthetic unknown holding"]["status"] == "unknown"


def test_nok_row_is_unknown_and_dkk_row_can_be_confirmed(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "unsupported-currency")
    nok_isin = _valid_market_isin("DK")
    dkk_isin = _valid_market_isin("DK")
    statement = (
        "Symbol,ISIN,Name,Quantity,Open Price,Currency\n"
        f"NOKROW,{nok_isin},Synthetic NOK row,2,20.00,NOK\n"
        f"DKROW,{dkk_isin},Synthetic DKK row,3,30.00,DKK\n"
    ).encode()
    uploaded = _upload(client, headers, "synthetic-currencies.csv", statement, broker="saxo")
    assert uploaded.status_code == 201, uploaded.text
    job_id = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    items_by_symbol = {item["symbol"]: item for item in detail["items"]}
    assert items_by_symbol["NOKROW"]["status"] == "unknown"
    assert items_by_symbol["NOKROW"]["currency"] == "NOK"
    assert items_by_symbol["DKROW"]["status"] == "new"
    assert items_by_symbol["DKROW"]["currency"] == "DKK"

    confirmed = client.post(f"/v1/imports/{job_id}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text
    with system_session(migrated_database.owner_engine) as session:
        assert session.scalar(select(Security).where(Security.isin == dkk_isin)) is not None
        assert session.scalar(select(Security).where(Security.isin == nok_isin)) is None


def test_valid_isins_use_safe_symbols_on_confirm_and_reimport(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "safe-symbols")
    first_isin = _valid_market_isin("DK")
    second_isin = _valid_market_isin("IE")
    statement = (
        "Symbol,ISIN,Name,Quantity,Open Price,Currency\n"
        f"SYNTH&B/X,{first_isin},Synthetic unusual symbol,2,20.00,DKK\n"
        f"SYNTHETICLONGSYMBOL12345,{second_isin},Synthetic long symbol,3,30.00,EUR\n"
    ).encode()
    uploaded = _upload(client, headers, "synthetic-safe-symbols.csv", statement, broker="saxo")
    assert uploaded.status_code == 201, uploaded.text
    first_job = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{first_job}", headers=headers).json()
    assert [item["status"] for item in detail["items"]] == ["new", "new"]
    confirmed = client.post(f"/v1/imports/{first_job}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text

    with system_session(migrated_database.owner_engine) as session:
        securities = {
            security.isin: security
            for security in session.scalars(
                select(Security).where(Security.isin.in_([first_isin, second_isin]))
            )
        }
        assert set(securities) == {first_isin, second_isin}
        assert {isin: security.symbol for isin, security in securities.items()} == {
            first_isin: first_isin,
            second_isin: second_isin,
        }
        security_ids = {isin: str(security.id) for isin, security in securities.items()}

    reuploaded = _upload(client, headers, "synthetic-safe-symbols.csv", statement, broker="saxo")
    assert reuploaded.status_code == 201, reuploaded.text
    reimport = client.get(f"/v1/imports/{reuploaded.json()['job_id']}", headers=headers).json()
    assert {
        item["isin"]: item["match"]["security_id"] for item in reimport["items"]
    } == security_ids


def test_random_us_symbols_stage_as_new_broker_holdings(
    broker_api_client: tuple[TestClient, str],
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "random-us-symbols")
    generator = random.Random(58)
    symbols: set[str] = set()
    while len(symbols) < 320:
        symbols.add(f"SYNTH{generator.getrandbits(24):06X}")

    rows = [
        f"{symbol},Synthetic US share {index},3,90.00,40.00"
        for index, symbol in enumerate(sorted(symbols))
    ]
    statement = (
        "Symbol,Name,Quantity,Invested Value,Current Market Price\n" + "\n".join(rows) + "\n"
    ).encode("ascii")
    uploaded = _upload(
        client, headers, "synthetic-random-us-shares.csv", statement, broker="indmoney"
    )
    assert uploaded.status_code == 201, uploaded.text

    detail = client.get(f"/v1/imports/{uploaded.json()['job_id']}", headers=headers).json()
    assert detail["auto_mapped"] is True
    assert len(detail["items"]) == len(symbols)
    assert {item["symbol"] for item in detail["items"]} == symbols
    assert all(item["market"] == "US" and item["currency"] == "USD" for item in detail["items"])
    assert all(item["status"] == "new" for item in detail["items"])


def test_ibkr_generic_headers_use_the_selected_us_defaults(
    broker_api_client: tuple[TestClient, str],
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "ibkr-generic-layout")
    statement = b"Symbol,Quantity,Average Price,Exchange\nSYNTH,2,20.00,XNAS\n"
    uploaded = _upload(client, headers, "synthetic-ibkr.csv", statement, broker="ibkr")
    assert uploaded.status_code == 201, uploaded.text
    detail = client.get(f"/v1/imports/{uploaded.json()['job_id']}", headers=headers).json()
    assert detail["auto_mapped"] is True
    assert detail["items"][0]["status"] == "new"
    assert (detail["items"][0]["market"], detail["items"][0]["currency"]) == (
        "US",
        "USD",
    )


def test_item_response_uses_a_user_selected_kind_for_security_type() -> None:
    item = SimpleNamespace(
        id=uuid4(),
        payload={"quantity": "12.500", "security_type": "equity", "kind": "bond"},
        user_override={"kind": "mf"},
        item_kind="unknown",
        match_lot_id=None,
        match_security_id=None,
        status="accepted",
        diff=None,
    )

    response = _get_item_response(item)

    assert response["kind"] == "mf"
    assert response["security_type"] == "mf"


def test_unknown_file_auto_maps_once_and_reuses_household_template(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "mapping")
    unknown_file = _fixture("unknown_broker.csv")

    uploaded = _upload(client, headers, "synthetic.csv", unknown_file)
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] == "ready"
    job_id = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert detail["headers"] == [
        "Ticker",
        "Units",
        "Buy price",
        "Venue",
        "Security ID",
        "Average value",
    ]
    assert len(detail["samples"]) == 1
    assert detail["auto_mapped"] is True
    assert detail["mapping_kind"] == "holdings"
    assert detail["columns_used"]["symbol"] == "Ticker"
    assert detail["columns_used"]["quantity"] == "Units"
    sample = detail["samples"][0]
    assert "SYNTH" not in "".join(sample)
    assert all(
        character in {"x", "9"} or not character.isalnum()
        for value in sample
        for character in value
    )

    first_confirm = client.post(f"/v1/imports/{job_id}/confirm", headers=headers)
    assert first_confirm.status_code == 200, first_confirm.text

    second = _upload(client, headers, "synthetic.csv", unknown_file)
    assert second.status_code == 201, second.text
    assert second.json()["status"] == "ready"
    second_job = second.json()["job_id"]
    second_detail = client.get(f"/v1/imports/{second_job}", headers=headers).json()
    assert second_detail["auto_mapped"] is False
    confirmed = client.post(f"/v1/imports/{second_job}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text

    user_id = _user_id(migrated_database, claims)
    with system_session(migrated_database.owner_engine) as session:
        member = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert member is not None
        job = session.get(ImportJob, UUID(second_job))
        assert job is not None and job.member_id == member.id and job.account_id is None
        assert (
            session.scalar(
                select(func.count())
                .select_from(ImportTemplate)
                .where(ImportTemplate.user_id == user_id)
            )
            == 1
        )
        template = session.scalar(select(ImportTemplate).where(ImportTemplate.user_id == user_id))
        assert template is not None and template.name == "Other"
        lot = session.scalar(
            select(HoldingLot).where(
                HoldingLot.user_id == user_id,
                HoldingLot.source == "broker",
            )
        )
        assert lot is not None and lot.member_id == member.id
        assert lot.quantity == Decimal("20.000000")


def test_broker_pdf_and_unknown_broker_return_friendly_errors(
    broker_api_client: tuple[TestClient, str],
) -> None:
    client, secret = broker_api_client
    headers, _claims = _identity(secret, "upload-errors")

    pdf = client.post(
        "/v1/imports",
        headers=headers,
        files={"file": ("synthetic.pdf", b"%PDF-1.4 synthetic statement", "application/pdf")},
        data={"doc_type": "broker_statement", "broker": "nordnet"},
    )
    assert pdf.status_code == 415
    assert pdf.json()["code"] == "broker_pdf"
    assert pdf.json()["detail"] == (
        "This looks like a PDF. Please download the Excel or CSV version from your broker."
    )

    unknown_broker = client.post(
        "/v1/imports",
        headers=headers,
        files={
            "file": (
                "synthetic.csv",
                b"Symbol,Quantity,Name\nSYNTH,2,Synthetic share\n",
                "text/csv",
            )
        },
        data={"doc_type": "broker_statement", "broker": "unsupported"},
    )
    assert unknown_broker.status_code == 422
    assert unknown_broker.json()["code"] == "invalid_broker"


def test_holdings_without_cost_confirm_and_existing_isin_security_keeps_its_symbol(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "cost-and-isin")
    user_id = _user_id(migrated_database, claims)
    with system_session(migrated_database.owner_engine) as session:
        member = ensure_self_household_member(session, user_id)
        existing = Security(
            market="US",
            symbol="AAPL",
            isin="US0378331005",
            exchange="NASDAQ",
            name="Synthetic existing security",
            security_type="equity",
            currency="USD",
            sleeve="equity_global",
        )
        session.add(existing)
        session.flush()
        existing_security_id = existing.id

    no_cost_upload = _upload(
        client,
        headers,
        "synthetic-no-cost.csv",
        b"Symbol,Quantity,Name\nSYNTHCO,2,Synthetic company\n",
        broker="other",
    )
    assert no_cost_upload.status_code == 201, no_cost_upload.text
    no_cost_job = no_cost_upload.json()["job_id"]
    no_cost_detail = client.get(f"/v1/imports/{no_cost_job}", headers=headers).json()
    assert no_cost_detail["auto_mapped"] is True
    assert no_cost_detail["has_missing_cost"] is True
    no_cost_confirm = client.post(f"/v1/imports/{no_cost_job}/confirm", headers=headers)
    assert no_cost_confirm.status_code == 200, no_cost_confirm.text

    share_upload = _upload(
        client,
        headers,
        "synthetic-us-share.csv",
        ("Symbol,ISIN,Quantity,Avg Cost Price\nAAPL.O,US0378331005,2,100.00\n").encode("ascii"),
        broker="indmoney",
    )
    assert share_upload.status_code == 201, share_upload.text
    share_job = share_upload.json()["job_id"]
    share_detail = client.get(f"/v1/imports/{share_job}", headers=headers).json()
    assert share_detail["items"][0]["market"] == "US"
    assert share_detail["items"][0]["currency"] == "USD"
    share_confirm = client.post(f"/v1/imports/{share_job}/confirm", headers=headers)
    assert share_confirm.status_code == 200, share_confirm.text

    with system_session(migrated_database.owner_engine) as session:
        member = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == user_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert member is not None
        existing = session.get(Security, existing_security_id)
        assert existing is not None
        assert (existing.market, existing.symbol, existing.exchange) == (
            "US",
            "AAPL",
            "NASDAQ",
        )
        linked_lot = session.scalar(
            select(HoldingLot).where(
                HoldingLot.user_id == user_id,
                HoldingLot.security_id == existing_security_id,
                HoldingLot.source == "broker",
            )
        )
        assert linked_lot is not None and linked_lot.member_id == member.id
        no_cost_lot = session.scalar(
            select(HoldingLot)
            .join(Security)
            .where(
                HoldingLot.user_id == user_id,
                HoldingLot.source == "broker",
                Security.symbol == "SYNTHCO",
            )
        )
        assert no_cost_lot is not None and no_cost_lot.cost_per_unit is None


def test_holdings_enrich_and_tradebook_choices_preserve_manual_lots(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "merge")
    user_id = _user_id(migrated_database, claims)
    isin = _valid_synthetic_isin("SYNTH0001")
    with system_session(migrated_database.owner_engine) as session:
        member = ensure_self_household_member(session, user_id)
        security = Security(
            market="IN",
            symbol="SYNTH-ONE",
            isin=isin,
            name="Synthetic Share One",
            security_type="equity",
            currency="INR",
            sleeve="equity_india",
            nse_symbol="SYNTH",
        )
        session.add(security)
        session.flush()
        account = Account(
            user_id=user_id,
            member_id=member.id,
            institution="Synthetic Custodian",
            account_type="broker",
            masked_number="X0000",
        )
        session.add(account)
        cas_lot = HoldingLot(
            user_id=user_id,
            member_id=member.id,
            security_id=security.id,
            quantity=Decimal("20"),
            source="cas",
        )
        manual_lot = HoldingLot(
            user_id=user_id,
            member_id=member.id,
            security_id=security.id,
            quantity=Decimal("2"),
            source="manual",
        )
        session.add_all([cas_lot, manual_lot])
        session.flush()
        account_id, cas_lot_id, manual_lot_id, security_id = (
            account.id,
            cas_lot.id,
            manual_lot.id,
            security.id,
        )

    holdings = _fixture("zerodha_holdings.csv").replace(b"IN00000000A3", isin.encode("ascii"))
    holdings_upload = _upload(
        client,
        headers,
        "synthetic-holdings.csv",
        holdings,
        account_id=str(account_id),
    )
    assert holdings_upload.status_code == 201, holdings_upload.text
    holding_job = holdings_upload.json()["job_id"]
    holding_detail = client.get(f"/v1/imports/{holding_job}", headers=headers).json()
    assert holding_detail["items"][0]["action"] == "enrich"
    assert holding_detail["member_id"] is not None
    enriched = client.post(f"/v1/imports/{holding_job}/confirm", headers=headers)
    assert enriched.status_code == 200, enriched.text
    with system_session(migrated_database.owner_engine) as session:
        cas_lot = session.get(HoldingLot, cas_lot_id)
        assert cas_lot is not None
        assert cas_lot.source == "cas"
        assert cas_lot.cost_per_unit == Decimal("250.0000")
        assert cas_lot.account_id == account_id

    tradebook = _fixture("zerodha_tradebook.csv").replace(b"IN00000000A3", isin.encode("ascii"))
    keep_upload = _upload(
        client,
        headers,
        "synthetic-trades.csv",
        tradebook,
        account_id=str(account_id),
    )
    assert keep_upload.status_code == 201, keep_upload.text
    keep_job = keep_upload.json()["job_id"]
    mismatch = client.get(f"/v1/imports/{keep_job}", headers=headers).json()["items"][0]
    assert mismatch["status"] == "quantity_mismatch"
    assert mismatch["diff"] == {"current": "22.000000", "tradebook": "3.00"}
    missing_choice = client.post(f"/v1/imports/{keep_job}/confirm", headers=headers)
    assert missing_choice.status_code == 409
    assert missing_choice.json()["code"] == "quantity_choice_required"
    rejected = client.patch(
        f"/v1/imports/{keep_job}/items/{mismatch['id']}",
        headers=headers,
        json={"status": "rejected"},
    )
    assert rejected.status_code == 200
    assert client.post(f"/v1/imports/{keep_job}/confirm", headers=headers).status_code == 200
    with system_session(migrated_database.owner_engine) as session:
        assert session.get(HoldingLot, cas_lot_id) is not None
        assert session.get(HoldingLot, manual_lot_id) is not None
        assert (
            session.scalar(
                select(func.count())
                .select_from(RealisedTrade)
                .where(RealisedTrade.user_id == user_id)
            )
            == 0
        )

    use_upload = _upload(
        client,
        headers,
        "synthetic-trades.csv",
        tradebook,
        account_id=str(account_id),
    )
    assert use_upload.status_code == 201, use_upload.text
    use_job = use_upload.json()["job_id"]
    use_item = client.get(f"/v1/imports/{use_job}", headers=headers).json()["items"][0]
    accepted = client.patch(
        f"/v1/imports/{use_job}/items/{use_item['id']}",
        headers=headers,
        json={"status": "accepted"},
    )
    assert accepted.status_code == 200
    confirmed = client.post(f"/v1/imports/{use_job}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text

    with system_session(migrated_database.owner_engine) as session:
        assert session.get(HoldingLot, cas_lot_id) is None
        assert session.get(HoldingLot, manual_lot_id) is not None
        broker_lot = session.scalar(
            select(HoldingLot).where(
                HoldingLot.user_id == user_id,
                HoldingLot.security_id == security_id,
                HoldingLot.source == "broker",
            )
        )
        assert broker_lot is not None
        assert broker_lot.quantity == Decimal("3.000000")
        assert broker_lot.cost_per_unit == Decimal("120.0000")
        assert broker_lot.acquired_on == date(2024, 6, 1)
        gains = session.execute(
            select(RealisedTrade.holding_period, func.sum(RealisedTrade.gain))
            .where(
                RealisedTrade.user_id == user_id,
                RealisedTrade.member_id == broker_lot.member_id,
                RealisedTrade.account_id == account_id,
            )
            .group_by(RealisedTrade.holding_period)
        )
        assert dict(gains.all()) == {"LT": Decimal("500.00"), "ST": Decimal("60.00")}


def test_realised_trades_are_member_scoped_for_plan_and_fund_tax_contexts(
    migrated_database: MigratedDatabase,
) -> None:
    user_id = resolve_user(
        migrated_database.api_engine,
        f"synthetic-tax-{uuid4()}",
        f"synthetic-tax-{uuid4()}@example.test",
        "Synthetic Tax User",
    )
    today = _TODAY
    with system_session(migrated_database.owner_engine) as session:
        self_member = ensure_self_household_member(session, user_id)
        child = HouseholdMember(
            user_id=user_id,
            display_name="Synthetic Child",
            relationship="child",
        )
        session.add(child)
        security = Security(
            market="IN",
            symbol=f"SYNTH-TAX-{uuid4().hex[:6]}",
            isin=_valid_synthetic_isin("SYNTH0002"),
            name="Synthetic Tax Share",
            security_type="equity",
            currency="INR",
        )
        session.add(security)
        session.flush()
        job = ImportJob(
            user_id=user_id,
            member_id=self_member.id,
            source="broker",
            doc_type="broker_tradebook",
            status="confirmed",
            counts={},
        )
        session.add(job)
        session.flush()
        rows = [
            (self_member.id, date(2026, 4, 15), "LT", Decimal("120000")),
            (self_member.id, date(2026, 5, 15), "ST", Decimal("30000")),
            (self_member.id, date(2026, 3, 31), "LT", Decimal("999999")),
            (child.id, date(2026, 5, 15), "LT", Decimal("2000")),
        ]
        for member_id, trade_date, period, gain in rows:
            session.add(
                RealisedTrade(
                    user_id=user_id,
                    member_id=member_id,
                    import_job_id=job.id,
                    security_id=security.id,
                    trade_date=trade_date,
                    acquired_on=date(2025, 1, 1),
                    quantity=Decimal("1"),
                    cost_per_unit=Decimal("100"),
                    sale_price=Decimal("100") + gain,
                    gain=gain,
                    holding_period=period,
                    source_broker="synthetic",
                )
            )

        user_data, _market, _research = load_policy_inputs(session, user_id, "2026-W44", today)
        assert user_data.realised_trades_this_fy.equity_ltcg == Decimal("120000.00")
        assert user_data.realised_trades_this_fy.equity_stcg == Decimal("30000.00")
        profile = session.scalar(select(Profile).where(Profile.user_id == user_id))
        fund_members, _profiles = _member_rows(session, user_id, profile, {}, today)

        self_fund_member = next(
            member for member in fund_members if member.member_id == str(self_member.id)
        )
        child_fund_member = next(
            member for member in fund_members if member.member_id == str(child.id)
        )
        assert self_fund_member.tax_context.fy_realised_equity_ltcg == Decimal("120000.00")
        assert self_fund_member.tax_context.fy_realised_equity_stcg == Decimal("30000.00")
        assert child_fund_member.tax_context.fy_realised_equity_ltcg == Decimal("2000.00")

        lot = Lot(
            lot_id="synthetic-lot",
            tax_class="listed_equity",
            quantity=Decimal("10"),
            cost_per_unit=Decimal("100"),
            acquired_on=date(2024, 1, 1),
        )
        ledger = _FundSaleLedger({self_fund_member.member_id: self_fund_member})
        quote = ledger.quote(
            self_fund_member,
            [(lot, Decimal("1"), Decimal("200"))],
            today,
        )
        assert quote.ltcg_exemption_remaining == Decimal("4900.00")


def test_broker_rows_are_hidden_from_another_login(
    migrated_database: MigratedDatabase,
) -> None:
    owner_id = resolve_user(
        migrated_database.api_engine,
        f"synthetic-broker-owner-{uuid4()}",
        f"synthetic-owner-{uuid4()}@example.test",
        "Synthetic Owner",
    )
    other_id = resolve_user(
        migrated_database.api_engine,
        f"synthetic-broker-other-{uuid4()}",
        f"synthetic-other-{uuid4()}@example.test",
        "Synthetic Other",
    )
    with system_session(migrated_database.owner_engine) as session:
        member = ensure_self_household_member(session, owner_id)
        security = Security(
            market="IN",
            symbol=f"SYNTH-ISO-{uuid4().hex[:6]}",
            isin=_valid_synthetic_isin("SYNTH0003"),
            name="Synthetic Isolation Share",
            security_type="equity",
            currency="INR",
        )
        session.add(security)
        session.flush()
        job = ImportJob(
            user_id=owner_id,
            member_id=member.id,
            source="broker",
            doc_type="broker_tradebook",
            status="confirmed",
            counts={},
        )
        template = ImportTemplate(
            user_id=owner_id,
            name="Synthetic mapping",
            kind="holdings",
            header_fingerprint="c" * 64,
            header_names=["quantity", "symbol"],
            columns={"quantity": "Quantity", "symbol": "Symbol"},
        )
        session.add_all([job, template])
        session.flush()
        session.add(
            RealisedTrade(
                user_id=owner_id,
                member_id=member.id,
                import_job_id=job.id,
                security_id=security.id,
                trade_date=date(2026, 5, 1),
                acquired_on=date(2025, 1, 1),
                quantity=Decimal("1"),
                cost_per_unit=Decimal("100"),
                sale_price=Decimal("200"),
                gain=Decimal("100"),
                holding_period="LT",
                source_broker="synthetic",
            )
        )

    with user_session(migrated_database.api_engine, other_id) as session:
        assert session.scalar(select(ImportJob.id).where(ImportJob.user_id == owner_id)) is None
        assert (
            session.scalar(select(ImportTemplate.id).where(ImportTemplate.user_id == owner_id))
            is None
        )
        assert (
            session.scalar(select(RealisedTrade.id).where(RealisedTrade.user_id == owner_id))
            is None
        )


def test_danish_cash_import_prompts_once_and_saves_member_deposits(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "danish-cash")
    user_id = _user_id(migrated_database, claims)
    with system_session(migrated_database.owner_engine) as session:
        member = ensure_self_household_member(session, user_id)
        account = Account(
            user_id=user_id,
            member_id=member.id,
            institution="Nordnet",
            account_type="broker",
            masked_number="X0000",
        )
        session.add(account)
        session.flush()
        account_id = account.id

    transactions = (
        b"Transaction date,Activity,Amount,Currency,Reference\n"
        b'2026-05-01,Deposit,"5.000,25",DKK,SYNTH-DEP-1\n'
    )
    uploaded = _upload(
        client,
        headers,
        "synthetic-transactions.csv",
        transactions,
        broker="nordnet",
        account_id=str(account_id),
    )
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] == "needs_mapping"
    job_id = uploaded.json()["job_id"]
    initial = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert initial["member_id"] is not None
    assert initial["tax_wrapper_question"] is True
    assert initial["market_default"] == "DK"
    assert initial["currency_default"] == "DKK"

    mapping = {
        "kind": "tradebook",
        "broker": "nordnet",
        "market_default": "DK",
        "currency_default": "DKK",
        "columns": {
            "trade_date": "Transaction date",
            "trade_type": "Activity",
            "amount": "Amount",
            "currency": "Currency",
            "trade_id": "Reference",
        },
    }
    mapped = client.post(f"/v1/imports/{job_id}/mapping", headers=headers, json=mapping)
    assert mapped.status_code == 200, mapped.text
    assert mapped.json()["cash_flow_count"] == 1
    assert mapped.json()["tax_wrapper_question"] is True
    confirmed = client.post(
        f"/v1/imports/{job_id}/confirm",
        headers=headers,
        json={"tax_wrapper": "ask"},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["counts"]["cash_flows"] == 1

    repeated_upload = _upload(
        client,
        headers,
        "synthetic-transactions.csv",
        transactions,
        broker="nordnet",
        account_id=str(account_id),
    )
    assert repeated_upload.status_code == 201, repeated_upload.text
    assert repeated_upload.json()["status"] == "needs_mapping"
    repeated_job_id = repeated_upload.json()["job_id"]
    repeated_mapping = client.post(
        f"/v1/imports/{repeated_job_id}/mapping", headers=headers, json=mapping
    )
    assert repeated_mapping.status_code == 200, repeated_mapping.text
    repeated_confirm = client.post(
        f"/v1/imports/{repeated_job_id}/confirm", headers=headers, json={}
    )
    assert repeated_confirm.status_code == 200, repeated_confirm.text
    assert "cash_flows" not in repeated_confirm.json()["counts"]

    with system_session(migrated_database.owner_engine) as session:
        stored_account = session.get(Account, account_id)
        flow = session.scalar(
            select(BrokerCashFlowRecord).where(
                BrokerCashFlowRecord.user_id == user_id,
                BrokerCashFlowRecord.account_id == account_id,
            )
        )
        assert stored_account is not None and stored_account.tax_wrapper == "ask"
        assert flow is not None
        assert flow.kind == "deposit"
        assert flow.amount == Decimal("5000.2500")
        assert flow.currency == "DKK"
        assert flow.member_id == UUID(initial["member_id"])
        assert flow.transaction_date == date(2026, 5, 1)
        assert (
            session.scalar(
                select(func.count())
                .select_from(BrokerCashFlowRecord)
                .where(BrokerCashFlowRecord.user_id == user_id)
            )
            == 1
        )


def test_ibkr_import_keeps_native_lots_and_converts_using_dated_fx(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "ibkr-fx")
    user_id = _user_id(migrated_database, claims)
    rates = (
        (date(2024, 1, 1), Decimal("83")),
        (date(2026, 1, 2), Decimal("85")),
        (date(2026, 10, 27), Decimal("86")),
    )
    with system_session(migrated_database.owner_engine) as session:
        for as_of, rate in rates:
            session.add(
                FxRate(
                    pair="USD/INR",
                    as_of=as_of,
                    rate=rate,
                    source="synthetic_test",
                    source_url="https://example.test/fx",
                    publication_lag_days=0,
                )
            )

    statement = (
        b"Open Positions,Header,Asset Category,Symbol,Currency,Quantity,Cost Price,"
        b"Close Price,Value,Listing Exchange\n"
        b"Open Positions,Data,Stocks,VOO,USD,7,400,500,3500,NYSE\n"
        b"Trades,Header,Asset Category,Symbol,Date/Time,Quantity,T. Price,Proceeds,"
        b"Comm/Fee,Currency,Listing Exchange,Trade ID\n"
        b"Trades,Data,Stocks,VOO,2024-01-01; 10:00:00,10,400,-4000,-1,USD,NYSE,BUY-1\n"
        b"Trades,Data,Stocks,VOO,2026-01-02; 10:00:00,-4,500,2000,-2,USD,NYSE,SELL-1\n"
        b"Trades,Data,Stocks,VOO,2026-08-01; 10:00:00,1,450,-450,0,USD,NYSE,BUY-2\n"
    )
    uploaded = _upload(client, headers, "synthetic-activity.csv", statement, broker="ibkr")
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] == "ready"
    job_id = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert detail["member_id"] is not None
    assert detail["items"][0]["status"] == "quantity_mismatch"
    assert detail["items"][0]["symbol"] == "VOO"
    with system_session(migrated_database.owner_engine) as session:
        assert session.scalar(
            text("SELECT :symbol ~ :pattern"),
            {"symbol": "VOO", "pattern": r"^[A-Z0-9.\-]{1,15}$"},
        )
    selected = client.patch(
        f"/v1/imports/{job_id}/items/{detail['items'][0]['id']}",
        headers=headers,
        json={"status": "accepted"},
    )
    assert selected.status_code == 200, selected.text
    confirmed = client.post(f"/v1/imports/{job_id}/confirm", headers=headers)
    assert confirmed.status_code == 200, confirmed.text

    with system_session(migrated_database.owner_engine) as session:
        security = session.scalar(select(Security).where(Security.symbol == "VOO"))
        lots = list(
            session.scalars(
                select(HoldingLot)
                .where(HoldingLot.user_id == user_id)
                .order_by(HoldingLot.acquired_on)
            )
        )
        realised = session.scalar(select(RealisedTrade).where(RealisedTrade.user_id == user_id))
        assert security is not None
        assert security.market == "US" and security.currency == "USD"
        assert security.domicile == "US" and security.sleeve == "equity_global"
        assert len(lots) == 2
        assert lots[0].currency == "USD"
        assert lots[0].cost_per_unit == Decimal("400.0000")
        assert lots[0].fx_at_acquisition == Decimal("83.000000")
        assert lots[0].cost_per_unit_inr == Decimal("33200.0000")
        assert lots[0].fx_missing is False
        assert lots[1].fx_at_acquisition is None and lots[1].fx_missing is True
        assert realised is not None
        assert realised.currency == "USD"
        assert realised.fx_at_acquisition == Decimal("83.000000")
        assert realised.fx_at_sale == Decimal("85.000000")
        assert realised.gain_inr == Decimal("37030.00")

    networth = client.get("/v1/networth", headers=headers)
    assert networth.status_code == 200, networth.text
    voo_rows = [row for row in networth.json()["holdings"] if row["symbol"] == "VOO"]
    assert voo_rows
    statement_row = next(row for row in voo_rows if row["value_usd"] is not None)
    assert statement_row["value_usd"] == "3500.00"
    assert statement_row["value"] == "301000.00"
    assert statement_row["currency"] == "USD"
    assert statement_row["fx_as_of"] == "2026-10-27"
