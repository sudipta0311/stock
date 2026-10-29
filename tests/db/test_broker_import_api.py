"""Broker upload mapping, household ownership, lot replacement, and tax feeds."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import jwt
import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from conftest import MigratedDatabase
from wealth.api.fund_check import _member_rows
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
        database_url="postgresql://wealth_api@localhost/wealth_test",
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
    command.downgrade(database.config, "-1")
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


def test_unknown_file_maps_once_and_reuses_household_template(
    broker_api_client: tuple[TestClient, str], migrated_database: MigratedDatabase
) -> None:
    client, secret = broker_api_client
    headers, claims = _identity(secret, "mapping")
    unknown_file = _fixture("unknown_broker.csv")

    uploaded = _upload(client, headers, "synthetic.csv", unknown_file)
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["status"] == "needs_mapping"
    job_id = uploaded.json()["job_id"]
    detail = client.get(f"/v1/imports/{job_id}", headers=headers).json()
    assert detail["headers"] == [
        "ticker",
        "units",
        "buy price",
        "venue",
        "security id",
        "average value",
    ]
    assert len(detail["samples"]) == 1
    sample = detail["samples"][0]
    assert "SYNTH" not in "".join(sample)
    assert all(
        character in {"x", "9"} or not character.isalnum()
        for value in sample
        for character in value
    )

    mapped = client.post(
        f"/v1/imports/{job_id}/mapping",
        headers=headers,
        json={
            "kind": "holdings",
            "columns": {
                "isin": "security id",
                "symbol": "ticker",
                "quantity": "units",
                "avg_price": "buy price",
            },
            "save_as": "Synthetic holdings",
        },
    )
    assert mapped.status_code == 200, mapped.text
    assert mapped.json()["status"] == "ready"
    assert mapped.json()["items"][0]["status"] == "new"

    second = _upload(client, headers, "synthetic.csv", unknown_file)
    assert second.status_code == 201, second.text
    assert second.json()["status"] == "ready"
    assert second.json()["counts"]["new"] == 1
    second_job = second.json()["job_id"]
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
        lot = session.scalar(
            select(HoldingLot).where(
                HoldingLot.user_id == user_id,
                HoldingLot.source == "broker",
            )
        )
        assert lot is not None and lot.member_id == member.id
        assert lot.quantity == Decimal("20.000000")


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
