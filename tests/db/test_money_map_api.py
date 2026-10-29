"""Money Map API behaviour over the C05 non-owner PostgreSQL login."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text, update

from conftest import MigratedDatabase
from wealth.api.main import create_app
from wealth.api.money_map import get_today
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import FxRate, HoldingLot, ManualAsset, NetworthSnapshot, User
from wealth.db.session import system_session

pytestmark = pytest.mark.db


@pytest.fixture
def api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, object]]:
    secret = secrets.token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    application = create_app(settings)
    with TestClient(application) as client:
        yield client, secret, application


def _headers(secret: str, label: str) -> dict[str, str]:
    token = jwt.encode(
        {
            "sub": f"synthetic-money-map-{label}-{uuid4()}",
            "email": f"{label}-{uuid4()}@example.test",
            "name": "Synthetic Money Map User",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def test_member_account_asset_goal_and_quick_add_endpoints(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers = _headers(secret, "endpoints")

    me = client.get("/v1/me", headers=headers)
    assert me.status_code == 200
    members = client.get("/v1/members", headers=headers)
    assert len(members.json()) == 1
    self_member_id = members.json()[0]["id"]
    assert members.json()[0]["relationship"] == "self"

    member = client.post(
        "/v1/members",
        headers=headers,
        json={"display_name": "Synthetic Parent", "relationship": "parent"},
    )
    assert member.status_code == 201
    member_id = member.json()["id"]
    changed_member = client.patch(
        f"/v1/members/{member_id}", headers=headers, json={"display_name": "Synthetic Relative"}
    )
    assert changed_member.status_code == 200
    assert changed_member.json()["display_name"] == "Synthetic Relative"

    bad_account = client.post(
        "/v1/accounts",
        headers=headers,
        json={
            "institution": "Synthetic Bank",
            "account_type": "bank",
            "masked_number": "12345",
        },
    )
    assert bad_account.status_code == 422
    account = client.post(
        "/v1/accounts",
        headers=headers,
        json={
            "institution": "Synthetic Bank",
            "account_type": "bank",
            "masked_number": "X1234",
            "member_id": self_member_id,
        },
    )
    assert account.status_code == 201
    account_id = account.json()["id"]
    assert len(client.get("/v1/accounts", headers=headers).json()) == 1
    assert (
        client.patch(
            f"/v1/accounts/{account_id}", headers=headers, json={"masked_number": "XX5678"}
        ).status_code
        == 200
    )
    assert client.delete(f"/v1/accounts/{account_id}", headers=headers).status_code == 204

    asset = client.post(
        "/v1/assets",
        headers=headers,
        json={
            "asset_type": "savings",
            "bank": "Synthetic Bank",
            "balance": "500000",
            "member_id": self_member_id,
        },
    )
    assert asset.status_code == 201
    asset_id = asset.json()["id"]
    assert asset.json()["valuation"]["value_today"] == "500000.00"
    assert asset.json()["valuation_status"] == "valued"
    assert len(client.get("/v1/assets", headers=headers).json()) == 1
    filtered_assets = client.get(
        f"/v1/assets?asset_type=savings&member_id={self_member_id}", headers=headers
    )
    assert [item["id"] for item in filtered_assets.json()] == [asset_id]
    gold = client.post(
        "/v1/assets",
        headers=headers,
        json={"asset_type": "gold", "kind": "physical", "grams": "20", "purity": 22},
    )
    assert gold.status_code == 201
    assert gold.json()["valuation"] is None
    assert gold.json()["valuation_note"] == "price_needed"
    assert gold.json()["valuation_status"] == "price_needed"
    unvalued = client.get("/v1/networth", headers=headers).json()["unvalued_assets"]
    assert unvalued == [
        {
            "id": gold.json()["id"],
            "asset_type": "gold",
            "label": "gold",
            "reason": "price_needed",
        }
    ]

    updated = client.patch(
        f"/v1/assets/{asset_id}", headers=headers, json={"bank": "Synthetic Credit Union"}
    )
    assert updated.status_code == 200
    detail = client.get(f"/v1/assets/{asset_id}", headers=headers)
    assert detail.status_code == 200
    assert len(detail.json()["history"]) == 1
    history = detail.json()["history"][0]["changed_fields"]
    assert list(history) == ["bank"]
    assert history["bank"] == {"old": "Synthetic Bank", "new": "Synthetic Credit Union"}
    assert client.delete(f"/v1/assets/{asset_id}", headers=headers).status_code == 204
    assert all(item["id"] != asset_id for item in client.get("/v1/assets", headers=headers).json())
    assert [
        item["id"] for item in client.get("/v1/assets?status=closed", headers=headers).json()
    ] == [asset_id]
    assert client.delete(f"/v1/members/{member_id}", headers=headers).status_code == 204

    blocked_member = client.delete(f"/v1/members/{self_member_id}", headers=headers)
    assert blocked_member.status_code == 409
    assert blocked_member.json()["detail"] == "The self member cannot be deleted."
    assert client.delete(f"/v1/assets/{asset_id}?hard=true", headers=headers).status_code == 204
    assert (
        client.delete(f"/v1/assets/{gold.json()['id']}?hard=true", headers=headers).status_code
        == 204
    )
    assert client.delete(f"/v1/members/{self_member_id}", headers=headers).status_code == 409

    goal = client.post(
        "/v1/goals",
        headers=headers,
        json={
            "name": "Synthetic Home",
            "goal_type": "home",
            "target_amount": "25L",
            "target_year": 2035,
            "currency": "INR",
            "priority": 2,
        },
    )
    assert goal.status_code == 201
    goal_id = goal.json()["id"]
    assert goal.json()["target_amount"] == "2500000.00"
    assert client.get("/v1/goals", headers=headers).json()[0]["id"] == goal_id
    assert client.get(f"/v1/goals/{goal_id}", headers=headers).status_code == 200
    changed_goal = client.patch(
        f"/v1/goals/{goal_id}", headers=headers, json={"target_amount": "30L"}
    )
    assert changed_goal.json()["target_amount"] == "3000000.00"
    assert client.delete(f"/v1/goals/{goal_id}", headers=headers).status_code == 204

    checklist = client.put(
        "/v1/me/checklist", headers=headers, json={"categories": ["epf", "fd", "insurance"]}
    )
    assert checklist.status_code == 200
    assert checklist.json()["categories"] == ["epf", "fd", "insurance"]
    parsed = client.post(
        "/v1/assets/quick-add", headers=headers, json={"text": "FD 5L SBI 7.1% till Mar 2027"}
    )
    assert parsed.status_code == 200
    assert parsed.json()["draft"]["principal"] == "500000.00"
    not_understood = client.post(
        "/v1/assets/quick-add", headers=headers, json={"text": "random words"}
    )
    assert not_understood.status_code == 422
    assert not_understood.json()["code"] == "not_understood"


def test_member_residency_changes_plan_and_remains_tenant_scoped(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers_a = _headers(secret, "household-a")
    headers_b = _headers(secret, "household-b")

    initial = client.get("/v1/plan", headers=headers_a)
    assert initial.status_code == 200
    spouse = client.post(
        "/v1/members",
        headers=headers_a,
        json={
            "display_name": "Synthetic Spouse",
            "relationship": "spouse",
            "residency": {"status": "RI", "country": "IN"},
            "tax": {
                "tax_regime": "new",
                "tax_slab_pct": "20",
                "annual_income": "1000000",
                "annual_income_currency": "INR",
                "employment_type": "salaried",
            },
        },
    )
    assert spouse.status_code == 201
    spouse_id = spouse.json()["id"]
    assert spouse.json()["is_self"] is False

    after_create = client.get("/v1/plan", headers=headers_a)
    assert after_create.status_code == 200
    assert after_create.json()["inputs_digest"] != initial.json()["inputs_digest"]
    assert [item["id"] for item in after_create.json()["members"]] == [
        item["id"] for item in client.get("/v1/members", headers=headers_a).json()
    ]

    changed = client.patch(
        f"/v1/members/{spouse_id}",
        headers=headers_a,
        json={"residency": {"status": "NRI", "country": "DK"}},
    )
    assert changed.status_code == 200
    assert changed.json()["residency"] == {
        "status": "NRI",
        "country": "DK",
        "us_person": False,
        "planned_return_year": None,
        "settle_plan": "undecided",
    }
    after_residency_change = client.get("/v1/plan", headers=headers_a)
    assert after_residency_change.status_code == 200
    assert after_residency_change.json()["inputs_digest"] != after_create.json()["inputs_digest"]
    spouse_plan = next(
        item for item in after_residency_change.json()["members"] if item["id"] == spouse_id
    )
    assert "dk_aktiesparekonto" in spouse_plan["notes"]

    other_members = client.get("/v1/members", headers=headers_b)
    assert other_members.status_code == 200
    assert all(item["id"] != spouse_id for item in other_members.json())
    assert (
        client.patch(
            f"/v1/members/{spouse_id}",
            headers=headers_b,
            json={"display_name": "Synthetic Other"},
        ).status_code
        == 404
    )


def test_rate_unavailable_returns_certificate_rate_problem(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    response = client.post(
        "/v1/assets",
        headers=_headers(secret, "rate-needed"),
        json={
            "asset_type": "po_scheme",
            "scheme": "nsc",
            "amount": "50000",
            "purchase": "2025-01",
        },
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "rate_needed"
    assert response.json()["field"] == "rate_pct"


def test_networth_snapshot_calculates_sleeves_debt_liquidity_and_calendar(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, application = api_client
    headers = _headers(secret, "networth")
    as_of = date(2027, 2, 1)
    application.dependency_overrides[get_today] = lambda: as_of

    payloads = [
        {"asset_type": "savings", "bank": "Synthetic Bank", "balance": "600000"},
        {"asset_type": "epf", "balance": "1500000", "balance_as_of": "2027-02"},
        {
            "asset_type": "ppf",
            "balance": "500000",
            "opened_fy_start_year": 2019,
            "balance_as_of": "2027-02",
        },
        {
            "asset_type": "fd",
            "bank": "Synthetic Bank",
            "principal": "400000",
            "rate_pct": "7",
            "start": "2026-01",
            "maturity": "2028-01",
        },
        {
            "asset_type": "fd",
            "bank": "Synthetic Bank",
            "principal": "1000",
            "rate_pct": "7",
            "start": "2026-01",
            "maturity": "2029-01",
        },
        {
            "asset_type": "term_insurance",
            "insurer": "Synthetic Life",
            "cover": "1cr",
            "ends_year": 2040,
        },
        {
            "asset_type": "loan",
            "loan_kind": "home",
            "outstanding": "2000000",
            "outstanding_as_of": "2027-02",
            "rate_pct": "8.5",
        },
    ]
    created = [client.post("/v1/assets", headers=headers, json=payload) for payload in payloads]
    assert all(response.status_code == 201 for response in created)
    assert created[5].json()["valuation_status"] == "not_applicable"
    assert created[6].json()["valuation_status"] == "not_applicable"
    stale_asset_id = created[0].json()["id"]
    with migrated_database.owner_engine.begin() as connection:
        connection.execute(
            update(ManualAsset)
            .where(ManualAsset.id == stale_asset_id)
            .values(value_cached=Decimal("1.00"), value_as_of=date(2027, 1, 1))
        )
    asset_values = [Decimal(response.json()["value_cached"] or "0.00") for response in created]
    response = client.get("/v1/networth", headers=headers)

    assert response.status_code == 200
    networth = response.json()
    assets_total = sum(asset_values, Decimal(0))
    assert networth["assets_total"] == f"{assets_total:.2f}"
    assert networth["liabilities_total"] == "2000000.00"
    assert networth["net_worth"] == f"{assets_total - Decimal('2000000.00'):.2f}"
    assert networth["liquidity_ladder"]["instant"] == "600000.00"
    debt_amount = sum(asset_values[1:5], Decimal(0))
    expected_debt_pct = (debt_amount * 100 / assets_total).quantize(Decimal("0.01"))
    assert networth["debt_insight"]["debt_pct_including_locked"] == f"{expected_debt_pct:.2f}"
    assert networth["debt_insight"]["locked_debt_amount"] == f"{debt_amount:.2f}"
    assert networth["debt_insight"]["message_key"] == "debt_share"
    maturity_events = [
        event for event in networth["maturity_calendar"] if event["kind"] == "maturity"
    ]
    maturities = {event["date"] for event in maturity_events}
    assert "2028-01-01" in maturities
    assert "2029-01-01" not in maturities
    assert networth["as_of"] == as_of.isoformat()

    with migrated_database.owner_engine.connect() as connection:
        rows = connection.scalar(
            select(func.count())
            .select_from(NetworthSnapshot)
            .where(NetworthSnapshot.on_date == as_of)
        )
        refreshed = connection.scalar(
            select(ManualAsset.value_as_of).where(ManualAsset.id == stale_asset_id)
        )
    assert rows == 1
    assert refreshed == as_of
    assert client.get("/v1/networth", headers=headers).status_code == 200
    with migrated_database.owner_engine.connect() as connection:
        assert (
            connection.scalar(
                select(func.count())
                .select_from(NetworthSnapshot)
                .where(NetworthSnapshot.on_date == as_of)
            )
            == 1
        )


def test_completeness_uses_ticked_categories_with_active_assets(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, application = api_client
    headers = _headers(secret, "completeness")
    application.dependency_overrides[get_today] = lambda: date(2027, 2, 1)
    assert (
        client.put(
            "/v1/me/checklist",
            headers=headers,
            json={"categories": ["epf", "fd", "insurance"]},
        ).status_code
        == 200
    )
    assets = [
        {"asset_type": "epf", "balance": "100000", "balance_as_of": "2027-02"},
        {
            "asset_type": "fd",
            "bank": "Synthetic Bank",
            "principal": "10000",
            "rate_pct": "7",
            "maturity": "2028-02",
        },
    ]
    assert all(
        client.post("/v1/assets", headers=headers, json=asset).status_code == 201
        for asset in assets
    )

    response = client.get("/v1/networth", headers=headers)
    assert response.json()["completeness"] == {
        "ticked": ["epf", "fd", "insurance"],
        "done": ["epf", "fd"],
        "pct": "66.67",
    }


@pytest.mark.parametrize(
    ("market", "currency", "isin", "symbol"),
    [
        pytest.param("US", "INR", "US0000000001", "SYNTHETIC-US", id="us-market"),
        pytest.param("IN", "USD", "IN0000000001", "SYNTHETIC-IN-USD", id="usd-currency"),
    ],
)
def test_us_market_or_usd_holding_satisfies_us_completeness_tile(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
    market: str,
    currency: str,
    isin: str,
    symbol: str,
) -> None:
    client, secret, _application = api_client
    auth_subject = f"synthetic-money-map-us-{uuid4()}"
    email = f"us-{uuid4()}@example.test"
    token = jwt.encode(
        {
            "sub": auth_subject,
            "email": email,
            "name": "Synthetic US Market User",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}
    checklist = client.put(
        "/v1/me/checklist", headers=headers, json={"categories": ["us_investments"]}
    )
    assert checklist.status_code == 200

    with migrated_database.owner_engine.connect() as connection:
        user_id = connection.scalar(select(User.id).where(User.auth_subject == auth_subject))
    assert user_id is not None

    with migrated_database.api_engine.begin() as connection:
        security_id = connection.scalar(
            text(
                "SELECT app_upsert_security(:market, :isin, :symbol, :name, :kind, "
                ":currency, :sleeve, :category)"
            ),
            {
                "market": market,
                "isin": isin,
                "symbol": symbol,
                "name": "Synthetic US Holding",
                "kind": "equity",
                "currency": currency,
                "sleeve": "equity_global",
                "category": "equity:global",
            },
        )
    assert security_id is not None

    with migrated_database.owner_engine.begin() as connection:
        connection.execute(
            HoldingLot.__table__.insert().values(
                id=uuid4(),
                user_id=user_id,
                security_id=security_id,
                quantity=Decimal("2"),
                source="manual",
                statement_value=Decimal("250.00"),
            )
        )

    completeness = client.get("/v1/networth", headers=headers).json()["completeness"]
    assert completeness == {
        "ticked": ["us_investments"],
        "done": ["us_investments"],
        "pct": "100.00",
    }


def test_cross_tenant_resources_and_networth_are_hidden(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers_a = _headers(secret, "tenant-a")
    headers_b = _headers(secret, "tenant-b")
    member = client.post(
        "/v1/members",
        headers=headers_a,
        json={"display_name": "Synthetic Member", "relationship": "spouse"},
    ).json()
    account = client.post(
        "/v1/accounts",
        headers=headers_a,
        json={
            "institution": "Synthetic Bank",
            "account_type": "bank",
            "masked_number": "X1234",
            "member_id": member["id"],
        },
    ).json()
    asset = client.post(
        "/v1/assets",
        headers=headers_a,
        json={
            "asset_type": "savings",
            "bank": "Synthetic Bank",
            "balance": "40000",
            "member_id": member["id"],
        },
    ).json()
    goal = client.post(
        "/v1/goals",
        headers=headers_a,
        json={
            "name": "Synthetic Goal",
            "goal_type": "retirement",
            "target_amount": "10L",
            "target_year": 2040,
            "currency": "INR",
            "priority": 3,
        },
    ).json()

    for resource, resource_id, patch in (
        ("members", member["id"], {"display_name": "Other"}),
        ("accounts", account["id"], {"masked_number": "X5678"}),
        ("assets", asset["id"], {"bank": "Other Bank"}),
        ("goals", goal["id"], {"name": "Other Goal"}),
    ):
        assert (
            client.patch(f"/v1/{resource}/{resource_id}", headers=headers_b, json=patch).status_code
            == 404
        )
        assert client.delete(f"/v1/{resource}/{resource_id}", headers=headers_b).status_code == 404
    assert client.get(f"/v1/assets/{asset['id']}", headers=headers_b).status_code == 404
    assert client.get(f"/v1/goals/{goal['id']}", headers=headers_b).status_code == 404
    assert client.get("/v1/networth", headers=headers_b).json()["assets_total"] == "0.00"


def test_dkk_savings_value_reaches_networth_household_and_plan(
    api_client: tuple[TestClient, str, object], migrated_database: MigratedDatabase
) -> None:
    client, secret, application = api_client
    today = date(2035, 6, 15)
    application.dependency_overrides[get_today] = lambda: today
    with system_session(migrated_database.owner_engine) as session:
        session.add(
            FxRate(
                pair="DKK/INR",
                as_of=today,
                rate=Decimal("12.50"),
                source="synthetic-test",
                source_url="https://example.test/fx",
            )
        )

    headers = _headers(secret, "dkk-savings")
    member_id = client.get("/v1/me", headers=headers).json()["me_member_id"]
    created = client.post(
        "/v1/assets",
        headers=headers,
        json={
            "asset_type": "savings",
            "bank": "Synthetic Bank",
            "balance": "10000",
            "currency": "DKK",
            "member_id": member_id,
        },
    )

    assert created.status_code == 201, created.text
    asset = created.json()
    assert asset["details"]["balance"] == "10000.00"
    assert asset["details"]["currency"] == "DKK"
    assert asset["valuation"]["value_today"] == "125000.00"
    updated = client.patch(
        f"/v1/assets/{asset['id']}", headers=headers, json={"bank": "Synthetic Credit Union"}
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["value_cached"] == "125000.00"

    networth = client.get("/v1/networth", headers=headers).json()
    cash = next(item for item in networth["by_sleeve"] if item["sleeve"] == "cash")
    assert networth["assets_total"] == "125000.00"
    assert cash["amount"] == "125000.00"

    plan = client.get("/v1/plan", headers=headers)
    assert plan.status_code == 200, plan.text
    emergency_fund = next(
        item for item in plan.json()["foundation"] if item["key"] == "emergency_fund"
    )
    assert emergency_fund["have"] == "125000.00"


@pytest.mark.parametrize("rate_age_days", [None, 11])
def test_dkk_savings_without_a_fresh_rate_is_unvalued(
    rate_age_days: int | None,
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, application = api_client
    today = date(2036, 6, 15)
    application.dependency_overrides[get_today] = lambda: today
    if rate_age_days is not None:
        with system_session(migrated_database.owner_engine) as session:
            session.add(
                FxRate(
                    pair="DKK/INR",
                    as_of=today - timedelta(days=rate_age_days),
                    rate=Decimal("12.50"),
                    source="synthetic-test",
                    source_url="https://example.test/fx",
                )
            )

    headers = _headers(secret, f"dkk-savings-stale-{rate_age_days}")
    created = client.post(
        "/v1/assets",
        headers=headers,
        json={
            "asset_type": "savings",
            "bank": "Synthetic Bank",
            "balance": "10000",
            "currency": "DKK",
        },
    )

    assert created.status_code == 201, created.text
    assert created.json()["valuation"] is None
    assert created.json()["valuation_status"] == "fx_needed"
    networth = client.get("/v1/networth", headers=headers).json()
    assert networth["assets_total"] == "0.00"
    assert networth["unvalued_assets"] == [
        {
            "id": created.json()["id"],
            "asset_type": "savings",
            "label": "savings",
            "reason": "fx_needed",
        }
    ]
