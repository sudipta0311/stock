"""Recorded purchases stay temporary until a statement confirms them."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import jwt
import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from conftest import MigratedDatabase
from wealth.api.main import create_app
from wealth.api.money_map import get_today, refresh_networth_snapshot
from wealth.api.recorded_buys import _find_security, match_recorded_buys_for_import
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import (
    Account,
    Action,
    AuditLog,
    BrokerCashFlowRecord,
    FxRate,
    HoldingLot,
    HouseholdAccess,
    HouseholdMember,
    ImportJob,
    NavDaily,
    RecordedBuy,
    Security,
    User,
)
from wealth.marketdata.provider import DbPriceProvider
from wealth.policy.actions import Action as PolicyAction
from wealth.policy.data import (
    _ask_deposits_dkk,
    _holdings,
    _weekly_spending_for_week,
    load_policy_inputs,
)
from wealth.policy.personalise import WeeklySpendInput, _apply_weekly_spending, _holding_value
from wealth.policy.us_situs import us_situs_value_usd

pytestmark = pytest.mark.db
_TODAY = date(2026, 10, 28)
_WEEK = "2026-W44"


@pytest.fixture
def api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, MigratedDatabase]]:
    secret = secrets.token_urlsafe(32)
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
        yield client, secret, migrated_database


def _login(client: TestClient, secret: str, label: str) -> tuple[dict[str, str], UUID]:
    suffix = uuid4().hex
    token = jwt.encode(
        {
            "sub": f"recorded-buys-{label}-{suffix}",
            "email": f"{label}-{suffix}@example.test",
            "name": "Synthetic Household",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/v1/me", headers=headers)
    assert response.status_code == 200
    return headers, UUID(response.json()["id"])


def _self_member_id(client: TestClient, headers: dict[str, str]) -> UUID:
    response = client.get("/v1/members", headers=headers)
    assert response.status_code == 200, response.text
    return UUID(next(row["id"] for row in response.json() if row["is_self"]))


def _seed_action(
    database: MigratedDatabase,
    user_id: UUID,
    member_id: UUID,
    *,
    kind: str = "BUY_LUMPSUM",
    status: str = "proposed",
    amount: str = "5000.00",
    sleeve: str = "equity_india",
    abroad_lines: list[dict[str, object]] | None = None,
) -> UUID:
    action_id = uuid4()
    params: dict[str, object] = {}
    if abroad_lines is not None:
        params["abroad_buy_list"] = abroad_lines
    payload = {
        "kind": kind,
        "amount": amount,
        "currency": "INR",
        "instrument_category": "in_core_index",
        "sleeve": sleeve,
        "member_id": str(member_id),
        "params": params,
    }
    with database.owner_engine.begin() as connection:
        connection.execute(
            Action.__table__.insert().values(
                id=action_id,
                user_id=user_id,
                week=_WEEK,
                rank=1,
                kind=kind,
                payload=payload,
                status=status,
            )
        )
    return action_id


def _recorded_buy(
    *,
    user_id: UUID,
    member_id: UUID,
    bought_on: date,
    amount: Decimal = Decimal("100.00"),
    currency: str = "INR",
    amount_inr: Decimal | None = Decimal("100.00"),
    action_id: UUID | None = None,
    line: int = 0,
    account_id: UUID | None = None,
    security_id: UUID | None = None,
    security_type: str = "mf",
    units: Decimal | None = None,
    cost_per_unit: Decimal | None = None,
    account_type: str | None = None,
    name: str = "Indian index fund",
    sleeve: str = "equity_india",
    category: str = "in_core_index",
) -> RecordedBuy:
    return RecordedBuy(
        user_id=user_id,
        member_id=member_id,
        action_id=action_id,
        account_id=account_id,
        line=line,
        bought_on=bought_on,
        amount=amount,
        currency=currency,
        amount_inr=amount_inr,
        units=units,
        units_entered=units is not None,
        cost_per_unit=cost_per_unit,
        cost_per_unit_inr=(amount_inr / units if amount_inr and units else None),
        security_id=security_id,
        security_type=security_type,
        security_symbol=None,
        security_isin=None,
        name=name,
        sleeve=sleeve,
        instrument_category=category,
        account_type=account_type,
        status="waiting",
    )


def _weekly_spending(database: MigratedDatabase, user_id: UUID) -> tuple[WeeklySpendInput, ...]:
    with Session(database.owner_engine) as session:
        buys = tuple(session.scalars(select(RecordedBuy).where(RecordedBuy.user_id == user_id)))
        return _weekly_spending_for_week(session, user_id, _WEEK, buys)


def _policy_abroad_action(
    member_id: UUID, amount_inr: str, amount_dkk: str, *, rank: int = 1
) -> PolicyAction:
    return PolicyAction(
        rank=rank,
        kind="BUY_ABROAD",
        sleeve="equity_global",
        instrument_category="abroad_etf",
        amount=Decimal(amount_inr),
        reason_code="buy_abroad",
        member_id=str(member_id),
        params={
            "abroad_buy_list": [
                {
                    "member_id": str(member_id),
                    "sleeve": "equity_global",
                    "account": "ordinary",
                    "instrument": {"name": "Synthetic European fund"},
                    "amount_dkk": amount_dkk,
                    "amount_inr": amount_inr,
                }
            ]
        },
    )


def test_record_bought_values_and_spent_money_once_then_reopens_on_remove(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, user_id = _login(client, secret, "record-flow")
    member_id = _self_member_id(client, headers)
    action_id = _seed_action(database, user_id, member_id)

    response = client.post(
        f"/v1/actions/{action_id}/bought",
        headers=headers,
        json={"lines": [{"line": 0, "amount": "1234.50", "currency": "INR"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["action"]["status"] == "done"
    assert body["action"]["has_recorded_buys"] is True
    assert body["recorded_buys"][0]["name"] == "Indian index fund"
    assert body["recorded_buys"][0]["amount_inr"] == "1234.50"
    buy_id = UUID(body["recorded_buys"][0]["id"])

    listed = client.get("/v1/recorded-buys?status=waiting", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    snapshot = client.get("/v1/networth", headers=headers).json()
    assert snapshot["assets_total"] == "1234.50"
    assert snapshot["holdings"][0]["added_by_you"] is True

    unrecorded_action = _seed_action(
        database,
        user_id,
        member_id,
        kind="PAY_DOWN_DEBT",
        status="done",
        amount="500.00",
    )
    del unrecorded_action
    assert _weekly_spending(database, user_id) == (
        WeeklySpendInput(
            member_id=str(member_id), sleeve="equity_india", amount_inr=Decimal("1234.50")
        ),
    )

    edited = client.patch(
        f"/v1/recorded-buys/{buy_id}",
        headers=headers,
        json={"amount": "1500.00", "units": "30", "bought_on": "2026-10-27"},
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["amount"] == "1500.00"
    assert edited.json()["units"] == "30"

    removed = client.delete(f"/v1/recorded-buys/{buy_id}", headers=headers)
    assert removed.status_code == 200
    assert removed.json()["status"] == "removed"
    with database.owner_engine.connect() as connection:
        stored_action = connection.scalar(select(Action.status).where(Action.id == action_id))
        audit_count = connection.scalar(
            select(func.count())
            .select_from(AuditLog)
            .where(
                AuditLog.user_id == user_id,
                AuditLog.action.in_(
                    ("recorded_buy_created", "recorded_buy_updated", "recorded_buy_removed")
                ),
            )
        )
    assert stored_action == "proposed"
    assert audit_count == 3


def test_prefilled_abroad_buy_does_not_reallocate_the_carry_forward_on_rebuild(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, user_id = _login(client, secret, "abroad-prefilled-spend")
    member_id = _self_member_id(client, headers)
    investing_saved = client.put(
        "/v1/me/investing",
        headers=headers,
        json={"new_money_this_week": "12345.60"},
    )
    assert investing_saved.status_code == 200
    action_id = _seed_action(
        database,
        user_id,
        member_id,
        kind="BUY_ABROAD",
        amount="2000.00",
        sleeve="equity_global",
        abroad_lines=[
            {
                "member_id": str(member_id),
                "sleeve": "equity_global",
                "account": "ordinary",
                "instrument": {"name": "Synthetic European fund"},
                "amount_dkk": "162",
                "amount_inr": "2000.00",
            }
        ],
    )
    with Session(database.owner_engine) as session, session.begin():
        session.add(
            FxRate(
                pair="DKK/INR",
                as_of=_TODAY,
                rate=Decimal("12.3456"),
                source="synthetic-test",
                source_url="https://example.test/fx-rate",
            )
        )

    recorded = client.post(
        f"/v1/actions/{action_id}/bought",
        headers=headers,
        json={"lines": [{"line": 0, "amount": "162", "currency": "DKK"}]},
    )
    assert recorded.status_code == 200, recorded.text
    assert recorded.json()["recorded_buys"][0]["amount_inr"] == "1999.99"

    with Session(database.owner_engine) as session:
        user_data, _market, _research = load_policy_inputs(session, user_id, _WEEK, _TODAY)
    assert user_data.investing.new_money_this_week == Decimal("12345.60")
    assert user_data.weekly_spending == (
        WeeklySpendInput(
            member_id=str(member_id), sleeve="equity_global", amount_inr=Decimal("2000.00")
        ),
    )

    sell = PolicyAction(
        rank=1,
        kind="SELL_TRIM",
        sleeve="equity_global",
        instrument_category="stock",
        amount=Decimal("500.00"),
        reason_code="synthetic_exit",
        member_id=str(member_id),
    )
    buy = _policy_abroad_action(member_id, "2000.00", "162", rank=2)
    adjusted, carry_forward = _apply_weekly_spending(
        [sell, buy],
        Decimal("10345.60"),
        user_data.weekly_spending,
        {"DKK/INR": Decimal("12.3456")},
    )
    assert [action.kind for action in adjusted] == ["SELL_TRIM"]
    assert adjusted[0].amount == Decimal("500.00")
    assert carry_forward == Decimal("10345.60")


def test_half_of_an_abroad_line_leaves_the_other_half_available(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, user_id = _login(client, secret, "abroad-half-spend")
    member_id = _self_member_id(client, headers)
    action_id = _seed_action(
        database,
        user_id,
        member_id,
        kind="BUY_ABROAD",
        amount="3000.00",
        sleeve="equity_global",
        abroad_lines=[
            {
                "member_id": str(member_id),
                "sleeve": "equity_global",
                "account": "ordinary",
                "instrument": {"name": "Synthetic European fund"},
                "amount_dkk": "244",
                "amount_inr": "3000.00",
            }
        ],
    )
    bought_on = _TODAY - timedelta(days=1)
    with Session(database.owner_engine) as session, session.begin():
        session.add(
            FxRate(
                pair="DKK/INR",
                as_of=bought_on,
                rate=Decimal("12.29508197"),
                source="synthetic-test",
                source_url="https://example.test/fx-rate",
            )
        )

    recorded = client.post(
        f"/v1/actions/{action_id}/bought",
        headers=headers,
        json={
            "bought_on": bought_on.isoformat(),
            "lines": [{"line": 0, "amount": "122", "currency": "DKK"}],
        },
    )
    assert recorded.status_code == 200, recorded.text
    assert recorded.json()["recorded_buys"][0]["amount_inr"] == "1500.00"

    spending = _weekly_spending(database, user_id)
    assert spending == (
        WeeklySpendInput(
            member_id=str(member_id), sleeve="equity_global", amount_inr=Decimal("1500.00")
        ),
    )
    adjusted, _carry_forward = _apply_weekly_spending(
        [_policy_abroad_action(member_id, "3000.00", "244")],
        Decimal(0),
        spending,
        {"DKK/INR": Decimal("12.29508197")},
    )
    assert len(adjusted) == 1
    assert adjusted[0].amount == Decimal("1500.00")
    assert adjusted[0].params["abroad_buy_list"][0]["amount_inr"] == "1500.00"


def test_full_non_abroad_step_spends_the_step_amount(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, user_id = _login(client, secret, "non-abroad-step-spend")
    member_id = _self_member_id(client, headers)
    action_id = _seed_action(database, user_id, member_id, amount="5000.00")
    with Session(database.owner_engine) as session, session.begin():
        session.add(
            _recorded_buy(
                user_id=user_id,
                member_id=member_id,
                bought_on=_TODAY,
                amount=Decimal("5000.00"),
                amount_inr=Decimal("4900.00"),
                action_id=action_id,
            )
        )

    assert _weekly_spending(database, user_id) == (
        WeeklySpendInput(
            member_id=str(member_id), sleeve="equity_india", amount_inr=Decimal("5000.00")
        ),
    )


def test_done_buy_step_without_a_recorded_buy_spends_the_step_amount(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, user_id = _login(client, secret, "done-step-without-buy")
    member_id = _self_member_id(client, headers)
    _seed_action(
        database,
        user_id,
        member_id,
        kind="BUY_LUMPSUM",
        status="done",
        amount="5000.00",
    )

    assert _weekly_spending(database, user_id) == (
        WeeklySpendInput(
            member_id=str(member_id), sleeve="equity_india", amount_inr=Decimal("5000.00")
        ),
    )


def test_recording_less_than_display_carries_small_remainder_forward(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, user_id = _login(client, secret, "abroad-small-remainder")
    member_id = _self_member_id(client, headers)
    action_id = _seed_action(
        database,
        user_id,
        member_id,
        kind="BUY_ABROAD",
        amount="1500.00",
        sleeve="equity_global",
        abroad_lines=[
            {
                "member_id": str(member_id),
                "sleeve": "equity_global",
                "account": "ordinary",
                "instrument": {"name": "Synthetic European fund"},
                "amount_dkk": "122",
                "amount_inr": "1500.00",
            }
        ],
    )
    bought_on = _TODAY - timedelta(days=2)
    with Session(database.owner_engine) as session, session.begin():
        session.add(
            FxRate(
                pair="DKK/INR",
                as_of=bought_on,
                rate=Decimal("12.29508197"),
                source="synthetic-test",
                source_url="https://example.test/fx-rate",
            )
        )

    recorded = client.post(
        f"/v1/actions/{action_id}/bought",
        headers=headers,
        json={
            "bought_on": bought_on.isoformat(),
            "lines": [{"line": 0, "amount": "61", "currency": "DKK"}],
        },
    )
    assert recorded.status_code == 200, recorded.text
    assert recorded.json()["recorded_buys"][0]["amount_inr"] == "750.00"

    spending = _weekly_spending(database, user_id)
    assert spending == (
        WeeklySpendInput(
            member_id=str(member_id), sleeve="equity_global", amount_inr=Decimal("750.00")
        ),
    )
    adjusted, carry_forward = _apply_weekly_spending(
        [_policy_abroad_action(member_id, "1500.00", "122")],
        Decimal(0),
        spending,
        {"DKK/INR": Decimal("12.29508197")},
    )
    assert adjusted == []
    assert carry_forward == Decimal("750.00")


def test_record_bought_validation_and_abroad_line_indexes(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, user_id = _login(client, secret, "validations")
    member_id = _self_member_id(client, headers)
    abroad = [
        {
            "member_id": str(member_id),
            "account": "ask",
            "instrument": {"name": "Synthetic European fund", "isin": "US0000000001"},
            "amount_dkk": "100",
            "reason_codes": ["ask_first"],
        },
        {
            "member_id": str(member_id),
            "account": "ordinary",
            "instrument": {"name": "Synthetic global fund", "isin": "US0000000002"},
            "amount_dkk": "200",
            "reason_codes": ["ordinary_after_ask"],
        },
    ]
    abroad_action = _seed_action(
        database, user_id, member_id, kind="BUY_ABROAD", abroad_lines=abroad
    )
    recorded = client.post(
        f"/v1/actions/{abroad_action}/bought",
        headers=headers,
        json={
            "bought_on": "2026-10-26",
            "lines": [
                {"line": 1, "amount": "200", "currency": "DKK"},
                {"line": 0, "amount": "100", "currency": "DKK"},
            ],
        },
    )
    assert recorded.status_code == 200, recorded.text
    assert [row["line"] for row in recorded.json()["recorded_buys"]] == [1, 0]
    assert {row["account_type"] for row in recorded.json()["recorded_buys"]} == {
        "ask",
        "ordinary",
    }

    invalid_cases = [
        ("PAY_DOWN_DEBT", "proposed", None, {"line": 0, "amount": "1", "currency": "INR"}, None),
        ("BUY_LUMPSUM", "done", None, {"line": 0, "amount": "1", "currency": "INR"}, None),
        ("BUY_ABROAD", "proposed", abroad, {"line": 2, "amount": "1", "currency": "DKK"}, None),
        ("BUY_LUMPSUM", "proposed", None, {"line": 0, "amount": "0", "currency": "INR"}, None),
        ("BUY_ABROAD", "proposed", abroad, {"line": 0, "amount": "1", "currency": "INR"}, None),
        (
            "BUY_LUMPSUM",
            "proposed",
            None,
            {"line": 0, "amount": "1", "currency": "INR", "units": "0"},
            None,
        ),
        (
            "BUY_LUMPSUM",
            "proposed",
            None,
            {"line": 0, "amount": "1", "currency": "INR"},
            "2026-10-11",
        ),
        (
            "BUY_LUMPSUM",
            "proposed",
            None,
            {"line": 0, "amount": "1", "currency": "INR"},
            "2026-10-29",
        ),
    ]
    for index, (kind, status, lines, line, bought_on) in enumerate(invalid_cases):
        action_id = _seed_action(
            database,
            user_id,
            member_id,
            kind=kind,
            status=status,
            abroad_lines=lines,
        )
        payload: dict[str, object] = {"lines": [line]}
        if bought_on:
            payload["bought_on"] = bought_on
        result = client.post(f"/v1/actions/{action_id}/bought", headers=headers, json=payload)
        assert result.status_code == 422, f"case {index}: {result.text}"


def test_ticker_fallback_uses_market_and_rejects_ambiguous_matches(
    migrated_database: MigratedDatabase,
) -> None:
    ticker = f"DUPE{uuid4().hex[:8].upper()}"
    in_security = Security(
        market="IN",
        symbol=ticker,
        exchange="NSE",
        name="Synthetic India security",
        security_type="equity",
        currency="INR",
        sleeve="equity_india",
        category="stock",
    )
    us_security = Security(
        market="US",
        symbol=ticker,
        exchange="NASDAQ",
        name="Synthetic US security",
        security_type="equity",
        currency="USD",
        sleeve="equity_global",
        category="stock",
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add_all([in_security, us_security])
        session.flush()

        stock_line = {
            "reason_codes": ["approved_us_stock_idea"],
            "instrument": {"ticker": ticker},
        }
        assert _find_security(session, {"kind": "BUY_ABROAD"}, stock_line) is us_security

        market_fund_line = {"instrument": {"ticker": ticker, "market": "US"}}
        assert _find_security(session, {"kind": "BUY_ABROAD"}, market_fund_line) is us_security
        exchange_fund_line = {"instrument": {"ticker": ticker, "exchange": "NASDAQ"}}
        assert _find_security(session, {"kind": "BUY_ABROAD"}, exchange_fund_line) is us_security

        india_step = {"kind": "BUY_LUMPSUM", "params": {"symbol": ticker}}
        assert _find_security(session, india_step, None) is in_security

        ambiguous_fund_line = {"instrument": {"ticker": ticker}}
        assert _find_security(session, {"kind": "BUY_ABROAD"}, ambiguous_fund_line) is None


def test_household_roles_scope_reads_and_writes(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, owner_id = _login(client, secret, "roles-owner")
    member_id = _self_member_id(client, owner_headers)
    viewer_member_response = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Synthetic viewer member", "relationship": "other"},
    )
    assert viewer_member_response.status_code == 201
    viewer_member_id = UUID(viewer_member_response.json()["id"])
    partner_headers, partner_id = _login(client, secret, "roles-partner")
    viewer_headers, viewer_id = _login(client, secret, "roles-viewer")
    other_headers, _other_id = _login(client, secret, "roles-other")
    with database.owner_engine.begin() as connection:
        connection.execute(
            HouseholdAccess.__table__.insert(),
            [
                {
                    "owner_user_id": owner_id,
                    "grantee_user_id": partner_id,
                    "member_id": member_id,
                    "role": "partner",
                },
                {
                    "owner_user_id": owner_id,
                    "grantee_user_id": viewer_id,
                    "member_id": viewer_member_id,
                    "role": "viewer",
                },
            ],
        )
    owner_action = _seed_action(database, owner_id, member_id)
    partner_action = _seed_action(database, owner_id, member_id, amount="200.00")
    owner_buy = client.post(
        f"/v1/actions/{owner_action}/bought",
        headers=owner_headers,
        json={"lines": [{"line": 0, "amount": "100.00", "currency": "INR"}]},
    )
    assert owner_buy.status_code == 200

    selected_partner_headers = {**partner_headers, "X-Household": str(owner_id)}
    selected_viewer_headers = {**viewer_headers, "X-Household": str(owner_id)}
    partner_buy = client.post(
        f"/v1/actions/{partner_action}/bought",
        headers=selected_partner_headers,
        json={"lines": [{"line": 0, "amount": "200.00", "currency": "INR"}]},
    )
    assert partner_buy.status_code == 200, partner_buy.text
    assert len(client.get("/v1/recorded-buys", headers=selected_viewer_headers).json()) == 2
    viewer_blocked = client.post(
        f"/v1/actions/{partner_action}/bought",
        headers=selected_viewer_headers,
        json={"lines": [{"line": 0, "amount": "200.00", "currency": "INR"}]},
    )
    assert viewer_blocked.status_code == 403
    assert client.get("/v1/recorded-buys", headers=other_headers).json() == []
    hidden_action = client.post(
        f"/v1/actions/{owner_action}/bought",
        headers=other_headers,
        json={"lines": [{"line": 0, "amount": "100.00", "currency": "INR"}]},
    )
    assert hidden_action.status_code == 404
    assert client.get("/v1/actions?week=2026-W44", headers=other_headers).json() == []
    assert client.get("/v1/recorded-buys", headers=selected_partner_headers).status_code == 200


def test_policy_money_snapshot_fund_check_and_us_situs_include_waiting_buys(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    from wealth.api.fund_check import _build_inputs

    client, secret, database = api_client
    headers, user_id = _login(client, secret, "valuation")
    member_id = _self_member_id(client, headers)
    action_id = _seed_action(database, user_id, member_id)
    security = Security(
        market="IN",
        symbol=f"SYNTH-{uuid4().hex[:8]}",
        isin="IN0000000001",
        name="Synthetic Indian index fund",
        security_type="mf",
        currency="INR",
        sleeve="equity_india",
        category="mf:equity_india",
        sub_category="Large Cap",
        plan_type="direct",
        option="growth",
    )
    with Session(database.owner_engine) as session, session.begin():
        session.add(security)
        session.flush()
        session.add(
            NavDaily(
                security_id=security.id,
                as_of=_TODAY,
                nav=Decimal("50"),
                source="synthetic-test",
            )
        )
        session.add(
            HoldingLot(
                user_id=user_id,
                member_id=member_id,
                security_id=security.id,
                quantity=Decimal("3"),
                cost_per_unit=Decimal("40"),
                acquired_on=date(2025, 1, 1),
                source="manual",
                currency="INR",
                cost_per_unit_inr=Decimal("40"),
            )
        )
        buy = _recorded_buy(
            user_id=user_id,
            member_id=member_id,
            action_id=action_id,
            bought_on=date(2026, 10, 27),
            amount=Decimal("90"),
            amount_inr=Decimal("90"),
            security_id=security.id,
            units=Decimal("2"),
            cost_per_unit=Decimal("45"),
        )
        session.add(buy)
        session.flush()
        holdings = _holdings(session, user_id, _TODAY, DbPriceProvider(session), (buy,))
        assert len(holdings) == 1
        assert holdings[0].quantity == Decimal("5")
        assert _holding_value(holdings[0], {"INR/INR": Decimal(1)}) == Decimal("250.00")
        fund_holdings, *_rest = _build_inputs(session, user_id, _TODAY)
        assert len(fund_holdings) == 1
        assert fund_holdings[0].quantity == Decimal("5")
        assert fund_holdings[0].value == Decimal("250")
        assert len(fund_holdings[0].tax_lots) == 2
        totals = refresh_networth_snapshot(session, user_id, _TODAY)
        assert totals["assets_total"] == "250.00"

    assert isinstance(holdings[0].quantity, Decimal)
    assert us_situs_value_usd(
        [
            {
                "domicile": "US",
                "currency": "USD",
                "price": Decimal("10"),
                "quantity": Decimal("2"),
            }
        ]
    ) == Decimal("20")
    assert us_situs_value_usd(
        [
            {
                "domicile": "US",
                "currency": "INR",
                "price": None,
                "quantity": Decimal(0),
                "cost_value_inr": Decimal("900"),
            }
        ],
        {"USD/INR": Decimal("90")},
    ) == Decimal("10")


def test_ask_room_uses_waiting_buy_once_when_later_cashflow_covers_it(
    migrated_database: MigratedDatabase,
) -> None:
    user_id = uuid4()
    member_id = uuid4()
    account_id = uuid4()
    bought_on = date(2026, 4, 4)
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(User(id=user_id, email="ask-room@example.test", auth_subject=f"ask-{user_id}"))
        session.flush()
        member_id = session.scalar(
            select(HouseholdMember.id).where(
                HouseholdMember.user_id == user_id, HouseholdMember.is_self.is_(True)
            )
        )
        assert member_id is not None
        account = Account(
            id=account_id,
            user_id=user_id,
            member_id=member_id,
            institution="Synthetic broker",
            account_type="investment",
            tax_wrapper="ask",
        )
        job = ImportJob(
            user_id=user_id,
            member_id=member_id,
            source="broker",
            doc_type="broker_statement",
            status="confirmed",
            counts={},
        )
        session.add_all([account, job])
        session.flush()
        buy = _recorded_buy(
            user_id=user_id,
            member_id=member_id,
            bought_on=bought_on,
            amount=Decimal("100"),
            currency="DKK",
            amount_inr=Decimal("1250"),
            account_id=account_id,
            account_type="ask",
        )
        session.add(buy)
        session.flush()
        assert _ask_deposits_dkk(session, user_id, member_id, _TODAY) == (Decimal("100"), False)
        session.add(
            BrokerCashFlowRecord(
                user_id=user_id,
                member_id=member_id,
                account_id=account_id,
                import_job_id=job.id,
                transaction_date=bought_on + timedelta(days=2),
                kind="deposit",
                amount=Decimal("100"),
                currency="DKK",
                source_broker="synthetic",
            )
        )
        session.flush()
        assert _ask_deposits_dkk(session, user_id, member_id, _TODAY) == (Decimal("100"), False)


def test_statement_matching_uses_member_date_settlement_and_fund_rules(
    migrated_database: MigratedDatabase,
) -> None:
    user_id = uuid4()
    member_id, other_member_id = uuid4(), uuid4()
    security = Security(
        market="US",
        symbol=f"SYNTH-{uuid4().hex[:8]}",
        isin="US0000000001",
        domicile="US",
        name="Synthetic company share",
        security_type="equity",
        currency="USD",
        sleeve="equity_global",
        category="stock",
    )
    fund_security = Security(
        market="IN",
        symbol="SYNTH-CAS-FUND",
        isin="IN0000000002",
        name="Synthetic confirmed fund",
        security_type="mf",
        currency="INR",
        sleeve="equity_india",
        category="mf:equity_india",
    )
    cams_security = Security(
        market="US",
        symbol="SYNTH-CAMS-SHARE",
        isin="US0000000002",
        name="Synthetic CAMS share",
        security_type="equity",
        currency="USD",
        sleeve="equity_global",
        category="stock",
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(User(id=user_id, email="match@example.test", auth_subject=f"match-{user_id}"))
        session.flush()
        member_id = session.scalar(
            select(HouseholdMember.id).where(
                HouseholdMember.user_id == user_id, HouseholdMember.is_self.is_(True)
            )
        )
        assert member_id is not None
        session.add_all(
            [
                HouseholdMember(
                    id=other_member_id,
                    user_id=user_id,
                    display_name="Second synthetic member",
                    relationship="spouse",
                ),
                security,
                fund_security,
                cams_security,
            ]
        )
        session.flush()
        known = _recorded_buy(
            user_id=user_id,
            member_id=member_id,
            bought_on=date(2026, 10, 10),
            amount=Decimal("10"),
            currency="USD",
            amount_inr=None,
            security_id=security.id,
            security_type="equity",
            units=Decimal("1"),
            cost_per_unit=Decimal("10"),
            sleeve="equity_global",
            category="stock",
        )
        category_only = _recorded_buy(
            user_id=user_id,
            member_id=member_id,
            bought_on=date(2026, 10, 20),
            amount=Decimal("50"),
            amount_inr=Decimal("50"),
        )
        cams_stock = _recorded_buy(
            user_id=user_id,
            member_id=member_id,
            bought_on=date(2026, 10, 10),
            amount=Decimal("10"),
            currency="USD",
            amount_inr=None,
            security_id=cams_security.id,
            security_type="equity",
            units=Decimal("1"),
            cost_per_unit=Decimal("10"),
            sleeve="equity_global",
            category="stock",
        )
        confirmed_fund = _recorded_buy(
            user_id=user_id,
            member_id=member_id,
            bought_on=date(2026, 10, 10),
            amount=Decimal("100"),
            amount_inr=Decimal("100"),
            security_id=fund_security.id,
        )
        other_member = _recorded_buy(
            user_id=user_id,
            member_id=other_member_id,
            bought_on=date(2026, 10, 20),
            amount=Decimal("50"),
            amount_inr=Decimal("50"),
        )
        session.add_all([known, category_only, other_member, cams_stock, confirmed_fund])
        session.flush()

        assert (
            match_recorded_buys_for_import(
                session,
                user_id,
                member_id,
                "broker",
                date(2026, 10, 9),
                _TODAY,
                {security.id},
            )
            == 0
        )
        assert known.status == "waiting"
        assert (
            match_recorded_buys_for_import(
                session,
                user_id,
                member_id,
                "broker",
                None,
                date(2026, 10, 28),
                {security.id},
            )
            == 1
        )
        assert known.status == "matched"
        assert cams_stock.status == "waiting"
        assert category_only.status == "waiting"
        assert other_member.status == "waiting"

        assert (
            match_recorded_buys_for_import(
                session,
                user_id,
                member_id,
                "nsdl",
                date(2026, 10, 11),
                _TODAY,
                {fund_security.id},
            )
            == 1
        )
        assert confirmed_fund.status == "matched"
        assert (
            match_recorded_buys_for_import(
                session,
                user_id,
                member_id,
                "cams_kfin",
                date(2026, 10, 20),
                date(2026, 10, 28),
                {cams_security.id},
            )
            == 0
        )
        assert known.status == "matched"
        assert cams_stock.status == "waiting"
        assert other_member.status == "waiting"

        assert (
            match_recorded_buys_for_import(
                session,
                user_id,
                member_id,
                "cdsl",
                date(2026, 10, 22),
                _TODAY,
                set(),
            )
            == 0
        )
        assert category_only.status == "waiting"
        assert (
            match_recorded_buys_for_import(
                session,
                user_id,
                member_id,
                "nsdl",
                date(2026, 10, 23),
                _TODAY,
                set(),
            )
            == 1
        )
        assert category_only.status == "matched"
        assert other_member.status == "waiting"


def test_recorded_buy_migration_can_downgrade_and_upgrade(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0026_audit_log_hardening")
    with migrated_database.owner_engine.connect() as connection:
        assert inspect(connection).has_table("recorded_buys") is False
    command.upgrade(migrated_database.config, "head")
    with migrated_database.owner_engine.connect() as connection:
        inspector = inspect(connection)
        assert inspector.has_table("recorded_buys") is True
        member_fk = next(
            foreign_key
            for foreign_key in inspector.get_foreign_keys("recorded_buys")
            if foreign_key["name"] == "fk_recorded_buys_member_tenant"
        )
        assert member_fk["options"]["deferrable"] is True
        deferrable, initially_deferred = connection.execute(
            text(
                "SELECT condeferrable, condeferred FROM pg_constraint "
                "WHERE conname = 'fk_recorded_buys_member_tenant'"
            )
        ).one()
        assert deferrable is True
        assert initially_deferred is False
