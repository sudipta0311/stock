"""Loan-event household access, audit, FX totals, and merge behaviour."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from conftest import MigratedDatabase
from wealth.api.main import create_app
from wealth.api.money_map import get_today
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import AuditLog, FxRate, LoanEventRecord
from wealth.db.session import system_session

pytestmark = pytest.mark.db
_TODAY = date(2026, 10, 28)


@pytest.fixture
def api_client(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, MigratedDatabase]]:
    from secrets import token_urlsafe

    secret = token_urlsafe(32)
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


def _login(client: TestClient, secret: str, label: str) -> tuple[dict[str, str], str, UUID, str]:
    suffix = uuid4().hex
    subject = f"loan-{label}-{suffix}"
    email = f"{label}-{suffix}@example.test"
    token = jwt.encode(
        {
            "sub": subject,
            "email": email,
            "name": f"Synthetic {label.title()}",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/v1/me", headers=headers)
    assert response.status_code == 200
    return headers, email, UUID(response.json()["id"]), subject


def _household(headers: dict[str, str], owner_id: UUID) -> dict[str, str]:
    return {**headers, "X-Household": str(owner_id)}


def _loan_payload(currency: str = "INR") -> dict[str, str]:
    return {
        "asset_type": "loan",
        "loan_kind": "home",
        "label": "Demo home loan",
        "lender": "Sample lender",
        "outstanding": "2000.00" if currency == "DKK" else "1000000.00",
        "rate_pct": "8.5",
        "emi": "200.00" if currency == "DKK" else "10000.00",
        "currency": currency,
        "outstanding_as_of": "2026-10",
        "emi_day": "10",
        "rate_type": "floating",
        "tax_relief": "none",
    }


def _create_loan(client: TestClient, headers: dict[str, str], currency: str = "INR") -> UUID:
    response = client.post("/v1/assets", headers=headers, json=_loan_payload(currency))
    assert response.status_code == 201, response.text
    return UUID(response.json()["id"])


def _invite(
    client: TestClient,
    owner_headers: dict[str, str],
    person_headers: dict[str, str],
    email: str,
    *,
    role: str = "partner",
    member_id: UUID | None = None,
    bring_my_data: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {"email": email, "role": role}
    if member_id is not None:
        payload["member_id"] = str(member_id)
    created = client.post("/v1/households/invites", headers=owner_headers, json=payload)
    assert created.status_code == 201, created.text
    token = parse_qs(urlparse(created.json()["share_link"]).query)["token"][0]
    joined = client.post(
        "/v1/households/join",
        headers=person_headers,
        json={"token": token, "bring_my_data": bring_my_data},
    )
    assert joined.status_code == 200, joined.text
    return joined.json()


def test_loan_events_enforce_household_roles_audit_writes_and_convert_dkk(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_email, owner_id, _ = _login(client, secret, "owner")
    partner_headers, partner_email, partner_id, _ = _login(client, secret, "partner")
    viewer_headers, viewer_email, _viewer_id, _ = _login(client, secret, "viewer")
    stranger_headers, _stranger_email, _stranger_id, _ = _login(client, secret, "stranger")
    loan_id = _create_loan(client, owner_headers, "DKK")
    with system_session(database.owner_engine) as session:
        session.add(
            FxRate(
                pair="DKK/INR",
                as_of=_TODAY,
                rate=Decimal("12.5"),
                source="synthetic-test",
                source_url="https://example.test/fx",
            )
        )

    networth = client.get("/v1/networth", headers=owner_headers)
    assert networth.status_code == 200, networth.text
    assert networth.json()["liabilities_total"] == "25000.00"
    assert (
        client.get(
            f"/v1/loans/{loan_id}", headers=_household(stranger_headers, owner_id)
        ).status_code
        == 404
    )

    _invite(client, owner_headers, partner_headers, partner_email)
    partner_household = _household(partner_headers, owner_id)
    event = client.post(
        f"/v1/loans/{loan_id}/events",
        headers=partner_household,
        json={"kind": "prepayment", "on_date": "2026-10-10", "amount": "100.00"},
    )
    assert event.status_code == 201, event.text
    event_id = UUID(event.json()["id"])

    _invite(client, owner_headers, viewer_headers, viewer_email, role="viewer")
    viewer_household = _household(viewer_headers, owner_id)
    assert client.get(f"/v1/loans/{loan_id}", headers=viewer_household).status_code == 200
    blocked = client.post(
        f"/v1/loans/{loan_id}/events",
        headers=viewer_household,
        json={"kind": "missed_emi", "on_date": "2026-10-11"},
    )
    assert blocked.status_code == 403
    assert (
        client.delete(
            f"/v1/loans/{loan_id}/events/{event_id}", headers=viewer_household
        ).status_code
        == 403
    )

    with system_session(database.owner_engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(
                    AuditLog.user_id == owner_id,
                    AuditLog.actor_user_id == partner_id,
                    AuditLog.action == "household_row_insert",
                    AuditLog.detail["table"].astext == "loan_events",
                )
            )
            == 1
        )


def test_loan_events_and_loan_rows_merge_into_the_target_household(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_email, owner_id, _ = _login(client, secret, "merge-owner")
    partner_headers, partner_email, partner_id, _ = _login(client, secret, "merge-partner")
    loan_id = _create_loan(client, partner_headers)
    event = client.post(
        f"/v1/loans/{loan_id}/events",
        headers=partner_headers,
        json={"kind": "prepayment", "on_date": "2026-10-10", "amount": "1000.00"},
    )
    assert event.status_code == 201, event.text

    member = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Demo partner", "relationship": "other"},
    )
    assert member.status_code == 201, member.text
    joined = _invite(
        client,
        owner_headers,
        partner_headers,
        partner_email,
        member_id=UUID(member.json()["id"]),
        bring_my_data=True,
    )
    assert joined["owner_id"] == str(owner_id)
    assert joined["merge_counts"]["manual_assets"] >= 1

    with system_session(database.owner_engine) as session:
        event_row = session.scalar(
            select(LoanEventRecord).where(LoanEventRecord.loan_id == loan_id)
        )
        assert event_row is not None
        assert event_row.user_id == owner_id
        assert event_row.member_id == UUID(member.json()["id"])
        assert event_row.user_id != partner_id
    merged = client.get("/v1/loans", headers=owner_headers)
    assert merged.status_code == 200, merged.text
    assert any(row["id"] == str(loan_id) for row in merged.json()["loans"])


def test_upcoming_loan_payment_shows_the_monthly_payment_not_the_balance(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    headers, _email, _owner_id, _ = _login(client, secret, "calendar")
    rupee_loan = _create_loan(client, headers, "INR")
    krone_loan = _create_loan(client, headers, "DKK")
    with system_session(database.owner_engine) as session:
        session.add(
            FxRate(
                pair="DKK/INR",
                # Earlier tests may already hold today's synthetic rate.
                as_of=_TODAY - timedelta(days=1),
                rate=Decimal("12.5"),
                source="synthetic-test",
                source_url="https://example.test/fx",
            )
        )

    networth = client.get("/v1/networth", headers=headers)
    assert networth.status_code == 200, networth.text
    payments = {
        event["asset_id"]: event["amount"]
        for event in networth.json()["maturity_calendar"]
        if event["kind"] == "loan_payment"
    }
    assert payments[str(rupee_loan)] == "10000.00"
    assert payments[str(krone_loan)] == "2500.00"
