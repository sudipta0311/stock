"""Shared-household access, invite, RLS, member attribution, and merge checks."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from conftest import MigratedDatabase
from wealth.api.main import create_app
from wealth.api.money_map import get_today
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import (
    Account,
    Action,
    AuditLog,
    Consent,
    FundLaggingCheck,
    FundTransaction,
    HoldingLot,
    HouseholdAccess,
    HouseholdInvite,
    HouseholdMember,
    ImportItem,
    ImportJob,
    ImportTemplate,
    ManualAsset,
    Profile,
    PushSubscription,
    Security,
)
from wealth.db.session import system_session, user_session

pytestmark = pytest.mark.db
_TODAY = date(2026, 10, 28)


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


def _login(client: TestClient, secret: str, label: str) -> tuple[dict[str, str], str, str, UUID]:
    suffix = uuid4().hex
    subject = f"shared-{label}-{suffix}"
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
    assert client.get("/v1/me", headers=headers).status_code == 200
    return headers, subject, email, UUID(str(client.get("/v1/me", headers=headers).json()["id"]))


def _household(headers: dict[str, str], owner_id: UUID) -> dict[str, str]:
    return {**headers, "X-Household": str(owner_id)}


def _invite_token(response: dict[str, object]) -> str:
    link = str(response["share_link"])
    token = parse_qs(urlparse(link).query)["token"][0]
    assert len(token) >= 43
    return token


def _make_invite(
    client: TestClient,
    headers: dict[str, str],
    email: str,
    *,
    role: str = "partner",
    member_id: UUID | None = None,
) -> tuple[dict[str, object], str]:
    payload: dict[str, object] = {"email": email, "role": role}
    if member_id is not None:
        payload["member_id"] = str(member_id)
    response = client.post("/v1/households/invites", headers=headers, json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    return body, _invite_token(body)


def _problem_code(response: object) -> tuple[int, str | None, str | None]:
    assert hasattr(response, "status_code")
    body = response.json()  # type: ignore[attr-defined]
    return response.status_code, body.get("code"), body.get("detail")  # type: ignore[attr-defined]


def _seed_invite(
    database: MigratedDatabase,
    owner_id: UUID,
    email: str,
    member_id: UUID,
) -> str:
    token = secrets.token_urlsafe(32)
    with system_session(database.owner_engine) as session:
        session.add(
            HouseholdInvite(
                owner_user_id=owner_id,
                email_lower=email.lower(),
                role="partner",
                member_id=member_id,
                token_sha256=hashlib.sha256(token.encode("utf-8")).hexdigest(),
                expires_at=datetime.now(UTC) + timedelta(days=7),
            )
        )
    return token


def test_shared_household_grant_uses_linked_member_and_enforces_viewer_rls(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_subject, _owner_email, owner_id = _login(client, secret, "owner")
    partner_headers, _partner_subject, partner_email, partner_id = _login(client, secret, "partner")
    stranger_headers, _stranger_subject, _stranger_email, stranger_id = _login(
        client, secret, "stranger"
    )
    viewer_headers, _viewer_subject, viewer_email, viewer_id = _login(client, secret, "viewer")

    owner_residency = client.put(
        "/v1/me/residency",
        headers=owner_headers,
        json={"status": "NRI", "country": "DK"},
    )
    assert owner_residency.status_code == 200
    member_response = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Anita", "relationship": "spouse"},
    )
    assert member_response.status_code == 201
    linked_member_id = UUID(member_response.json()["id"])

    partner_invite, partner_token = _make_invite(
        client, owner_headers, partner_email.upper(), member_id=linked_member_id
    )
    with database.owner_engine.connect() as connection:
        stored_hash = connection.scalar(
            select(HouseholdInvite.token_sha256).where(
                HouseholdInvite.id == UUID(str(partner_invite["id"]))
            )
        )
    assert stored_hash == hashlib.sha256(partner_token.encode("utf-8")).hexdigest()
    assert partner_token not in str(stored_hash)

    hidden_before_join = client.get("/v1/assets", headers=_household(partner_headers, owner_id))
    assert hidden_before_join.status_code == 404
    wrong_email = client.post(
        "/v1/households/join",
        headers=stranger_headers,
        json={"token": partner_token, "bring_my_data": False},
    )
    assert _problem_code(wrong_email) == (400, "invalid_invite", "invalid_invite")

    joined = client.post(
        "/v1/households/join",
        headers=partner_headers,
        json={"token": partner_token, "bring_my_data": False},
    )
    assert joined.status_code == 200
    assert joined.json()["owner_id"] == str(owner_id)
    households = client.get("/v1/households", headers=partner_headers)
    assert households.status_code == 200
    assert households.json()[0]["owner_id"] == str(partner_id)
    shared = next(item for item in households.json() if item["owner_id"] == str(owner_id))
    assert shared["role"] == "partner"
    assert shared["member_count"] == 2

    partner_me = client.get("/v1/me", headers=_household(partner_headers, owner_id))
    owner_me = client.get("/v1/me", headers=owner_headers)
    assert partner_me.status_code == owner_me.status_code == 200
    assert partner_me.json()["me_member_id"] == str(linked_member_id)
    assert owner_me.json()["me_member_id"] != partner_me.json()["me_member_id"]
    assert partner_me.json()["role"] == "partner"
    assert partner_me.json()["household_owner_id"] == str(owner_id)

    added = client.post(
        "/v1/assets",
        headers=_household(partner_headers, owner_id),
        json={"asset_type": "savings", "bank": "Synthetic Bank", "balance": "1250"},
    )
    assert added.status_code == 201, added.text
    asset_id = UUID(added.json()["id"])
    assert added.json()["member_id"] == str(linked_member_id)
    assert client.get("/v1/assets", headers=_household(owner_headers, owner_id)).json()[0][
        "id"
    ] == str(asset_id)

    changed_residency = client.put(
        "/v1/me/residency",
        headers=_household(partner_headers, owner_id),
        json={"status": "RI", "country": "IN"},
    )
    assert changed_residency.status_code == 200
    assert (
        client.get("/v1/me/residency", headers=_household(partner_headers, owner_id)).json()[
            "status"
        ]
        == "RI"
    )
    assert client.get("/v1/me/residency", headers=owner_headers).json() == owner_residency.json()

    owner_plan = client.get("/v1/plan", headers=owner_headers)
    partner_plan = client.get("/v1/plan", headers=_household(partner_headers, owner_id))
    assert owner_plan.status_code == partner_plan.status_code == 200
    assert owner_plan.json()["inputs_digest"] == partner_plan.json()["inputs_digest"]
    assert owner_plan.json()["me_member_id"] != partner_plan.json()["me_member_id"]

    action_id = uuid4()
    with database.owner_engine.begin() as connection:
        connection.execute(
            Action.__table__.insert().values(
                id=action_id,
                user_id=owner_id,
                week="2026-W44",
                rank=1,
                kind="BUILD_EMERGENCY_FUND",
                payload={"member_id": str(linked_member_id), "member_name": "Anita"},
                status="proposed",
            )
        )
    marked_done = client.patch(
        f"/v1/actions/{action_id}",
        headers=_household(partner_headers, owner_id),
        json={"status": "done"},
    )
    assert marked_done.status_code == 200
    assert marked_done.json()["done_by"] == str(partner_id)
    owner_actions = client.get("/v1/actions?week=2026-W44", headers=owner_headers).json()
    done_action = next(action for action in owner_actions if action["id"] == str(action_id))
    assert done_action["done_by"] == str(partner_id)

    viewer_invite, viewer_token = _make_invite(client, owner_headers, viewer_email, role="viewer")
    viewer_join = client.post(
        "/v1/households/join",
        headers=viewer_headers,
        json={"token": viewer_token, "bring_my_data": False},
    )
    assert viewer_join.status_code == 200
    assert client.get("/v1/assets", headers=_household(viewer_headers, owner_id)).status_code == 200
    blocked_write = client.post(
        "/v1/assets",
        headers=_household(viewer_headers, owner_id),
        json={"asset_type": "savings", "bank": "Blocked Bank", "balance": "1"},
    )
    assert blocked_write.status_code == 403
    assert blocked_write.json()["code"] == "read_only"
    assert (
        client.patch(
            f"/v1/actions/{action_id}",
            headers=_household(viewer_headers, owner_id),
            json={"status": "done"},
        ).status_code
        == 403
    )

    with (
        pytest.raises(DBAPIError),
        user_session(database.api_engine, viewer_id, owner_id) as session,
    ):
        session.add(
            ManualAsset(
                user_id=owner_id,
                member_id=linked_member_id,
                asset_type="savings",
                details={"asset_type": "savings", "bank": "Blocked Bank", "balance": "1"},
            )
        )
        session.flush()

    assert (
        client.get("/v1/assets", headers=_household(stranger_headers, owner_id)).status_code == 404
    )
    with user_session(database.api_engine, stranger_id, owner_id) as session:
        assert session.scalar(select(func.count()).select_from(ManualAsset)) == 0
    with user_session(database.api_engine, partner_id, owner_id) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(
                    AuditLog.user_id == owner_id,
                    AuditLog.actor_user_id == partner_id,
                    AuditLog.action == "household_row_insert",
                )
            )
            >= 1
        )

    partner_cannot_invite = client.post(
        "/v1/households/invites",
        headers=_household(partner_headers, owner_id),
        json={"email": f"extra-{uuid4().hex}@example.test", "role": "partner"},
    )
    assert partner_cannot_invite.status_code == 403
    partner_cannot_change_role = client.patch(
        f"/v1/households/access/{partner_id}",
        headers=_household(partner_headers, owner_id),
        json={"role": "viewer"},
    )
    assert partner_cannot_change_role.status_code == 403
    partner_cannot_revoke_owner = client.delete(
        f"/v1/households/access/{owner_id}",
        headers=_household(partner_headers, owner_id),
    )
    assert partner_cannot_revoke_owner.status_code == 403
    assert (
        client.get("/v1/assets", headers=_household(partner_headers, stranger_id)).status_code
        == 404
    )

    revoked = client.delete(f"/v1/households/access/{partner_id}", headers=owner_headers)
    assert revoked.status_code == 204
    assert (
        client.get("/v1/assets", headers=_household(partner_headers, owner_id)).status_code == 404
    )
    with user_session(database.api_engine, partner_id, owner_id) as session:
        assert session.scalar(select(func.count()).select_from(ManualAsset)) == 0

    with Session(database.api_engine) as session, session.begin():
        session.execute(text("SET LOCAL ROLE wealth_app"))
        session.execute(
            text("SELECT set_config('app.user_id', :user_id, true)"),
            {"user_id": str(owner_id)},
        )
        session.execute(text("RESET app.household_owner"))
        assert session.scalar(select(func.count()).select_from(ManualAsset)) == 0


def test_invalid_invites_share_one_error_for_expired_revoked_and_reused_tokens(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_subject, _owner_email, owner_id = _login(client, secret, "invite-owner")
    grantee_headers, _subject, grantee_email, _grantee_id = _login(client, secret, "invite-grantee")

    expired, expired_token = _make_invite(
        client, owner_headers, f"expired-{uuid4().hex}@example.test"
    )
    with database.owner_engine.begin() as connection:
        connection.execute(
            update(HouseholdInvite)
            .where(HouseholdInvite.id == UUID(str(expired["id"])))
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    expired_user, _expired_subject, _expired_email, _expired_id = _login(
        client, secret, "expired-user"
    )
    expired_response = client.post(
        "/v1/households/join",
        headers=expired_user,
        json={"token": expired_token},
    )

    revoked, revoked_token = _make_invite(
        client, owner_headers, f"revoked-{uuid4().hex}@example.test"
    )
    assert (
        client.delete(f"/v1/households/invites/{revoked['id']}", headers=owner_headers).status_code
        == 204
    )
    revoked_user, _revoked_subject, _revoked_email, _revoked_id = _login(
        client, secret, "revoked-user"
    )
    revoked_response = client.post(
        "/v1/households/join",
        headers=revoked_user,
        json={"token": revoked_token},
    )

    accepted, accepted_token = _make_invite(client, owner_headers, grantee_email)
    accepted_response = client.post(
        "/v1/households/join",
        headers=grantee_headers,
        json={"token": accepted_token},
    )
    assert accepted_response.status_code == 200
    reused_response = client.post(
        "/v1/households/join",
        headers=grantee_headers,
        json={"token": accepted_token},
    )

    assert _problem_code(expired_response) == (400, "invalid_invite", "invalid_invite")
    assert _problem_code(revoked_response) == _problem_code(expired_response)
    assert _problem_code(reused_response) == _problem_code(expired_response)
    assert owner_id == UUID(accepted_response.json()["owner_id"])


def test_owner_self_member_cannot_be_invited_or_accepted_as_a_grantee(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_subject, _owner_email, owner_id = _login(client, secret, "self-owner")
    grantee_headers, _subject, grantee_email, _grantee_id = _login(client, secret, "self-grantee")
    self_member_id = UUID(client.get("/v1/me", headers=owner_headers).json()["me_member_id"])

    refused = client.post(
        "/v1/households/invites",
        headers=owner_headers,
        json={"email": grantee_email, "role": "partner", "member_id": str(self_member_id)},
    )
    assert _problem_code(refused)[:2] == (409, "member_is_owner")

    forged_token = _seed_invite(database, owner_id, grantee_email, self_member_id)
    forged_acceptance = client.post(
        "/v1/households/join",
        headers=grantee_headers,
        json={"token": forged_token, "bring_my_data": False},
    )
    assert _problem_code(forged_acceptance) == (400, "invalid_invite", "invalid_invite")


def test_member_cannot_be_linked_to_two_logins_or_pending_invites(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_subject, _owner_email, owner_id = _login(client, secret, "unique-owner")
    first_headers, _first_subject, first_email, first_id = _login(client, secret, "first-grantee")
    second_headers, _second_subject, second_email, second_id = _login(
        client, secret, "second-grantee"
    )

    linked_member = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Anita", "relationship": "spouse"},
    )
    linked_member_id = UUID(linked_member.json()["id"])
    _invite, first_token = _make_invite(
        client, owner_headers, first_email, member_id=linked_member_id
    )
    accepted = client.post(
        "/v1/households/join",
        headers=first_headers,
        json={"token": first_token, "bring_my_data": False},
    )
    assert accepted.status_code == 200

    active_duplicate = client.post(
        "/v1/households/invites",
        headers=owner_headers,
        json={"email": second_email, "role": "partner", "member_id": str(linked_member_id)},
    )
    assert _problem_code(active_duplicate)[:2] == (409, "member_already_linked")

    forged_duplicate_token = _seed_invite(database, owner_id, second_email, linked_member_id)
    forged_duplicate = client.post(
        "/v1/households/join",
        headers=second_headers,
        json={"token": forged_duplicate_token, "bring_my_data": False},
    )
    assert _problem_code(forged_duplicate) == (400, "invalid_invite", "invalid_invite")

    with pytest.raises(IntegrityError), system_session(database.owner_engine) as session:
        session.add(
            HouseholdAccess(
                owner_user_id=owner_id,
                grantee_user_id=second_id,
                member_id=linked_member_id,
                role="viewer",
            )
        )
        session.flush()

    pending_member = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Pending member", "relationship": "other"},
    )
    pending_member_id = UUID(pending_member.json()["id"])
    _pending, _pending_token = _make_invite(
        client, owner_headers, f"pending-{uuid4().hex}@example.test", member_id=pending_member_id
    )
    pending_duplicate = client.post(
        "/v1/households/invites",
        headers=owner_headers,
        json={
            "email": f"different-{uuid4().hex}@example.test",
            "role": "viewer",
            "member_id": str(pending_member_id),
        },
    )
    assert _problem_code(pending_duplicate)[:2] == (409, "member_already_linked")
    assert first_id != second_id


def test_partner_and_viewer_can_leave_and_access_ends_immediately(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_subject, _owner_email, owner_id = _login(client, secret, "leave-owner")
    partner_headers, _partner_subject, partner_email, partner_id = _login(
        client, secret, "leave-partner"
    )
    viewer_headers, _viewer_subject, viewer_email, viewer_id = _login(
        client, secret, "leave-viewer"
    )

    for name in ("Partner member", "Viewer member"):
        created = client.post(
            "/v1/members",
            headers=owner_headers,
            json={"display_name": name, "relationship": "other"},
        )
        assert created.status_code == 201
        member_id = UUID(created.json()["id"])
        grantee_headers, email = (
            (partner_headers, partner_email)
            if name.startswith("Partner")
            else (viewer_headers, viewer_email)
        )
        _invite, token = _make_invite(
            client,
            owner_headers,
            email,
            role="partner" if name.startswith("Partner") else "viewer",
            member_id=member_id,
        )
        joined = client.post("/v1/households/join", headers=grantee_headers, json={"token": token})
        assert joined.status_code == 200

    partner_household = _household(partner_headers, owner_id)
    partner_leave = client.delete("/v1/households/leave", headers=partner_household)
    assert partner_leave.status_code == 204
    assert client.get("/v1/assets", headers=partner_household).status_code == 404
    assert client.delete("/v1/households/leave", headers=partner_household).status_code == 404
    with (
        pytest.raises(DBAPIError) as second_partner_leave,
        user_session(database.api_engine, partner_id, owner_id) as session,
    ):
        session.scalar(text("SELECT app_leave_household(:owner_id)"), {"owner_id": owner_id})
    diagnostic = getattr(getattr(second_partner_leave.value, "orig", None), "diag", None)
    assert getattr(diagnostic, "message_primary", None) == "not_found"

    viewer_household = _household(viewer_headers, owner_id)
    viewer_leave = client.delete("/v1/households/leave", headers=viewer_household)
    assert viewer_leave.status_code == 204
    assert client.get("/v1/assets", headers=viewer_household).status_code == 404
    assert client.delete("/v1/households/leave", headers=viewer_household).status_code == 404
    assert client.delete("/v1/households/leave", headers=owner_headers).status_code == 404

    with system_session(database.owner_engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(
                    AuditLog.user_id == owner_id,
                    AuditLog.action == "household_access_left",
                    AuditLog.actor_user_id.in_([partner_id, viewer_id]),
                )
            )
            == 2
        )


def test_member_with_active_login_requires_access_revocation_before_deletion(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_subject, _owner_email, owner_id = _login(client, secret, "delete-owner")
    partner_headers, _partner_subject, partner_email, partner_id = _login(
        client, secret, "delete-partner"
    )
    other_partner_headers, _other_subject, other_email, _other_partner_id = _login(
        client, secret, "other-delete-partner"
    )
    member = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Linked member", "relationship": "spouse"},
    )
    member_id = UUID(member.json()["id"])
    _invite, token = _make_invite(client, owner_headers, partner_email, member_id=member_id)
    assert (
        client.post(
            "/v1/households/join", headers=partner_headers, json={"token": token}
        ).status_code
        == 200
    )
    other_member = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Other member", "relationship": "other"},
    )
    _other_invite, other_token = _make_invite(
        client,
        owner_headers,
        other_email,
        role="partner",
        member_id=UUID(other_member.json()["id"]),
    )
    assert (
        client.post(
            "/v1/households/join", headers=other_partner_headers, json={"token": other_token}
        ).status_code
        == 200
    )

    refused = client.delete(f"/v1/members/{member_id}", headers=owner_headers)
    assert _problem_code(refused)[:2] == (409, "member_linked_to_login")
    partner_refused = client.delete(
        f"/v1/members/{member_id}", headers=_household(other_partner_headers, owner_id)
    )
    assert _problem_code(partner_refused)[:2] == (409, "member_linked_to_login")

    revoked = client.delete(f"/v1/households/access/{partner_id}", headers=owner_headers)
    assert revoked.status_code == 204
    with system_session(database.owner_engine) as session:
        access = session.get(HouseholdAccess, (owner_id, partner_id))
        assert access is not None and access.revoked_at is not None

    deleted = client.delete(f"/v1/members/{member_id}", headers=owner_headers)
    assert deleted.status_code == 204


def test_bring_my_data_moves_rows_to_linked_member_and_preserves_login_data(
    api_client: tuple[TestClient, str, MigratedDatabase],
) -> None:
    client, secret, database = api_client
    owner_headers, _owner_subject, _owner_email, owner_id = _login(client, secret, "merge-owner")
    partner_headers, _partner_subject, partner_email, partner_id = _login(
        client, secret, "merge-partner"
    )
    assert (
        client.put(
            "/v1/me/residency",
            headers=owner_headers,
            json={"status": "NRI", "country": "DK"},
        ).status_code
        == 200
    )
    owner_member = client.post(
        "/v1/members",
        headers=owner_headers,
        json={"display_name": "Anita", "relationship": "spouse"},
    ).json()
    linked_member_id = UUID(owner_member["id"])
    assert (
        client.put(
            "/v1/me/residency",
            headers=partner_headers,
            json={"status": "RI", "country": "IN"},
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/v1/me/investing",
            headers=partner_headers,
            json={"new_money_this_week": "25000"},
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/v1/me/consents",
            headers=partner_headers,
            json={
                "weekly_email": True,
                "reminder_email": False,
                "push": False,
                "urgent_alerts": False,
            },
        ).status_code
        == 200
    )

    partner_asset = client.post(
        "/v1/assets",
        headers=partner_headers,
        json={"asset_type": "savings", "bank": "Partner Bank", "balance": "5000"},
    )
    assert partner_asset.status_code == 201
    source_asset_id = UUID(partner_asset.json()["id"])
    partner_child = client.post(
        "/v1/members",
        headers=partner_headers,
        json={"display_name": "Synthetic Child", "relationship": "child"},
    )
    assert partner_child.status_code == 201, partner_child.text
    partner_child_id = UUID(partner_child.json()["id"])

    with system_session(database.owner_engine) as session:
        source_member = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == partner_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        assert source_member is not None
        source_member.tax = {"tax_regime": "new", "tax_slab_pct": "20"}
        source_member.investing = {"new_money_this_week": "25000"}
        source_member.residency = {"status": "RI", "country": "IN", "us_person": False}
        partner_profile = session.scalar(select(Profile).where(Profile.user_id == partner_id))
        assert partner_profile is not None
        partner_profile.residency = {"status": "RI", "country": "IN", "us_person": False}
        partner_profile.investing = {"new_money_this_week": "25000"}
        account = Account(
            user_id=partner_id,
            member_id=source_member.id,
            institution="Synthetic Custodian",
            account_type="broker",
            masked_number="X1234",
        )
        session.add(account)
        session.flush()
        job = ImportJob(user_id=partner_id, doc_type="cas", status="ready")
        session.add(job)
        session.flush()
        session.add_all(
            [
                ImportTemplate(
                    user_id=owner_id,
                    name="Owner mapping",
                    kind="holdings",
                    header_fingerprint="a" * 64,
                    header_names=["quantity", "symbol"],
                    columns={"quantity": "Quantity", "symbol": "Symbol"},
                ),
                ImportTemplate(
                    user_id=partner_id,
                    name="Partner duplicate",
                    kind="holdings",
                    header_fingerprint="a" * 64,
                    header_names=["quantity", "symbol"],
                    columns={"quantity": "Qty", "symbol": "Ticker"},
                ),
                ImportTemplate(
                    user_id=partner_id,
                    name="Partner unique",
                    kind="tradebook",
                    header_fingerprint="b" * 64,
                    header_names=["quantity", "symbol", "trade date"],
                    columns={"quantity": "Units", "symbol": "Ticker", "trade_date": "Date"},
                ),
            ]
        )
        security = Security(
            market="IN",
            symbol=f"SYN{uuid4().hex[:8]}",
            name="Synthetic Index",
            security_type="equity",
            currency="INR",
        )
        session.add(security)
        session.flush()
        session.add(
            HoldingLot(
                user_id=partner_id,
                account_id=account.id,
                member_id=source_member.id,
                security_id=security.id,
                quantity=Decimal("2"),
                cost_per_unit=Decimal("100"),
                source="manual",
                import_job_id=job.id,
            )
        )
        # Member-keyed rows in tables added after 0011 must move with their member. The
        # child row only works if the member foreign key is deferrable: the table sorts
        # before household_members in the merge loop.
        for member_id in (source_member.id, partner_child_id):
            session.add(
                FundTransaction(
                    user_id=partner_id,
                    member_id=member_id,
                    security_id=security.id,
                    transaction_date=date(2026, 1, 15),
                    kind="buy",
                    amount=Decimal("200"),
                    units=Decimal("2"),
                )
            )
        session.add(
            FundLaggingCheck(
                user_id=partner_id,
                member_id=source_member.id,
                security_id=security.id,
                month_end=date(2026, 9, 30),
                bottom_quartile=True,
            )
        )
        session.add(
            ImportItem(
                user_id=partner_id,
                job_id=job.id,
                item_kind="holding",
                payload={"symbol": security.symbol},
            )
        )
        session.add(
            PushSubscription(
                user_id=partner_id,
                endpoint=f"https://push.example.test/{uuid4().hex}",
                keys={"p256dh": "synthetic", "auth": "synthetic"},
            )
        )

    invite, token = _make_invite(client, owner_headers, partner_email, member_id=linked_member_id)
    consent_before_merge = client.get("/v1/me/consents", headers=partner_headers).json()
    merged = client.post(
        "/v1/households/join",
        headers=partner_headers,
        json={"token": token, "bring_my_data": True},
    )
    assert merged.status_code == 200, merged.text
    body = merged.json()
    assert body["owner_id"] == str(owner_id)
    assert body["merge_counts"]["manual_assets"] >= 1
    assert body["merge_counts"]["accounts"] == 1
    assert body["merge_counts"]["holdings_lots"] == 1
    assert body["merge_counts"]["import_jobs"] == 1
    assert body["merge_counts"]["import_items"] == 1
    assert body["merge_counts"]["fund_transactions"] == 2
    assert body["merge_counts"]["fund_lagging_checks"] == 1
    assert body["merge_counts"]["import_templates"] == 1
    assert {"residency", "tax", "investing"} <= set(body["overwritten_fields"])
    assert invite["member_id"] == str(linked_member_id)

    with system_session(database.owner_engine) as session:
        moved_asset = session.get(ManualAsset, source_asset_id)
        assert moved_asset is not None
        assert moved_asset.user_id == owner_id
        assert moved_asset.member_id == linked_member_id
        moved_transactions = session.scalars(
            select(FundTransaction).where(FundTransaction.user_id == owner_id)
        ).all()
        assert {row.member_id for row in moved_transactions} == {
            linked_member_id,
            partner_child_id,
        }
        moved_checks = session.scalars(
            select(FundLaggingCheck).where(FundLaggingCheck.user_id == owner_id)
        ).all()
        moved_templates = session.scalars(
            select(ImportTemplate).where(ImportTemplate.user_id == owner_id)
        ).all()
        assert {row.name for row in moved_templates} == {"Owner mapping", "Partner unique"}
        assert (
            session.scalar(
                select(func.count())
                .select_from(ImportTemplate)
                .where(ImportTemplate.user_id == partner_id)
            )
            == 0
        )
        assert [row.member_id for row in moved_checks] == [linked_member_id]
        assert (
            session.scalar(
                select(func.count()).select_from(Account).where(Account.user_id == partner_id)
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count()).select_from(HoldingLot).where(HoldingLot.user_id == partner_id)
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count()).select_from(ImportJob).where(ImportJob.user_id == partner_id)
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count()).select_from(ImportItem).where(ImportItem.user_id == partner_id)
            )
            == 0
        )
        source_self = session.scalar(
            select(HouseholdMember).where(
                HouseholdMember.user_id == partner_id,
                HouseholdMember.is_self.is_(True),
            )
        )
        linked_member = session.get(HouseholdMember, linked_member_id)
        assert source_self is not None and source_self.user_id == partner_id
        assert linked_member is not None
        assert linked_member.residency["country"] == "IN"
        assert linked_member.investing["new_money_this_week"] == "25000"
        assert (
            session.scalar(
                select(func.count()).select_from(Consent).where(Consent.user_id == partner_id)
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(PushSubscription)
                .where(PushSubscription.user_id == partner_id)
            )
            == 1
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(AuditLog.user_id == owner_id, AuditLog.action == "household_merge_performed")
            )
            == 1
        )

    assert (
        client.get("/v1/me/residency", headers=_household(partner_headers, owner_id)).json()[
            "country"
        ]
        == "IN"
    )
    assert client.get("/v1/me/residency", headers=owner_headers).json()["country"] == "DK"
    assert client.get("/v1/me/consents", headers=partner_headers).json() == consent_before_merge
    with user_session(database.api_engine, partner_id, owner_id) as session:
        assert (
            session.scalar(
                select(func.count()).select_from(ManualAsset).where(ManualAsset.user_id == owner_id)
            )
            >= 1
        )
        repeated = session.scalar(
            text("SELECT app_merge_into_household(:owner_id)"), {"owner_id": owner_id}
        )
        assert all(count == 0 for count in repeated["counts"].values())
        assert repeated["overwritten_fields"] == []


def test_every_member_foreign_key_is_deferrable_for_household_merge(
    migrated_database: MigratedDatabase,
) -> None:
    # app_merge_into_household defers constraints and moves tables before their members.
    with migrated_database.owner_engine.connect() as connection:
        immediate = connection.execute(
            text(
                """
                SELECT con.conrelid::regclass::text
                FROM pg_catalog.pg_constraint AS con
                WHERE con.contype = 'f'
                  AND con.confrelid = 'household_members'::regclass
                  AND cardinality(con.conkey) > 1
                  AND NOT con.condeferrable
                """
            )
        ).scalars()
        assert sorted(immediate) == []
