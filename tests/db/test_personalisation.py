"""Tenant isolation and storage tests for weekly plans and investing preferences."""

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
from sqlalchemy import func, inspect, select, text, update
from sqlalchemy.orm import Session

from conftest import MigratedDatabase
from wealth.api.main import create_app
from wealth.api.money_map import get_today
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import (
    Action,
    AuditLog,
    HighlightDismissal,
    HouseholdMember,
    ResearchPackRecord,
    User,
)
from wealth.db.models import WeeklyPlan as WeeklyPlanRow
from wealth.db.session import user_session
from wealth.policy.personalise import WeeklyPlan
from wealth.policy.store import save_weekly_plan

pytestmark = pytest.mark.db
_TODAY = date(2026, 10, 28)
_WEEK = "2026-W44"


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
    application.dependency_overrides[get_today] = lambda: _TODAY
    with TestClient(application) as client:
        yield client, secret, application


def _auth(secret: str, label: str) -> tuple[dict[str, str], str]:
    subject = f"synthetic-policy-{label}-{uuid4()}"
    email = f"{label}-{uuid4()}@example.test"
    token = jwt.encode(
        {
            "sub": subject,
            "email": email,
            "name": "Synthetic Policy User",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}, subject


def _user_id(database: MigratedDatabase, subject: str) -> UUID:
    with database.owner_engine.connect() as connection:
        result = connection.scalar(select(User.id).where(User.auth_subject == subject))
    assert result is not None
    return result


def test_research_ideas_reads_selector_and_track_record_from_existing_pack(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, _application = api_client
    headers, _subject = _auth(secret, "c16b-pack-reader")
    pack = ResearchPackRecord(
        week=_WEEK,
        pack={
            "ideas": [],
            "core_picks": {
                "in_nifty50": [
                    {
                        "category_label": "India large companies",
                        "symbol": "SYNTH",
                        "name": "Synthetic large-company fund",
                        "metrics": {"expense_ratio_pct": None},
                        "reason_codes": ["ter_unavailable"],
                    }
                ]
            },
            "inav_premium_pct": {"in_nifty50": None},
            "track_record": {
                "markets": {
                    "IN": {
                        "status": "available",
                        "summary": "Research ideas since 2026-10-01: +1.0% vs Nifty 50.",
                    }
                }
            },
            "cio_agenda": {"week": _WEEK},
            "amc_international_suspended": None,
        },
    )
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(pack)

    response = client.get("/v1/research/ideas", params={"week": _WEEK}, headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ideas"] == []
    assert payload["core_picks"]["in_nifty50"][0]["symbol"] == "SYNTH"
    assert payload["track_record"]["markets"]["IN"]["status"] == "available"


def test_investing_preferences_validate_and_round_trip(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers, _subject = _auth(secret, "investing")
    initial = client.get("/v1/me/investing", headers=headers)
    assert initial.status_code == 200
    assert initial.json()["new_money_this_week"] == "0.00"
    assert initial.json()["satellite_pct"] == "20"

    saved = client.put(
        "/v1/me/investing",
        headers=headers,
        json={
            "monthly_sip_capacity": "45000",
            "existing_sips": [
                {
                    "sleeve": "equity_india",
                    "instrument_category": "nifty_index_fund",
                    "monthly_amount": "15000",
                }
            ],
            "new_money_this_week": "100000",
            "has_lrs_broker": True,
            "prefers_no_paperwork": False,
            "satellite_pct": "25",
            "birth_year": 1990,
            "lrs_used_this_fy": "50000",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["monthly_sip_capacity"] == "45000.00"
    assert client.get("/v1/me/investing", headers=headers).json() == saved.json()

    invalid = client.put(
        "/v1/me/investing",
        headers=headers,
        json={"satellite_pct": "26"},
    )
    assert invalid.status_code == 422


def test_plan_is_computed_on_demand_and_reused_when_inputs_match(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers, _subject = _auth(secret, "plan-on-demand")

    first = client.get("/v1/plan", headers=headers)
    assert first.status_code == 200
    assert first.json()["week"] == _WEEK
    assert first.json()["fresh"] is True
    assert len(first.json()["inputs_digest"]) == 64

    second = client.get("/v1/plan", headers=headers)
    assert second.status_code == 200
    assert second.json()["fresh"] is False
    assert second.json()["inputs_digest"] == first.json()["inputs_digest"]

    changed = client.put(
        "/v1/me/investing",
        headers=headers,
        json={"new_money_this_week": "25000"},
    )
    assert changed.status_code == 200
    third = client.get("/v1/plan", headers=headers)
    assert third.status_code == 200
    assert third.json()["fresh"] is True
    assert third.json()["inputs_digest"] != first.json()["inputs_digest"]

    stress = client.get("/v1/stress", headers=headers)
    assert stress.status_code == 200
    assert len(stress.json()["stress"]) == 5
    ideas = client.get("/v1/research/ideas", headers=headers)
    assert ideas.status_code == 200
    assert ideas.json()["ideas"] == []


def test_highlight_dismissal_is_persisted_per_login_and_household(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wealth.api import personalisation
    from wealth.policy.highlights import Highlight, HighlightContext

    client, secret, _application = api_client
    owner_headers, owner_subject = _auth(secret, "highlight-owner")
    other_headers, _other_subject = _auth(secret, "highlight-other")
    fingerprint = "a" * 24

    def fake_highlights(_plan: WeeklyPlan, context: HighlightContext) -> list[Highlight]:
        dismissed = context.dismissed_fingerprints
        if dismissed.get("synthetic_notice") == fingerprint:
            return []
        return [
            Highlight(
                key="synthetic_notice",
                screen=context.screen,
                title="Review one detail",
                body="This detail may change your next step.",
                learn_more_key="plan_details",
                severity="info",
            )
        ]

    monkeypatch.setattr(personalisation, "highlights", fake_highlights)
    monkeypatch.setattr(
        personalisation,
        "highlight_fingerprint",
        lambda _plan, key, _copy: fingerprint if key == "synthetic_notice" else None,
    )

    before = client.get("/v1/plan", headers=owner_headers)
    assert before.status_code == 200
    assert before.json()["highlights_by_screen"]["home"][0]["key"] == "synthetic_notice"

    dismissed = client.post("/v1/highlights/synthetic_notice/dismiss", headers=owner_headers)
    assert dismissed.status_code == 204

    saved = client.get("/v1/highlights/dismissals", headers=owner_headers)
    assert saved.status_code == 200
    assert len(saved.json()) == 1
    assert saved.json()[0]["key"] == "synthetic_notice"
    assert saved.json()[0]["fingerprint"] == fingerprint
    assert saved.json()[0]["dismissed_at"]
    after = client.get("/v1/plan", headers=owner_headers)
    assert after.status_code == 200
    assert after.json()["highlights_by_screen"]["home"] == []

    other_login = client.get("/v1/highlights/dismissals", headers=other_headers)
    assert other_login.status_code == 200
    assert other_login.json() == []

    owner_id = _user_id(migrated_database, owner_subject)
    with migrated_database.owner_engine.connect() as connection:
        dismissal_count = connection.scalar(
            select(func.count())
            .select_from(HighlightDismissal)
            .where(
                HighlightDismissal.login_user_id == owner_id,
                HighlightDismissal.household_owner_id == owner_id,
            )
        )
    assert dismissal_count == 1

    removed = client.delete("/v1/me", headers=owner_headers)
    assert removed.status_code == 204
    with migrated_database.owner_engine.connect() as connection:
        remaining = connection.scalar(
            select(func.count())
            .select_from(HighlightDismissal)
            .where(HighlightDismissal.login_user_id == owner_id)
        )
    assert remaining == 0


def test_plan_loads_fx_for_member_residency_without_a_matching_goal(
    api_client: tuple[TestClient, str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wealth.policy.data as policy_data

    client, secret, _application = api_client
    headers, _subject = _auth(secret, "member-fx")
    assert (
        client.put(
            "/v1/me/residency",
            headers=headers,
            json={"status": "NRI", "country": "DK"},
        ).status_code
        == 200
    )
    requested_pairs: list[str] = []

    def fake_latest_fx_rate(_session: object, pair: str, as_of: date) -> tuple[Decimal, date]:
        requested_pairs.append(pair)
        return Decimal("12"), as_of

    monkeypatch.setattr(policy_data.store, "latest_fx_rate", fake_latest_fx_rate)

    response = client.get("/v1/plan", headers=headers)

    assert response.status_code == 200
    assert {"USD/INR", "DKK/INR", "EUR/INR"} <= set(requested_pairs)


def test_plan_uses_defaults_when_onboarding_skipped_the_optional_questions(
    api_client: tuple[TestClient, str, object],
) -> None:
    client, secret, _application = api_client
    headers, _subject = _auth(secret, "skipped-optional-questions")
    # The short onboarding path saves only the tax regime; every optional answer is null.
    saved = client.put("/v1/me/profile", headers=headers, json={"tax_regime": "new"})
    assert saved.status_code == 200

    plan = client.get("/v1/plan", headers=headers)
    loans = client.get("/v1/loans", headers=headers)

    assert plan.status_code == 200
    assert loans.status_code == 200


def test_get_plan_loads_legacy_stored_null_residency_country(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, _application = api_client
    headers, subject = _auth(secret, "legacy-null-residency-country")
    first = client.get("/v1/plan", headers=headers)
    assert first.status_code == 200
    assert first.json()["residency"]["country"] == "IN"

    user_id = _user_id(migrated_database, subject)
    with migrated_database.owner_engine.begin() as connection:
        stored_plan = connection.scalar(
            select(WeeklyPlanRow.plan).where(
                WeeklyPlanRow.user_id == user_id,
                WeeklyPlanRow.week == _WEEK,
            )
        )
        assert stored_plan is not None
        legacy_plan = dict(stored_plan)
        legacy_residency = dict(legacy_plan["residency"])
        legacy_residency["country"] = None
        legacy_plan["residency"] = legacy_residency
        connection.execute(
            update(WeeklyPlanRow)
            .where(WeeklyPlanRow.user_id == user_id, WeeklyPlanRow.week == _WEEK)
            .values(plan=legacy_plan)
        )

    restored = client.get("/v1/plan", headers=headers)

    assert restored.status_code == 200
    assert restored.json()["fresh"] is False
    assert restored.json()["residency"]["country"] == "IN"


def test_residency_change_updates_me_and_recomputes_the_plan(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, _application = api_client
    headers, subject = _auth(secret, "residency-change")
    first_plan = client.get("/v1/plan", headers=headers)
    assert first_plan.status_code == 200

    saved = client.put(
        "/v1/me/residency",
        headers=headers,
        json={"status": "NRI", "country": "DK"},
    )
    assert saved.status_code == 200
    assert saved.json()["country"] == "DK"
    assert client.get("/v1/me/residency", headers=headers).json() == saved.json()
    assert client.get("/v1/me", headers=headers).json()["residency"] == saved.json()

    changed_plan = client.get("/v1/plan", headers=headers)
    assert changed_plan.status_code == 200
    assert changed_plan.json()["fresh"] is True
    assert changed_plan.json()["inputs_digest"] != first_plan.json()["inputs_digest"]
    assert changed_plan.json()["route"]["vehicle"] == "ucits_global_etf"
    assert changed_plan.json()["disclaimer"] == (
        "Personal tool for family and friends. Information and education only, "
        "not investment advice."
    )

    invalid = client.put("/v1/me/residency", headers=headers, json={"status": "NRI"})
    assert invalid.status_code == 422
    for country in ("US", "XX", "OTHER"):
        invalid = client.put(
            "/v1/me/residency",
            headers=headers,
            json={"status": "NRI", "country": country},
        )
        assert invalid.status_code == 422
    invalid_us_person = client.put(
        "/v1/me/residency",
        headers=headers,
        json={"status": "NRI", "country": "DK", "us_person": True},
    )
    assert invalid_us_person.status_code == 422

    user_id = _user_id(migrated_database, subject)
    with migrated_database.owner_engine.begin() as connection:
        connection.execute(
            update(HouseholdMember)
            .where(HouseholdMember.user_id == user_id, HouseholdMember.is_self.is_(True))
            .values(residency={"status": "NRI", "country": "US", "us_person": True})
        )
    stored = client.get("/v1/me/residency", headers=headers)
    assert stored.status_code == 200
    assert stored.json()["country"] == "OTHER"
    legacy_plan = client.get("/v1/plan", headers=headers)
    assert legacy_plan.status_code == 200
    assert "residency_country_unavailable" in legacy_plan.json()["residency_notes"]
    assert "estate_tax_us_domiciled_assets" not in legacy_plan.json()["route"]["warnings"]


def test_residency_migration_up_and_down_round_trip(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0008_alerts_delivery")
    with migrated_database.owner_engine.connect() as connection:
        profile_columns = {column["name"] for column in inspect(connection).get_columns("profiles")}
        goal_constraints = {
            item["name"]: item.get("sqltext", "")
            for item in inspect(connection).get_check_constraints("goals")
        }
    assert "residency" not in profile_columns
    assert "DKK" not in goal_constraints["ck_goals_currency"]

    command.upgrade(migrated_database.config, "head")
    with migrated_database.owner_engine.connect() as connection:
        profile_columns = {column["name"] for column in inspect(connection).get_columns("profiles")}
        goal_constraints = {
            item["name"]: item.get("sqltext", "")
            for item in inspect(connection).get_check_constraints("goals")
        }
    assert "residency" in profile_columns
    assert "DKK" in goal_constraints["ck_goals_currency"]
    assert all(
        currency in goal_constraints["ck_goals_currency"] for currency in ("INR", "EUR", "USD")
    )
    assert all(
        currency not in goal_constraints["ck_goals_currency"]
        for currency in ("SEK", "NOK", "GBP", "SGD", "CAD", "AUD")
    )


def test_action_patch_is_tenant_scoped_and_audited(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, _application = api_client
    headers_a, subject_a = _auth(secret, "action-a")
    headers_b, _subject_b = _auth(secret, "action-b")
    assert client.get("/v1/plan", headers=headers_a).status_code == 200
    assert client.get("/v1/plan", headers=headers_b).status_code == 200
    user_a = _user_id(migrated_database, subject_a)
    action_id = uuid4()
    with migrated_database.owner_engine.begin() as connection:
        connection.execute(
            Action.__table__.insert().values(
                id=action_id,
                user_id=user_a,
                week=_WEEK,
                rank=1,
                kind="BUY_LUMPSUM",
                payload={"instrument_category": "synthetic_index_fund"},
                status="proposed",
            )
        )

    assert client.get(f"/v1/actions?week={_WEEK}", headers=headers_b).json() == []
    hidden = client.patch(f"/v1/actions/{action_id}", headers=headers_b, json={"status": "done"})
    assert hidden.status_code == 404
    updated = client.patch(f"/v1/actions/{action_id}", headers=headers_a, json={"status": "done"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "done"
    with migrated_database.owner_engine.connect() as connection:
        audits = connection.scalar(
            select(func.count())
            .select_from(AuditLog)
            .where(AuditLog.user_id == user_a, AuditLog.action == "action_status_changed")
        )
    assert audits == 1


def test_weekly_plan_upsert_replaces_proposed_and_preserves_user_statuses(
    api_client: tuple[TestClient, str, object],
    migrated_database: MigratedDatabase,
) -> None:
    client, secret, _application = api_client
    headers, subject = _auth(secret, "weekly-upsert")
    assert client.get("/v1/me", headers=headers).status_code == 200
    user_id = _user_id(migrated_database, subject)
    with migrated_database.owner_engine.begin() as connection:
        for status, kind in (
            ("done", "PAY_DOWN_DEBT"),
            ("dismissed", "BUILD_EMERGENCY_FUND"),
            ("proposed", "BUY_STP"),
        ):
            connection.execute(
                Action.__table__.insert().values(
                    id=uuid4(),
                    user_id=user_id,
                    week=_WEEK,
                    rank=1,
                    kind=kind,
                    payload={"instrument_category": "synthetic_product"},
                    status=status,
                )
            )

    def plan(kind: str, digest: str) -> WeeklyPlan:
        return WeeklyPlan(
            week=_WEEK,
            as_of=_TODAY,
            allocation={},
            band_breaches=[],
            debt_answer={},
            foundation=[],
            route={},
            actions=[{"rank": 1, "kind": kind}],
            exit_cards=[],
            stress={},
            notes=[],
            inputs_digest=digest,
        )

    with user_session(migrated_database.owner_engine, user_id) as session:
        save_weekly_plan(session, user_id, plan("BUY_LUMPSUM", "a" * 64))
    with user_session(migrated_database.owner_engine, user_id) as session:
        save_weekly_plan(session, user_id, plan("BUY_STP", "b" * 64))

    with migrated_database.owner_engine.connect() as connection:
        statuses = list(
            connection.execute(
                select(Action.status, Action.kind).where(
                    Action.user_id == user_id,
                    Action.week == _WEEK,
                )
            )
        )
        plan_count = connection.scalar(
            select(func.count())
            .select_from(WeeklyPlanRow)
            .where(WeeklyPlanRow.user_id == user_id, WeeklyPlanRow.week == _WEEK)
        )
    assert {status for status, _kind in statuses} == {"done", "dismissed", "proposed"}
    assert {(status, kind) for status, kind in statuses if status != "proposed"} == {
        ("done", "PAY_DOWN_DEBT"),
        ("dismissed", "BUILD_EMERGENCY_FUND"),
    }
    assert [(status, kind) for status, kind in statuses if status == "proposed"] == [
        ("proposed", "BUY_STP")
    ]
    assert plan_count == 1


def test_weekly_job_loads_each_user_inside_user_session(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wealth.policy.run_weekly as weekly

    original_loader = weekly.load_policy_inputs
    user_ids: list[UUID] = []

    def checked_loader(
        session, user_id: UUID, week: str, as_of: date, *, deployment_mode: str = "personal"
    ):
        assert session.scalar(text("SELECT current_user")) == "wealth_app"
        assert session.scalar(text("SELECT current_setting('app.user_id', true)")) == str(user_id)
        user_ids.append(user_id)
        return original_loader(session, user_id, week, as_of, deployment_mode=deployment_mode)

    monkeypatch.setattr(weekly, "load_policy_inputs", checked_loader)
    counts = weekly.run_weekly(migrated_database.owner_engine, _WEEK, _TODAY)
    assert counts.failed == 0
    assert counts.users == counts.plans
    assert len(user_ids) == counts.users


def test_z_personalisation_migration_can_downgrade_and_upgrade(
    migrated_database: MigratedDatabase,
) -> None:
    command.downgrade(migrated_database.config, "0006_market_data")
    inspector = inspect(migrated_database.owner_engine)
    assert "weekly_plans" not in inspector.get_table_names()
    assert "research_packs" not in inspector.get_table_names()
    assert "investing" not in {column["name"] for column in inspector.get_columns("profiles")}

    command.upgrade(migrated_database.config, "head")
    inspector = inspect(migrated_database.owner_engine)
    assert "weekly_plans" in inspector.get_table_names()
    assert "research_packs" in inspector.get_table_names()
    assert "investing" in {column["name"] for column in inspector.get_columns("profiles")}
