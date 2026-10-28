"""End-to-end API checks over the C05 non-owner PostgreSQL login."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from conftest import MigratedDatabase
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from wealth.api import main as api_main
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import Profile, User

pytestmark = pytest.mark.db


def _token(secret: str, subject: str, email: str) -> str:
    return jwt.encode(
        {
            "sub": subject,
            "email": email,
            "name": "Synthetic API User",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )


def test_api_engine_accepts_throwaway_member_login(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert migrated_database.test_login.startswith("wealth_api_test_")
    monkeypatch.setattr(db_session, "make_engine", lambda _url: migrated_database.api_engine)

    assert db_session.api_engine("synthetic-test-url") is migrated_database.api_engine


def test_api_engine_rejects_owner_login(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(db_session, "make_engine", lambda _url: migrated_database.owner_engine)

    with pytest.raises(ValueError, match="API database login must be non-superuser"):
        db_session.api_engine("synthetic-test-url")


def test_validation_problem_lists_safe_fields_without_echoing_inputs(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = secrets.token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql+psycopg://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    token = _token(secret, "synthetic-validation-user", "validation@example.test")
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(api_main.create_app(settings)) as client:
        response = client.put(
            "/v1/me/profile",
            headers=headers,
            json={"tax_regime": "invalid-regime", "monthly_expenses": "5L5"},
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    errors = {item["field"]: item["message"] for item in response.json()["errors"]}
    assert errors["tax_regime"] == "Input should be 'old' or 'new'"
    assert errors["monthly_expenses"] == "Input should be a valid INR amount"
    assert "invalid-regime" not in response.text
    assert "5L5" not in response.text
    assert '"input"' not in response.text


def test_api_user_bootstrap_profiles_risk_delete_and_deep_health(
    migrated_database: MigratedDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert migrated_database.test_login.startswith("wealth_api_test_")
    assert migrated_database.test_login != migrated_database.owner_role
    secret = secrets.token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql+psycopg://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    token_a = _token(secret, "synthetic-api-user-a", "api-user-a@example.test")
    token_b = _token(secret, "synthetic-api-user-b", "api-user-b@example.test")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    with TestClient(api_main.create_app(settings)) as client:
        first = client.get("/v1/me", headers=headers_a)
        second = client.get("/v1/me", headers=headers_a)
        other = client.get("/v1/me", headers=headers_b)
        health = client.get("/healthz?deep=1")

        assert first.status_code == 200
        assert second.status_code == 200
        first_user = first.json()
        second_user = second.json()
        other_user = other.json()
        assert first_user["id"] == second_user["id"]
        assert first_user["profile"] is None
        assert other_user["id"] != first_user["id"]
        assert other_user["profile"] is None
        assert health.status_code == 200
        assert health.json()["db"] == "ok"

        profile = client.put(
            "/v1/me/profile",
            headers=headers_a,
            json={
                "tax_regime": "new",
                "tax_slab_pct": "30.0000",
                "city_tier": 2,
                "monthly_expenses": "12500.00",
                "annual_income": "900000.00",
                "dependants": 0,
                "employment_type": "salaried",
                "income_stability": "high",
            },
        )
        assert profile.status_code == 200
        assert profile.json()["tax_slab_pct"] == "30.0000"
        assert profile.json()["financial_context"]["monthly_expenses"] == "12500.00"
        assert profile.json()["financial_context"]["annual_income"] == "900000.00"

        isolated = client.get("/v1/me", headers=headers_b)
        assert isolated.json()["profile"] is None

        questions = client.get("/v1/meta/risk-questions", headers=headers_a)
        assert questions.status_code == 200
        assert len(questions.json()) == 6
        answers = {question["id"]: question["options"][-1]["id"] for question in questions.json()}
        risk = client.put(
            "/v1/me/risk-answers",
            headers=headers_a,
            json={"answers": answers},
        )
        assert risk.status_code == 200
        assert risk.json()["final"] == "aggressive"

        removed = client.delete("/v1/me", headers=headers_a)
        assert removed.status_code == 204

    deleted_user_id = first_user["id"]
    with migrated_database.owner_engine.connect() as connection:
        assert (
            connection.scalar(
                select(func.count()).select_from(User).where(User.id == deleted_user_id)
            )
            == 0
        )
        assert (
            connection.scalar(
                select(func.count()).select_from(Profile).where(Profile.user_id == deleted_user_id)
            )
            == 0
        )
