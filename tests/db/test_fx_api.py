"""The PWA reads the DKK/INR rate from the API, not from an outside rate service."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from secrets import token_urlsafe
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient

from conftest import MigratedDatabase
from wealth.api.main import create_app
from wealth.api.money_map import get_today
from wealth.api.settings import Settings
from wealth.db import session as db_session
from wealth.db.models import FxRate
from wealth.db.session import system_session

pytestmark = pytest.mark.db
# Dates far from other tests' synthetic rates, which share the fx_rates table.
_RATE_DAY = date(2001, 3, 13)


def _client(
    migrated_database: MigratedDatabase, monkeypatch: pytest.MonkeyPatch, today: date
) -> tuple[TestClient, dict[str, str]]:
    secret = token_urlsafe(32)
    settings = Settings(
        env="test",
        auth_mode="dev",
        dev_auth_secret=secret,
        database_url="postgresql://wealth_api@localhost/wealth_test",
    )
    monkeypatch.setattr(db_session, "api_engine", lambda _url: migrated_database.api_engine)
    application = create_app(settings)
    application.dependency_overrides[get_today] = lambda: today
    token = jwt.encode(
        {
            "sub": f"fx-{uuid4().hex}",
            "email": f"fx-{uuid4().hex}@example.test",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    return TestClient(application), {"Authorization": f"Bearer {token}"}


@pytest.fixture
def stored_rate(migrated_database: MigratedDatabase) -> Iterator[MigratedDatabase]:
    with system_session(migrated_database.owner_engine) as session:
        if session.get(FxRate, ("DKK/INR", _RATE_DAY)) is None:
            session.add(
                FxRate(
                    pair="DKK/INR",
                    as_of=_RATE_DAY,
                    rate=Decimal("12.34560000"),
                    source="synthetic-test",
                    source_url="https://example.test/fx",
                )
            )
    yield migrated_database


def test_the_latest_stored_rate_is_returned_to_a_signed_in_person(
    stored_rate: MigratedDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, headers = _client(stored_rate, monkeypatch, _RATE_DAY + timedelta(days=2))
    with client:
        assert client.get("/v1/fx/dkk-inr").status_code == 401
        response = client.get("/v1/fx/dkk-inr", headers=headers)

    assert response.status_code == 200, response.text
    assert response.json() == {"date": "2001-03-13", "rate": "12.3456"}


def test_a_rate_older_than_ten_days_is_not_offered(
    stored_rate: MigratedDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, headers = _client(stored_rate, monkeypatch, _RATE_DAY + timedelta(days=30))
    with client:
        response = client.get("/v1/fx/dkk-inr", headers=headers)

    assert response.status_code == 404
