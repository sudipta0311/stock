from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session
from tests.db.conftest import MigratedDatabase

from wealth.db.models import IndexValuation
from wealth.marketdata.store import latest_index_valuation_percentile

pytestmark = pytest.mark.db


def test_api_role_can_read_the_stored_sp500_valuation(
    migrated_database: MigratedDatabase,
) -> None:
    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add(
            IndexValuation(
                index_name="S&P 500",
                as_of=date(2026, 10, 30),
                pe=Decimal("25.5"),
                source="synthetic_test",
                source_url="https://example.test/us-index",
            )
        )

    with Session(migrated_database.api_engine) as session:
        valuation = latest_index_valuation_percentile(
            session,
            "S&P 500",
            date(2026, 10, 31),
        )

    assert valuation is not None
    assert valuation.pe == Decimal("25.50000000")
