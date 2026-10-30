"""Core-pick candidates keep the price history the fund selector needs."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from conftest import MigratedDatabase
from wealth.db.models import PriceEod, Security
from wealth.marketdata import store

pytestmark = pytest.mark.db


def _security(
    name: str, kind: str, *, plan: str | None = None, option: str | None = None
) -> Security:
    return Security(
        id=uuid4(),
        market="IN",
        symbol=f"CORE{uuid4().hex[:8].upper()}",
        name=name,
        security_type=kind,
        currency="INR",
        plan_type=plan,
        option=option,
    )


def test_core_candidates_keep_their_prices_through_the_nightly_cleanup(
    migrated_database: MigratedDatabase,
) -> None:
    etf = _security("Synthetic Nifty 50 ETF", "etf")
    index_fund = _security("Synthetic Nifty 50 Index Fund", "mf", plan="direct", option="growth")
    lock_in = _security(
        "Synthetic ELSS Tax Saver Nifty 50 Fund", "mf", plan="direct", option="growth"
    )
    regular = _security("Synthetic Nifty 50 Index Fund", "mf", plan="regular", option="growth")
    active = _security("Synthetic Flexi Cap Fund", "mf", plan="direct", option="growth")
    securities = (etf, index_fund, lock_in, regular, active)
    etf_id, index_id, lock_in_id, regular_id, active_id = (item.id for item in securities)

    with Session(migrated_database.owner_engine) as session, session.begin():
        session.add_all(securities)
        session.flush()
        candidates = store.core_candidate_ids(session)
        for security in securities:
            store.upsert_eod_price(session, security.id, date(2026, 10, 29), Decimal("100"))
        store.prune(session, date(2026, 10, 30))
        kept = {
            security_id
            for (security_id,) in session.execute(
                select(PriceEod.security_id)
                .where(
                    PriceEod.security_id.in_([etf_id, index_id, lock_in_id, regular_id, active_id])
                )
                .group_by(PriceEod.security_id)
                .having(func.count() > 0)
            )
        }

    assert {etf_id, index_id} <= candidates
    assert not {lock_in_id, regular_id, active_id} & candidates
    assert kept == {etf_id, index_id}
