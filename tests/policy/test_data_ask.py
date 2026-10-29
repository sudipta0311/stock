from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest

from wealth.marketdata import store
from wealth.policy import data


class FakeRows:
    def __init__(self, rows: list[tuple[Decimal, str, date, str]]) -> None:
        self.rows = rows

    def all(self) -> list[tuple[Decimal, str, date, str]]:
        return self.rows


class FakeSession:
    def __init__(self, rows: list[tuple[Decimal, str, date, str]]) -> None:
        self.rows = rows

    def execute(self, _statement: Any) -> FakeRows:
        return FakeRows(self.rows)


def test_ask_deposits_convert_on_transaction_date_and_net_withdrawals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_date = date(2026, 5, 1)
    flows = [
        (Decimal("100"), "EUR", trade_date, "deposit"),
        (Decimal("500"), "DKK", trade_date, "deposit"),
        (Decimal("250"), "DKK", trade_date, "withdrawal"),
    ]
    rates = {
        "EUR/INR": (Decimal("100"), trade_date, "ECB"),
        "DKK/INR": (Decimal("10"), trade_date, "ECB"),
    }
    monkeypatch.setattr(store, "latest_fx_rate", lambda _s, pair, _on: rates.get(pair))
    used, missing = data._ask_deposits_dkk(
        FakeSession(flows),  # type: ignore[arg-type]
        user_id=UUID("10000000-0000-4000-8000-000000000001"),
        member_id=UUID("20000000-0000-4000-8000-000000000001"),
        as_of=date(2026, 10, 28),
    )
    assert used == Decimal("1250")
    assert missing is False


def test_ask_deposits_fall_back_only_when_no_rows_or_fx_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    used, missing = data._ask_deposits_dkk(
        FakeSession([]),  # type: ignore[arg-type]
        user_id=UUID("10000000-0000-4000-8000-000000000001"),
        member_id=UUID("20000000-0000-4000-8000-000000000001"),
        as_of=date(2026, 10, 28),
    )
    assert used is None and missing is False

    monkeypatch.setattr(store, "latest_fx_rate", lambda *_args: None)
    used, missing = data._ask_deposits_dkk(
        FakeSession([(Decimal("100"), "EUR", date(2026, 5, 1), "deposit")]),
        user_id=UUID("10000000-0000-4000-8000-000000000001"),
        member_id=UUID("20000000-0000-4000-8000-000000000001"),
        as_of=date(2026, 10, 28),
    )
    assert used is None and missing is True
