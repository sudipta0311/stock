from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

import wealth.api.broker_imports as broker_imports


def test_indian_symbol_with_ampersand_is_preserved_for_a_valid_isin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security_id = UUID("00000000-0000-0000-0000-000000000068")
    expected_security = SimpleNamespace(id=security_id, symbol="M&M")

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0
            self.upsert_parameters: dict[str, Any] | None = None

        def scalar(
            self, _statement: object, parameters: dict[str, Any] | None = None
        ) -> object | None:
            self.calls += 1
            if parameters is not None:
                self.upsert_parameters = parameters
                return security_id
            return None if self.calls == 1 else expected_security

    session = FakeSession()
    item = SimpleNamespace(
        match_security_id=None,
        payload={
            "isin": "IN0000000000",
            "market": "IN",
            "currency": "INR",
            "symbol": "M&M",
            "name": "Synthetic Company",
            "security_type": "equity",
        },
        user_override={},
    )
    monkeypatch.setattr(broker_imports, "isin_check_digit_ok", lambda _isin: True)

    result = broker_imports._security_for_confirmation(
        session,
        UUID("00000000-0000-0000-0000-000000000001"),
        item,
    )

    assert result is expected_security
    assert session.upsert_parameters is not None
    assert session.upsert_parameters["symbol"] == "M&M"
