from datetime import date, timedelta
from decimal import Decimal

import pytest

from wealth.marketdata import store


def test_fresh_fx_rate_to_inr_accepts_ten_day_rates_and_rejects_older_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = date(2036, 6, 15)
    monkeypatch.setattr(
        store,
        "latest_fx_rate",
        lambda _session, pair, _on: (
            (
                Decimal("12.50"),
                today - timedelta(days=10),
                "synthetic-test",
            )
            if pair == "DKK/INR"
            else None
        ),
    )

    assert store.fresh_fx_rate_to_inr(object(), "DKK", today) == Decimal("12.50")
    assert store.fresh_fx_rate_to_inr(object(), "INR", today) == Decimal(1)

    monkeypatch.setattr(
        store,
        "latest_fx_rate",
        lambda _session, _pair, _on: (
            Decimal("12.50"),
            today - timedelta(days=11),
            "synthetic-test",
        ),
    )
    assert store.fresh_fx_rate_to_inr(object(), "DKK", today) is None
