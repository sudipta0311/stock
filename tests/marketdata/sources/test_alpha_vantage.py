from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from wealth.marketdata.sources import alpha_vantage


def test_parse_daily_series_picks_newest_valid_decimal_close() -> None:
    close = alpha_vantage.parse_daily_series(
        {
            "Meta Data": {"2. Symbol": "WLDX"},
            "Time Series (Daily)": {
                "2026-10-27": {"4. close": "101.2400"},
                "2026-10-28": {"4. close": "102.50"},
            },
        },
        "WLDX",
    )
    assert close.symbol == "WLDX"
    assert close.as_of == date(2026, 10, 28)
    assert close.close == Decimal("102.50")


def test_fetch_daily_uses_fixture_and_source_url_has_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def json(self) -> dict[str, object]:
            return {"Time Series (Daily)": {"2026-10-28": {"4. close": "12.50"}}}

    requested: list[str] = []

    def fake_get(url: str) -> FakeResponse:
        requested.append(url)
        return FakeResponse()

    monkeypatch.setattr(alpha_vantage.http, "get", fake_get)
    quote = alpha_vantage.fetch_daily("WLDX.LON", "fixture-key")
    assert quote.close == Decimal("12.50")
    assert "fixture-key" in requested[0]
    assert "fixture-key" not in alpha_vantage.SOURCE_URL


def test_fetch_daily_rejects_missing_key_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpha_vantage.http,
        "get",
        lambda _url: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )
    with pytest.raises(ValueError, match="not configured"):
        alpha_vantage.fetch_daily("WLDX.LON", "")
