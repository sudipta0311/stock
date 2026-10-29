from __future__ import annotations

from datetime import date

import pytest

from wealth.marketdata import refresh


def test_overseas_refresh_skips_when_price_key_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("PRICE_API_KEY", raising=False)
    fetchers = refresh._default_fetchers(date(2026, 10, 28), session=object())
    assert fetchers["eu_etfs"]() == []
    assert fetchers["us_stocks"]() == []


def test_us_stock_quote_budget_reserves_all_curated_funds() -> None:
    assert refresh._remaining_abroad_quote_slots(11) == 13
    assert refresh._remaining_abroad_quote_slots(30) == 0


class FakeSession:
    def scalar(self, _statement):
        return "security-id"


def test_curated_london_price_uses_only_the_usd_listing() -> None:
    instrument = {
        "name": "Example fund",
        "isin": "IE00TEST0001",
        "exposure": "global_all_world",
        "tickers": {
            "LSE_GBP": {"symbol": "EXGBX", "currency": "GBP"},
            "LSE_USD": {"symbol": "EXUSD", "currency": "USD"},
            "Xetra": {"symbol": "EXEUR", "currency": "EUR"},
        },
    }
    security_id, provider_symbol = refresh._upsert_curated_security(
        FakeSession(),
        instrument,  # type: ignore[arg-type]
    )
    assert security_id == "security-id"
    assert provider_symbol == "EXUSD.LON"


def test_curated_london_price_skips_when_no_usd_listing_exists() -> None:
    instrument = {
        "name": "Example fund",
        "isin": "IE00TEST0001",
        "exposure": "global_all_world",
        "tickers": {
            "LSE_GBP": {"symbol": "EXGBX", "currency": "GBP"},
            "Xetra": {"symbol": "EXEUR", "currency": "EUR"},
        },
    }
    with pytest.raises(ValueError, match="USD-quoted London listing"):
        refresh._upsert_curated_security(FakeSession(), instrument)  # type: ignore[arg-type]
