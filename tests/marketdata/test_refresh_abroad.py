from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from wealth.db.models import MarketdataRun
from wealth.marketdata import refresh, store
from wealth.marketdata.sources import alpha_vantage
from wealth.marketdata.sources.alpha_vantage import DailyClose
from wealth.policy.denmark_picks import load_eu_etfs


def test_overseas_refresh_skips_when_price_key_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICE_API_KEY", raising=False)
    fetchers = refresh._default_fetchers(date(2026, 10, 28), session=object())
    assert fetchers["eu_etfs"]() == []
    assert fetchers["us_stocks"]() == []


def test_us_stock_quote_budget_reserves_all_curated_funds() -> None:
    assert refresh._remaining_abroad_quote_slots(11) == 13
    assert refresh._remaining_abroad_quote_slots(30) == 0


def test_curated_listing_prefers_london_usd() -> None:
    listing = refresh._choose_curated_listing(
        {
            "tickers": {
                "Xetra": {"symbol": "SYNTH-EUR", "currency": "EUR"},
                "LSE_GBP": {"symbol": "SYNTH-GBP", "currency": "GBP"},
                "LSE_USD": {"symbol": "SYNTH-USD", "currency": "USD"},
            }
        }
    )

    assert listing == refresh.CuratedListing("LSE_USD", "SYNTH-USD", "USD", "SYNTH-USD.LON")


def test_curated_listing_falls_back_to_xetra_eur_with_exchange_suffix() -> None:
    listing = refresh._choose_curated_listing(
        {"tickers": {"Xetra": {"symbol": "SYNTH-EUR", "currency": "EUR"}}}
    )

    assert listing == refresh.CuratedListing("Xetra", "SYNTH-EUR", "EUR", "SYNTH-EUR.DEX")


def test_curated_listing_skips_gbp_and_euronext_only_funds() -> None:
    assert (
        refresh._choose_curated_listing(
            {
                "tickers": {
                    "LSE_GBP": {"symbol": "SYNTH-GBP", "currency": "GBP"},
                    "Euronext": {"symbol": "SYNTH-EUR", "currency": "EUR"},
                }
            }
        )
        is None
    )


def test_all_curated_funds_have_a_supported_av_listing() -> None:
    selections = [
        refresh._choose_curated_listing(instrument) for instrument in load_eu_etfs()["instruments"]
    ]

    assert len(selections) == 11
    assert all(selection is not None for selection in selections)
    assert sum(selection.exchange == "LSE_USD" for selection in selections if selection) == 8
    assert sum(selection.exchange == "Xetra" for selection in selections if selection) == 3


def test_alpha_vantage_pacing_and_one_rate_limit_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []
    waits: list[float] = []

    def fetch_daily_series(symbol: str, _api_key: str, **_kwargs: object) -> tuple[DailyClose, ...]:
        requests.append(symbol)
        if len(requests) == 1:
            raise alpha_vantage.RateLimitResponse("synthetic rate limit")
        return (DailyClose(symbol, date(2026, 10, 28), Decimal("12.50")),)

    monkeypatch.setattr(alpha_vantage, "fetch_daily_series", fetch_daily_series)

    closes = refresh._fetch_alpha_vantage_daily(
        "SYNTH.LON",
        "fixture-key",
        sleeper=waits.append,
    )

    assert closes[-1].close == Decimal("12.50")
    assert requests == ["SYNTH.LON", "SYNTH.LON"]
    assert waits == [refresh._ALPHA_VANTAGE_REQUEST_SPACING_SECONDS] * 2


def test_abroad_etf_retries_once_then_counts_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wealth.policy import denmark_picks

    monkeypatch.setenv("PRICE_API_KEY", "fixture-key")
    monkeypatch.setattr(
        denmark_picks,
        "load_eu_etfs",
        lambda: {
            "instruments": [
                {
                    "name": "Synthetic fund",
                    "isin": "IE00TEST0001",
                    "exposure": "global_all_world",
                    "tickers": {"LSE_USD": {"symbol": "SYNTH", "currency": "USD"}},
                }
            ]
        },
    )
    requests: list[str] = []

    def rate_limited(symbol: str, _api_key: str, **_kwargs: object) -> tuple[DailyClose, ...]:
        requests.append(symbol)
        raise alpha_vantage.RateLimitResponse("synthetic rate limit")

    monkeypatch.setattr(alpha_vantage, "fetch_daily_series", rate_limited)
    result = refresh._fetch_abroad_etfs(date(2026, 10, 28), sleeper=lambda _seconds: None)

    assert isinstance(result, refresh.AbroadBatch)
    assert result.requested == 1
    assert result.skipped["rate_limited"] == 1
    assert result.records == []
    assert requests == ["SYNTH.LON", "SYNTH.LON"]


def test_retries_share_the_daily_quote_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []

    def fetch_daily_series(symbol: str, _api_key: str, **_kwargs: object) -> tuple[DailyClose, ...]:
        requests.append(symbol)
        return (DailyClose(symbol, date(2026, 10, 28), Decimal("12.50")),)

    monkeypatch.setattr(alpha_vantage, "fetch_daily_series", fetch_daily_series)
    budget = refresh.AbroadRequestBudget(used=22)
    securities = [(f"security-{index}", f"SYNTH{index}") for index in range(5)]

    result = refresh._fetch_abroad_us_stocks(
        securities,
        date(2026, 10, 28),
        curated_count=11,
        api_key="fixture-key",
        sleeper=lambda _seconds: None,
        request_budget=budget,
    )

    assert isinstance(result, refresh.AbroadBatch)
    assert result.requested == 2
    assert len(result.records) == 2
    assert budget.used == refresh._MAX_ABROAD_QUOTES_PER_DAY
    assert requests == ["SYNTH0", "SYNTH1"]


def test_held_us_stock_refresh_stores_the_recorded_compact_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "Time Series (Daily)": {
            "2026-10-26": {"4. close": "11.50"},
            "2026-10-27": {"4. close": "12.00"},
            "2026-10-28": {"4. close": "12.50"},
            "2026-10-29": {"4. close": "13.00"},
        }
    }

    class FakeResponse:
        def json(self) -> dict[str, object]:
            return payload

    requests: list[str] = []

    def fake_get(url: str, **_kwargs: object) -> FakeResponse:
        requests.append(url)
        return FakeResponse()

    monkeypatch.setattr(alpha_vantage.http, "get", fake_get)
    budget = refresh.AbroadRequestBudget()

    result = refresh._fetch_abroad_us_stocks(
        [("synthetic-security", "SYNTH")],
        date(2026, 10, 28),
        curated_count=0,
        api_key="fixture-key",
        sleeper=lambda _seconds: None,
        request_budget=budget,
    )

    assert isinstance(result, refresh.AbroadBatch)
    assert [(record.close.as_of, record.close.close) for record in result.records] == [
        (date(2026, 10, 26), Decimal("11.50")),
        (date(2026, 10, 27), Decimal("12.00")),
        (date(2026, 10, 28), Decimal("12.50")),
    ]
    assert len(requests) == 1
    assert budget.used == 1


class FakeSession:
    def __init__(self) -> None:
        self.rows: list[MarketdataRun] = []

    def add(self, row: MarketdataRun) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        for row in self.rows:
            if row.id is None:
                row.id = uuid4()


def test_partial_overseas_result_counts_skips_without_exposing_api_key_or_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wealth.policy import denmark_picks

    api_key = "synthetic-api-key"
    monkeypatch.setenv("PRICE_API_KEY", api_key)
    instruments = [
        {
            "name": "Synthetic supported fund",
            "isin": "IE00TEST0001",
            "exposure": "global_all_world",
            "tickers": {"LSE_USD": {"symbol": "GOOD", "currency": "USD"}},
        },
        {
            "name": "Synthetic failing fund",
            "isin": "IE00TEST0002",
            "exposure": "global_all_world",
            "tickers": {"LSE_USD": {"symbol": "BAD", "currency": "USD"}},
        },
    ]
    monkeypatch.setattr(denmark_picks, "load_eu_etfs", lambda: {"instruments": instruments})

    def fetch_daily_series(symbol: str, key: str, **_kwargs: object) -> tuple[DailyClose, ...]:
        if symbol.startswith("BAD"):
            raise RuntimeError(f"transport failure {key} {alpha_vantage.API_URL}")
        return (DailyClose(symbol, date(2026, 10, 28), Decimal("12.50")),)

    monkeypatch.setattr(alpha_vantage, "fetch_daily_series", fetch_daily_series)
    monkeypatch.setattr(refresh, "_upsert_curated_security", lambda *_args: ("synthetic-id", ""))
    monkeypatch.setattr(store, "upsert_eod_price", lambda *_args, **_kwargs: None)
    for name in (
        "prune_fundamentals",
        "prune_feed_items",
        "prune_index_valuations",
        "prune_macro_series",
    ):
        monkeypatch.setattr(store, name, lambda *_args, **_kwargs: 0)
    session = FakeSession()
    monkeypatch.setattr(refresh, "system_session", lambda _engine: nullcontext(session))

    result = refresh._run_refresh_on_engine(
        object(),
        date(2026, 10, 28),
        only=("eu_etfs",),
        fetchers={
            "eu_etfs": lambda: refresh._fetch_abroad_etfs(
                date(2026, 10, 28),
                sleeper=lambda _seconds: None,
            )
        },
    )
    serialized = json.dumps(result)

    assert result["status"] == "partial"
    assert result["counts"]["eu_etfs"] == {
        "prices_eod": 1,
        "requested": 2,
        "skipped": {
            "rate_limited": 0,
            "no_supported_listing": 0,
            "unknown_symbol": 0,
            "no_series": 0,
            "http_error": 1,
        },
    }
    assert result["errors"] == {"eu_etfs:http_error": "1 skipped"}
    assert api_key not in serialized
    assert alpha_vantage.API_URL not in serialized
