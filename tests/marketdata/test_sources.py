"""Offline parser tests for public market-data formats."""

from __future__ import annotations

import csv
import io
from contextlib import nullcontext
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import wealth.marketdata.refresh as refresh_module
from wealth.marketdata.refresh import _fetch_fx, _fetch_nse_master
from wealth.marketdata.sources import amfi, ecb, fred, ibja, manual, nse
from wealth.marketdata.sources.fred import ECB_SOURCE, FRED_SOURCE

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _ecb_series(currency: str) -> str:
    rows = csv.DictReader(io.StringIO(_fixture("ecb_multi_eur_cross.csv")))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=("TIME_PERIOD", "OBS_VALUE"))
    writer.writeheader()
    writer.writerows(
        {"TIME_PERIOD": row["TIME_PERIOD"], "OBS_VALUE": row["OBS_VALUE"]}
        for row in rows
        if row["CURRENCY"] == currency
    )
    return output.getvalue()


def test_nse_bhavcopy_parser_reads_closes_and_isins_from_udiff_format() -> None:
    rows = nse.parse_bhavcopy(_fixture("nse_bhavcopy.csv"))

    assert len(rows) == 20
    assert rows[0].symbol == "ABB"
    assert rows[0].isin == "INE117A01022"
    assert rows[0].close == Decimal("6350.00")
    assert rows[0].as_of == date(2026, 10, 26)


def test_nse_bhavcopy_parser_skips_empty_holidays_and_warns_on_bad_rows() -> None:
    assert nse.parse_bhavcopy("\ufeff") == []
    assert nse.parse_bhavcopy("TradDt,BizDt,ISIN,TckrSymb,ClsPric\n") == []
    malformed = "TradDt,TckrSymb,ClsPric\n2026-10-26,ABB,not-a-price\n"

    with pytest.warns(RuntimeWarning, match="row 2 was skipped"):
        assert nse.parse_bhavcopy(malformed) == []


def test_nse_bhavcopy_parser_accepts_legacy_column_names() -> None:
    text = "SYMBOL,SERIES,DATE1,CLOSE_PRICE,ISIN\nABB,EQ,27-Oct-2026,6350.00,INE117A01022\n"

    assert nse.parse_bhavcopy(text) == [
        nse.NseClose("ABB", Decimal("6350.00"), date(2026, 10, 27), "INE117A01022", "EQ")
    ]


def test_nse_bhavcopy_parser_reads_volume_and_traded_value() -> None:
    rows = nse.parse_bhavcopy(
        "TradDt,TckrSymb,ClsPric,TtlTradgVol,TtlTrfVal\n2026-10-27,SYNTH,100.50,2500,251250.00\n"
    )

    assert rows[0].volume == 2500
    assert rows[0].traded_value == Decimal("251250.00")

    turnover_lacs = nse.parse_bhavcopy(
        "DATE1,SYMBOL,CLOSE_PRICE,TURNOVER_LACS\n27-Oct-2026,SYNTH,100.50,12.5\n"
    )
    assert turnover_lacs[0].traded_value == Decimal("1250000.0")


def test_nse_fetch_skips_weekends_and_missing_holiday_archives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nse, "get", lambda _url: pytest.fail("weekend should not fetch"))
    assert nse.fetch_bhavcopy(date(2026, 10, 31)) == []

    calls: list[str] = []

    def missing_archive(url: str) -> httpx.Response:
        calls.append(url)
        request = httpx.Request("GET", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)

    monkeypatch.setattr(nse, "get", missing_archive)
    assert nse.fetch_bhavcopy(date(2026, 10, 27)) == []
    assert len(calls) == 2


def test_nse_equity_list_parser_reads_master_fields_and_etf_type() -> None:
    text = _fixture("nse_equity_list.csv")
    text += "GOLDBEES,Synthetic Gold ETF,EQ,01-JAN-2002,1,1,INF204KB17I5,1\n"
    rows = nse.parse_equity_list(text)

    assert rows[0].symbol == "20MICRONS"
    assert rows[0].name == "20 Microns Limited"
    assert rows[0].isin == "INE144J01027"
    assert rows[0].series == "EQ"
    assert rows[-1].security_type == "etf"


def test_nse_equity_list_parser_warns_and_skips_malformed_rows() -> None:
    text = (
        "SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\n"
        ",Missing name,EQ,\nABB,ABB India,EQ,INE117A01022\n"
    )

    with pytest.warns(RuntimeWarning, match="equity row 2 was skipped"):
        rows = nse.parse_equity_list(text)

    assert [row.symbol for row in rows] == ["ABB"]


def test_nse_etf_list_parser_classifies_proxies_and_sleeves() -> None:
    rows = nse.parse_etf_list(_fixture("nse_etfseclist.csv"))
    by_symbol = {row.symbol: row for row in rows}

    assert len(rows) == 7
    assert by_symbol["NIFTYBEES"].security_type == "etf"
    assert by_symbol["NIFTYBEES"].sleeve == "equity_india"
    assert by_symbol["GOLDBEES"].sleeve == "gold"
    assert by_symbol["LIQUIDBEES"].sleeve == "debt"
    assert by_symbol["SETF10GILT"].sleeve == "debt"
    assert by_symbol["MON100"].sleeve == "equity_global"
    assert by_symbol["MIDSELIETF"].sleeve == "equity_india"


def test_master_keeps_equities_when_the_etf_endpoint_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    equities = nse.parse_equity_list(_fixture("nse_equity_list.csv"))

    def fail_etf_fetch() -> list[nse.NseSecurity]:
        raise httpx.ConnectTimeout(
            "synthetic timeout", request=httpx.Request("GET", "https://nse.test")
        )

    monkeypatch.setattr(nse, "fetch_equity_list", lambda: equities)
    monkeypatch.setattr(nse, "fetch_etf_list", fail_etf_fetch)

    batch = _fetch_nse_master()

    assert batch.securities == equities
    assert batch.etf_count == 0
    assert batch.etf_error == "ConnectTimeout"


def test_master_fetch_combines_nse_equities_and_etfs(monkeypatch: pytest.MonkeyPatch) -> None:
    equities = nse.parse_equity_list(_fixture("nse_equity_list.csv"))
    etfs = nse.parse_etf_list(_fixture("nse_etfseclist.csv"))
    monkeypatch.setattr(nse, "fetch_equity_list", lambda: equities)
    monkeypatch.setattr(nse, "fetch_etf_list", lambda: etfs)

    batch = _fetch_nse_master()

    assert batch.securities == equities + etfs
    assert batch.etf_count == 7
    assert batch.etf_error is None


def test_master_reports_an_empty_etf_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nse, "fetch_equity_list", lambda: [])
    monkeypatch.setattr(nse, "fetch_etf_list", lambda: [])

    batch = _fetch_nse_master()

    assert batch.etf_error == "EmptyETFList"


def test_amfi_parser_reads_three_sections_and_both_isin_columns() -> None:
    rows = amfi.parse_nav_all(_fixture("amfi_nav_excerpt.txt"))

    assert len(rows) == 9
    assert rows[0].scheme_code == "135762"
    assert rows[0].isin_primary == "INF846K01WO1"
    assert rows[0].category is None
    assert rows[3].category == "mf:debt"
    assert rows[3].sleeve == "debt"
    assert rows[5].isin_reinvestment == "INF090I01KT4"
    assert rows[6].category == "mf:debt"
    assert rows[6].as_of == date(2026, 10, 27)


def test_amfi_fetch_uses_portal_before_www_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    response = httpx.Response(200, text=_fixture("amfi_nav_excerpt.txt"))

    def fetch(url: str) -> httpx.Response:
        calls.append(url)
        return response

    monkeypatch.setattr(amfi, "get", fetch)

    assert len(amfi.fetch_nav_all()) == 9
    assert calls == [amfi.AMFI_NAV_URL]
    assert amfi.AMFI_NAV_URL == "https://portal.amfiindia.com/spages/NAVAll.txt"


def test_amfi_fetch_falls_back_to_www_after_portal_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    response = httpx.Response(200, text=_fixture("amfi_nav_excerpt.txt"))

    def fetch(url: str) -> httpx.Response:
        calls.append(url)
        if url == amfi.AMFI_NAV_URL:
            raise httpx.ConnectTimeout("synthetic timeout", request=httpx.Request("GET", url))
        return response

    monkeypatch.setattr(amfi, "get", fetch)

    assert len(amfi.fetch_nav_all()) == 9
    assert calls == [amfi.AMFI_NAV_URL, amfi.AMFI_NAV_FALLBACK_URL]


@pytest.mark.parametrize(
    ("section", "expected_sleeve", "expected_category"),
    [
        ("Debt Scheme - Liquid Fund", "debt", "mf:debt:liquid"),
        ("Debt Scheme - Gilt Fund", "debt", "mf:debt:gilt"),
        ("Debt Scheme - Corporate Bond Fund", "debt", "mf:debt"),
        ("Gold ETF", "gold", "mf:gold"),
        ("Fund of Funds Overseas", "equity_global", "mf:equity_global"),
        ("Equity Scheme - Large Cap Fund", "equity_india", "mf:equity_india"),
        ("Hybrid Scheme - Aggressive Hybrid Fund", "equity_india", "mf:equity_india"),
        ("Arbitrage Fund", "debt", "mf:arbitrage"),
    ],
)
def test_amfi_section_categories_map_to_policy_sleeves(
    section: str, expected_sleeve: str, expected_category: str
) -> None:
    classification = amfi.classify_amfi_section(section)

    assert classification is not None
    assert classification.sleeve == expected_sleeve
    assert classification.category == expected_category
    if "Arbitrage" in section:
        assert classification.tax_note is not None


def test_amfi_parser_warns_on_malformed_nav_rows() -> None:
    text = (
        "Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;"
        "Option;Net Asset Value;Date\n"
        "Open Ended Schemes(Gold ETF)\n"
        "bad;row\n"
        "135762;INF846K01WO1;-;Synthetic Gold Fund;Direct;Growth;bad;27-Oct-2026\n"
    )

    with pytest.warns(RuntimeWarning):
        assert amfi.parse_nav_all(text) == []


def test_fred_parser_reads_ten_observations_and_computes_calendar_lag() -> None:
    rows = fred.parse_dexinus_csv(_fixture("fred_dexinus.csv"), fetched_on=date(2026, 10, 28))

    assert len(rows) == 10
    assert rows[0].pair == "USD/INR"
    assert rows[0].rate == Decimal("94.9400")
    assert rows[0].as_of == date(2026, 10, 23)
    assert rows[0].publication_lag_days == 5


def test_fred_parser_skips_holidays_and_warns_on_malformed_observations() -> None:
    text = "observation_date,DEXINUS\n2026-10-26,.\nnot-a-date,10\n"

    with pytest.warns(RuntimeWarning, match="FRED row 3 was skipped"):
        assert fred.parse_dexinus_csv(text) == []


def test_fx_refresh_falls_back_to_fred_when_ecb_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = fred.parse_dexinus_csv(_fixture("fred_dexinus.csv"), fetched_on=date(2026, 10, 28))

    def fail_ecb(*, fetched_on: date) -> ecb.FxFetchResult:
        raise httpx.ConnectTimeout(
            "synthetic timeout", request=httpx.Request("GET", ecb.ECB_INR_EUR_URL)
        )

    monkeypatch.setattr(ecb, "fetch_ecb_fx_rates", fail_ecb)
    monkeypatch.setattr(fred, "fetch_dexinus", lambda *, fetched_on: expected)

    result = _fetch_fx(date(2026, 10, 28))
    by_pair = {row.pair: row for row in result.observations}

    assert by_pair["USD/INR"] == expected[0]
    assert by_pair["USD/INR"].source == FRED_SOURCE
    assert set(by_pair) == {"USD/INR"}
    assert result.errors["DKK"] == "ConnectTimeout"
    assert "USD" not in result.errors


def test_fx_refresh_prefers_ecb_when_both_sources_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ecb.parse_ecb_multi_fx_rates(
        _fixture("ecb_multi_eur_cross.csv"),
        fetched_on=date(2026, 10, 28),
    )
    monkeypatch.setattr(ecb, "fetch_ecb_fx_rates", lambda **_kwargs: expected)
    monkeypatch.setattr(fred, "fetch_dexinus", lambda **_kwargs: pytest.fail("FRED is fallback"))

    result = _fetch_fx(date(2026, 10, 28))
    by_pair = {row.pair: row for row in result.observations}

    assert result.errors == {}
    assert by_pair["USD/INR"].source == ECB_SOURCE
    assert by_pair["DKK/INR"].source == ECB_SOURCE


def test_ecb_cross_parser_uses_overlapping_dates_and_rounds_reviewed_rate() -> None:
    rows = ecb.parse_ecb_usd_inr(
        _ecb_series("INR"),
        _ecb_series("USD"),
        fetched_on=date(2026, 10, 28),
    )
    reviewed_rate = next(row for row in rows if row.as_of == date(2026, 10, 27))

    assert [row.as_of for row in rows] == [date(2026, 10, 27)]
    assert reviewed_rate.rate == Decimal("95.1242")
    assert reviewed_rate.pair == "USD/INR"
    assert reviewed_rate.source == ECB_SOURCE
    assert reviewed_rate.source_url.splitlines() == [ecb.ECB_INR_EUR_URL, ecb.ECB_USD_EUR_URL]
    assert reviewed_rate.publication_lag_days == 1


def test_ecb_fetch_uses_both_official_series_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    responses = {
        ecb.ECB_INR_EUR_URL: _ecb_series("INR"),
        ecb.ECB_USD_EUR_URL: _ecb_series("USD"),
    }

    def fetch(url: str) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, text=responses[url])

    monkeypatch.setattr(ecb, "get", fetch)

    rows = ecb.fetch_ecb_usd_inr(fetched_on=date(2026, 10, 28))

    assert len(rows) == 1
    assert calls == [ecb.ECB_INR_EUR_URL, ecb.ECB_USD_EUR_URL]


def test_ecb_multi_series_parser_uses_real_csv_for_supported_reference_pairs() -> None:
    result = ecb.parse_ecb_multi_fx_rates(
        _fixture("ecb_multi_eur_cross.csv"),
        fetched_on=date(2026, 10, 28),
    )

    by_pair = {row.pair: row for row in result.observations}
    assert result.errors == {}
    assert by_pair["USD/INR"].rate == Decimal("95.1242")
    assert by_pair["DKK/INR"].rate == Decimal("14.1906")
    assert by_pair["EUR/INR"].rate == Decimal("106.0635")


def test_ecb_rejects_currency_without_series_or_configured_peg() -> None:
    with pytest.raises(ValueError, match="unsupported ECB currency: JPY"):
        ecb.fetch_ecb_fx_rates(("JPY",), fetched_on=date(2026, 10, 28))


def test_ecb_fetch_uses_one_multi_series_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fetch(url: str) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, text=_fixture("ecb_multi_eur_cross.csv"))

    monkeypatch.setattr(ecb, "get", fetch)
    result = ecb.fetch_ecb_fx_rates(("DKK", "USD", "SEK"), fetched_on=date(2026, 10, 28))

    assert calls == [
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        "D.INR+DKK+USD+SEK.EUR.SP00.A?lastNObservations=10&format=csvdata"
    ]
    assert result.errors == {}
    assert {item.pair for item in result.observations} == {
        "EUR/INR",
        "DKK/INR",
        "USD/INR",
        "SEK/INR",
    }


def test_ecb_multi_request_failure_isolated_by_currency(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fetch(url: str) -> httpx.Response:
        calls.append(url)
        if url.endswith("D.INR+DKK+USD+SEK.EUR.SP00.A?lastNObservations=10&format=csvdata"):
            raise httpx.ConnectTimeout(
                "synthetic multi-series outage", request=httpx.Request("GET", url)
            )
        if url == ecb.ECB_INR_EUR_URL:
            return httpx.Response(200, text=_ecb_series("INR"))
        if url == ecb.ECB_EUR_CROSS_URLS["DKK"]:
            return httpx.Response(200, text=_ecb_series("DKK"))
        if url == ecb.ECB_EUR_CROSS_URLS["USD"]:
            return httpx.Response(200, text=_ecb_series("USD"))
        request = httpx.Request("GET", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("synthetic missing series", request=request, response=response)

    monkeypatch.setattr(ecb, "get", fetch)
    result = ecb.fetch_ecb_fx_rates(("DKK", "USD", "SEK"), fetched_on=date(2026, 10, 28))

    assert {item.pair for item in result.observations} == {"EUR/INR", "DKK/INR", "USD/INR"}
    assert result.errors == {"SEK": "HTTPStatusError"}
    assert calls == [
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        "D.INR+DKK+USD+SEK.EUR.SP00.A?lastNObservations=10&format=csvdata",
        ecb.ECB_INR_EUR_URL,
        ecb.ECB_EUR_CROSS_URLS["DKK"],
        ecb.ECB_EUR_CROSS_URLS["USD"],
        ecb.ECB_EUR_CROSS_URLS["SEK"],
    ]


def test_fx_refresh_cli_returns_partial_exit_code_after_currency_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.delenv("OWNER_DATABASE_URL", raising=False)
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite://")
    monkeypatch.setattr(refresh_module, "make_engine", lambda _url: engine)
    monkeypatch.setattr(refresh_module, "system_session", lambda _engine: nullcontext(object()))
    monkeypatch.setattr(
        refresh_module,
        "run_refresh",
        lambda *_args, **_kwargs: {
            "run_id": "test-run",
            "status": "partial",
            "counts": {"fx": {"fx_rates": 8}},
            "errors": {"fx:SEK": "NoObservations"},
        },
    )

    exit_code = refresh_module.main(["--date", "2026-10-28", "--only", "fx"])

    assert exit_code == 2
    assert '"status": "partial"' in capsys.readouterr().out


def test_ibja_parser_reads_latest_pm_24k_rate_and_converts_to_per_gram() -> None:
    observation = ibja.parse_ibja_html(_fixture("ibja_rates.html"), on=date(2026, 10, 28))

    assert observation is not None
    assert observation.as_of == date(2026, 10, 27)
    assert observation.inr_per_gram_24k == Decimal("14143.1")


def test_ibja_parser_returns_none_for_unparseable_rate_page() -> None:
    with pytest.warns(RuntimeWarning, match="no dated 999 gold rates"):
        assert ibja.parse_ibja_html("<html><body>Rate unavailable</body></html>") is None


def test_manual_gold_loader_requires_an_explicit_value_and_as_of() -> None:
    observation = manual.load_manual_gold_price(FIXTURES / "manual_gold_price.yaml")

    assert observation is not None
    assert observation.inr_per_gram_24k == Decimal("8200.00")
    assert observation.as_of == date(2026, 10, 27)
    assert observation.source_url == "rules/manual_prices.yaml"


def test_source_fetch_wrappers_use_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    amfi_response = httpx.Response(200, text=_fixture("amfi_nav_excerpt.txt"))
    fred_response = httpx.Response(200, text=_fixture("fred_dexinus.csv"))
    equity_response = httpx.Response(200, text=_fixture("nse_equity_list.csv"))
    etf_response = httpx.Response(200, text=_fixture("nse_etfseclist.csv"))
    monkeypatch.setattr(amfi, "get", lambda _url: amfi_response)
    monkeypatch.setattr(fred, "get", lambda _url: fred_response)
    monkeypatch.setattr(
        nse,
        "get",
        lambda url: etf_response if url == nse.NSE_ETF_LIST_URL else equity_response,
    )

    assert len(amfi.fetch_nav_all()) == 9
    assert len(fred.fetch_dexinus(fetched_on=date(2026, 10, 28))) == 10
    assert len(nse.fetch_equity_list()) == 14
    assert len(nse.fetch_etf_list()) == 7


@pytest.mark.parametrize(
    ("section", "scheme", "expected_sleeve"),
    [
        ("Equity Scheme - Sectoral/Thematic", "Synthetic US Bluechip Equity Fund", "equity_global"),
        ("Other Scheme - Index Funds", "Synthetic S and P 500 Index Fund", "equity_global"),
        ("Other Scheme - Index Funds", "Synthetic NYSE FANG+ Fund of Fund", "equity_global"),
        (
            "Equity Scheme - Sectoral/Thematic",
            "Synthetic Feeder - U.S. Opportunities",
            "equity_global",
        ),
        ("Other Scheme - Index Funds", "Synthetic S&P BSE Sensex Index Fund", "equity_india"),
        ("Equity Scheme - Sectoral/Thematic", "SBI Magnum Global Fund", "equity_india"),
        ("Equity Scheme - Sectoral/Thematic", "Synthetic Business Cycle Fund", "equity_india"),
        ("Debt Scheme - Short Duration Fund", "Synthetic US Dollar Bond Fund", "debt"),
    ],
)
def test_overseas_funds_in_indian_amfi_sections_count_as_abroad(
    section: str, scheme: str, expected_sleeve: str
) -> None:
    text = (
        "Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;"
        "Option;Net Asset Value;Date\n"
        f"Open Ended Schemes({section})\n"
        f"135762;INF846K01WO1;-;{scheme};Direct;Growth;10.5;27-Oct-2026\n"
    )

    (row,) = amfi.parse_nav_all(text)

    assert row.sleeve == expected_sleeve


def test_daily_nse_refresh_rereads_the_last_week(monkeypatch: pytest.MonkeyPatch) -> None:
    # A late GitHub start lands on the next IST day, before that day's file exists.
    requested: list[date] = []

    def fake_bhavcopy(day: date) -> list[nse.NseClose]:
        requested.append(day)
        return []

    monkeypatch.setattr(refresh_module.nse, "fetch_bhavcopy", fake_bhavcopy)

    assert refresh_module._fetch_recent_bhavcopies(date(2026, 10, 31)) == []
    assert requested == [date(2026, 10, 25) + timedelta(days=offset) for offset in range(7)]
