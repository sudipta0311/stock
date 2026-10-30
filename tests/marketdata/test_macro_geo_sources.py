from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import httpx

from wealth.marketdata.sources import macro_geo

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"
_FETCHED_AT = datetime(2026, 10, 30, 12, tzinfo=UTC)


def _fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def test_nse_index_archive_parser_reads_four_supported_indices() -> None:
    rows = macro_geo.parse_nse_index_valuations(
        _fixture("nse_all_indices.csv"),
        source_url=("https://nsearchives.nseindia.com/content/indices/ind_close_all_29102026.csv"),
        as_of_max=date(2026, 10, 30),
    )

    assert [row.index_name for row in rows] == [
        "NIFTY 50",
        "NIFTY 500",
        "NIFTY MIDCAP 150",
        "NIFTY NEXT 50",
    ]
    assert rows[0].pe.as_tuple().digits == (2, 2, 1, 0)
    assert rows[0].as_of == date(2026, 10, 29)
    assert rows[0].close == Decimal("25100")
    assert all(row.source == "NSE" for row in rows)


def test_nse_fetch_uses_archives_csv_url_and_tries_the_latest_available_day() -> None:
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if str(request.url).endswith("ind_close_all_29102026.csv"):
            return httpx.Response(200, content=_fixture("nse_all_indices.csv"), request=request)
        return httpx.Response(404, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        rows = macro_geo.fetch_nse_index_valuations(date(2026, 10, 30), client=client)

    assert len(rows) == 4
    assert requested[0].endswith("ind_close_all_30102026.csv")
    assert requested[1] == (
        "https://nsearchives.nseindia.com/content/indices/ind_close_all_29102026.csv"
    )


def test_treasury_xml_parser_reads_ten_year_and_three_month_yields() -> None:
    rows = macro_geo.parse_treasury_yield_curve(
        _fixture("treasury_yields.xml"),
        source_url="https://home.treasury.gov/synthetic-feed",
        as_of_max=date(2026, 10, 30),
    )

    assert [(row.series, row.as_of, row.value) for row in rows] == [
        ("UST_10Y", date(2026, 10, 29), rows[0].value),
        ("UST_3M", date(2026, 10, 29), rows[1].value),
        ("UST_10Y", date(2026, 10, 28), rows[2].value),
        ("UST_3M", date(2026, 10, 28), rows[3].value),
    ]
    assert [row.value for row in rows] == [
        Decimal("4.25"),
        Decimal("3.90"),
        Decimal("4.20"),
        Decimal("3.88"),
    ]


def test_treasury_fetch_keeps_current_year_when_prior_year_source_fails() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("field_tdr_date_value") == "2025":
            return httpx.Response(503, request=request)
        return httpx.Response(200, content=_fixture("treasury_yields.xml"), request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        batch = macro_geo.fetch_treasury_yields(date(2026, 10, 30), client=client)

    assert len(batch.observations) == 4
    assert batch.errors == {"2025": "HTTPStatusError"}


def test_each_official_rss_or_atom_fixture_is_parsed() -> None:
    for source in ("pib", "rbi", "ustr"):
        rows = macro_geo.parse_rss_atom_feed(
            _fixture(f"{source}_feed.xml"),
            source=source,
            feed_url=f"https://{source}.example.test/feed",
            fetched_at=_FETCHED_AT,
        )
        assert len(rows) == 1
        assert rows[0].source == source
        assert rows[0].title
        assert rows[0].url.startswith("https://")

    rows = macro_geo.parse_federal_register_results(
        json.loads(_fixture("federal_register.json")), fetched_at=_FETCHED_AT
    )
    assert len(rows) == 1
    assert rows[0].source == "federal_register"
    assert rows[0].guid == "SYNTHETIC-2026-0001"


def test_feed_text_is_capped_in_utf8_bytes() -> None:
    content = (
        "<rss><channel><item><guid>synthetic</guid><title>"
        + "é" * 1000
        + "</title><link>https://source.example.test/item</link><description>"
        + "summary " * 300
        + "</description></item></channel></rss>"
    )
    rows = macro_geo.parse_rss_atom_feed(
        content,
        source="pib",
        feed_url="https://source.example.test/feed",
        fetched_at=_FETCHED_AT,
    )

    assert len(rows) == 1
    assert len(rows[0].title.encode("utf-8")) <= 1024
    assert len(rows[0].summary.encode("utf-8")) <= 1024


def test_feed_refresh_continues_after_a_source_failure_and_uses_descriptive_agent() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if str(request.url) == macro_geo.PIB_FEED_URL:
            return httpx.Response(200, content=_fixture("pib_feed.xml"), request=request)
        if str(request.url) == macro_geo.RBI_FEED_URL:
            return httpx.Response(503, request=request)
        if str(request.url) == macro_geo.USTR_FEED_URL:
            return httpx.Response(200, content=_fixture("ustr_feed.xml"), request=request)
        return httpx.Response(
            200,
            json=json.loads(_fixture("federal_register.json")),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        batch = macro_geo.fetch_official_feeds(
            date(2026, 10, 30), client=client, fetched_at=_FETCHED_AT
        )

    assert {item.source for item in batch.items} == {"pib", "federal_register"}
    assert batch.errors["rbi"] == "HTTPStatusError"
    assert batch.errors["ustr"] == "StaleFeed"
    assert all(request.headers["user-agent"] == macro_geo.MACRO_USER_AGENT for request in requests)
    assert all("@" not in request.headers["user-agent"] for request in requests)
