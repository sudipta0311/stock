from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from openpyxl import Workbook

from wealth.marketdata.sources import edgar, nse_fundamentals, universes
from wealth.marketdata.sources.nse_fundamentals import NseFiling

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"


def test_edgar_parser_maps_fallback_tags_and_uses_filing_date_for_known_at() -> None:
    payload = json.loads((FIXTURES / "sec_companyfacts_excerpt.json").read_text(encoding="utf-8"))

    observations = edgar.parse_company_facts(payload, as_of=date(2026, 12, 31))
    by_metric = {row.metric: row for row in observations}

    assert by_metric["revenue"].value == Decimal("110")
    assert by_metric["net_income"].value == Decimal("20")
    assert by_metric["total_equity"].value == Decimal("250")
    assert by_metric["total_debt"].value == Decimal("100")
    assert by_metric["eps_diluted"].unit == "USD/shares"
    restatement = [row for row in observations if row.metric == "revenue" and row.value == 110]
    assert restatement[0].filed_at == datetime(2026, 2, 10, tzinfo=UTC)
    assert restatement[0].known_at == restatement[0].filed_at


def test_edgar_requires_a_contact_user_agent_before_making_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(edgar.EdgarConfigurationError, match="SEC_USER_AGENT"):
        edgar.SecEdgarClient()


def test_company_ticker_lookup_normalizes_share_class_punctuation() -> None:
    companies = edgar.parse_company_tickers(
        {"0": {"ticker": "BRK-B", "cik_str": 1067983, "title": "Example Holdings"}}
    )

    assert edgar.ticker_lookup(companies, "BRK.B") == edgar.EdgarCompany(
        "BRK-B", "0001067983", "Example Holdings"
    )


def test_nse_xbrl_parser_extracts_quarterly_and_instant_facts() -> None:
    filing = NseFiling(
        symbol="SAMPLE",
        period_end=date(2025, 12, 31),
        filed_at=datetime(2026, 1, 30, 12, tzinfo=UTC),
        xbrl_url="https://nsearchives.nseindia.com/corporate/xbrl/sample.xml",
        consolidated=True,
        submission_type="Original",
    )

    observations = nse_fundamentals.parse_xbrl_filing(
        (FIXTURES / "nse_xbrl_excerpt.xml").read_bytes(), filing, as_of=date(2026, 2, 1)
    )
    by_metric = {row.metric: row for row in observations}

    assert by_metric["revenue"].value == Decimal("1000")
    assert by_metric["net_income"].value == Decimal("80")
    assert by_metric["operating_income"].value == Decimal("135")
    assert by_metric["eps_diluted"].value == Decimal("2.50")
    assert by_metric["total_assets"].period_type == "Q"
    assert by_metric["total_equity"].value == Decimal("2800")
    assert by_metric["total_debt"].value == Decimal("400")
    assert all(row.known_at == filing.filed_at for row in observations)
    assert all(row.source_url == filing.xbrl_url for row in observations)


def test_nse_filing_index_prefers_consolidated_rows_for_a_submission_day() -> None:
    rows = [
        {
            "symbol": "SAMPLE",
            "qe_Date": "30-JUN-2026",
            "broadcast_Date": "30-Jul-2026 18:00:00",
            "revised_Date": None,
            "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/standalone.xml",
            "consolidated": "Standalone",
            "type_Sub": "Original",
        },
        {
            "symbol": "SAMPLE",
            "qe_Date": "30-JUN-2026",
            "broadcast_Date": "30-Jul-2026 18:10:00",
            "revised_Date": None,
            "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/consolidated.xml",
            "consolidated": "Consolidated",
            "type_Sub": "Original",
        },
    ]

    filings = nse_fundamentals.parse_filing_rows(rows, symbol="SAMPLE")

    assert len(filings) == 1
    assert filings[0].consolidated
    assert filings[0].xbrl_url.endswith("consolidated.xml")


def test_universe_parsers_keep_issuer_members_and_survivorship_label() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Fund Name:", "Example S&P ETF"])
    sheet.append(["Ticker Symbol:", "SPY"])
    sheet.append(["Holdings:", "As of 27-Oct-2026"])
    sheet.append([])
    sheet.append(["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector"])
    sheet.append(["Example One Corp", "EXA", "123456789", "ABC1234", 1.0, "Industrials"])
    sheet.append(["Cash Position", "USD", "CASH", "", 0.0, "Unassigned"])
    sheet.append(["Other Payable", "-", "-", "", 0.0, "-"])
    sheet.append(["Placeholder Line", "2602335D", "", "", 0.0, "Industrials"])
    import io

    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()

    sp500 = universes.parse_sp500_holdings(stream.getvalue())
    nifty500 = universes.parse_nifty500_csv(
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        "Example One Limited,Services,EXA,EQ,\n"
        "Example Two Limited,Manufacturing,EXB,EQ,\n"
        "Example Two Limited,Manufacturing,EXB,EQ,\n"
        "Example Three Limited,N/A,EXC,EQ,\n"
    )

    assert [item.symbol for item in sp500.securities] == ["EXA"]
    assert sp500.as_of == date(2026, 10, 27)
    assert sp500.securities[0].sector == "Industrials"
    assert [item.symbol for item in nifty500] == ["EXA", "EXB", "EXC"]
    assert nifty500[0].sector == "Services"
    assert nifty500[0].industry == "Services"
    assert nifty500[2].sector == nifty500[2].industry == "Unclassified"
    assert universes.SURVIVORSHIP_LABEL == "current constituents only (survivorship bias)"
    assert "ssga.com" in universes.SP500_HOLDINGS_URL
    assert "archives.nseindia.com" in universes.NIFTY500_ARCHIVE_URL


def test_spy_files_give_each_holding_a_close_from_weight_and_net_assets() -> None:
    import io

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Fund Name:", "Example S&P ETF"])
    sheet.append([])
    sheet.append(["Date", "NAV", "Shares Outstanding", "Total Net Assets"])
    sheet.append(["27-Oct-2026", "500.0", "2000000.0", "1000000000.00"])
    sheet.append(["26-Oct-2026", "499.0", "2000000.0", "998000000.00"])
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()

    assets = universes.parse_spy_total_net_assets(stream.getvalue())
    holdings = universes.Sp500Holdings(
        securities=[
            universes.UniverseSecurity(
                "EXA", "Example One", weight_pct=Decimal("2.5"), shares_held=Decimal("100000")
            ),
            universes.UniverseSecurity("EXB", "Example Two", weight_pct=Decimal("1.0")),
        ],
        as_of=date(2026, 10, 27),
    )
    stale = universes.Sp500Holdings(holdings.securities, as_of=date(2026, 10, 28))

    # 2.5% of 1,000,000,000 is 25,000,000 across 100,000 shares.
    assert assets[date(2026, 10, 27)] == Decimal("1000000000.00")
    assert universes.implied_closes(holdings, assets) == {"EXA": Decimal("250.000000")}
    assert universes.implied_closes(stale, assets) == {}


def test_sp500_sector_table_maps_tickers_to_gics_sectors() -> None:
    html = """
    <table class="wikitable"><tr><th>Other</th></tr><tr><td>ignored</td></tr></table>
    <table id="constituents"><tbody>
    <tr><th><a>Symbol</a></th><th>Security</th><th><a>GICS</a> Sector</th>
    <th>GICS Sub-Industry</th><th>Founded</th></tr>
    <tr><td><a>EXA</a></td><td>Example One</td><td>Industrials</td>
    <td>Industrial Machinery</td><td>1900</td></tr>
    <tr><td>EXB-B</td><td>Example Two</td><td>Health Care</td>
    <td>Life Sciences</td><td>1950</td></tr>
    <tr><td>-</td><td>Bad row</td><td>Energy</td><td>Oil</td><td>2000</td></tr>
    </tbody></table>
    """

    sectors = universes.parse_sp500_sectors(html)

    assert sectors == {
        "EXA": universes.Sp500Sector("Industrials", "Industrial Machinery"),
        "EXB.B": universes.Sp500Sector("Health Care", "Life Sciences"),
    }


def test_edgar_turns_year_to_date_cash_flow_and_annual_eps_into_four_quarters() -> None:
    def fact(start: str, end: str, value: str, form: str, filed: str) -> dict[str, str]:
        return {"start": start, "end": end, "val": value, "form": form, "filed": filed}

    # Synthetic issuer: 10-Q cash flows are year-to-date; the 10-K reports only the full year.
    cash_flow = [
        fact("2025-01-01", "2025-03-31", "10", "10-Q", "2025-05-01"),
        fact("2025-01-01", "2025-06-30", "25", "10-Q", "2025-08-01"),
        fact("2025-01-01", "2025-09-30", "45", "10-Q", "2025-11-01"),
        fact("2025-01-01", "2025-12-31", "70", "10-K", "2026-02-15"),
    ]
    eps = [
        fact("2025-01-01", "2025-03-31", "1.00", "10-Q", "2025-05-01"),
        fact("2025-04-01", "2025-06-30", "1.10", "10-Q", "2025-08-01"),
        fact("2025-07-01", "2025-09-30", "1.20", "10-Q", "2025-11-01"),
        fact("2025-01-01", "2025-12-31", "4.60", "10-K", "2026-02-15"),
    ]
    payload = {
        "cik": 1,
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": cash_flow}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
            }
        },
    }

    observations = edgar.parse_company_facts(payload, as_of=date(2026, 3, 31))

    def quarters(metric: str) -> dict[date, tuple[Decimal, date]]:
        return {
            row.period_end: (row.value, row.known_at.date())
            for row in observations
            if row.metric == metric and row.period_type == "Q"
        }

    assert quarters("operating_cash_flow") == {
        date(2025, 3, 31): (Decimal("10"), date(2025, 5, 1)),
        date(2025, 6, 30): (Decimal("15"), date(2025, 8, 1)),
        date(2025, 9, 30): (Decimal("20"), date(2025, 11, 1)),
        date(2025, 12, 31): (Decimal("25"), date(2026, 2, 15)),
    }
    assert quarters("eps_diluted")[date(2025, 12, 31)] == (Decimal("1.30"), date(2026, 2, 15))
