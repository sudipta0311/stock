from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from openpyxl import Workbook
from openpyxl.utils.datetime import WINDOWS_EPOCH, to_excel
from xlwt import Workbook as XlsWorkbook

from wealth.api.broker_imports import (
    can_create_security,
    header_names,
    mapping_suggestions,
    masked_samples,
)
from wealth.api.imports import _looks_like_text_csv
from wealth.core.ids import isin_check_digit_ok
from wealth.importer.brokers import (
    Layout,
    Trade,
    auto_map,
    detect_broker_file,
    detect_layout,
    fifo_lots,
    is_activity_statement_csv,
    load_broker_config,
    load_layouts,
    mapping_default_requirements,
    normalize_header,
    parse_activity_statement_csv,
    parse_cash_flows,
    parse_holdings,
    parse_tradebook,
    read_table,
)
from wealth.importer.brokers.parser import (
    _currency_decimal,
    _date_value,
    header_fingerprint,
    suggested_mapping,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "brokers"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _xlsx(headers: list[str], rows: list[list[object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Synthetic export"])
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def _xls(headers: list[str], rows: list[list[object]]) -> bytes:
    workbook = XlsWorkbook()
    sheet = workbook.add_sheet("Synthetic")
    for column, header in enumerate(headers):
        sheet.write(0, column, header)
    for row_index, row in enumerate(rows, start=1):
        for column, value in enumerate(row):
            sheet.write(row_index, column, value)
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_header_normalization_handles_vendor_units_and_punctuation() -> None:
    cases = {
        "Avg. cost (₹)": "avg cost inr",
        "Qty.": "qty",
        "ISIN Code": "isin code",
        " Average__Price ": "average price",
        "Last.Price": "last price",
        "Trade_Date": "trade date",
        "Current price (INR)": "current price inr",
        "Market Value (DKK)": "market value dkk",
        "Market Value (NOK)": "market value nok",
        "Market Value (€)": "market value eur",
        "Scrip.Name": "scrip name",
        "Order ID [ref]": "order id",
        "Trade No.": "trade no",
        "Exchange (NSE/BSE)": "exchange",
        "Stock   Name": "stock name",
        "Buy price (₹ per share)": "buy price inr",
    }
    assert len(cases) >= 12
    assert {value: normalize_header(value) for value in cases} == cases


def test_each_seeded_layout_matches_csv_and_generated_xlsx() -> None:
    layouts = load_layouts()
    fixture_names = {
        ("zerodha", "holdings"): "zerodha_holdings.csv",
        ("zerodha", "tradebook"): "zerodha_tradebook.csv",
        ("groww", "holdings"): "groww_holdings.csv",
        ("icici_direct", "holdings"): "icici_direct_holdings.csv",
        ("angel_one", "holdings"): "angel_one_holdings.csv",
        ("upstox", "holdings"): "upstox_holdings.csv",
    }
    for expected, filename in fixture_names.items():
        csv_headers, csv_rows = read_table(_fixture(filename), filename)
        csv_layout = detect_layout(csv_headers, layouts)
        assert csv_layout is not None
        assert (csv_layout.broker, csv_layout.kind) == expected
        assert csv_rows

        sheet_headers = list(csv_rows[0])
        sheet_values = [[row.get(header, "") for header in sheet_headers] for row in csv_rows]
        data = _xlsx(sheet_headers, sheet_values)
        xlsx_headers, xlsx_rows = read_table(data, filename.removesuffix(".csv") + ".xlsx")
        xlsx_layout = detect_layout(xlsx_headers, layouts)
        assert xlsx_layout is not None
        assert (xlsx_layout.broker, xlsx_layout.kind) == expected
        assert xlsx_rows


def test_unknown_headers_are_not_guessed_as_a_broker_layout() -> None:
    headers, rows = read_table(_fixture("unknown_broker.csv"), "unknown_broker.csv")
    assert detect_layout(headers, load_layouts()) is None
    assert mapping_suggestions(headers)["quantity"] == "Units"
    samples = masked_samples(headers, rows)
    assert len(samples) == 1
    expected = [
        re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "x", str(rows[0].get(header) or "")))
        for header in headers
    ]
    assert samples[0] == expected
    assert set(character for value in samples[0] for character in value if character.isalpha()) <= {
        "x"
    }
    assert set(character for value in samples[0] for character in value if character.isdigit()) <= {
        "9"
    }


def test_all_broker_choices_have_configured_market_and_currency_defaults() -> None:
    config = load_broker_config()
    expected_indian = {
        "zerodha",
        "hdfc_securities",
        "groww",
        "icici_direct",
        "angel_one",
        "upstox",
        "other_indian_broker",
    }
    expected_us = {"indmoney", "vested", "ibkr", "other"}
    expected_danish = {"nordnet", "saxo", "danske_bank", "other_danish_broker"}

    assert set(config) == expected_indian | expected_us | expected_danish
    assert all(
        (config[broker]["market"], config[broker]["currency"]) == ("IN", "INR")
        for broker in expected_indian
    )
    assert all(
        (config[broker]["market"], config[broker]["currency"]) == ("US", "USD")
        for broker in expected_us
    )
    assert all(
        (config[broker]["market"], config[broker]["currency"]) == ("DK", "DKK")
        for broker in expected_danish
    )


def test_norwegian_aliases_and_trade_date_fallback_are_loaded() -> None:
    broker_rules = yaml.safe_load(
        (Path(__file__).parents[2] / "rules" / "brokers.yaml").read_text(encoding="utf-8")
    )
    assert set(broker_rules["aliases"]["holdings"]["name"]) == {"en", "da", "no", "sv"}
    norwegian = suggested_mapping(["Verdipapir", "Antall", "Gjennomsnittskurs"], kind="holdings")
    assert (norwegian["name"], norwegian["quantity"], norwegian["avg_price"]) == (
        "Verdipapir",
        "Antall",
        "Gjennomsnittskurs",
    )
    holdings = suggested_mapping(["Værdipapir", "Antal", "Kurs"], kind="holdings")
    trades = suggested_mapping(
        ["Handelsdag", "Transaktionstype", "Antal", "Kurs", "ISIN"],
        kind="tradebook",
    )
    assert (holdings["last_price"], holdings.get("price")) == ("Kurs", None)
    assert trades["price"] == "Kurs"
    assert (
        suggested_mapping(
            ["Booking Date", "Trade Type", "Quantity", "Price", "Symbol"],
            kind="tradebook",
        )["trade_date"]
        == "Booking Date"
    )


def test_mapping_defaults_are_asked_only_without_a_row_market_or_currency() -> None:
    assert mapping_default_requirements(
        [{"Symbol": "SYNTH", "Quantity": "2"}],
        {"symbol": "Symbol", "quantity": "Quantity"},
    ) == (True, True)
    assert mapping_default_requirements(
        [{"ISIN": "US0378331005", "Quantity": "2"}],
        {"isin": "ISIN", "quantity": "Quantity"},
    ) == (False, False)
    assert mapping_default_requirements(
        [{"Market": "EU", "Quantity": "2"}],
        {"market": "Market", "quantity": "Quantity"},
    ) == (False, True)


def test_saved_template_is_selected_for_the_same_normalized_headers() -> None:
    headers = ["Ticker", "Units", "Buy price", "Venue", "Security ID", "Average value"]
    template = {
        "name": "Synthetic mapping",
        "kind": "holdings",
        "header_fingerprint": header_fingerprint(headers),
        "header_names": sorted(normalize_header(header) for header in headers),
        "columns": {
            "symbol": "Ticker",
            "quantity": "Units",
            "avg_price": "Buy price",
            "isin": "Security ID",
        },
    }
    layout = detect_layout(headers, load_layouts(), [template])

    assert layout is not None
    assert layout.broker == "Synthetic mapping"
    assert layout.columns == template["columns"]


def test_saved_template_headers_keep_the_pre_currency_unit_fingerprint() -> None:
    headers = ["Avg. cost (₹)", "Current price (INR)"]

    assert header_fingerprint(headers) == (
        "2eb3ab040869e7a969236a85cc3146c7dd7dde6e148f1d57087d42f7aed9f09d"
    )
    assert header_names(headers) == ["avg cost", "current price"]


def test_csv_reader_handles_bom_and_cp1252_without_writing_files() -> None:
    utf8_bom = (
        "Symbol,Quantity,Average price,Company name\nSYNTH,2,125.50,Synthetic share\n"
    ).encode("utf-8-sig")
    cp1252 = (
        "Symbol,Quantity,Average price,Company name\nSYNTH,2,125.50,Synthetic € class\n"
    ).encode("cp1252")

    for data in (utf8_bom, cp1252):
        headers, rows = read_table(data, "synthetic.csv")
        assert headers == ["Symbol", "Quantity", "Average price", "Company name"]
        assert rows[0]["Company name"] in {"Synthetic share", "Synthetic € class"}


def test_detection_matrix_and_table_reader_share_file_checks() -> None:
    utf8 = b"Symbol,Quantity,Name\nSYNTH,2,Synthetic share\n"
    utf8_bom_semicolon = "Symbol;Quantity;Name\nSYNTH;2;Synthetic share\n".encode("utf-8-sig")
    utf16_bom = "Værdipapir\tAntal\tNavn\nSynthetic share\t2\tSynthetic\n".encode("utf-16")
    utf16_no_bom = "Symbol\tQuantity\tName\nSYNTH\t2\tSynthetic\n".encode("utf-16-le")
    cp1252 = "Symbol,Quantity,Name\nSYNTH,2,Synthetic € class\n".encode("cp1252")
    html_xls = (
        b"<html><script>not executed</script><table><tr><th>Symbol</th>"
        b"<th>Quantity</th><th>Name</th></tr><tr><td>SYNTH</td><td>2</td>"
        b"<td>Synthetic share</td></tr></table></html>"
    )
    real_xls = _xls(["Symbol", "Quantity", "Name"], [["SYNTH", 2, "Synthetic share"]])
    xlsx = _xlsx(["Symbol", "Quantity", "Name"], [["SYNTH", 2, "Synthetic share"]])
    cases = [
        (utf8, "synthetic.csv", "text"),
        (utf8_bom_semicolon, "synthetic.csv", "text"),
        (utf16_bom, "synthetic.tsv", "text"),
        (utf16_no_bom, "synthetic.tsv", "text"),
        (cp1252, "synthetic.csv", "text"),
        (html_xls, "synthetic.xls", "html_xls"),
        (real_xls, "synthetic.xls", "xls"),
        (xlsx, "synthetic.xlsx", "xlsx"),
        (b"%PDF-1.4 synthetic statement", "synthetic.pdf", "pdf"),
    ]

    for data, filename, expected_type in cases:
        assert detect_broker_file(data, filename) == expected_type
        assert _looks_like_text_csv(data, filename) is (expected_type == "text")
        if expected_type == "pdf":
            with pytest.raises(ValueError):
                read_table(data, filename)
            continue
        headers, rows = read_table(data, filename)
        assert len(headers) == 3
        assert rows[0].get("Symbol") == "SYNTH" or "Værdipapir" in rows[0]


def test_plain_text_without_table_delimiters_is_not_an_uploadable_table() -> None:
    data = b"not a PDF"
    assert detect_broker_file(data, "wrong.txt") is None
    assert not _looks_like_text_csv(data, "wrong.txt")


@pytest.mark.parametrize(
    ("raw", "currency", "expected"),
    [
        ("1,23,456.78", "INR", "123456.78"),
        ("₹ 1,23,456.78", "INR", "123456.78"),
        ("INR -12.34", "INR", "-12.34"),
        ("(₹1,000.00)", "INR", "-1000.00"),
        ("−₹ 1,234.56", "INR", "-1234.56"),
        ("1.234,56", "DKK", "1234.56"),
        ("1 234,56", "DKK", "1234.56"),
        ("1\u00a0234,56", "DKK", "1234.56"),
        ("1\u202f234,56", "EUR", "1234.56"),
        ("−1.234,56", "EUR", "-1234.56"),
        ("(1.234,56)", "DKK", "-1234.56"),
        ("DKK 1.234,56", "DKK", "1234.56"),
        ("€ 1.234,56", "EUR", "1234.56"),
        ("$1,234.56", "USD", "1234.56"),
        ("1,23", "USD", "1.23"),
        ("1,234", "USD", "1234"),
        ("1543.2367", "INR", "1543.2367"),
        ("1,23,456.7891", "INR", "123456.7891"),
        ("12,345.678", "INR", "12345.678"),
        ("312,4567", "DKK", "312.4567"),
    ],
)
def test_broker_numbers_accept_indian_european_and_symbol_formats(
    raw: str, currency: str, expected: str
) -> None:
    assert _currency_decimal(raw, field="amount", currency=currency) == Decimal(expected)


@pytest.mark.parametrize(
    ("raw", "currency"),
    [
        ("12,34", "INR"),
        ("1,234,56.78", "INR"),
        ("1,234,56.789", "INR"),
        ("1,234,56", "USD"),
        ("1..2", "DKK"),
        ("--1", "EUR"),
        ("not money", "USD"),
    ],
)
def test_broker_numbers_reject_malformed_grouping_and_text(raw: str, currency: str) -> None:
    with pytest.raises(ValueError):
        _currency_decimal(raw, field="amount", currency=currency)


@pytest.mark.parametrize(
    "raw",
    [
        "2024-01-10",
        "10-01-2024",
        "10/01/2024",
        "10-Jan-2024",
        "10.01.2024",
        "10-01-24",
        "2024/01/10",
        "10 Jan 2024",
        "Jan 10, 2024",
    ],
)
def test_broker_dates_accept_supported_text_formats(raw: str) -> None:
    assert _date_value(raw) == date(2024, 1, 10)


def test_broker_dates_accept_excel_serial_values() -> None:
    serial = to_excel(date(2024, 1, 10), WINDOWS_EPOCH)
    assert _date_value(serial) == date(2024, 1, 10)


def _synthetic_isin(prefix: str) -> str:
    for check_digit in range(10):
        candidate = f"{prefix}{'0' * (11 - len(prefix))}{check_digit}"
        if isin_check_digit_ok(candidate):
            return candidate
    raise AssertionError("Could not make a check-digit-valid synthetic ISIN.")


def test_saxo_holdings_clean_symbols_and_read_open_price() -> None:
    headers = ["Symbol", "ISIN", "Name", "Quantity", "Open Price", "Currency"]
    rows = [
        {
            "Symbol": "SYNTH B:xcse",
            "ISIN": _synthetic_isin("DK"),
            "Name": "Synthetic Danish share",
            "Quantity": "2",
            "Open Price": "20.50",
            "Currency": "DKK",
        },
        {
            "Symbol": "SYNTHF:xams",
            "ISIN": _synthetic_isin("IE"),
            "Name": "Synthetic European fund",
            "Quantity": "3",
            "Open Price": "30.25",
            "Currency": "EUR",
        },
    ]
    layout, kind, _mapping = auto_map(
        headers,
        rows,
        broker="saxo",
        market_default="DK",
        currency_default="DKK",
    )
    assert kind == "holdings"
    assert layout is not None
    holdings = parse_holdings(rows, layout)
    assert [(holding.symbol, holding.exchange, holding.avg_price) for holding in holdings] == [
        ("SYNTHB", "XCSE", Decimal("20.50")),
        ("SYNTHF", "XAMS", Decimal("30.25")),
    ]


def test_valid_isin_allows_unusual_symbols_but_missing_isin_keeps_symbol_rule() -> None:
    assert can_create_security(
        {
            "isin": _synthetic_isin("DK"),
            "symbol": "SYNTH B?",
            "market": "DK",
            "currency": "DKK",
        }
    )
    assert not can_create_security(
        {"isin": None, "symbol": "SYNTH B?", "market": "DK", "currency": "DKK"}
    )


def test_zerodha_previous_closing_price_and_indian_fund_units() -> None:
    holdings_headers = [
        "Symbol",
        "ISIN",
        "Sector",
        "Quantity Available",
        "Quantity Discrepant",
        "Quantity Long Term",
        "Quantity Pledged (Margin)",
        "Quantity Pledged (Loan)",
        "Average Price",
        "Previous Closing Price",
        "Unrealized P&L",
        "Unrealized P&L Pct.",
    ]
    holdings_data = io.StringIO(newline="")
    writer = csv.writer(holdings_data)
    writer.writerow(["Synthetic holdings report"])
    writer.writerow(["Generated for testing"])
    writer.writerow(holdings_headers)
    writer.writerow(
        [
            "SYNTH",
            _synthetic_isin("IN"),
            "Equity",
            "10",
            "0",
            "10",
            "0",
            "0",
            "10.00",
            "12.50",
            "25.00",
            "2.50",
        ]
    )
    writer.writerow(
        [
            "SYNTHE",
            _synthetic_isin("INF"),
            "ETF",
            "25",
            "0",
            "25",
            "0",
            "0",
            "20.00",
            "26.00",
            "150.00",
            "7.50",
        ]
    )
    holdings_headers, holdings_rows = read_table(
        holdings_data.getvalue().encode("utf-8"), "synthetic_zerodha.csv"
    )
    holdings_layout, kind, _mapping = auto_map(
        holdings_headers,
        holdings_rows,
        broker="zerodha",
        market_default="IN",
        currency_default="INR",
    )
    assert kind == "holdings"
    assert holdings_layout is not None
    holdings = parse_holdings(holdings_rows, holdings_layout)
    assert [holding.last_price for holding in holdings] == [
        Decimal("12.50"),
        Decimal("26.00"),
    ]
    assert [holding.security_type for holding in holdings] == ["equity", "etf"]


def test_generic_ibkr_holdings_use_the_selected_us_defaults() -> None:
    headers = ["Symbol", "Quantity", "Average Price", "Exchange"]
    rows = [
        {
            "Symbol": "SYNTH",
            "Quantity": "2",
            "Average Price": "20.00",
            "Exchange": "XNAS",
        }
    ]

    assert detect_layout(headers, load_layouts()) is None
    layout, kind, _mapping = auto_map(
        headers,
        rows,
        broker="ibkr",
        market_default="US",
        currency_default="USD",
    )
    assert kind == "holdings"
    assert layout is not None
    holding = parse_holdings(rows, layout)[0]
    assert (holding.market, holding.currency) == ("US", "USD")


def test_currency_unit_mismatch_leaves_market_value_out() -> None:
    headers = ["Symbol", "ISIN", "Quantity", "Currency", "Market Value (DKK)"]
    rows = [
        {
            "Symbol": "SYNTHF",
            "ISIN": _synthetic_isin("IE"),
            "Quantity": "2",
            "Currency": "EUR",
            "Market Value (DKK)": "100.00",
        }
    ]
    layout, _kind, _mapping = auto_map(
        headers,
        rows,
        broker="saxo",
        market_default="DK",
        currency_default="DKK",
    )
    assert layout is not None
    mismatches: dict[str, int] = {}
    holding = parse_holdings(rows, layout, currency_mismatches=mismatches)[0]
    assert holding.currency == "EUR"
    assert holding.market_value is None
    assert mismatches == {"market_value": 1}


def test_nok_header_unit_is_still_checked_against_a_dkk_row() -> None:
    headers = ["Symbol", "Quantity", "Currency", "Market Value (NOK)"]
    rows = [
        {
            "Symbol": "SYNTH",
            "Quantity": "2",
            "Currency": "DKK",
            "Market Value (NOK)": "100.00",
        }
    ]
    layout, _kind, _mapping = auto_map(
        headers,
        rows,
        broker="saxo",
        market_default="DK",
        currency_default="DKK",
    )
    assert layout is not None
    mismatches: dict[str, int] = {}
    holding = parse_holdings(rows, layout, currency_mismatches=mismatches)[0]
    assert holding.market_value is None
    assert mismatches == {"market_value": 1}


def test_kr_header_unit_does_not_override_a_swedish_row_currency() -> None:
    headers = ["Symbol", "Quantity", "Currency", "Marknadsvärde (kr)"]
    rows = [
        {
            "Symbol": "SYNTH",
            "Quantity": "2",
            "Currency": "SEK",
            "Marknadsvärde (kr)": "100.00",
        }
    ]
    layout, kind, _mapping = auto_map(
        headers,
        rows,
        broker="nordnet",
        market_default="DK",
        currency_default="DKK",
    )
    assert kind == "holdings"
    assert layout is not None
    holding = parse_holdings(rows, layout)[0]
    assert holding.currency == "SEK"
    assert holding.market_value == Decimal("100.00")
    assert normalize_header("Marknadsvärde (kr)") == "marknadsvärde"


def test_kr_in_a_currency_cell_still_means_danish_kroner() -> None:
    headers = ["Symbol", "Quantity", "Currency", "Market Value"]
    rows = [{"Symbol": "SYNTH", "Quantity": "2", "Currency": "kr.", "Market Value": "100.00"}]
    layout, _kind, _mapping = auto_map(
        headers,
        rows,
        broker="nordnet",
        market_default="DK",
        currency_default="DKK",
    )
    assert layout is not None
    holding = parse_holdings(rows, layout)[0]
    assert holding.currency == "DKK"
    assert holding.market_value == Decimal("100.00")


def test_danish_headers_auto_map_and_market_follows_each_isin() -> None:
    headers = [
        "Værdipapir",
        "ISIN",
        "Antal",
        "GAK",
        "Kurs",
        "Markedsværdi",
        "Valuta",
    ]
    rows = [
        {
            "Værdipapir": "Synthetic US share",
            "ISIN": "US0378331005",
            "Antal": "2",
            "GAK": "10.50",
            "Kurs": "12.50",
            "Markedsværdi": "25.00",
            "Valuta": "USD",
        },
        {
            "Værdipapir": "Synthetic EU fund",
            "ISIN": _synthetic_isin("IE"),
            "Antal": "3",
            "GAK": "1.234,56",
            "Kurs": "1.300,00",
            "Markedsværdi": "3.900,00",
            "Valuta": "EUR",
        },
    ]
    layout, kind, mapping = auto_map(
        headers,
        rows,
        broker="nordnet",
        market_default="DK",
        currency_default="DKK",
    )

    assert kind == "holdings"
    assert layout is not None
    assert mapping["last_price"] == "Kurs"
    holdings = parse_holdings(rows, layout)
    assert [(holding.market, holding.currency) for holding in holdings] == [
        ("US", "USD"),
        ("EU", "EUR"),
    ]
    assert holdings[0].name == "Synthetic US share"
    assert holdings[1].avg_price == Decimal("1234.56")

    invalid_market_rows = [{"ISIN": rows[0]["ISIN"], "Market": "unknown", "Quantity": "1"}]
    explicit_layout = Layout(
        broker="nordnet",
        kind="holdings",
        fingerprint=frozenset(),
        columns={"isin": "ISIN", "market": "Market", "quantity": "Quantity"},
        market_default="DK",
        currency_default="DKK",
    )
    explicit_market_fallback = parse_holdings(invalid_market_rows, explicit_layout)[0]
    assert (explicit_market_fallback.market, explicit_market_fallback.currency) == (
        "US",
        "USD",
    )


def test_danish_trade_words_skip_other_rows_and_prefer_trade_date() -> None:
    headers = [
        "Værdipapirer",
        "ISIN",
        "Antal",
        "Handelsdag",
        "Bogføringsdag",
        "Transaktionstype",
        "Kurs",
        "Beløb",
        "Kurtage",
        "Valuta",
    ]
    isin = _synthetic_isin("DK")
    rows = [
        {
            "Værdipapirer": "Synthetic Danish fund",
            "ISIN": isin,
            "Antal": "2",
            "Handelsdag": "2024-01-10",
            "Bogføringsdag": "2024-01-12",
            "Transaktionstype": "KØBT",
            "Kurs": "10,00",
            "Beløb": "20,00",
            "Kurtage": "1,00",
            "Valuta": "DKK",
        },
        {
            "Værdipapirer": "Synthetic Danish fund",
            "ISIN": isin,
            "Antal": "1",
            "Handelsdag": "2024-02-10",
            "Bogføringsdag": "2024-02-12",
            "Transaktionstype": "KØBT",
            "Kurs": "11,00",
            "Beløb": "11,00",
            "Kurtage": "1,00",
            "Valuta": "DKK",
        },
        {
            "Værdipapirer": "Synthetic Danish fund",
            "ISIN": isin,
            "Antal": "1",
            "Handelsdag": "2024-03-10",
            "Bogføringsdag": "2024-03-12",
            "Transaktionstype": "SOLGT",
            "Kurs": "12,00",
            "Beløb": "12,00",
            "Kurtage": "1,00",
            "Valuta": "DKK",
        },
        {
            "Værdipapirer": "Synthetic Danish fund",
            "ISIN": isin,
            "Antal": "0",
            "Handelsdag": "2024-04-10",
            "Bogføringsdag": "2024-04-12",
            "Transaktionstype": "UDBYTTE",
            "Kurs": "",
            "Beløb": "2,00",
            "Kurtage": "",
            "Valuta": "DKK",
        },
        {
            "Værdipapirer": "Synthetic Danish fund",
            "ISIN": isin,
            "Antal": "0",
            "Handelsdag": "2024-05-10",
            "Bogføringsdag": "2024-05-12",
            "Transaktionstype": "INDBETALING",
            "Kurs": "",
            "Beløb": "100,00",
            "Kurtage": "",
            "Valuta": "DKK",
        },
    ]
    layout, kind, mapping = auto_map(
        headers,
        rows,
        broker="nordnet",
        market_default="DK",
        currency_default="DKK",
    )
    assert kind == "tradebook"
    assert layout is not None
    assert mapping["trade_date"] == "Handelsdag"
    skipped: dict[str, int] = {}
    trades = parse_tradebook(rows, layout, skipped_rows=skipped)
    assert [trade.trade_type for trade in trades] == ["buy", "buy", "sell"]
    assert [trade.trade_date for trade in trades] == [
        date(2024, 1, 10),
        date(2024, 2, 10),
        date(2024, 3, 10),
    ]
    assert all(trade.market == "DK" and trade.currency == "DKK" for trade in trades)
    assert skipped == {"not_a_trade": 2}


def test_indian_and_us_headers_auto_map_and_cost_value_is_divided_exactly() -> None:
    indian_headers = ["Scrip Name", "ISIN", "Qty", "Avg Cost Price", "LTP"]
    indian_rows = [
        {
            "Scrip Name": "Synthetic Indian share",
            "ISIN": _synthetic_isin("IN"),
            "Qty": "4",
            "Avg Cost Price": "12.50",
            "LTP": "14.00",
        },
        {
            "Scrip Name": "Synthetic Indian fund",
            "ISIN": "",
            "Qty": "2",
            "Avg Cost Price": "10.00",
            "LTP": "11.00",
        },
    ]
    indian_layout, _kind, _mapping = auto_map(
        indian_headers,
        indian_rows,
        broker="hdfc_securities",
        market_default="IN",
        currency_default="INR",
    )
    assert indian_layout is not None
    indian = parse_holdings(indian_rows, indian_layout)
    assert [(holding.market, holding.currency) for holding in indian] == [
        ("IN", "INR"),
        ("IN", "INR"),
    ]

    us_headers = ["Symbol", "Quantity", "Invested Value"]
    us_rows = [{"Symbol": "SYNTH", "Quantity": "3", "Invested Value": "100.00"}]
    us_layout, _kind, _mapping = auto_map(
        us_headers,
        us_rows,
        broker="indmoney",
        market_default="US",
        currency_default="USD",
    )
    assert us_layout is not None
    holding = parse_holdings(us_rows, us_layout)[0]
    assert holding.avg_price == Decimal("100.00") / Decimal("3")
    assert (holding.market, holding.currency) == ("US", "USD")


def test_xlsx_with_a_wrong_sheet_size_tag_is_still_read_in_full() -> None:
    # INDmoney's holdings export declares the sheet as just A1, with the real header on row 5.
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Holdings Report"
    sheet.append(["Synthetic holdings report"])
    sheet.append(["Synthetic Person", "Synthetic account"])
    sheet.append([])
    sheet.append([])
    sheet.append(
        [
            "Symbol",
            "Name",
            "Quantity",
            "($) Cost Price",
            "($) Invested Value",
            "($) Market Price",
            "($) Market Value",
            "($) Unrealized Returns",
        ]
    )
    sheet.append(["SYNA", "Synthetic Alpha Inc", 0.25, 120.5, 30.125, 130.0, 32.5, 2.375])
    sheet.append(["SYNB", "Synthetic Broad Market ETF", 2.0, 50.0, 100.0, 55.0, 110.0, 10.0])
    # A US fund from the look-through rules whose name doesn't say ETF.
    sheet.append(["QQQM", "Synthetic Nasdaq Tracker Trust", 1.0, 200.0, 200.0, 210.0, 210.0, 10.0])
    sheet.append([])
    sheet.append(["Synthetic note: prices are delayed."])
    output = io.BytesIO()
    workbook.save(output)
    source = zipfile.ZipFile(io.BytesIO(output.getvalue()))
    rewritten = io.BytesIO()
    with zipfile.ZipFile(rewritten, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            content = source.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                wrong_size = b'<dimension ref="A1"'
                content, count = re.subn(rb'<dimension ref="[^"]*"', wrong_size, content)
                assert count == 1
            target.writestr(item, content)

    headers, rows = read_table(rewritten.getvalue(), "synthetic_indmoney.xlsx")
    layout, kind, _mapping = auto_map(
        headers, rows, broker="indmoney", market_default="US", currency_default="USD"
    )

    assert headers[0] == "Symbol"
    assert kind == "holdings"
    assert layout is not None
    holdings = parse_holdings(rows, layout)
    assert [
        (holding.symbol, holding.market, holding.currency, holding.security_type)
        for holding in holdings
    ] == [
        ("SYNA", "US", "USD", "equity"),
        ("SYNB", "US", "USD", "etf"),
        ("QQQM", "US", "USD", "etf"),
    ]
    assert holdings[0].quantity == Decimal("0.25")
    assert holdings[0].avg_price == Decimal("120.5")


def test_shares_registered_abroad_keep_an_indian_or_us_brokers_market() -> None:
    headers = ["Symbol", "ISIN", "Quantity", "Average Price"]
    rows = [
        {
            "Symbol": "SYNIE",
            "ISIN": _synthetic_isin("IE"),
            "Quantity": "2",
            "Average Price": "310.25",
        }
    ]
    for broker, market, currency in (("indmoney", "US", "USD"), ("zerodha", "IN", "INR")):
        layout, _kind, _mapping = auto_map(
            headers, rows, broker=broker, market_default=market, currency_default=currency
        )
        assert layout is not None
        holding = parse_holdings(rows, layout)[0]
        assert (holding.market, holding.currency) == (market, currency)

    layout, _kind, _mapping = auto_map(
        headers, rows, broker="nordnet", market_default="DK", currency_default="DKK"
    )
    assert layout is not None
    assert parse_holdings(rows, layout)[0].market == "EU"


def test_indian_average_prices_with_more_than_two_decimals_are_kept_exactly() -> None:
    headers = ["Scrip Name", "ISIN", "Qty", "Avg Cost Price", "LTP"]
    rows = [
        {
            "Scrip Name": "Synthetic Indian share",
            "ISIN": _synthetic_isin("IN"),
            "Qty": "3",
            "Avg Cost Price": "1543.2367",
            "LTP": "1601.05",
        }
    ]
    layout, _kind, _mapping = auto_map(
        headers, rows, broker="zerodha", market_default="IN", currency_default="INR"
    )
    assert layout is not None
    holding = parse_holdings(rows, layout)[0]
    assert holding.avg_price == Decimal("1543.2367")


def test_missing_buying_price_is_kept_and_equal_alias_ties_require_mapping() -> None:
    headers = ["Name", "Quantity", "Last Price"]
    rows = [{"Name": "Synthetic holding", "Quantity": "2", "Last Price": "12.00"}]
    layout, _kind, _mapping = auto_map(
        headers,
        rows,
        broker="other",
        market_default="US",
        currency_default="USD",
    )
    assert layout is not None
    holding = parse_holdings(rows, layout)[0]
    assert holding.avg_price is None

    ambiguous, _kind, suggestions = auto_map(
        ["Symbol", "Quantity", "LTP", "CMP"],
        [{"Symbol": "SYNTH", "Quantity": "2", "LTP": "12", "CMP": "13"}],
        broker="other",
        market_default="US",
        currency_default="USD",
    )
    assert ambiguous is None
    assert suggestions["last_price"] is None


def test_holdings_parse_decimal_prices_and_skips_summary_rows() -> None:
    headers, rows = read_table(_fixture("zerodha_holdings.csv"), "zerodha_holdings.csv")
    layout = detect_layout(headers, load_layouts())
    assert layout is not None
    rows.append({header: "" for header in headers} | {"Symbol": "Total"})
    holdings = parse_holdings(rows, layout)
    assert len(holdings) == 1
    assert holdings[0].quantity == Decimal("20.00")
    assert holdings[0].avg_price == Decimal("250.00")
    assert holdings[0].isin is not None and isin_check_digit_ok(holdings[0].isin)


def test_tradebook_parses_supported_dates_and_indian_grouping() -> None:
    rows = [
        {
            "Trade date": raw_date,
            "Trade type": "Buy",
            "Symbol": "SYNTH",
            "Qty.": "1,00,000",
            "Price": "1,25,000.00",
            "Trade ID": str(index),
            "ISIN": "IN00000000A3",
        }
        for index, raw_date in enumerate(("2024-01-10", "10-01-2024", "10/01/2024", "10-Jan-2024"))
    ]
    from wealth.importer.brokers import Layout

    layout = Layout(
        broker="synthetic",
        kind="tradebook",
        fingerprint=frozenset(),
        columns={
            "trade_date": "Trade date",
            "trade_type": "Trade type",
            "symbol": "Symbol",
            "quantity": "Qty.",
            "price": "Price",
            "trade_id": "Trade ID",
            "isin": "ISIN",
        },
    )
    trades = parse_tradebook(rows, layout)
    assert [trade.trade_date for trade in trades] == [date(2024, 1, 10)] * 4
    assert trades[0].quantity == Decimal("100000")
    assert trades[0].price == Decimal("125000.00")


def test_fifo_consumes_oldest_lots_and_splits_realised_holding_periods() -> None:
    trades = [
        Trade(
            "IN00000000A3",
            "SYNTH",
            date(2024, 1, 10),
            "buy",
            Decimal(10),
            Decimal(100),
            trade_id="A",
        ),
        Trade(
            "IN00000000A3", "SYNTH", date(2024, 6, 1), "buy", Decimal(5), Decimal(120), trade_id="B"
        ),
        Trade(
            "IN00000000A3",
            "SYNTH",
            date(2025, 2, 1),
            "sell",
            Decimal(12),
            Decimal(150),
            trade_id="C",
        ),
    ]
    open_lots, realised, warnings = fifo_lots(trades)
    assert [(lot.quantity, lot.cost_per_unit, lot.acquired_on) for lot in open_lots] == [
        (Decimal(3), Decimal(120), date(2024, 6, 1))
    ]
    assert [(lot.quantity, lot.cost_per_unit, lot.holding_period) for lot in realised] == [
        (Decimal(10), Decimal(100), "LT"),
        (Decimal(2), Decimal(120), "ST"),
    ]
    assert warnings == []


def test_fifo_warns_for_sells_without_buys_and_nets_intraday_trades() -> None:
    trades = [
        Trade(
            "IN00000000A3",
            "SYNTH",
            date(2025, 2, 1),
            "sell",
            Decimal(3),
            Decimal(150),
            trade_id="A",
        ),
        Trade(
            "IN00000000A3", "SYNTH", date(2025, 2, 2), "buy", Decimal(2), Decimal(100), trade_id="B"
        ),
        Trade(
            "IN00000000A3",
            "SYNTH",
            date(2025, 2, 2),
            "sell",
            Decimal(2),
            Decimal(110),
            trade_id="C",
        ),
    ]
    open_lots, realised, warnings = fifo_lots(trades)
    assert open_lots == []
    assert realised == []
    assert warnings == ["intraday_netted", "sell_without_buy"]


def _synthetic_isin(prefix: str) -> str:
    base = f"{prefix}{'0' * (11 - len(prefix))}"
    for digit in range(10):
        candidate = f"{base}{digit}"
        if isin_check_digit_ok(candidate):
            return candidate
    raise AssertionError("A synthetic security code could not be generated.")


def test_ibkr_activity_reads_supported_sections_and_keeps_foreign_currency() -> None:
    ucits_isin = _synthetic_isin("IE")
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(
        [
            [
                "Open Positions",
                "Header",
                "Asset Category",
                "Symbol",
                "Currency",
                "Quantity",
                "Cost Price",
                "Close Price",
                "Value",
                "Listing Exchange",
            ],
            [
                "Open Positions",
                "Data",
                "Stocks",
                "VOO",
                "USD",
                "100",
                "400",
                "500",
                "50000",
                "NYSE",
            ],
            [
                "Open Positions",
                "Data",
                "ETFs",
                "CSPX",
                "GBP",
                "50",
                "550",
                "600",
                "30000",
                "LSEETF",
            ],
            [
                "Open Positions",
                "Data",
                "Forex",
                "EUR.USD",
                "USD",
                "5000",
                "1.1",
                "1.1",
                "5500",
                "IDEALPRO",
            ],
            [
                "Trades",
                "Header",
                "Asset Category",
                "Symbol",
                "Date/Time",
                "Quantity",
                "T. Price",
                "Proceeds",
                "Comm/Fee",
                "Currency",
                "Listing Exchange",
                "Trade ID",
            ],
            [
                "Trades",
                "Data",
                "Stocks",
                "VOO",
                "2024-01-01; 10:00:00",
                "10",
                "400",
                "-4000",
                "-1",
                "USD",
                "NYSE",
                "BUY-1",
            ],
            [
                "Trades",
                "Data",
                "Stocks",
                "VOO",
                "2026-01-02; 10:00:00",
                "-4",
                "500",
                "2000",
                "-2",
                "USD",
                "NYSE",
                "SELL-1",
            ],
            [
                "Financial Instrument Information",
                "Header",
                "Symbol",
                "Security ID",
                "Security ID Type",
                "Description",
                "Listing Exchange",
            ],
            [
                "Financial Instrument Information",
                "Data",
                "CSPX",
                ucits_isin,
                "ISIN",
                "Global Index Fund",
                "LSEETF",
            ],
            ["Cash Report", "Data", "USD", "50000"],
        ]
    )
    statement = output.getvalue()

    assert is_activity_statement_csv(statement)
    activity = parse_activity_statement_csv(statement)

    assert [(row.symbol, row.market, row.currency, row.quantity) for row in activity.positions] == [
        ("VOO", "US", "USD", Decimal("100")),
        ("CSPX", "GB", "GBP", Decimal("50")),
    ]
    assert activity.positions[1].isin == ucits_isin
    assert activity.positions[1].market_value == Decimal("30000")
    assert activity.warnings == ("unknown_section", "unsupported_asset_category")
    assert len(activity.trades) == 2
    open_lots, realised, warnings = fifo_lots(activity.trades)
    assert warnings == []
    assert [(lot.quantity, lot.cost_per_unit, lot.currency) for lot in open_lots] == [
        (Decimal("6"), Decimal("400"), "USD")
    ]
    assert [(lot.quantity, lot.holding_period, lot.commission, lot.gain) for lot in realised] == [
        (Decimal("4"), "LT", Decimal("2"), Decimal("398"))
    ]


def test_danish_layout_mapping_keeps_native_amounts_and_only_explicit_deposits() -> None:
    rows = [
        {
            "Transaction date": "2026-05-01",
            "Activity": "Deposit",
            "Amount": "5.000,25",
            "Currency": "DKK",
            "Reference": "SYNTH-DEP-1",
        },
        {
            "Transaction date": "2026-05-02",
            "Activity": "Buy",
            "Amount": "-1.250,00",
            "Currency": "DKK",
            "Reference": "SYNTH-BUY-1",
        },
        {
            "Transaction date": "2026-05-03",
            "Activity": "Withdrawal",
            "Amount": "-750,50",
            "Currency": "DKK",
            "Reference": "SYNTH-WD-1",
        },
    ]
    layout = Layout(
        broker="nordnet",
        kind="tradebook",
        fingerprint=frozenset(),
        columns={
            "trade_date": "Transaction date",
            "trade_type": "Activity",
            "amount": "Amount",
            "currency": "Currency",
            "trade_id": "Reference",
        },
        market_default="DK",
        currency_default="DKK",
    )

    flows = parse_cash_flows(rows, layout)
    rows_without_reference = [
        {key: value for key, value in row.items() if key != "Reference"} for row in rows[:1]
    ]
    repeat_flows = parse_cash_flows(rows_without_reference, layout)

    assert [(flow.kind, flow.amount, flow.currency) for flow in flows] == [
        ("deposit", Decimal("5000.25"), "DKK"),
        ("withdrawal", Decimal("750.50"), "DKK"),
    ]
    assert (
        repeat_flows[0].transaction_id
        == parse_cash_flows(rows_without_reference, layout)[0].transaction_id
    )
