from __future__ import annotations

import csv
import io
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from openpyxl import Workbook

from wealth.api.broker_imports import mapping_suggestions, masked_samples
from wealth.core.ids import isin_check_digit_ok
from wealth.importer.brokers import (
    Layout,
    Trade,
    detect_layout,
    fifo_lots,
    is_activity_statement_csv,
    load_layouts,
    normalize_header,
    parse_activity_statement_csv,
    parse_cash_flows,
    parse_holdings,
    parse_tradebook,
    read_table,
)
from wealth.importer.brokers.parser import header_fingerprint

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


def test_header_normalization_handles_vendor_units_and_punctuation() -> None:
    cases = {
        "Avg. cost (₹)": "avg cost",
        "Qty.": "qty",
        "ISIN Code": "isin code",
        " Average__Price ": "average price",
        "Last.Price": "last price",
        "Trade_Date": "trade date",
        "Current price (INR)": "current price",
        "Scrip.Name": "scrip name",
        "Order ID [ref]": "order id",
        "Trade No.": "trade no",
        "Exchange (NSE/BSE)": "exchange",
        "Stock   Name": "stock name",
        "Buy price (₹ per share)": "buy price",
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
    base = f"{prefix}{'0' * 9}"
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
