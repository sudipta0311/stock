"""Synthetic, network-free fixtures for public basket source parsers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import BytesIO
from typing import Any

from openpyxl import Workbook

from wealth.core.ids import isin_check_digit_ok
from wealth.marketdata.sources import fund_baskets
from wealth.marketdata.sources.universes import (
    Sp500Holdings,
    UniverseSecurity,
    parse_sp500_holdings,
)


def _isin(serial: int) -> str:
    prefix = f"IN{serial:09d}"
    return next(value for digit in "0123456789" if isin_check_digit_ok(value := f"{prefix}{digit}"))


def _foreign_isin(serial: int) -> str:
    prefix = f"US{serial:09d}"
    return next(value for digit in "0123456789" if isin_check_digit_ok(value := f"{prefix}{digit}"))


def _ssga_workbook(rows: list[list[Any]], ticker: str = "spym") -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "holdings"
    worksheet.append(["Fund Name:", "Synthetic fund"])
    worksheet.append(["ISIN:", "IE0000000000"])
    worksheet.append(["Ticker Symbol:", ticker])
    worksheet.append(["Holdings As Of:", "03-Nov-2026"])
    worksheet.append([])
    worksheet.append(
        [
            "ISIN",
            "SEDOL",
            "Security Name",
            "Currency",
            "Number of Shares",
            "Percent of Fund",
            "Trade Country Name",
            "Local Price",
            "Sector Classification",
            "Industry Classification",
        ]
    )
    for row in rows:
        worksheet.append(row)
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def _ssga_row(
    isin: str,
    name: str,
    weight: str,
    country: str = "India",
    sedol: str = "",
) -> list[Any]:
    return [isin, sedol, name, "INR", 1, weight, country, 1, "Industry", "Sector"]


def _spy_workbook() -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["Holdings:", "03-Nov-2026"])
    worksheet.append(["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Industry"])
    worksheet.append(
        ["Synthetic Global Company", "AAPL", "037833100", "2046251", "7", "Technology", "Hardware"]
    )
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def test_ssga_parser_skips_cash_and_invalid_rows_and_sums_duplicate_isins() -> None:
    first = _isin(1)
    second = _isin(2)
    valid_but_corrupted = _isin(99)
    invalid = valid_but_corrupted[:-1] + ("0" if valid_but_corrupted[-1] != "0" else "1")
    content = _ssga_workbook(
        [
            _ssga_row(first, "Synthetic Alpha", "10.25"),
            _ssga_row(first, "Synthetic Alpha duplicate", "2.00"),
            _ssga_row(second, "Synthetic Beta", "0"),
            _ssga_row("Unassigned", "Cash", "4.00"),
            _ssga_row(invalid, "Invalid check digit", "3.00"),
        ]
    )

    parsed = fund_baskets.parse_ssga_emea_holdings(content)

    assert parsed.as_of == date(2026, 11, 3)
    assert len(parsed.weights) == 1
    assert parsed.weights[0].isin == first
    assert parsed.weights[0].weight_pct == Decimal("12.25")


def test_ssga_parser_keeps_sedol_and_spy_parser_keeps_cusip_and_sedol() -> None:
    parsed_ssga = fund_baskets.parse_ssga_emea_holdings(
        _ssga_workbook(
            [_ssga_row("US0378331005", "Synthetic Global Company", "7", "United States", "2046251")]
        )
    )
    parsed_spy = parse_sp500_holdings(_spy_workbook())

    assert parsed_ssga.weights[0].sedol == "2046251"
    assert parsed_spy.securities[0].cusip == "037833100"
    assert parsed_spy.securities[0].sedol == "2046251"


def test_invesco_parser_keeps_only_common_shares() -> None:
    parsed = fund_baskets.parse_invesco_nasdaq100(
        {
            "effectiveDate": "2026-11-03T00:00:00Z",
            "holdings": [
                {
                    "ticker": "SYN1",
                    "issuerName": "Synthetic One",
                    "percentageOfTotalNetAssets": "4.25",
                    "cusip": "037833100",
                    "securityTypeCode": "COM",
                },
                {
                    "ticker": "SYN2",
                    "issuerName": "Synthetic Note",
                    "percentageOfTotalNetAssets": "0.75",
                    "securityTypeCode": "DEBT",
                },
            ],
        }
    )

    assert parsed.as_of == date(2026, 11, 3)
    assert [(row.symbol, row.weight_pct) for row in parsed.weights] == [("SYN1", Decimal("4.25"))]
    assert parsed.weights[0].isin == "US0378331005"


def test_spy_and_qqq_rows_resolve_isin_from_batch_sedol_and_shared_ticker() -> None:
    apple_isin = "US0378331005"
    emea = _ssga_workbook(
        [_ssga_row(apple_isin, "Synthetic Global Company", "7", "United States", "2046251")],
        "sppw",
    )

    batch = fund_baskets.fetch_fund_baskets(
        date(2026, 11, 4),
        fetch_ssga=lambda _ticker: emea,
        fetch_invesco=lambda: {
            "effectiveDate": "2026-11-03",
            "holdings": [
                {
                    "ticker": "AAPL",
                    "issuerName": "Synthetic Global Company",
                    "percentageOfTotalNetAssets": "7",
                    "cusip": None,
                    "securityTypeCode": "COM",
                }
            ],
        },
        fetch_nifty_csv=lambda _index: (
            "Company Name,Industry,Symbol,Series,ISIN Code\n"
            "Synthetic India Company,Technology,SYNIN,EQ,IN0000000017\n"
        ),
        fetch_nifty500=lambda: [
            UniverseSecurity("SYNIN", "Synthetic India Company", isin="IN0000000017")
        ],
        fetch_sp500=lambda: Sp500Holdings(
            [
                UniverseSecurity(
                    "AAPL",
                    "Synthetic Global Company",
                    weight_pct=Decimal("7"),
                    sedol="2046251",
                )
            ],
            date(2026, 11, 3),
        ),
    )

    sp500_row = batch.baskets["sp500"].weights[0]
    qqq_row = batch.baskets["nasdaq100"].weights[0]
    assert (sp500_row.isin, sp500_row.symbol) == (apple_isin, "AAPL")
    assert (qqq_row.isin, qqq_row.symbol) == (apple_isin, "AAPL")


def test_nse_index_csv_parser_uses_named_columns() -> None:
    first = _isin(1)
    text = (
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        f"Synthetic Alpha,Services,SYN1,EQ,{first}\n"
        "Synthetic Beta,Banks,SYN2,EQ,-\n"
    )

    members = fund_baskets.parse_nse_index_csv(text)

    assert [(member.symbol, member.name, member.isin) for member in members] == [
        ("SYN1", "Synthetic Alpha", first),
        ("SYN2", "Synthetic Beta", None),
    ]


def test_nifty_estimate_scales_and_renormalizes_four_members() -> None:
    first, second, third, fourth = (_isin(number) for number in range(1, 5))
    spym = (
        fund_baskets.BasketWeight("IN", first, None, "Alpha", Decimal("30")),
        fund_baskets.BasketWeight("IN", second, None, "Beta", Decimal("10")),
    )
    spyi = (
        fund_baskets.BasketWeight("IN", first, None, "Alpha", Decimal("10")),
        fund_baskets.BasketWeight("IN", second, None, "Beta", Decimal("10")),
        fund_baskets.BasketWeight("IN", third, None, "Gamma", Decimal("7")),
    )
    members = [
        fund_baskets.NseIndexMember("A", "Alpha", first),
        fund_baskets.NseIndexMember("B", "Beta", second),
        fund_baskets.NseIndexMember("C", "Gamma", third),
        fund_baskets.NseIndexMember("D", "Delta", fourth),
    ]

    rows, found, scale = fund_baskets.estimate_nifty_weights(members, spym, spyi)

    assert scale == Decimal("2")
    assert found == 3
    assert [row.weight_pct for row in rows] == [
        Decimal("55.555555"),
        Decimal("18.518519"),
        Decimal("25.925926"),
        Decimal("0"),
    ]
    assert sum((row.weight_pct for row in rows), Decimal(0)) == Decimal("100.000000")


def test_failed_msci_source_skips_nifty_baskets_without_network() -> None:
    spyi = _ssga_workbook([_ssga_row(_isin(1), "Synthetic Alpha", "100")], "spyi")

    def fetch_ssga(ticker: str) -> bytes:
        if ticker == "spyi":
            return spyi
        raise ConnectionError("synthetic unavailable source")

    batch = fund_baskets.fetch_fund_baskets(
        date(2026, 11, 4),
        fetch_ssga=fetch_ssga,
        fetch_invesco=lambda: {
            "effectiveDate": "2026-11-03",
            "holdings": [
                {
                    "ticker": "SYN1",
                    "issuerName": "Synthetic One",
                    "percentageOfTotalNetAssets": "100",
                    "securityTypeCode": "COM",
                }
            ],
        },
        fetch_sp500=lambda: Sp500Holdings(
            [UniverseSecurity("SYN1", "Synthetic One", weight_pct=Decimal("100"))],
            date(2026, 11, 3),
        ),
    )

    assert "nifty50" not in batch.baskets
    assert batch.errors["nifty50"] == "MissingMSCIWeightSource"


def test_derived_msci_baskets_and_nifty_members_are_built_from_shared_sources() -> None:
    alpha, beta, gamma = (_isin(number) for number in range(1, 4))
    gamma = _foreign_isin(3)
    em = _ssga_workbook(
        [
            _ssga_row(alpha, "Synthetic Alpha", "30"),
            _ssga_row(beta, "Synthetic Beta", "10"),
            _ssga_row(gamma, "Synthetic Gamma", "60", "United States"),
        ],
        "spym",
    )
    imi = _ssga_workbook(
        [
            _ssga_row(alpha, "Synthetic Alpha", "10"),
            _ssga_row(beta, "Synthetic Beta", "10"),
            _ssga_row(_isin(4), "Synthetic Delta", "7"),
        ],
        "spyi",
    )

    def fetch_ssga(ticker: str) -> bytes:
        if ticker == "spym":
            return em
        if ticker == "spyi":
            return imi
        raise ConnectionError("synthetic unavailable source")

    batch = fund_baskets.fetch_fund_baskets(
        date(2026, 11, 4),
        fetch_ssga=fetch_ssga,
        fetch_invesco=lambda: {
            "effectiveDate": "2026-11-03",
            "holdings": [
                {
                    "ticker": "SYN1",
                    "issuerName": "Synthetic One",
                    "percentageOfTotalNetAssets": "100",
                    "securityTypeCode": "COM",
                }
            ],
        },
        fetch_nifty_csv=lambda _index: (
            "Company Name,Industry,Symbol,Series,ISIN Code\n"
            f"Synthetic Alpha,Services,SYN1,EQ,{alpha}\n"
            f"Synthetic Beta,Banks,SYN2,EQ,{beta}\n"
        ),
        fetch_nifty500=lambda: [
            UniverseSecurity("SYN1", "Synthetic Alpha", isin=alpha),
            UniverseSecurity("SYN2", "Synthetic Beta", isin=beta),
        ],
        fetch_sp500=lambda: Sp500Holdings(
            [UniverseSecurity("SYN1", "Synthetic One", weight_pct=Decimal("100"))],
            date(2026, 11, 3),
        ),
    )

    assert [row.weight_pct for row in batch.baskets["msci_india"].weights] == [
        Decimal("75.000000"),
        Decimal("25.000000"),
    ]
    assert [row.weight_pct for row in batch.baskets["msci_em_ex_india"].weights] == [
        Decimal("100.000000")
    ]
    assert batch.baskets["nifty50"].weights_estimated is True
    assert batch.baskets["nifty50"].weight_source == "msci_em+imi"
