"""casparser output maps onto our parsed statement without personal details."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import casparser
import pytest
from casparser.enums import FileType, TransactionType
from casparser.exceptions import CASParseError, IncorrectPasswordError
from casparser.types import (
    Bond,
    CASData,
    DematAccount,
    DematOwner,
    Equity,
    Folio,
    InvestorInfo,
    MutualFund,
    NSDLCASData,
    Scheme,
    SchemeValuation,
    StatementPeriod,
    TransactionData,
)

from wealth.importer.cas import parse_cas
from wealth.importer.cas.common import ParseError

# Built at runtime so no PAN-shaped literal lives in the repository.
_SYNTHETIC_PAN = "".join(["ZZ", "ZZZ", "0000", "Z"])
_INVESTOR = InvestorInfo(
    name="Synthetic Investor",
    email="investor@example.test",
    address="1 Example Road",
    mobile="0000000000",
)
_PERIOD = StatementPeriod(**{"from": "01-Sep-2026", "to": "30-Sep-2026"})


def _use(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    def fake_read(_file: object, _password: str, output: str = "dict") -> Any:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(casparser, "read_cas_pdf", fake_read)


def _nsdl() -> NSDLCASData:
    return NSDLCASData(
        accounts=[
            DematAccount(
                name="Synthetic DP",
                type="NSDL Demat Account",
                dp_id="IN000000",
                client_id="10001234",
                folios=0,
                balance=Decimal("0"),
                owners=[DematOwner(name="Synthetic Investor", PAN=_SYNTHETIC_PAN)],
                equities=[
                    Equity(
                        name="SYNTHETIC INDUSTRIES LTD",
                        isin="INE000A01019",
                        num_shares=Decimal("10"),
                        price=Decimal("1500.25"),
                        value=Decimal("15002.50"),
                    ),
                    Equity(
                        name="SOLD LTD",
                        isin="INE000B01017",
                        num_shares=Decimal("0"),
                        price=Decimal("10"),
                        value=Decimal("0"),
                    ),
                ],
                mutual_funds=[
                    MutualFund(
                        name="SYNTHETIC NIFTY ETF",
                        isin="INF000A01013",
                        balance=Decimal("40"),
                        nav=Decimal("250.10"),
                        value=Decimal("10004.00"),
                    )
                ],
                bonds=[
                    Bond(
                        name="SGB SERIES SYNTHETIC",
                        isin="IN0020000017",
                        num_bonds=Decimal("5"),
                        value=Decimal("37500.004"),
                    )
                ],
            ),
            DematAccount(
                name="Mutual Fund Folios",
                type="MF Folios",
                folios=1,
                balance=Decimal("0"),
                owners=[],
                equities=[],
                mutual_funds=[
                    MutualFund(
                        name="SYNTHETIC FLEXI CAP FUND - DIRECT GROWTH",
                        isin="INF000B01011",
                        balance=Decimal("123.456"),
                        nav=Decimal("81.00"),
                        value=Decimal("10000.00"),
                        avg_cost=Decimal("60.00"),
                        total_cost=Decimal("7407.36"),
                        folio="98765432/10",
                        pnl=Decimal("-50.00"),
                        **{"return": Decimal("-1.5")},
                    )
                ],
            ),
        ],
        statement_period=_PERIOD,
        investor_info=_INVESTOR,
        file_type=FileType.NSDL,
    )


def test_an_nsdl_statement_maps_every_holding_and_drops_personal_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, _nsdl())

    parsed = parse_cas(b"%PDF-synthetic", "synthetic-password")

    assert parsed.source == "nsdl"
    assert parsed.statement_date == date(2026, 9, 30)
    by_isin = {holding.isin: holding for holding in parsed.holdings}
    assert set(by_isin) == {"INE000A01019", "INF000A01013", "IN0020000017", "INF000B01011"}
    share = by_isin["INE000A01019"]
    assert (share.kind, share.quantity, share.price, share.market_value) == (
        "equity",
        Decimal("10"),
        Decimal("1500.25"),
        Decimal("15002.50"),
    )
    assert share.account_ref == "XXXX1234"
    assert by_isin["INF000A01013"].kind == "etf"
    gold = by_isin["IN0020000017"]
    assert (gold.kind, gold.market_value) == ("sgb", Decimal("37500.00"))
    # A folio row with a loss still imports, at its units, NAV and value.
    fund = by_isin["INF000B01011"]
    assert (fund.quantity, fund.price, fund.market_value, fund.cost_value) == (
        Decimal("123.456"),
        Decimal("81.00"),
        Decimal("10000.00"),
        Decimal("7407.36"),
    )
    assert fund.account_ref == "XXXX3210"
    dumped = parsed.model_dump_json()
    for personal in (_SYNTHETIC_PAN, "Synthetic Investor", "example.test", "Example Road"):
        assert personal not in dumped


def test_a_cams_statement_keeps_holdings_and_buying_and_selling_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def row(day: int, kind: TransactionType, amount: str, units: str | None) -> TransactionData:
        return TransactionData(
            date=date(2026, 9, day),
            description="synthetic",
            amount=Decimal(amount),
            units=Decimal(units) if units else None,
            nav=Decimal("50") if units else None,
            type=kind,
        )

    scheme = Scheme(
        scheme="SYNTHETIC LARGE CAP FUND - DIRECT GROWTH",
        rta_code="SYN1",
        rta="KFINTECH",
        isin="INF000C01019",
        open=Decimal("0"),
        close=Decimal("190"),
        close_calculated=Decimal("190"),
        valuation=SchemeValuation(
            date=date(2026, 9, 30), nav=Decimal("52"), value=Decimal("9880"), cost=Decimal("9500")
        ),
        transactions=[
            row(5, TransactionType.PURCHASE_SIP, "5000", "100"),
            row(5, TransactionType.STAMP_DUTY_TAX, "0.25", None),
            row(15, TransactionType.PURCHASE_SIP, "5000", "100"),
            row(20, TransactionType.REDEMPTION, "-500", "-10"),
        ],
    )
    _use(
        monkeypatch,
        CASData(
            statement_period=_PERIOD,
            folios=[Folio(folio="11223344/55", amc="Synthetic AMC", schemes=[scheme])],
            investor_info=_INVESTOR,
            cas_type="DETAILED",
            file_type=FileType.KFINTECH,
        ),
    )

    parsed = parse_cas(b"%PDF-synthetic", "synthetic-password")

    assert parsed.source == "cams_kfin"
    (holding,) = parsed.holdings
    assert (holding.quantity, holding.price, holding.market_value, holding.registrar) == (
        Decimal("190"),
        Decimal("52"),
        Decimal("9880.00"),
        "KFintech",
    )
    assert [(row.kind, row.units, row.amount) for row in parsed.transactions] == [
        ("buy", Decimal("100"), Decimal("5000.00")),
        ("buy", Decimal("100"), Decimal("5000.00")),
        ("sell", Decimal("10"), Decimal("500.00")),
    ]


def test_a_wrong_password_is_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, IncorrectPasswordError("Incorrect PDF password!"))

    with pytest.raises(ParseError) as error:
        parse_cas(b"%PDF-synthetic", "incorrect-synthetic-password")

    assert error.value.code == "wrong_password"


def test_a_layout_casparser_cannot_read_falls_back_to_the_line_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, CASParseError("Could not identify the CAS issuer."))

    with pytest.raises(ParseError) as error:
        parse_cas(b"not a pdf", "synthetic-password")

    # The line parser then ran and reached its own verdict on the bytes.
    assert error.value.code == "not_a_pdf"


def test_a_cdsl_statement_read_by_casparser_asks_to_check_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statement = _nsdl().model_copy(update={"file_type": FileType.CDSL})
    _use(monkeypatch, statement)

    parsed = parse_cas(b"%PDF-synthetic", "synthetic-password")

    assert parsed.source == "cdsl"
    assert "cdsl_beta_verify_values" in parsed.warnings


def test_a_folio_row_named_after_the_statement_period_gets_its_fund_name_by_isin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statement = _nsdl()
    folio_account = statement.accounts[1]
    period_named = folio_account.mutual_funds[0].model_copy(
        update={"name": "APR-26 TO SEP-26", "isin": "INF879O01027"}
    )
    statement.accounts[1] = folio_account.model_copy(update={"mutual_funds": [period_named]})
    _use(monkeypatch, statement)

    parsed = parse_cas(b"%PDF-synthetic", "synthetic-password")

    names = {holding.isin: holding.name for holding in parsed.holdings}
    # Public scheme name from casparser's bundled ISIN list.
    assert names["INF879O01027"] == "PARAG PARIKH FLEXI CAP FUND - DIRECT PLAN GROWTH"
