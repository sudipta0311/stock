"""Synthetic CAS parser and encrypted-PDF tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pytest
from tests.fixtures.cas.make_pdf import write_encrypted_pdf

from wealth.importer.cas import (
    ParseError,
    detect_cas_type,
    parse_cams_kfin,
    parse_cas,
    parse_cdsl,
    parse_nsdl,
)
from wealth.importer.cas.common import clean_security_name
from wealth.importer.models import ParsedStatement

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "cas"
_PASSWORD = "synthetic-test-password"


def _lines(name: str) -> list[str]:
    return (_FIXTURES / name).read_text(encoding="utf-8").splitlines()


@contextmanager
def _temporary_pdf(lines: list[str], filename: str) -> Iterator[Path]:
    with TemporaryDirectory(dir=_FIXTURES) as temporary_directory:
        path = Path(temporary_directory) / filename
        yield write_encrypted_pdf(lines, _PASSWORD, path)


@pytest.mark.parametrize(
    ("fixture", "source"),
    [
        ("nsdl.txt", "nsdl"),
        ("cdsl.txt", "cdsl"),
        ("cams_kfin.txt", "cams_kfin"),
    ],
)
def test_detect_cas_type_uses_issuer_header(fixture: str, source: str) -> None:
    assert detect_cas_type(_lines(fixture)) == source


def test_detect_cas_type_returns_unknown_for_unrelated_text() -> None:
    assert detect_cas_type(["A synthetic test document", "No statement data here"]) == "unknown"


def test_nsdl_parser_normalizes_demat_and_folio_holdings() -> None:
    parsed = parse_nsdl(_lines("nsdl.txt"))

    assert isinstance(parsed, ParsedStatement)
    assert parsed.source == "nsdl"
    assert parsed.statement_date == date(2026, 6, 30)
    assert len(parsed.holdings) == 7
    assert [holding.name for holding in parsed.holdings] == [
        "Synthetic Tools Limited",
        "Synthetic Bank Limited",
        "Synthetic Nifty ETF",
        "Synthetic Balanced Fund",
        "Synthetic Index Fund - Direct Growth",
        "Synthetic Taxable Bond",
        "Synthetic Sovereign Gold Bond",
    ]
    assert [holding.kind for holding in parsed.holdings] == [
        "equity",
        "equity",
        "etf",
        "mf",
        "mf",
        "bond",
        "sgb",
    ]
    assert [holding.quantity for holding in parsed.holdings] == [
        Decimal("20"),
        Decimal("5"),
        Decimal("10.000"),
        Decimal("15.500"),
        Decimal("2.500"),
        Decimal("10"),
        Decimal("5"),
    ]
    assert [holding.market_value for holding in parsed.holdings] == [
        Decimal("2000.00"),
        Decimal("1000.00"),
        Decimal("1000.00"),
        Decimal("310.00"),
        Decimal("100.00"),
        Decimal("10500.00"),
        Decimal("37500.00"),
    ]
    assert {holding.account_ref for holding in parsed.holdings[:4]} == {"XXXX5678"}
    assert parsed.holdings[4].account_ref == "XXXX3210"
    assert all(
        holding.account_ref and holding.account_ref.startswith("XXXX")
        for holding in parsed.holdings
    )
    assert not parsed.warnings


def test_cams_kfin_parser_keeps_transactions_and_calculates_cost_basis() -> None:
    parsed = parse_cams_kfin(_lines("cams_kfin.txt"))

    assert parsed.source == "cams_kfin"
    assert parsed.statement_date == date(2026, 6, 30)
    assert len(parsed.holdings) == 1
    holding = parsed.holdings[0]
    assert holding.kind == "mf"
    assert holding.quantity == Decimal("82.000")
    assert holding.market_value == Decimal("9020.00")
    assert holding.price == Decimal("110.00")
    assert holding.cost_value == Decimal("8200.00")
    assert holding.account_ref == "XXXX5678"
    assert holding.amc == "Synthetic Asset Management Company"
    assert holding.registrar == "CAMS"
    assert [holding.name for holding in parsed.holdings] == [
        "Synthetic Equity Fund - Direct Growth"
    ]
    assert [transaction.kind for transaction in parsed.transactions] == [
        "buy",
        "sell",
        "switch_in",
        "switch_out",
        "dividend_reinvest",
        "other",
    ]
    assert parsed.transactions[0].amount == Decimal("10000.00")
    assert parsed.transactions[0].account_ref == "XXXX5678"
    assert parsed.transactions[1].units == Decimal("20.000")
    assert parsed.transactions[-1].nav == Decimal("0")


@pytest.mark.parametrize(
    ("value", "numeric_fields", "expected"),
    [
        ("Synthetic   Nifty ETF 10.000 1", 2, "Synthetic Nifty ETF"),
        ("Synthetic Taxable Bond 10 10 10", 3, "Synthetic Taxable Bond"),
        ("Synthetic Balanced Fund 15.5", 1, "Synthetic Balanced Fund"),
        ("Synthetic Tools Limited 10.0", 1, "Synthetic Tools Limited"),
        ("Synthetic Fund 12,34,567.89 1,234,567.89", 2, "Synthetic Fund"),
        ("Synthetic Fund 1234567.89 100.00", 2, "Synthetic Fund"),
        ("Nifty 50 10.000 1,000.00", 2, "Nifty 50"),
        ("Bharat Bond ETF April 2030 10 100.00", 2, "Bharat Bond ETF April 2030"),
        ("SGB 2.50% Series IV 5 7,500.00", 2, "SGB 2.50% Series IV"),
        ("Synthetic Equity Fund INE000A01012", 0, "Synthetic Equity Fund"),
    ],
)
def test_clean_security_name_removes_columns_and_preserves_name_digits(
    value: str, numeric_fields: int, expected: str
) -> None:
    assert clean_security_name(value, trailing_numeric_fields=numeric_fields) == expected


def test_kfin_header_uses_kfin_registrar() -> None:
    lines = _lines("cams_kfin.txt")
    lines[0] = "KFINCASWS KFintech Consolidated Account Statement"

    assert detect_cas_type(lines) == "cams_kfin"
    parsed = parse_cams_kfin(lines)
    assert parsed.holdings[0].registrar == "KFintech"


def test_cdsl_parser_reads_complete_synthetic_holding_rows() -> None:
    parsed = parse_cdsl(_lines("cdsl.txt"))

    assert parsed.source == "cdsl"
    assert parsed.statement_date == date(2026, 6, 30)
    assert len(parsed.holdings) == 2
    assert parsed.holdings[0].kind == "equity"
    assert parsed.holdings[0].quantity == Decimal("12")
    assert parsed.holdings[0].market_value == Decimal("1200.00")
    assert parsed.holdings[0].account_ref == "XXXX3456"
    assert parsed.holdings[1].kind == "mf"
    assert parsed.holdings[1].quantity == Decimal("5.500")


def test_parse_cas_disables_cdsl_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CDSL_PARSER_BETA", raising=False)
    with (
        _temporary_pdf(_lines("cdsl.txt"), "cdsl-disabled.pdf") as pdf_path,
        pytest.raises(ParseError) as error,
    ):
        parse_cas(pdf_path.read_bytes(), _PASSWORD)

    assert error.value.code == "unsupported_format"
    assert (
        error.value.detail
        == "CDSL CAS support is coming soon. Upload your NSDL CAS or broker statement instead."
    )


def test_parse_cas_cdsl_beta_adds_verification_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CDSL_PARSER_BETA", "true")
    with _temporary_pdf(_lines("cdsl.txt"), "cdsl-beta.pdf") as pdf_path:
        parsed = parse_cas(pdf_path.read_bytes(), _PASSWORD)

    assert parsed.source == "cdsl"
    assert "cdsl_beta_verify_values" in parsed.warnings


def test_invalid_isin_is_kept_with_warning() -> None:
    parsed = parse_nsdl(_lines("nsdl_bad_isin.txt"))

    assert len(parsed.holdings) == 1
    assert parsed.holdings[0].isin == "INE000A01019"
    assert parsed.warnings == ["bad_isin"]


@pytest.mark.parametrize(
    ("fixture", "source"),
    [("nsdl.txt", "nsdl"), ("cams_kfin.txt", "cams_kfin")],
)
def test_parse_cas_reads_generated_encrypted_pdfs(fixture: str, source: str) -> None:
    with _temporary_pdf(_lines(fixture), f"{source}.pdf") as pdf_path:
        parsed = parse_cas(pdf_path.read_bytes(), _PASSWORD)

    assert parsed.source == source
    assert parsed.holdings


def test_parse_cas_rejects_wrong_password_without_echoing_it() -> None:
    with (
        _temporary_pdf(_lines("nsdl.txt"), "encrypted.pdf") as pdf_path,
        pytest.raises(ParseError) as error,
    ):
        parse_cas(pdf_path.read_bytes(), "incorrect-synthetic-password")

    assert error.value.code == "wrong_password"
    assert "incorrect-synthetic-password" not in str(error.value)
    assert _PASSWORD not in str(error.value)


def test_parse_cas_rejects_non_pdf_and_oversized_input() -> None:
    with pytest.raises(ParseError) as invalid:
        parse_cas(b"hello", _PASSWORD)
    assert invalid.value.code == "not_a_pdf"

    with pytest.raises(ParseError) as too_large:
        parse_cas(b"x" * (11 * 1024 * 1024), _PASSWORD)
    assert too_large.value.code == "too_large"


def test_parse_cas_rejects_unknown_pdf_layout() -> None:
    with (
        _temporary_pdf(["Synthetic report with no CAS issuer marker"], "unknown.pdf") as pdf_path,
        pytest.raises(ParseError) as error,
    ):
        parse_cas(pdf_path.read_bytes(), _PASSWORD)

    assert error.value.code == "unsupported_format"


def test_generated_nsdl_pdf_parses_under_three_seconds() -> None:
    with _temporary_pdf(_lines("nsdl.txt"), "timed.pdf") as pdf_path:
        started = perf_counter()
        parsed = parse_cas(pdf_path.read_bytes(), _PASSWORD)
        elapsed = perf_counter() - started

    assert parsed.source == "nsdl"
    assert elapsed < 3
