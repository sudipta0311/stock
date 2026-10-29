"""Read real CAS PDFs with casparser and map them to a ParsedStatement.

Real NSDL, CDSL, CAMS and KFintech statements are tables whose columns only line up by
their position on the page; plain text extraction loses that (NSDL folio rows carry eight
numbers). casparser reads the page layout, so it is tried first and the line parsers in
this package stay as the fallback. Investor name, PAN, address and phone are dropped.
"""

from __future__ import annotations

import io
import re
import threading
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from wealth.importer.cas.common import (
    ParseError,
    isin_warning,
    mask_account_ref,
    parse_date_text,
)
from wealth.importer.models import (
    HoldingKind,
    ParsedHolding,
    ParsedStatement,
    ParsedTransaction,
    TransactionKind,
)

_PAISE = Decimal("0.01")
_TRANSACTION_KINDS: dict[str, TransactionKind] = {
    "PURCHASE": "buy",
    "PURCHASE_SIP": "buy",
    "REDEMPTION": "sell",
    "SWITCH_IN": "switch_in",
    "SWITCH_IN_MERGER": "switch_in",
    "SWITCH_OUT": "switch_out",
    "SWITCH_OUT_MERGER": "switch_out",
    "DIVIDEND_REINVEST": "dividend_reinvest",
}
# pdfium is not thread-safe, and parsing runs in a worker thread per request.
_PDFIUM_LOCK = threading.Lock()
# Real NSDL folio rows can come back named after the statement period ("APR-26 TO
# SEP-26"); the fund is then found by ISIN, or its type would be guessed from the period.
_PERIOD_NAME = re.compile(r"^[A-Z]{3}-\d{2,4}\s+TO\s+[A-Z]{3}-\d{2,4}$", re.I)


class CasparserFailed(Exception):
    """casparser could not read this file; the caller falls back to the line parsers."""


def parse_with_casparser(data: bytes, password: str | None) -> ParsedStatement:
    """Parse in memory; a wrong password raises ParseError, anything else CasparserFailed."""
    try:
        from casparser import read_cas_pdf
        from casparser.exceptions import IncorrectPasswordError
    except ImportError as error:
        raise CasparserFailed("not_installed") from error

    try:
        with _PDFIUM_LOCK:
            parsed = read_cas_pdf(io.BytesIO(data), password or "", output="dict")
    except IncorrectPasswordError:
        raise ParseError("wrong_password") from None
    except Exception as error:
        raise CasparserFailed(type(error).__name__) from None

    statement = _from_depository(parsed) if hasattr(parsed, "accounts") else _from_rta(parsed)
    if not statement.holdings:
        raise CasparserFailed("no_holdings")
    return statement


def _from_depository(data: Any) -> ParsedStatement:
    file_type = str(getattr(data.file_type, "name", data.file_type)).upper()
    source = "cdsl" if "CDSL" in file_type else "nsdl"
    statement_date = _statement_date(data)
    holdings: list[ParsedHolding] = []
    warnings: list[str] = []

    for account in data.accounts:
        account_ref = mask_account_ref(account.client_id)
        for equity in account.equities:
            _add_holding(
                holdings,
                warnings,
                kind="equity",
                isin=equity.isin,
                name=equity.name,
                quantity=equity.num_shares,
                price=equity.price,
                value=equity.value,
                account_ref=account_ref,
            )
        for fund in account.mutual_funds:
            _add_holding(
                holdings,
                warnings,
                kind="etf" if re.search(r"\bETF\b", fund.name or "", re.I) else "mf",
                isin=fund.isin,
                name=fund.name,
                quantity=fund.balance,
                price=fund.nav,
                value=fund.value,
                cost=fund.total_cost,
                account_ref=mask_account_ref(fund.folio) or account_ref,
            )
        for bond in account.bonds:
            name = bond.name or ""
            price = bond.market_price
            if price is None and bond.num_bonds:
                price = bond.value / bond.num_bonds
            _add_holding(
                holdings,
                warnings,
                kind="sgb" if re.search(r"\b(?:SGB|SOVEREIGN GOLD)\b", name, re.I) else "bond",
                isin=bond.isin,
                name=bond.name,
                quantity=bond.num_bonds,
                price=price,
                value=bond.value,
                account_ref=account_ref,
            )

    if getattr(data, "nps", None) is not None:
        warnings.append("nps_not_imported")
    if getattr(data, "parse_warnings", None):
        warnings.append("verify_values")
    if source == "cdsl":
        warnings.append("cdsl_beta_verify_values")
    return ParsedStatement(
        source=source,
        statement_date=statement_date,
        holdings=holdings,
        transactions=[],
        warnings=list(dict.fromkeys(warnings)),
    )


def _from_rta(data: Any) -> ParsedStatement:
    statement_date = _statement_date(data)
    holdings: list[ParsedHolding] = []
    transactions: list[ParsedTransaction] = []
    warnings: list[str] = []

    for folio in data.folios:
        account_ref = mask_account_ref(folio.folio)
        for scheme in folio.schemes:
            if not scheme.isin:
                warnings.append("scheme_without_isin")
                continue
            registrar = "KFintech" if "KFIN" in str(scheme.rta).upper() else "CAMS"
            valuation = scheme.valuation
            _add_holding(
                holdings,
                warnings,
                kind="mf",
                isin=scheme.isin,
                name=scheme.scheme,
                quantity=scheme.close,
                price=valuation.nav,
                value=valuation.value,
                cost=valuation.cost,
                account_ref=account_ref,
                amc=folio.amc or None,
                registrar=registrar,
            )
            for row in scheme.transactions:
                transaction = _transaction(scheme.isin, row, account_ref)
                if transaction is not None:
                    transactions.append(transaction)

    if getattr(data, "parse_warnings", None):
        warnings.append("verify_values")
    return ParsedStatement(
        source="cams_kfin",
        statement_date=statement_date,
        holdings=holdings,
        transactions=transactions,
        warnings=list(dict.fromkeys(warnings)),
    )


def _statement_date(data: Any) -> date:
    period = getattr(data, "statement_period", None)
    parsed = parse_date_text(str(getattr(period, "to", "") or ""))
    if parsed is None:
        raise CasparserFailed("no_statement_date")
    return parsed


def _add_holding(
    holdings: list[ParsedHolding],
    warnings: list[str],
    *,
    kind: HoldingKind,
    isin: str,
    name: str | None,
    quantity: Decimal | None,
    price: Decimal | None,
    value: Decimal | None,
    account_ref: str | None,
    cost: Decimal | None = None,
    amc: str | None = None,
    registrar: str | None = None,
) -> None:
    # A closed folio or sold position still prints a zero row; it isn't a holding.
    if quantity is None or quantity <= 0:
        return
    if warning := isin_warning(isin):
        warnings.append(warning)
    holdings.append(
        ParsedHolding(
            kind=kind,
            isin=isin,
            name=_holding_name(name, isin),
            quantity=quantity,
            market_value=_paise(value),
            price=abs(price) if price is not None else None,
            cost_value=_paise(cost),
            account_ref=account_ref,
            amc=amc,
            registrar=registrar,
        )
    )


def _holding_name(name: str | None, isin: str) -> str:
    cleaned = re.sub(r"\s+", " ", name or "").strip()
    if cleaned and not _PERIOD_NAME.match(cleaned) and re.search(r"[A-Za-z]{3}", cleaned):
        return cleaned
    return _isin_database_name(isin) or cleaned or isin


def _isin_database_name(isin: str) -> str | None:
    """Fund name from casparser's bundled ISIN list (read-only, no network)."""
    try:
        from casparser_isin import MFISINDb

        database = MFISINDb()
        database.initialize()
        try:
            rows = database.direct_isin_lookup(isin)
        finally:
            database.close()
    except Exception:
        return None
    name = rows[0].get("name") if rows else None
    return re.sub(r"\s+", " ", name).strip() if isinstance(name, str) and name.strip() else None


def _transaction(isin: str, row: Any, account_ref: str | None) -> ParsedTransaction | None:
    kind = _TRANSACTION_KINDS.get(str(getattr(row.type, "name", row.type)).upper())
    when = row.date if isinstance(row.date, date) else parse_date_text(str(row.date))
    if kind is None or when is None:
        return None
    units = abs(row.units or Decimal(0))
    amount = abs(row.amount or Decimal(0))
    if units == 0 and amount == 0:
        return None
    nav = abs(row.nav) if row.nav else (amount / units if units else Decimal(0))
    return ParsedTransaction(
        isin=isin,
        date=when,
        kind=kind,
        units=units,
        amount=amount.quantize(_PAISE, rounding=ROUND_HALF_UP),
        nav=nav,
        account_ref=account_ref,
    )


def _paise(value: Decimal | None) -> Decimal | None:
    if value is None or value < 0:
        return None
    return value.quantize(_PAISE, rounding=ROUND_HALF_UP)
