"""Pure line parser for CAMS and KFintech mutual-fund CAS statements."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from wealth.importer.cas.common import (
    ParseError,
    clean_lines,
    clean_security_name,
    decimal_value,
    isin_from_line,
    isin_warning,
    mask_account_ref,
    normalized_name,
    parse_date_text,
    parse_statement_date,
)
from wealth.importer.models import ParsedHolding, ParsedStatement, ParsedTransaction

_FOLIO = re.compile(r"\bfolio\s*(?:no\.?|number)?\s*:?\s*([A-Z0-9/-]+)", re.I)
_ISIN_LABEL = re.compile(r"\bISIN\s*:?\s*([A-Z]{2}[A-Z0-9]{9}[0-9])\b", re.I)
_SCHEME = re.compile(r"^(?:scheme(?:\s+name)?|fund\s+name)\s*:?\s*(.+)$", re.I)
_NAV = re.compile(r"\b(?:closing\s+)?NAV(?:\s+on\s+[\dA-Za-z/-]+)?\s*:?\s*([\d,]+(?:\.\d+)?)", re.I)
_VALUE = re.compile(
    r"\b(?:market\s+value|valuation)(?:\s+as\s+on\s+[\dA-Za-z/-]+)?\s*:?\s*(?:[Rr]s\.?\s*)?([\d,]+(?:\.\d+)?)",
    re.I,
)
_BALANCE = re.compile(r"\b(Opening|Closing)\s+Unit\s+Balance\s*:?\s*([\d,]+(?:\.\d+)?)", re.I)
_DATE_AT_START = re.compile(r"^\s*(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})\b\s*(.*)$", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d[\d,]*)(?:\.\d+)?(?![A-Za-z0-9])")
_TRANSACTION_HEADER = re.compile(r"\bdate\b.*\b(?:transaction|description)\b", re.I)


@dataclass
class _HoldingDraft:
    isin: str
    folio: str
    name: str = ""
    opening_units: Decimal = Decimal(0)
    closing_units: Decimal | None = None
    nav: Decimal | None = None
    market_value: Decimal | None = None
    amc: str | None = None
    registrar: str = "CAMS"
    transactions: list[ParsedTransaction] = field(default_factory=list)


def parse_cams_kfin(lines: list[str]) -> ParsedStatement:
    """Parse mutual-fund scheme balances and detailed transaction rows."""
    text_lines = clean_lines(lines)
    statement_date = parse_statement_date(text_lines)
    if statement_date is None:
        raise ParseError("unsupported_format")

    registrar = (
        "KFintech"
        if any("KFIN" in line.upper() or "KFINTECH" in line.upper() for line in text_lines[:12])
        else "CAMS"
    )
    current_folio = ""
    current_isin: str | None = None
    current_name = ""
    current_amc: str | None = None
    drafts: dict[tuple[str, str], _HoldingDraft] = {}
    warnings: list[str] = []
    transactions: list[ParsedTransaction] = []

    for line in text_lines:
        folio_match = _FOLIO.search(line)
        if folio_match is not None:
            current_folio = folio_match.group(1)
            current_isin = None
            current_name = ""

        amc_match = re.match(r"\s*AMC\s*:?\s*(.+)$", line, re.I)
        if amc_match is not None:
            current_amc = normalized_name(amc_match.group(1))

        scheme_match = _SCHEME.match(line)
        if scheme_match is not None:
            current_name = clean_security_name(scheme_match.group(1))
            if current_isin is not None:
                draft = _draft(drafts, current_folio, current_isin, registrar)
                draft.name = current_name
                draft.amc = current_amc

        isin_match = _ISIN_LABEL.search(line)
        isin = isin_match.group(1).upper() if isin_match is not None else isin_from_line(line)
        if isin is not None:
            current_isin = isin
            draft = _draft(drafts, current_folio, current_isin, registrar)
            if current_name:
                draft.name = current_name
            draft.amc = current_amc

        if current_isin is None:
            continue
        draft = _draft(drafts, current_folio, current_isin, registrar)

        balance_match = _BALANCE.search(line)
        if balance_match is not None:
            balance = decimal_value(balance_match.group(2))
            if balance is not None:
                if balance_match.group(1).lower() == "opening":
                    draft.opening_units = balance
                else:
                    draft.closing_units = balance

        nav_match = _NAV.search(line)
        if nav_match is not None:
            draft.nav = decimal_value(nav_match.group(1))

        value_match = _VALUE.search(line)
        if value_match is not None:
            draft.market_value = decimal_value(value_match.group(1))

        parsed = _parse_transaction(line, current_isin, mask_account_ref(current_folio))
        if parsed is not None:
            draft.transactions.append(parsed)
            transactions.append(parsed)

    holdings: list[ParsedHolding] = []
    for draft in drafts.values():
        if not draft.name:
            continue
        quantity = draft.closing_units
        if quantity is None:
            quantity = _units_after_transactions(draft.opening_units, draft.transactions)
        market_value = draft.market_value
        if market_value is None and draft.nav is not None:
            market_value = quantity * draft.nav
        cost_value = _remaining_cost_basis(draft.transactions)
        if warning := isin_warning(draft.isin):
            warnings.append(warning)
        holdings.append(
            ParsedHolding(
                kind="mf",
                isin=draft.isin,
                name=draft.name,
                quantity=quantity,
                market_value=market_value,
                price=draft.nav,
                cost_value=cost_value,
                account_ref=mask_account_ref(draft.folio),
                amc=draft.amc,
                registrar=draft.registrar,
            )
        )

    if not holdings:
        raise ParseError("unsupported_format")
    return ParsedStatement(
        source="cams_kfin",
        statement_date=statement_date,
        holdings=holdings,
        transactions=transactions,
        warnings=list(dict.fromkeys(warnings)),
    )


def _draft(
    drafts: dict[tuple[str, str], _HoldingDraft], folio: str, isin: str, registrar: str
) -> _HoldingDraft:
    key = (folio, isin)
    if key not in drafts:
        drafts[key] = _HoldingDraft(isin=isin, folio=folio, registrar=registrar)
    return drafts[key]


def _parse_transaction(
    line: str, isin: str, account_ref: str | None = None
) -> ParsedTransaction | None:
    if _TRANSACTION_HEADER.search(line):
        return None
    date_match = _DATE_AT_START.match(line)
    if date_match is None:
        return None
    transaction_date = parse_date_text(date_match.group(1))
    body = date_match.group(2).strip()
    if transaction_date is None:
        return None

    number_matches = list(_NUMBER.finditer(body))
    if len(number_matches) < 2:
        return None
    selected = number_matches[-4:] if len(number_matches) >= 4 else number_matches[-3:]
    numbers = [decimal_value(match.group()) for match in selected]
    if any(number is None for number in numbers):
        return None
    values = [number for number in numbers if number is not None]
    if len(values) == 4:
        amount, units, nav, _balance = values
    else:
        amount, units, nav = values
    amount = abs(amount)
    units = abs(units)
    nav = abs(nav) if nav else amount / units if units else Decimal(0)

    description = body[: selected[0].start()].strip(" |\t:-")
    kind = _transaction_kind(description)
    if units == 0 and amount == 0 and kind != "other":
        return None
    return ParsedTransaction(
        isin=isin,
        date=transaction_date,
        kind=kind,
        units=units,
        amount=amount,
        nav=nav,
        account_ref=account_ref,
    )


def _transaction_kind(description: str) -> str:
    text = re.sub(r"\s+", " ", description).lower()
    if "switch" in text and re.search(r"\b(in|purchase)\b", text):
        return "switch_in"
    if "switch" in text and re.search(r"\b(out|redemption)\b", text):
        return "switch_out"
    if "dividend" in text and re.search(r"reinvest|reinvestment", text):
        return "dividend_reinvest"
    if re.search(r"\b(redemption|redeemed|repurchase|sell|withdrawal)\b", text):
        return "sell"
    if re.search(r"\b(purchase|purchased|sip|subscription|buy|investment)\b", text):
        return "buy"
    return "other"


def _units_after_transactions(
    opening_units: Decimal, transactions: list[ParsedTransaction]
) -> Decimal:
    quantity = opening_units
    for transaction in transactions:
        if transaction.kind in {"buy", "switch_in", "dividend_reinvest"}:
            quantity += transaction.units
        elif transaction.kind in {"sell", "switch_out"}:
            quantity = max(Decimal(0), quantity - transaction.units)
    return quantity


def _remaining_cost_basis(transactions: list[ParsedTransaction]) -> Decimal | None:
    acquired_units = Decimal(0)
    cost = Decimal(0)
    has_purchase = False
    for transaction in transactions:
        if transaction.kind in {"buy", "switch_in", "dividend_reinvest"}:
            acquired_units += transaction.units
            cost += transaction.amount
            has_purchase = True
        elif transaction.kind in {"sell", "switch_out"} and acquired_units > 0:
            removed_units = min(transaction.units, acquired_units)
            average_cost = cost / acquired_units
            cost -= average_cost * removed_units
            acquired_units -= removed_units
    return max(Decimal(0), cost) if has_purchase else None
