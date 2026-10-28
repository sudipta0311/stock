"""Pure line parser for CDSL CAS tables with explicit numeric holding rows."""

from __future__ import annotations

import re
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
    parse_statement_date,
)
from wealth.importer.models import ParsedHolding, ParsedStatement

_ACCOUNT = re.compile(
    r"(?:BO\s*ID|DP\s*ID\s*/\s*Client\s*ID|Client\s*ID)\s*:?\s*([A-Z0-9/-]+)", re.I
)
_FOLIO = re.compile(r"folio(?:\s+no(?:\.)?)?\s*:?\s*([A-Z0-9/-]+)", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d[\d,]*)(?:\.\d+)?(?![A-Za-z0-9])")
_LABELS = re.compile(
    r"^(?:ISIN|Security Name|Name|Quantity|Balance|Rate|Price|Value|Market Value)\b", re.I
)


def parse_cdsl(lines: list[str]) -> ParsedStatement:
    """Parse CDSL text when it exposes an ISIN and a complete holding row."""
    text_lines = clean_lines(lines)
    statement_date = parse_statement_date(text_lines)
    if statement_date is None:
        raise ParseError("unsupported_format")

    account_ref: str | None = None
    records: list[tuple[str, list[str], str | None]] = []
    current_isin: str | None = None
    current_lines: list[str] = []
    current_account: str | None = None
    for line in text_lines:
        account_match = _ACCOUNT.search(line)
        if account_match is not None:
            account_ref = mask_account_ref(account_match.group(1))
        folio_match = _FOLIO.search(line)
        if folio_match is not None:
            account_ref = mask_account_ref(folio_match.group(1))

        isin = isin_from_line(line)
        if isin is not None:
            if current_isin is not None:
                records.append((current_isin, current_lines, current_account))
            current_isin = isin
            current_lines = [line]
            current_account = account_ref
        elif current_isin is not None:
            if re.match(r"^(?:Total|Sub Total|Notes:)", line, re.I):
                records.append((current_isin, current_lines, current_account))
                current_isin = None
                current_lines = []
                current_account = None
            else:
                current_lines.append(line)
    if current_isin is not None:
        records.append((current_isin, current_lines, current_account))

    holdings: list[ParsedHolding] = []
    warnings: list[str] = []
    for isin, record_lines, account in records:
        parsed = _holding_from_record(isin, record_lines, account, warnings)
        if parsed is not None:
            holdings.append(parsed)
    if not holdings:
        raise ParseError("unsupported_format")
    return ParsedStatement(
        source="cdsl",
        statement_date=statement_date,
        holdings=holdings,
        transactions=[],
        warnings=list(dict.fromkeys(warnings)),
    )


def _holding_from_record(
    isin: str,
    lines: list[str],
    account_ref: str | None,
    warnings: list[str],
) -> ParsedHolding | None:
    candidates: list[tuple[list[Decimal], str, int]] = []
    for line in lines:
        matches = list(_NUMBER.finditer(line))
        if len(matches) < 3 or line[matches[-1].end() :].strip(" \t.,"):
            continue
        field_count = 4 if len(matches) >= 4 and "face value" in line.casefold() else 3
        values = [decimal_value(match.group()) for match in matches[-field_count:]]
        if any(value is None for value in values):
            continue
        candidates.append(([value for value in values if value is not None], line, field_count))
    if not candidates:
        return None

    values, value_line, field_count = candidates[-1]
    if field_count == 4:
        quantity, price, market_value = values[-3:]
    else:
        quantity, price, market_value = values
    if min(quantity, price, market_value) < Decimal(0):
        return None

    names: list[str] = []
    folio_ref: str | None = None
    for line in lines:
        folio_match = _FOLIO.search(line)
        if folio_match is not None:
            folio_ref = mask_account_ref(folio_match.group(1))
        if line == value_line:
            matches = list(_NUMBER.finditer(line))
            if len(matches) >= field_count:
                line = line[: matches[-field_count].start()]
        line = re.sub(r"(?i)[A-Z]{2}[A-Z0-9]{9}[0-9]", "", line)
        if _LABELS.match(line):
            continue
        cleaned = normalized_name(line)
        if cleaned and not re.fullmatch(r"[\d,./ -]+", cleaned):
            names.append(cleaned)
    name = clean_security_name(" ".join(names))
    if not name:
        return None

    lowered = name.casefold()
    if "sovereign gold" in lowered or re.search(r"\bsgb\b", lowered):
        kind = "sgb"
    elif re.search(r"\betf\b", lowered):
        kind = "etf"
    elif re.search(r"\b(bond|ncd|debenture)\b", lowered):
        kind = "bond"
    elif re.search(r"\b(fund|mutual fund|scheme)\b", lowered):
        kind = "mf"
    else:
        kind = "equity"

    if warning := isin_warning(isin):
        warnings.append(warning)
    return ParsedHolding(
        kind=kind,
        isin=isin,
        name=name,
        quantity=quantity,
        price=price,
        market_value=market_value,
        account_ref=folio_ref or account_ref,
    )
