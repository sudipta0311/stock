"""Pure line parser for synthetic and text-extracted NSDL CAS statements."""

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
    parse_statement_date,
)
from wealth.importer.models import ParsedHolding, ParsedStatement

_ACCOUNT = re.compile(r"DP\s*ID\s*:?\s*(\S+)\s+Client\s*ID\s*:?\s*(\S+)", re.I)
_FOLIO = re.compile(r"(?:folio(?:\s+no(?:\.)?)?)\s*:?\s*([A-Z0-9/-]+)", re.I)
_SECTION_STARTS = {
    "equity": re.compile(r"^Equity Shares\b", re.I),
    "demat_mf": re.compile(r"^Mutual Funds\s*\(M\)", re.I),
    "mf_folio": re.compile(r"^Mutual Fund Folios\s*\(F\)", re.I),
    "bond": re.compile(r"^(?:Corporate )?Bonds\b|^Sovereign Gold Bonds\b", re.I),
}
_SECTION_END = re.compile(
    r"^(?:Sub Total|Notes:|Transactions for the period|Total Portfolio)", re.I
)
_SYMBOL = re.compile(r"\b[A-Z0-9._-]{1,18}\.(?:NSE|BSE)\b", re.I)
_TRAIL_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?(?:\d[\d,]*)(?:\.\d+)?(?![A-Za-z0-9])")


def parse_nsdl(lines: list[str]) -> ParsedStatement:
    """Parse NSDL demat equities, debt, demat funds and folio funds."""
    text_lines = clean_lines(lines)
    statement_date = parse_statement_date(text_lines)
    if statement_date is None:
        raise ParseError("unsupported_format")

    holdings: list[ParsedHolding] = []
    warnings: list[str] = []
    section: str | None = None
    account_ref: str | None = None
    section_lines: list[tuple[str, str | None]] = []

    def flush() -> None:
        if section is None or not section_lines:
            section_lines.clear()
            return
        for holding_lines, account in _isin_segments(section_lines):
            holding = _parse_segment(section, holding_lines, account, warnings)
            if holding is not None:
                holdings.append(holding)
        section_lines.clear()

    for line in text_lines:
        account_match = _ACCOUNT.search(line)
        if account_match is not None:
            account_ref = mask_account_ref(account_match.group(2))

        next_section = next(
            (name for name, pattern in _SECTION_STARTS.items() if pattern.search(line)), None
        )
        if next_section is not None:
            flush()
            section = next_section
            continue
        if _SECTION_END.match(line):
            flush()
            section = None
            continue
        if section is not None:
            section_lines.append((line, account_ref))
    flush()

    if not holdings:
        raise ParseError("unsupported_format")
    return ParsedStatement(
        source="nsdl",
        statement_date=statement_date,
        holdings=holdings,
        transactions=[],
        warnings=list(dict.fromkeys(warnings)),
    )


def _isin_segments(
    lines: list[tuple[str, str | None]],
) -> list[tuple[list[tuple[str, str | None]], str | None]]:
    segments: list[tuple[list[tuple[str, str | None]], str | None]] = []
    current: list[tuple[str, str | None]] = []
    current_account: str | None = None
    for line, account in lines:
        isin = isin_from_line(line)
        if isin is not None:
            if current:
                segments.append((current, current_account))
            current = [(line, account)]
            current_account = account
        elif current:
            current.append((line, account))
    if current:
        segments.append((current, current_account))
    return segments


def _parse_segment(
    section: str,
    lines: list[tuple[str, str | None]],
    account_ref: str | None,
    warnings: list[str],
) -> ParsedHolding | None:
    isin = next((found for line, _ in lines if (found := isin_from_line(line)) is not None), None)
    if isin is None:
        return None

    numeric_fields = 4 if section == "equity" else 3
    values: list[Decimal] | None = None
    for line, _ in lines:
        tail = _trailing_numbers(line, numeric_fields)
        if tail is not None:
            values = tail
    if values is None:
        return None

    name_parts: list[str] = []
    folio_ref: str | None = None
    for line, _ in lines:
        folio_match = _FOLIO.search(line)
        if folio_match is not None:
            folio_ref = mask_account_ref(folio_match.group(1))
        clean = re.sub(_SYMBOL, "", line)
        clean = re.sub(r"(?i)\b(?:ISIN|UCC)\s*:?.*$", "", clean)
        clean = re.sub(r"(?i)\bFolio(?:\s+No(?:\.)?)?\s*:?\s*[A-Z0-9/-]+", "", clean)
        clean = re.sub(
            r"(?i)\b(?:Sub Total|Equity Shares|Mutual Funds|Mutual Fund Folios)\b.*", "", clean
        )
        cleaned = clean_security_name(clean, trailing_numeric_fields=numeric_fields)
        if cleaned and not re.fullmatch(r"(?i)(?:DP\s*ID|Client\s*ID|NSDL|CDSL).*", cleaned):
            name_parts.append(cleaned)
    name = clean_security_name(" ".join(name_parts))
    if not name:
        return None

    quantity = values[-3]
    price = values[-2]
    market_value = values[-1]
    if min(quantity, price, market_value) < Decimal(0):
        return None

    if section == "equity":
        kind = "equity"
    elif section == "bond":
        kind = "sgb" if re.search(r"\b(?:SGB|SOVEREIGN GOLD)\b", name, re.I) else "bond"
    elif section == "demat_mf" and re.search(r"\bETF\b", name, re.I):
        kind = "etf"
    else:
        kind = "mf"

    if warning := isin_warning(isin):
        warnings.append(warning)
    return ParsedHolding(
        kind=kind,
        isin=isin,
        name=name,
        quantity=quantity,
        market_value=market_value,
        price=price,
        cost_value=None,
        account_ref=folio_ref or account_ref,
    )


def _trailing_numbers(line: str, count: int) -> list[Decimal] | None:
    matches = list(_TRAIL_NUMBER.finditer(line))
    if len(matches) < count:
        return None
    selected = matches[-count:]
    if line[selected[-1].end() :].strip(" \t.,"):
        return None
    values = [decimal_value(match.group()) for match in selected]
    if any(value is None for value in values):
        return None
    return [value for value in values if value is not None]
