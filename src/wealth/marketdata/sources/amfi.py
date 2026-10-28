"""AMFI NAVAll parser and adapter."""

from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import httpx

from wealth.marketdata.sources.http import get

AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
AMFI_NAV_FALLBACK_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
AMFI_SOURCE = "AMFI"
_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


@dataclass(frozen=True)
class AmfiCategory:
    category: str
    sleeve: str
    tax_note: str | None = None


@dataclass(frozen=True)
class AmfiNav:
    scheme_code: str
    isin_primary: str | None
    isin_reinvestment: str | None
    name: str
    plan: str | None
    option: str | None
    nav: Decimal
    as_of: date
    category: str | None
    sleeve: str | None
    tax_note: str | None = None


def classify_amfi_section(section: str) -> AmfiCategory | None:
    """Map an AMFI section title to the deterministic sleeve and compact category."""
    value = re.sub(r"\s+", " ", section.casefold()).strip()
    if not value:
        return None
    if "arbitrage" in value:
        return AmfiCategory("mf:arbitrage", "debt", "arbitrage funds have distinct tax treatment")
    if "gold" in value or "silver" in value:
        return AmfiCategory("mf:gold", "gold")
    if any(term in value for term in ("overseas", "international", "global", "foreign")):
        return AmfiCategory("mf:equity_global", "equity_global")
    if "liquid" in value:
        return AmfiCategory("mf:debt:liquid", "debt")
    if "gilt" in value:
        return AmfiCategory("mf:debt:gilt", "debt")
    if "debt" in value or any(
        term in value
        for term in ("bond", "money market", "overnight", "credit risk", "duration fund")
    ):
        return AmfiCategory("mf:debt", "debt")
    if any(term in value for term in ("equity", "hybrid", "index")):
        return AmfiCategory("mf:equity_india", "equity_india")
    return None


def _section_title(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.casefold().startswith(("open ended schemes(", "close ended schemes(")):
        return None
    start = stripped.find("(")
    end = stripped.rfind(")")
    return stripped[start + 1 : end] if end > start else stripped


def _parse_date(value: str) -> date:
    for pattern in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    raise ValueError("unrecognized NAV date")


def parse_nav_all(text: str) -> list[AmfiNav]:
    """Parse AMFI's semicolon-delimited NAVAll file and retain section categories."""
    if not text.strip():
        return []
    section: AmfiCategory | None = None
    rows: list[AmfiNav] = []
    for line_number, line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        if title := _section_title(line):
            section = classify_amfi_section(title)
            continue
        if not line.strip() or line.casefold().startswith("scheme code;"):
            continue
        fields = next(csv.reader([line], delimiter=";"), [])
        if len(fields) != 8:
            if len(fields) > 1:
                warnings.warn(
                    f"AMFI NAV row {line_number} was skipped", RuntimeWarning, stacklevel=2
                )
            continue
        try:
            code, isin_primary, isin_reinvestment, name, plan, option, nav_text, date_text = (
                field.strip() for field in fields
            )
            if not code.isdigit() or not name:
                raise ValueError("missing scheme code or name")
            nav = Decimal(nav_text.replace(",", ""))
            if not nav.is_finite() or nav <= 0:
                raise ValueError("invalid NAV")
            date_value = _parse_date(date_text)
            primary = isin_primary.upper() if _ISIN.fullmatch(isin_primary.upper()) else None
            reinvestment = (
                isin_reinvestment.upper() if _ISIN.fullmatch(isin_reinvestment.upper()) else None
            )
            if (isin_primary not in {"", "-"} and primary is None) or (
                isin_reinvestment not in {"", "-"} and reinvestment is None
            ):
                warnings.warn(
                    f"AMFI NAV row {line_number} has an invalid ISIN", RuntimeWarning, stacklevel=2
                )
            rows.append(
                AmfiNav(
                    scheme_code=code,
                    isin_primary=primary,
                    isin_reinvestment=reinvestment,
                    name=name,
                    plan=plan or None,
                    option=option or None,
                    nav=nav,
                    as_of=date_value,
                    category=section.category if section else None,
                    sleeve=section.sleeve if section else None,
                    tax_note=section.tax_note if section else None,
                )
            )
        except (InvalidOperation, ValueError):
            warnings.warn(f"AMFI NAV row {line_number} was skipped", RuntimeWarning, stacklevel=2)
    return rows


def fetch_nav_all() -> list[AmfiNav]:
    """Fetch AMFI NAVAll from the portal, falling back to the www host."""
    try:
        primary_text = get(AMFI_NAV_URL).text
    except httpx.HTTPError:
        return parse_nav_all(get(AMFI_NAV_FALLBACK_URL).text)
    rows = parse_nav_all(primary_text)
    if rows:
        return rows
    return parse_nav_all(get(AMFI_NAV_FALLBACK_URL).text)
