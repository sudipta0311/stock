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
    sub_category: str | None = None


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
    sub_category: str | None = None
    plan_type: str | None = None
    option_type: str | None = None


def classify_amfi_scheme_name(
    name: str, plan_field: str | None = None, option_field: str | None = None
) -> tuple[str | None, str | None]:
    """Return normalized plan and payout option from a scheme name or AMFI fields."""
    value = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    plan_value = re.sub(r"[^a-z0-9]+", " ", (plan_field or "").casefold()).strip()
    option_value = re.sub(r"[^a-z0-9]+", " ", (option_field or "").casefold()).strip()
    if re.search(r"\bdirect(?: plan)?\b", value):
        plan_type = "direct"
    elif re.search(r"\bregular(?: plan)?\b", value):
        plan_type = "regular"
    elif "direct" in plan_value:
        plan_type = "direct"
    elif "regular" in plan_value:
        plan_type = "regular"
    else:
        plan_type = None

    if re.search(r"\b(idcw|dividend|payout|reinvestment)\b", value):
        option_type = "idcw"
    elif re.search(r"\bgrowth\b", value):
        option_type = "growth"
    elif re.search(r"\b(idcw|dividend|payout|reinvestment)\b", option_value):
        option_type = "idcw"
    elif "growth" in option_value:
        option_type = "growth"
    else:
        option_type = None
    return plan_type, option_type


def classify_amfi_section(section: str) -> AmfiCategory | None:
    """Map an AMFI section title to the deterministic sleeve and compact category."""
    value = re.sub(r"\s+", " ", section.casefold()).strip()
    if not value:
        return None
    if any(term in value for term in ("overseas", "international", "global", "foreign")):
        return AmfiCategory("mf:equity_global", "equity_global", sub_category="FoF Overseas")
    if "arbitrage" in value:
        return AmfiCategory(
            "mf:arbitrage",
            "debt",
            "arbitrage funds have distinct tax treatment",
            "Arbitrage",
        )
    if "gold" in value or "silver" in value:
        return AmfiCategory("mf:gold", "gold", sub_category="Gold")
    if "liquid" in value:
        return AmfiCategory("mf:debt:liquid", "debt", sub_category="Liquid")
    if "gilt" in value:
        return AmfiCategory("mf:debt:gilt", "debt", sub_category="Gilt")
    if "debt" in value or any(
        term in value
        for term in ("bond", "money market", "overnight", "credit risk", "duration fund")
    ):
        return AmfiCategory("mf:debt", "debt", sub_category="Other Debt")
    sub_category = _equity_sub_category(value)
    if sub_category is not None:
        return AmfiCategory("mf:equity_india", "equity_india", sub_category=sub_category)
    if any(term in value for term in ("equity", "hybrid", "index")):
        return AmfiCategory("mf:equity_india", "equity_india", sub_category="Other Equity")
    return None


def _equity_sub_category(value: str) -> str | None:
    if "sectoral" in value or "thematic" in value:
        return "Sectoral/Thematic"
    if "large" in value and ("mid" in value or "&" in value):
        return "Large & Mid Cap"
    for term, label in (
        ("large cap", "Large Cap"),
        ("mid cap", "Mid Cap"),
        ("small cap", "Small Cap"),
        ("flexi cap", "Flexi Cap"),
        ("multi cap", "Multi Cap"),
        ("elss", "ELSS"),
        ("tax saver", "ELSS"),
    ):
        if term in value:
            return label
    if "index" in value or "etf" in value:
        return "Index Funds/ETFs"
    if "hybrid" in value or "balanced" in value or "asset allocation" in value:
        if "aggressive" in value:
            return "Aggressive Hybrid"
        if "conservative" in value:
            return "Conservative Hybrid"
        return "Balanced Hybrid"
    return None


def _section_title(line: str) -> str | None:
    stripped = line.strip()
    if not re.match(r"^(?:open|close) ended schemes\s*\(", stripped, re.IGNORECASE):
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
                    sub_category=section.sub_category if section else None,
                    plan_type=classify_amfi_scheme_name(name, plan, option)[0],
                    option_type=classify_amfi_scheme_name(name, plan, option)[1],
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
