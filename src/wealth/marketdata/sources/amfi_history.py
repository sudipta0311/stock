"""Parser and bounded downloader for AMFI's official historical NAV report."""

from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import httpx

from wealth.marketdata.sources.amfi import (
    _ISIN,
    _parse_date,
    classify_amfi_scheme_name,
    classify_amfi_section,
)
from wealth.marketdata.sources.http import get

AMFI_NAV_HISTORY_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"
AMFI_HISTORY_SOURCE = "AMFI"
MAX_RANGE_DAYS = 90


@dataclass(frozen=True)
class AmfiHistoricalNav:
    scheme_code: str
    isin_primary: str | None
    isin_reinvestment: str | None
    name: str
    nav: Decimal
    as_of: date
    category: str | None
    sleeve: str | None
    sub_category: str | None
    plan_type: str | None
    option_type: str | None


def parse_nav_history(text: str) -> list[AmfiHistoricalNav]:
    """Parse AMFI's eight-column NAV history text without retaining unrelated fields."""
    if not text.strip():
        return []
    section = None
    rows: list[AmfiHistoricalNav] = []
    for line_number, line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        title = _section_title(line)
        if title is not None:
            section = classify_amfi_section(title)
            continue
        if not line.strip() or line.casefold().startswith(("scheme code;", "schemecode;")):
            continue
        fields = next(csv.reader([line], delimiter=";"), [])
        if len(fields) != 8:
            if len(fields) > 1:
                warnings.warn(
                    f"AMFI history row {line_number} was skipped", RuntimeWarning, stacklevel=2
                )
            continue
        try:
            code, name, primary_raw, reinvestment_raw, nav_raw, _repurchase, _sale, date_raw = (
                field.strip() for field in fields
            )
            if not code.isdigit() or not name:
                raise ValueError("missing scheme code or name")
            nav = Decimal(nav_raw.replace(",", ""))
            if not nav.is_finite() or nav <= 0:
                raise ValueError("invalid NAV")
            primary = _validated_isin(primary_raw)
            reinvestment = _validated_isin(reinvestment_raw)
            plan_type, option_type = classify_amfi_scheme_name(name)
            rows.append(
                AmfiHistoricalNav(
                    scheme_code=code,
                    isin_primary=primary,
                    isin_reinvestment=reinvestment,
                    name=name,
                    nav=nav,
                    as_of=_parse_date(date_raw),
                    category=section.category if section else None,
                    sleeve=section.sleeve if section else None,
                    sub_category=section.sub_category if section else None,
                    plan_type=plan_type,
                    option_type=option_type,
                )
            )
        except (InvalidOperation, ValueError):
            warnings.warn(
                f"AMFI history row {line_number} was skipped", RuntimeWarning, stacklevel=2
            )
    return rows


def fetch_nav_history(start: date, end: date) -> list[AmfiHistoricalNav]:
    """Fetch one date-bounded report; AMFI's public page limits each download to 90 days."""
    if not isinstance(start, date) or not isinstance(end, date) or end < start:
        raise ValueError("start and end must be valid dates with end on or after start")
    if (end - start).days + 1 > MAX_RANGE_DAYS:
        raise ValueError(f"AMFI history windows cannot exceed {MAX_RANGE_DAYS} days")
    query = urlencode(
        {
            "frmdt": start.strftime("%d-%b-%Y"),
            "todt": end.strftime("%d-%b-%Y"),
        }
    )
    url = f"{AMFI_NAV_HISTORY_URL}?{query}"
    try:
        response = get(url, timeout=120)
    except httpx.HTTPError:
        raise
    rows = parse_nav_history(response.text)
    return [row for row in rows if start <= row.as_of <= end]


def _section_title(line: str) -> str | None:
    match = re.match(r"^\s*(?:open|close) ended schemes\s*\((.*?)\)\s*$", line, re.I)
    return match.group(1).strip() if match else None


def _validated_isin(value: str) -> str | None:
    candidate = value.upper()
    if candidate in {"", "-"}:
        return None
    return candidate if _ISIN.fullmatch(candidate) else None
