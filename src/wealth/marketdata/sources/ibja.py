"""Best-effort parser for IBJA's published daily gold rates."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser

from wealth.marketdata.sources.http import get

IBJA_URL = "https://www.ibjarates.com/"
IBJA_SOURCE = "IBJA"


@dataclass(frozen=True)
class GoldObservation:
    inr_per_gram_24k: Decimal
    as_of: date
    source_url: str = IBJA_URL


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "tr":
            self._row = []
        elif tag.casefold() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _rate_row(row: list[str]) -> tuple[date, Decimal] | None:
    if not row:
        return None
    match = re.fullmatch(r"(\d{2}/\d{2}/\d{4})", row[0].strip())
    if match is None:
        return None
    try:
        as_of = datetime.strptime(match.group(1), "%d/%m/%Y").date()
        rate_text = next(
            (cell for cell in row[1:] if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", cell)), None
        )
        if rate_text is None:
            return None
        per_ten_grams = Decimal(rate_text.replace(",", ""))
        if not per_ten_grams.is_finite() or per_ten_grams <= 0:
            return None
        return as_of, per_ten_grams / Decimal(10)
    except (InvalidOperation, ValueError):
        return None


def parse_ibja_html(text: str, *, on: date | None = None) -> GoldObservation | None:
    """Read the latest dated 999-purity table row, expressed in rupees per gram."""
    parser = _TableParser()
    parser.feed(text)
    found = [candidate for row in parser.rows if (candidate := _rate_row(row)) is not None]
    if not found:
        if text.strip():
            warnings.warn("IBJA page has no dated 999 gold rates", RuntimeWarning, stacklevel=2)
        return None
    eligible = [row for row in found if on is None or row[0] <= on]
    if not eligible:
        return None
    latest_date = max(row[0] for row in eligible)
    # IBJA publishes opening and closing tables; the later row is the close.
    selected = [row for row in eligible if row[0] == latest_date][-1]
    return GoldObservation(selected[1], selected[0])


def fetch_gold_rate(*, on: date | None = None) -> GoldObservation | None:
    """Fetch IBJA's public rates page; a parse failure is handled by manual fallback."""
    return parse_ibja_html(get(IBJA_URL).text, on=on)
