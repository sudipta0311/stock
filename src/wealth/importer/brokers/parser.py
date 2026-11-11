"""Header-driven broker statement parsing for common export formats."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import yaml
from openpyxl import load_workbook
from openpyxl.utils.datetime import WINDOWS_EPOCH, from_excel

from wealth.core.dates import add_months
from wealth.core.ids import isin_check_digit_ok
from wealth.core.money import parse_inr
from wealth.importer.brokers.types import BrokerCashFlow, BrokerHolding, Layout, Trade
from wealth.policy.lookthrough import load_lookthrough_rules

_MAX_TABLE_ROWS = 100_000
_MAX_TABLE_COLUMNS = 256
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%d.%m.%Y",
    "%d-%m-%y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%b %d, %Y",
)
_BUY_WORDS = {"buy", "b", "bought", "purchase", "køb", "købt", "kjøpt", "köp", "köpt"}
_SELL_WORDS = {"sell", "s", "sold", "sale", "salg", "solgt", "solgte", "sålt"}
_MARKET_BY_ISIN_PREFIX = {"IN": "IN", "US": "US", "DK": "DK", "GB": "GB"}
_CURRENCY_BY_MARKET = {"IN": "INR", "US": "USD", "DK": "DKK"}
_CURRENCY_CODES = {"INR", "USD", "EUR", "DKK", "GBP", "SEK"}
_MARKETS = {"IN", "US", "DK", "EU", "GB"}
# Indian grouping with more than 2 decimals, which prices and fund units often have.
_INR_LONG_FRACTION = re.compile(r"(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3})\.\d{3,}")
_ETF_WORD = re.compile(r"\betfs?\b", re.IGNORECASE)
_CURRENCY_MARKERS = re.compile(r"(?:INR|USD|EUR|DKK|GBP|SEK|NOK|RS\.?|KR\.?)", re.IGNORECASE)
_MARKET_NAMES = {
    "india": "IN",
    "indian": "IN",
    "united states": "US",
    "usa": "US",
    "us": "US",
    "denmark": "DK",
    "danish": "DK",
    "europe": "EU",
    "eu": "EU",
    "united kingdom": "GB",
    "uk": "GB",
}
_CURRENCY_NAMES = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "kr": "DKK",
    "kr.": "DKK",
}
_CURRENCY_HEADER_CODES = {
    **{currency.casefold(): currency for currency in _CURRENCY_CODES},
    "nok": "NOK",
}


def _header_currency_code(value: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    symbols = {"\u20b9": "INR", "$": "USD", "\u20ac": "EUR", "\u00a3": "GBP"}
    if normalized in symbols:
        return symbols[normalized]
    for symbol, currency in symbols.items():
        if normalized.startswith(f"{symbol} per "):
            return currency
    for code, currency in _CURRENCY_HEADER_CODES.items():
        if normalized == code or normalized.startswith(f"{code} per "):
            return currency
    if normalized in {"rs", "rs."}:
        return "INR"
    return None


def normalize_header(value: object) -> str:
    """Normalize a vendor header while retaining recognized currency units."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()

    def replace_unit(match: re.Match[str]) -> str:
        unit = match.group(1) if match.group(1) is not None else match.group(2)
        currency = _header_currency_code(unit or "")
        return f" {currency.casefold()} " if currency else " "

    text = re.sub(r"\(([^)]*)\)|\[([^]]*)\]", replace_unit, text)
    text = text.replace("₨", " ").replace("₹", " ")
    text = re.sub(r"[._]+", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"[_\s]+", " ", text)
    return text.strip()


def normalize_template_header(value: object) -> str:
    """Keep the pre-currency-unit header form used by saved templates."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", text)
    return normalize_header(text)


def _header_alias_base(value: object) -> str:
    normalized = normalize_header(value)
    currency_suffix = "|".join(sorted(_CURRENCY_HEADER_CODES, key=len, reverse=True))
    normalized = re.sub(rf"^(?:{currency_suffix})\s+", "", normalized)
    return re.sub(rf"\s+(?:{currency_suffix})$", "", normalized)


def _header_currency_unit(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"\(([^)]*)\)|\[([^]]*)\]", value)
    if match is None:
        return None
    unit = match.group(1) if match.group(1) is not None else match.group(2)
    return _header_currency_code(unit or "")


def header_fingerprint(headers: list[str] | tuple[str, ...]) -> str:
    normalized = sorted(
        {
            normalize_template_header(header)
            for header in headers
            if normalize_template_header(header)
        }
    )
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _rules_path(path: Path | None = None) -> Path:
    return path or Path(__file__).resolve().parents[4] / "rules" / "brokers.yaml"


@lru_cache(maxsize=4)
def _load_broker_document(path_text: str) -> dict[str, Any]:
    document = yaml.safe_load(Path(path_text).read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


def _broker_document(path: Path | None = None) -> dict[str, Any]:
    return _load_broker_document(str(_rules_path(path).resolve()))


def load_broker_config(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load broker labels and defaults from the versioned rules file."""
    result: dict[str, dict[str, str]] = {}
    for broker, values in _broker_document(path).get("brokers", {}).items():
        result[str(broker)] = {
            "label": str(values["label"]),
            "market": str(values["market"]).upper(),
            "currency": str(values["currency"]).upper(),
        }
    return result


def load_aliases(
    kind: str | None = None, path: Path | None = None
) -> dict[str, tuple[str, ...]] | dict[str, dict[str, tuple[str, ...]]]:
    """Return canonical column aliases grouped by import kind."""
    raw_aliases = _broker_document(path).get("aliases", {})
    kinds = (kind,) if kind else tuple(raw_aliases)
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for selected_kind in kinds:
        fields: dict[str, tuple[str, ...]] = {}
        for canonical, language_groups in raw_aliases.get(selected_kind, {}).items():
            values: list[str] = []
            for language, aliases in language_groups.items():
                if language == "fallback":
                    for fallback_aliases in aliases.values():
                        values.extend(str(alias) for alias in fallback_aliases)
                elif isinstance(aliases, list):
                    values.extend(str(alias) for alias in aliases)
            fields[str(canonical)] = tuple(dict.fromkeys(values))
        result[selected_kind] = fields
    if kind:
        return result.get(kind, {})
    return result


def _alias_entries(kind: str, canonical: str) -> tuple[tuple[str, bool], ...]:
    groups = _broker_document().get("aliases", {}).get(kind, {}).get(canonical, {})
    result: list[tuple[str, bool]] = []
    for language, aliases in groups.items():
        if language == "fallback":
            result.extend(
                (str(alias), True)
                for fallback_aliases in aliases.values()
                for alias in fallback_aliases
            )
        elif isinstance(aliases, list):
            result.extend((str(alias), False) for alias in aliases)
    return tuple(result)


def load_layouts(path: Path | None = None) -> tuple[Layout, ...]:
    document = _broker_document(path)
    broker_config = load_broker_config(path)
    result: list[Layout] = []
    for entry in document.get("layouts", []):
        columns = {
            str(field): tuple(str(alias) for alias in aliases)
            for field, aliases in entry.get("columns", {}).items()
        }
        broker_defaults = broker_config.get(entry["broker"], {})
        result.append(
            Layout(
                broker=str(entry["broker"]),
                kind=entry["kind"],
                fingerprint=frozenset(normalize_header(v) for v in entry["fingerprint"]),
                columns=columns,
                verified=bool(entry.get("verified", True)),
                source_note=str(entry.get("source_note", "")),
                market_default=str(
                    entry.get("market_default", broker_defaults.get("market", "IN"))
                ),
                currency_default=str(
                    entry.get("currency_default", broker_defaults.get("currency", "INR"))
                ),
                wins_over_picker=bool(entry.get("wins_over_picker", False)),
            )
        )
    return tuple(result)


def _layout_from_template(template: Mapping[str, Any]) -> Layout:
    columns = {
        str(field): str(header)
        for field, header in dict(template.get("columns") or {}).items()
        if header is not None
    }
    return Layout(
        broker=str(template.get("name") or "saved template"),
        kind=template["kind"],
        fingerprint=frozenset(str(value) for value in template["header_names"]),
        columns=columns,
        source_note="Saved by this household.",
        header_fingerprint=str(template["header_fingerprint"]),
        market_default=str(template.get("market_default", "IN")),
        currency_default=str(template.get("currency_default", "INR")),
    )


def detect_layout(
    headers: list[str] | tuple[str, ...],
    layouts: tuple[Layout, ...] | list[Layout] | None = None,
    templates: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
) -> Layout | None:
    normalized = {_header_alias_base(header) for header in headers}
    fingerprint = header_fingerprint(headers)
    for template in templates:
        if template.get("header_fingerprint") == fingerprint:
            return _layout_from_template(template)
    for layout in layouts if layouts is not None else load_layouts():
        if layout.fingerprint <= normalized:
            return layout
    return None


def _mapping_for_kind(
    headers: list[str] | tuple[str, ...], kind: str
) -> tuple[dict[str, str | None], set[str], set[str]]:
    result: dict[str, str | None] = {}
    ambiguous: set[str] = set()
    matched: set[str] = set()
    normalized_headers = [(_header_alias_base(header), header) for header in headers]
    for canonical in load_aliases(kind):
        alias_scores = {
            _header_alias_base(alias): (0 if fallback else 1, len(_header_alias_base(alias)))
            for alias, fallback in _alias_entries(kind, canonical)
        }
        hits: list[tuple[tuple[int, int], str]] = []
        for normalized, original in normalized_headers:
            score = alias_scores.get(normalized)
            if score is not None:
                hits.append((score, original))
        if not hits:
            result[canonical] = None
            continue
        matched.add(canonical)
        best_score = max(score for score, _header in hits)
        best_headers = list(dict.fromkeys(header for score, header in hits if score == best_score))
        if len(best_headers) != 1:
            result[canonical] = None
            ambiguous.add(canonical)
        else:
            result[canonical] = best_headers[0]
    return result, ambiguous, matched


def _required_fields(kind: str) -> set[str]:
    if kind == "holdings":
        return {"quantity"}
    return {"trade_date", "trade_type", "quantity", "price"}


def _structurally_fits(kind: str, mapping: Mapping[str, str | None], matched: set[str]) -> bool:
    required = _required_fields(kind)
    if not all(mapping.get(field) is not None for field in required):
        return False
    identity_fields = ("isin", "symbol", "name")
    return any(mapping.get(field) is not None for field in identity_fields)


def _trade_words_in_rows(rows: list[dict[str, Any]], type_header: str | None) -> tuple[int, int]:
    if not type_header:
        return 0, 0
    recognized = 0
    populated = 0
    for row in rows:
        value = _normal_trade_word(row.get(type_header))
        if not value:
            continue
        populated += 1
        if value in _BUY_WORDS or value in _SELL_WORDS:
            recognized += 1
    return recognized, populated


def _guess_kind(
    headers: list[str] | tuple[str, ...], rows: list[dict[str, Any]]
) -> tuple[str, dict[str, str | None], set[str], set[str]]:
    holdings, holdings_ambiguous, holdings_matched = _mapping_for_kind(headers, "holdings")
    trades, trades_ambiguous, trades_matched = _mapping_for_kind(headers, "tradebook")
    holdings_fit = _structurally_fits("holdings", holdings, holdings_matched)
    trades_fit = _structurally_fits("tradebook", trades, trades_matched)
    if holdings_fit and trades_fit:
        recognized, populated = _trade_words_in_rows(rows, trades.get("trade_type"))
        choose_tradebook = populated > 0 and recognized * 2 > populated
    else:
        choose_tradebook = trades_fit
    if choose_tradebook:
        return "tradebook", trades, trades_ambiguous, trades_matched
    if holdings_fit:
        return "holdings", holdings, holdings_ambiguous, holdings_matched
    if any(trades.get(field) for field in ("trade_type", "trade_date")):
        return "tradebook", trades, trades_ambiguous, trades_matched
    return "holdings", holdings, holdings_ambiguous, holdings_matched


def suggested_mapping(
    headers: list[str] | tuple[str, ...],
    kind: str | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, str | None]:
    """Suggest one header per field, preferring specific labels and exposing ties as blank."""
    selected_kind = kind or _guess_kind(headers, rows or [])[0]
    return _mapping_for_kind(headers, selected_kind)[0]


def auto_map(
    headers: list[str] | tuple[str, ...],
    rows: list[dict[str, Any]],
    *,
    broker: str,
    market_default: str,
    currency_default: str,
) -> tuple[Layout | None, str, dict[str, str | None]]:
    """Return an unambiguous layout when headers cover a safe set of required fields."""
    kind, mapping, ambiguous, matched = _guess_kind(headers, rows)
    identity_fields = {"isin", "symbol", "name"}
    required = _required_fields(kind)
    required_ambiguous = required & ambiguous
    identity_is_ambiguous = not any(mapping.get(field) for field in identity_fields) and bool(
        identity_fields & ambiguous
    )
    if (
        not _structurally_fits(kind, mapping, matched)
        or required_ambiguous
        or identity_is_ambiguous
        or ambiguous
    ):
        return None, kind, mapping
    columns = {field: header for field, header in mapping.items() if header is not None}
    return (
        Layout(
            broker=broker,
            kind=kind,  # type: ignore[arg-type]
            fingerprint=frozenset(),
            columns=columns,
            source_note="Automatically matched from column names.",
            market_default=market_default,
            currency_default=currency_default,
        ),
        kind,
        mapping,
    )


def _known_headers() -> frozenset[str]:
    known = {
        _header_alias_base(alias)
        for kind in ("holdings", "tradebook")
        for aliases in load_aliases(kind).values()
        for alias in aliases
    }
    for layout in load_layouts():
        known.update(_header_alias_base(value) for value in layout.fingerprint)
        for aliases in layout.columns.values():
            values = (aliases,) if isinstance(aliases, str) else aliases
            known.update(_header_alias_base(alias) for alias in values)
    return frozenset(known)


def _header_row(rows: list[tuple[Any, ...]]) -> int | None:
    known_headers = _known_headers()
    for index, row in enumerate(rows):
        non_empty = [value for value in row if value is not None and str(value).strip()]
        if len(non_empty) >= 3 and any(
            _header_alias_base(value) in known_headers for value in non_empty
        ):
            return index
    return None


def _decode_table_text(data: bytes) -> str | None:
    if not data:
        return None
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    else:
        sample = data[:4096]
        even_zeroes = sample[::2].count(0)
        odd_zeroes = sample[1::2].count(0)
        sample_pairs = max(1, len(sample) // 2)
        if odd_zeroes / sample_pairs > 0.2 and odd_zeroes > even_zeroes * 2:
            encodings = ("utf-16-le", "utf-8-sig", "cp1252")
        elif even_zeroes / sample_pairs > 0.2 and even_zeroes > odd_zeroes * 2:
            encodings = ("utf-16-be", "utf-8-sig", "cp1252")
        else:
            encodings = ("utf-8-sig", "cp1252")
    for encoding in encodings:
        try:
            decoded = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if all(character in "\t\n\r\f" or ord(character) >= 32 for character in decoded):
            return decoded
    return None


def _has_table_delimiter(text: str) -> bool:
    return any(
        max(line.count(delimiter) for delimiter in (",", ";", "\t")) >= 2
        for line in text.splitlines()[:20]
    )


def detect_broker_file(data: bytes, filename: str) -> str | None:
    """Shared content and extension check for upload validation and table parsing."""
    suffix = Path(filename).suffix.casefold()
    if suffix == ".pdf" and data.startswith(b"%PDF"):
        return "pdf"
    if suffix == ".xlsx" and data.startswith(b"PK"):
        return "xlsx"
    if suffix in {".csv", ".tsv", ".txt"}:
        decoded = _decode_table_text(data)
        if decoded is not None and _has_table_delimiter(decoded):
            return "text"
    if suffix == ".xls":
        if data.startswith(bytes.fromhex("D0CF11E0")):
            return "xls"
        decoded = _decode_table_text(data)
        if decoded is not None and (
            decoded.lstrip().startswith("<") or re.search(r"<\s*table\b", decoded, re.I)
        ):
            return "html_xls"
    return None


class _FirstHTMLTable(HTMLParser):
    """Collect text from the first HTML table without loading linked resources."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self._active = False
        self._finished = False
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth or self._finished:
            return
        if lowered == "table" and not self._active:
            self._active = True
        elif self._active and lowered == "tr" and self._row is None:
            self._row = []
        elif self._active and lowered in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth or not self._active:
            return
        if lowered in {"td", "th"} and self._cell is not None and self._row is not None:
            if len(self._row) >= _MAX_TABLE_COLUMNS:
                raise ValueError("The table has too many columns.")
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            if len(self.rows) >= _MAX_TABLE_ROWS:
                raise ValueError("The table has too many rows.")
            self.rows.append(tuple(self._row))
            self._row = None
        elif lowered == "table":
            self._active = False
            self._finished = True

    def handle_data(self, data: str) -> None:
        if self._cell is not None and self._ignored_depth == 0:
            self._cell.append(data)


def _rows_from_xlsx(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except (BadZipFile, OSError, ValueError) as error:
        raise ValueError("The XLSX workbook could not be read.") from error
    try:
        for worksheet in workbook.worksheets:
            # Some exports (INDmoney) declare the sheet as just A1, and read-only mode
            # trusts that, so it would never reach the header row.
            worksheet.reset_dimensions()
            values: list[tuple[Any, ...]] = []
            for row in worksheet.iter_rows(values_only=True):
                if len(row) > _MAX_TABLE_COLUMNS:
                    raise ValueError("The table has too many columns.")
                if len(values) >= _MAX_TABLE_ROWS:
                    raise ValueError("The table has too many rows.")
                values.append(tuple(row))
            row_index = _header_row(values)
            if row_index is not None:
                return _rows_from_values(values, row_index)
    finally:
        workbook.close()
    raise ValueError("No worksheet contains a recognizable header row.")


def _rows_from_xls(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import xlrd
    except ImportError as error:
        raise ValueError("The XLS reader is not installed.") from error
    try:
        workbook = xlrd.open_workbook(file_contents=data, on_demand=True)
    except (OSError, ValueError, xlrd.XLRDError) as error:
        raise ValueError("The XLS workbook could not be read.") from error
    try:
        for sheet in workbook.sheets():
            if sheet.ncols > _MAX_TABLE_COLUMNS or sheet.nrows > _MAX_TABLE_ROWS:
                raise ValueError("The table exceeds the supported row or column limit.")
            values: list[tuple[Any, ...]] = []
            for row_index in range(sheet.nrows):
                cells: list[Any] = []
                for column_index in range(sheet.ncols):
                    cell = sheet.cell(row_index, column_index)
                    if cell.ctype == xlrd.XL_CELL_DATE:
                        cells.append(xlrd.xldate_as_datetime(cell.value, workbook.datemode))
                    else:
                        cells.append(cell.value)
                values.append(tuple(cells))
            header_index = _header_row(values)
            if header_index is not None:
                return _rows_from_values(values, header_index)
    finally:
        workbook.release_resources()
    raise ValueError("No worksheet contains a recognizable header row.")


def read_table(data: bytes, filename: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a text table, HTML XLS export, XLSX, or cached-value XLS workbook."""
    file_type = detect_broker_file(data, filename)
    if file_type == "xlsx":
        return _rows_from_xlsx(data)
    if file_type == "xls":
        return _rows_from_xls(data)
    if file_type == "html_xls":
        decoded = _decode_table_text(data)
        if decoded is None:
            raise ValueError("The HTML table encoding is not supported.")
        parser = _FirstHTMLTable()
        parser.feed(decoded)
        parser.close()
        if not parser.rows:
            raise ValueError("The HTML file does not contain a table.")
        row_index = _header_row(parser.rows)
        if row_index is None:
            raise ValueError("No recognizable header row was found.")
        return _rows_from_values(parser.rows, row_index)
    if file_type != "text":
        raise ValueError("The file does not match a supported broker table format.")
    decoded = _decode_table_text(data)
    if decoded is None:
        raise ValueError("The table encoding is not supported.")
    try:
        dialect = csv.Sniffer().sniff(decoded[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel_tab if Path(filename).suffix.casefold() == ".tsv" else csv.excel
    rows: list[tuple[Any, ...]] = []
    for row in csv.reader(io.StringIO(decoded, newline=""), dialect=dialect):
        if len(row) > _MAX_TABLE_COLUMNS:
            raise ValueError("The table has too many columns.")
        if len(rows) >= _MAX_TABLE_ROWS:
            raise ValueError("The table has too many rows.")
        rows.append(tuple(row))
    row_index = _header_row(rows)
    if row_index is None:
        if rows and len([cell for cell in rows[0] if cell.strip()]) >= 3:
            row_index = 0
        else:
            raise ValueError("No recognizable header row was found.")
    return _rows_from_values(rows, row_index)


def _rows_from_values(
    values: list[tuple[Any, ...]], row_index: int
) -> tuple[list[str], list[dict[str, Any]]]:
    raw_headers = list(values[row_index])
    while raw_headers and (raw_headers[-1] is None or not str(raw_headers[-1]).strip()):
        raw_headers.pop()
    if len(raw_headers) > _MAX_TABLE_COLUMNS:
        raise ValueError("The table has too many columns.")
    headers = [str(value or "").strip() for value in raw_headers]
    if len(values) - row_index - 1 > _MAX_TABLE_ROWS:
        raise ValueError("The table has too many rows.")
    rows: list[dict[str, Any]] = []
    for row in values[row_index + 1 :]:
        if not row or not any(value is not None and str(value).strip() for value in row):
            continue
        rows.append(
            {
                headers[index]: row[index] if index < len(row) else None
                for index in range(len(headers))
            }
        )
    return headers, rows


def _column_header(row: Mapping[str, Any], layout: Layout, canonical: str) -> str | None:
    configured = layout.columns.get(canonical)
    if configured is None:
        return None
    aliases = (configured,) if isinstance(configured, str) else configured
    row_headers = [(normalize_header(header), _header_alias_base(header), header) for header in row]
    for alias in aliases:
        key = normalize_header(alias)
        for normalized, _base, header in row_headers:
            if normalized == key:
                return header
    for alias in aliases:
        base = _header_alias_base(alias)
        for _normalized, row_base, header in row_headers:
            if row_base == base:
                return header
    return None


def _column_value(row: Mapping[str, Any], layout: Layout, canonical: str) -> Any:
    header = _column_header(row, layout, canonical)
    value = row.get(header) if header is not None else None
    return None if isinstance(value, str) and value.strip() == "-" else value


def _decimal_value(value: Any, *, field: str) -> Decimal:
    return _parse_decimal(value, field=field, currency="INR")


def _currency_decimal(value: Any, *, field: str, currency: str) -> Decimal:
    """Read a broker amount while keeping INR grouping strict."""
    return _parse_decimal(value, field=field, currency=currency)


def _parse_decimal(value: Any, *, field: str, currency: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a number")
    raw = str(value).strip().replace("−", "-")
    if not raw:
        raise ValueError(f"{field} must be a number")
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1].strip()
    raw = raw.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    for marker in ("₹", "€", "$", "£"):
        raw = raw.replace(marker, "")
    raw = _CURRENCY_MARKERS.sub("", raw)
    raw = raw.replace("−", "-")
    if raw.startswith("-"):
        negative = not negative
        raw = raw[1:]
    if raw.startswith("+"):
        raw = raw[1:]
    if not raw or "-" in raw or "+" in raw:
        raise ValueError(f"{field} must be a number")

    if currency == "INR":
        try:
            number = parse_inr(raw)
        except ValueError as error:
            # parse_inr stops at paise; an average price like 1543.2367 is still exact.
            if _INR_LONG_FRACTION.fullmatch(raw) is None:
                raise ValueError(f"{field} must be a valid INR amount") from error
            number = Decimal(raw.replace(",", ""))
        return -number if negative else number

    if "," in raw and "." in raw:
        decimal_separator = "," if raw.rfind(",") > raw.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        integer, fraction = raw.rsplit(decimal_separator, 1)
        if not fraction.isdigit():
            raise ValueError(f"{field} must be a number")
        groups = integer.split(grouping_separator)
        if len(groups) > 1 and not _valid_grouping(groups, grouping_separator):
            raise ValueError(f"{field} must be a number")
        raw = "".join(groups) + "." + fraction
    elif "," in raw:
        left, _, right = raw.rpartition(",")
        groups = left.split(",")
        if right.isdigit() and _valid_grouping([*groups, right], ","):
            raw = "".join([*groups, right])
        elif right.isdigit() and len(right) != 3 and "," not in left:
            raw = left + "." + right
        else:
            raise ValueError(f"{field} must be a number")
    elif raw.count(".") > 1:
        groups = raw.split(".")
        if not _valid_grouping(groups, "."):
            raise ValueError(f"{field} must be a number")
        raw = "".join(groups)
    if not re.fullmatch(r"\d+(?:\.\d+)?", raw):
        raise ValueError(f"{field} must be a number")
    try:
        number = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"{field} must be a number") from error
    if not number.is_finite():
        raise ValueError(f"{field} must be finite") from None
    return -number if negative else number


def _valid_grouping(groups: list[str], separator: str) -> bool:
    if len(groups) < 2 or not all(group.isdigit() for group in groups):
        return False
    if separator == "," and len(groups) > 2 and all(len(group) == 2 for group in groups[1:-1]):
        return len(groups[0]) in {1, 2} and len(groups[-1]) == 3
    return len(groups[0]) in {1, 2, 3} and all(len(group) == 3 for group in groups[1:])


def _date_value(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            excel_date = from_excel(float(value), WINDOWS_EPOCH)
        except (OverflowError, TypeError, ValueError):
            excel_date = None
        if isinstance(excel_date, datetime):
            return excel_date.date()
        if isinstance(excel_date, date):
            return excel_date
    text = str(value or "").strip()
    for pattern in _DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError("trade date must use a supported date format")


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _normal_trade_word(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[^\w]+", " ", text).strip()


def _market_without_default(
    row: Mapping[str, Any],
    layout: Layout,
    isin: str | None,
    default_market: str | None = None,
) -> str | None:
    raw_market = _text(_column_value(row, layout, "market"))
    if raw_market:
        normalized_market = raw_market.strip().casefold()
        market = _MARKET_NAMES.get(normalized_market, raw_market.upper())
        if market in _MARKETS:
            return market
    if isin and len(isin) == 12 and isin_check_digit_ok(isin.upper()):
        market = _MARKET_BY_ISIN_PREFIX.get(isin[:2].upper())
        if market is not None:
            return market
        # An ISIN's country is where a company is registered, not where it trades:
        # Accenture (IE...) trades in New York, so an Indian or US broker keeps its market.
        if default_market in {"IN", "US"}:
            return default_market
        return "EU"
    return None


def mapping_default_requirements(
    rows: list[dict[str, Any]], columns: Mapping[str, str | None]
) -> tuple[bool, bool]:
    """Report whether any mapped row can supply market and currency itself."""
    layout = Layout(
        broker="mapped",
        kind="holdings",
        fingerprint=frozenset(),
        columns={field: header for field, header in columns.items() if header is not None},
    )
    market_inferred = False
    currency_inferred = False
    for row in rows:
        isin = _text(_column_value(row, layout, "isin"))
        market = _market_without_default(row, layout, isin)
        market_inferred = market_inferred or market is not None
        explicit_currency = _text(_column_value(row, layout, "currency"))
        if explicit_currency:
            normalized_currency = explicit_currency.strip().casefold()
            currency = _CURRENCY_NAMES.get(normalized_currency, explicit_currency.upper())
            currency_inferred = currency_inferred or currency in _CURRENCY_CODES
        if market in _CURRENCY_BY_MARKET:
            currency_inferred = True
    return not market_inferred, not currency_inferred


def _row_market_currency(
    row: Mapping[str, Any], layout: Layout, isin: str | None
) -> tuple[str, str] | None:
    default_market = layout.market_default.upper()
    market = _market_without_default(row, layout, isin, default_market) or default_market
    if market not in _MARKETS:
        return None

    explicit_currency = _text(_column_value(row, layout, "currency"))
    if explicit_currency:
        normalized_currency = explicit_currency.strip().casefold()
        currency = _CURRENCY_NAMES.get(normalized_currency, explicit_currency.upper())
        if currency not in _CURRENCY_CODES:
            return market, currency
    else:
        currency = ""
    if currency not in _CURRENCY_CODES:
        currency = _CURRENCY_BY_MARKET.get(market, layout.currency_default.upper())
    if currency not in _CURRENCY_CODES:
        return None
    return market, currency


def _security_type(
    row: Mapping[str, Any],
    layout: Layout,
    *,
    market: str,
    symbol: str | None,
    isin: str | None,
) -> str:
    """File a row as an ETF from its type column, its name, or the look-through fund lists."""
    if isin and isin.upper().startswith("INF"):
        return "etf"
    raw_type = (_text(_column_value(row, layout, "security_type")) or "").casefold()
    if raw_type in {"etf", "etfs"}:
        return "etf"
    # Exports like INDmoney's have no type column; VOO and QQQM only show it in the name.
    name = _text(_column_value(row, layout, "name"))
    if name and _ETF_WORD.search(name):
        return "etf"
    rules = load_lookthrough_rules()
    if market == "US" and symbol and symbol.upper() in rules.get("by_us_symbol", {}):
        return "etf"
    if isin and isin.upper() in rules.get("_curated_by_isin", {}):
        return "etf"
    return "equity"


def _clean_broker_symbol(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    symbol_text = value.strip()
    exchange_hint: str | None = None
    if ":" in symbol_text:
        symbol_text, raw_exchange = symbol_text.split(":", 1)
        exchange_hint = re.sub(r"\s+", "", raw_exchange).upper() or None
    symbol = re.sub(r"\s+", "", symbol_text).upper()
    return symbol or None, exchange_hint


def _currency_mismatch(
    row: Mapping[str, Any], layout: Layout, canonical: str, currency: str
) -> bool:
    header = _column_header(row, layout, canonical)
    unit = _header_currency_unit(header)
    return unit is not None and unit != currency


def _record_currency_mismatch(mismatches: dict[str, int] | None, canonical: str) -> None:
    if mismatches is not None:
        mismatches[canonical] = mismatches.get(canonical, 0) + 1


def _is_summary(row: Mapping[str, Any], layout: Layout) -> bool:
    values = [str(value).strip().casefold() for value in row.values() if value is not None]
    return not any(values) or any(
        value in {"total", "grand total", "summary", "subtotal"} for value in values
    )


def parse_holdings(
    rows: list[dict[str, Any]],
    layout: Layout,
    *,
    currency_mismatches: dict[str, int] | None = None,
) -> list[BrokerHolding]:
    if layout.kind != "holdings":
        raise ValueError("A holdings layout is required.")
    result: list[BrokerHolding] = []
    for row in rows:
        if _is_summary(row, layout):
            continue
        isin_text = _text(_column_value(row, layout, "isin"))
        market_currency = _row_market_currency(row, layout, isin_text)
        if market_currency is None:
            continue
        market, currency = market_currency
        try:
            quantity = _currency_decimal(
                _column_value(row, layout, "quantity"), field="quantity", currency=currency
            )
        except ValueError:
            continue
        if quantity <= 0:
            continue
        raw_avg_price = _column_value(row, layout, "avg_price")
        raw_cost_value = _column_value(row, layout, "cost_value")
        avg_price: Decimal | None = None
        if raw_avg_price not in (None, "") and _currency_mismatch(
            row, layout, "avg_price", currency
        ):
            _record_currency_mismatch(currency_mismatches, "avg_price")
        elif raw_avg_price not in (None, ""):
            with suppress(ValueError):
                avg_price = _currency_decimal(
                    raw_avg_price, field="average price", currency=currency
                )
        elif raw_cost_value not in (None, ""):
            if _currency_mismatch(row, layout, "cost_value", currency):
                _record_currency_mismatch(currency_mismatches, "cost_value")
            else:
                with suppress(ValueError):
                    cost_value = _currency_decimal(
                        raw_cost_value, field="cost value", currency=currency
                    )
                    avg_price = cost_value / quantity
        if avg_price is not None and avg_price < 0:
            continue
        symbol = _text(_column_value(row, layout, "symbol"))
        name = _text(_column_value(row, layout, "name"))
        if (symbol or name or _text(_column_value(row, layout, "isin"))) is None:
            continue
        last_price = None
        raw_last = _column_value(row, layout, "last_price")
        if raw_last not in (None, ""):
            if _currency_mismatch(row, layout, "last_price", currency):
                _record_currency_mismatch(currency_mismatches, "last_price")
            else:
                with suppress(ValueError):
                    last_price = _currency_decimal(raw_last, field="last price", currency=currency)
        parsed_market_value = None
        raw_market_value = _column_value(row, layout, "market_value")
        if raw_market_value not in (None, ""):
            if _currency_mismatch(row, layout, "market_value", currency):
                _record_currency_mismatch(currency_mismatches, "market_value")
            else:
                with suppress(ValueError):
                    parsed_market_value = _currency_decimal(
                        raw_market_value, field="market value", currency=currency
                    )
        isin = _text(_column_value(row, layout, "isin"))
        symbol, exchange_hint = _clean_broker_symbol(_text(_column_value(row, layout, "symbol")))
        exchange = _text(_column_value(row, layout, "exchange")) or exchange_hint
        result.append(
            BrokerHolding(
                isin=isin.upper() if isin else None,
                symbol=symbol,
                name=name,
                quantity=quantity,
                avg_price=avg_price,
                last_price=last_price,
                exchange=exchange,
                market=market,
                currency=currency,
                market_value=parsed_market_value,
                security_type=_security_type(row, layout, market=market, symbol=symbol, isin=isin),
            )
        )
    return result


def parse_tradebook(
    rows: list[dict[str, Any]],
    layout: Layout,
    *,
    skipped_rows: dict[str, int] | None = None,
) -> list[Trade]:
    if layout.kind != "tradebook":
        raise ValueError("A tradebook layout is required.")
    result: list[Trade] = []
    for row in rows:
        if _is_summary(row, layout):
            continue
        raw_type = _normal_trade_word(_column_value(row, layout, "trade_type"))
        if raw_type in _BUY_WORDS:
            trade_type = "buy"
        elif raw_type in _SELL_WORDS:
            trade_type = "sell"
        else:
            if skipped_rows is not None:
                skipped_rows["not_a_trade"] = skipped_rows.get("not_a_trade", 0) + 1
            continue
        isin_text = _text(_column_value(row, layout, "isin"))
        market_currency = _row_market_currency(row, layout, isin_text)
        if market_currency is None:
            continue
        market, currency = market_currency
        try:
            trade_date = _date_value(_column_value(row, layout, "trade_date"))
            quantity = abs(
                _currency_decimal(
                    _column_value(row, layout, "quantity"), field="quantity", currency=currency
                )
            )
            price = _currency_decimal(
                _column_value(row, layout, "price"), field="price", currency=currency
            )
            raw_commission = _column_value(row, layout, "commission")
            commission = (
                _currency_decimal(raw_commission, field="commission", currency=currency)
                if raw_commission not in (None, "")
                else Decimal(0)
            )
            raw_proceeds = _column_value(row, layout, "amount")
            proceeds = (
                _currency_decimal(raw_proceeds, field="proceeds", currency=currency)
                if raw_proceeds not in (None, "")
                else None
            )
        except ValueError:
            continue
        if quantity <= 0 or price < 0:
            continue
        isin = _text(_column_value(row, layout, "isin"))
        symbol, exchange_hint = _clean_broker_symbol(_text(_column_value(row, layout, "symbol")))
        name = _text(_column_value(row, layout, "name"))
        if isin is None and symbol is None and name is None:
            continue
        exchange = _text(_column_value(row, layout, "exchange")) or exchange_hint
        result.append(
            Trade(
                isin=isin.upper() if isin else None,
                symbol=symbol,
                trade_date=trade_date,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                name=name,
                exchange=exchange,
                order_id=_text(_column_value(row, layout, "order_id")),
                trade_id=_text(_column_value(row, layout, "trade_id")),
                market=market,
                currency=currency,
                commission=commission,
                proceeds=proceeds,
                security_type=_security_type(row, layout, market=market, symbol=symbol, isin=isin),
            )
        )
    return result


def parse_cash_flows(rows: list[dict[str, Any]], layout: Layout) -> list[BrokerCashFlow]:
    """Extract only explicit cash deposits and withdrawals from mapped transaction rows."""
    result: list[BrokerCashFlow] = []
    fingerprint_counts: dict[str, int] = {}
    for row in rows:
        raw_kind = str(_column_value(row, layout, "trade_type") or "").strip().casefold()
        if any(word in raw_kind for word in ("deposit", "contribution", "transfer in")):
            kind = "deposit"
        elif any(word in raw_kind for word in ("withdraw", "transfer out")):
            kind = "withdrawal"
        else:
            continue
        currency = (
            _text(_column_value(row, layout, "currency")) or layout.currency_default
        ).upper()
        if currency not in {"INR", "USD", "EUR", "DKK", "GBP", "SEK"}:
            continue
        try:
            transaction_date = _date_value(_column_value(row, layout, "trade_date"))
            amount = abs(
                _currency_decimal(
                    _column_value(row, layout, "amount"), field="amount", currency=currency
                )
            )
        except ValueError:
            continue
        if amount <= 0:
            continue
        transaction_id = _text(_column_value(row, layout, "trade_id"))
        if transaction_id is not None:
            transaction_id = transaction_id[:80]
        if transaction_id is None:
            row_text = "\x1f".join(
                f"{normalize_header(key)}={str(value or '').strip()}"
                for key, value in sorted(row.items(), key=lambda item: normalize_header(item[0]))
            )
            fingerprint = hashlib.sha256(row_text.encode("utf-8")).hexdigest()[:32]
            fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1
            transaction_id = f"row-{fingerprint}-{fingerprint_counts[fingerprint]}"
        result.append(
            BrokerCashFlow(
                transaction_date=transaction_date,
                kind=kind,
                amount=amount,
                currency=currency,
                transaction_id=transaction_id,
            )
        )
    return result


def is_long_term(acquired_on: date, sold_on: date, *, months: int = 12) -> bool:
    """Apply the configured listed or foreign equity holding period."""
    return sold_on > add_months(acquired_on, months)
