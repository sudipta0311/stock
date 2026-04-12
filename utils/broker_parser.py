from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional until dependency is installed
    pdfplumber = None


BROKER_COLUMN_ALIASES = {
    "symbol": [
        "symbol",
        "scrip",
        "stock",
        "ticker",
        "instrument",
        "scrip name",
        "stock name",
        "trading symbol",
    ],
    "quantity": ["qty", "quantity", "shares", "units", "holdings"],
    "avg_price": [
        "avg price",
        "average price",
        "buy price",
        "cost price",
        "avg cost",
        "purchase price",
        "acquisition price",
        "avg. price",
        "avg buy price",
    ],
    "current_price": ["ltp", "last price", "current price", "market price", "close price", "cmp"],
    "buy_date": ["buy date", "purchase date", "date of purchase", "acquisition date", "date"],
}


def _canonical_column(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def _normalise_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    symbol = symbol.replace(".NS", "").replace(".BO", "").replace("-EQ", "")
    return symbol


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    if cleaned in {"", "-", ".", "-."}:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: dict[str, str] = {}
    canonical = {_canonical_column(col): col for col in df.columns}
    for standard, aliases in BROKER_COLUMN_ALIASES.items():
        for alias in aliases:
            original = canonical.get(alias)
            if original is not None:
                col_map[original] = standard
                break
    return df.rename(columns=col_map)


def _extract_holdings(df: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    df = df.copy()
    df.columns = [_canonical_column(c) for c in df.columns]
    df = normalise_columns(df)
    required = ["symbol", "quantity", "avg_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print(f"Available: {list(df.columns)}")
        return []

    holdings: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        symbol = _normalise_symbol(row.get("symbol"))
        if not symbol or symbol == "NAN":
            continue
        holding = {
            "symbol": symbol,
            "quantity": _to_float(row.get("quantity")),
            "avg_buy_price": _to_float(row.get("avg_price")),
            "current_price": (
                _to_float(row.get("current_price"), default=0.0) if "current_price" in df.columns else None
            ),
            "buy_date": str(row.get("buy_date", "unknown") or "unknown").strip(),
            "source": source,
        }
        if holding["quantity"] > 0 and holding["avg_buy_price"] > 0:
            holdings.append(holding)
    print(f"Parsed {len(holdings)} holdings from {source}")
    return holdings


def parse_broker_csv(file_path: str | Path) -> list[dict[str, Any]]:
    try:
        df = pd.read_csv(file_path)
        return _extract_holdings(df, source="broker_csv")
    except Exception as exc:
        print(f"Broker CSV parse error: {exc}")
        return []


def _table_to_frame(table: list[list[Any]]) -> pd.DataFrame:
    cleaned_rows = [[str(cell or "").strip() for cell in row] for row in table if row and any(cell for cell in row)]
    if len(cleaned_rows) < 2:
        return pd.DataFrame()
    header = cleaned_rows[0]
    body = cleaned_rows[1:]
    return pd.DataFrame(body, columns=header)


def _extract_pdf_tables(file_path: str | Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if pdfplumber is None:
        print("pdfplumber is not installed; broker PDF parsing unavailable")
        return frames
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                frame = _table_to_frame(table)
                if not frame.empty:
                    frames.append(frame)
    return frames


def _extract_pdf_text_rows(file_path: str | Path) -> pd.DataFrame:
    if pdfplumber is None:
        return pd.DataFrame()
    pattern = re.compile(
        r"(?P<symbol>[A-Z][A-Z0-9&.\-]{1,20})\s+"
        r"(?P<quantity>\d[\d,]*(?:\.\d+)?)\s+"
        r"(?P<avg_price>\d[\d,]*(?:\.\d+)?)"
        r"(?:\s+(?P<current_price>\d[\d,]*(?:\.\d+)?))?"
    )
    rows: list[dict[str, Any]] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                match = pattern.search(line.upper())
                if not match:
                    continue
                rows.append(match.groupdict())
    return pd.DataFrame(rows)


def parse_broker_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    try:
        for frame in _extract_pdf_tables(file_path):
            holdings = _extract_holdings(frame, source="broker_pdf")
            if holdings:
                return holdings
        fallback = _extract_pdf_text_rows(file_path)
        return _extract_holdings(fallback, source="broker_pdf")
    except Exception as exc:
        print(f"Broker PDF parse error: {exc}")
        return []


def parse_broker_file(file_path: str | Path) -> list[dict[str, Any]]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return parse_broker_csv(file_path)
    if suffix == ".pdf":
        return parse_broker_pdf(file_path)
    print(f"Unsupported broker statement type: {suffix}")
    return []


def save_broker_holdings_to_db(holdings: list[dict[str, Any]], db_path: str | Path) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS direct_equity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE,
            quantity REAL,
            avg_buy_price REAL,
            current_price REAL,
            buy_date TEXT,
            source TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    saved = 0
    for holding in holdings:
        conn.execute(
            """
            INSERT INTO direct_equity
                (symbol, quantity, avg_buy_price, current_price, buy_date, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = excluded.quantity,
                avg_buy_price = excluded.avg_buy_price,
                current_price = excluded.current_price,
                buy_date = excluded.buy_date,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                _normalise_symbol(holding["symbol"]),
                _to_float(holding.get("quantity")),
                _to_float(holding.get("avg_buy_price")),
                holding.get("current_price"),
                str(holding.get("buy_date", "unknown") or "unknown"),
                holding.get("source", "broker_csv"),
            ),
        )
        saved += 1
    conn.commit()
    conn.close()
    return saved
