"""Pure parser for the supported sections of an IBKR activity CSV."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from wealth.core.ids import isin_check_digit_ok
from wealth.importer.brokers.parser import normalize_header
from wealth.importer.brokers.types import BrokerHolding, Trade

_SUPPORTED_CATEGORIES = {"stocks", "etfs"}
_SUPPORTED_CURRENCIES = {"USD", "EUR", "DKK", "GBP", "SEK"}


@dataclass(frozen=True)
class IBKRActivity:
    positions: tuple[BrokerHolding, ...]
    trades: tuple[Trade, ...]
    warnings: tuple[str, ...]


def is_activity_statement_csv(data: bytes | str) -> bool:
    """Identify only the known sectioned statement shape."""
    if isinstance(data, bytes):
        try:
            raw = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                raw = data.decode("cp1252")
            except UnicodeDecodeError:
                return False
    else:
        raw = data
    sections = {
        (normalize_header(row[0]), normalize_header(row[1]))
        for row in csv.reader(io.StringIO(raw))
        if len(row) > 1
    }
    return ("open positions", "header") in sections and ("trades", "header") in sections


def _decimal(value: str | None, *, field: str) -> Decimal:
    raw = (value or "").strip().replace(",", "")
    if not raw:
        raise ValueError(f"{field} is missing")
    try:
        result = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"{field} is invalid") from error
    if not result.is_finite():
        raise ValueError(f"{field} is invalid")
    return result


def _date(value: str | None) -> date:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("trade date is missing")
    iso_candidate = raw[:10]
    try:
        return date.fromisoformat(iso_candidate)
    except ValueError:
        pass
    for pattern in ("%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(iso_candidate, pattern).date()
        except ValueError:
            continue
    raise ValueError("trade date is invalid")


def _row(section_headers: dict[str, list[str]], section: str, values: list[str]) -> dict[str, str]:
    headers = section_headers.get(section, [])
    return {
        normalize_header(header): value.strip()
        for header, value in zip(headers, values, strict=False)
    }


def _get(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = row.get(normalize_header(name))
        if value:
            return value
    return None


def _isin(row: dict[str, str]) -> str | None:
    direct = _get(row, "isin")
    security_id = _get(row, "security id")
    security_id_type = (_get(row, "security id type") or "").casefold()
    candidate = (direct or (security_id if security_id_type == "isin" else None) or "").upper()
    return candidate if isin_check_digit_ok(candidate) else None


def _market(exchange: str | None, currency: str) -> str:
    venue = (exchange or "").strip().upper()
    if any(token in venue for token in ("COPENHAGEN", "CPH", "XCSE")):
        return "DK"
    if any(token in venue for token in ("XETRA", "IBIS", "FWB", "EURONEXT")):
        return "EU"
    if "LSE" in venue or "LONDON" in venue:
        return "GB"
    if venue or currency == "USD":
        return "US" if currency == "USD" else "EU"
    return "EU"


def parse_activity_statement_csv(data: bytes | str) -> IBKRActivity:
    """Parse positions, trades, and instrument information from a multi-section CSV."""
    if isinstance(data, bytes):
        decoded = None
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                decoded = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            raise ValueError("The CSV encoding is not supported.")
    else:
        decoded = data.lstrip("\ufeff")

    section_headers: dict[str, list[str]] = {}
    section_rows: dict[str, list[dict[str, str]]] = {}
    warnings: set[str] = set()
    recognized = {"open positions", "trades", "financial instrument information"}
    for values in csv.reader(io.StringIO(decoded, newline="")):
        if len(values) < 2:
            continue
        section = normalize_header(values[0])
        record_type = normalize_header(values[1])
        if not section or record_type not in {"header", "data", "subtotal", "total"}:
            continue
        if section not in recognized:
            if record_type == "data":
                warnings.add("unknown_section")
            continue
        if record_type == "header":
            section_headers[section] = values[2:]
        elif record_type == "data":
            section_rows.setdefault(section, []).append(_row(section_headers, section, values[2:]))

    instruments: dict[str, dict[str, str]] = {}
    for row in section_rows.get("financial instrument information", []):
        symbol = (_get(row, "symbol") or "").upper()
        if symbol:
            instruments[symbol] = row

    positions: list[BrokerHolding] = []
    for row in section_rows.get("open positions", []):
        category = (_get(row, "asset category") or "").casefold()
        if category not in _SUPPORTED_CATEGORIES:
            warnings.add("unsupported_asset_category")
            continue
        symbol = (_get(row, "symbol") or "").upper()
        currency = (_get(row, "currency") or "").upper()
        if not symbol or currency not in _SUPPORTED_CURRENCIES:
            if currency and currency not in _SUPPORTED_CURRENCIES:
                warnings.add("unsupported_currency")
            continue
        instrument = instruments.get(symbol, {})
        exchange = _get(row, "listing exchange", "exchange") or _get(
            instrument, "listing exchange", "exchange"
        )
        try:
            quantity = _decimal(_get(row, "quantity"), field="quantity")
            average_cost = _decimal(_get(row, "cost price"), field="cost price")
            last_price = _decimal(_get(row, "close price"), field="close price")
            market_value = _decimal(_get(row, "value"), field="value")
        except ValueError:
            continue
        if quantity <= 0:
            continue
        positions.append(
            BrokerHolding(
                isin=_isin(instrument),
                symbol=symbol,
                name=_get(instrument, "description", "name") or symbol,
                quantity=quantity,
                avg_price=average_cost,
                last_price=last_price,
                exchange=exchange,
                market=_market(exchange, currency),
                currency=currency,
                market_value=market_value,
                security_type="etf" if category == "etfs" else "equity",
            )
        )

    trades: list[Trade] = []
    for row in section_rows.get("trades", []):
        category = (_get(row, "asset category") or "").casefold()
        if category not in _SUPPORTED_CATEGORIES:
            warnings.add("unsupported_asset_category")
            continue
        symbol = (_get(row, "symbol") or "").upper()
        currency = (_get(row, "currency") or "").upper()
        if not symbol or currency not in _SUPPORTED_CURRENCIES:
            if currency and currency not in _SUPPORTED_CURRENCIES:
                warnings.add("unsupported_currency")
            continue
        instrument = instruments.get(symbol, {})
        exchange = _get(row, "listing exchange", "exchange") or _get(
            instrument, "listing exchange", "exchange"
        )
        try:
            signed_quantity = _decimal(_get(row, "quantity"), field="quantity")
            trade_date = _date(_get(row, "date/time", "date time", "trade date"))
            price = _decimal(_get(row, "t. price", "t price", "price"), field="price")
            commission = _decimal(
                _get(row, "comm/fee", "comm fee", "commission"), field="commission"
            )
            proceeds_raw = _get(row, "proceeds")
            proceeds = _decimal(proceeds_raw, field="proceeds") if proceeds_raw else None
        except ValueError:
            continue
        if signed_quantity == 0:
            continue
        trades.append(
            Trade(
                isin=_isin(instrument),
                symbol=symbol,
                trade_date=trade_date,
                trade_type="sell" if signed_quantity < 0 else "buy",
                quantity=abs(signed_quantity),
                price=price,
                exchange=exchange,
                trade_id=_get(row, "trade id", "transaction id"),
                market=_market(exchange, currency),
                currency=currency,
                commission=abs(commission),
                proceeds=proceeds,
                security_type="etf" if category == "etfs" else "equity",
            )
        )

    return IBKRActivity(tuple(positions), tuple(trades), tuple(sorted(warnings)))
