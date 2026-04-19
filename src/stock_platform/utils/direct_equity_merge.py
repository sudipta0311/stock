from __future__ import annotations

from typing import Any

from stock_platform.utils.symbol_resolver import resolve_symbol_base


def _is_missing_buy_price(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        clean = value.strip().lower()
        return clean in {"", "none", "nan", "null"}
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric <= 0


def apply_saved_buy_prices(
    direct_equities: list[dict[str, Any]],
    direct_equity_holdings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for holding in direct_equity_holdings:
        symbol = resolve_symbol_base(str(holding.get("symbol") or "").strip())
        if symbol:
            lookup[symbol] = holding

    hydrated_rows: list[dict[str, Any]] = []
    for row in direct_equities:
        merged = dict(row)
        symbol = resolve_symbol_base(str(merged.get("symbol") or "").strip())
        saved_holding = lookup.get(symbol)
        if saved_holding and _is_missing_buy_price(merged.get("avg_buy_price")):
            merged["avg_buy_price"] = saved_holding.get("avg_buy_price")
        hydrated_rows.append(merged)
    return hydrated_rows
