"""Pure estate-tax exposure totals for assets domiciled in the United States."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any


def _field(holding: Mapping[str, Any] | object, name: str) -> Any:
    if isinstance(holding, Mapping):
        return holding.get(name)
    return getattr(holding, name, None)


def _decimal(value: object, field: str) -> Decimal:
    if isinstance(value, (float, bool)):
        raise ValueError(f"{field} must be a decimal value")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a decimal value") from error
    if not amount.is_finite():
        raise ValueError(f"{field} must be finite")
    return amount


def us_situs_value_usd(holdings: Iterable[Mapping[str, Any] | object]) -> Decimal:
    """Sum USD values for US-domiciled holdings; other domiciles are excluded."""
    total = Decimal(0)
    for holding in holdings:
        if str(_field(holding, "domicile") or "").upper() != "US":
            continue
        value_usd = _field(holding, "value_usd")
        if value_usd is None:
            if str(_field(holding, "currency") or "").upper() != "USD":
                continue
            price = _field(holding, "price")
            if price is None:
                price = _field(holding, "statement_value")
                value_usd = _decimal(price, "statement value") if price is not None else None
            else:
                quantity = _decimal(_field(holding, "quantity"), "quantity")
                value_usd = quantity * _decimal(price, "price")
        if value_usd is not None:
            total += _decimal(value_usd, "USD value")
    return total
