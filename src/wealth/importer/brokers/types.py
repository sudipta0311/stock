"""Typed values shared by the broker file parsers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class Layout:
    broker: str
    kind: Literal["holdings", "tradebook"]
    fingerprint: frozenset[str]
    columns: dict[str, str | tuple[str, ...]]
    verified: bool = True
    source_note: str = ""
    header_fingerprint: str | None = None
    market_default: str = "IN"
    currency_default: str = "INR"


@dataclass(frozen=True)
class BrokerHolding:
    isin: str | None
    symbol: str | None
    name: str | None
    quantity: Decimal
    avg_price: Decimal
    last_price: Decimal | None
    exchange: str | None
    market: str = "IN"
    currency: str = "INR"
    market_value: Decimal | None = None
    security_type: str = "equity"


@dataclass(frozen=True)
class Trade:
    isin: str | None
    symbol: str | None
    trade_date: date
    trade_type: Literal["buy", "sell"]
    quantity: Decimal
    price: Decimal
    exchange: str | None = None
    order_id: str | None = None
    trade_id: str | None = None
    market: str = "IN"
    currency: str = "INR"
    commission: Decimal = Decimal(0)
    proceeds: Decimal | None = None
    security_type: str = "equity"

    @property
    def security_key(self) -> str:
        identity = self.isin or f"{self.exchange or ''}:{self.symbol or ''}"
        return f"{self.market}:{self.currency}:{identity}".casefold()


@dataclass(frozen=True)
class OpenLot:
    isin: str | None
    symbol: str | None
    exchange: str | None
    quantity: Decimal
    cost_per_unit: Decimal
    acquired_on: date
    trade_ids: tuple[str, ...]
    market: str = "IN"
    currency: str = "INR"
    security_type: str = "equity"


@dataclass(frozen=True)
class RealisedLot:
    isin: str | None
    symbol: str | None
    exchange: str | None
    quantity: Decimal
    cost_per_unit: Decimal
    sale_price: Decimal
    acquired_on: date
    sold_on: date
    holding_period: Literal["LT", "ST"]
    buy_trade_ids: tuple[str, ...]
    sell_trade_id: str | None
    market: str = "IN"
    currency: str = "INR"
    commission: Decimal = Decimal(0)
    security_type: str = "equity"

    @property
    def gain(self) -> Decimal:
        return (self.sale_price - self.cost_per_unit) * self.quantity - self.commission


@dataclass(frozen=True)
class BrokerCashFlow:
    transaction_date: date
    kind: Literal["deposit", "withdrawal"]
    amount: Decimal
    currency: str
    transaction_id: str | None = None
