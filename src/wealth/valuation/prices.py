"""Price-provider interface and deterministic manual price overrides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Protocol


class PriceUnavailable(LookupError):
    """Raised when a requested market price is not present."""


@dataclass(frozen=True)
class PriceQuote:
    """A price together with the market date, source and calendar-day staleness."""

    value: Decimal
    as_of: date
    source: str
    stale_days: int


class PriceProvider(Protocol):
    """Supplies prices required by market-valued assets."""

    def gold_24k_per_gram(self, on: date) -> Decimal | PriceQuote: ...

    def nav(self, key: str, on: date) -> Decimal | PriceQuote: ...

    def price(self, security_id: str, on: date) -> Decimal | PriceQuote: ...


@dataclass(frozen=True)
class StaticPrices:
    """Exact-date dictionaries for tests and manual price overrides.

    Gold values are keyed by date. NAV values are keyed by ``(key, date)``.
    """

    gold_prices: Mapping[date, Decimal] = field(default_factory=dict)
    navs: Mapping[tuple[str, date], Decimal] = field(default_factory=dict)

    def gold_24k_per_gram(self, on: date) -> Decimal:
        try:
            return self.gold_prices[on]
        except KeyError as error:
            raise PriceUnavailable(f"No 24k gold price for {on}") from error

    def nav(self, key: str, on: date) -> Decimal:
        try:
            return self.navs[(key, on)]
        except KeyError as error:
            raise PriceUnavailable(f"No NAV for {key!r} on {on}") from error


def quote_value(value: Decimal | PriceQuote) -> tuple[Decimal, tuple[str, ...]]:
    """Return the Decimal value and a human-readable provenance note if available."""
    if isinstance(value, PriceQuote):
        return value.value, (f"price as of {value.as_of.isoformat()} ({value.source})",)
    return value, ()
