"""Database-backed implementation of the deterministic valuation price provider."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from wealth.db.models import Security
from wealth.marketdata import store
from wealth.valuation.prices import PriceQuote, PriceUnavailable

MarketPrice = PriceQuote


class DbPriceProvider:
    """Read current database prices without making network requests."""

    def __init__(self, session: Session, *, max_age_days: int = 10) -> None:
        self.session = session
        self.max_age_days = max_age_days

    def _point(
        self, row: tuple[Decimal, date, str] | None, on: date, description: str
    ) -> PriceQuote:
        if row is None:
            raise PriceUnavailable(f"No {description} price exists for {on} or earlier")
        value, as_of, source = row
        stale_days = (on - as_of).days
        if stale_days < 0 or stale_days > self.max_age_days:
            raise PriceUnavailable(f"Latest {description} price is {stale_days} days old")
        return PriceQuote(value, as_of, source, stale_days)

    def gold_24k_per_gram(self, on: date) -> PriceQuote:
        return self._point(store.latest_gold_price(self.session, on), on, "24k gold")

    def _security_ids(self, key: str) -> list[UUID]:
        security_ids = select(Security.id).where(
            or_(
                Security.isin == key.upper(),
                Security.nse_symbol == key.upper(),
                Security.symbol == key.upper(),
                Security.amfi_scheme_code == key,
            )
        )
        return list(dict.fromkeys(self.session.scalars(security_ids)))

    def nav(self, key: str, on: date) -> PriceQuote:
        matches = [
            point
            for security_id in self._security_ids(key)
            if (point := store.latest_nav(self.session, security_id, on)) is not None
        ]
        latest = max(matches, key=lambda point: point[1]) if matches else None
        return self._point(latest, on, f"NAV for {key}")

    def price(self, security_id: UUID | str, on: date) -> PriceQuote:
        identifier: UUID | None
        try:
            identifier = UUID(str(security_id))
        except ValueError:
            matches = self._security_ids(str(security_id))
            identifier = matches[0] if matches else None
        if identifier is None:
            return self._point(None, on, f"security {security_id}")
        return self._point(store.latest_price(self.session, identifier, on), on, "security")

    def market_data_as_of(self, on: date) -> date | None:
        """Return the freshest daily price date represented by the database."""
        return store.latest_market_data_as_of(self.session, on)
