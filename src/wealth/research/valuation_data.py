"""Database inputs shared by dossier and weekly stock-check valuations."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from wealth.db.models import FundamentalPIT, PriceEod, Security
from wealth.research.factors import fundamental_factors
from wealth.research.rules import ResearchRules
from wealth.research.valuation import ValuationStats, calculate_valuation_stats


def valuation_stats_for_security(
    session: Session,
    security: Security,
    as_of: date,
    rules: ResearchRules,
) -> tuple[ValuationStats, str | None]:
    """Use the same peer, filing, price, and monthly-history inputs as dossier checks."""
    if security.market not in {"IN", "US"}:
        raise ValueError("Research valuation supports India and US securities")

    peers = list(
        session.scalars(
            select(Security).where(
                Security.market == security.market,
                Security.sector == security.sector,
                Security.security_type == "equity",
            )
        )
    )
    if (security.sector or "").casefold() == "unclassified":
        peers = [security]
    peer_ids = [peer.id for peer in peers]
    cutoff = as_of - timedelta(days=3 * 366 + 35)
    price_rows = list(
        session.execute(
            select(
                PriceEod.security_id,
                PriceEod.as_of,
                PriceEod.close,
                PriceEod.source_url,
            )
            .where(
                PriceEod.security_id.in_(peer_ids),
                PriceEod.as_of >= cutoff,
                PriceEod.as_of <= as_of,
            )
            .order_by(PriceEod.security_id, PriceEod.as_of)
        )
    )
    fact_rows = list(
        session.scalars(
            select(FundamentalPIT).where(
                FundamentalPIT.security_id.in_(peer_ids),
                FundamentalPIT.metric.in_(
                    ("eps_diluted", "operating_cash_flow", "capex", "shares_outstanding")
                ),
                FundamentalPIT.period_end <= as_of,
                FundamentalPIT.known_at <= datetime.combine(as_of, time.max, tzinfo=UTC),
            )
        )
    )

    prices_by_security: dict[Any, list[Any]] = {}
    for row in price_rows:
        prices_by_security.setdefault(row.security_id, []).append(row)
    facts_by_security: dict[Any, list[FundamentalPIT]] = {}
    for row in fact_rows:
        facts_by_security.setdefault(row.security_id, []).append(row)

    current_prices: dict[Any, Decimal] = {}
    for peer in peers:
        rows = prices_by_security.get(peer.id, [])
        if rows and (as_of - rows[-1].as_of).days <= 20:
            current_prices[peer.id] = rows[-1].close

    security_price = current_prices.get(security.id)
    own_facts = facts_by_security.get(security.id, [])
    own_factors = fundamental_factors(own_facts, as_of=as_of, price=security_price)
    peer_yields: list[Decimal] = []
    for peer in peers:
        peer_price = current_prices.get(peer.id)
        if peer_price is None:
            continue
        peer_factors = fundamental_factors(
            facts_by_security.get(peer.id, []), as_of=as_of, price=peer_price
        )
        if peer_factors.earnings_yield is not None:
            peer_yields.append(peer_factors.earnings_yield)

    monthly_prices: dict[tuple[int, int], tuple[date, Decimal]] = {}
    for row in prices_by_security.get(security.id, []):
        monthly_prices[(row.as_of.year, row.as_of.month)] = (row.as_of, row.close)
    historical_pe: list[Decimal] = []
    for on, close in monthly_prices.values():
        historical_factors = fundamental_factors(own_facts, as_of=on, price=close)
        if historical_factors.earnings_yield is not None and historical_factors.earnings_yield > 0:
            historical_pe.append(Decimal(1) / historical_factors.earnings_yield)

    stats = calculate_valuation_stats(
        security_id=str(security.id),
        market=security.market,
        price=security_price,
        factors=own_factors,
        sector_earnings_yields=peer_yields,
        historical_pe=historical_pe,
        rules=rules,
    )
    own_prices = prices_by_security.get(security.id, [])
    return stats, own_prices[-1].source_url if own_prices else None
