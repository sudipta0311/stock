"""Build deterministic weekly checks for screened and directly held shares."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from wealth.db.models import FundamentalPIT, HoldingLot, RecordedBuy, Security, StockCheck
from wealth.db.session import make_engine, system_session
from wealth.research.factors import _facts_as_of, _metric_rows
from wealth.research.rules import ResearchRules
from wealth.research.screener import (
    ScreenerRules,
    _load_market_rows,
    screen_universe,
)
from wealth.research.valuation_data import valuation_stats_for_security

_WEEK_PATTERN = re.compile(r"^(\d{4})-W(\d{2})$")
_PRICE_FRESH_DAYS = 20
_SESSIONS_FOR_DMA = 200
_REVENUE_YOY_QUARTERS = 4


def _as_of_for_week(week: str) -> date:
    matched = _WEEK_PATTERN.fullmatch(week)
    if matched is None:
        raise ValueError("week must use YYYY-Www")
    year, week_number = (int(value) for value in matched.groups())
    return date.fromisocalendar(year, week_number, 5)


def revenue_yoy_percentages(rows: list[FundamentalPIT], as_of: date) -> tuple[Decimal, ...]:
    """Return up to four consecutive, point-in-time quarterly revenue changes."""
    selected = _facts_as_of(rows, as_of)
    quarters = _metric_rows(selected, "revenue", "Q")
    if not quarters:
        return ()
    values: list[Decimal] = []
    prior_quarter: date | None = None
    for current in reversed(quarters[-_REVENUE_YOY_QUARTERS:]):
        if prior_quarter is not None and not 70 <= (prior_quarter - current.period_end).days <= 120:
            break
        target = _anniversary(current.period_end, -1)
        comparable = [
            row
            for row in quarters
            if 330 <= (current.period_end - row.period_end).days <= 400
            and abs((row.period_end - target).days) <= 45
        ]
        if not comparable:
            break
        previous = min(
            comparable,
            key=lambda row: (abs((row.period_end - target).days), -row.period_end.toordinal()),
        )
        if previous.value == 0:
            break
        values.append((current.value / previous.value - Decimal(1)) * Decimal(100))
        prior_quarter = current.period_end
    return tuple(values[:_REVENUE_YOY_QUARTERS])


def _anniversary(value: date, years: int) -> date:
    year = value.year + years
    try:
        return value.replace(year=year)
    except ValueError:
        return value.replace(year=year, day=28)


def dma200_from_sessions(prices: list[Decimal]) -> Decimal | None:
    """Average the latest 200 stored closes, or return None while history builds."""
    if len(prices) < _SESSIONS_FOR_DMA:
        return None
    return sum(prices[-_SESSIONS_FOR_DMA:], Decimal(0)) / Decimal(_SESSIONS_FOR_DMA)


def _held_security_ids(session: Any) -> set[UUID]:
    lot_ids = set(
        session.scalars(
            select(HoldingLot.security_id)
            .join(Security, Security.id == HoldingLot.security_id)
            .where(HoldingLot.quantity > 0, Security.security_type == "equity")
        )
    )
    waiting_ids = set(
        session.scalars(
            select(RecordedBuy.security_id)
            .join(Security, Security.id == RecordedBuy.security_id)
            .where(
                RecordedBuy.status == "waiting",
                RecordedBuy.security_id.is_not(None),
                Security.security_type == "equity",
            )
        )
    )
    return lot_ids | waiting_ids


def _screened_rows(session: Any, as_of: date, rules: ScreenerRules) -> dict[str, dict[str, Any]]:
    rows_by_market, _snapshots = _load_market_rows(session, as_of, rules)
    ranked: dict[str, dict[str, Any]] = {}
    for market in ("IN", "US"):
        market_rows = rows_by_market[market]
        if not market_rows:
            continue
        result = screen_universe(
            market_rows,
            market=market,
            max_per_market=rules.max_per_market,
            liquidity_floor=rules.liquidity_floor.get(market),
        )
        ranked.update(
            {
                row.security_id: {"decile": row.decile, "coverage": row.factor_coverage}
                for row in result.ranked
            }
        )
    return ranked


def _price_sessions(session: Any, security_ids: set[UUID], as_of: date) -> dict[UUID, list[Any]]:
    if not security_ids:
        return {}
    rows = session.execute(
        text(
            """
            SELECT security_id, as_of, close, source, source_url
            FROM (
                SELECT security_id, as_of, close, source, source_url,
                       row_number() OVER (
                           PARTITION BY security_id ORDER BY as_of DESC
                       ) AS session_rank
                FROM prices_eod
                WHERE security_id = ANY(:security_ids) AND as_of <= :as_of
            ) AS recent_prices
            WHERE session_rank <= :session_limit
            ORDER BY security_id, as_of
            """
        ),
        {
            "security_ids": list(security_ids),
            "as_of": as_of,
            "session_limit": _SESSIONS_FOR_DMA,
        },
    ).all()
    grouped: dict[UUID, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row.security_id].append(row)
    return dict(grouped)


def _revenue_rows(
    session: Any, security_ids: set[UUID], as_of: date
) -> dict[UUID, list[FundamentalPIT]]:
    if not security_ids:
        return {}
    facts = session.scalars(
        select(FundamentalPIT)
        .where(
            FundamentalPIT.security_id.in_(security_ids),
            FundamentalPIT.metric == "revenue",
            FundamentalPIT.period_type == "Q",
            FundamentalPIT.period_end <= as_of,
            FundamentalPIT.known_at <= datetime.combine(as_of, time.max, tzinfo=UTC),
        )
        .order_by(FundamentalPIT.security_id, FundamentalPIT.period_end, FundamentalPIT.known_at)
    )
    grouped: dict[UUID, list[FundamentalPIT]] = defaultdict(list)
    for fact in facts:
        grouped[fact.security_id].append(fact)
    return dict(grouped)


def _stock_check_payloads(
    engine: Engine,
    week: str,
    as_of: date,
    *,
    screener_rules: ScreenerRules,
) -> tuple[list[dict[str, Any]], set[UUID], set[UUID], int]:
    with system_session(engine) as session:
        held_ids = _held_security_ids(session)
        screened = _screened_rows(session, as_of, screener_rules)
        screened_ids = {UUID(security_id) for security_id in screened}
        security_ids = held_ids | screened_ids
        if not security_ids:
            return [], held_ids, screened_ids, 0

        securities = {
            security.id: security
            for security in session.scalars(
                select(Security).where(
                    Security.id.in_(security_ids),
                    Security.security_type == "equity",
                )
            )
        }
        prices_by_security = _price_sessions(session, set(securities), as_of)
        revenue_by_security = _revenue_rows(session, screened_ids, as_of)
        research_rules = ResearchRules.load()
        rows: list[dict[str, Any]] = []
        valuation_checked = 0
        for security_id in sorted(securities, key=str):
            security = securities[security_id]
            prices = prices_by_security.get(security_id, [])
            latest_price = prices[-1] if prices else None
            price = latest_price.close if latest_price is not None else None
            dma200 = dma200_from_sessions([row.close for row in prices])
            below_dma200 = price < dma200 if price is not None and dma200 is not None else None
            price_as_of = latest_price.as_of if latest_price is not None else None
            price_is_fresh = (
                price_as_of is not None and 0 <= (as_of - price_as_of).days <= _PRICE_FRESH_DAYS
            )
            if not price_is_fresh:
                price = None
                dma200 = None
                below_dma200 = None
            rank = screened.get(str(security_id))
            revenue: tuple[Decimal, ...] = ()
            pe_percentile: Decimal | None = None
            pe_vs_sector: Decimal | None = None
            valuation_source: str | None = None
            revenue_source_dates: list[str] = []
            if rank is not None:
                revenue_rows = revenue_by_security.get(security_id, [])
                revenue = revenue_yoy_percentages(revenue_rows, as_of)
                revenue_source_dates = sorted(
                    {row.period_end.isoformat() for row in revenue_rows}, reverse=True
                )[:8]
            if security_id in held_ids and security.market in {"IN", "US"}:
                valuation, valuation_source = valuation_stats_for_security(
                    session, security, as_of, research_rules
                )
                valuation_checked += 1
                pe_percentile = valuation.pe_percentile
                relative_yield = valuation.earnings_yield_vs_sector
                if relative_yield is not None and relative_yield > 0:
                    pe_vs_sector = Decimal(1) / relative_yield

            price_source = latest_price.source if latest_price is not None else None
            latest_known_at = max(
                (row.known_at for row in revenue_by_security.get(security_id, [])),
                default=None,
            )
            payload = {
                "security_id": security_id,
                "week": week,
                "as_of": as_of,
                "quant_decile": rank["decile"] if rank is not None else None,
                "factor_coverage": rank["coverage"] if rank is not None else None,
                "pe_percentile": pe_percentile,
                "pe_vs_sector": pe_vs_sector,
                "revenue_yoy_pct": [format(value, "f") for value in revenue],
                "price": price,
                "dma200": dma200,
                "below_dma200": below_dma200,
                "sources": {
                    "screener": (
                        {"week": week, "factor_coverage": rank["coverage"]}
                        if rank is not None
                        else None
                    ),
                    "price": {
                        "source": price_source,
                        "as_of": price_as_of.isoformat() if price_as_of else None,
                        "sessions": len(prices),
                    },
                    "fundamentals": {
                        "latest_known_at": latest_known_at.isoformat()
                        if latest_known_at is not None
                        else None,
                        "revenue_period_ends": revenue_source_dates,
                    }
                    if rank is not None
                    else None,
                    "valuation": {
                        "as_of": as_of.isoformat(),
                        "price_source_url": valuation_source,
                    }
                    if security_id in held_ids and security.market in {"IN", "US"}
                    else None,
                },
            }
            rows.append(payload)
        return rows, held_ids, screened_ids, valuation_checked


def run_stock_checks(
    engine: Engine,
    week: str,
    *,
    screener_rules: ScreenerRules | None = None,
) -> dict[str, Any]:
    """Store this week's screened and held share facts, then prune older rows."""
    as_of = _as_of_for_week(week)
    rows, held_ids, screened_ids, valuation_checked = _stock_check_payloads(
        engine,
        week,
        as_of,
        screener_rules=screener_rules or ScreenerRules.load(),
    )
    if rows:
        statement = pg_insert(StockCheck).values(rows)
        update_fields = (
            "as_of",
            "quant_decile",
            "factor_coverage",
            "pe_percentile",
            "pe_vs_sector",
            "revenue_yoy_pct",
            "price",
            "dma200",
            "below_dma200",
            "sources",
        )
        statement = statement.on_conflict_do_update(
            index_elements=[StockCheck.security_id, StockCheck.week],
            set_={name: getattr(statement.excluded, name) for name in update_fields},
        )
    minimum_date = as_of - timedelta(weeks=11)
    minimum_iso = minimum_date.isocalendar()
    minimum_week = f"{minimum_iso.year:04d}-W{minimum_iso.week:02d}"
    with system_session(engine) as session:
        if rows:
            session.execute(statement)
        session.execute(delete(StockCheck).where(StockCheck.week < minimum_week))

    checked = len(rows)
    coverage = {
        name: sum(row[name] is not None for row in rows)
        for name in (
            "quant_decile",
            "factor_coverage",
            "pe_percentile",
            "pe_vs_sector",
            "price",
            "dma200",
        )
    }
    coverage["revenue_yoy_pct"] = sum(bool(row["revenue_yoy_pct"]) for row in rows)
    return {
        "week": week,
        "as_of": as_of.isoformat(),
        "checked": checked,
        "held_checked": len(held_ids & {row["security_id"] for row in rows}),
        "held_not_covered": len(held_ids - screened_ids),
        "valuation_checked": valuation_checked,
        "coverage": coverage,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Store weekly deterministic share checks")
    parser.add_argument("--week", required=True, help="ISO week such as 2026-W44")
    arguments = parser.parse_args(argv)
    try:
        _as_of_for_week(arguments.week)
    except ValueError as error:
        parser.error(str(error))
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    database_url = os.getenv("OWNER_DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "unavailable", "reason": "database is not configured"}))
        return 2
    engine = make_engine(database_url)
    try:
        result = run_stock_checks(engine, arguments.week)
    except Exception as error:
        print(json.dumps({"status": "unavailable", "reason": type(error).__name__}))
        return 1
    finally:
        engine.dispose()
    print(json.dumps({"status": "complete", **result}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
