"""Idempotent writes and point-in-time reads for shared market data."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from wealth.db.models import (
    FxRate,
    GoldPrice,
    HoldingLot,
    NavDaily,
    PriceEod,
    Security,
)

if TYPE_CHECKING:
    from wealth.marketdata.sources.amfi import AmfiNav
    from wealth.marketdata.sources.fred import FxObservation
    from wealth.marketdata.sources.ibja import GoldObservation
    from wealth.marketdata.sources.nse import NseClose, NseSecurity

_INDEX_PROXIES = ("NIFTYBEES", "JUNIORBEES", "GOLDBEES")


def _upsert_security(
    session: Session,
    values: dict[str, Any],
    *,
    by_isin: bool,
    preserve_nse_security: bool = False,
) -> UUID:
    statement = pg_insert(Security).values(id=uuid4(), **values)
    if by_isin:
        excluded = statement.excluded
        update_values: dict[str, Any] = {
            "market": excluded.market,
            "symbol": excluded.symbol,
            "name": excluded.name,
            "security_type": excluded.security_type,
            "currency": excluded.currency,
            "sleeve": func.coalesce(excluded.sleeve, Security.sleeve),
            "category": func.coalesce(excluded.category, Security.category),
            "amfi_scheme_code": func.coalesce(excluded.amfi_scheme_code, Security.amfi_scheme_code),
            "nse_symbol": func.coalesce(excluded.nse_symbol, Security.nse_symbol),
            "series": func.coalesce(excluded.series, Security.series),
        }
        if preserve_nse_security:
            has_nse_identity = Security.nse_symbol.is_not(None)
            for field in ("symbol", "name", "security_type", "sleeve", "category", "series"):
                update_values[field] = case(
                    (has_nse_identity, getattr(Security, field)),
                    else_=update_values[field],
                )
        statement = statement.on_conflict_do_update(
            index_elements=[Security.isin],
            set_=update_values,
        )
    else:
        statement = statement.on_conflict_do_update(
            constraint="uq_securities_market_symbol",
            set_={
                "name": statement.excluded.name,
                "security_type": statement.excluded.security_type,
                "currency": statement.excluded.currency,
                "sleeve": func.coalesce(statement.excluded.sleeve, Security.sleeve),
                "category": func.coalesce(statement.excluded.category, Security.category),
                "amfi_scheme_code": func.coalesce(
                    statement.excluded.amfi_scheme_code, Security.amfi_scheme_code
                ),
                "nse_symbol": func.coalesce(statement.excluded.nse_symbol, Security.nse_symbol),
                "series": func.coalesce(statement.excluded.series, Security.series),
            },
        )
    security_id = session.scalar(statement.returning(Security.id))
    if security_id is None:
        raise RuntimeError("security upsert returned no identifier")
    return security_id if isinstance(security_id, UUID) else UUID(str(security_id))


def upsert_nse_master(session: Session, records: list[NseSecurity]) -> int:
    """Insert or refresh NSE securities by ISIN or exchange symbol."""
    count = 0
    for record in records:
        if record.security_type == "etf":
            sleeve = record.sleeve or (
                "gold" if "gold" in record.name.casefold() else "equity_india"
            )
            category = record.category or f"etf:{sleeve}"
        else:
            sleeve = "equity_india"
            category = "equity:india"
        _upsert_security(
            session,
            {
                "market": "IN",
                "symbol": record.symbol,
                "isin": record.isin,
                "name": record.name[:160],
                "security_type": record.security_type,
                "currency": "INR",
                "sleeve": sleeve,
                "category": category,
                "nse_symbol": record.symbol,
                "series": record.series,
            },
            by_isin=record.isin is not None,
        )
        count += 1
    return count


def upsert_amfi_master(session: Session, records: list[AmfiNav]) -> int:
    """Create/update AMFI scheme securities and apply section-based classifications."""
    variants: dict[tuple[str, str | None], tuple[AmfiNav, str]] = {}
    for record in records:
        isins = list(
            dict.fromkeys(isin for isin in (record.isin_primary, record.isin_reinvestment) if isin)
        )
        if not isins:
            variants[(record.scheme_code, None)] = (record, "")
        for index, isin in enumerate(isins):
            variants[(record.scheme_code, isin)] = (record, "G" if index == 0 else "R")

    for (scheme_code, isin), (record, variant) in variants.items():
        symbol = f"AMFI-{scheme_code}-{variant}" if isin else f"AMFI-{scheme_code}"
        _upsert_security(
            session,
            {
                "market": "IN",
                "symbol": symbol[:32],
                "isin": isin,
                "name": record.name[:160],
                "security_type": "mf",
                "currency": "INR",
                "sleeve": record.sleeve,
                "category": record.category,
                "amfi_scheme_code": scheme_code,
                "nse_symbol": None,
                "series": None,
            },
            by_isin=isin is not None,
            preserve_nse_security=True,
        )
    return len(variants)


def upsert_eod_price(
    session: Session,
    security_id: UUID,
    as_of: date,
    close: Decimal,
    *,
    source: str = "NSE",
    source_url: str = "https://www.nseindia.com/all-reports",
    fetched_at: datetime | None = None,
) -> None:
    values: dict[str, Any] = {
        "security_id": security_id,
        "as_of": as_of,
        "close": close,
        "source": source,
        "source_url": source_url,
    }
    if fetched_at is not None:
        values["fetched_at"] = fetched_at
    statement = pg_insert(PriceEod).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[PriceEod.security_id, PriceEod.as_of],
        set_={
            "close": statement.excluded.close,
            "source": statement.excluded.source,
            "source_url": statement.excluded.source_url,
            "fetched_at": statement.excluded.fetched_at,
        },
    )
    session.execute(statement)


def upsert_nav(
    session: Session,
    security_id: UUID,
    as_of: date,
    nav: Decimal,
    *,
    source: str = "AMFI",
    fetched_at: datetime | None = None,
) -> None:
    values: dict[str, Any] = {
        "security_id": security_id,
        "as_of": as_of,
        "nav": nav,
        "source": source,
    }
    if fetched_at is not None:
        values["fetched_at"] = fetched_at
    statement = pg_insert(NavDaily).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[NavDaily.security_id, NavDaily.as_of],
        set_={
            "nav": statement.excluded.nav,
            "source": statement.excluded.source,
            "fetched_at": statement.excluded.fetched_at,
        },
    )
    session.execute(statement)


def upsert_fx_rate(
    session: Session,
    observation: FxObservation,
    *,
    source: str | None = None,
    fetched_at: datetime | None = None,
) -> None:
    values: dict[str, Any] = {
        "pair": observation.pair,
        "as_of": observation.as_of,
        "rate": observation.rate,
        "source": source or observation.source,
        "source_url": observation.source_url,
        "publication_lag_days": observation.publication_lag_days,
    }
    if fetched_at is not None:
        values["fetched_at"] = fetched_at
    statement = pg_insert(FxRate).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[FxRate.pair, FxRate.as_of],
        set_={
            "rate": statement.excluded.rate,
            "source": statement.excluded.source,
            "source_url": statement.excluded.source_url,
            "publication_lag_days": statement.excluded.publication_lag_days,
            "fetched_at": statement.excluded.fetched_at,
        },
    )
    session.execute(statement)


def upsert_gold_price(
    session: Session,
    observation: GoldObservation,
    *,
    source: str = "IBJA",
    fetched_at: datetime | None = None,
) -> None:
    values: dict[str, Any] = {
        "as_of": observation.as_of,
        "inr_per_gram_24k": observation.inr_per_gram_24k,
        "source": source,
        "source_url": observation.source_url,
    }
    if fetched_at is not None:
        values["fetched_at"] = fetched_at
    statement = pg_insert(GoldPrice).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[GoldPrice.as_of],
        set_={
            "inr_per_gram_24k": statement.excluded.inr_per_gram_24k,
            "source": statement.excluded.source,
            "source_url": statement.excluded.source_url,
            "fetched_at": statement.excluded.fetched_at,
        },
    )
    session.execute(statement)


def upsert_nse_closes(
    session: Session,
    records: list[NseClose],
    *,
    fetched_at: datetime | None = None,
) -> int:
    """Store NSE closes only for held securities and the configured index proxies."""
    held = set(session.scalars(select(HoldingLot.security_id).distinct()))
    proxy_rows = session.scalars(
        select(Security.id).where(
            or_(Security.symbol.in_(_INDEX_PROXIES), Security.nse_symbol.in_(_INDEX_PROXIES))
        )
    )
    allowed_ids = held | set(proxy_rows)
    count = 0
    for record in records:
        if record.as_of is None:
            continue
        if record.series:
            clauses = [
                (Security.nse_symbol == record.symbol) & (Security.series == record.series),
                (Security.symbol == record.symbol) & (Security.series == record.series),
            ]
        else:
            clauses = [Security.nse_symbol == record.symbol, Security.symbol == record.symbol]
        if record.isin:
            clauses.append(Security.isin == record.isin)
        matches = list(session.scalars(select(Security).where(or_(*clauses))))
        for security in matches:
            if security.id in allowed_ids:
                upsert_eod_price(
                    session,
                    security.id,
                    record.as_of,
                    record.close,
                    source_url=(
                        "https://nsearchives.nseindia.com/content/cm/"
                        f"BhavCopy_NSE_CM_0_0_0_{record.as_of:%Y%m%d}_F_0000.csv.zip"
                    ),
                    fetched_at=fetched_at,
                )
                count += 1
    return count


def securities_for_amfi_nav(session: Session, record: AmfiNav) -> list[UUID]:
    """Return the share-class security IDs represented by one AMFI NAV row."""
    identifiers = [
        identifier for identifier in (record.isin_primary, record.isin_reinvestment) if identifier
    ]
    query = select(Security.id).where(
        or_(
            Security.amfi_scheme_code == record.scheme_code,
            Security.isin.in_(identifiers) if identifiers else Security.isin.is_(None),
        )
    )
    return list(dict.fromkeys(session.scalars(query)))


def upsert_amfi_navs(
    session: Session,
    records: list[AmfiNav],
    *,
    fetched_at: datetime | None = None,
) -> int:
    """Store NAV rows only for securities represented in current user holdings."""
    held_ids = set(session.scalars(select(HoldingLot.security_id).distinct()))
    count = 0
    seen: set[tuple[UUID, date]] = set()
    for record in records:
        for security_id in securities_for_amfi_nav(session, record):
            if security_id not in held_ids or (security_id, record.as_of) in seen:
                continue
            upsert_nav(session, security_id, record.as_of, record.nav, fetched_at=fetched_at)
            seen.add((security_id, record.as_of))
            count += 1
    return count


def latest_price(session: Session, security_id: UUID, on: date) -> tuple[Decimal, date, str] | None:
    """Return the newest NSE close no later than ``on`` with its provenance source."""
    row = session.execute(
        select(PriceEod.close, PriceEod.as_of, PriceEod.source)
        .where(PriceEod.security_id == security_id, PriceEod.as_of <= on)
        .order_by(PriceEod.as_of.desc())
        .limit(1)
    ).one_or_none()
    return tuple(row) if row is not None else None


def latest_nav(session: Session, security_id: UUID, on: date) -> tuple[Decimal, date, str] | None:
    """Return the newest AMFI NAV no later than ``on`` with its provenance source."""
    row = session.execute(
        select(NavDaily.nav, NavDaily.as_of, NavDaily.source)
        .where(NavDaily.security_id == security_id, NavDaily.as_of <= on)
        .order_by(NavDaily.as_of.desc())
        .limit(1)
    ).one_or_none()
    return tuple(row) if row is not None else None


def latest_gold_price(session: Session, on: date) -> tuple[Decimal, date, str] | None:
    row = session.execute(
        select(GoldPrice.inr_per_gram_24k, GoldPrice.as_of, GoldPrice.source)
        .where(GoldPrice.as_of <= on)
        .order_by(GoldPrice.as_of.desc())
        .limit(1)
    ).one_or_none()
    return tuple(row) if row is not None else None


def latest_fx_rate(session: Session, pair: str, on: date) -> tuple[Decimal, date, str] | None:
    row = session.execute(
        select(FxRate.rate, FxRate.as_of, FxRate.source)
        .where(FxRate.pair == pair, FxRate.as_of <= on)
        .order_by(FxRate.as_of.desc())
        .limit(1)
    ).one_or_none()
    return tuple(row) if row is not None else None


def latest_market_data_as_of(session: Session, on: date) -> date | None:
    """Return the newest available daily market-data date no later than ``on``."""
    dates = [
        session.scalar(select(func.max(PriceEod.as_of)).where(PriceEod.as_of <= on)),
        session.scalar(select(func.max(NavDaily.as_of)).where(NavDaily.as_of <= on)),
        session.scalar(select(func.max(GoldPrice.as_of)).where(GoldPrice.as_of <= on)),
        session.scalar(select(func.max(FxRate.as_of)).where(FxRate.as_of <= on)),
    ]
    values = [as_of for as_of in dates if as_of is not None]
    return max(values) if values else None


def prune(session: Session, on: date, *, retention_days: int = 400) -> dict[str, int]:
    """Remove old daily rows, preserving prices for held securities and index proxies."""
    cutoff = on - timedelta(days=retention_days)
    held_ids = set(session.scalars(select(HoldingLot.security_id).distinct()))
    proxy_ids = set(
        session.scalars(
            select(Security.id).where(
                or_(Security.symbol.in_(_INDEX_PROXIES), Security.nse_symbol.in_(_INDEX_PROXIES))
            )
        )
    )
    keep_ids = held_ids | proxy_ids
    price_delete = delete(PriceEod).where(PriceEod.as_of < cutoff)
    nav_delete = delete(NavDaily).where(NavDaily.as_of < cutoff)
    if keep_ids:
        price_delete = delete(PriceEod).where(
            or_(PriceEod.as_of < cutoff, PriceEod.security_id.not_in(keep_ids))
        )
        nav_delete = delete(NavDaily).where(
            or_(NavDaily.as_of < cutoff, NavDaily.security_id.not_in(keep_ids))
        )
    else:
        price_delete = delete(PriceEod)
        nav_delete = delete(NavDaily)
    counts = {
        "prices_eod": session.execute(price_delete).rowcount or 0,
        "nav_daily": session.execute(nav_delete).rowcount or 0,
        "gold_prices": session.execute(delete(GoldPrice).where(GoldPrice.as_of < cutoff)).rowcount
        or 0,
        "fx_rates": session.execute(delete(FxRate).where(FxRate.as_of < cutoff)).rowcount or 0,
    }
    return counts
