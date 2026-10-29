"""Idempotent writes and point-in-time reads for shared market data."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from wealth.db.models import (
    FundamentalPIT,
    FxRate,
    GoldPrice,
    HoldingLot,
    NavDaily,
    NavMonthly,
    PriceEod,
    Security,
    UniverseMember,
)
from wealth.marketdata.sources.fundamentals import retention_cutoff

if TYPE_CHECKING:
    from wealth.marketdata.sources.amfi import AmfiNav
    from wealth.marketdata.sources.amfi_history import AmfiHistoricalNav
    from wealth.marketdata.sources.fred import FxObservation
    from wealth.marketdata.sources.fundamentals import FundamentalObservation
    from wealth.marketdata.sources.ibja import GoldObservation
    from wealth.marketdata.sources.nse import NseClose, NseSecurity

_INDEX_PROXIES = ("NIFTYBEES", "JUNIORBEES", "GOLDBEES")


def _month_end(year: int, month: int) -> date:
    next_year, next_month = _shift_month(year, month, 1)
    return date(next_year, next_month, 1) - timedelta(days=1)


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    index = year * 12 + month - 1 + offset
    return index // 12, index % 12 + 1


def _previous_month(value: date) -> tuple[int, int]:
    return _shift_month(value.year, value.month, -1)


def _upsert_security(
    session: Session,
    values: dict[str, Any],
    *,
    by_isin: bool,
    preserve_nse_security: bool = False,
) -> UUID:
    statement = _with_security_conflict(
        pg_insert(Security).values(id=uuid4(), **values),
        by_isin=by_isin,
        preserve_nse_security=preserve_nse_security,
    )
    security_id = session.scalar(statement.returning(Security.id))
    if security_id is None:
        raise RuntimeError("security upsert returned no identifier")
    return security_id if isinstance(security_id, UUID) else UUID(str(security_id))


def _upsert_securities_batch(
    session: Session,
    rows: list[dict[str, Any]],
    *,
    by_isin: bool,
    preserve_nse_security: bool = False,
    batch_size: int = 1000,
) -> None:
    """Upsert many securities with one round trip per batch instead of one per row.

    The daily AMFI file has ~18,000 share classes; per-row upserts from a GitHub runner
    to Neon took longer than the job's 30-minute limit.
    """
    statement = _with_security_conflict(
        pg_insert(Security), by_isin=by_isin, preserve_nse_security=preserve_nse_security
    )
    for start in range(0, len(rows), batch_size):
        session.execute(
            statement, [{"id": uuid4(), **row} for row in rows[start : start + batch_size]]
        )


def _with_security_conflict(statement: Any, *, by_isin: bool, preserve_nse_security: bool) -> Any:
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
            "sub_category": func.coalesce(excluded.sub_category, Security.sub_category),
            "plan_type": func.coalesce(excluded.plan_type, Security.plan_type),
            "option": func.coalesce(excluded.option, Security.option),
            "nse_symbol": func.coalesce(excluded.nse_symbol, Security.nse_symbol),
            "series": func.coalesce(excluded.series, Security.series),
        }
        if preserve_nse_security:
            has_nse_identity = Security.nse_symbol.is_not(None)
            for field in (
                "symbol",
                "name",
                "security_type",
                "sleeve",
                "category",
                "sub_category",
                "plan_type",
                "option",
                "series",
            ):
                update_values[field] = case(
                    (has_nse_identity, getattr(Security, field)),
                    else_=update_values[field],
                )
        statement = statement.on_conflict_do_update(
            index_elements=[Security.isin],
            set_=update_values,
        )
    else:
        # Migration 0014 made (market, symbol) a partial unique index, which ON CONFLICT
        # can only infer from its columns and predicate, not by name.
        statement = statement.on_conflict_do_update(
            index_elements=[Security.market, Security.symbol],
            index_where=Security.symbol.is_not(None),
            set_={
                "name": statement.excluded.name,
                "security_type": statement.excluded.security_type,
                "currency": statement.excluded.currency,
                "sleeve": func.coalesce(statement.excluded.sleeve, Security.sleeve),
                "category": func.coalesce(statement.excluded.category, Security.category),
                "sub_category": func.coalesce(
                    statement.excluded.sub_category, Security.sub_category
                ),
                "plan_type": func.coalesce(statement.excluded.plan_type, Security.plan_type),
                "option": func.coalesce(statement.excluded.option, Security.option),
                "amfi_scheme_code": func.coalesce(
                    statement.excluded.amfi_scheme_code, Security.amfi_scheme_code
                ),
                "nse_symbol": func.coalesce(statement.excluded.nse_symbol, Security.nse_symbol),
                "series": func.coalesce(statement.excluded.series, Security.series),
            },
        )
    return statement


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


def upsert_us_equity_security(session: Session, symbol: str, name: str) -> UUID:
    """Insert/update a US listing through the shared security upsert function."""
    normalized_symbol = symbol.strip().upper()
    security_id = session.scalar(
        select(
            func.app_upsert_security_by_symbol(
                "US",
                normalized_symbol,
                "US",
                name.strip()[:160],
                "equity",
                "USD",
                None,
                "equity_global",
                "equity:us",
            )
        )
    )
    if security_id is None:
        raise RuntimeError("US security upsert returned no identifier")
    return security_id if isinstance(security_id, UUID) else UUID(str(security_id))


def upsert_universe_members(
    session: Session,
    universe: str,
    security_ids: set[UUID],
    *,
    loaded_on: date,
    source: str,
    source_url: str,
    survivorship_label: str,
) -> dict[str, int]:
    """Reconcile a current constituent snapshot without duplicate active rows."""
    rows = list(session.scalars(select(UniverseMember).where(UniverseMember.universe == universe)))
    active_by_security = {row.security_id: row for row in rows if row.removed_on is None}
    added = 0
    removed = 0
    for security_id, row in active_by_security.items():
        if security_id not in security_ids:
            row.removed_on = max(loaded_on, row.added_on)
            removed += 1
    for security_id in security_ids:
        row = active_by_security.get(security_id)
        if row is not None:
            row.source = source
            row.source_url = source_url
            row.survivorship_label = survivorship_label
            continue
        same_day = next(
            (
                previous
                for previous in rows
                if previous.security_id == security_id
                and previous.added_on == loaded_on
                and previous.removed_on is not None
            ),
            None,
        )
        if same_day is not None:
            same_day.removed_on = None
            same_day.source = source
            same_day.source_url = source_url
            same_day.survivorship_label = survivorship_label
            active_by_security[security_id] = same_day
            continue
        session.add(
            UniverseMember(
                universe=universe,
                security_id=security_id,
                added_on=loaded_on,
                removed_on=None,
                source=source,
                source_url=source_url,
                survivorship_label=survivorship_label,
            )
        )
        added += 1
    return {"added": added, "removed": removed, "active": len(security_ids)}


def _fundamentals_cutoff(as_of: date) -> date:
    return retention_cutoff(as_of)


def prune_fundamentals(session: Session, as_of: date) -> int:
    """Prune facts older than the rolling fundamentals storage window."""
    cutoff = _fundamentals_cutoff(as_of)
    result = session.execute(delete(FundamentalPIT).where(FundamentalPIT.period_end < cutoff))
    return int(result.rowcount or 0)


def upsert_fundamentals(
    session: Session,
    security_id: UUID,
    observations: list[FundamentalObservation],
    *,
    as_of: date,
    fetched_at: datetime | None = None,
) -> int:
    """Store changed values only, preserving each value's first reporting filing."""
    cutoff = _fundamentals_cutoff(as_of)
    selected = [
        item
        for item in observations
        if cutoff <= item.period_end <= as_of
        and item.period_type in {"Q", "FY"}
        and item.value.is_finite()
    ]
    if not selected:
        return 0
    retrieved_at = fetched_at or datetime.now(UTC)
    session.execute(
        select(
            func.pg_advisory_xact_lock(func.hashtextextended(f"fundamentals-pit:{security_id}", 0))
        )
    )
    groups: dict[tuple[str, date, str, str], list[FundamentalObservation]] = {}
    for observation in selected:
        key = (
            observation.metric,
            observation.period_end,
            observation.period_type,
            observation.source,
        )
        groups.setdefault(key, []).append(observation)
    existing_rows = list(
        session.scalars(
            select(FundamentalPIT).where(
                FundamentalPIT.security_id == security_id,
                FundamentalPIT.period_end >= cutoff,
                FundamentalPIT.source.in_({key[3] for key in groups}),
            )
        )
    )
    existing_groups: dict[tuple[str, date, str, str], list[FundamentalPIT]] = {}
    for row in existing_rows:
        key = (row.metric, row.period_end, row.period_type, row.source)
        existing_groups.setdefault(key, []).append(row)

    inserted = 0
    for key, incoming in groups.items():
        rows = existing_groups.get(key, [])
        by_known_at = {row.known_at: row for row in rows}
        for observation in sorted(incoming, key=lambda item: (item.known_at, item.filed_at)):
            existing = by_known_at.get(observation.known_at)
            if existing is not None:
                if observation.filed_at >= existing.filed_at:
                    existing.value = observation.value
                    existing.unit = observation.unit
                    existing.filed_at = observation.filed_at
                    existing.source_url = observation.source_url
                    existing.fetched_at = retrieved_at
                continue
            earlier_rows = [
                row for row in by_known_at.values() if row.known_at < observation.known_at
            ]
            previous = max(earlier_rows, key=lambda row: row.known_at) if earlier_rows else None
            if previous is not None and previous.value == observation.value:
                continue
            later_rows = [
                row for row in by_known_at.values() if row.known_at > observation.known_at
            ]
            current = FundamentalPIT(
                security_id=security_id,
                metric=observation.metric,
                period_end=observation.period_end,
                period_type=observation.period_type,
                source=observation.source,
                known_at=observation.known_at,
                value=observation.value,
                unit=observation.unit,
                filed_at=observation.filed_at,
                source_url=observation.source_url,
                fetched_at=retrieved_at,
                is_latest_for_period=not later_rows,
            )
            session.add(current)
            by_known_at[observation.known_at] = current
            inserted += 1
        if by_known_at:
            latest_known_at = max(by_known_at)
            for row in by_known_at.values():
                row.is_latest_for_period = row.known_at == latest_known_at
    return inserted


def latest_fundamental_filed_at(
    session: Session, security_id: UUID, source: str
) -> datetime | None:
    """Return the latest filing timestamp already observed for one security/source."""
    return session.scalar(
        select(func.max(FundamentalPIT.filed_at)).where(
            FundamentalPIT.security_id == security_id,
            FundamentalPIT.source == source,
        )
    )


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

    # Keyed by each batch's conflict target, so one batch never updates a row twice.
    isin_rows: dict[str, dict[str, Any]] = {}
    symbol_rows: dict[str, dict[str, Any]] = {}
    for (scheme_code, isin), (record, variant) in variants.items():
        symbol = (f"AMFI-{scheme_code}-{variant}" if isin else f"AMFI-{scheme_code}")[:32]
        row = {
            "market": "IN",
            "symbol": symbol,
            "isin": isin,
            "name": record.name[:160],
            "security_type": "mf",
            "currency": "INR",
            "sleeve": record.sleeve,
            "category": record.category,
            "sub_category": record.sub_category,
            "plan_type": record.plan_type,
            "option": record.option_type,
            "amfi_scheme_code": scheme_code,
            "nse_symbol": None,
            "series": None,
        }
        if isin is None:
            symbol_rows[symbol] = row
        else:
            isin_rows[isin] = row
    _upsert_securities_batch(
        session, list(isin_rows.values()), by_isin=True, preserve_nse_security=True
    )
    _upsert_securities_batch(
        session, list(symbol_rows.values()), by_isin=False, preserve_nse_security=True
    )
    return len(variants)


def amfi_history_targets(session: Session) -> set[UUID]:
    """Return direct-growth equity/hybrid schemes plus every Indian MF currently held."""
    direct_growth = set(
        session.scalars(
            select(Security.id).where(
                Security.market == "IN",
                Security.security_type == "mf",
                Security.plan_type == "direct",
                Security.option == "growth",
                Security.category.in_(("mf:equity_india", "mf:equity_global")),
            )
        )
    )
    held = set(
        session.scalars(
            select(HoldingLot.security_id)
            .join(Security, Security.id == HoldingLot.security_id)
            .where(
                Security.market == "IN",
                Security.security_type == "mf",
                HoldingLot.quantity > 0,
            )
        )
    )
    return direct_growth | held


def missing_amfi_history_windows(session: Session, on: date) -> list[tuple[date, date]]:
    """Return month-aligned gaps in the rolling five-year history, in AMFI-safe windows."""
    target_ids = amfi_history_targets(session)
    if not target_ids:
        return []
    last_month_end = (
        _month_end(on.year, on.month)
        if on.day == _month_end(on.year, on.month).day
        else _month_end(*_previous_month(on))
    )
    months = [
        _month_end(*_shift_month(last_month_end.year, last_month_end.month, offset))
        for offset in range(-59, 1)
    ]
    saved = session.execute(
        select(NavMonthly.security_id, NavMonthly.month_end).where(
            NavMonthly.security_id.in_(target_ids), NavMonthly.month_end.in_(months)
        )
    ).all()
    saved_by_security: dict[UUID, set[date]] = {}
    for security_id, month_end in saved:
        saved_by_security.setdefault(security_id, set()).add(month_end)
    missing: set[date] = set()
    for security_id in target_ids:
        available = saved_by_security.get(security_id, set())
        missing.update(month_end for month_end in months if month_end not in available)
    if not missing:
        return []
    ordered = sorted(missing)
    ranges: list[tuple[date, date]] = []
    range_start = _month_start(ordered[0])
    range_end = ordered[0]
    for month_end in ordered[1:]:
        next_start = _month_start(month_end)
        proposed_end = month_end
        if (
            next_start == range_end + timedelta(days=1)
            and (proposed_end - range_start).days + 1 <= 90
        ):
            range_end = proposed_end
            continue
        ranges.append((range_start, range_end))
        range_start, range_end = next_start, month_end
    ranges.append((range_start, range_end))
    return ranges


def prune_amfi_monthly_navs(session: Session, on: date) -> int:
    """Keep only the latest 60 completed calendar months of shared fund history."""
    month_end = (
        _month_end(on.year, on.month)
        if on.day == _month_end(on.year, on.month).day
        else _month_end(*_previous_month(on))
    )
    oldest_month_end = _month_end(*_shift_month(month_end.year, month_end.month, -59))
    result = session.execute(delete(NavMonthly).where(NavMonthly.month_end < oldest_month_end))
    return result.rowcount or 0


def upsert_amfi_monthly_navs(session: Session, records: list[AmfiHistoricalNav]) -> int:
    """Keep the last available NAV for each target security and calendar month."""
    target_ids = amfi_history_targets(session)
    if not target_ids or not records:
        return 0
    security_rows = session.execute(
        select(Security.id, Security.amfi_scheme_code, Security.isin).where(
            Security.id.in_(target_ids)
        )
    ).all()
    by_scheme_code: dict[str, set[UUID]] = {}
    by_isin: dict[str, UUID] = {}
    for security_id, scheme_code, isin in security_rows:
        if scheme_code:
            by_scheme_code.setdefault(scheme_code, set()).add(security_id)
        if isin:
            by_isin[isin] = security_id

    latest: dict[tuple[UUID, date], tuple[date, Decimal]] = {}
    for record in records:
        identifiers = {record.isin_primary, record.isin_reinvestment} - {None}
        matched = {by_isin[isin] for isin in identifiers if isin in by_isin}
        if not identifiers:
            matched.update(by_scheme_code.get(record.scheme_code, set()))
        month_end = _month_end(record.as_of.year, record.as_of.month)
        for security_id in matched:
            key = (security_id, month_end)
            previous = latest.get(key)
            if previous is None or record.as_of > previous[0]:
                latest[key] = (record.as_of, record.nav)

    if not latest:
        return 0
    values = [
        {
            "security_id": security_id,
            "month_end": month_end,
            "nav": nav,
            "source": "AMFI",
        }
        for (security_id, month_end), (_as_of, nav) in sorted(
            latest.items(), key=lambda item: (str(item[0][0]), item[0][1])
        )
    ]
    for start in range(0, len(values), 1000):
        statement = pg_insert(NavMonthly).values(values[start : start + 1000])
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[NavMonthly.security_id, NavMonthly.month_end],
                set_={"nav": statement.excluded.nav, "source": statement.excluded.source},
            )
        )
    return len(values)


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


def upsert_amfi_navs(
    session: Session,
    records: list[AmfiNav],
    *,
    fetched_at: datetime | None = None,
) -> int:
    """Store NAV rows only for securities represented in current user holdings."""
    # Match in memory against the few held securities: one query instead of one per NAV
    # row. A row without an ISIN matches by scheme code only, never by "ISIN is null".
    held = session.execute(
        select(Security.id, Security.isin, Security.amfi_scheme_code).where(
            Security.id.in_(select(HoldingLot.security_id).distinct())
        )
    ).all()
    held_by_isin = {isin: security_id for security_id, isin, _code in held if isin}
    held_by_code: dict[str, list[UUID]] = {}
    for security_id, _isin, code in held:
        if code:
            held_by_code.setdefault(code, []).append(security_id)
    count = 0
    seen: set[tuple[UUID, date]] = set()
    for record in records:
        matches = [
            *held_by_code.get(record.scheme_code, []),
            *(
                held_by_isin[isin]
                for isin in (record.isin_primary, record.isin_reinvestment)
                if isin and isin in held_by_isin
            ),
        ]
        for security_id in dict.fromkeys(matches):
            if (security_id, record.as_of) in seen:
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
    # Current index members keep their price history so the research factors (momentum,
    # volatility, 200-day distance) have the year of closes they need.
    universe_ids = set(
        session.scalars(
            select(UniverseMember.security_id).where(UniverseMember.removed_on.is_(None)).distinct()
        )
    )
    keep_ids = held_ids | proxy_ids | universe_ids
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
