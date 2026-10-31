"""CLI-only refresh job for official and fallback market-data sources."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from wealth.db.models import HoldingLot, MarketdataRun, Security, UniverseMember
from wealth.db.session import make_engine, system_session
from wealth.marketdata import store
from wealth.marketdata.sources import (
    alpha_vantage,
    amfi,
    amfi_history,
    ecb,
    edgar,
    fred,
    ibja,
    macro_geo,
    manual,
    nse,
    nse_fundamentals,
    universes,
)
from wealth.marketdata.sources.alpha_vantage import DailyClose
from wealth.marketdata.sources.amfi import AmfiNav
from wealth.marketdata.sources.amfi_history import AmfiHistoricalNav
from wealth.marketdata.sources.ecb import FxFetchResult
from wealth.marketdata.sources.edgar import EdgarCompany
from wealth.marketdata.sources.fred import FxObservation
from wealth.marketdata.sources.fundamentals import FundamentalObservation
from wealth.marketdata.sources.ibja import GoldObservation
from wealth.marketdata.sources.macro_geo import (
    FeedBatch,
    FeedItemObservation,
    IndexValuationObservation,
    MacroSeriesBatch,
    MacroSeriesObservation,
)
from wealth.marketdata.sources.nse import NseClose, NseSecurity
from wealth.marketdata.sources.nse_fundamentals import NseIntegratedClient
from wealth.marketdata.sources.universes import Sp500Holdings, Sp500Sector

_SOURCES = (
    "nse",
    "amfi",
    "fx",
    "gold",
    "master",
    "amfi_history",
    "nse_history",
    "eu_etfs",
    "us_stocks",
    "universes",
    "edgar",
    "nse_fundamentals",
    "index_pe",
    "ust",
    "feeds",
)
_RETENTION_DAYS = 400
_MAX_ABROAD_QUOTES_PER_DAY = 24
# GitHub can start the 19:00 IST job after midnight IST, and NSE posts the day's file in
# the evening, so the run date alone is often a missing session. Re-reading the last
# week fills any gap.
_NSE_CATCH_UP_DAYS = 7


@dataclass(frozen=True)
class AbroadPrice:
    """A provider close already matched to the shared securities key."""

    security_id: Any
    close: DailyClose


@dataclass(frozen=True)
class NseMasterBatch:
    securities: list[NseSecurity]
    etf_count: int
    etf_error: str | None = None


@dataclass(frozen=True)
class FundamentalsBatch:
    observations: dict[str, list[FundamentalObservation]]
    errors: dict[str, str]
    companies: dict[str, EdgarCompany] | None = None


@dataclass(frozen=True)
class UniverseBatch:
    sp500: Sp500Holdings | None
    nifty500: list[NseSecurity] | None
    errors: dict[str, str]
    sp500_closes: dict[str, Decimal] = field(default_factory=dict)
    sp500_sectors: dict[str, Sp500Sector] = field(default_factory=dict)


@dataclass(frozen=True)
class NseHistoryBatch:
    closes: list[NseClose]
    requested_sessions: int
    failed_sessions: int


@dataclass(frozen=True)
class SourceBatchResult:
    counts: dict[str, Any]
    errors: dict[str, str]


def _fetch_fx(on: date) -> FxFetchResult:
    observations: list[FxObservation] = []
    errors: dict[str, str] = {}
    try:
        ecb_result = ecb.fetch_ecb_fx_rates(fetched_on=on)
        observations.extend(ecb_result.observations)
        errors.update(ecb_result.errors)
    except Exception as error:
        errors.update({currency: type(error).__name__ for currency in ecb.ECB_CROSS_CURRENCIES})

    if not any(item.pair == "USD/INR" for item in observations):
        try:
            fallback = fred.fetch_dexinus(fetched_on=on)
        except Exception as error:
            errors.setdefault("USD", type(error).__name__)
        else:
            observations.extend(fallback)
            if any(item.pair == "USD/INR" for item in fallback):
                errors.pop("USD", None)

    by_key = {(item.pair, item.as_of): item for item in observations}
    ordered = tuple(sorted(by_key.values(), key=lambda item: (item.pair, item.as_of)))
    return FxFetchResult(ordered, errors)


def _fetch_nse_master() -> NseMasterBatch:
    equities = nse.fetch_equity_list()
    try:
        etfs = nse.fetch_etf_list()
    except Exception as error:
        return NseMasterBatch(equities, 0, type(error).__name__)
    if not etfs:
        return NseMasterBatch(equities, 0, "EmptyETFList")
    return NseMasterBatch(equities + etfs, len(etfs))


def _fetch_missing_amfi_history(session: Session, on: date) -> list[AmfiHistoricalNav]:
    records: list[AmfiHistoricalNav] = []
    for start, end in store.missing_amfi_history_windows(session, on):
        records.extend(amfi_history.fetch_nav_history(start, end))
    return records


def _fetch_recent_bhavcopies(on: date) -> list[NseClose]:
    records: list[NseClose] = []
    for offset in range(_NSE_CATCH_UP_DAYS - 1, -1, -1):
        records.extend(nse.fetch_bhavcopy(on - timedelta(days=offset)))
    return records


def _fetch_missing_nse_history(session: Session, on: date) -> NseHistoryBatch:
    """Fetch weekday sessions in the retention window that have no NIFTY 500 closes.

    Momentum and volatility need 200 sessions, so the first run loads about a year of
    official bhavcopies; later runs only retry gaps (holidays return no file).
    """
    start = on - timedelta(days=_RETENTION_DAYS)
    loaded = store.loaded_nse_sessions(session, start, on)
    wanted = [
        start + timedelta(days=offset)
        for offset in range((on - start).days + 1)
        if (start + timedelta(days=offset)).weekday() < 5
        and start + timedelta(days=offset) not in loaded
    ]
    candidates = store.nse_price_keys(session)
    closes: list[NseClose] = []
    failed = 0
    for day in wanted:
        try:
            rows = nse.fetch_bhavcopy(day)
        except Exception:
            failed += 1
            continue
        # Keep only rows a stored security can match; a year of full files is ~1M rows.
        closes.extend(
            row for row in rows if row.symbol in candidates or (row.isin or "") in candidates
        )
    return NseHistoryBatch(closes, len(wanted), failed)


def _upsert_curated_security(session: Session, instrument: Mapping[str, Any]) -> tuple[Any, str]:
    tickers = instrument.get("tickers", {})
    if not isinstance(tickers, Mapping):
        raise ValueError("curated ETF has no sourced listing")
    choices: list[tuple[int, str, str, str]] = []
    currency_rank = {"EUR": 0, "GBP": 1, "USD": 2, "DKK": 3}
    for exchange, listing in tickers.items():
        if not isinstance(exchange, str) or not isinstance(listing, Mapping):
            continue
        symbol = listing.get("symbol")
        currency = listing.get("currency")
        if isinstance(symbol, str) and currency in currency_rank:
            choices.append((currency_rank[currency], exchange, symbol, currency))
    if not choices:
        raise ValueError("curated ETF has no supported listing")
    london = [choice for choice in choices if choice[1].startswith("LSE_")]
    london_usd = [choice for choice in london if choice[1] == "LSE_USD" and choice[3] == "USD"]
    if london and not london_usd:
        raise ValueError("curated ETF has no USD-quoted London listing")
    _rank, exchange, symbol, currency = min(london_usd or choices)
    market = "GB" if exchange.startswith("LSE_") else "EU"
    security_id = session.scalar(
        select(
            func.app_upsert_security_by_symbol(
                market,
                symbol,
                exchange,
                str(instrument["name"]),
                "etf",
                currency,
                str(instrument["isin"]),
                "equity_india" if instrument.get("exposure") == "india_equity" else "equity_global",
                f"ucits:{instrument.get('exposure', 'global_all_world')}",
            )
        )
    )
    return security_id, f"{symbol}.LON" if exchange.startswith("LSE_") else symbol


def _fetch_abroad_etfs(session: Session, on: date) -> list[AbroadPrice]:
    api_key = os.getenv("PRICE_API_KEY", "")
    if not api_key:
        return []
    from wealth.policy.denmark_picks import load_eu_etfs

    records: list[AbroadPrice] = []
    instruments = load_eu_etfs()["instruments"][:_MAX_ABROAD_QUOTES_PER_DAY]
    for instrument in instruments:
        try:
            security_id, provider_symbol = _upsert_curated_security(session, instrument)
            if security_id is None:
                continue
            close = alpha_vantage.fetch_daily(provider_symbol, api_key)
            if close.as_of <= on:
                records.append(AbroadPrice(security_id, close))
        except Exception:
            continue
    return records


def _fetch_abroad_us_stocks(session: Session, on: date) -> list[AbroadPrice]:
    api_key = os.getenv("PRICE_API_KEY", "")
    if not api_key:
        return []
    from wealth.policy.denmark_picks import load_eu_etfs

    curated_count = min(
        _MAX_ABROAD_QUOTES_PER_DAY,
        len(load_eu_etfs()["instruments"]),
    )
    securities = session.scalars(
        select(Security)
        .join(HoldingLot, HoldingLot.security_id == Security.id)
        .where(Security.market == "US", Security.security_type == "equity")
        .distinct()
        .order_by(Security.symbol, Security.id)
    )
    records: list[AbroadPrice] = []
    for security in list(securities)[: _remaining_abroad_quote_slots(curated_count)]:
        if not security.symbol:
            continue
        try:
            close = alpha_vantage.fetch_daily(security.symbol, api_key)
        except Exception:
            continue
        if close.as_of <= on:
            records.append(AbroadPrice(security.id, close))
    return records


def _remaining_abroad_quote_slots(curated_count: int) -> int:
    """Reserve one free-tier call for each curated fund, capped at 24 calls daily."""
    reserved = min(_MAX_ABROAD_QUOTES_PER_DAY, max(0, curated_count))
    return _MAX_ABROAD_QUOTES_PER_DAY - reserved


def _universe_symbols(
    session: Session,
    market: str,
    universe: str,
    symbols: tuple[str, ...] | None,
) -> list[str]:
    if symbols:
        return list(dict.fromkeys(symbols))
    return list(
        session.scalars(
            select(Security.symbol)
            .join(UniverseMember, UniverseMember.security_id == Security.id)
            .where(
                UniverseMember.universe == universe,
                UniverseMember.removed_on.is_(None),
                Security.market == market,
                Security.symbol.is_not(None),
            )
            .order_by(Security.symbol)
        )
    )


def _fetch_edgar(
    session: Session | None, on: date, symbols: tuple[str, ...] | None
) -> FundamentalsBatch:
    if session is None:
        raise ValueError("EDGAR refresh requires a database session")
    client = edgar.SecEdgarClient()
    companies = edgar.fetch_company_tickers(client)
    target_symbols = _universe_symbols(session, "US", "sp500", symbols)
    if not target_symbols:
        return FundamentalsBatch({}, {"universe": "NoSymbols"}, {})
    observations: dict[str, list[FundamentalObservation]] = {}
    errors: dict[str, str] = {}
    selected_companies: dict[str, EdgarCompany] = {}
    for ticker in target_symbols:
        company = edgar.ticker_lookup(companies, ticker)
        if company is None:
            errors[ticker] = "TickerNotFound"
            continue
        selected_companies[ticker] = company
        try:
            payload = edgar.fetch_companyfacts(company.cik, client)
            observations[ticker] = edgar.parse_company_facts(payload, as_of=on)
        except Exception as error:
            errors[ticker] = type(error).__name__
    return FundamentalsBatch(observations, errors, selected_companies)


def _securities_for_symbols(
    session: Session, market: str, symbols: list[str]
) -> dict[str, Security]:
    if not symbols:
        return {}
    field = Security.nse_symbol if market == "IN" else Security.symbol
    matches = session.scalars(
        select(Security).where(
            Security.market == market,
            or_(field.in_(symbols), Security.symbol.in_(symbols)),
        )
    )
    return {
        (security.nse_symbol or security.symbol or "").upper(): security
        for security in matches
        if (security.nse_symbol or security.symbol)
    }


def _fetch_nse_fundamentals(
    session: Session | None, on: date, symbols: tuple[str, ...] | None
) -> FundamentalsBatch:
    if session is None:
        raise ValueError("NSE fundamentals refresh requires a database session")
    target_symbols = _universe_symbols(session, "IN", "nifty500", symbols)
    if not target_symbols:
        return FundamentalsBatch({}, {"universe": "NoSymbols"})
    securities = _securities_for_symbols(session, "IN", target_symbols)
    observations: dict[str, list[FundamentalObservation]] = {}
    errors: dict[str, str] = {}
    with NseIntegratedClient() as client:  # type: ignore[attr-defined]
        for symbol in target_symbols:
            security = securities.get(symbol.upper())
            if security is None:
                errors[symbol] = "SecurityNotFound"
                continue
            try:
                after = store.latest_fundamental_filed_at(session, security.id, "NSE_XBRL")
                observations[symbol] = nse_fundamentals.fetch_symbol_fundamentals(
                    client,
                    symbol,
                    as_of=on,
                    after=after,
                )
            except Exception as error:
                errors[symbol] = type(error).__name__
    return FundamentalsBatch(observations, errors)


def _fetch_universes() -> UniverseBatch:
    sp500: Sp500Holdings | None = None
    nifty500: list[NseSecurity] | None = None
    errors: dict[str, str] = {}
    sp500_closes: dict[str, Decimal] = {}
    sp500_sectors: dict[str, Sp500Sector] = {}
    try:
        sp500 = universes.fetch_sp500()
    except Exception as error:
        errors["sp500"] = type(error).__name__
    if sp500 is not None:
        try:
            sp500_closes = universes.fetch_sp500_closes(sp500)
        except Exception as error:
            errors["sp500_prices"] = type(error).__name__
        try:
            sp500_sectors = universes.fetch_sp500_sectors()
        except Exception as error:
            errors["sp500_sectors"] = type(error).__name__
    try:
        nifty500 = universes.fetch_nifty500()
    except Exception as error:
        errors["nifty500"] = type(error).__name__
    return UniverseBatch(sp500, nifty500, errors, sp500_closes, sp500_sectors)


def _default_fetchers(
    on: date,
    session: Session | None = None,
    symbols: tuple[str, ...] | None = None,
) -> dict[str, Callable[[], Any]]:
    return {
        "nse": lambda: _fetch_recent_bhavcopies(on),
        "amfi": amfi.fetch_nav_all,
        "fx": lambda: _fetch_fx(on),
        "gold": lambda: ibja.fetch_gold_rate(on=on),
        "master": _fetch_nse_master,
        "amfi_history": (
            lambda: _fetch_missing_amfi_history(session, on) if session is not None else []
        ),
        "nse_history": (
            lambda: (
                _fetch_missing_nse_history(session, on)
                if session is not None
                else NseHistoryBatch([], 0, 0)
            )
        ),
        "eu_etfs": lambda: _fetch_abroad_etfs(session, on) if session is not None else [],
        "us_stocks": lambda: _fetch_abroad_us_stocks(session, on) if session is not None else [],
        "universes": _fetch_universes,
        "edgar": lambda: _fetch_edgar(session, on, symbols),
        "nse_fundamentals": lambda: _fetch_nse_fundamentals(session, on, symbols),
        "index_pe": lambda: macro_geo.fetch_nse_index_valuations(on),
        "ust": lambda: macro_geo.fetch_treasury_yields(on),
        "feeds": lambda: macro_geo.fetch_official_feeds(on),
    }


def _store_source(
    session: Session,
    source: str,
    records: Any,
    *,
    on: date,
    fetched_at: datetime,
) -> int | dict[str, int] | str | SourceBatchResult:
    if source in {"edgar", "nse_fundamentals"}:
        if not isinstance(records, FundamentalsBatch):
            raise TypeError(f"{source} adapter returned invalid rows")
        errors = dict(records.errors)
        inserted = 0
        stored_symbols = 0
        for ticker, observations in records.observations.items():
            try:
                with session.begin_nested():
                    if source == "edgar":
                        company = (records.companies or {}).get(ticker)
                        if company is None:
                            raise ValueError("TickerNotFound")
                        security_id = store.upsert_us_equity_security(session, ticker, company.name)
                    else:
                        security = _securities_for_symbols(session, "IN", [ticker]).get(
                            ticker.upper()
                        )
                        if security is None:
                            raise ValueError("SecurityNotFound")
                        security_id = security.id
                    inserted += store.upsert_fundamentals(
                        session,
                        security_id,
                        observations,
                        as_of=on,
                        fetched_at=fetched_at,
                    )
                stored_symbols += 1
            except Exception as error:
                errors[ticker] = type(error).__name__
        return SourceBatchResult({"fundamentals_pit": inserted, "symbols": stored_symbols}, errors)
    if source == "universes":
        if not isinstance(records, UniverseBatch):
            raise TypeError("universe adapter returned invalid rows")
        errors = dict(records.errors)
        counts: dict[str, Any] = {"universes_loaded": 0}
        if records.sp500 is not None:
            try:
                with session.begin_nested():
                    security_ids: set[Any] = set()
                    price_rows: list[dict[str, Any]] = []
                    for row in records.sp500.securities:
                        security_id = store.upsert_us_equity_security(session, row.symbol, row.name)
                        security_ids.add(security_id)
                        # SSGA now lists every sector as "-"; the GICS table fills the gap.
                        listed = records.sp500_sectors.get(row.symbol)
                        sector, industry = (
                            row.sector or "Unclassified",
                            row.industry or "Unclassified",
                        )
                        if listed is not None and sector == "Unclassified":
                            sector = listed.sector
                        if listed is not None and industry == "Unclassified":
                            industry = listed.industry
                        session.execute(
                            update(Security)
                            .where(Security.id == security_id)
                            .values(sector=sector, industry=industry)
                        )
                        close = records.sp500_closes.get(row.symbol)
                        if close is not None:
                            price_rows.append(
                                {
                                    "security_id": security_id,
                                    "as_of": records.sp500.as_of,
                                    "close": close,
                                    "volume": None,
                                    "traded_value": None,
                                    "source": universes.SP500_PRICE_SOURCE,
                                    "source_url": universes.SPY_NAV_HISTORY_URL,
                                    "fetched_at": fetched_at,
                                }
                            )
                    prices = store.upsert_eod_prices(session, price_rows)
                    membership = store.upsert_universe_members(
                        session,
                        "sp500",
                        security_ids,
                        loaded_on=on,
                        source=universes.SP500_SOURCE,
                        source_url=universes.SP500_HOLDINGS_URL,
                        survivorship_label=universes.SURVIVORSHIP_LABEL,
                    )
                counts["sp500"] = {
                    **membership,
                    "as_of": records.sp500.as_of.isoformat(),
                    "prices_eod": prices,
                    "sectors": sum(
                        1 for row in records.sp500.securities if row.symbol in records.sp500_sectors
                    ),
                }
                counts["universes_loaded"] += 1
            except Exception as error:
                errors["sp500_store"] = type(error).__name__
        if records.nifty500 is not None:
            try:
                with session.begin_nested():
                    store.upsert_nse_master(session, records.nifty500)
                    symbols = {row.symbol for row in records.nifty500}
                    security_ids = set(
                        session.scalars(
                            select(Security.id).where(
                                Security.market == "IN",
                                or_(
                                    Security.nse_symbol.in_(symbols),
                                    Security.symbol.in_(symbols),
                                ),
                            )
                        )
                    )
                    membership = store.upsert_universe_members(
                        session,
                        "nifty500",
                        security_ids,
                        loaded_on=on,
                        source=universes.NIFTY500_SOURCE,
                        source_url=universes.NIFTY500_ARCHIVE_URL,
                        survivorship_label=universes.SURVIVORSHIP_LABEL,
                    )
                counts["nifty500"] = membership
                counts["universes_loaded"] += 1
            except Exception as error:
                errors["nifty500_store"] = type(error).__name__
        return SourceBatchResult(counts, errors)
    if source == "master":
        securities = records.securities if isinstance(records, NseMasterBatch) else records
        if not isinstance(securities, list) or any(
            not isinstance(item, NseSecurity) for item in securities
        ):
            raise TypeError("NSE master adapter returned invalid rows")
        return store.upsert_nse_master(session, securities)
    if source == "nse":
        if not isinstance(records, list) or any(not isinstance(item, NseClose) for item in records):
            raise TypeError("NSE price adapter returned invalid rows")
        count = store.upsert_nse_closes(session, records, fetched_at=fetched_at)
        return {"prices_eod": count}
    if source == "nse_history":
        if not isinstance(records, NseHistoryBatch):
            raise TypeError("NSE history adapter returned invalid rows")
        count = store.upsert_nse_closes(session, records.closes, fetched_at=fetched_at)
        return SourceBatchResult(
            {
                "prices_eod": count,
                "sessions_requested": records.requested_sessions,
            },
            {"failed_sessions": str(records.failed_sessions)} if records.failed_sessions else {},
        )
    if source == "amfi":
        if not isinstance(records, list) or any(not isinstance(item, AmfiNav) for item in records):
            raise TypeError("AMFI adapter returned invalid rows")
        master_count = store.upsert_amfi_master(session, records)
        cutoff = on - timedelta(days=_RETENTION_DAYS)
        recent = [item for item in records if cutoff <= item.as_of <= on]
        return {
            "amfi_master": master_count,
            "nav_daily": store.upsert_amfi_navs(session, recent, fetched_at=fetched_at),
        }
    if source == "amfi_history":
        if not isinstance(records, list) or any(
            not isinstance(item, AmfiHistoricalNav) for item in records
        ):
            raise TypeError("AMFI history adapter returned invalid rows")
        return {
            "nav_monthly": store.upsert_amfi_monthly_navs(session, records),
            "nav_monthly_pruned": store.prune_amfi_monthly_navs(session, on),
        }
    if source == "fx":
        if not isinstance(records, list) or any(
            not isinstance(item, FxObservation) for item in records
        ):
            raise TypeError("FX adapter returned invalid rows")
        cutoff = on - timedelta(days=_RETENTION_DAYS)
        selected = [item for item in records if cutoff <= item.as_of <= on]
        for item in selected:
            store.upsert_fx_rate(session, item, source=item.source, fetched_at=fetched_at)
        return {"fx_rates": len(selected)}
    if source == "index_pe":
        if not isinstance(records, list) or any(
            not isinstance(item, IndexValuationObservation) for item in records
        ):
            raise TypeError("NSE index-valuation adapter returned invalid rows")
        return {"index_valuations": store.upsert_index_valuations(session, records)}
    if source == "ust":
        if not isinstance(records, MacroSeriesBatch) or any(
            not isinstance(item, MacroSeriesObservation) for item in records.observations
        ):
            raise TypeError("Treasury adapter returned invalid rows")
        count = store.upsert_macro_series(session, records.observations)
        return SourceBatchResult(
            {
                "macro_series": count,
                "series": len({item.series for item in records.observations}),
            },
            dict(records.errors),
        )
    if source == "feeds":
        if not isinstance(records, FeedBatch) or any(
            not isinstance(item, FeedItemObservation) for item in records.items
        ):
            raise TypeError("Official-feed adapter returned invalid rows")
        count = store.upsert_feed_items(session, records.items)
        source_counts: dict[str, int] = {}
        for item in records.items:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        return SourceBatchResult(
            {"feed_items": count, "sources": source_counts},
            dict(records.errors),
        )
    if source == "gold":
        if not isinstance(records, GoldObservation):
            raise TypeError("IBJA adapter returned no parseable gold rate")
        selected_source = "manual" if records.source_url == "rules/manual_prices.yaml" else "IBJA"
        store.upsert_gold_price(session, records, source=selected_source, fetched_at=fetched_at)
        return {"gold_prices": 1, "gold_source": selected_source}
    if source in {"eu_etfs", "us_stocks"}:
        if not isinstance(records, list) or any(
            not isinstance(item, AbroadPrice) for item in records
        ):
            raise TypeError("Alpha Vantage adapter returned invalid rows")
        for item in records:
            store.upsert_eod_price(
                session,
                item.security_id,
                item.close.as_of,
                item.close.close,
                source="AlphaVantage",
                source_url=alpha_vantage.SOURCE_URL,
                fetched_at=fetched_at,
            )
        return {"prices_eod": len(records)}
    raise ValueError(f"Unknown market-data source {source}")


def _run_gold_fetch(fetch: Callable[[], Any]) -> tuple[GoldObservation, str]:
    try:
        observation = fetch()
    except Exception:
        observation = None
    if isinstance(observation, GoldObservation):
        return observation, "IBJA"
    fallback = manual.load_manual_gold_price()
    if fallback is None:
        raise ValueError("IBJA supplied no rate and manual_prices.yaml has no configured value")
    return fallback, "manual"


def run_refresh(
    session: Session,
    on: date,
    only: tuple[str, ...] | list[str] | None = None,
    *,
    fetchers: Mapping[str, Callable[[], Any]] | None = None,
    prune_old: bool = False,
    fetched_at: datetime | None = None,
    symbols: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Fetch selected sources, continue after failures, and record one run row."""
    selected_sources = tuple(only or _SOURCES)
    unknown = set(selected_sources) - set(_SOURCES)
    if unknown:
        raise ValueError(f"Unknown market-data sources: {', '.join(sorted(unknown))}")
    if "universes" in selected_sources:
        selected_sources = ("universes",) + tuple(
            source for source in selected_sources if source != "universes"
        )
    normalized_symbols = (
        tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip()))
        if symbols
        else None
    )
    effective_fetchers = dict(fetchers or _default_fetchers(on, session, normalized_symbols))
    started_at = fetched_at or datetime.now(UTC)
    run = MarketdataRun(
        started_at=started_at,
        status="failed",
        counts={},
        errors={},
    )
    session.add(run)
    session.flush()
    counts: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        with session.begin_nested():
            fundamentals_pruned = store.prune_fundamentals(session, on)
        if fundamentals_pruned:
            counts["fundamentals_pit_pruned"] = fundamentals_pruned
    except Exception as error:
        errors["fundamentals_pit_retention"] = type(error).__name__

    try:
        with session.begin_nested():
            feed_items_pruned = store.prune_feed_items(
                session,
                on,
                retention_days=macro_geo.FEED_RETENTION_DAYS,
            )
        if feed_items_pruned:
            counts["feed_items_pruned"] = feed_items_pruned
    except Exception as error:
        errors["feed_items_retention"] = type(error).__name__

    for retention_name, prune in (
        ("index_valuations_retention", store.prune_index_valuations),
        ("macro_series_retention", store.prune_macro_series),
    ):
        try:
            with session.begin_nested():
                deleted = prune(session, on)
            if deleted:
                counts[f"{retention_name.removesuffix('_retention')}_pruned"] = deleted
        except Exception as error:
            errors[retention_name] = type(error).__name__

    for source in selected_sources:
        fetcher = effective_fetchers.get(source)
        if fetcher is None:
            errors[source] = "MissingAdapter"
            continue
        try:
            if source == "gold":
                observation, source_name = _run_gold_fetch(fetcher)
                with session.begin_nested():
                    result = _store_source(
                        session,
                        source,
                        observation,
                        on=on,
                        fetched_at=started_at,
                    )
                counts[source] = result
                if source_name == "manual":
                    counts["gold_fallback"] = "manual"
            else:
                records = fetcher()
                if source == "fx" and isinstance(records, FxFetchResult):
                    for currency, error_code in records.errors.items():
                        errors[f"fx:{currency}"] = error_code
                    records = list(records.observations)
                with session.begin_nested():
                    result = _store_source(
                        session,
                        source,
                        records,
                        on=on,
                        fetched_at=started_at,
                    )
                if isinstance(result, SourceBatchResult):
                    counts[source] = result.counts
                    errors.update(
                        {f"{source}:{key}": value for key, value in result.errors.items()}
                    )
                else:
                    counts[source] = result
                if source == "master" and isinstance(records, NseMasterBatch):
                    counts["master_etf"] = records.etf_count
                    if records.etf_error is not None:
                        errors["etf_master"] = records.etf_error
        except Exception as error:
            errors[source] = type(error).__name__

    if prune_old:
        try:
            with session.begin_nested():
                counts["pruned"] = store.prune(session, on, retention_days=_RETENTION_DAYS)
        except Exception as error:
            errors["prune"] = type(error).__name__

    successful_sources = 0
    for source in selected_sources:
        source_errors = [key for key in errors if key == source or key.startswith(f"{source}:")]
        if (
            not source_errors
            or (
                source == "fx"
                and isinstance(counts.get("fx"), dict)
                and counts["fx"].get("fx_rates", 0) > 0
            )
            or (
                source in {"edgar", "nse_fundamentals"}
                and isinstance(counts.get(source), dict)
                and counts[source].get("symbols", 0) > 0
            )
            or (
                source == "nse_history"
                and isinstance(counts.get(source), dict)
                and counts[source].get("prices_eod", 0) > 0
            )
            or (
                source == "index_pe"
                and isinstance(counts.get(source), dict)
                and counts[source].get("index_valuations", 0) > 0
            )
            or (
                source == "ust"
                and isinstance(counts.get(source), dict)
                and counts[source].get("macro_series", 0) > 0
            )
            or (
                source == "feeds"
                and isinstance(counts.get(source), dict)
                and counts[source].get("feed_items", 0) > 0
            )
            or (
                source == "universes"
                and isinstance(counts.get(source), dict)
                and counts[source].get("universes_loaded", 0) > 0
            )
        ):
            successful_sources += 1
    run.status = "succeeded" if not errors else "partial" if successful_sources else "failed"
    run.finished_at = datetime.now(UTC)
    run.counts = counts
    run.errors = errors
    session.flush()
    return {
        "run_id": str(run.id),
        "status": run.status,
        "counts": counts,
        "errors": errors,
    }


def _date_argument(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh dated market-data reference rows")
    parser.add_argument("--date", type=_date_argument)
    parser.add_argument("--only", default=",".join(_SOURCES))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args(argv)
    args.sources = tuple(part.strip().casefold() for part in args.only.split(",") if part.strip())
    args.symbols = tuple(
        dict.fromkeys(part.strip().upper() for part in args.symbols.split(",") if part.strip())
    )
    if any(not re.fullmatch(r"[A-Z0-9.-]{1,15}", symbol) for symbol in args.symbols):
        parser.error("--symbols must contain comma-separated ticker symbols")
    unknown = set(args.sources) - set(_SOURCES)
    if not args.sources or unknown:
        parser.error(f"--only must contain any of: {', '.join(_SOURCES)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    on = args.date or datetime.now(ZoneInfo("Asia/Kolkata")).date()
    database_url = os.getenv("OWNER_DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
    if not database_url:
        print(
            "Set OWNER_DATABASE_URL or TEST_DATABASE_URL for a market-data refresh.",
            file=sys.stderr,
        )
        return 2

    engine = make_engine(database_url)
    try:
        with system_session(engine) as session:
            result = run_refresh(
                session,
                on,
                args.sources,
                prune_old=args.prune,
                symbols=args.symbols,
            )
    finally:
        engine.dispose()
    print(json.dumps({"date": on.isoformat(), **result}, sort_keys=True))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
