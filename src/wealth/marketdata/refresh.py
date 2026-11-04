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
from decimal import ROUND_HALF_UP, Decimal
from time import sleep as sleep_for
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, delete, func, or_, select, update
from sqlalchemy.orm import Session

from wealth.db.models import (
    FundamentalPIT,
    FundBasket,
    FundBasketWeight,
    HoldingLot,
    IndexValuation,
    MarketdataRun,
    PriceEod,
    Security,
    UniverseMember,
)
from wealth.db.session import make_engine, system_session
from wealth.marketdata import store
from wealth.marketdata.sources import (
    alpha_vantage,
    amfi,
    amfi_history,
    ecb,
    edgar,
    fred,
    fund_baskets,
    ibja,
    macro_geo,
    manual,
    nse,
    nse_fundamentals,
    universes,
    us_index_pe,
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
    "fund_baskets",
    "eu_etfs",
    "us_stocks",
    "universes",
    "edgar",
    "us_index_pe",
    "nse_fundamentals",
    "index_pe",
    "ust",
    "feeds",
)
_RETENTION_DAYS = 400
_MAX_ABROAD_QUOTES_PER_DAY = 24
# Alpha Vantage identifies 5 free-tier calls per minute and 25 per day; 12 seconds stays below it.
# https://www.alphavantage.co/stock-price-tracker-website-python-django/
# https://www.alphavantage.co/support/
_ALPHA_VANTAGE_REQUEST_SPACING_SECONDS = 12
_ALPHA_VANTAGE_REQUEST_TIMEOUT_SECONDS = 20.0
_ALPHA_VANTAGE_REQUEST_RETRIES = 1
# Alpha Vantage global symbols include exchange suffixes (for example, TSCO.LON).
# https://www.alphavantage.co/documentation/#weekly
_ALPHA_VANTAGE_EXCHANGE_SUFFIXES = {"LSE_USD": ".LON", "Xetra": ".DEX"}
_ABROAD_SKIP_REASONS = (
    "rate_limited",
    "no_supported_listing",
    "unknown_symbol",
    "no_series",
    "http_error",
)
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
class CuratedListing:
    exchange: str
    symbol: str
    currency: str
    provider_symbol: str


@dataclass(frozen=True)
class CuratedAbroadPrice:
    instrument: Mapping[str, Any]
    listing: CuratedListing
    close: DailyClose


@dataclass(frozen=True)
class AbroadBatch:
    records: list[AbroadPrice | CuratedAbroadPrice]
    requested: int
    skipped: dict[str, int]


@dataclass
class AbroadRequestBudget:
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, _MAX_ABROAD_QUOTES_PER_DAY - self.used)


@dataclass(frozen=True)
class NseFundamentalTarget:
    symbol: str
    security_id: Any
    after: datetime | None


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
    xbrl_missing: int = 0


@dataclass(frozen=True)
class UniverseBatch:
    sp500: Sp500Holdings | None
    nifty500: list[NseSecurity] | None
    errors: dict[str, str]
    sp500_closes: dict[str, Decimal] = field(default_factory=dict)
    sp500_sectors: dict[str, Sp500Sector] = field(default_factory=dict)


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
    return _fetch_amfi_history_windows(store.missing_amfi_history_windows(session, on))


def _fetch_amfi_history_windows(
    windows: list[tuple[date, date]],
) -> list[AmfiHistoricalNav]:
    records: list[AmfiHistoricalNav] = []
    for start, end in windows:
        records.extend(amfi_history.fetch_nav_history(start, end))
    return records


def _fetch_recent_bhavcopies(on: date) -> list[NseClose]:
    records: list[NseClose] = []
    for offset in range(_NSE_CATCH_UP_DAYS - 1, -1, -1):
        records.extend(nse.fetch_bhavcopy(on - timedelta(days=offset)))
    return records


def _choose_curated_listing(instrument: Mapping[str, Any]) -> CuratedListing | None:
    tickers = instrument.get("tickers", {})
    if not isinstance(tickers, Mapping):
        return None
    for exchange in ("LSE_USD", "Xetra"):
        listing = tickers.get(exchange)
        if not isinstance(listing, Mapping):
            continue
        currency = "USD" if exchange == "LSE_USD" else "EUR"
        symbol = listing.get("symbol")
        if isinstance(symbol, str) and symbol.strip() and listing.get("currency") == currency:
            return CuratedListing(
                exchange=exchange,
                symbol=symbol.strip(),
                currency=currency,
                provider_symbol=f"{symbol.strip()}{_ALPHA_VANTAGE_EXCHANGE_SUFFIXES[exchange]}",
            )
    return None


def _upsert_curated_security(
    session: Session,
    instrument: Mapping[str, Any],
    listing: CuratedListing | None = None,
) -> tuple[Any, str]:
    selected = listing or _choose_curated_listing(instrument)
    if selected is None:
        raise ValueError("curated ETF has no supported USD London or EUR Xetra listing")
    market = "GB" if selected.exchange == "LSE_USD" else "EU"
    security_id = session.scalar(
        select(
            func.app_upsert_security_by_symbol(
                market,
                selected.symbol,
                selected.exchange,
                str(instrument["name"]),
                "etf",
                selected.currency,
                str(instrument["isin"]),
                "equity_india" if instrument.get("exposure") == "india_equity" else "equity_global",
                f"ucits:{instrument.get('exposure', 'global_all_world')}",
            )
        )
    )
    return security_id, selected.provider_symbol


def _fetch_alpha_vantage_daily(
    symbol: str,
    api_key: str,
    *,
    sleeper: Callable[[float], None] = sleep_for,
    request_budget: AbroadRequestBudget | None = None,
) -> DailyClose:
    for attempt in range(2):
        if request_budget is not None:
            if request_budget.remaining == 0:
                raise RuntimeError("Alpha Vantage daily request budget exhausted")
            request_budget.used += 1
        sleeper(_ALPHA_VANTAGE_REQUEST_SPACING_SECONDS)
        try:
            return alpha_vantage.fetch_daily(
                symbol,
                api_key,
                timeout=_ALPHA_VANTAGE_REQUEST_TIMEOUT_SECONDS,
                retries=_ALPHA_VANTAGE_REQUEST_RETRIES,
            )
        except alpha_vantage.RateLimitResponse:
            if attempt == 1:
                raise
    raise RuntimeError("Alpha Vantage retry loop ended without a response")


def _abroad_skip_reason(error: Exception) -> str:
    if isinstance(error, alpha_vantage.RateLimitResponse):
        return "rate_limited"
    if isinstance(error, alpha_vantage.UnknownSymbolResponse):
        return "unknown_symbol"
    if isinstance(error, alpha_vantage.NoSeriesResponse):
        return "no_series"
    return "http_error"


def _new_abroad_skip_counts() -> dict[str, int]:
    return dict.fromkeys(_ABROAD_SKIP_REASONS, 0)


def _fetch_abroad_etfs(
    on: date,
    *,
    sleeper: Callable[[float], None] = sleep_for,
    request_budget: AbroadRequestBudget | None = None,
) -> list[AbroadPrice | CuratedAbroadPrice] | AbroadBatch:
    api_key = os.getenv("PRICE_API_KEY", "")
    if not api_key:
        return []
    budget = request_budget or AbroadRequestBudget()
    from wealth.policy.denmark_picks import load_eu_etfs

    records: list[AbroadPrice | CuratedAbroadPrice] = []
    skipped = _new_abroad_skip_counts()
    instruments = load_eu_etfs()["instruments"][:_MAX_ABROAD_QUOTES_PER_DAY]
    for instrument in instruments:
        listing = _choose_curated_listing(instrument)
        if listing is None:
            skipped["no_supported_listing"] += 1
            continue
        try:
            close = _fetch_alpha_vantage_daily(
                listing.provider_symbol,
                api_key,
                sleeper=sleeper,
                request_budget=budget,
            )
            if close.as_of <= on:
                records.append(CuratedAbroadPrice(instrument, listing, close))
        except Exception as error:
            skipped[_abroad_skip_reason(error)] += 1
    return AbroadBatch(records, len(instruments), skipped)


def _fetch_abroad_us_stocks(
    securities: list[tuple[Any, str]],
    on: date,
    *,
    curated_count: int,
    api_key: str | None = None,
    sleeper: Callable[[float], None] = sleep_for,
    request_budget: AbroadRequestBudget | None = None,
) -> list[AbroadPrice] | AbroadBatch:
    selected_api_key = api_key if api_key is not None else os.getenv("PRICE_API_KEY", "")
    if not selected_api_key:
        return []
    budget = request_budget or AbroadRequestBudget(
        used=_MAX_ABROAD_QUOTES_PER_DAY - _remaining_abroad_quote_slots(curated_count)
    )
    records: list[AbroadPrice] = []
    skipped = _new_abroad_skip_counts()
    selected = securities[: budget.remaining]
    for security_id, symbol in selected:
        try:
            close = _fetch_alpha_vantage_daily(
                symbol,
                selected_api_key,
                sleeper=sleeper,
                request_budget=budget,
            )
            if close.as_of <= on:
                records.append(AbroadPrice(security_id, close))
        except Exception as error:
            skipped[_abroad_skip_reason(error)] += 1
    return AbroadBatch(records, len(selected), skipped)


def _remaining_abroad_quote_slots(curated_count: int) -> int:
    """Reserve one free-tier call for each curated fund, capped at 24 calls daily."""
    reserved = min(_MAX_ABROAD_QUOTES_PER_DAY, max(0, curated_count))
    return _MAX_ABROAD_QUOTES_PER_DAY - reserved


def _held_us_stock_keys(session: Session) -> list[tuple[Any, str]]:
    return [
        (security_id, symbol)
        for security_id, symbol in session.execute(
            select(Security.id, Security.symbol)
            .join(HoldingLot, HoldingLot.security_id == Security.id)
            .where(
                Security.market == "US",
                Security.security_type == "equity",
                Security.symbol.is_not(None),
            )
            .distinct()
            .order_by(Security.symbol, Security.id)
        )
        if symbol
    ]


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
    return _fetch_edgar_for_symbols(on, _universe_symbols(session, "US", "sp500", symbols))


def _fetch_edgar_for_symbols(on: date, target_symbols: list[str]) -> FundamentalsBatch:
    client = edgar.SecEdgarClient()
    companies = edgar.fetch_company_tickers(client)
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
    targets = [
        NseFundamentalTarget(
            symbol=symbol,
            security_id=security.id,
            after=store.latest_fundamental_filed_at(session, security.id, "NSE_XBRL"),
        )
        for symbol in target_symbols
        if (security := securities.get(symbol.upper())) is not None
    ]
    missing = [symbol for symbol in target_symbols if symbol.upper() not in securities]
    return _fetch_nse_fundamental_targets(on, targets, missing)


def _fetch_nse_fundamental_targets(
    on: date,
    targets: list[NseFundamentalTarget],
    missing: list[str] | None = None,
) -> FundamentalsBatch:
    observations: dict[str, list[FundamentalObservation]] = {}
    errors: dict[str, str] = {}
    xbrl_missing = 0
    errors.update({symbol: "SecurityNotFound" for symbol in missing or []})
    with NseIntegratedClient() as client:  # type: ignore[attr-defined]
        for target in targets:
            symbol = target.symbol
            try:
                result = nse_fundamentals.fetch_symbol_fundamentals(
                    client,
                    symbol,
                    as_of=on,
                    after=target.after,
                )
                xbrl_missing += result.xbrl_missing
                if result.xbrl_missing and not result.readable_filings:
                    errors[symbol] = "XbrlMissing"
                    continue
                observations[symbol] = result.observations
            except Exception as error:
                errors[symbol] = type(error).__name__
    return FundamentalsBatch(observations, errors, xbrl_missing=xbrl_missing)


def _fetch_us_index_pe(session: Session, on: date) -> us_index_pe.UsIndexPeBatch:
    """Build the S&P valuation from the current SSGA snapshot and stored SEC facts."""
    holdings = universes.fetch_sp500()
    workbook = universes.get(universes.SPY_NAV_HISTORY_URL).content
    nav_history = universes.parse_spy_nav_history(workbook)
    return _build_us_index_pe(session, on, holdings, nav_history)


def _fetch_us_index_pe_for_engine(engine: Engine, on: date) -> us_index_pe.UsIndexPeBatch:
    """Download public inputs before opening the short database read transaction."""
    holdings = universes.fetch_sp500()
    workbook = universes.get(universes.SPY_NAV_HISTORY_URL).content
    nav_history = universes.parse_spy_nav_history(workbook)
    with system_session(engine) as session:
        return _build_us_index_pe(session, on, holdings, nav_history)


def _build_us_index_pe(
    session: Session,
    on: date,
    holdings: Sp500Holdings,
    nav_history: dict[date, Decimal],
) -> us_index_pe.UsIndexPeBatch:
    symbols = {member.symbol for member in holdings.securities}
    security_rows = session.execute(
        select(Security.id, Security.symbol)
        .join(UniverseMember, UniverseMember.security_id == Security.id)
        .where(
            UniverseMember.universe == "sp500",
            UniverseMember.removed_on.is_(None),
            Security.market == "US",
            Security.symbol.in_(symbols),
        )
    ).all()
    security_ids = {security_id for security_id, _symbol in security_rows}
    if not security_ids:
        return us_index_pe.UsIndexPeBatch([], {"universe": "NoCurrentSp500Members"})

    price_rows = session.execute(
        select(PriceEod.security_id, PriceEod.as_of, PriceEod.close)
        .where(
            PriceEod.security_id.in_(security_ids),
            PriceEod.as_of <= holdings.as_of,
            PriceEod.as_of >= holdings.as_of - timedelta(days=us_index_pe.MAX_PRICE_AGE_DAYS),
            PriceEod.source == universes.SP500_PRICE_SOURCE,
        )
        .order_by(PriceEod.security_id, PriceEod.as_of)
    ).all()
    facts = list(
        session.scalars(
            select(FundamentalPIT)
            .where(
                FundamentalPIT.security_id.in_(security_ids),
                FundamentalPIT.metric.in_({"eps_diluted", "net_income"}),
                FundamentalPIT.period_end <= holdings.as_of,
                FundamentalPIT.known_at
                <= datetime.combine(holdings.as_of, datetime.max.time(), tzinfo=UTC),
            )
            .order_by(
                FundamentalPIT.security_id, FundamentalPIT.period_end, FundamentalPIT.known_at
            )
        )
    )
    prices_by_id: dict[Any, list[tuple[date, Decimal]]] = {}
    facts_by_id: dict[Any, list[FundamentalPIT]] = {}
    for row in price_rows:
        prices_by_id.setdefault(row.security_id, []).append((row.as_of, row.close))
    for row in facts:
        facts_by_id.setdefault(row.security_id, []).append(row)
    constituents = {
        str(symbol): us_index_pe.Sp500ConstituentData(
            prices=prices_by_id.get(security_id, []),
            facts=facts_by_id.get(security_id, []),
        )
        for security_id, symbol in security_rows
        if symbol is not None
    }
    return us_index_pe.build_sp500_pe_valuations(
        holdings,
        constituents,
        nav_history,
        on=on,
    )


def _fetch_universes(
    *,
    fetch_sp500: Callable[[], Sp500Holdings] | None = None,
    fetch_nifty500: Callable[[], list[NseSecurity]] | None = None,
) -> UniverseBatch:
    sp500: Sp500Holdings | None = None
    nifty500: list[NseSecurity] | None = None
    errors: dict[str, str] = {}
    sp500_closes: dict[str, Decimal] = {}
    sp500_sectors: dict[str, Sp500Sector] = {}
    try:
        sp500 = (fetch_sp500 or universes.fetch_sp500)()
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
        nifty500 = (fetch_nifty500 or universes.fetch_nifty500)()
    except Exception as error:
        errors["nifty500"] = type(error).__name__
    return UniverseBatch(sp500, nifty500, errors, sp500_closes, sp500_sectors)


def _default_fetchers(
    on: date,
    session: Session | None = None,
    symbols: tuple[str, ...] | None = None,
    request_budget: AbroadRequestBudget | None = None,
) -> dict[str, Callable[[], Any]]:
    abroad_budget = request_budget or AbroadRequestBudget()
    cached_reference_data: dict[str, Any] = {}

    def cached_reference(key: str, fetch: Callable[[], Any]) -> Any:
        if key not in cached_reference_data:
            try:
                cached_reference_data[key] = fetch()
            except Exception as error:
                cached_reference_data[key] = error
        result = cached_reference_data[key]
        if isinstance(result, Exception):
            raise result
        return result

    def shared_sp500() -> Sp500Holdings:
        return cached_reference("sp500", universes.fetch_sp500)

    def shared_nifty500() -> list[NseSecurity]:
        return cached_reference("nifty500", universes.fetch_nifty500)

    def fetch_universes() -> UniverseBatch:
        return _fetch_universes(fetch_sp500=shared_sp500, fetch_nifty500=shared_nifty500)

    def fetch_fund_baskets() -> fund_baskets.FundBasketBatch:
        return fund_baskets.fetch_fund_baskets(
            on,
            fetch_sp500=shared_sp500,
            fetch_nifty500=shared_nifty500,
        )

    def fetch_us_stocks() -> list[AbroadPrice] | AbroadBatch:
        if not os.getenv("PRICE_API_KEY", "") or session is None:
            return []
        from wealth.policy.denmark_picks import load_eu_etfs

        return _fetch_abroad_us_stocks(
            _held_us_stock_keys(session),
            on,
            curated_count=len(load_eu_etfs()["instruments"]),
            request_budget=abroad_budget,
        )

    return {
        "nse": lambda: _fetch_recent_bhavcopies(on),
        "amfi": amfi.fetch_nav_all,
        "fx": lambda: _fetch_fx(on),
        "gold": lambda: ibja.fetch_gold_rate(on=on),
        "master": _fetch_nse_master,
        "amfi_history": (
            lambda: _fetch_missing_amfi_history(session, on) if session is not None else []
        ),
        "eu_etfs": lambda: _fetch_abroad_etfs(on, request_budget=abroad_budget),
        "us_stocks": fetch_us_stocks,
        "universes": fetch_universes,
        "fund_baskets": fetch_fund_baskets,
        "edgar": lambda: _fetch_edgar(session, on, symbols),
        "us_index_pe": lambda: _fetch_us_index_pe(session, on) if session is not None else None,
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
        edgar_securities = (
            _securities_for_symbols(
                session, "US", [ticker.upper() for ticker in records.observations]
            )
            if source == "edgar"
            else {}
        )
        for ticker, observations in records.observations.items():
            try:
                with session.begin_nested():
                    if source == "edgar":
                        company = (records.companies or {}).get(ticker)
                        if company is None:
                            raise ValueError("TickerNotFound")
                        security = edgar_securities.get(ticker.upper())
                        security_id = (
                            security.id
                            if security is not None
                            else store.upsert_us_equity_security(session, ticker, company.name)
                        )
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
        counts = {"fundamentals_pit": inserted, "symbols": stored_symbols}
        if source == "nse_fundamentals":
            counts["xbrl_missing"] = records.xbrl_missing
        return SourceBatchResult(counts, errors)
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
                        listed = records.sp500_sectors.get(row.symbol)
                        name = listed.name if listed is not None and listed.name else row.name
                        security_id = store.upsert_us_equity_security(session, row.symbol, name)
                        security_ids.add(security_id)
                        # SSGA now lists every sector as "-"; the GICS table fills the gap.
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
    if source == "fund_baskets":
        if not isinstance(records, fund_baskets.FundBasketBatch):
            raise TypeError("fund-basket adapter returned invalid rows")
        errors = dict(records.errors)
        counts: dict[str, Any] = {"baskets_stored": 0, "baskets": {}}
        for basket_id, snapshot in records.baskets.items():
            try:
                with session.begin_nested():
                    basket = session.get(FundBasket, basket_id)
                    if basket is None:
                        basket = FundBasket(
                            id=basket_id,
                            label=snapshot.label,
                            as_of=snapshot.as_of,
                            source=snapshot.source,
                            source_url=snapshot.source_url,
                            weights_estimated=snapshot.weights_estimated,
                            weight_source=snapshot.weight_source,
                            members_total=snapshot.members_total,
                            members_found=snapshot.members_found,
                            fetched_at=fetched_at,
                        )
                        session.add(basket)
                    else:
                        basket.label = snapshot.label
                        basket.as_of = snapshot.as_of
                        basket.source = snapshot.source
                        basket.source_url = snapshot.source_url
                        basket.weights_estimated = snapshot.weights_estimated
                        basket.weight_source = snapshot.weight_source
                        basket.members_total = snapshot.members_total
                        basket.members_found = snapshot.members_found
                        basket.fetched_at = fetched_at
                    session.flush()
                    session.execute(
                        delete(FundBasketWeight).where(FundBasketWeight.basket_id == basket_id)
                    )
                    session.add_all(
                        [
                            FundBasketWeight(
                                basket_id=basket_id,
                                market=row.market,
                                isin=row.isin,
                                symbol=row.symbol,
                                name=row.name,
                                weight_pct=row.weight_pct.quantize(
                                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                                ),
                            )
                            for row in snapshot.weights
                        ]
                    )
                    session.flush()
                counts["baskets"][basket_id] = {
                    "rows": len(snapshot.weights),
                    "members_total": snapshot.members_total,
                    "members_found": snapshot.members_found,
                    "as_of": snapshot.as_of.isoformat(),
                }
                counts["baskets_stored"] += 1
            except Exception as error:
                errors[basket_id] = type(error).__name__
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
    if source == "us_index_pe":
        if not isinstance(records, us_index_pe.UsIndexPeBatch):
            raise TypeError("S&P index-valuation adapter returned invalid rows")
        backfill_dates = [
            item.as_of
            for item in records.observations
            if item.source == us_index_pe.BACKFILL_SOURCE
        ]
        session.execute(
            delete(IndexValuation).where(
                IndexValuation.index_name == us_index_pe.INDEX_NAME,
                IndexValuation.source == us_index_pe.BACKFILL_SOURCE,
                IndexValuation.as_of.not_in(backfill_dates),
            )
        )
        backfill_count = sum(
            1 for item in records.observations if item.source == us_index_pe.BACKFILL_SOURCE
        )
        current_count = sum(
            1 for item in records.observations if item.source == us_index_pe.CURRENT_SOURCE
        )
        count = store.upsert_index_valuations(session, records.observations)
        return SourceBatchResult(
            {
                "index_valuations": count,
                "current_points": current_count,
                "backfill_points": backfill_count,
            },
            dict(records.errors),
        )
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
        if isinstance(records, AbroadBatch):
            inserted = 0
            for item in records.records:
                if isinstance(item, CuratedAbroadPrice):
                    security_id, _provider_symbol = _upsert_curated_security(
                        session,
                        item.instrument,
                        item.listing,
                    )
                else:
                    security_id = item.security_id
                if security_id is None:
                    raise ValueError("Alpha Vantage quote has no matching security")
                store.upsert_eod_price(
                    session,
                    security_id,
                    item.close.as_of,
                    item.close.close,
                    source="AlphaVantage",
                    source_url=alpha_vantage.SOURCE_URL,
                    fetched_at=fetched_at,
                )
                inserted += 1
            counts: dict[str, Any] = {
                "prices_eod": inserted,
                "requested": records.requested,
                "skipped": records.skipped,
            }
            errors = {
                reason: f"{amount} skipped" for reason, amount in records.skipped.items() if amount
            }
            return SourceBatchResult(counts, errors)
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


def _prepare_engine_fetchers(
    engine: Engine,
    on: date,
    selected_sources: tuple[str, ...],
    symbols: tuple[str, ...] | None,
    overrides: Mapping[str, Callable[[], Any]] | None,
) -> dict[str, Callable[[], Any]]:
    abroad_budget = AbroadRequestBudget()
    fetchers = _default_fetchers(on, symbols=symbols, request_budget=abroad_budget)
    fetchers.update(overrides or {})
    overridden = set(overrides or {})
    prepared_windows: list[tuple[date, date]] = []
    prepared_us_stocks: list[tuple[Any, str]] = []
    curated_count = 0
    edgar_symbols: list[str] = []
    nse_targets: list[NseFundamentalTarget] = []
    nse_missing: list[str] = []

    needs_preparation = {
        source
        for source in ("amfi_history", "us_stocks", "edgar", "nse_fundamentals")
        if source in selected_sources and source not in overridden
    }
    if needs_preparation:
        with system_session(engine) as session:
            if "amfi_history" in needs_preparation:
                prepared_windows = store.missing_amfi_history_windows(session, on)
            if "us_stocks" in needs_preparation:
                prepared_us_stocks = _held_us_stock_keys(session)
            if "edgar" in needs_preparation:
                edgar_symbols = _universe_symbols(session, "US", "sp500", symbols)
            if "nse_fundamentals" in needs_preparation:
                target_symbols = _universe_symbols(session, "IN", "nifty500", symbols)
                if target_symbols:
                    securities = _securities_for_symbols(session, "IN", target_symbols)
                    nse_targets = [
                        NseFundamentalTarget(
                            symbol=symbol,
                            security_id=security.id,
                            after=store.latest_fundamental_filed_at(
                                session, security.id, "NSE_XBRL"
                            ),
                        )
                        for symbol in target_symbols
                        if (security := securities.get(symbol.upper())) is not None
                    ]
                    nse_missing = [
                        symbol for symbol in target_symbols if symbol.upper() not in securities
                    ]
                else:
                    nse_missing = ["universe"]

    if "amfi_history" in needs_preparation:
        fetchers["amfi_history"] = lambda: _fetch_amfi_history_windows(prepared_windows)
    if "eu_etfs" in selected_sources and "eu_etfs" not in overridden:
        fetchers["eu_etfs"] = lambda: _fetch_abroad_etfs(
            on,
            request_budget=abroad_budget,
        )
    if "us_stocks" in needs_preparation:
        from wealth.policy.denmark_picks import load_eu_etfs

        curated_count = len(load_eu_etfs()["instruments"])
        fetchers["us_stocks"] = lambda: _fetch_abroad_us_stocks(
            prepared_us_stocks,
            on,
            curated_count=curated_count,
            request_budget=abroad_budget,
        )
    if "edgar" in needs_preparation:
        fetchers["edgar"] = lambda: _fetch_edgar_for_symbols(on, edgar_symbols)
    if "nse_fundamentals" in needs_preparation:
        if nse_missing == ["universe"]:
            fetchers["nse_fundamentals"] = lambda: FundamentalsBatch({}, {"universe": "NoSymbols"})
        else:
            fetchers["nse_fundamentals"] = lambda: _fetch_nse_fundamental_targets(
                on,
                nse_targets,
                nse_missing,
            )
    if "us_index_pe" in selected_sources and "us_index_pe" not in overridden:
        fetchers["us_index_pe"] = lambda: _fetch_us_index_pe_for_engine(engine, on)
    return fetchers


def _refresh_sources_succeeded(
    selected_sources: tuple[str, ...], errors: Mapping[str, str], counts: Mapping[str, Any]
) -> int:
    successful = 0
    for source in selected_sources:
        source_errors = [key for key in errors if key == source or key.startswith(f"{source}:")]
        source_counts = counts.get(source)
        if (
            not source_errors
            or (
                source in {"eu_etfs", "us_stocks"}
                and isinstance(source_counts, dict)
                and source_counts.get("prices_eod", 0) > 0
            )
            or (
                source == "fx"
                and isinstance(source_counts, dict)
                and source_counts.get("fx_rates", 0) > 0
            )
            or (
                source in {"edgar", "nse_fundamentals"}
                and isinstance(source_counts, dict)
                and source_counts.get("symbols", 0) > 0
            )
            or (
                source in {"index_pe", "us_index_pe"}
                and isinstance(source_counts, dict)
                and source_counts.get("index_valuations", 0) > 0
            )
            or (
                source == "ust"
                and isinstance(source_counts, dict)
                and source_counts.get("macro_series", 0) > 0
            )
            or (
                source == "feeds"
                and isinstance(source_counts, dict)
                and source_counts.get("feed_items", 0) > 0
            )
            or (
                source == "universes"
                and isinstance(source_counts, dict)
                and source_counts.get("universes_loaded", 0) > 0
            )
            or (
                source == "fund_baskets"
                and isinstance(source_counts, dict)
                and source_counts.get("baskets_stored", 0) > 0
            )
        ):
            successful += 1
    return successful


def _run_refresh_on_engine(
    engine: Engine,
    on: date,
    only: tuple[str, ...] | list[str] | None = None,
    *,
    fetchers: Mapping[str, Callable[[], Any]] | None = None,
    prune_old: bool = False,
    fetched_at: datetime | None = None,
    symbols: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    selected_sources = tuple(
        source for source in (only or _SOURCES) if only or source != "nse_history"
    )
    if "nse_history" in selected_sources:
        raise ValueError("nse_history commits day by day; use run_nse_history")
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
    effective_fetchers = _prepare_engine_fetchers(
        engine,
        on,
        selected_sources,
        normalized_symbols,
        fetchers,
    )
    started_at = fetched_at or datetime.now(UTC)
    counts: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for retention_name, operation in (
        ("fundamentals_pit_retention", store.prune_fundamentals),
        ("feed_items_retention", store.prune_feed_items),
        ("index_valuations_retention", store.prune_index_valuations),
        ("macro_series_retention", store.prune_macro_series),
    ):
        try:
            with system_session(engine) as session:
                if retention_name == "feed_items_retention":
                    deleted = operation(
                        session,
                        on,
                        retention_days=macro_geo.FEED_RETENTION_DAYS,
                    )
                elif retention_name == "fundamentals_pit_retention":
                    deleted = operation(session, on)
                else:
                    deleted = operation(session, on)
            if deleted:
                counts[f"{retention_name.removesuffix('_retention')}_pruned"] = deleted
        except Exception as error:
            errors[retention_name] = type(error).__name__

    for source in selected_sources:
        fetcher = effective_fetchers.get(source)
        if fetcher is None:
            errors[source] = "MissingAdapter"
            continue
        source_name: str | None = None
        try:
            if source == "gold":
                records, source_name = _run_gold_fetch(fetcher)
            else:
                records = fetcher()
            if source == "fx" and isinstance(records, FxFetchResult):
                for currency, error_code in records.errors.items():
                    errors[f"fx:{currency}"] = error_code
                records = list(records.observations)
        except Exception as error:
            errors[source] = type(error).__name__
            continue

        try:
            with system_session(engine) as session:
                result = _store_source(
                    session,
                    source,
                    records,
                    on=on,
                    fetched_at=started_at,
                )
            if isinstance(result, SourceBatchResult):
                counts[source] = result.counts
                errors.update({f"{source}:{key}": value for key, value in result.errors.items()})
            else:
                counts[source] = result
            if source_name == "manual":
                counts["gold_fallback"] = "manual"
            if source == "master" and isinstance(records, NseMasterBatch):
                counts["master_etf"] = records.etf_count
                if records.etf_error is not None:
                    errors["etf_master"] = records.etf_error
        except Exception as error:
            errors[source] = type(error).__name__

    if prune_old:
        try:
            with system_session(engine) as session:
                counts["pruned"] = store.prune(session, on, retention_days=_RETENTION_DAYS)
        except Exception as error:
            errors["prune"] = type(error).__name__

    successful_sources = _refresh_sources_succeeded(selected_sources, errors, counts)
    status = "succeeded" if not errors else "partial" if successful_sources else "failed"
    finished_at = datetime.now(UTC)
    with system_session(engine) as session:
        run = MarketdataRun(
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            counts=counts,
            errors=errors,
        )
        session.add(run)
        session.flush()
        run_id = str(run.id)
    return {"run_id": run_id, "status": status, "counts": counts, "errors": errors}


def run_refresh(
    session: Session | Engine,
    on: date,
    only: tuple[str, ...] | list[str] | None = None,
    *,
    fetchers: Mapping[str, Callable[[], Any]] | None = None,
    prune_old: bool = False,
    fetched_at: datetime | None = None,
    symbols: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Fetch selected sources, continue after failures, and record one run row."""
    if isinstance(session, Engine):
        return _run_refresh_on_engine(
            session,
            on,
            only,
            fetchers=fetchers,
            prune_old=prune_old,
            fetched_at=fetched_at,
            symbols=symbols,
        )
    if fetchers is None:
        if session.in_transaction():
            raise ValueError("Pass an Engine to fetch refresh sources outside a transaction")
        return _run_refresh_on_engine(
            session.get_bind(),
            on,
            only,
            prune_old=prune_old,
            fetched_at=fetched_at,
            symbols=symbols,
        )
    selected_sources = tuple(
        source for source in (only or _SOURCES) if only or source != "nse_history"
    )
    if "nse_history" in selected_sources:
        raise ValueError("nse_history commits day by day; use run_nse_history")
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

    successful_sources = _refresh_sources_succeeded(selected_sources, errors, counts)
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


def run_nse_history(
    engine: Engine,
    on: date,
    *,
    fetch: Callable[[date], list[NseClose]] | None = None,
) -> dict[str, Any]:
    """Load weekday sessions in the retention window that have no NIFTY 500 closes.

    Momentum and volatility need 200 sessions, so the first run loads about a year of
    official bhavcopies; later runs only retry gaps (holidays return no file). Each day
    commits on its own: one transaction across ~285 downloads lost its Neon connection and
    rolled everything back, while a day-sized commit keeps progress for the next run.
    """
    fetch_day = fetch or nse.fetch_bhavcopy
    started_at = datetime.now(UTC)
    start = on - timedelta(days=_RETENTION_DAYS)
    with system_session(engine) as session:
        loaded = store.loaded_nse_sessions(session, start, on)
        candidates = store.nse_price_keys(session)
    wanted = [
        day
        for day in (start + timedelta(days=offset) for offset in range((on - start).days + 1))
        if day.weekday() < 5 and day not in loaded
    ]
    stored = 0
    failed = 0
    last_error = ""
    for day in wanted:
        try:
            # Keep only rows a stored security can match; a full file has ~3,600 rows.
            rows = [
                row
                for row in fetch_day(day)
                if row.symbol in candidates or (row.isin or "") in candidates
            ]
            if rows:
                with system_session(engine) as session:
                    stored += store.upsert_nse_closes(session, rows, fetched_at=started_at)
        except Exception as error:
            failed += 1
            last_error = type(error).__name__
    counts: dict[str, Any] = {
        "nse_history": {"prices_eod": stored, "sessions_requested": len(wanted)}
    }
    errors = (
        {"nse_history:failed_sessions": str(failed), "nse_history:last_error": last_error}
        if failed
        else {}
    )
    status = "succeeded" if not failed else "partial" if stored else "failed"
    with system_session(engine) as session:
        run = MarketdataRun(
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status=status,
            counts=counts,
            errors=errors,
        )
        session.add(run)
        session.flush()
        run_id = str(run.id)
    return {"run_id": run_id, "status": status, "counts": counts, "errors": errors}


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

    engine = make_engine(database_url, pool_pre_ping=True)
    others = tuple(source for source in args.sources if source != "nse_history")
    results: list[dict[str, Any]] = []
    try:
        if others:
            results.append(
                run_refresh(
                    engine,
                    on,
                    others,
                    prune_old=args.prune,
                    symbols=args.symbols,
                )
            )
        if "nse_history" in args.sources:
            results.append(run_nse_history(engine, on))
    finally:
        engine.dispose()
    statuses = {item["status"] for item in results}
    result = {
        "run_id": ",".join(item["run_id"] for item in results),
        "status": "failed"
        if "failed" in statuses
        else "partial"
        if "partial" in statuses
        else "succeeded",
        "counts": {key: value for item in results for key, value in item["counts"].items()},
        "errors": {key: value for item in results for key, value in item["errors"].items()},
    }
    print(json.dumps({"date": on.isoformat(), **result}, sort_keys=True))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
