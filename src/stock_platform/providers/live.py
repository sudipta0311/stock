from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from stock_platform.utils.index_config import NSE_INDEX_CSV_URLS
from stock_platform.utils.rules import clamp
from stock_platform.utils.screener_fetcher import get_stock_fundamentals
from stock_platform.utils.sector_config import get_sector
from stock_platform.utils.symbol_resolver import get_symbol_display_name, resolve_nse_symbol, resolve_symbol_base

# Disk-cache directory for index constituents (7-day TTL — indices rebalance monthly).
_INDEX_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "index_constituents"
_INDEX_CACHE_TTL_DAYS = 7


class LiveMarketDataProvider:
    """Live market-data provider backed by NSE constituents, yfinance, and AMC look-through data."""

    # All supported index CSV URLs — sourced from index_config.NSE_INDEX_CSV_URLS.
    INDEX_URLS = NSE_INDEX_CSV_URLS

    def __init__(self, holdings_client: Any | None = None, repo: Any | None = None) -> None:
        self.holdings_client = holdings_client
        self.repo = repo
        self.today = date.today()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
                )
            }
        )
        self._index_cache: dict[str, list[dict[str, Any]]] = {}
        self._info_cache: dict[str, dict[str, Any]] = {}
        self._snapshot_cache: dict[str, dict[str, Any]] = {}
        self._history_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._financial_cache: dict[str, dict[str, Any]] = {}
        self._sector_overview: dict[str, dict[str, Any]] | None = None

    def _ticker_symbol(self, symbol: str) -> str:
        return resolve_nse_symbol(symbol)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _series_first(frame: Any, labels: list[str], *, latest_two: bool = False) -> float | list[float] | None:
        if frame is None or getattr(frame, "empty", True):
            return None
        for label in labels:
            if label not in frame.index:
                continue
            values = []
            for raw in frame.loc[label].tolist():
                numeric = LiveMarketDataProvider._as_float(raw)
                if numeric is not None:
                    values.append(numeric)
            if not values:
                continue
            return values[:2] if latest_two else values[0]
        return None

    @staticmethod
    def _series_values(frame: Any, labels: list[str], *, limit: int | None = None) -> list[float]:
        if frame is None or getattr(frame, "empty", True):
            return []
        for label in labels:
            if label not in frame.index:
                continue
            values: list[float] = []
            for raw in frame.loc[label].tolist():
                numeric = LiveMarketDataProvider._as_float(raw)
                if numeric is not None:
                    values.append(numeric)
            return values[:limit] if limit is not None else values
        return []

    @staticmethod
    def _consecutive_negative_quarters(frame: Any) -> int | None:
        if frame is None or getattr(frame, "empty", True):
            return None
        for label in [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
        ]:
            if label not in frame.index:
                continue
            negatives = 0
            for raw in frame.loc[label].tolist():
                numeric = LiveMarketDataProvider._as_float(raw)
                if numeric is None:
                    continue
                if numeric < 0:
                    negatives += 1
                else:
                    break
            return negatives
        return None

    @staticmethod
    def _percent_return(closes: pd.Series, periods_back: int) -> float | None:
        clean = closes.dropna()
        if len(clean) <= periods_back:
            return None
        base = float(clean.iloc[-periods_back - 1])
        current = float(clean.iloc[-1])
        if base == 0:
            return None
        return (current / base) - 1.0

    @staticmethod
    def _market_conviction(score: float) -> str:
        if score >= 0.72:
            return "BUY"
        if score >= 0.58:
            return "POSITIVE"
        if score >= 0.42:
            return "NEUTRAL"
        return "AVOID"

    def _parse_index_csv_text(self, text: str) -> list[dict[str, Any]]:
        """Parse NSE index CSV text into a list of symbol dicts."""
        rows: list[dict[str, Any]] = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            symbol = self.normalize_symbol(row.get("Symbol") or row.get("symbol") or "")
            if not symbol:
                continue
            rows.append({
                "symbol": symbol,
                "company_name": (row.get("Company Name") or row.get("Company") or symbol).strip(),
                "sector": (row.get("Industry") or row.get("Sector") or "Unknown").strip() or "Unknown",
            })
        return rows

    def _index_cache_key(self, index_name: str) -> str:
        return f"index_constituents:{index_name}"

    def _load_db_index_cache(self, index_name: str, *, allow_expired: bool = False) -> list[dict[str, Any]]:
        if self.repo is None:
            return []
        rows = self.repo.get_cache(
            self._index_cache_key(index_name),
            default=[],
            allow_expired=allow_expired,
        )
        return rows if isinstance(rows, list) else []

    def _persist_index_cache(self, index_name: str, rows: list[dict[str, Any]]) -> None:
        if self.repo is not None and rows:
            self.repo.set_cache(
                self._index_cache_key(index_name),
                rows,
                ttl_seconds=_INDEX_CACHE_TTL_DAYS * 86400,
            )

    def _download_index_csv(self, index_name: str) -> list[dict[str, Any]]:
        # L1: in-memory cache (per session).
        if index_name in self._index_cache:
            return self._index_cache[index_name]

        # L2: DB cache - shared across app sessions and synced to Turso.
        db_rows = self._load_db_index_cache(index_name)
        if db_rows:
            self._index_cache[index_name] = db_rows
            return db_rows

        # L3: disk cache with 7-day TTL.
        cache_file = _INDEX_CACHE_DIR / f"{index_name}.json"
        if cache_file.exists():
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days < _INDEX_CACHE_TTL_DAYS:
                try:
                    rows = json.loads(cache_file.read_text(encoding="utf-8"))
                    self._persist_index_cache(index_name, rows)
                    self._index_cache[index_name] = rows
                    return rows
                except Exception:
                    pass  # corrupt cache — fall through to fresh fetch

        # L4: fetch from NSE archives.
        url = self.INDEX_URLS.get(index_name, self.INDEX_URLS["NIFTY50"])
        rows: list[dict[str, Any]] = []
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            rows = self._parse_index_csv_text(response.text)
        except Exception:
            rows = []

        if rows:
            self._persist_index_cache(index_name, rows)
            try:
                _INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(rows), encoding="utf-8")
            except Exception:
                pass  # disk write failure is non-fatal

        self._index_cache[index_name] = rows
        return rows

    def _load_stale_index_cache(self, index_name: str) -> list[dict[str, Any]]:
        db_rows = self._load_db_index_cache(index_name, allow_expired=True)
        if db_rows:
            return db_rows
        cache_file = _INDEX_CACHE_DIR / f"{index_name}.json"
        if not cache_file.exists():
            return []
        try:
            rows = json.loads(cache_file.read_text(encoding="utf-8"))
            return rows if isinstance(rows, list) else []
        except Exception:
            return []

    def _fallback_index_members(self, index_name: str) -> list[dict[str, Any]]:
        """
        Last-resort offline universe so the buy flow can still proceed when
        NSE archive downloads are unavailable on hosted environments.
        """
        from stock_platform.providers.demo import DemoDataProvider

        demo = DemoDataProvider()
        if index_name in demo.index_members:
            return demo.get_index_members(index_name)

        combined: dict[str, dict[str, Any]] = {}
        for fallback_index in ("NIFTY50", "NIFTYNEXT50"):
            for row in demo.get_index_members(fallback_index):
                combined.setdefault(row["symbol"], row)
        return list(combined.values())

    def _combined_universe(self) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        for index_name in ("NIFTY50", "NIFTYNEXT50"):
            for row in self._download_index_csv(index_name):
                combined.setdefault(row["symbol"], row)
        return list(combined.values())

    def _lookup_cached_index_row(self, symbol: str) -> dict[str, Any] | None:
        normalized = self.normalize_symbol(symbol)
        for row in self._combined_universe():
            if row["symbol"] == normalized:
                return row
        return None

    def _should_skip_market_lookup(self, symbol: str) -> bool:
        normalized = self.normalize_symbol(symbol)
        if self._lookup_cached_index_row(normalized):
            return False
        return len(normalized) > 12 or normalized[:1].isdigit()

    def _get_info(self, symbol: str) -> dict[str, Any]:
        normalized = self.normalize_symbol(symbol)
        if normalized in self._info_cache:
            return self._info_cache[normalized]
        info: dict[str, Any] = {}
        if self._should_skip_market_lookup(normalized):
            self._info_cache[normalized] = {}
            return {}
        try:
            info = yf.Ticker(self._ticker_symbol(normalized)).info or {}
        except Exception:
            info = {}
        self._info_cache[normalized] = info
        return info

    def _get_history(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        normalized = self.normalize_symbol(symbol)
        cache_key = (normalized, period)
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]
        if self._should_skip_market_lookup(normalized):
            self._history_cache[cache_key] = pd.DataFrame()
            return self._history_cache[cache_key]
        try:
            frame = yf.Ticker(self._ticker_symbol(normalized)).history(
                period=period,
                interval="1d",
                auto_adjust=False,
            )
        except Exception:
            frame = pd.DataFrame()
        self._history_cache[cache_key] = frame
        return frame

    def _compute_price_metrics_from_history(self, history: pd.DataFrame) -> dict[str, Any]:
        if history.empty:
            return {}
        closes = history.get("Close", pd.Series(dtype=float)).dropna()
        volumes = history.get("Volume", pd.Series(dtype=float)).fillna(0.0)
        if closes.empty:
            return {}
        current_price = float(closes.iloc[-1])
        ret_1m = self._percent_return(closes, 21)
        ret_3m = self._percent_return(closes, 63)
        ret_6m = self._percent_return(closes, min(max(len(closes) - 1, 1), 126))
        high_6m = float(closes.max())
        drawdown = (current_price / high_6m) - 1.0 if high_6m else None
        aligned_close = closes.reindex(history.index).ffill()
        daily_value = (aligned_close * volumes).dropna().tail(21)
        avg_daily_value_cr = float(daily_value.mean() / 1e7) if not daily_value.empty else None
        return {
            "current_price": current_price,
            "price_change_1m": None if ret_1m is None else ret_1m * 100.0,
            "price_change_3m": None if ret_3m is None else ret_3m * 100.0,
            "price_change_6m": None if ret_6m is None else ret_6m * 100.0,
            "drawdown_from_high_pct": None if drawdown is None else abs(drawdown) * 100.0,
            "drawdown_raw": drawdown,
            "avg_daily_value_cr": avg_daily_value_cr,
            "entry_anchor": float(closes.iloc[0]),
        }

    def _compute_price_metrics(self, symbol: str) -> dict[str, Any]:
        history = self._get_history(symbol, period="6mo")
        return self._compute_price_metrics_from_history(history)

    def _build_sector_overview(self) -> dict[str, dict[str, Any]]:
        if self._sector_overview is not None:
            return self._sector_overview

        members = self._combined_universe()
        tickers = [self._ticker_symbol(row["symbol"]) for row in members]
        overview: dict[str, dict[str, Any]] = {}
        if not tickers:
            self._sector_overview = overview
            return overview

        try:
            universe_prices = yf.download(
                tickers,
                period="6mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            universe_prices = pd.DataFrame()

        sector_buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
        for member in members:
            ticker = self._ticker_symbol(member["symbol"])
            frame = pd.DataFrame()
            if isinstance(universe_prices, pd.DataFrame) and not universe_prices.empty:
                if isinstance(universe_prices.columns, pd.MultiIndex) and ticker in universe_prices.columns.get_level_values(0):
                    frame = universe_prices[ticker].copy()
                elif not isinstance(universe_prices.columns, pd.MultiIndex):
                    frame = universe_prices.copy()
            if frame.empty:
                frame = self._get_history(member["symbol"], period="6mo")
            if frame.empty or "Close" not in frame:
                continue
            metrics = self._compute_price_metrics_from_history(frame)
            if not metrics:
                continue
            sector_buckets[member["sector"]].append(
                {
                    "ret_1m": (metrics.get("price_change_1m") or 0.0) / 100.0,
                    "ret_3m": (metrics.get("price_change_3m") or 0.0) / 100.0,
                    "ret_6m": (metrics.get("price_change_6m") or 0.0) / 100.0,
                    "drawdown": metrics.get("drawdown_raw") or 0.0,
                    "avg_daily_value_cr": metrics.get("avg_daily_value_cr") or 0.0,
                }
            )

        for sector, values in sector_buckets.items():
            one_month = median(row["ret_1m"] for row in values)
            three_month = median(row["ret_3m"] for row in values)
            six_month = median(row["ret_6m"] for row in values)
            drawdown = median(row["drawdown"] for row in values)
            breadth = sum(1 for row in values if row["ret_1m"] > 0) / max(len(values), 1)
            liquidity = median(row["avg_daily_value_cr"] for row in values)

            momentum_score = clamp((three_month + 0.15) / 0.30, 0.0, 1.0)
            reversal_score = clamp((one_month + 0.10) / 0.20, 0.0, 1.0)
            drawdown_score = clamp(1.0 + (drawdown / 0.25), 0.0, 1.0)
            liquidity_score = clamp(liquidity / 60.0, 0.0, 1.0)

            sector_signal = clamp(0.45 * momentum_score + 0.30 * breadth + 0.25 * liquidity_score, 0.0, 1.0)
            policy_signal = clamp(0.45 * reversal_score + 0.30 * momentum_score + 0.25 * drawdown_score, 0.0, 1.0)
            flow_signal = clamp(0.60 * breadth + 0.20 * momentum_score + 0.20 * liquidity_score, 0.0, 1.0)
            contrarian_signal = clamp(0.55 * (1.0 - drawdown_score) + 0.45 * reversal_score, 0.0, 1.0)

            if drawdown <= -0.18 and one_month > 0.03:
                contrarian_conviction = "BUY"
            elif one_month < -0.05 and three_month < -0.10:
                contrarian_conviction = "AVOID"
            else:
                contrarian_conviction = "CAUTION"

            overview[sector] = {
                "sector": sector,
                "member_count": len(values),
                "one_month_return": one_month,
                "three_month_return": three_month,
                "six_month_return": six_month,
                "drawdown": drawdown,
                "breadth": breadth,
                "avg_daily_value_cr": liquidity,
                "signal_score": round(sector_signal, 3),
                "policy_score": round(policy_signal, 3),
                "flow_score": round(flow_signal, 3),
                "contrarian_score": round(contrarian_signal, 3),
                "contrarian_conviction": contrarian_conviction,
            }

        self._sector_overview = overview
        return overview

    def normalize_symbol(self, symbol: str) -> str:
        return resolve_symbol_base(symbol)

    def infer_sector(self, instrument_name: str) -> str:
        name = (instrument_name or "").lower()
        rules = [
            ("liquid", "Fixed Income"),
            ("short term", "Fixed Income"),
            ("sdl", "Fixed Income"),
            ("bond", "Fixed Income"),
            ("nasdaq", "US Equity"),
            ("global", "Global Equity"),
            ("technology", "Technology"),
            ("mid cap", "Mid Cap"),
            ("midcap", "Mid Cap"),
            ("small cap", "Small Cap"),
            ("large & mid", "Large & Mid Cap"),
            ("large cap", "Large Cap"),
            ("flexi cap", "Flexi Cap"),
            ("value", "Value"),
            ("nifty next 50", "Next 50 Index"),
            ("nifty index", "Large Cap Index"),
            ("elss", "Tax Saver"),
            ("defence", "Aerospace & Defense"),
            ("bank", "Banks"),
            ("pharma", "Drug Manufacturers"),
        ]
        for needle, sector in rules:
            if needle in name:
                return sector
        return "Diversified Equity"

    def get_index_members(self, index_name: str) -> list[dict[str, Any]]:
        members = self._download_index_csv(index_name)
        if not members:
            members = self._load_stale_index_cache(index_name)
        if not members and index_name not in {"NIFTY50", "NIFTYNEXT50"}:
            members = self._load_stale_index_cache("NIFTY50") + self._load_stale_index_cache("NIFTYNEXT50")
        if not members and index_name not in {"NIFTY50", "NIFTYNEXT50"}:
            members = self._combined_universe()
        if not members:
            members = self._fallback_index_members(index_name)
        if not members:
            raise ValueError(
                f"Unable to load constituents for {index_name}. "
                "NSE archive download returned no rows and no fallback universe was available."
            )
        return [dict(row) for row in members]

    def get_stock_snapshot(self, symbol: str) -> dict[str, Any]:
        normalized = self.normalize_symbol(symbol)
        if normalized in self._snapshot_cache:
            return self._snapshot_cache[normalized]

        index_row = self._lookup_cached_index_row(normalized) or {}
        info = self._get_info(normalized)
        price_metrics = self._compute_price_metrics(normalized)

        company_name = (
            info.get("longName")
            or info.get("shortName")
            or index_row.get("company_name")
            or get_symbol_display_name(normalized)
        )
        raw_sector = (
            info.get("industry")
            or info.get("sector")
            or index_row.get("sector")
            or self.infer_sector(normalized)
            or "Unknown"
        )
        sector = get_sector(normalized, raw_sector)
        current_price = (
            self._as_float(info.get("regularMarketPrice"))
            or self._as_float(info.get("currentPrice"))
            or price_metrics.get("current_price")
        )

        snapshot = {
            "symbol": normalized,
            "company_name": company_name,
            "sector": sector,
            "price": current_price,
            "analyst_target": self._as_float(info.get("targetMeanPrice")),
            "us_revenue_pct": None,
            "beta": self._as_float(info.get("beta")),
            "avg_daily_value_cr": price_metrics.get("avg_daily_value_cr"),
            "promoter_pledge_pct": None,
            "sebi_flag": False,
            "drawdown_from_52w": price_metrics.get("drawdown_from_high_pct"),
            "pe_trailing": self._as_float(info.get("trailingPE")),
            "pe_5yr_avg": None,
            "sector_pe": None,
            "pe_forward": self._as_float(info.get("forwardPE")),
        }
        self._snapshot_cache[normalized] = snapshot
        return snapshot

    def build_proxy_holding(
        self,
        instrument_name: str,
        source: str,
        lookthrough_weight: float,
        holdings_source: str = "proxy:scheme-level",
    ) -> dict[str, Any]:
        token = hashlib.md5(f"{source}:{instrument_name}".encode("utf-8")).hexdigest()[:8].upper()
        symbol = f"{source[:2].upper()}_{token}"
        return {
            "instrument_name": instrument_name,
            "fund_weight": 100.0,
            "lookthrough_weight": round(lookthrough_weight, 3),
            "symbol": symbol,
            "company_name": instrument_name,
            "sector": self.infer_sector(instrument_name),
            "source": source,
            "proxy": True,
            "holdings_source": holdings_source,
        }

    def get_geopolitical_signals(self, macro_thesis: str = "") -> list[dict[str, Any]]:
        thesis = (macro_thesis or "").lower()
        rows: list[dict[str, Any]] = []
        for sector, payload in self._build_sector_overview().items():
            boost = 0.08 if sector.lower() in thesis else 0.0
            score = round(clamp(payload["signal_score"] + boost, 0.0, 1.0), 3)
            rows.append(
                {
                    "signal_key": f"MARKET_{sector.upper().replace(' ', '_')}",
                    "sector": sector,
                    "conviction": self._market_conviction(score),
                    "score": score,
                    "source": "NSE_ARCHIVES+YFINANCE",
                    "horizon": "1-3m",
                    "detail": (
                        f"{sector}: median 3M return {payload['three_month_return'] * 100:+.1f}% "
                        f"with {payload['breadth'] * 100:.0f}% breadth"
                    ),
                    "as_of_date": self.today.isoformat(),
                    "payload": {
                        "one_month_return_pct": round(payload["one_month_return"] * 100, 2),
                        "three_month_return_pct": round(payload["three_month_return"] * 100, 2),
                        "breadth_pct": round(payload["breadth"] * 100, 2),
                    },
                }
            )
        return rows

    def get_tariff_signals(self) -> list[dict[str, Any]]:
        return []

    def get_policy_signals(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sector, payload in self._build_sector_overview().items():
            score = payload["policy_score"]
            rows.append(
                {
                    "signal_key": f"POLICY_{sector.upper().replace(' ', '_')}",
                    "sector": sector,
                    "conviction": self._market_conviction(score),
                    "score": score,
                    "source": "YFINANCE_PRICE_ACTION",
                    "horizon": "1-6m",
                    "detail": (
                        f"{sector}: 1M {payload['one_month_return'] * 100:+.1f}% "
                        f"after {abs(payload['drawdown']) * 100:.1f}% drawdown from 6M high"
                    ),
                    "as_of_date": self.today.isoformat(),
                    "payload": {
                        "drawdown_pct": round(abs(payload["drawdown"]) * 100, 2),
                        "six_month_return_pct": round(payload["six_month_return"] * 100, 2),
                    },
                }
            )
        return rows

    def get_flow_signals(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sector, payload in self._build_sector_overview().items():
            score = payload["flow_score"]
            rows.append(
                {
                    "signal_key": f"FLOW_{sector.upper().replace(' ', '_')}",
                    "sector": sector,
                    "conviction": self._market_conviction(score),
                    "score": score,
                    "source": "NSE_BREADTH",
                    "horizon": "2-6w",
                    "detail": (
                        f"{sector}: {payload['breadth'] * 100:.0f}% of tracked constituents are positive over 1 month"
                    ),
                    "as_of_date": self.today.isoformat(),
                    "payload": {
                        "breadth_pct": round(payload["breadth"] * 100, 2),
                        "avg_daily_value_cr": round(payload["avg_daily_value_cr"], 2),
                    },
                }
            )
        return rows

    def get_contrarian_signals(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sector, payload in self._build_sector_overview().items():
            rows.append(
                {
                    "signal_key": f"CONTRA_{sector.upper().replace(' ', '_')}",
                    "sector": sector,
                    "conviction": payload["contrarian_conviction"],
                    "score": payload["contrarian_score"],
                    "source": "YFINANCE_REVERSAL",
                    "horizon": "entry",
                    "detail": (
                        f"{sector}: drawdown {payload['drawdown'] * 100:+.1f}% and 1M return "
                        f"{payload['one_month_return'] * 100:+.1f}%"
                    ),
                    "as_of_date": self.today.isoformat(),
                    "payload": {
                        "drawdown_pct": round(abs(payload["drawdown"]) * 100, 2),
                        "one_month_return_pct": round(payload["one_month_return"] * 100, 2),
                    },
                }
            )
        return rows

    def get_current_market_signal(self, sector: str) -> float:
        payload = self._build_sector_overview().get(sector)
        return payload["signal_score"] if payload else 0.5

    def get_stock_news(self, symbol: str) -> dict[str, Any]:
        snapshot = self.get_stock_snapshot(symbol)
        price = self.get_price_context(symbol)
        current_price = price.get("price")
        analyst_target = price.get("analyst_target")
        upside = None
        if current_price and analyst_target:
            upside = ((analyst_target - current_price) / current_price) * 100.0

        components: list[float] = []
        if price.get("price_change_1m") is not None:
            components.append(clamp(price["price_change_1m"] / 20.0, -1.0, 1.0))
        if price.get("price_change_6m") is not None:
            components.append(clamp(price["price_change_6m"] / 30.0, -1.0, 1.0))
        if upside is not None:
            components.append(clamp(upside / 25.0, -1.0, 1.0))
        sentiment_score = round(sum(components) / len(components), 3) if components else 0.0

        parts = []
        if price.get("price_change_1m") is not None:
            parts.append(f"1M {price['price_change_1m']:+.1f}%")
        if price.get("price_change_6m") is not None:
            parts.append(f"6M {price['price_change_6m']:+.1f}%")
        if upside is not None:
            parts.append(f"target upside {upside:+.1f}%")
        headline = f"{snapshot['company_name']}: " + (", ".join(parts) if parts else "live market data updated")

        return {
            "symbol": snapshot["symbol"],
            "sentiment_score": sentiment_score,
            "headline": headline,
        }

    def get_sector_news(self, sector: str) -> dict[str, Any]:
        payload = self._build_sector_overview().get(sector)
        if not payload:
            return {
                "sector": sector,
                "signal_score": 0.5,
                "summary": f"{sector}: live sector breadth is unavailable right now.",
            }
        return {
            "sector": sector,
            "signal_score": payload["signal_score"],
            "summary": (
                f"{sector}: 1M {payload['one_month_return'] * 100:+.1f}%, "
                f"3M {payload['three_month_return'] * 100:+.1f}%, "
                f"breadth {payload['breadth'] * 100:.0f}%."
            ),
        }

    def get_financials(self, symbol: str) -> dict[str, Any]:
        normalized = self.normalize_symbol(symbol)
        if normalized in self._financial_cache:
            return self._financial_cache[normalized]

        snapshot = self.get_stock_snapshot(normalized)
        screener = get_stock_fundamentals(normalized)
        result: dict[str, Any] = {
            "symbol": screener.get("symbol", normalized),
            "company_name": snapshot["company_name"],
            "sector": snapshot["sector"],
            "source": screener.get("source"),
            "resolved_symbol": screener.get("resolved_symbol"),
            "symbol_mapped": screener.get("symbol_mapped", False),
            "roce_pct": screener.get("roce_pct"),
            "roe_pct": screener.get("roe_pct"),
            "roce_ttm": None if screener.get("roce_pct") is None else screener["roce_pct"] / 100.0,
            "returnOnCapitalEmployed": None if screener.get("roce_pct") is None else screener["roce_pct"] / 100.0,
            "roce_5y": screener.get("roce_pct"),
            "freeCashflow": None,
            "free_cashflow": None,
            "fcf_positive_years": None,
            "revenueGrowth": None if screener.get("revenue_growth_pct") is None else screener["revenue_growth_pct"] / 100.0,
            "revenue_growth": None if screener.get("revenue_growth_pct") is None else screener["revenue_growth_pct"] / 100.0,
            "revenue_growth_pct": screener.get("revenue_growth_pct"),
            "debtToEquity": screener.get("debt_to_equity"),
            "debt_to_equity": screener.get("debt_to_equity"),
            "pe_trailing": screener.get("pe_ratio"),
            "pe_ratio": screener.get("pe_ratio"),
            "pe_5yr_avg": None,
            "sector_pe": None,
            "pe_forward": None,
            "profit_margins": None,
            "trailingEps": screener.get("eps"),
            "eps": screener.get("eps"),
            "currentPrice": screener.get("current_price"),
            "targetMeanPrice": screener.get("target_mean_price"),
            "week52_high": screener.get("week52_high"),
            "week52_low": screener.get("week52_low"),
            "fiftyTwoWeekHigh": screener.get("week52_high"),
            "fiftyTwoWeekLow": screener.get("week52_low"),
            "beta": snapshot.get("beta"),
            "promoter_holding_pct": None if screener.get("promoter_holding") is None else screener["promoter_holding"] / 100.0,
            "promoter_holding": screener.get("promoter_holding"),
            "negative_pat_quarters": None,
        }

        filtered = {key: value for key, value in result.items() if value is not None}
        self._financial_cache[normalized] = filtered
        return filtered

    def get_risk_metrics(self, symbol: str) -> dict[str, Any]:
        snapshot = self.get_stock_snapshot(symbol)
        return {
            "symbol": snapshot["symbol"],
            "avg_daily_value_cr": snapshot.get("avg_daily_value_cr"),
            "beta": snapshot.get("beta"),
            "promoter_pledge_pct": snapshot.get("promoter_pledge_pct"),
            "sebi_flag": snapshot.get("sebi_flag", False),
        }

    def get_price_context(self, symbol: str) -> dict[str, Any]:
        snapshot = self.get_stock_snapshot(symbol)
        metrics = self._compute_price_metrics(symbol)
        return {
            "symbol": snapshot["symbol"],
            "price": snapshot.get("price") or metrics.get("current_price"),
            "analyst_target": snapshot.get("analyst_target"),
            "drawdown_from_52w": metrics.get("drawdown_from_high_pct"),
            "price_change_1m": metrics.get("price_change_1m"),
            "price_change_6m": metrics.get("price_change_6m"),
        }

    def get_fund_holdings(self, fund_name: str, month: str | None = None) -> tuple[dict[str, float], str]:
        if not self.holdings_client:
            return {}, "unresolved"
        holdings, source = self.holdings_client.resolve_with_meta(fund_name, month=month)
        return holdings or {}, source

    def get_etf_holdings(self, etf_name: str, month: str | None = None) -> tuple[dict[str, float], str]:
        if not self.holdings_client:
            return {}, "unresolved"
        holdings, source = self.holdings_client.resolve_with_meta(etf_name, month=month)
        return holdings or {}, source

    def get_sector_target_weights(self) -> dict[str, float]:
        members = self._combined_universe()
        total = max(len(members), 1)
        sector_counts: dict[str, int] = defaultdict(int)
        for row in members:
            sector_counts[row["sector"]] += 1
        return {
            sector: round((count / total) * 100.0, 2)
            for sector, count in sector_counts.items()
        }

    def get_monitoring_price_series(self, symbol: str, entry_price: float | None = None) -> dict[str, Any]:
        metrics = self._compute_price_metrics(symbol)
        current = metrics.get("current_price")
        assumed_entry = entry_price or metrics.get("entry_anchor") or current
        drawdown_pct = None
        if current and assumed_entry:
            drawdown_pct = round(((current - assumed_entry) / assumed_entry) * 100.0, 2)
        return {
            "symbol": self.normalize_symbol(symbol),
            "entry_price": round(assumed_entry, 2) if assumed_entry else None,
            "current_price": current,
            "drawdown_pct": drawdown_pct or 0.0,
            "as_of_date": self.today.isoformat(),
        }

    def recent_date(self, days_back: int) -> str:
        return (self.today - timedelta(days=days_back)).isoformat()
