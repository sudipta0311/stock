"""
snapshot.py — fetch and persist historical price + fundamental snapshots.

Two modes:
  snapshot prices      — bulk yfinance download, stored in historical_prices
  snapshot fundamentals — yfinance quarterly income statement, stored in historical_fundamentals

Run via run_backtest.py --mode snapshot, or import directly.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running as a script from repo root.
_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import yfinance as yf

from stock_platform.config import AppConfig, load_app_env
from stock_platform.data.repository import PlatformRepository

_log = logging.getLogger(__name__)

# NIFTY index ticker for benchmark forward-return calculation.
NIFTY_TICKER = "^NSEI"

# The NIFTY200 symbols used for universe — snapshot stores whichever are returned
# by the provider; for standalone backtest we use a static list of large-caps.
_NIFTY200_SAMPLE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "KOTAKBANK", "LT",
    "SBIN", "AXISBANK", "BAJFINANCE", "HINDUNILVR", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "POWERGRID", "NTPC",
    "TECHM", "HCLTECH", "ITC", "DIVISLAB", "DRREDDY", "CIPLA",
    "BAJAJFINSV", "TATAMOTORS", "JSWSTEEL", "HINDALCO", "COALINDIA",
    "ADANIPORTS", "GRASIM", "BPCL", "EICHERMOT", "APOLLOHOSP",
    "BHARTIARTL", "INDUSINDBK", "TATACONSUM", "SHREECEM", "NESTLEIND",
    "ONGC", "IOC", "M&M", "HDFCLIFE", "SBILIFE", "BRITANNIA",
    "BAJAJ-AUTO", "PIDILITIND", "SIEMENS", "ABB", "BEL", "HAL",
]


def _nse_ticker(symbol: str) -> str:
    """Convert NSE symbol to yfinance ticker (appends .NS, handles special chars)."""
    return symbol.replace("&", "") + ".NS"


def snapshot_prices(
    repo: PlatformRepository,
    symbols: list[str] | None = None,
    period: str = "2y",
    interval: str = "1wk",
) -> int:
    """
    Bulk-download weekly price history via yfinance and upsert into historical_prices.

    Includes the NIFTY index (^NSEI) for benchmark comparison.
    Returns the number of rows written.
    """
    universe = symbols or _NIFTY200_SAMPLE
    tickers  = [_nse_ticker(s) for s in universe] + [NIFTY_TICKER]

    _log.info("Downloading %d tickers (%s, %s) …", len(tickers), period, interval)
    raw: pd.DataFrame = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        _log.warning("yfinance returned no data")
        return 0

    # yfinance multi-ticker download returns MultiIndex columns (metric, ticker).
    # We only need Close.
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]

    rows_written = 0
    with repo.connect() as conn:
        for ticker in closes.columns:
            # Map ticker back to NSE symbol.
            if ticker == NIFTY_TICKER:
                symbol = "NIFTY"
            else:
                symbol = ticker.replace(".NS", "")

            series = closes[ticker].dropna()
            for dt, price in series.items():
                date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
                conn.execute(
                    """
                    INSERT INTO historical_prices(symbol, date, close_price, volume)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(symbol, date) DO UPDATE SET
                        close_price=excluded.close_price,
                        volume=excluded.volume
                    """,
                    (symbol, date_str, float(price), None),
                )
                rows_written += 1
        conn.commit()

    _log.info("snapshot_prices: wrote %d rows", rows_written)
    return rows_written


def snapshot_fundamentals(
    repo: PlatformRepository,
    symbols: list[str] | None = None,
) -> int:
    """
    Fetch yfinance quarterly income statement for each symbol and store the most
    recent 8 quarters in historical_fundamentals.

    Screener.in doesn't expose historical data via its public interface, so we
    rely on yfinance for v1 — fields will be marked with source='yfinance'.
    Missing fields are stored as NULL.
    """
    universe     = symbols or _NIFTY200_SAMPLE
    rows_written = 0

    for symbol in universe:
        try:
            ticker = yf.Ticker(_nse_ticker(symbol))
            qis    = ticker.quarterly_income_stmt   # columns = quarter-end dates

            if qis is None or getattr(qis, "empty", True):
                _log.debug("%s: no quarterly income statement", symbol)
                continue

            # yfinance quarterly_financials has a quarterly_balance_sheet for D/E.
            qbs = ticker.quarterly_balance_sheet

            info = ticker.fast_info

            for col in list(qis.columns)[:8]:   # up to 8 most recent quarters
                snapshot_date = pd.Timestamp(col).strftime("%Y-%m-%d")

                def _val(frame: Any, *labels: str) -> float | None:
                    if frame is None or getattr(frame, "empty", True):
                        return None
                    for label in labels:
                        if label in frame.index:
                            raw = frame.loc[label, col] if col in frame.columns else None
                            try:
                                v = float(raw)
                                return None if pd.isna(v) else v
                            except (TypeError, ValueError):
                                pass
                    return None

                revenue      = _val(qis, "Total Revenue", "Revenue")
                prev_col_idx = list(qis.columns).index(col) + 4  # same Q prior year
                prev_col     = list(qis.columns)[prev_col_idx] if prev_col_idx < len(qis.columns) else None
                rev_growth   = None
                if revenue and prev_col is not None:
                    prev_rev = _val(qis, "Total Revenue", "Revenue") if False else None
                    # Simple: use yfinance revenueGrowth from info as proxy.
                    try:
                        rev_growth = float(info.revenue_growth or 0) * 100
                    except Exception:
                        rev_growth = None

                eps           = _val(qis, "Basic EPS", "Diluted EPS", "EPS")
                total_debt    = _val(qbs, "Total Debt", "Long Term Debt") if qbs is not None else None
                equity        = _val(qbs, "Stockholders Equity", "Total Stockholders Equity") if qbs is not None else None
                debt_equity   = round(total_debt / equity, 3) if (total_debt and equity and equity != 0) else None
                promoter_hold = None   # not available via yfinance

                with repo.connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO historical_fundamentals
                            (symbol, snapshot_date, roce, eps, debt_equity,
                             revenue_growth, promoter_holding, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, snapshot_date) DO UPDATE SET
                            roce=excluded.roce, eps=excluded.eps,
                            debt_equity=excluded.debt_equity,
                            revenue_growth=excluded.revenue_growth,
                            promoter_holding=excluded.promoter_holding,
                            source=excluded.source
                        """,
                        (symbol, snapshot_date, None, eps, debt_equity,
                         rev_growth, promoter_hold, "yfinance"),
                    )
                    conn.commit()
                    rows_written += 1

        except Exception as exc:
            _log.warning("snapshot_fundamentals %s: %r", symbol, exc)

    _log.info("snapshot_fundamentals: wrote %d rows", rows_written)
    return rows_written


def run_snapshot(
    repo: PlatformRepository,
    symbols: list[str] | None = None,
    period: str = "2y",
) -> dict[str, int]:
    """Run both price and fundamental snapshots; return row counts."""
    price_rows = snapshot_prices(repo, symbols=symbols, period=period)
    fund_rows  = snapshot_fundamentals(repo, symbols=symbols)
    return {"price_rows": price_rows, "fundamental_rows": fund_rows}
