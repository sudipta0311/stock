"""
snapshot.py — fetch and persist historical price + fundamental snapshots.

Two modes:
  snapshot prices      — bulk yfinance download, stored in historical_prices
  snapshot fundamentals — yfinance quarterly income + balance sheet, stored in
                          historical_fundamentals

Run via run_backtest.py --mode snapshot, or import directly.

Coverage helpers:
  get_coverage_counts(repo) — aggregate null% per field
  report_coverage(repo, before=None) — print before/after comparison table
"""
from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import yfinance as yf

from stock_platform.config import AppConfig, load_app_env
from stock_platform.data.repository import PlatformRepository

_log = logging.getLogger(__name__)

NIFTY_TICKER = "^NSEI"

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
    return symbol.replace("&", "") + ".NS"


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def _get_val(frame: Any, col: Any, *labels: str) -> float | None:
    """Read a float from (label, col) in a pandas DataFrame, trying label alternatives."""
    if frame is None or getattr(frame, "empty", True):
        return None
    if col not in frame.columns:
        return None
    for label in labels:
        if label not in frame.index:
            continue
        try:
            v = float(frame.loc[label, col])
            return None if pd.isna(v) else v
        except (TypeError, ValueError):
            pass
    return None


def _nearest_col(target_col: Any, cols: list) -> Any | None:
    """Return the element in cols whose timestamp is closest to target_col."""
    if not cols:
        return None
    t = pd.Timestamp(target_col)
    return min(cols, key=lambda c: abs(pd.Timestamp(c) - t))


# ── Price snapshot ────────────────────────────────────────────────────────────

def snapshot_prices(
    repo: PlatformRepository,
    symbols: list[str] | None = None,
    period: str = "2y",
    interval: str = "1wk",
) -> int:
    """
    Bulk-download weekly price history via yfinance and upsert into historical_prices.
    Includes the NIFTY index (^NSEI) for benchmark forward-return calculation.
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

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]

    rows_written = 0
    with repo.connect() as conn:
        for ticker in closes.columns:
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


# ── Fundamental snapshot ──────────────────────────────────────────────────────

def snapshot_fundamentals(
    repo: PlatformRepository,
    symbols: list[str] | None = None,
) -> int:
    """
    Fetch yfinance quarterly income statement + balance sheet for each symbol
    and store the most recent 8 quarters in historical_fundamentals.

    Fields computed per quarter:
      eps            — Basic/Diluted EPS from income statement
      revenue_growth — YoY % growth: (rev_q - rev_q4) / |rev_q4| × 100
      debt_equity    — Total Debt / Equity (balance sheet, nearest quarter)
      roce           — Operating Income × 4 / Capital Employed × 100
                       Capital Employed = Total Assets - Current Liabilities
      promoter_holding — always NULL (Screener enrichment fills this)

    fetched_source is set to "yfinance" for all rows written here.
    """
    universe     = symbols or _NIFTY200_SAMPLE
    rows_written = 0

    with repo.connect() as conn:
        for symbol in universe:
            try:
                ticker = yf.Ticker(_nse_ticker(symbol))
                qis    = ticker.quarterly_income_stmt
                qbs    = ticker.quarterly_balance_sheet

                if qis is None or getattr(qis, "empty", True):
                    _log.debug("%s: no quarterly income statement", symbol)
                    continue

                qis_cols = list(qis.columns)
                qbs_cols = list(qbs.columns) if (
                    qbs is not None and not getattr(qbs, "empty", True)
                ) else []

                n_sym = 0
                for i, col in enumerate(qis_cols[:8]):
                    snapshot_date = pd.Timestamp(col).strftime("%Y-%m-%d")

                    # Nearest balance sheet quarter (may differ from income stmt quarter)
                    bs_col = _nearest_col(col, qbs_cols)

                    # ── EPS ──────────────────────────────────────────────────
                    eps = _get_val(qis, col, "Basic EPS", "Diluted EPS", "EPS")

                    # ── Revenue and per-quarter YoY growth ───────────────────
                    revenue   = _get_val(qis, col,
                                         "Total Revenue", "Revenue", "Net Revenue",
                                         "Operating Revenue")
                    rev_growth = None
                    if revenue is not None and i + 4 < len(qis_cols):
                        col_4    = qis_cols[i + 4]
                        prev_rev = _get_val(qis, col_4,
                                            "Total Revenue", "Revenue", "Net Revenue",
                                            "Operating Revenue")
                        if prev_rev and abs(prev_rev) > 0:
                            rev_growth = round(
                                (revenue - prev_rev) / abs(prev_rev) * 100, 2
                            )

                    # ── D/E (aligned to nearest balance sheet quarter) ───────
                    total_debt = _get_val(qbs, bs_col,
                                          "Total Debt", "Long Term Debt") if bs_col else None
                    equity     = _get_val(qbs, bs_col,
                                          "Stockholders Equity",
                                          "Total Stockholders Equity") if bs_col else None
                    debt_equity = (
                        round(total_debt / equity, 3)
                        if (total_debt and equity and equity != 0)
                        else None
                    )

                    # ── ROCE (annualised quarterly operating income / cap employed) ──
                    op_income  = _get_val(qis, col,
                                          "Operating Income", "EBIT",
                                          "Total Operating Income As Reported")
                    total_assets = _get_val(qbs, bs_col,
                                            "Total Assets") if bs_col else None
                    cur_liab     = _get_val(qbs, bs_col,
                                            "Current Liabilities",
                                            "Total Current Liabilities",
                                            "Current Liabilities Net Minority Interest",
                                            ) if bs_col else None
                    roce = None
                    if (
                        op_income is not None
                        and total_assets is not None
                        and total_assets > 0
                    ):
                        cap_employed = total_assets - (cur_liab or 0)
                        if cap_employed > 0:
                            roce = round(op_income * 4 / cap_employed * 100, 2)

                    conn.execute(
                        """
                        INSERT INTO historical_fundamentals
                            (symbol, snapshot_date, roce, eps, debt_equity,
                             revenue_growth, promoter_holding, source, fetched_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, snapshot_date) DO UPDATE SET
                            roce=excluded.roce,
                            eps=excluded.eps,
                            debt_equity=excluded.debt_equity,
                            revenue_growth=excluded.revenue_growth,
                            promoter_holding=excluded.promoter_holding,
                            source=excluded.source,
                            fetched_source=excluded.fetched_source
                        """,
                        (symbol, snapshot_date, roce, eps, debt_equity,
                         rev_growth, None, "yfinance", "yfinance"),
                    )
                    n_sym += 1

                conn.commit()
                rows_written += n_sym
                _log.info("%s: %d quarters written (ROCE computed: %s)",
                          symbol, n_sym,
                          sum(1 for k in range(min(8, len(qis_cols)))
                              if _get_val(qis, qis_cols[k], "Operating Income", "EBIT") is not None))

            except Exception as exc:
                try:
                    conn.rollback()
                except Exception:
                    pass
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


# ── Coverage reporting ────────────────────────────────────────────────────────

def get_coverage_counts(repo: PlatformRepository) -> dict:
    """
    Return aggregate null counts from historical_fundamentals.
    Returns dict with keys: symbol_count, row_count, null (per-field null counts).
    Returns empty dict on error or when table is empty.
    """
    try:
        with repo.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT symbol)                               AS symbol_count,
                    COUNT(*)                                             AS row_count,
                    SUM(CASE WHEN roce             IS NULL THEN 1 ELSE 0 END) AS null_roce,
                    SUM(CASE WHEN eps              IS NULL THEN 1 ELSE 0 END) AS null_eps,
                    SUM(CASE WHEN debt_equity      IS NULL THEN 1 ELSE 0 END) AS null_de,
                    SUM(CASE WHEN revenue_growth   IS NULL THEN 1 ELSE 0 END) AS null_rev,
                    SUM(CASE WHEN promoter_holding IS NULL THEN 1 ELSE 0 END) AS null_promo
                FROM historical_fundamentals
                """
            ).fetchone()
        if row is None or (row["row_count"] or 0) == 0:
            return {"symbol_count": 0, "row_count": 0, "null": {}}
        total = row["row_count"]
        return {
            "symbol_count": row["symbol_count"],
            "row_count":    total,
            "null": {
                "roce":             row["null_roce"]  or 0,
                "eps":              row["null_eps"]   or 0,
                "debt_equity":      row["null_de"]    or 0,
                "revenue_growth":   row["null_rev"]   or 0,
                "promoter_holding": row["null_promo"] or 0,
            },
        }
    except Exception as exc:
        _log.warning("get_coverage_counts: %r", exc)
        return {}


def report_coverage(repo: PlatformRepository, before: dict | None = None) -> None:
    """
    Query historical_fundamentals and print a coverage table.
    If before is provided (output of get_coverage_counts before snapshot),
    prints a before/after comparison including per-field delta.
    Also prints the 15 worst-covered symbols.
    """
    after  = get_coverage_counts(repo)
    n_after  = after.get("row_count", 0)
    symbols  = after.get("symbol_count", 0)
    n_before = (before or {}).get("row_count", 0)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Fundamental Coverage Report")
    print(f"  {symbols} symbols  |  {n_after} rows")
    print(sep)

    fields = [
        ("roce",             "ROCE"),
        ("eps",              "EPS"),
        ("debt_equity",      "D/E"),
        ("revenue_growth",   "RevGrowth"),
        ("promoter_holding", "Promoter"),
    ]

    if before and n_before > 0:
        print(f"\n  {'Field':<18}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
        print("  " + "-" * 54)
        for key, label in fields:
            b_null = before.get("null", {}).get(key, 0)
            a_null = after.get("null", {}).get(key, 0)
            b_pct  = b_null / n_before * 100 if n_before else 0.0
            a_pct  = a_null / n_after  * 100 if n_after  else 0.0
            delta  = a_pct - b_pct
            print(
                f"  {label:<18}  {b_pct:>9.1f}%  {a_pct:>9.1f}%  {delta:>+9.1f}pp"
            )
    else:
        print(f"\n  {'Field':<18}  {'Null rows':>10}  {'Null%':>10}")
        print("  " + "-" * 42)
        for key, label in fields:
            a_null = after.get("null", {}).get(key, 0)
            a_pct  = a_null / n_after * 100 if n_after else 0.0
            print(f"  {label:<18}  {a_null:>10}  {a_pct:>9.1f}%")

    # Per-symbol detail: 15 worst covered
    try:
        with repo.connect() as conn:
            detail = conn.execute(
                """
                SELECT symbol,
                       COUNT(*) AS quarters,
                       SUM(CASE WHEN roce             IS NULL THEN 1 ELSE 0 END) AS null_roce,
                       SUM(CASE WHEN eps              IS NULL THEN 1 ELSE 0 END) AS null_eps,
                       SUM(CASE WHEN debt_equity      IS NULL THEN 1 ELSE 0 END) AS null_de,
                       SUM(CASE WHEN revenue_growth   IS NULL THEN 1 ELSE 0 END) AS null_rev,
                       SUM(CASE WHEN promoter_holding IS NULL THEN 1 ELSE 0 END) AS null_promo
                FROM historical_fundamentals
                GROUP BY symbol
                ORDER BY (
                    SUM(CASE WHEN roce             IS NULL THEN 1 ELSE 0 END) +
                    SUM(CASE WHEN eps              IS NULL THEN 1 ELSE 0 END) +
                    SUM(CASE WHEN debt_equity      IS NULL THEN 1 ELSE 0 END) +
                    SUM(CASE WHEN revenue_growth   IS NULL THEN 1 ELSE 0 END) +
                    SUM(CASE WHEN promoter_holding IS NULL THEN 1 ELSE 0 END)
                ) DESC, symbol
                LIMIT 15
                """
            ).fetchall()
        print(f"\n  Bottom 15 symbols by total null count:")
        print(f"  {'Symbol':<14}  {'Qtrs':>5}  {'ROCE':>5}  {'EPS':>4}  {'D/E':>4}  {'RevG':>4}  {'Promo':>5}")
        print("  " + "-" * 50)
        for row in detail:
            print(
                f"  {row['symbol']:<14}  {row['quarters']:>5}  "
                f"{row['null_roce']:>5}  {row['null_eps']:>4}  {row['null_de']:>4}  "
                f"{row['null_rev']:>4}  {row['null_promo']:>5}"
            )
    except Exception as exc:
        _log.warning("report_coverage detail query: %r", exc)

    print(f"{sep}\n")
