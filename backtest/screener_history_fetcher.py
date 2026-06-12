"""
screener_history_fetcher.py — Screener.in historical enrichment for backtest fundamentals.

Fetches per-quarter promoter holding and ROCE estimates from Screener.in HTML
and fills NULL fields in historical_fundamentals (first-non-null-wins policy).

fetched_source values written:
  "screener"          — directly reported value (promoter holding from quarterly-shp;
                        or current ROCE from #top-ratios, applied to the latest quarter)
  "screener_computed" — ROCE estimated from quarterly P&L:
                        (Operating Profit × 4) / Total Assets × 100
                        This approximates annualised ROCE using most-recent balance-sheet
                        total assets as the denominator (operating margin × asset turnover).

Rate limit: delay=5.0 s between symbol fetches (non-negotiable — live platform
depends on Screener access and an IP ban cannot be risked).
"""
from __future__ import annotations

import calendar
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_platform.data.repository import PlatformRepository

_log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-app/1.0)"}

_SCREENER_MONTHS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _clean_number(text: str) -> float | None:
    cleaned = (
        (text or "")
        .strip()
        .replace(",", "")
        .replace("%", "")
        .replace("Cr.", "")
        .replace("₹", "")
        .replace("Rs.", "")
    )
    if not cleaned or cleaned == "-":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _quarter_label_to_date(label: str) -> str | None:
    """Convert Screener quarter label to ISO date string.

    "Dec 2024" → "2024-12-31"
    "Sep 2024" → "2024-09-30"
    """
    parts = (label or "").strip().split()
    if len(parts) < 2:
        return None
    month = _SCREENER_MONTHS.get(parts[0][:3].lower())
    if month is None:
        return None
    try:
        year = int(parts[1])
    except ValueError:
        return None
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last_day:02d}"


def _fetch_soup(symbol: str, consolidated: bool = True) -> BeautifulSoup | None:
    clean  = symbol.replace("&", "")
    suffix = "consolidated/" if consolidated else ""
    url    = f"https://www.screener.in/company/{clean}/{suffix}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            _log.debug("_fetch_soup %s: HTTP %d", symbol, resp.status_code)
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        _log.debug("_fetch_soup %s: %r", symbol, exc)
        return None


def _get_soup(symbol: str) -> BeautifulSoup | None:
    """Try consolidated page first, fall back to standalone."""
    soup = _fetch_soup(symbol, consolidated=True)
    if soup is None:
        soup = _fetch_soup(symbol, consolidated=False)
    return soup


# ── Data parsers ──────────────────────────────────────────────────────────────

def _parse_shareholding_history(soup: BeautifulSoup) -> dict[str, float]:
    """
    Parse #quarterly-shp (or #yearly-shp) table.
    Returns {snapshot_date_str: promoter_holding_pct} for up to 8 quarters.
    Quarterly takes precedence over yearly when both exist.
    """
    result: dict[str, float] = {}
    for table in soup.select("#quarterly-shp table, #yearly-shp table"):
        header_row = table.select_one("thead tr") or table.select_one("tr")
        if not header_row:
            continue
        header_cells = header_row.select("th, td")
        date_strs    = [
            _quarter_label_to_date(c.get_text(" ", strip=True))
            for c in header_cells[1:]
        ]

        data_rows = table.select("tbody tr") or table.select("tr")[1:]
        for row in data_rows:
            cells = row.select("td")
            if not cells:
                continue
            label = cells[0].get_text(" ", strip=True).lower()
            if "promoter" not in label:
                continue
            for i, cell in enumerate(cells[1:]):
                if i >= len(date_strs) or date_strs[i] is None:
                    continue
                val = _clean_number(cell.get_text(" ", strip=True))
                if val is not None and date_strs[i] not in result:
                    result[date_strs[i]] = val

        if result:
            return result  # quarterly found — stop

    return result


def _parse_quarterly_pl_history(
    soup: BeautifulSoup,
) -> dict[str, tuple[float | None, float | None]]:
    """
    Parse #quarters table for quarterly Operating Profit and Sales.
    Returns {snapshot_date_str: (operating_profit_cr, sales_cr)}.
    Values are in Crores as reported by Screener.
    """
    result: dict[str, tuple[float | None, float | None]] = {}
    rows = soup.select("#quarters table tr")
    if len(rows) < 2:
        return result

    header_cells = rows[0].select("th, td")
    date_strs    = [
        _quarter_label_to_date(c.get_text(" ", strip=True))
        for c in header_cells[1:]
    ]
    if not any(date_strs):
        return result

    sales_by_date:     dict[str, float] = {}
    op_profit_by_date: dict[str, float] = {}

    for row in rows[1:]:
        cells = row.select("th, td")
        if not cells:
            continue
        raw_label = (
            cells[0].get_text(" ", strip=True)
            .lower()
            .replace("+", "")
            .strip()
        )
        is_sales     = any(kw in raw_label for kw in ("sales", "revenue"))
        is_op_profit = "operating profit" in raw_label

        if not (is_sales or is_op_profit):
            continue

        for i, cell in enumerate(cells[1:]):
            if i >= len(date_strs) or date_strs[i] is None:
                continue
            val = _clean_number(cell.get_text(" ", strip=True))
            if val is None:
                continue
            d = date_strs[i]
            if is_sales and d not in sales_by_date:
                sales_by_date[d] = val
            elif is_op_profit and d not in op_profit_by_date:
                op_profit_by_date[d] = val

    all_dates = set(sales_by_date) | set(op_profit_by_date)
    for d in all_dates:
        result[d] = (op_profit_by_date.get(d), sales_by_date.get(d))
    return result


def _parse_total_assets_latest(soup: BeautifulSoup) -> float | None:
    """
    Get most recent Total Assets from the Balance Sheet section.
    Screener's balance sheet is annual; used as denominator for ROCE estimates
    across all historical quarters (accepted approximation).
    """
    for section in soup.select("section"):
        heading = section.select_one("h2")
        if not heading or "Balance Sheet" not in heading.get_text(" ", strip=True):
            continue
        table = section.select_one("table")
        if not table:
            continue
        for row in table.select("tbody tr"):
            cells = row.select("td")
            if not cells:
                continue
            label = cells[0].get_text(" ", strip=True)
            if "Total Assets" in label or "Total Liabilities" in label:
                for cell in reversed(cells[1:]):
                    val = _clean_number(cell.get_text(" ", strip=True))
                    if val is not None:
                        return val
    return None


def _parse_top_ratios_roce(soup: BeautifulSoup) -> float | None:
    """Get current ROCE % from #top-ratios li."""
    for item in soup.select("#top-ratios li"):
        name  = item.select_one(".name")
        value = item.select_one(".value .number, .number")
        if not name or not value:
            continue
        if "roce" in name.get_text(" ", strip=True).lower():
            return _clean_number(value.get_text(" ", strip=True))
    return None


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_screener_history(symbol: str) -> dict[str, dict[str, Any]]:
    """
    Fetch historical promoter holding and ROCE estimates from Screener.in.

    Returns {snapshot_date: {
        "promoter_holding":   float | None,
        "roce":               float | None,
        "fetched_source_roce": "screener" | "screener_computed" | None,
    }}

    ROCE strategy:
    - Quarterly P&L available → ROCE_estimated = (OP × 4) / Total_Assets × 100
      tagged as "screener_computed"
    - Only current-period top-ratios ROCE (no P&L) → apply to most recent date only,
      tagged as "screener"
    """
    soup = _get_soup(symbol)
    if soup is None:
        _log.warning("fetch_screener_history %s: page unavailable", symbol)
        return {}

    shp_history  = _parse_shareholding_history(soup)
    pl_history   = _parse_quarterly_pl_history(soup)
    current_roce = _parse_top_ratios_roce(soup)
    total_assets = _parse_total_assets_latest(soup)

    result: dict[str, dict[str, Any]] = {}

    all_dates = set(shp_history) | set(pl_history)
    for date_str in all_dates:
        entry: dict[str, Any] = {
            "promoter_holding":    shp_history.get(date_str),
            "roce":                None,
            "fetched_source_roce": None,
        }

        # ROCE from quarterly P&L (screener_computed)
        if date_str in pl_history and total_assets is not None and total_assets > 0:
            op_profit, _ = pl_history[date_str]
            if op_profit is not None:
                entry["roce"]                = round(op_profit * 4 / total_assets * 100, 2)
                entry["fetched_source_roce"] = "screener_computed"

        result[date_str] = entry

    # Current ROCE from top-ratios: apply to the most recent date where ROCE still NULL
    if current_roce is not None:
        candidate_dates = sorted(
            (d for d in result if result[d]["roce"] is None),
            reverse=True,
        )
        if not candidate_dates and shp_history:
            # No P&L dates at all — synthesise a row for the most recent shp date
            most_recent = max(shp_history.keys())
            candidate_dates = [most_recent]

        if candidate_dates:
            most_recent = candidate_dates[0]
            result.setdefault(most_recent, {
                "promoter_holding":    shp_history.get(most_recent),
                "roce":                None,
                "fetched_source_roce": None,
            })
            if result[most_recent]["roce"] is None:
                result[most_recent]["roce"]                = current_roce
                result[most_recent]["fetched_source_roce"] = "screener"

    return result


# ── Extended quarterly history (12 quarters of revenue, PAT, EPS) ────────────

def _parse_quarterly_full(
    soup: BeautifulSoup,
) -> dict[str, dict[str, float | None]]:
    """
    Parse the Screener #quarters table for Revenue, PAT (net profit), and EPS.
    Returns {snapshot_date: {revenue, pat, eps}} for up to 12 quarters.
    Values are in Crores for revenue/PAT; EPS is rupees per share.
    """
    result: dict[str, dict[str, float | None]] = {}
    rows = soup.select("#quarters table tr")
    if len(rows) < 2:
        return result

    header_cells = rows[0].select("th, td")
    date_strs = [
        _quarter_label_to_date(c.get_text(" ", strip=True))
        for c in header_cells[1:]
    ]
    if not any(date_strs):
        return result

    # Pre-populate result dict for all valid dates
    for d in date_strs:
        if d:
            result[d] = {"revenue": None, "pat": None, "eps": None}

    for row in rows[1:]:
        cells = row.select("th, td")
        if not cells:
            continue
        raw_label = cells[0].get_text(" ", strip=True).lower().replace("+", "").strip()

        is_sales  = any(kw in raw_label for kw in ("sales", "revenue"))
        is_pat    = any(kw in raw_label for kw in ("net profit", "profit after tax", "pat"))
        is_eps    = "eps" in raw_label

        if not (is_sales or is_pat or is_eps):
            continue

        for i, cell in enumerate(cells[1:]):
            if i >= len(date_strs) or not date_strs[i]:
                continue
            val = _clean_number(cell.get_text(" ", strip=True))
            d   = date_strs[i]
            if val is None or d not in result:
                continue
            if is_sales and result[d]["revenue"] is None:
                result[d]["revenue"] = val
            elif is_pat and result[d]["pat"] is None:
                result[d]["pat"] = val
            elif is_eps and result[d]["eps"] is None:
                result[d]["eps"] = val

    return result


def _parse_annual_de(soup: BeautifulSoup) -> float | None:
    """
    Get the most recent Debt/Equity from the balance-sheet section (annual).
    Screener shows long-term borrowings and total equity; we approximate D/E.
    Returns None when unavailable.
    """
    for section in soup.select("section"):
        heading = section.select_one("h2")
        if not heading or "Balance Sheet" not in heading.get_text():
            continue
        table = section.select_one("table")
        if not table:
            continue
        rows = table.select("tr")
        borrowings_row = equity_row = None
        for row in rows:
            cells = row.select("th, td")
            if not cells:
                continue
            label = cells[0].get_text(" ", strip=True).lower()
            if any(kw in label for kw in ("borrowing", "long term debt", "debt")):
                borrowings_row = cells
            elif any(kw in label for kw in ("equity", "net worth", "shareholders")):
                equity_row = cells
        if borrowings_row and equity_row:
            # Use most recent column (last data cell)
            for b_cell, e_cell in zip(reversed(borrowings_row[1:]), reversed(equity_row[1:])):
                b = _clean_number(b_cell.get_text(" ", strip=True))
                e = _clean_number(e_cell.get_text(" ", strip=True))
                if b is not None and e is not None and e != 0:
                    return round(b / e, 3)
    return None


def fetch_quarterly_history(
    symbol: str,
) -> dict[str, dict[str, Any]]:
    """
    Fetch up to 12 quarters of fundamental data from Screener.in for a symbol.

    Returns {snapshot_date: {revenue_cr, pat_cr, eps, debt_equity}} where
    snapshot_date is the quarter-end date string "YYYY-MM-DD".

    Revenue and PAT are in Crores (raw from Screener).
    EPS is rupees per share.
    debt_equity is annual (forward-filled to all quarters).
    """
    soup = _get_soup(symbol)
    if soup is None:
        _log.warning("fetch_quarterly_history %s: page unavailable", symbol)
        return {}

    quarterly = _parse_quarterly_full(soup)
    de        = _parse_annual_de(soup)

    result: dict[str, dict[str, Any]] = {}
    for date_str, vals in quarterly.items():
        result[date_str] = {
            "revenue_cr":  vals.get("revenue"),
            "pat_cr":      vals.get("pat"),
            "eps":         vals.get("eps"),
            "debt_equity": de,     # annual, forward-filled to all quarters
        }

    return result


def store_quarterly_history(
    repo: PlatformRepository,
    symbol: str,
    history: dict[str, dict[str, Any]],
) -> int:
    """
    Upsert quarterly history rows into historical_fundamentals.

    Computes revenue_growth_pct as YoY if the same quarter 4 periods back exists.
    Sets available_date = snapshot_date + 60d (conservative Indian disclosure lag).
    Uses first-non-null-wins for all fields (never overwrites yfinance values).

    Returns the number of rows written.
    """
    from datetime import date, timedelta

    if not history:
        return 0

    sorted_dates = sorted(history.keys())
    rows_written = 0

    with repo.connect() as conn:
        for i, snap_date_str in enumerate(sorted_dates):
            data           = history[snap_date_str]
            snap_date      = date.fromisoformat(snap_date_str)
            avail_date_str = (snap_date + timedelta(days=60)).isoformat()

            eps        = data.get("eps")
            de         = data.get("debt_equity")
            revenue    = data.get("revenue_cr")
            rev_growth = None

            # YoY revenue growth: compare to same quarter 4 periods back (≈ 1 year)
            if revenue is not None and i >= 4:
                prev_date_str = sorted_dates[i - 4]
                prev_rev = history[prev_date_str].get("revenue_cr")
                if prev_rev and abs(prev_rev) > 0:
                    rev_growth = round((revenue - prev_rev) / abs(prev_rev) * 100, 2)

            conn.execute(
                """
                INSERT INTO historical_fundamentals
                    (symbol, snapshot_date, available_date, eps, debt_equity,
                     revenue_growth, source, fetched_source)
                VALUES (?, ?, ?, ?, ?, ?, 'screener', 'screener')
                ON CONFLICT(symbol, snapshot_date) DO UPDATE SET
                    available_date  = COALESCE(historical_fundamentals.available_date, excluded.available_date),
                    eps             = COALESCE(historical_fundamentals.eps,            excluded.eps),
                    debt_equity     = COALESCE(historical_fundamentals.debt_equity,    excluded.debt_equity),
                    revenue_growth  = COALESCE(historical_fundamentals.revenue_growth, excluded.revenue_growth),
                    fetched_source  = CASE
                        WHEN historical_fundamentals.fetched_source IS NULL THEN 'screener'
                        ELSE historical_fundamentals.fetched_source
                    END
                """,
                (symbol, snap_date_str, avail_date_str, eps, de, rev_growth),
            )
            rows_written += 1

        conn.commit()

    return rows_written


# ── DB enrichment ─────────────────────────────────────────────────────────────

def enrich_from_screener(
    repo: PlatformRepository,
    symbols: list[str] | None = None,
    delay: float = 5.0,
) -> dict[str, int]:
    """
    Fill NULL fields in historical_fundamentals using Screener.in data.

    First-non-null-wins: never overwrites a yfinance-populated value.
    Rate limit: `delay` seconds between symbol requests (default 5.0).

    Returns {symbol: n_dates_with_screener_data} summary.
    """
    if symbols is None:
        with repo.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM historical_fundamentals ORDER BY symbol"
            ).fetchall()
        symbols = [row["symbol"] for row in rows]

    if not symbols:
        _log.info("enrich_from_screener: no symbols to process")
        return {}

    _log.info(
        "enrich_from_screener: %d symbols  delay=%.1fs", len(symbols), delay
    )
    summary: dict[str, int] = {}

    for i, symbol in enumerate(symbols):
        if i > 0:
            time.sleep(delay)

        try:
            history = fetch_screener_history(symbol)
            if not history:
                _log.debug("enrich_from_screener %s: no history", symbol)
                summary[symbol] = 0
                continue

            with repo.connect() as conn:
                for date_str, data in history.items():
                    # ── Find the matching row (±15-day tolerance) ──────────
                    target     = datetime.strptime(date_str, "%Y-%m-%d")
                    low_date   = (target - timedelta(days=15)).strftime("%Y-%m-%d")
                    high_date  = (target + timedelta(days=15)).strftime("%Y-%m-%d")

                    match = conn.execute(
                        """
                        SELECT snapshot_date
                        FROM   historical_fundamentals
                        WHERE  symbol = ?
                          AND  snapshot_date BETWEEN ? AND ?
                          AND  (roce IS NULL OR promoter_holding IS NULL)
                        LIMIT  1
                        """,
                        (symbol, low_date, high_date),
                    ).fetchone()

                    if match is None:
                        continue

                    exact_date = match["snapshot_date"]

                    # ── Fill ROCE where NULL ───────────────────────────────
                    roce_val    = data.get("roce")
                    roce_source = data.get("fetched_source_roce")
                    if roce_val is not None and roce_source:
                        conn.execute(
                            """
                            UPDATE historical_fundamentals
                            SET    roce           = ?,
                                   fetched_source = CASE
                                       WHEN fetched_source IN ('yfinance') OR fetched_source IS NULL
                                       THEN ?
                                       ELSE fetched_source
                                   END
                            WHERE  symbol = ? AND snapshot_date = ? AND roce IS NULL
                            RETURNING snapshot_date
                            """,
                            (roce_val, roce_source, symbol, exact_date),
                        ).fetchone()

                    # ── Fill promoter_holding where NULL ───────────────────
                    promo_val = data.get("promoter_holding")
                    if promo_val is not None:
                        conn.execute(
                            """
                            UPDATE historical_fundamentals
                            SET    promoter_holding = ?,
                                   fetched_source   = CASE
                                       WHEN fetched_source IN ('yfinance') OR fetched_source IS NULL
                                       THEN 'screener'
                                       ELSE fetched_source
                                   END
                            WHERE  symbol = ? AND snapshot_date = ? AND promoter_holding IS NULL
                            RETURNING snapshot_date
                            """,
                            (promo_val, symbol, exact_date),
                        ).fetchone()

                conn.commit()

            summary[symbol] = len(history)
            _log.info("enrich_from_screener %s: %d dates", symbol, len(history))

        except Exception as exc:
            _log.warning("enrich_from_screener %s: %r", symbol, exc)
            summary[symbol] = -1

    total_dates = sum(v for v in summary.values() if v > 0)
    _log.info(
        "enrich_from_screener: done — %d symbol-date entries across %d symbols",
        total_dates, len(symbols),
    )
    return summary
