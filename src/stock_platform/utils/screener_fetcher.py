from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from stock_platform.utils.symbol_resolver import normalize_input_symbol, resolve_nse_symbol, resolve_symbol_base

_log = logging.getLogger(__name__)


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-app/1.0)"}
_SCREENER_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_REVENUE_ROW_CANDIDATES = (
    "Total Revenue",
    "Revenue",
    "Net Revenue",
    "Operating Revenue",
)
_PAT_ROW_CANDIDATES = (
    "Net Income",
    "Net Profit",
    "Profit After Tax",
    "Net Income Common Stockholders",
)


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


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _parse_screener_quarter_label(label: str) -> tuple[int, int] | None:
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

    return month, year


def _format_fiscal_quarter_label(label: str) -> str | None:
    parsed = _parse_screener_quarter_label(label)
    if parsed is None:
        return None

    month, year = parsed
    if month in (4, 5, 6):
        quarter = "Q1"
        fy_year = year + 1
    elif month in (7, 8, 9):
        quarter = "Q2"
        fy_year = year + 1
    elif month in (10, 11, 12):
        quarter = "Q3"
        fy_year = year + 1
    else:
        quarter = "Q4"
        fy_year = year

    return f"{quarter} FY{fy_year % 100:02d}"


def _find_same_quarter_last_year_index(quarter_labels: list[str]) -> int | None:
    if not quarter_labels:
        return None

    latest = _parse_screener_quarter_label(quarter_labels[0])
    if latest is not None:
        latest_month, latest_year = latest
        for idx, label in enumerate(quarter_labels[1:], start=1):
            parsed = _parse_screener_quarter_label(label)
            if parsed == (latest_month, latest_year - 1):
                return idx

    # Screener quarterly tables are usually reverse-chronological quarter columns.
    # When parsing fails, fall back to the fifth visible quarter as the prior-year comp.
    return 4 if len(quarter_labels) >= 5 else None


def _quarter_tuple_from_column(column: Any) -> tuple[int, int, int] | None:
    if hasattr(column, "to_pydatetime"):
        try:
            column = column.to_pydatetime()
        except Exception:
            pass

    if hasattr(column, "year") and hasattr(column, "month"):
        try:
            return int(column.year), int(column.month), int(getattr(column, "day", 1))
        except Exception:
            return None

    text = str(column or "").strip()
    if not text:
        return None
    text = text.split()[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.year, parsed.month, parsed.day
        except ValueError:
            continue
    return None


def _format_fy_quarter_from_column(column: Any) -> str | None:
    parsed = _quarter_tuple_from_column(column)
    if parsed is None:
        return None

    year, month, _day = parsed
    if month in (4, 5, 6):
        quarter = "Q1"
        fy_year = year + 1
    elif month in (7, 8, 9):
        quarter = "Q2"
        fy_year = year + 1
    elif month in (10, 11, 12):
        quarter = "Q3"
        fy_year = year + 1
    else:
        quarter = "Q4"
        fy_year = year

    return f"{quarter} FY{fy_year % 100:02d}"


def _ordered_statement_columns(statement: Any) -> list[Any]:
    raw_columns = getattr(statement, "columns", None)
    if raw_columns is None:
        return []
    columns = list(raw_columns)
    if not columns:
        return []

    parsed = [(column, _quarter_tuple_from_column(column)) for column in columns]
    if any(item[1] is not None for item in parsed):
        known = [item for item in parsed if item[1] is not None]
        unknown = [item[0] for item in parsed if item[1] is None]
        known_sorted = sorted(known, key=lambda item: item[1], reverse=True)
        return [item[0] for item in known_sorted] + unknown
    return columns


def _extract_fiscal_quarter_number(label: str | None) -> str | None:
    """Return 'Q1'/'Q2'/'Q3'/'Q4' from a label like 'Q3 FY26', or None."""
    if not label:
        return None
    for q in ("Q1", "Q2", "Q3", "Q4"):
        if label.startswith(q):
            return q
    return None


def find_yoy_column(columns: list[Any], base_col_idx: int = 0) -> int | None:
    """
    Find the same-quarter-last-year column by date proximity, not fixed offset.
    Accepts the already-ordered quarterly columns and matches against the date
    closest to one year before the base column within a 45-day window.
    """
    if len(columns) < 2 or base_col_idx < 0 or base_col_idx >= len(columns):
        return None

    base_tuple = _quarter_tuple_from_column(columns[base_col_idx])
    if base_tuple is None:
        return None

    try:
        import pandas as pd

        base_date = pd.Timestamp(datetime(*base_tuple))
        target_date = base_date - pd.DateOffset(years=1)
    except Exception:
        return None

    best_idx: int | None = None
    best_diff = float("inf")

    for idx, column in enumerate(columns):
        if idx == base_col_idx:
            continue
        parsed = _quarter_tuple_from_column(column)
        if parsed is None:
            continue
        try:
            col_date = pd.Timestamp(datetime(*parsed))
        except Exception:
            continue
        diff = abs((col_date - target_date).days)
        if diff <= 45 and diff < best_diff:
            best_diff = diff
            best_idx = idx

    return best_idx


def _pick_statement_row(statement: Any, candidates: tuple[str, ...]) -> str | None:
    index = getattr(statement, "index", None)
    if index is None:
        return None
    for candidate in candidates:
        if candidate in index:
            return candidate
    lowered = {str(item).strip().lower(): item for item in index}
    for candidate in candidates:
        match = lowered.get(candidate.strip().lower())
        if match is not None:
            return str(match)
    return None


def _fetch_quarterly_income_stmt(symbol: str) -> Any | None:
    try:
        import yfinance as yf

        ticker = yf.Ticker(resolve_nse_symbol(symbol))
        statement = ticker.quarterly_income_stmt
        if statement is None or getattr(statement, "empty", True):
            return None
        return statement
    except Exception as exc:
        print(f"Quarterly income stmt fetch error for {symbol}: {exc}")
        return None


def compute_revenue_momentum(
    symbol: str,
    fin_data: dict[str, Any],
    quarterly_income_stmt: Any | None = None,
) -> dict[str, Any]:
    """
    Compute revenue momentum using most recent quarter vs same quarter last year.
    """
    clean_symbol = resolve_symbol_base(normalize_input_symbol(symbol))
    del fin_data
    statement = quarterly_income_stmt if quarterly_income_stmt is not None else _fetch_quarterly_income_stmt(clean_symbol)
    if statement is None:
        return {"momentum": "UNKNOWN", "growth_pct": None}

    columns = _ordered_statement_columns(statement)
    if len(columns) < 2:
        return {"momentum": "UNKNOWN", "growth_pct": None}

    revenue_row = _pick_statement_row(statement, _REVENUE_ROW_CANDIDATES)
    if not revenue_row:
        return {"momentum": "UNKNOWN", "growth_pct": None}

    q0_col = columns[0]
    yoy_idx = find_yoy_column(columns, 0)
    if yoy_idx is None:
        return {"momentum": "UNKNOWN", "growth_pct": None, "note": "YoY column not found"}
    q4_col = columns[yoy_idx]
    q0 = _as_float(statement.at[revenue_row, q0_col])
    q4 = _as_float(statement.at[revenue_row, q4_col])
    q1 = _as_float(statement.at[revenue_row, columns[1]]) if len(columns) > 1 else None
    if q0 is None or q4 in (None, 0):
        return {"momentum": "UNKNOWN", "growth_pct": None}

    # Validate: same fiscal quarter (Q1/Q2/Q3/Q4 must match)
    latest_label = _format_fy_quarter_from_column(q0_col)
    prior_label = _format_fy_quarter_from_column(q4_col)
    q0_qnum = _extract_fiscal_quarter_number(latest_label)
    q4_qnum = _extract_fiscal_quarter_number(prior_label)
    if q0_qnum and q4_qnum and q0_qnum != q4_qnum:
        _log.error(
            "YoY MISMATCH %s: comparing %s vs %s — must be same fiscal Q. "
            "Returning DATA_ERROR to prevent wrong momentum label.",
            clean_symbol, latest_label, prior_label,
        )
        return {
            "momentum": "UNKNOWN",
            "growth_pct": None,
            "status": "DATA_ERROR",
            "note": f"YoY quarter mismatch: {latest_label} vs {prior_label}",
        }

    growth = ((q0 - q4) / abs(q4)) * 100.0

    if abs(growth) > 200:
        _log.warning(
            "YoY OUTLIER %s: %.1f%% — verify data (latest=%.4g, prior=%.4g)",
            clean_symbol, growth, q0, q4,
        )

    signal = (
        "STRONG" if growth >= 20 else
        "MODERATE" if growth >= 10 else
        "WEAK" if growth >= 0 else
        "DECLINING"
    )
    period = (
        f"{latest_label} vs {prior_label}"
        if latest_label and prior_label
        else "latest quarter vs same quarter last year"
    )

    revenue_qoq_pct: float | None = None
    if q1 is not None and q1 != 0:
        revenue_qoq_pct = round(((q0 - q1) / abs(q1)) * 100.0, 1)

    return {
        "momentum": signal,
        "growth_pct": round(growth, 1),
        "period": period,
        "latest_quarter_revenue": q0,
        "prev_quarter_revenue": q1,
        "same_quarter_last_year_revenue": q4,
        "revenue_yoy_growth_pct": round(growth, 1),
        "revenue_qoq_pct": revenue_qoq_pct,
        "comparison_label": period,
        "source": "yfinance_quarterly_income_stmt",
    }


def compute_pat_momentum(
    symbol: str,
    fin_data: dict[str, Any],
    quarterly_income_stmt: Any | None = None,
) -> dict[str, Any]:
    """
    Compute PAT momentum separately so revenue/PAT divergence is explicit.
    """
    clean_symbol = resolve_symbol_base(normalize_input_symbol(symbol))
    statement = quarterly_income_stmt if quarterly_income_stmt is not None else _fetch_quarterly_income_stmt(clean_symbol)
    if statement is None:
        return {"pat_momentum": "UNKNOWN"}

    columns = _ordered_statement_columns(statement)
    if len(columns) < 2:
        return {"pat_momentum": "UNKNOWN"}

    pat_row = _pick_statement_row(statement, _PAT_ROW_CANDIDATES)
    if not pat_row:
        return {"pat_momentum": "UNKNOWN"}

    q0_col = columns[0]
    yoy_idx = find_yoy_column(columns, 0)
    if yoy_idx is None:
        return {"pat_momentum": "UNKNOWN", "note": "YoY column not found"}
    q4_col = columns[yoy_idx]
    pat_q0 = _as_float(statement.at[pat_row, q0_col])
    pat_q4 = _as_float(statement.at[pat_row, q4_col])
    pat_q1 = _as_float(statement.at[pat_row, columns[1]]) if len(columns) > 1 else None
    if pat_q0 is None or pat_q4 in (None, 0):
        return {"pat_momentum": "UNKNOWN"}

    pat_growth = ((pat_q0 - pat_q4) / abs(pat_q4)) * 100.0

    # QoQ PAT
    pat_qoq_pct: float | None = None
    if pat_q1 is not None and pat_q1 != 0:
        pat_qoq_pct = round(((pat_q0 - pat_q1) / abs(pat_q1)) * 100.0, 1)

    signal = (
        "STRONG" if pat_growth >= 20 else
        "MODERATE" if pat_growth >= 5 else
        "FLAT" if pat_growth >= -5 else
        "DECLINING" if pat_growth >= -25 else
        "COLLAPSING"
    )
    low_base_threshold_cr = 100.0
    pat_abs_cr = abs(pat_q0) / 1e7
    low_base_flag = pat_abs_cr < low_base_threshold_cr and pat_growth > 50
    qualifier = (
        f"(Rs.{pat_abs_cr:.0f}Cr absolute - high growth on small base)"
        if low_base_flag
        else ""
    )
    period = (
        f"{_format_fy_quarter_from_column(q0_col)} vs {_format_fy_quarter_from_column(q4_col)}"
        if _format_fy_quarter_from_column(q0_col) and _format_fy_quarter_from_column(q4_col)
        else "latest quarter vs same quarter last year"
    )
    rev_growth = (
        _as_float(fin_data.get("revenue_growth_latest_qtr"))
        or _as_float((fin_data.get("recent_results") or {}).get("revenue_yoy_growth_pct"))
        or _as_float(fin_data.get("revenue_growth_pct"))
        or 0.0
    )
    # YoY divergence: revenue growing but PAT collapsing
    yoy_divergence = rev_growth > 10 and pat_growth < -15
    # QoQ collapse: PAT dropped >50% vs prior quarter (catches seasonal/one-time spikes)
    qoq_pat_collapse = pat_qoq_pct is not None and pat_qoq_pct < -50
    divergence = yoy_divergence or qoq_pat_collapse

    if qoq_pat_collapse:
        _log.warning(
            "QoQ PAT COLLAPSE %s: %.1f%% — possible seasonality or one-time prior-quarter gain. "
            "Verify before entry.",
            clean_symbol, pat_qoq_pct,
        )

    return {
        "pat_momentum": signal,
        "pat_growth_pct": round(pat_growth, 1),
        "pat_qoq_pct": pat_qoq_pct,
        "period": period,
        "qualifier": qualifier,
        "pat_abs_cr": round(pat_abs_cr, 1),
        "rev_pat_divergence": divergence,
        "yoy_divergence": yoy_divergence,
        "qoq_pat_collapse": qoq_pat_collapse,
        "source": "yfinance_quarterly_income_stmt",
    }


def compute_profitability_snapshot(
    symbol: str,
    quarterly_income_stmt: Any | None = None,
) -> dict[str, Any]:
    """
    Compute a simple latest-quarter PAT margin snapshot from the quarterly statement.
    Returns margins in percentage terms so downstream callers can choose their own
    ratio/percent normalisation.
    """
    clean_symbol = resolve_symbol_base(normalize_input_symbol(symbol))
    statement = quarterly_income_stmt if quarterly_income_stmt is not None else _fetch_quarterly_income_stmt(clean_symbol)
    if statement is None:
        return {}

    columns = _ordered_statement_columns(statement)
    if not columns:
        return {}

    revenue_row = _pick_statement_row(statement, _REVENUE_ROW_CANDIDATES)
    pat_row = _pick_statement_row(statement, _PAT_ROW_CANDIDATES)
    if not revenue_row or not pat_row:
        return {}

    latest_col = columns[0]
    revenue = _as_float(statement.at[revenue_row, latest_col])
    pat = _as_float(statement.at[pat_row, latest_col])
    if revenue in (None, 0) or pat is None:
        return {}

    margin_pct = round((pat / revenue) * 100.0, 2)
    return {
        "pat_margin_pct": margin_pct,
        "net_margin_pct": margin_pct,
        "latest_quarter_label": _format_fy_quarter_from_column(latest_col),
        "source": "yfinance_quarterly_income_stmt",
    }


def _fetch_soup(clean_symbol: str, consolidated: bool) -> BeautifulSoup | None:
    suffix = "consolidated/" if consolidated else ""
    url = f"https://www.screener.in/company/{clean_symbol}/{suffix}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
        return BeautifulSoup(response.text, "html.parser")
    except Exception:
        return None


def _parse_top_ratios(soup: BeautifulSoup | None) -> dict[str, float]:
    if soup is None:
        return {}
    ratios: dict[str, float] = {}
    for item in soup.select("#top-ratios li"):
        name = item.select_one(".name")
        value = item.select_one(".value .number, .number")
        if not name or not value:
            continue
        parsed = _clean_number(value.get_text(" ", strip=True))
        if parsed is None:
            continue
        ratios[name.get_text(" ", strip=True)] = parsed
    return ratios


def _get_ratio_value(ratios: dict[str, float], *aliases: str) -> float | None:
    lowered = {key.strip().lower(): value for key, value in ratios.items()}
    for alias in aliases:
        value = lowered.get(alias.strip().lower())
        if value is not None:
            return value
    for key, value in lowered.items():
        if any(alias.strip().lower() in key for alias in aliases):
            return value
    return None


def _parse_ranges_table_value(
    soup: BeautifulSoup | None,
    title: str,
    preferred_rows: list[str],
) -> float | None:
    if soup is None:
        return None
    for table in soup.select("table.ranges-table"):
        heading = table.select_one("th")
        if not heading or heading.get_text(" ", strip=True) != title:
            continue
        values = {
            row[0].rstrip(":"): _clean_number(row[1])
            for row in (
                [td.get_text(" ", strip=True) for td in tr.select("td")]
                for tr in table.select("tr")[1:]
            )
            if len(row) == 2
        }
        for label in preferred_rows:
            value = values.get(label)
            if value is not None:
                return value
    return None


def _parse_promoter_change(soup: BeautifulSoup | None) -> float | None:
    """
    Return promoter holding change in percentage points (latest quarter minus previous).
    Positive = promoter increased stake; negative = reduced stake.
    """
    if soup is None:
        return None
    try:
        sh_tables = soup.select(".shareholding-table")
        if sh_tables:
            rows = sh_tables[0].select("tr")
            for row in rows:
                if "promoter" in row.text.lower():
                    cells = row.select("td")
                    if len(cells) >= 3:
                        try:
                            latest = float(cells[1].text.strip().replace("%", ""))
                            prev = float(cells[2].text.strip().replace("%", ""))
                            return latest - prev
                        except Exception:
                            pass
    except Exception:
        pass
    return None


def _parse_promoter_change_3yr(soup: BeautifulSoup | None) -> float | None:
    """
    Return the relative promoter holding change versus roughly 3 years ago.
    Negative values mean promoters have been selling.
    """
    if soup is None:
        return None
    try:
        for table in soup.select(".shareholding-table, #quarterly-shp table, #yearly-shp table"):
            for row in table.select("tr"):
                cells = row.select("td, th")
                if len(cells) < 3:
                    continue
                label = cells[0].get_text(" ", strip=True).lower()
                if "promoter" not in label:
                    continue
                values = [_clean_number(cell.get_text(" ", strip=True)) for cell in cells[1:]]
                values = [value for value in values if value is not None]
                if len(values) < 2:
                    continue
                compare_idx = 12 if len(values) > 12 else len(values) - 1
                latest = values[0]
                oldest = values[compare_idx]
                if latest is None or oldest in (None, 0):
                    continue
                return (latest - oldest) / abs(oldest)
    except Exception:
        pass
    return None


def _parse_promoter_holding(soup: BeautifulSoup | None) -> float | None:
    if soup is None:
        return None
    for table in soup.select("#quarterly-shp table, #yearly-shp table"):
        for row in table.select("tbody tr"):
            cells = row.select("td")
            if len(cells) < 2:
                continue
            label = cells[0].get_text(" ", strip=True).lower()
            if "promoter" not in label:
                continue
            for cell in reversed(cells[1:]):
                value = _clean_number(cell.get_text(" ", strip=True))
                if value is not None:
                    return value
    return None


def _parse_pledge_data(soup: BeautifulSoup | None) -> dict[str, Any]:
    """
    Parse promoter pledge % from the quarterly shareholding table on Screener.in.
    Returns pledge_pct (latest), pledge_history (up to 4 quarters), and
    pledge_trend (RISING/STABLE/FALLING) derived from the quarterly series.
    """
    if soup is None:
        return {"pledge_pct": None, "pledge_history": [], "pledge_trend": None}

    history: list[float] = []
    for table in soup.select("#quarterly-shp table, #yearly-shp table"):
        for row in table.select("tbody tr"):
            cells = row.select("td")
            if len(cells) < 2:
                continue
            label = cells[0].get_text(" ", strip=True).lower()
            if "pledge" not in label:
                continue
            for cell in cells[1:5]:  # up to 4 quarters
                val = _clean_number(cell.get_text(" ", strip=True))
                if val is not None:
                    history.append(val)
            break  # found the pledge row

    if not history:
        return {"pledge_pct": None, "pledge_history": [], "pledge_trend": None}

    pledge_pct = history[0]  # most recent quarter first on Screener

    # Trend: compare latest vs oldest of the available quarters
    if len(history) >= 2:
        delta = history[0] - history[-1]
        if delta > 2.0:
            trend = "RISING"
        elif delta < -2.0:
            trend = "FALLING"
        else:
            trend = "STABLE"
    else:
        trend = "STABLE"

    return {"pledge_pct": pledge_pct, "pledge_history": history, "pledge_trend": trend}


def _parse_balance_sheet_value(soup: BeautifulSoup | None, row_label: str) -> float | None:
    if soup is None:
        return None
    for section in soup.select("section"):
        heading = section.select_one("h2")
        if not heading or heading.get_text(" ", strip=True) != "Balance Sheet":
            continue
        table = section.select_one("table")
        if table is None:
            continue
        for row in table.select("tbody tr"):
            cells = row.select("td")
            if len(cells) < 2:
                continue
            label = cells[0].get_text(" ", strip=True)
            if label != row_label:
                continue
            for cell in reversed(cells[1:]):
                value = _clean_number(cell.get_text(" ", strip=True))
                if value is not None:
                    return value
    return None


def _parse_debt_to_equity(soup: BeautifulSoup | None) -> float | None:
    if soup is None:
        return None
    borrowings = _parse_balance_sheet_value(soup, "Borrowings +") or _parse_balance_sheet_value(soup, "Borrowings")
    equity_capital = _parse_balance_sheet_value(soup, "Equity Capital")
    reserves = _parse_balance_sheet_value(soup, "Reserves")
    if borrowings is None or equity_capital is None or reserves is None:
        return None
    equity = equity_capital + reserves
    if equity <= 0:
        return None
    return borrowings / equity


def _parse_recent_results_from_soup(soup: BeautifulSoup | None) -> dict[str, Any]:
    if soup is None:
        return {}

    quarters = soup.select("#quarters table tr")
    if len(quarters) < 2:
        return {}

    header_cells = quarters[0].select("th, td")
    quarter_labels = [cell.get_text(" ", strip=True) for cell in header_cells[1:]]
    if not quarter_labels:
        return {}

    sales_row = None
    for row in quarters[1:]:
        first_cell = row.select_one("th, td")
        label = first_cell.get_text(" ", strip=True).lower() if first_cell else ""
        if "sales" in label or "revenue" in label:
            sales_row = row
            break

    if sales_row is None:
        return {}

    cells = sales_row.select("td, th")
    if len(cells) < 3:
        return {}

    sales_values = [_clean_number(cell.get_text(" ", strip=True)) for cell in cells[1:]]
    latest = sales_values[0] if sales_values else None
    prev_quarter = sales_values[1] if len(sales_values) > 1 else None
    same_qtr_idx = _find_same_quarter_last_year_index(quarter_labels)
    same_qtr_last_year = (
        sales_values[same_qtr_idx]
        if same_qtr_idx is not None and same_qtr_idx < len(sales_values)
        else None
    )
    if latest is None or same_qtr_last_year is None:
        return {}

    latest_label = quarter_labels[0]
    same_qtr_label = quarter_labels[same_qtr_idx] if same_qtr_idx is not None else ""
    formatted_latest = _format_fiscal_quarter_label(latest_label)
    formatted_same_qtr = _format_fiscal_quarter_label(same_qtr_label)
    comparison_label = (
        f"{formatted_latest} vs {formatted_same_qtr}"
        if formatted_latest and formatted_same_qtr
        else (
            f"{latest_label} vs {same_qtr_label}"
            if latest_label and same_qtr_label
            else "latest quarter vs same quarter last year"
        )
    )

    growth = ((latest - same_qtr_last_year) / same_qtr_last_year * 100.0) if same_qtr_last_year > 0 else 0.0
    if growth > 30:
        momentum = "STRONG"
    elif growth > 15:
        momentum = "GOOD"
    elif growth > 5:
        momentum = "MODERATE"
    else:
        momentum = "WEAK"

    return {
        "latest_quarter_revenue": latest,
        "prev_quarter_revenue": prev_quarter,
        "same_quarter_last_year_revenue": same_qtr_last_year,
        "revenue_yoy_growth_pct": round(growth, 1),
        "quarter_names": [latest_label, same_qtr_label] if same_qtr_label else [latest_label],
        "latest_quarter_label": latest_label,
        "same_quarter_last_year_label": same_qtr_label,
        "comparison_label": comparison_label,
        "momentum": momentum,
    }


def fetch_recent_results(symbol: str) -> dict[str, Any]:
    """
    Fetch latest quarterly revenue trend so timing can override overly
    conservative WAIT signals when momentum is clearly improving.
    """
    clean_symbol = resolve_symbol_base(normalize_input_symbol(symbol))
    try:
        quarterly_stmt = _fetch_quarterly_income_stmt(clean_symbol)
        revenue_momentum = compute_revenue_momentum(clean_symbol, {}, quarterly_income_stmt=quarterly_stmt)
        if revenue_momentum.get("growth_pct") is not None:
            return revenue_momentum

        soup = _fetch_soup(clean_symbol, consolidated=True)
        if soup is None:
            soup = _fetch_soup(clean_symbol, consolidated=False)
        return _parse_recent_results_from_soup(soup)
    except Exception as exc:
        print(f"Recent results fetch error for {clean_symbol}: {exc}")
        return {}


def _parse_bse_table_row(row: dict) -> dict:
    """Extract revenue/PAT from a BSE filing table row, handling varied field names."""
    revenue = _clean_number(str(
        row.get("Revenue") or row.get("SALES") or row.get("NetSales")
        or row.get("NET_SALES") or row.get("Income") or ""
    ))
    pat = _clean_number(str(
        row.get("PAT") or row.get("NET_PROFIT") or row.get("NetProfit")
        or row.get("Profit") or ""
    ))
    period = row.get("PERIOD_END") or row.get("TO_DATE") or row.get("QuarterEnded") or row.get("PeriodEnd")
    return {"revenue": revenue, "pat": pat, "period_end": period, "source": "BSE_FILING"}


def fetch_bse_quarterly(symbol: str) -> dict | None:
    """
    BSE corporate filings — most authoritative source when accessible.
    Returns latest and prior-year quarterly revenue/PAT, or None if unavailable.

    Tries two endpoints in order:
      1. AnnSubCategoryGetData (quarterly results, less session-gated)
      2. AnnGetData (legacy, session-gated on cloud)
    """
    from stock_platform.utils.bse_codes import SYMBOL_TO_BSE_CODE

    bse_code = SYMBOL_TO_BSE_CODE.get(symbol)
    if not bse_code:
        return None

    endpoints = [
        (
            f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
            f"?strCat=Result&strScrip={bse_code}&strType=C&strPType=Quarterly"
        ),
        (
            f"https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
            f"?strCat=Result&strScrip={bse_code}&strType=C"
        ),
    ]
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    }
    for url in endpoints:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue
            data = response.json()
            table = data.get("Table") or data.get("Table1") or []
            if not table:
                continue
            latest = _parse_bse_table_row(table[0])
            prior = _parse_bse_table_row(table[4]) if len(table) > 4 else None
            if latest.get("revenue") is None:
                continue
            _log.info("BSE filing fetched for %s via %s", symbol, url.split("?")[0].split("/")[-1])
            return {"latest": latest, "prior_year": prior}
        except Exception as exc:
            _log.debug("BSE endpoint failed %s (%s): %s", symbol, url.split("?")[0].split("/")[-1], exc)
    _log.warning("BSE fetch unavailable for %s — both endpoints returned no data", symbol)
    return None


def _compute_yoy_from_bse(bse_data: dict) -> float | None:
    """Compute YoY revenue growth from BSE filing data (latest vs prior-year quarter)."""
    latest = (bse_data or {}).get("latest") or {}
    prior = (bse_data or {}).get("prior_year") or {}
    rev_latest = _as_float(latest.get("revenue"))
    rev_prior = _as_float(prior.get("revenue"))
    if rev_latest is None or rev_prior in (None, 0):
        return None
    return round((rev_latest - rev_prior) / abs(rev_prior) * 100.0, 1)


def compute_yoy_from_quarterly(data: dict) -> float | None:
    """Compute YoY revenue growth from quarterly filing data (latest vs prior-year quarter).

    Accepts the {'latest': {'revenue': X}, 'prior_year': {'revenue': Y}} shape
    returned by fetch_nse_quarterly (and historically fetch_bse_quarterly).
    """
    latest = (data or {}).get("latest") or {}
    prior = (data or {}).get("prior_year") or {}
    rev_latest = _as_float(latest.get("revenue"))
    rev_prior = _as_float(prior.get("revenue"))
    if rev_latest is None or rev_prior in (None, 0):
        return None
    return round((rev_latest - rev_prior) / abs(rev_prior) * 100.0, 1)


def fetch_nse_quarterly(symbol: str) -> dict | None:
    """
    NSE corporate filings — authoritative tiebreaker for YoY disagreements.
    Replaced fetch_bse_quarterly (2026-04-26) after BSE API silently started
    returning empty Table responses on all symbols.

    Two-call flow:
      1. List endpoint  → find most-recent consolidated quarterly + same-quarter prior year
      2. Detail endpoint → extract re_net_sale for each filing

    Returns {'latest': {'revenue': X, 'period_end': ...},
             'prior_year': {'revenue': Y, 'period_end': ...}}
    or None if unavailable.
    """
    import datetime as _dt

    _headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        session = requests.Session()
        session.headers.update(_headers)
        session.get("https://www.nseindia.com/", timeout=10)

        r = session.get(
            f"https://www.nseindia.com/api/corporates-financial-results"
            f"?index=equities&symbol={symbol}&period=Quarterly",
            timeout=15,
        )
        if r.status_code != 200:
            _log.debug("NSE list endpoint HTTP %s for %s", r.status_code, symbol)
            return None
        items = r.json()
        if not isinstance(items, list) or not items:
            return None

        consol = [
            x for x in items
            if x.get("consolidated") == "Consolidated"
            and x.get("cumulative") == "Non-cumulative"
        ]
        if len(consol) < 2:
            _log.debug("NSE: insufficient consolidated quarterly filings for %s (%d)", symbol, len(consol))
            return None

        latest_item = consol[0]
        latest_end = _dt.datetime.strptime(latest_item["toDate"], "%d-%b-%Y")
        yoy_target = latest_end.replace(year=latest_end.year - 1)

        prior_candidates = [
            x for x in consol[1:]
            if _dt.datetime.strptime(x["toDate"], "%d-%b-%Y") < latest_end - _dt.timedelta(days=300)
        ]
        if not prior_candidates:
            _log.debug("NSE: no prior-year quarter found for %s", symbol)
            return None
        prior_item = min(
            prior_candidates,
            key=lambda x: abs((_dt.datetime.strptime(x["toDate"], "%d-%b-%Y") - yoy_target).days),
        )

        def _nse_revenue(item: dict) -> dict | None:
            rd = session.get(
                f"https://www.nseindia.com/api/corporates-financial-results-data"
                f"?index=equities"
                f"&params={item['params']}"
                f"&seq_id={item['seqNumber']}"
                f"&industry={item.get('industry', '-')}"
                f"&ind=C"
                f"&format={item.get('format', 'New')}",
                timeout=15,
            )
            if rd.status_code != 200:
                return None
            d = rd.json()
            results = d.get("resultsData2") or d.get("resultsData") or {}
            net_sale = (
                results.get("re_net_sale")
                or results.get("re_income_rev_op")
                or results.get("re_income")
            )
            if net_sale is None:
                return None
            return {
                "revenue": _as_float(str(net_sale)),
                "period_end": item.get("toDate"),
                "source": "NSE_FILING",
            }

        latest_data = _nse_revenue(latest_item)
        if not latest_data or not latest_data.get("revenue"):
            _log.debug("NSE: could not extract revenue for %s latest quarter", symbol)
            return None

        prior_data = _nse_revenue(prior_item)
        _log.info(
            "NSE filing fetched for %s: latest=%s rev=%.0f, prior=%s rev=%s",
            symbol, latest_item.get("toDate"), latest_data["revenue"],
            prior_item.get("toDate"), prior_data.get("revenue") if prior_data else "N/A",
        )
        return {"latest": latest_data, "prior_year": prior_data}

    except Exception as exc:
        _log.debug("NSE quarterly fetch failed for %s: %s", symbol, exc)
        return None


def resolve_yoy_disagreement(
    symbol: str,
    yfinance_yoy: float | None,
    screener_consol_yoy: float | None,
    standalone_yoy: float | None = None,
    tolerance: float = 5.0,  # unused in new logic; kept for call-site compat
) -> dict:
    """
    Resolve YoY revenue growth across sources.

    Priority:
      1. consolidated + yfinance agree within 15pp → average → HIGH
      2. consolidated + yfinance disagree → try NSE filing → HIGH
      3. consolidated + yfinance disagree, NSE unavailable → exclude → LOW
      4. only consolidated available → MEDIUM (no exclude)
      5. only yfinance available → MEDIUM (no exclude)
      6. only standalone available → LOW, exclude
      7. no data → exclude
    """
    yf = yfinance_yoy
    sc_consol = screener_consol_yoy
    sc_standalone = standalone_yoy

    # Priority 1–3: both primary sources present
    if sc_consol is not None and yf is not None:
        if abs(yf - sc_consol) <= 15:
            avg = round((yf + sc_consol) / 2, 1)
            return {
                "yoy_pct": avg,
                "source": "yf+consolidated_avg",
                "yoy_confidence": "HIGH",
                "exclude_from_recommendations": False,
            }
        # Disagree >15pp — try NSE filing as authoritative tiebreaker
        nse_data = fetch_nse_quarterly(symbol)
        nse_yoy = compute_yoy_from_quarterly(nse_data) if nse_data else None
        if nse_yoy is not None:
            _log.info(
                "YoY RESOLVED via NSE %s: yf=%.1f%% vs sc_consol=%.1f%% disagree — NSE=%.1f%%",
                symbol, yf, sc_consol, nse_yoy,
            )
            return {
                "yoy_pct": nse_yoy,
                "source": "NSE_FILING",
                "yoy_confidence": "HIGH",
                "exclude_from_recommendations": False,
            }
        _log.warning(
            "YoY EXCLUDED %s: yf=%.1f%% vs sc_consol=%.1f%% disagree by %.1fpp — "
            "NSE unavailable. Excluding from recommendations.",
            symbol, yf, sc_consol, abs(yf - sc_consol),
        )
        return {
            "yoy_pct": min(yf, sc_consol),
            "source": "disagree_excluded",
            "yoy_confidence": "LOW",
            "exclude_from_recommendations": True,
        }

    # Priority 4: only consolidated available (holding companies, multi-segment)
    if sc_consol is not None:
        return {
            "yoy_pct": sc_consol,
            "source": "consolidated_only",
            "yoy_confidence": "MEDIUM",
            "exclude_from_recommendations": False,
        }

    # Priority 5: only yfinance available
    if yf is not None:
        return {
            "yoy_pct": yf,
            "source": "yfinance_only",
            "yoy_confidence": "MEDIUM",
            "exclude_from_recommendations": False,
        }

    # Priority 6: standalone only — unreliable for holding companies / multi-segment
    if sc_standalone is not None:
        _log.info("%s: only standalone available — using with low confidence", symbol)
        return {
            "yoy_pct": sc_standalone,
            "source": "standalone_only",
            "yoy_confidence": "LOW",
            "exclude_from_recommendations": True,
        }

    return {
        "yoy_pct": None,
        "source": "no_data",
        "yoy_confidence": "NONE",
        "exclude_from_recommendations": True,
    }



def fetch_screener_data(nse_symbol: str) -> dict[str, Any]:
    """
    Fetch standardized Indian-equity fundamentals from Screener.in.
    Consolidated is preferred, with standalone fallback for sparse/recently demerged listings.
    """
    clean_symbol = resolve_symbol_base(normalize_input_symbol(nse_symbol))
    consolidated_soup = _fetch_soup(clean_symbol, consolidated=True)
    standalone_soup = _fetch_soup(clean_symbol, consolidated=False)
    if consolidated_soup is None and standalone_soup is None:
        print(f"Screener.in: {clean_symbol} unavailable")
        return {}

    consolidated_ratios = _parse_top_ratios(consolidated_soup)
    standalone_ratios = _parse_top_ratios(standalone_soup)

    current_price = consolidated_ratios.get("Current Price") or standalone_ratios.get("Current Price")
    pe_ratio = consolidated_ratios.get("Stock P/E") or standalone_ratios.get("Stock P/E")
    week52_high = (
        _get_ratio_value(consolidated_ratios, "52 week high", "high")
        or _get_ratio_value(standalone_ratios, "52 week high", "high")
    )
    week52_low = (
        _get_ratio_value(consolidated_ratios, "52 week low", "low")
        or _get_ratio_value(standalone_ratios, "52 week low", "low")
    )
    eps = None
    if current_price and pe_ratio and pe_ratio != 0:
        eps = round(current_price / pe_ratio, 2)

    roe_pct = (
        consolidated_ratios.get("ROE")
        or standalone_ratios.get("ROE")
        or _parse_ranges_table_value(standalone_soup, "Return on Equity", ["Last Year", "3 Years", "5 Years"])
        or _parse_ranges_table_value(consolidated_soup, "Return on Equity", ["Last Year", "3 Years", "5 Years"])
    )
    roce_pct = consolidated_ratios.get("ROCE") or standalone_ratios.get("ROCE") or roe_pct
    revenue_growth_ttm = (
        _parse_ranges_table_value(consolidated_soup, "Compounded Sales Growth", ["TTM"])
        or _parse_ranges_table_value(standalone_soup, "Compounded Sales Growth", ["TTM"])
    )
    revenue_growth_pct = (
        revenue_growth_ttm
        if revenue_growth_ttm is not None
        else (
            _parse_ranges_table_value(consolidated_soup, "Compounded Sales Growth", ["3 Years", "5 Years"])
            or _parse_ranges_table_value(standalone_soup, "Compounded Sales Growth", ["3 Years", "5 Years"])
        )
    )
    promoter_holding = _parse_promoter_holding(consolidated_soup) or _parse_promoter_holding(standalone_soup)
    promoter_change = (
        _parse_promoter_change(consolidated_soup)
        or _parse_promoter_change(standalone_soup)
    )
    promoter_change_3yr = (
        _parse_promoter_change_3yr(consolidated_soup)
        or _parse_promoter_change_3yr(standalone_soup)
    )
    debt_to_equity = _parse_debt_to_equity(consolidated_soup)
    if debt_to_equity is None:
        debt_to_equity = _parse_debt_to_equity(standalone_soup)
    market_cap_cr = (
        _get_ratio_value(consolidated_ratios, "market cap")
        or _get_ratio_value(standalone_ratios, "market cap")
    )
    dma_200 = (
        _get_ratio_value(consolidated_ratios, "200 DMA", "200D MA")
        or _get_ratio_value(standalone_ratios, "200 DMA", "200D MA")
    )
    dma_50 = (
        _get_ratio_value(consolidated_ratios, "50 DMA", "50D MA")
        or _get_ratio_value(standalone_ratios, "50 DMA", "50D MA")
    )
    pe_5yr_median = (
        _get_ratio_value(consolidated_ratios, "median pe", "p/e median", "pe median")
        or _get_ratio_value(standalone_ratios, "median pe", "p/e median", "pe median")
    )
    pb_ratio = (
        _get_ratio_value(consolidated_ratios, "price to book value", "p/b", "pb")
        or _get_ratio_value(standalone_ratios, "price to book value", "p/b", "pb")
    )
    quarterly_stmt = _fetch_quarterly_income_stmt(clean_symbol)
    revenue_momentum = compute_revenue_momentum(clean_symbol, {}, quarterly_income_stmt=quarterly_stmt)
    # Parse consolidated and standalone Screener HTML independently (both soups already fetched).
    _sc_consol_momentum = _parse_recent_results_from_soup(consolidated_soup) or {}
    _sc_standalone_momentum = _parse_recent_results_from_soup(standalone_soup) or {}
    _yf_pct = revenue_momentum.get("growth_pct") if revenue_momentum.get("status") != "DATA_ERROR" else None
    _sc_consol_pct = _sc_consol_momentum.get("revenue_yoy_growth_pct")
    _sc_standalone_pct = _sc_standalone_momentum.get("revenue_yoy_growth_pct")

    # 3-source resolution: yfinance + Screener consolidated + Screener standalone.
    # Falls back to NSE filing when primary sources disagree >15pp.
    _yoy_resolution = resolve_yoy_disagreement(
        clean_symbol, _yf_pct, _sc_consol_pct, _sc_standalone_pct
    )
    _resolved_yoy = _yoy_resolution.get("yoy_pct")
    _yoy_confidence = _yoy_resolution.get("yoy_confidence", "HIGH")
    _yoy_source = _yoy_resolution.get("source", "unknown")
    _exclude_for_data_quality = bool(_yoy_resolution.get("exclude_from_recommendations"))

    # Pick the best revenue_momentum dict to drive downstream labels.
    if _yf_pct is not None:
        revenue_momentum = revenue_momentum  # yfinance preferred when valid
    elif _sc_consol_momentum:
        revenue_momentum = _sc_consol_momentum
    else:
        revenue_momentum = _sc_standalone_momentum

    # Overwrite the YoY figure in whichever momentum dict we chose.
    if _resolved_yoy is not None:
        revenue_momentum = dict(revenue_momentum)
        revenue_momentum["revenue_yoy_growth_pct"] = _resolved_yoy
        revenue_momentum["growth_pct"] = _resolved_yoy
        revenue_momentum["yoy_source"] = _yoy_source
        revenue_momentum["yoy_confidence"] = _yoy_confidence

    revenue_growth_latest_qtr = (
        revenue_momentum.get("growth_pct")
        if revenue_momentum.get("growth_pct") is not None
        else revenue_momentum.get("revenue_yoy_growth_pct")
    )
    pat_momentum = compute_pat_momentum(
        clean_symbol,
        {
            "revenue_growth_latest_qtr": revenue_growth_latest_qtr,
            "recent_results": revenue_momentum,
            "revenue_growth_pct": revenue_growth_pct,
        },
        quarterly_income_stmt=quarterly_stmt,
    )
    profitability = compute_profitability_snapshot(clean_symbol, quarterly_income_stmt=quarterly_stmt)
    recent_results = dict(revenue_momentum or {})
    if pat_momentum:
        recent_results.update(pat_momentum)
    pledge_data = _parse_pledge_data(consolidated_soup) or _parse_pledge_data(standalone_soup)

    result = {
        "roce_pct": roce_pct,
        "roe_pct": roe_pct,
        "debt_to_equity": debt_to_equity,
        "pe_ratio": pe_ratio,
        "pe_5yr_median": pe_5yr_median,
        "pb_ratio": pb_ratio,
        "eps": eps,
        "revenue_growth_pct": revenue_growth_pct,
        "revenue_growth_ttm": revenue_growth_ttm,
        "revenue_growth_latest_qtr": revenue_growth_latest_qtr,
        "revenue_momentum": revenue_momentum,
        "pat_momentum": pat_momentum,
        "pat_margin_pct": profitability.get("pat_margin_pct"),
        "net_margin_pct": profitability.get("net_margin_pct"),
        "promoter_holding": promoter_holding,
        "promoter_change": promoter_change,
        "promoter_holding_change_3yr": promoter_change_3yr,
        "market_cap_cr": market_cap_cr,
        "pledge_pct": pledge_data.get("pledge_pct"),
        "pledge_trend": pledge_data.get("pledge_trend"),
        "pledge_history": pledge_data.get("pledge_history") or [],
        "dma_200": dma_200,
        "dma_50": dma_50,
        "current_price": current_price,
        "week52_high": week52_high,
        "week52_low": week52_low,
        "recent_results": recent_results,
        "source": "screener.in",
        "symbol": clean_symbol,
        "resolved_symbol": resolve_nse_symbol(clean_symbol),
        "consolidated_available": consolidated_soup is not None,
        "standalone_available": standalone_soup is not None,
        "exclude_from_recommendations": _exclude_for_data_quality,
        "yoy_confidence": _yoy_confidence,
        "yoy_source": _yoy_source,
    }
    print(
        f"Screener.in {clean_symbol}: ROCE={result['roce_pct']}%, "
        f"D/E={result['debt_to_equity']}, PE={result['pe_ratio']}, "
        f"PE5yrMedian={result['pe_5yr_median']}, EPS={result['eps']}"
    )
    return {key: value for key, value in result.items() if value is not None}


def compute_pe_context(
    symbol: str,
    fin_data: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """
    Compute verified PE context from real screener data.
    Prevents the LLM from estimating PE incorrectly.
    Returns a dict with pe_current, pe_5yr_median, pe_assessment, pe_signal.
    """
    pe_current = fin_data.get("pe_ratio") or fin_data.get("Stock P/E")
    pe_5yr_med = fin_data.get("pe_5yr_median")
    eps = fin_data.get("eps") or fin_data.get("trailingEps")

    # Derive PE from price/EPS if the ratio wasn't fetched directly
    if not pe_current and eps and float(eps) > 0 and current_price and current_price > 0:
        pe_current = round(float(current_price) / float(eps), 1)

    if not pe_current:
        return {
            "pe_current": None,
            "pe_5yr_median": None,
            "pe_vs_median_pct": None,
            "pe_assessment": "PE data unavailable",
            "pe_signal": "NEUTRAL",
        }

    pe_current = float(pe_current)

    if pe_5yr_med and float(pe_5yr_med) > 0:
        pe_5yr_med = float(pe_5yr_med)
        pe_vs_median_pct = round((pe_current - pe_5yr_med) / pe_5yr_med * 100, 1)
    else:
        pe_5yr_med = None
        pe_vs_median_pct = None

    if pe_vs_median_pct is not None:
        if pe_vs_median_pct < -20:
            pe_signal = "CHEAP_VS_HISTORY"
            assessment = (
                f"PE {pe_current:.1f}x is {abs(pe_vs_median_pct):.0f}% BELOW "
                f"5-year median {pe_5yr_med:.1f}x — stock is CHEAP relative to own history"
            )
        elif pe_vs_median_pct < 0:
            pe_signal = "FAIR_VS_HISTORY"
            assessment = (
                f"PE {pe_current:.1f}x is {abs(pe_vs_median_pct):.0f}% below "
                f"5-year median {pe_5yr_med:.1f}x — fair value relative to own history"
            )
        elif pe_vs_median_pct < 20:
            pe_signal = "SLIGHT_PREMIUM"
            assessment = (
                f"PE {pe_current:.1f}x is {pe_vs_median_pct:.0f}% above "
                f"5-year median {pe_5yr_med:.1f}x — slight premium to own history"
            )
        else:
            pe_signal = "EXPENSIVE_VS_HISTORY"
            assessment = (
                f"PE {pe_current:.1f}x is {pe_vs_median_pct:.0f}% ABOVE "
                f"5-year median {pe_5yr_med:.1f}x — expensive relative to own history"
            )
    else:
        pe_signal = "NEUTRAL"
        assessment = f"PE {pe_current:.1f}x — no historical median available for comparison"

    return {
        "pe_current": pe_current,
        "pe_5yr_median": pe_5yr_med,
        "pe_vs_median_pct": pe_vs_median_pct,
        "pe_assessment": assessment,
        "pe_signal": pe_signal,
    }


def get_stock_fundamentals(symbol: str, include_market_data: bool = True) -> dict[str, Any]:
    """
    Resolve a symbol, fetch Screener fundamentals, then enrich with yfinance price metadata.
    """
    resolved = resolve_nse_symbol(symbol)
    clean = resolve_symbol_base(symbol)
    data = fetch_screener_data(clean)
    data.setdefault("symbol", clean)
    data["resolved_symbol"] = resolved
    data["input_symbol"] = normalize_input_symbol(symbol)
    data["symbol_mapped"] = normalize_input_symbol(symbol) != clean

    if include_market_data:
        try:
            import yfinance as yf

            ticker = yf.Ticker(resolved)
            history = ticker.history(period="1d")
            if not history.empty:
                data["current_price"] = float(history["Close"].iloc[-1])
            info = ticker.info or {}
            target = info.get("targetMeanPrice")
            if target is not None:
                data["target_mean_price"] = float(target)
        except Exception:
            pass

    return data
