from __future__ import annotations

from typing import Any

import requests
from bs4 import BeautifulSoup

from stock_platform.utils.symbol_resolver import normalize_input_symbol, resolve_nse_symbol, resolve_symbol_base


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-app/1.0)"}


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
    quarter_names = [cell.get_text(" ", strip=True) for cell in header_cells[1:3]]
    if len(quarter_names) < 2:
        quarter_names = []

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

    latest = _clean_number(cells[1].get_text(" ", strip=True))
    prev = _clean_number(cells[2].get_text(" ", strip=True))
    if latest is None or prev is None:
        return {}

    growth = ((latest - prev) / prev * 100.0) if prev > 0 else 0.0
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
        "prev_quarter_revenue": prev,
        "revenue_yoy_growth_pct": round(growth, 1),
        "quarter_names": quarter_names,
        "momentum": momentum,
    }


def fetch_recent_results(symbol: str) -> dict[str, Any]:
    """
    Fetch latest quarterly revenue trend so timing can override overly
    conservative WAIT signals when momentum is clearly improving.
    """
    clean_symbol = resolve_symbol_base(normalize_input_symbol(symbol))
    try:
        soup = _fetch_soup(clean_symbol, consolidated=True)
        if soup is None:
            soup = _fetch_soup(clean_symbol, consolidated=False)
        return _parse_recent_results_from_soup(soup)
    except Exception as exc:
        print(f"Recent results fetch error for {clean_symbol}: {exc}")
        return {}


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
    revenue_growth_pct = (
        _parse_ranges_table_value(consolidated_soup, "Compounded Sales Growth", ["5 Years", "3 Years", "TTM"])
        or _parse_ranges_table_value(standalone_soup, "Compounded Sales Growth", ["5 Years", "3 Years", "TTM"])
    )
    promoter_holding = _parse_promoter_holding(consolidated_soup) or _parse_promoter_holding(standalone_soup)
    promoter_change = (
        _parse_promoter_change(consolidated_soup)
        or _parse_promoter_change(standalone_soup)
    )
    debt_to_equity = _parse_debt_to_equity(consolidated_soup)
    if debt_to_equity is None:
        debt_to_equity = _parse_debt_to_equity(standalone_soup)
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
    recent_results = _parse_recent_results_from_soup(consolidated_soup) or _parse_recent_results_from_soup(standalone_soup)
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
        "promoter_holding": promoter_holding,
        "promoter_change": promoter_change,
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


def get_stock_fundamentals(symbol: str) -> dict[str, Any]:
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
