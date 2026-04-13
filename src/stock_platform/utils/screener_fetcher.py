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
    debt_to_equity = _parse_debt_to_equity(consolidated_soup)
    if debt_to_equity is None:
        debt_to_equity = _parse_debt_to_equity(standalone_soup)

    result = {
        "roce_pct": roce_pct,
        "roe_pct": roe_pct,
        "debt_to_equity": debt_to_equity,
        "pe_ratio": pe_ratio,
        "eps": eps,
        "revenue_growth_pct": revenue_growth_pct,
        "promoter_holding": promoter_holding,
        "current_price": current_price,
        "week52_high": week52_high,
        "week52_low": week52_low,
        "source": "screener.in",
        "symbol": clean_symbol,
        "resolved_symbol": resolve_nse_symbol(clean_symbol),
        "consolidated_available": consolidated_soup is not None,
        "standalone_available": standalone_soup is not None,
    }
    print(
        f"Screener.in {clean_symbol}: ROCE={result['roce_pct']}%, "
        f"D/E={result['debt_to_equity']}, PE={result['pe_ratio']}, EPS={result['eps']}"
    )
    return {key: value for key, value in result.items() if value is not None}


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
