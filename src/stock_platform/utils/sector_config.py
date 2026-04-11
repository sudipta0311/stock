"""
Sector overrides and governance risk configuration.

Screener.in and yfinance sometimes misclassify stocks (e.g. BEL as Capital Goods
instead of Defence). SECTOR_OVERRIDES takes priority over any data-source label.

ELEVATED_GOVERNANCE_RISK lists conglomerate stocks that require a higher net-return
threshold before a recommendation is issued.
"""

from __future__ import annotations

# Symbol → correct sector.  Applied in get_stock_snapshot() after all data-source
# lookups, so it takes priority over Screener, yfinance, and NSE CSV labels.
SECTOR_OVERRIDES: dict[str, str] = {
    # Defence PSUs — Screener.in classifies these as Capital Goods
    "BEL":        "Defence",
    "HAL":        "Defence",
    "MTAR":       "Defence",
    "BEML":       "Defence",
    "COCHINSHIP": "Defence",
    "MAZDOCK":    "Defence",
    "GRSE":       "Defence",
    "PARAS":      "Defence",
    # CDMO / pharma services — often grouped with generic pharma
    "DIVISLAB":   "CDMO",
    "LAURUSLABS": "CDMO",
    "SYNGENE":    "CDMO",
    # Electronics manufacturing services
    "DIXON":      "Electronics Manufacturing",
    "KAYNES":     "Electronics Manufacturing",
    "AMBER":      "Electronics Manufacturing",
}


def get_sector(symbol: str, data_source_sector: str) -> str:
    """
    Return the correct sector for a stock.
    SECTOR_OVERRIDES takes priority over whatever the data source returned.
    """
    return SECTOR_OVERRIDES.get(symbol.upper().replace(".NS", ""), data_source_sector)


# Adani Group and other conglomerate stocks with elevated governance risk.
# Stocks in this set need a higher net-return to justify entry.
ELEVATED_GOVERNANCE_RISK: frozenset[str] = frozenset({
    "ADANIPOWER", "ADANIENT", "ADANIPORTS", "ADANIGREEN",
    "ADANITRANS", "ADANIGAS", "ADANIWILMAR", "NDTV",
})

# Minimum analyst-target net-return (after tax) for governance-risk stocks.
# At 2.16% the risk premium is not justified.
GOVERNANCE_RISK_MIN_NET_RETURN_PCT = 10.0  # percent (matches compute_net_return output scale)


def governance_risk_blocks(symbol: str, net_return_pct: float | None) -> tuple[bool, str]:
    """
    Return (blocked, reason).

    A governance-risk stock is blocked when its analyst-target net return is
    below GOVERNANCE_RISK_MIN_NET_RETURN_PCT or is unavailable.
    """
    clean = symbol.upper().replace(".NS", "")
    if clean not in ELEVATED_GOVERNANCE_RISK:
        return False, ""
    if net_return_pct is None or net_return_pct < GOVERNANCE_RISK_MIN_NET_RETURN_PCT:
        actual = f"{net_return_pct:.1f}%" if net_return_pct is not None else "unavailable"
        return True, (
            f"{symbol} is in the elevated governance risk group — "
            f"minimum {GOVERNANCE_RISK_MIN_NET_RETURN_PCT:.0f}% net return required, "
            f"current analyst-target return is {actual}."
        )
    return False, ""
