"""Deterministic classification for CAS holdings."""

from __future__ import annotations

import re

from wealth.importer.models import ParsedHolding
from wealth.schemas.common import Sleeve

_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_DEBT_NAMES = (
    "liquid",
    "overnight",
    "money market",
    "gilt",
    "debt",
    "bond",
    "banking & psu",
    "banking and psu",
    "short duration",
)
_GLOBAL_NAMES = (
    "nasdaq",
    "s&p",
    "global",
    "international",
    "overseas",
    "hang seng",
    "nyse",
    "fang",
)


def classify_holding(
    holding: ParsedHolding, amfi_category: str | None = None
) -> tuple[str, str] | None:
    """Prefer an AMFI master category, then fall back to deterministic name rules."""
    isin = holding.isin.upper()
    if not _ISIN.fullmatch(isin):
        return None

    if holding.kind == "sgb":
        return Sleeve.GOLD.value, "sgb"
    if holding.kind == "bond":
        return Sleeve.DEBT.value, "bond"
    if holding.kind == "equity":
        market = "india" if isin.startswith("IN") else "global"
        sleeve = Sleeve.EQUITY_INDIA if market == "india" else Sleeve.EQUITY_GLOBAL
        return sleeve.value, f"equity:{market}"
    if holding.kind not in {"mf", "etf"}:
        return None

    if holding.kind == "mf" and amfi_category and amfi_category.startswith("mf:"):
        category = amfi_category.casefold()
        if "arbitrage" in category:
            return Sleeve.DEBT.value, "mf:arbitrage:tax_note"
        if "gold" in category:
            return Sleeve.GOLD.value, "mf:gold"
        if "equity_global" in category:
            return Sleeve.EQUITY_GLOBAL.value, "mf:equity_global"
        if "debt" in category:
            return Sleeve.DEBT.value, category
        return Sleeve.EQUITY_INDIA.value, "mf:equity_india"

    name = re.sub(r"\s+", " ", holding.name.casefold()).strip()
    instrument = holding.kind
    if "arbitrage" in name:
        return Sleeve.DEBT.value, f"{instrument}:arbitrage"
    if "bharat bond" in name or any(keyword in name for keyword in _DEBT_NAMES):
        return Sleeve.DEBT.value, f"{instrument}:debt"
    if "gold" in name or "silver" in name:
        return Sleeve.GOLD.value, f"{instrument}:gold"
    if any(keyword in name for keyword in _GLOBAL_NAMES) or re.search(r"\bus\b", name):
        return Sleeve.EQUITY_GLOBAL.value, f"{instrument}:equity_global"
    if "fof" in name and "overseas" in name:
        return Sleeve.EQUITY_GLOBAL.value, f"{instrument}:equity_global"
    return Sleeve.EQUITY_INDIA.value, f"{instrument}:equity_india"


def classify_broker_security(
    market: str, symbol: str, name: str, security_type: str
) -> tuple[str, str]:
    """Classify broker securities using market and transparent name rules."""
    normalized_name = re.sub(r"\s+", " ", name.casefold()).strip()
    normalized_symbol = symbol.upper()
    if security_type == "etf" and (
        "gold" in normalized_name or normalized_symbol in {"GLD", "IAU", "GLDM", "SGOL"}
    ):
        return Sleeve.GOLD.value, "etf:gold"
    if security_type == "etf" and any(
        keyword in normalized_name for keyword in ("bond", "treasury", "tips", "aggregate")
    ):
        return Sleeve.DEBT.value, "etf:debt"
    if market == "IN":
        return Sleeve.EQUITY_INDIA.value, "equity:india"
    return Sleeve.EQUITY_GLOBAL.value, "equity:global"
