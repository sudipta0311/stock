"""Deterministic fund ranking and optional US stock-pick limits for Denmark."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from wealth.policy.actions import SatelliteIdea

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "eu_etfs.yaml"
_SIZE_RANK = {"small": 0, "medium": 1, "large": 2}
_CURRENCY_RANK = {"DKK": 0, "EUR": 1, "USD": 2, "GBP": 3}
_BROKERS = {"nordnet", "saxo", "danske_bank", "other"}


@dataclass(frozen=True)
class EtfPick:
    """One rulebook ETF and the selected exchange listing."""

    instrument: Mapping[str, Any]
    exchange: str
    ticker: str
    currency: str


def plain_fund_name(name: str) -> str:
    """Hide provider jargon in names shown to people while keeping source names in rules."""
    return re.sub(r"\bUCITS\s+ETF\b", "fund", name, flags=re.IGNORECASE)


@lru_cache(maxsize=1)
def load_eu_etfs() -> dict[str, Any]:
    """Load the checked-in sourced fund universe and fail closed on incomplete rows."""
    document = yaml.safe_load(_RULES_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("unsupported EU ETF rulebook")
    rows = document.get("instruments")
    if not isinstance(rows, list):
        raise ValueError("EU ETF rules must contain an instruments list")
    seen_isins: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("EU ETF rows must be mappings")
        isin = row.get("isin")
        if not isinstance(isin, str) or len(isin) != 12 or isin in seen_isins:
            raise ValueError("EU ETF ISINs must be unique 12-character identifiers")
        seen_isins.add(isin)
        for key in (
            "factsheet_url",
            "source_url",
            "abis_source_url",
            "ask_source_url",
            "fund_size_source_url",
        ):
            if not isinstance(row.get(key), str) or not row[key].startswith("https://"):
                raise ValueError(f"EU ETF {isin} requires {key}")
        if row.get("verified") is not False or row.get("tax_verified") is not False:
            raise ValueError("Danish tax facts must remain unverified")
        if not isinstance(row.get("abis_list_year"), int):
            raise ValueError("EU ETF rows require the year of the source list")
        if not isinstance(row.get("tickers"), dict) or not row["tickers"]:
            raise ValueError("EU ETF rows require sourced exchange tickers")
        if not isinstance(row.get("available_at"), list):
            raise ValueError("EU ETF rows require broker availability")
    return document


def _listing(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    tickers = row.get("tickers")
    if not isinstance(tickers, Mapping):
        return None
    listings: list[tuple[int, str, str, str]] = []
    for exchange, listing in tickers.items():
        if not isinstance(exchange, str) or not isinstance(listing, Mapping):
            continue
        symbol = listing.get("symbol")
        currency = listing.get("currency")
        if isinstance(symbol, str) and currency in _CURRENCY_RANK:
            listings.append((_CURRENCY_RANK[currency], exchange, symbol, currency))
    if not listings:
        return None
    _rank, exchange, symbol, currency = min(listings)
    return exchange, symbol, currency


def ranked_etfs(
    exposure: str,
    broker: str,
    *,
    ask_only: bool = False,
    broker_filter: bool = True,
    rules: Mapping[str, Any] | None = None,
) -> list[EtfPick]:
    """Filter the sourced products and rank by fee, income policy, size, then currency."""
    selected = rules or load_eu_etfs()
    if broker not in _BROKERS:
        return []
    eligible: list[tuple[tuple[Decimal, int, int, int, str], EtfPick]] = []
    for row in selected.get("instruments", []):
        if (
            not isinstance(row, Mapping)
            or row.get("exposure") != exposure
            or row.get("abis_listed") is not True
            or (broker_filter and broker not in row.get("available_at", []))
            or (ask_only and row.get("ask_eligible") is not True)
        ):
            continue
        listing = _listing(row)
        if listing is None:
            continue
        exchange, ticker, currency = listing
        key = (
            Decimal(str(row["ter_pct"])),
            0 if row.get("distribution") == "accumulating" else 1,
            -_SIZE_RANK.get(str(row.get("fund_size_band")), -1),
            _CURRENCY_RANK[currency],
            str(row.get("id", "")),
        )
        eligible.append((key, EtfPick(row, exchange, ticker, currency)))
    eligible.sort(key=lambda item: item[0])
    return [pick for _key, pick in eligible]


def pick_india_vehicle(
    *, has_indian_nri_share_account: bool, broker: str, rules: Mapping[str, Any] | None = None
) -> str:
    """Compare direct Indian shares with a listed India fund using the member's preference."""
    selected = rules or load_eu_etfs()
    preferences = selected.get("selection", {}).get("indian_vehicle", {})
    if has_indian_nri_share_account:
        return str(preferences.get("direct_if_nri_account", "in_equity_direct_core"))
    if ranked_etfs("india_equity", broker, rules=selected):
        return str(preferences.get("default", "ucits_india_etf"))
    return "in_equity_direct_core"


def approved_us_stock_buys(
    ideas: Sequence[Mapping[str, Any] | SatelliteIdea],
    *,
    investable_wealth_dkk: Decimal,
    available_dkk: Decimal,
    satellite_pct: Decimal,
    max_per_stock_pct: Decimal = Decimal("5"),
    minimum_buy_dkk: Decimal = Decimal("2000"),
) -> list[tuple[Mapping[str, Any] | SatelliteIdea, Decimal]]:
    """Size only approved US ideas, with code-owned total and per-stock caps."""
    if investable_wealth_dkk <= 0 or available_dkk <= 0:
        return []
    total_pct = min(max(Decimal(0), satellite_pct), Decimal("25"))
    total_cap = investable_wealth_dkk * total_pct / Decimal(100)
    per_stock_cap = investable_wealth_dkk * max_per_stock_pct / Decimal(100)
    remaining = min(available_dkk, total_cap)
    sorted_ideas = sorted(
        ideas,
        key=lambda idea: (
            -int(_idea_value(idea, "quant_decile") or 0),
            str(_idea_value(idea, "security_id") or ""),
        ),
    )
    output: list[tuple[Mapping[str, Any] | SatelliteIdea, Decimal]] = []
    for idea in sorted_ideas:
        if (
            _idea_value(idea, "market") != "US"
            or _idea_value(idea, "dossier_status") != "APPROVED"
            or not all(
                _idea_value(idea, field) for field in ("name", "isin", "exchange", "currency")
            )
        ):
            continue
        amount = min(remaining, per_stock_cap)
        amount = amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        if amount < minimum_buy_dkk:
            continue
        output.append((idea, amount))
        remaining -= amount
        if remaining < minimum_buy_dkk:
            break
    return output


def _idea_value(idea: Mapping[str, Any] | SatelliteIdea, key: str) -> Any:
    if isinstance(idea, Mapping):
        return idea.get(key)
    return getattr(idea, key, None)


def comparison_for_etf(
    pick: EtfPick, *, rules: Mapping[str, Any] | None = None
) -> dict[str, str] | None:
    """Compute a sourced annual TER and dividend-withholding comparison in percentage points."""
    selected = rules or load_eu_etfs()
    equivalent = pick.instrument.get("us_equivalent")
    comparison = selected.get("selection", {}).get("comparison", {})
    yields = comparison.get("dividend_yields_pct", {})
    us_fees = comparison.get("us_equivalent_ter_pct", {})
    yield_row = yields.get(pick.instrument.get("exposure"))
    fee_row = us_fees.get(equivalent)
    if not isinstance(yield_row, Mapping) or not isinstance(fee_row, Mapping):
        return None
    eu_fee = Decimal(str(pick.instrument["ter_pct"]))
    us_fee = Decimal(str(fee_row["value"]))
    yield_pct = Decimal(str(yield_row["value"]))
    withholding = Decimal(str(comparison["dividend_withholding_pct"]))
    annual_difference = eu_fee - us_fee + (yield_pct * withholding / Decimal(100))
    displayed_difference = abs(annual_difference).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "eu_ter_pct": format(eu_fee, "f"),
        "us_ter_pct": format(us_fee, "f"),
        "dividend_yield_pct": format(yield_pct, "f"),
        "annual_difference_pct": format(displayed_difference, ".2f"),
        "annual_difference_direction": "more" if annual_difference >= 0 else "less",
        "us_ter_source_url": str(fee_row["source_url"]),
        "dividend_yield_source_url": str(yield_row["source_url"]),
        "verified": "false",
    }
