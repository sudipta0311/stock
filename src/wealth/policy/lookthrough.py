"""Pure basket selection and Decimal look-through calculations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CompanyKey = str | tuple[str, str]
_RULES_DIR = Path(__file__).resolve().parents[3] / "rules"


@dataclass(frozen=True)
class BasketChoice:
    basket_id: str
    equity_share: Decimal = Decimal(1)
    proxy: bool = False


@dataclass(frozen=True)
class FundHolding:
    name: str
    basket_id: str
    value_inr: Decimal
    equity_share: Decimal = Decimal(1)
    proxy: bool = False
    market: str = "IN"
    symbol: str | None = None
    isin: str | None = None
    member_id: str | None = None


@dataclass(frozen=True)
class FundPart:
    name: str
    amount_inr: Decimal
    proxy: bool
    member_id: str | None = None


@dataclass(frozen=True)
class LookthroughResult:
    amounts: dict[CompanyKey, Decimal]
    parts: dict[CompanyKey, tuple[FundPart, ...]]
    amounts_by_member: dict[tuple[str, CompanyKey], Decimal]
    company_keys_by_symbol: dict[tuple[str, str], tuple[CompanyKey, ...]]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _normal(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _keywords_match(name: str, rule: Mapping[str, Any]) -> bool:
    lower = _normal(name)
    include = rule.get("any", rule.get("include_any", []))
    exclude = rule.get("exclude_any", [])
    return (
        isinstance(include, list)
        and any(_normal(keyword) in lower for keyword in include)
        and not any(_normal(keyword) in lower for keyword in exclude)
    )


def _choice(value: Mapping[str, Any], *, proxy_default: bool = False) -> BasketChoice | None:
    basket_id = value.get("basket")
    if not isinstance(basket_id, str) or not basket_id:
        return None
    equity_share = Decimal(str(value.get("equity_share", "1")))
    if not Decimal(0) <= equity_share <= Decimal(1):
        return None
    return BasketChoice(basket_id, equity_share, bool(value.get("proxy", proxy_default)))


@lru_cache(maxsize=1)
def load_lookthrough_rules() -> dict[str, Any]:
    """Load versioned rules and the two existing catalogues used to resolve fund names."""
    with (_RULES_DIR / "fund_lookthrough.yaml").open(encoding="utf-8") as stream:
        rules = yaml.safe_load(stream)
    if not isinstance(rules, dict):
        raise ValueError("fund look-through rules must be a mapping")
    with (_RULES_DIR / "etf_categories.yaml").open(encoding="utf-8") as stream:
        etf_categories = yaml.safe_load(stream)
    with (_RULES_DIR / "eu_etfs.yaml").open(encoding="utf-8") as stream:
        eu_etfs = yaml.safe_load(stream)
    curated: dict[str, str] = {}
    by_index = {_normal(key): value for key, value in rules.get("by_index_name", {}).items()}
    for item in eu_etfs.get("instruments", []):
        if not isinstance(item, Mapping):
            continue
        isin = str(item.get("isin", "")).strip().upper()
        index = _normal(item.get("index"))
        basket = by_index.get(index)
        if isin and isinstance(basket, str):
            curated[isin] = basket
    rules["_etf_categories"] = etf_categories
    rules["_curated_by_isin"] = curated
    return rules


def basket_for_holding(holding: Any, rules: Mapping[str, Any]) -> BasketChoice | None:
    """Choose a public basket using direct ticker, curated fund, and category rules."""
    market = str(_value(holding, "market", "")).upper()
    security_type = str(_value(holding, "security_type", "")).casefold()
    sleeve = str(_value(holding, "sleeve", "") or "").casefold()
    name = str(_value(holding, "name", "") or "")
    sub_category = str(_value(holding, "sub_category", "") or "")
    symbol = str(_value(holding, "symbol", "") or "").strip().upper()
    isin = str(_value(holding, "isin", "") or "").strip().upper()
    if security_type not in {"etf", "mf"}:
        return None
    if sleeve in {"debt", "gold"}:
        return None
    no_lookthrough = _mapping(rules.get("not_looked_through"))
    category_key = _normal(sub_category)
    excluded_categories = {_normal(key) for key in no_lookthrough}
    category_words = set(category_key.replace("/", " ").replace("-", " ").split())
    if category_words & excluded_categories:
        return None
    lower_name = _normal(name)
    if any(
        term in lower_name
        for term in (
            "arbitrage",
            "liquid fund",
            "liquid etf",
            "gold fund",
            "debt fund",
            "bond fund",
        )
    ):
        return None

    if market == "US" and security_type == "etf":
        choice = _choice(_mapping(_mapping(rules.get("by_us_symbol")).get(symbol)))
        if choice is not None:
            return choice

    curated = _mapping(rules.get("_curated_by_isin"))
    curated_basket = curated.get(isin)
    if curated_basket:
        return BasketChoice(str(curated_basket), proxy=True)

    if market == "IN" and security_type in {"mf", "etf"}:
        etf_categories = _mapping(rules.get("_etf_categories"))
        for category in etf_categories.get("categories", []):
            if not isinstance(category, Mapping):
                continue
            category_market = str(category.get("market", "")).upper()
            security_types = category.get("security_types", [])
            if category_market != "IN" or security_type not in security_types:
                continue
            if _keywords_match(
                name,
                {
                    "any": category.get("include_any", []),
                    "exclude_any": category.get("exclude_any", []),
                },
            ):
                index_name = category.get("index_name")
                basket = _mapping(rules.get("by_index_name")).get(index_name)
                if basket:
                    return BasketChoice(str(basket), proxy=True)

    by_sub_category = _mapping(rules.get("by_sub_category"))
    sub_rule = next(
        (value for key, value in by_sub_category.items() if _normal(key) == category_key),
        None,
    )
    if isinstance(sub_rule, Mapping):
        for keyword_rule in sub_rule.get("name_keywords", []):
            if isinstance(keyword_rule, Mapping) and _keywords_match(name, keyword_rule):
                selected = _choice(
                    keyword_rule,
                    proxy_default=bool(sub_rule.get("proxy", False)),
                )
                if selected is not None:
                    return selected
        selected = _choice(sub_rule)
        if selected is not None:
            return selected

    for keyword_rule in rules.get("by_name_keywords", []):
        if isinstance(keyword_rule, Mapping) and _keywords_match(name, keyword_rule):
            selected = _choice(keyword_rule)
            if selected is not None:
                return selected
    return None


def _same_company_aliases(rules: Mapping[str, Any] | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    configured = (rules or {}).get("same_company", [])
    for group in configured:
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)) or not group:
            continue
        canonical = str(group[0]).strip().upper()
        for symbol in group:
            aliases[str(symbol).strip().upper()] = canonical
    return aliases


def company_key(
    market: str | None,
    isin: str | None,
    symbol: str | None,
    *,
    rules: Mapping[str, Any] | None = None,
) -> CompanyKey | None:
    """Return an ISIN key or a normalized market-and-symbol key."""
    normalized_market = str(market or "").strip().upper()
    normalized_symbol = str(symbol or "").strip().upper()
    aliases = _same_company_aliases(rules)
    if normalized_market and normalized_symbol in aliases:
        return (normalized_market, aliases[normalized_symbol])
    normalized_isin = str(isin or "").strip().upper()
    if normalized_isin:
        return normalized_isin
    if normalized_market and normalized_symbol:
        return (normalized_market, normalized_symbol)
    return None


def resolve_company_keys(
    market: str,
    isin: str | None,
    symbol: str | None,
    company_keys_by_symbol: Mapping[tuple[str, str], Sequence[CompanyKey]],
    *,
    rules: Mapping[str, Any] | None = None,
) -> tuple[CompanyKey, ...]:
    """Resolve a security to its ISIN and symbol keys using fresh basket rows."""
    normalized_market = market.strip().upper()
    normalized_isin = str(isin or "").strip().upper()
    normalized_symbol = str(symbol or "").strip().upper()
    keys: list[CompanyKey] = []
    if normalized_isin:
        keys.append(normalized_isin)
    if normalized_symbol:
        aliases = _same_company_aliases(rules)
        canonical = aliases.get(normalized_symbol)
        # Every share class of one company (GOOGL and GOOG) brings its own ISIN rows.
        group = (
            sorted(symbol for symbol, target in aliases.items() if target == canonical)
            if canonical is not None
            else [normalized_symbol]
        )
        for member_symbol in group:
            keys.extend(company_keys_by_symbol.get((normalized_market, member_symbol), ()))
        symbol_key = company_key(normalized_market, None, normalized_symbol, rules=rules)
        if symbol_key is not None:
            keys.append(symbol_key)
    elif normalized_isin:
        keys.append(normalized_isin)
    return tuple(dict.fromkeys(keys))


def same_company(
    left_market: str,
    left_isin: str | None,
    left_symbol: str | None,
    right_market: str,
    right_isin: str | None,
    right_symbol: str | None,
    company_keys_by_symbol: Mapping[tuple[str, str], Sequence[CompanyKey]],
    *,
    rules: Mapping[str, Any] | None = None,
) -> bool:
    """Compare two security references after resolving available basket keys."""
    left_keys = set(
        resolve_company_keys(
            left_market,
            left_isin,
            left_symbol,
            company_keys_by_symbol,
            rules=rules,
        )
    )
    right_keys = set(
        resolve_company_keys(
            right_market,
            right_isin,
            right_symbol,
            company_keys_by_symbol,
            rules=rules,
        )
    )
    return bool(left_keys.intersection(right_keys))


def _weights_list(weights: Sequence[Any] | Mapping[str, Sequence[Any]]) -> list[Any]:
    if isinstance(weights, Mapping):
        return [item for basket_rows in weights.values() for item in basket_rows]
    return list(weights)


def through_fund_result(
    fund_holdings: Sequence[FundHolding],
    weights: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    rules: Mapping[str, Any] | None = None,
) -> LookthroughResult:
    """Calculate unrounded company amounts and sorted fund-level explanations."""
    weights_by_basket: dict[str, list[Any]] = defaultdict(list)
    for row in _weights_list(weights):
        basket_id = str(_value(row, "basket_id", ""))
        weight = _value(row, "weight_pct")
        if basket_id and weight is not None:
            weights_by_basket[basket_id].append(row)

    amounts: dict[CompanyKey, Decimal] = defaultdict(Decimal)
    fund_parts: dict[CompanyKey, dict[tuple[str, bool], Decimal]] = defaultdict(
        lambda: defaultdict(Decimal)
    )
    member_amounts: dict[tuple[str, CompanyKey], Decimal] = defaultdict(Decimal)
    company_keys_by_symbol: dict[tuple[str, str], set[CompanyKey]] = defaultdict(set)
    for row in _weights_list(weights):
        market = str(_value(row, "market", "") or "").strip().upper()
        symbol = str(_value(row, "symbol", "") or "").strip().upper()
        key = company_key(
            market,
            _value(row, "isin"),
            symbol,
            rules=rules,
        )
        if market and symbol and key is not None:
            company_keys_by_symbol[(market, symbol)].add(key)
            # An alias key hides the row's ISIN, which ISIN-only fund rows still use.
            row_isin = str(_value(row, "isin", "") or "").strip().upper()
            if row_isin:
                company_keys_by_symbol[(market, symbol)].add(row_isin)
    for fund in fund_holdings:
        if fund.value_inr <= 0 or fund.equity_share <= 0:
            continue
        for weight in weights_by_basket.get(fund.basket_id, []):
            weight_pct = Decimal(str(_value(weight, "weight_pct")))
            if weight_pct <= 0:
                continue
            key = company_key(
                _value(weight, "market"),
                _value(weight, "isin"),
                _value(weight, "symbol"),
                rules=rules,
            )
            if key is None:
                continue
            amount = fund.value_inr * fund.equity_share * weight_pct / Decimal(100)
            amounts[key] += amount
            fund_parts[key][(fund.name, fund.proxy)] += amount
            if fund.member_id is not None:
                member_amounts[(fund.member_id, key)] += amount
    parts = {
        key: tuple(
            FundPart(name, amount, proxy)
            for (name, proxy), amount in sorted(
                records.items(), key=lambda pair: (-pair[1], pair[0][0])
            )
        )
        for key, records in fund_parts.items()
    }
    return LookthroughResult(
        dict(amounts),
        parts,
        dict(member_amounts),
        {key: tuple(sorted(values, key=str)) for key, values in company_keys_by_symbol.items()},
    )


def through_fund_amounts(
    fund_holdings: Sequence[FundHolding],
    weights: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    rules: Mapping[str, Any] | None = None,
) -> dict[CompanyKey, Decimal]:
    """Return summed company amounts without rounding the underlying calculation."""
    return through_fund_result(fund_holdings, weights, rules=rules).amounts


def matching_company_keys(
    result: LookthroughResult,
    *,
    market: str,
    isin: str | None,
    symbol: str | None,
    rules: Mapping[str, Any] | None = None,
) -> tuple[Decimal, tuple[FundPart, ...]]:
    """Find all basket rows for one security, preferring exact ISIN matches."""
    matches = [
        key
        for key in resolve_company_keys(
            market,
            isin,
            symbol,
            result.company_keys_by_symbol,
            rules=rules,
        )
        if key in result.amounts
    ]
    total = sum((result.amounts[key] for key in matches), Decimal(0))
    combined: dict[tuple[str, bool], Decimal] = defaultdict(Decimal)
    for key in matches:
        for part in result.parts.get(key, ()):
            combined[(part.name, part.proxy)] += part.amount_inr
    parts = tuple(
        FundPart(name, amount, proxy)
        for (name, proxy), amount in sorted(
            combined.items(), key=lambda pair: (-pair[1], pair[0][0])
        )
    )
    return total, parts


def remaining_stock_room(
    cap: Decimal,
    direct_value: Decimal,
    through_funds_value: Decimal,
    *other_limits: Decimal,
) -> Decimal:
    """Return deterministic room beneath all supplied stock limits."""
    room = cap - direct_value - through_funds_value
    for limit in other_limits:
        room = min(room, limit)
    return max(Decimal(0), room)
