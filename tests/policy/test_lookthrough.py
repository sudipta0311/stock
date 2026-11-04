"""Pure selection, alias, and money math tests for fund look-through."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import yaml

from wealth.policy.lookthrough import (
    FundHolding,
    basket_for_holding,
    company_key,
    load_lookthrough_rules,
    matching_company_keys,
    same_company,
    through_fund_result,
)

_RULES_DIR = Path(__file__).resolve().parents[2] / "rules"


def _holding(
    *,
    name: str = "Synthetic fund",
    market: str = "IN",
    security_type: str = "mf",
    sleeve: str = "equity_india",
    sub_category: str | None = None,
    symbol: str | None = None,
    isin: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        market=market,
        security_type=security_type,
        sleeve=sleeve,
        sub_category=sub_category,
        symbol=symbol,
        isin=isin,
    )


def test_every_catalogued_index_name_maps_to_a_basket_or_exclusion() -> None:
    rules = load_lookthrough_rules()
    mapped = {" ".join(key.casefold().split()) for key in rules["by_index_name"]}
    excluded = {" ".join(key.casefold().split()) for key in rules["not_looked_through"]}
    with (_RULES_DIR / "etf_categories.yaml").open(encoding="utf-8") as stream:
        categories = yaml.safe_load(stream)["categories"]
    with (_RULES_DIR / "eu_etfs.yaml").open(encoding="utf-8") as stream:
        curated = yaml.safe_load(stream)["instruments"]

    for category in categories:
        index_name = category.get("index_name")
        if index_name:
            normalized = " ".join(index_name.casefold().split())
            assert normalized in mapped or normalized in excluded
    for instrument in curated:
        index_name = " ".join(instrument["index"].casefold().split())
        assert index_name in mapped or index_name in excluded


def test_each_sub_category_rule_has_a_choice() -> None:
    rules = load_lookthrough_rules()
    examples = {
        "Large Cap": ("Synthetic active fund", "nifty100", Decimal(1)),
        "Large & Mid Cap": ("Synthetic active fund", "niftylargemidcap250", Decimal(1)),
        "Mid Cap": ("Synthetic active fund", "niftymidcap150", Decimal(1)),
        "Small Cap": ("Synthetic active fund", "niftysmallcap250", Decimal(1)),
        "Flexi Cap": ("Synthetic active fund", "nifty500", Decimal(1)),
        "Multi Cap": ("Synthetic active fund", "nifty500", Decimal(1)),
        "ELSS": ("Synthetic active fund", "nifty500", Decimal(1)),
        "Other Equity": ("Synthetic active fund", "nifty500", Decimal(1)),
        "Aggressive Hybrid": ("Synthetic hybrid fund", "nifty500", Decimal("0.75")),
        "Balanced Hybrid": ("Synthetic hybrid fund", "nifty500", Decimal("0.5")),
        "Conservative Hybrid": ("Synthetic hybrid fund", "nifty500", Decimal("0.2")),
        "Sectoral/Thematic": ("Synthetic bank fund", "niftybank", Decimal(1)),
        "Index Funds/ETFs": ("Nifty Next 50 index fund", "niftynext50", Decimal(1)),
        "International / FoF Overseas": ("S&P 500 index fund", "sp500", Decimal(1)),
    }

    for sub_category, (name, basket_id, equity_share) in examples.items():
        choice = basket_for_holding(
            _holding(name=name, sub_category=sub_category),
            rules,
        )
        assert choice is not None
        assert (choice.basket_id, choice.equity_share, choice.proxy) == (
            basket_id,
            equity_share,
            True,
        )


def test_nifty_keywords_do_not_confuse_nifty_50_with_next_50() -> None:
    rules = load_lookthrough_rules()
    nifty_50 = basket_for_holding(
        _holding(name="Nifty 50 Index Fund", sub_category="Index Funds/ETFs"), rules
    )
    next_50 = basket_for_holding(
        _holding(name="Nifty Next 50 Index Fund", sub_category="Index Funds/ETFs"),
        rules,
    )

    assert nifty_50 is not None and nifty_50.basket_id == "nifty50"
    assert next_50 is not None and next_50.basket_id == "niftynext50"


def test_curated_funds_and_danish_name_fallback_select_their_baskets() -> None:
    rules = load_lookthrough_rules()
    curated_isin = next(iter(rules["_curated_by_isin"]))
    curated = basket_for_holding(
        _holding(market="EU", security_type="etf", isin=curated_isin), rules
    )
    danish_name = basket_for_holding(
        _holding(
            name="iShares Core S&P 500 UCITS",
            market="EU",
            security_type="etf",
            sub_category=None,
        ),
        rules,
    )

    assert curated is not None
    assert curated.proxy is True
    assert danish_name is not None and danish_name.basket_id == "sp500"
    assert danish_name.proxy is True


def test_danish_global_fund_names_select_proxy_baskets_in_specific_order() -> None:
    rules = load_lookthrough_rules()
    examples = {
        "Vanguard FTSE All-World UCITS ETF (USD) Accumulating": "msci_acwi",
        "Synthetic All-World fund": "msci_acwi",
        "Sparindex INDEX Globale Aktier KL": "msci_world",
        "Synthetic Total World fund": "msci_acwi_imi",
        "Synthetic FTSE Emerging fund": "msci_em",
    }

    for name, basket_id in examples.items():
        choice = basket_for_holding(
            _holding(
                name=name,
                market="EU",
                security_type="etf",
                sleeve="equity_global",
            ),
            rules,
        )
        assert choice is not None
        assert choice.basket_id == basket_id
        assert choice.proxy is True


def test_debt_gold_arbitrage_and_liquid_funds_are_not_looked_through() -> None:
    rules = load_lookthrough_rules()
    excluded = [
        _holding(name="Synthetic debt fund", sleeve="debt", sub_category="Debt"),
        _holding(name="Synthetic gilt fund", sub_category="Debt - Gilt"),
        _holding(name="Synthetic gold ETF", sleeve="gold", security_type="etf"),
        _holding(name="Synthetic arbitrage fund", sub_category="Arbitrage"),
        _holding(name="Synthetic liquid fund", sub_category="Liquid"),
    ]

    assert all(basket_for_holding(holding, rules) is None for holding in excluded)


def test_name_fallback_does_not_confuse_nifty_50_with_nifty_500() -> None:
    rules = load_lookthrough_rules()
    nifty_500 = basket_for_holding(_holding(name="Nifty 500 index fund"), rules)

    assert nifty_500 is not None and nifty_500.basket_id == "nifty500"


def test_hybrid_and_broad_funds_sum_without_rounding_and_keep_member_parts() -> None:
    rules = load_lookthrough_rules()
    company_isin = "IN0000000010"
    funds = [
        FundHolding("Nifty 50 index fund", "nifty50", Decimal("2000000"), member_id="owner"),
        FundHolding(
            "Aggressive hybrid fund",
            "nifty500",
            Decimal("400000"),
            equity_share=Decimal("0.75"),
            proxy=True,
            member_id="partner",
        ),
    ]
    weights = [
        {
            "basket_id": "nifty50",
            "market": "IN",
            "isin": company_isin,
            "symbol": "ALPHA",
            "name": "Synthetic Alpha",
            "weight_pct": Decimal("12.5"),
        },
        {
            "basket_id": "nifty500",
            "market": "IN",
            "isin": company_isin,
            "symbol": "ALPHA",
            "name": "Synthetic Alpha",
            "weight_pct": Decimal("7"),
        },
    ]

    result = through_fund_result(funds, weights, rules=rules)
    key = company_key("IN", company_isin, "ALPHA", rules=rules)

    assert key is not None
    assert result.amounts[key] == Decimal("271000.000")
    assert result.amounts_by_member[("owner", key)] == Decimal("250000.00")
    assert result.amounts_by_member[("partner", key)] == Decimal("21000.000")
    assert [part.name for part in result.parts[key]] == [
        "Nifty 50 index fund",
        "Aggressive hybrid fund",
    ]
    assert result.parts[key][1].proxy is True
    matched, parts = matching_company_keys(
        result,
        market="IN",
        isin=company_isin,
        symbol="ALPHA",
        rules=rules,
    )
    assert matched == Decimal("271000.000")
    assert parts == result.parts[key]


def test_same_company_share_classes_add_their_basket_weights() -> None:
    rules = load_lookthrough_rules()
    fund = FundHolding("Nasdaq index fund", "nasdaq100", Decimal("100000"))
    weights = [
        {
            "basket_id": "nasdaq100",
            "market": "US",
            "isin": None,
            "symbol": "GOOGL",
            "name": "Alphabet Class A",
            "weight_pct": Decimal("12"),
        },
        {
            "basket_id": "nasdaq100",
            "market": "US",
            "isin": None,
            "symbol": "GOOG",
            "name": "Alphabet Class C",
            "weight_pct": Decimal("3"),
        },
    ]

    result = through_fund_result([fund], weights, rules=rules)
    matched, _parts = matching_company_keys(
        result,
        market="US",
        isin=None,
        symbol="GOOG",
        rules=rules,
    )

    assert matched == Decimal("15000")


def test_share_classes_also_match_isin_only_rows_from_global_funds() -> None:
    rules = load_lookthrough_rules()
    class_a_isin = "US02079K3059"
    class_c_isin = "US02079K1079"
    weights = [
        {
            "basket_id": "msci_acwi",
            "market": "US",
            "isin": class_a_isin,
            "symbol": None,
            "name": "Synthetic class A",
            "weight_pct": Decimal("2"),
        },
        {
            "basket_id": "msci_acwi",
            "market": "US",
            "isin": class_c_isin,
            "symbol": None,
            "name": "Synthetic class C",
            "weight_pct": Decimal("1"),
        },
        {
            "basket_id": "sp500",
            "market": "US",
            "isin": class_a_isin,
            "symbol": "GOOGL",
            "name": "Synthetic class A",
            "weight_pct": Decimal("4"),
        },
        {
            "basket_id": "sp500",
            "market": "US",
            "isin": class_c_isin,
            "symbol": "GOOG",
            "name": "Synthetic class C",
            "weight_pct": Decimal("3"),
        },
    ]
    funds = [
        FundHolding("Synthetic ACWI fund", "msci_acwi", Decimal("100000")),
        FundHolding("Synthetic S&P fund", "sp500", Decimal("100000")),
    ]
    result = through_fund_result(funds, weights, rules=rules)

    for symbol, isin in (("GOOGL", None), ("GOOG", None), ("GOOGL", class_a_isin)):
        amount, parts = matching_company_keys(
            result, market="US", isin=isin, symbol=symbol, rules=rules
        )
        assert amount == Decimal("10000")
        assert {part.name: part.amount_inr for part in parts} == {
            "Synthetic S&P fund": Decimal("7000"),
            "Synthetic ACWI fund": Decimal("3000"),
        }

    assert same_company(
        "US",
        class_c_isin,
        None,
        "US",
        None,
        "GOOGL",
        result.company_keys_by_symbol,
        rules=rules,
    )


def test_isin_and_symbol_rows_are_counted_once_under_their_available_keys() -> None:
    result = through_fund_result(
        [FundHolding("Synthetic fund", "sp500", Decimal("100000"))],
        [
            {
                "basket_id": "sp500",
                "market": "US",
                "isin": "US0000000010",
                "symbol": "SYNTH",
                "name": "Synthetic exact company row",
                "weight_pct": Decimal("10"),
            },
            {
                "basket_id": "sp500",
                "market": "US",
                "isin": None,
                "symbol": "SYNTH",
                "name": "Synthetic fallback company row",
                "weight_pct": Decimal("20"),
            },
        ],
    )

    amount, parts = matching_company_keys(
        result,
        market="US",
        isin="US0000000010",
        symbol="SYNTH",
        rules=load_lookthrough_rules(),
    )

    assert amount == Decimal("30000")
    assert len(parts) == 1
    assert parts[0].name == "Synthetic fund"
    assert parts[0].amount_inr == Decimal("30000")


def test_acwi_isin_only_and_sp500_dual_key_rows_both_match_symbol_only_idea() -> None:
    rules = load_lookthrough_rules()
    apple_isin = "US0378331005"
    weights = [
        {
            "basket_id": "msci_acwi",
            "market": "US",
            "isin": apple_isin,
            "symbol": None,
            "name": "Synthetic Global Company",
            "weight_pct": Decimal("8"),
        },
        {
            "basket_id": "sp500",
            "market": "US",
            "isin": apple_isin,
            "symbol": "AAPL",
            "name": "Synthetic S&P Company",
            "weight_pct": Decimal("7"),
        },
    ]
    funds = [
        FundHolding("Synthetic ACWI fund", "msci_acwi", Decimal("100000")),
        FundHolding("Synthetic S&P fund", "sp500", Decimal("100000")),
    ]
    result = through_fund_result(funds, weights, rules=rules)
    amount, _parts = matching_company_keys(
        result,
        market="US",
        isin=None,
        symbol="AAPL",
        rules=rules,
    )

    assert amount == Decimal("15000")
    assert len(result.amounts) == 1

    acwi_only = through_fund_result(funds[:1], weights, rules=rules)
    only_global_amount, _parts = matching_company_keys(
        acwi_only,
        market="US",
        isin=None,
        symbol="AAPL",
        rules=rules,
    )
    assert only_global_amount == Decimal("8000")


def test_isin_only_direct_holding_matches_symbol_only_idea_via_basket_row() -> None:
    result = through_fund_result(
        [],
        [
            {
                "basket_id": "sp500",
                "market": "US",
                "isin": "US0378331005",
                "symbol": "AAPL",
                "name": "Synthetic reference row",
                "weight_pct": Decimal("7"),
            }
        ],
        rules=load_lookthrough_rules(),
    )

    assert same_company(
        "US",
        "US0378331005",
        None,
        "US",
        None,
        "AAPL",
        result.company_keys_by_symbol,
        rules=load_lookthrough_rules(),
    )


def test_symbol_only_rows_still_match_when_isin_cannot_be_resolved() -> None:
    result = through_fund_result(
        [FundHolding("Synthetic S&P fund", "sp500", Decimal("100000"))],
        [
            {
                "basket_id": "sp500",
                "market": "US",
                "isin": None,
                "symbol": "AAPL",
                "name": "Synthetic unresolved company",
                "weight_pct": Decimal("3"),
            }
        ],
        rules=load_lookthrough_rules(),
    )

    amount, _parts = matching_company_keys(
        result,
        market="US",
        isin=None,
        symbol="AAPL",
        rules=load_lookthrough_rules(),
    )
    assert amount == Decimal("3000")
