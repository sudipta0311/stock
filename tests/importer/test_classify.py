"""CAS holding classifications are deterministic and cover common scheme names."""

from __future__ import annotations

import pytest

from wealth.importer.classify import classify_holding
from wealth.importer.models import ParsedHolding


@pytest.mark.parametrize(
    ("kind", "isin", "name", "expected"),
    [
        ("sgb", "IN0000000001", "Synthetic Sovereign Gold Bond", ("gold", "sgb")),
        ("bond", "IN0000000002", "Synthetic Taxable Bond", ("debt", "bond")),
        ("equity", "INE000A01012", "Synthetic Tools", ("equity_india", "equity:india")),
        ("equity", "US0000000001", "Synthetic Global Co", ("equity_global", "equity:global")),
        ("mf", "INF000000001", "Synthetic Liquid Fund", ("debt", "mf:debt")),
        ("mf", "INF000000002", "Synthetic Overnight Fund", ("debt", "mf:debt")),
        ("mf", "INF000000003", "Synthetic Money Market Fund", ("debt", "mf:debt")),
        ("mf", "INF000000004", "Synthetic Gilt Fund", ("debt", "mf:debt")),
        ("mf", "INF000000005", "Synthetic Debt Fund", ("debt", "mf:debt")),
        ("mf", "INF000000006", "Synthetic Corporate Bond Fund", ("debt", "mf:debt")),
        ("mf", "INF000000007", "Synthetic Banking & PSU Fund", ("debt", "mf:debt")),
        ("mf", "INF000000008", "Synthetic Short Duration Fund", ("debt", "mf:debt")),
        ("mf", "INF000000009", "Synthetic Bharat Bond Index", ("debt", "mf:debt")),
        ("mf", "INF000000010", "Synthetic Gold Fund", ("gold", "mf:gold")),
        ("mf", "INF000000011", "Synthetic Silver ETF Fund", ("gold", "mf:gold")),
        ("mf", "INF000000012", "Synthetic Nasdaq 100 Fund", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000013", "Synthetic S&P 500 Fund", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000014", "Synthetic US Equity Fund", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000015", "Synthetic Global Fund", ("equity_global", "mf:equity_global")),
        (
            "mf",
            "INF000000016",
            "Synthetic International Fund",
            ("equity_global", "mf:equity_global"),
        ),
        ("mf", "INF000000017", "Synthetic Overseas Fund", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000018", "Synthetic Hang Seng Fund", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000019", "Synthetic NYSE Fund", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000020", "Synthetic FANG Fund", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000021", "Synthetic Overseas FoF", ("equity_global", "mf:equity_global")),
        ("mf", "INF000000022", "Synthetic Arbitrage Fund", ("debt", "mf:arbitrage")),
        ("mf", "INF000000023", "Synthetic Nifty Index Fund", ("equity_india", "mf:equity_india")),
        ("etf", "INF000000024", "Synthetic Nifty ETF", ("equity_india", "etf:equity_india")),
        ("etf", "INF000000025", "Synthetic Gold ETF", ("gold", "etf:gold")),
    ],
)
def test_classify_holding_maps_asset_and_name_to_sleeve(
    kind: str, isin: str, name: str, expected: tuple[str, str]
) -> None:
    holding = ParsedHolding(kind=kind, isin=isin, name=name, quantity="1")

    assert classify_holding(holding) == expected


def test_classify_holding_returns_none_for_unknown_kind_or_invalid_isin() -> None:
    unknown = ParsedHolding(
        kind="other", isin="IN0000000026", name="Synthetic Unknown", quantity="1"
    )
    invalid_isin = ParsedHolding(kind="equity", isin="bad", name="Synthetic Equity", quantity="1")

    assert classify_holding(unknown) is None
    assert classify_holding(invalid_isin) is None


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("mf:debt:liquid", ("debt", "mf:debt:liquid")),
        ("mf:gold", ("gold", "mf:gold")),
        ("mf:equity_global", ("equity_global", "mf:equity_global")),
        ("mf:equity_india", ("equity_india", "mf:equity_india")),
        ("mf:arbitrage", ("debt", "mf:arbitrage:tax_note")),
    ],
)
def test_classify_holding_prefers_amfi_master_category(
    category: str, expected: tuple[str, str]
) -> None:
    holding = ParsedHolding(
        kind="mf", isin="INF000000001", name="Synthetic Ambiguous Fund", quantity="1"
    )

    assert classify_holding(holding, amfi_category=category) == expected
