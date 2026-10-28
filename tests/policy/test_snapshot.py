from collections.abc import Callable
from decimal import Decimal

from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import parse_asset
from wealth.schemas.common import LiquidityClass, Sleeve
from wealth.valuation.base import Valuation


def test_snapshot_aggregates_sleeves_liquidity_locked_assets_and_investable_total(
    make_valuation: Callable[[str], Valuation],
) -> None:
    savings = parse_asset({"asset_type": "savings", "bank": "Demo Bank", "balance": "10"})
    epf = parse_asset({"asset_type": "epf", "balance": "20"})
    jewellery = parse_asset(
        {
            "asset_type": "gold",
            "kind": "physical",
            "grams": "1",
            "purity": 22,
            "purpose": "jewellery",
        }
    )
    property_asset = parse_asset(
        {"asset_type": "property", "property_kind": "home", "estimated_value": "100"}
    )
    loan = parse_asset(
        {"asset_type": "loan", "loan_kind": "car", "outstanding": "7", "rate_pct": "8"}
    )
    snapshot = WealthSnapshot(
        [
            (savings, make_valuation("10")),
            (epf, make_valuation("20")),
            (jewellery, make_valuation("5")),
            (property_asset, make_valuation("100")),
            (loan, make_valuation("0")),
        ],
        [("equity_global", Decimal("20"), LiquidityClass.INSTANT)],
    )

    assert snapshot.totals_by_sleeve[Sleeve.CASH] == Decimal("10.00")
    assert snapshot.totals_by_liquidity[LiquidityClass.INSTANT] == Decimal("30.00")
    assert snapshot.locked_by_sleeve[Sleeve.DEBT] == Decimal("20.00")
    assert snapshot.investable_totals_by_sleeve == {
        "equity": Decimal("20.00"),
        "debt": Decimal("20.00"),
        "gold": Decimal("0.00"),
        "cash": Decimal("10.00"),
    }
    assert snapshot.investable_total == Decimal("50.00")
    assert snapshot.liabilities_total == Decimal("7.00")
    assert snapshot.net_worth == Decimal("148.00")


def test_snapshot_splits_mixed_sleeves_without_losing_paise(
    make_valuation: Callable[[str], Valuation],
) -> None:
    ulip = parse_asset(
        {
            "asset_type": "ulip",
            "insurer": "Demo Cover",
            "fund_value": "0",
            "annual_premium": "0",
            "equity_pct": "33.33",
        }
    )
    snapshot = WealthSnapshot([(ulip, make_valuation("10.01"))])

    assert snapshot.totals_by_sleeve[Sleeve.EQUITY_INDIA] == Decimal("3.34")
    assert snapshot.totals_by_sleeve[Sleeve.DEBT] == Decimal("6.67")
    assert snapshot.investable_total == Decimal("10.01")


def test_snapshot_rejects_float_market_values() -> None:
    try:
        WealthSnapshot(market_holdings=[("debt", 10.0, "instant")])  # type: ignore[list-item]
    except ValueError as error:
        assert "Decimal" in str(error)
    else:
        raise AssertionError("float market values must be rejected")
