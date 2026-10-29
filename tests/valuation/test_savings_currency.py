"""Foreign-currency savings never reach a rupee valuation unconverted."""

from datetime import date
from decimal import Decimal

import pytest

from wealth.schemas.assets import SavingsAsset
from wealth.valuation import value_asset


def test_a_kroner_balance_is_refused_until_it_is_converted_to_rupees() -> None:
    asset = SavingsAsset(
        asset_type="savings", bank="Synthetic Bank", balance="85000", currency="DKK"
    )

    with pytest.raises(ValueError, match="DKK"):
        value_asset(asset, date(2026, 10, 1))

    converted = asset.model_copy(
        update={"balance": asset.balance * Decimal("12.50"), "currency": "INR"}
    )
    assert value_asset(converted, date(2026, 10, 1)).value_today == Decimal("1062500.00")
