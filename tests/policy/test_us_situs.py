from decimal import Decimal

import pytest

from wealth.policy.us_situs import us_situs_value_usd


def test_us_situs_value_counts_us_domiciled_shares_and_excludes_ucits() -> None:
    holdings = [
        {
            "symbol": "VOO",
            "domicile": "US",
            "currency": "USD",
            "quantity": Decimal("100"),
            "price": Decimal("500"),
        },
        {
            "symbol": "CSPX",
            "domicile": "IE",
            "currency": "USD",
            "quantity": Decimal("50"),
            "price": Decimal("600"),
        },
    ]

    assert us_situs_value_usd(holdings) == Decimal("50000")


def test_us_situs_accepts_statement_values_and_rejects_float_money() -> None:
    assert us_situs_value_usd(
        [{"domicile": "US", "currency": "USD", "statement_value": "1250.50"}]
    ) == Decimal("1250.50")

    with pytest.raises(ValueError, match="decimal value"):
        us_situs_value_usd([{"domicile": "US", "currency": "USD", "value_usd": 1.25}])
