"""Statement lots of one security show as a single holding."""

from __future__ import annotations

from typing import Any

from wealth.api.money_map import _merge_lots


def _lot(quantity: str, value: str | None, status: str = "statement_value_used") -> dict[str, Any]:
    return {
        "security_id": "00000000-0000-0000-0000-000000000001",
        "symbol": "SYNTHFUND",
        "name": "Synthetic Equity Fund",
        "security_type": "mf",
        "quantity": quantity,
        "value": value,
        "value_usd": None,
        "currency": "INR",
        "price_as_of": None,
        "fx_as_of": None,
        "fx_missing": False,
        "price_source": "statement",
        "status": status,
    }


def test_lots_of_one_fund_add_up_to_one_holding() -> None:
    merged = _merge_lots(
        [_lot("75.000000", "8250.00"), _lot("5.000000", "550.00"), _lot("2.000000", "220.00")]
    )

    assert len(merged) == 1
    assert merged[0]["quantity"] == "82.000000"
    assert merged[0]["value"] == "9020.00"


def test_an_unpriced_lot_keeps_the_known_value_and_flags_the_gap() -> None:
    merged = _merge_lots(
        [_lot("75.000000", "8250.00", "priced"), _lot("5.000000", None, "price_unavailable")]
    )

    assert merged[0]["quantity"] == "80.000000"
    assert merged[0]["value"] == "8250.00"
    assert merged[0]["status"] == "price_unavailable"


def test_a_holding_with_no_priced_lot_has_no_value() -> None:
    merged = _merge_lots([_lot("5.000000", None, "price_unavailable")] * 2)

    assert merged[0]["value"] is None


def test_different_securities_stay_separate() -> None:
    other = {**_lot("1.000000", "100.00"), "security_id": "00000000-0000-0000-0000-000000000002"}

    assert len(_merge_lots([_lot("1.000000", "100.00"), other])) == 2
