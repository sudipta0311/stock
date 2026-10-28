from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal

from wealth.policy.allocation import AllocationPlan
from wealth.policy.debt_answer import debt_answer
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import parse_asset
from wealth.valuation.base import Valuation


def test_debt_answer_counts_epf_ppf_and_fd_and_orders_underweight_equity_first(
    make_valuation: Callable[[str], Valuation],
) -> None:
    epf = parse_asset({"asset_type": "epf", "balance": "100000"})
    ppf = parse_asset({"asset_type": "ppf", "balance": "200000", "opened_fy_start_year": 2015})
    fd = parse_asset(
        {
            "asset_type": "fd",
            "bank": "Demo Bank",
            "principal": "110000",
            "rate_pct": "6.5",
            "maturity": "2030-01",
        }
    )
    snapshot = WealthSnapshot(
        [
            (epf, make_valuation("100000")),
            (ppf, make_valuation("200000")),
            (fd, make_valuation("110000")),
        ],
        [("equity_india", Decimal("590000"), "instant")],
    )
    target = {
        "equity": Decimal("670000"),
        "debt": Decimal("330000"),
        "gold": Decimal(0),
        "cash": Decimal(0),
    }
    current = {
        "equity": Decimal("590000"),
        "debt": Decimal("410000"),
        "gold": Decimal(0),
        "cash": Decimal(0),
    }
    plan = AllocationPlan(
        target_rupees=target,
        target_percent={
            "equity": Decimal("67"),
            "debt": Decimal("33"),
            "gold": Decimal(0),
            "cash": Decimal(0),
        },
        current_rupees=current,
        current_percent={
            "equity": Decimal("59"),
            "debt": Decimal("41"),
            "gold": Decimal(0),
            "cash": Decimal(0),
        },
        gap={sleeve: target[sleeve] - current[sleeve] for sleeve in target},
        buckets=(),
        investable_total=Decimal("1000000"),
    )

    answer = debt_answer(plan, snapshot)

    assert answer.target_debt == Decimal("330000")
    assert answer.current_debt == Decimal("410000")
    assert answer.locked_debt == Decimal("410000.00")
    assert answer.gap == Decimal("-80000")
    assert answer.status == "enough"
    assert answer.new_money_order == ("equity",)
    assert answer.message_key == "debt_target_met"
    assert answer.params == {"current_pct": "41.0", "target_pct": "33.0"}


def test_new_money_order_omits_at_target_and_overweight_sleeves() -> None:
    zero_amounts = {sleeve: Decimal(0) for sleeve in ("equity", "debt", "gold", "cash")}
    plan = AllocationPlan(
        target_rupees=zero_amounts,
        target_percent=zero_amounts,
        current_rupees=zero_amounts,
        current_percent=zero_amounts,
        gap={
            "equity": Decimal("810000"),
            "debt": Decimal("-380000"),
            "gold": Decimal("-80000"),
            "cash": Decimal("-350000"),
        },
        buckets=(),
        investable_total=Decimal("0"),
    )

    assert debt_answer(plan, WealthSnapshot()).new_money_order == ("equity",)

    tied_gaps = replace(
        plan,
        gap={
            "equity": Decimal("500"),
            "debt": Decimal("500"),
            "gold": Decimal(0),
            "cash": Decimal("250"),
        },
    )
    assert debt_answer(tied_gaps, WealthSnapshot()).new_money_order == (
        "equity",
        "debt",
        "cash",
    )

    no_underweight_sleeves = replace(
        plan,
        gap={
            "equity": Decimal("-100"),
            "debt": Decimal(0),
            "gold": Decimal("-200"),
            "cash": Decimal(0),
        },
    )
    assert debt_answer(no_underweight_sleeves, WealthSnapshot()).new_money_order == ()
