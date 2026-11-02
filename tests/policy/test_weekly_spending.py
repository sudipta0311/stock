from __future__ import annotations

from decimal import Decimal

from wealth.policy.actions import Action
from wealth.policy.personalise import WeeklySpendInput, _apply_weekly_spending

_FX = {"DKK/INR": Decimal("12.345679")}


def _abroad_action(
    amount_inr: str = "2000.00", amount_dkk: str = "162", *, rank: int = 1
) -> Action:
    return Action(
        rank=rank,
        kind="BUY_ABROAD",
        sleeve="equity_global",
        instrument_category="abroad_etf",
        amount=Decimal(amount_inr),
        reason_code="buy_abroad",
        member_id="member-a",
        params={
            "abroad_buy_list": [
                {
                    "member_id": "member-a",
                    "sleeve": "equity_global",
                    "account": "ordinary",
                    "instrument": {"name": "Synthetic European fund"},
                    "amount_dkk": amount_dkk,
                    "amount_inr": amount_inr,
                }
            ]
        },
    )


def _lumpsum_action(
    amount: str, *, member_id: str = "member-a", sleeve: str = "equity_global"
) -> Action:
    return Action(
        rank=1,
        kind="BUY_LUMPSUM",
        sleeve=sleeve,
        instrument_category="global_index_fund",
        amount=Decimal(amount),
        reason_code="target_gap",
        member_id=member_id,
    )


def test_full_abroad_spend_removes_same_step_and_preserves_carry_forward() -> None:
    sell = Action(
        rank=1,
        kind="SELL_TRIM",
        sleeve="equity_global",
        instrument_category="stock",
        amount=Decimal("500.00"),
        reason_code="synthetic_exit",
        member_id="member-a",
    )
    buy = _abroad_action(rank=2)

    actions, carry_forward = _apply_weekly_spending(
        [sell, buy],
        Decimal("10345.60"),
        [WeeklySpendInput("member-a", "equity_global", Decimal("2000.00"))],
        _FX,
    )

    assert [action.kind for action in actions] == ["SELL_TRIM"]
    assert actions[0].amount == Decimal("500.00")
    assert carry_forward == Decimal("10345.60")


def test_half_abroad_spend_keeps_half_of_the_step() -> None:
    actions, carry_forward = _apply_weekly_spending(
        [_abroad_action()],
        Decimal(0),
        [WeeklySpendInput("member-a", "equity_global", Decimal("1000.00"))],
        _FX,
    )

    assert len(actions) == 1
    assert actions[0].amount == Decimal("1000.00")
    assert actions[0].params["abroad_buy_list"][0]["amount_inr"] == "1000.00"
    assert actions[0].params["abroad_buy_list"][0]["amount_dkk"] == "81"
    assert carry_forward == 0


def test_subminimum_abroad_remainder_moves_to_carry_forward() -> None:
    actions, carry_forward = _apply_weekly_spending(
        [_abroad_action("1500.00", "122")],
        Decimal(0),
        [WeeklySpendInput("member-a", "equity_global", Decimal("1000.00"))],
        _FX,
    )

    assert actions == []
    assert carry_forward == Decimal("500.00")


def test_denmark_stock_line_below_its_dkk_minimum_moves_to_carry_forward() -> None:
    action = _abroad_action("30000.00", "2430")
    line = action.params["abroad_buy_list"][0]
    line["reason_codes"] = ["approved_us_stock_idea"]

    actions, carry_forward = _apply_weekly_spending(
        [action],
        Decimal(0),
        [WeeklySpendInput("member-a", "equity_global", Decimal("10000.00"))],
        _FX,
    )

    assert actions == []
    assert carry_forward == Decimal("20000.00")


def test_extra_money_keeps_only_the_new_share_after_prior_spend() -> None:
    actions, carry_forward = _apply_weekly_spending(
        [_abroad_action("3000.00", "243")],
        Decimal("9345.60"),
        [WeeklySpendInput("member-a", "equity_global", Decimal("2000.00"))],
        _FX,
    )

    assert len(actions) == 1
    assert actions[0].amount == Decimal("1000.00")
    assert actions[0].params["abroad_buy_list"][0]["amount_inr"] == "1000.00"
    assert carry_forward == Decimal("9345.60")


def test_spending_stays_with_its_member_and_sleeve() -> None:
    actions, _carry_forward = _apply_weekly_spending(
        [
            _lumpsum_action("1500.00", sleeve="equity_india"),
            _lumpsum_action("2000.00", member_id="member-b"),
        ],
        Decimal(0),
        [WeeklySpendInput("member-a", "equity_global", Decimal("2000.00"))],
        _FX,
    )

    assert [action.amount for action in actions] == [Decimal("1500.00"), Decimal("2000.00")]


def test_spending_is_subtracted_from_same_sleeve_steps_in_rank_order() -> None:
    first = _lumpsum_action("1500.00")
    second = _abroad_action("2000.00", "162", rank=2)
    actions, _carry_forward = _apply_weekly_spending(
        [second, first],
        Decimal(0),
        [WeeklySpendInput("member-a", "equity_global", Decimal("2000.00"))],
        _FX,
    )

    assert [action.kind for action in actions] == ["BUY_ABROAD"]
    assert actions[0].rank == 1
    assert actions[0].amount == Decimal("1500.00")
