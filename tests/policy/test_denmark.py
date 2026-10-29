from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from wealth.policy import personalise
from wealth.policy.actions import Action
from wealth.policy.denmark import (
    account_lines,
    ask_deposit_room_dkk,
    dkk_from_inr,
    inr_from_dkk,
    round_dkk_for_display,
)
from wealth.policy.denmark_picks import (
    EtfPick,
    approved_us_stock_buys,
    pick_india_vehicle,
    plain_fund_name,
    ranked_etfs,
)
from wealth.policy.household import MemberProfile
from wealth.policy.personalise import MarketContext, _abroad_action_rows
from wealth.schemas.profile import InvestingPrefs, MemberTax, Residency


def _row(
    item_id: str,
    *,
    ter: str,
    distribution: str = "accumulating",
    size: str = "large",
    currency: str = "EUR",
    broker: str = "nordnet",
    abis: bool = True,
    ask: bool = True,
) -> dict[str, object]:
    return {
        "id": item_id,
        "name": item_id,
        "isin": f"IE00TEST{len(item_id):03d}1",
        "exposure": "global_all_world",
        "ter_pct": ter,
        "distribution": distribution,
        "fund_size_band": size,
        "abis_listed": abis,
        "ask_eligible": ask,
        "available_at": [broker],
        "tickers": {"Xetra": {"symbol": item_id.upper(), "currency": currency}},
        "us_equivalent": "ACWI",
    }


def test_dkk_rounding_and_fx_conversion_use_decimal() -> None:
    assert round_dkk_for_display(Decimal("999.5")) == Decimal("1000")
    assert round_dkk_for_display(Decimal("1049")) == Decimal("1000")
    assert round_dkk_for_display(Decimal("1050")) == Decimal("1100")
    assert inr_from_dkk(Decimal("100"), Decimal("12.5")) == Decimal("1250.00")
    assert dkk_from_inr(Decimal("1250"), Decimal("12.5")) == Decimal("100")


def test_ask_room_uses_start_value_and_nets_calendar_year_withdrawals() -> None:
    cap = Decimal("174200")
    assert ask_deposit_room_dkk(cap, Decimal("105000"), Decimal(0)) == Decimal("69200")
    assert ask_deposit_room_dkk(cap, Decimal("180000"), Decimal("-10000")) == Decimal("4200")
    assert ask_deposit_room_dkk(cap, Decimal("100000"), Decimal("2000")) == Decimal("72200")


def test_displayed_fund_name_uses_plain_language() -> None:
    assert plain_fund_name("SPDR MSCI World UCITS ETF (Acc)") == "SPDR MSCI World fund (Acc)"


def test_ranked_etfs_apply_availability_abis_and_ordering() -> None:
    rules = {
        "instruments": [
            _row("higher_fee", ter="0.30"),
            _row("distributing", ter="0.12", distribution="distributing"),
            _row("lower_fee", ter="0.12", size="medium"),
            _row("not_abis", ter="0.01", abis=False),
            _row("wrong_broker", ter="0.01", broker="saxo"),
        ]
    }
    selected = ranked_etfs("global_all_world", "nordnet", rules=rules)
    assert [item.instrument["id"] for item in selected] == [
        "lower_fee",
        "distributing",
        "higher_fee",
    ]
    assert (
        ranked_etfs("global_all_world", "saxo", rules=rules)[0].instrument["id"] == "wrong_broker"
    )
    assert (
        ranked_etfs(
            "global_all_world",
            "nordnet",
            ask_only=True,
            rules={"instruments": [_row("ordinary_only", ter="0.1", ask=False)]},
        )
        == []
    )


def test_ranked_etfs_can_apply_the_same_order_without_a_broker_filter() -> None:
    rules = {"instruments": [_row("nordnet_only", ter="0.1")]}
    assert ranked_etfs("global_all_world", "other", rules=rules) == []
    assert [
        pick.instrument["id"]
        for pick in ranked_etfs("global_all_world", "other", broker_filter=False, rules=rules)
    ] == ["nordnet_only"]


def test_comparison_direction_is_separate_from_rounded_absolute_cost() -> None:
    row = _row("comparison", ter="0.12")
    pick = EtfPick(row, "Xetra", "CMP", "EUR")
    rules = {
        "instruments": [row],
        "selection": {
            "comparison": {
                "dividend_withholding_pct": "15",
                "dividend_yields_pct": {
                    "global_all_world": {"value": "1.5", "source_url": "https://example.test/yield"}
                },
                "us_equivalent_ter_pct": {
                    "ACWI": {"value": "0.03", "source_url": "https://example.test/fee"}
                },
            }
        },
    }
    from wealth.policy.denmark_picks import comparison_for_etf

    comparison = comparison_for_etf(pick, rules=rules)
    assert comparison is not None
    assert comparison["annual_difference_pct"] == "0.32"
    assert comparison["annual_difference_direction"] == "more"

    lower_fee_row = {**row, "ter_pct": "0.01"}
    less_rules = {
        "selection": {
            "comparison": {
                "dividend_withholding_pct": "15",
                "dividend_yields_pct": {
                    "global_all_world": {"value": "1.5", "source_url": "https://example.test/yield"}
                },
                "us_equivalent_ter_pct": {
                    "ACWI": {"value": "0.50", "source_url": "https://example.test/fee"}
                },
            }
        }
    }
    less = comparison_for_etf(EtfPick(lower_fee_row, "Xetra", "CMP", "EUR"), rules=less_rules)
    assert less is not None
    assert less["annual_difference_pct"] == "0.27"
    assert less["annual_difference_direction"] == "less"


def test_account_lines_use_share_savings_room_then_ordinary() -> None:
    pick = EtfPick(
        instrument={
            "name": "World fund",
            "isin": "IE00TEST0001",
            "ask_eligible": True,
            "factsheet_url": "https://example.invalid/factsheet.pdf",
            "abis_source_url": "https://example.invalid/abis.xlsx",
            "ask_source_url": "https://example.invalid/ask",
        },
        exchange="Xetra",
        ticker="WLDX",
        currency="EUR",
    )
    partial = account_lines(
        pick,
        member_id="member",
        broker="nordnet",
        amount_dkk=Decimal("10000"),
        ask_room_dkk=Decimal("6000"),
        inr_per_dkk=Decimal("12.5"),
    )
    full = account_lines(
        pick,
        member_id="member",
        broker="nordnet",
        amount_dkk=Decimal("10000"),
        ask_room_dkk=Decimal(0),
        inr_per_dkk=Decimal("12.5"),
    )
    assert [(line["account"], line["amount_dkk"]) for line in partial] == [
        ("ask", "6000"),
        ("ordinary", "4000"),
    ]
    assert len(full) == 1 and full[0]["account"] == "ordinary"
    assert full[0]["amount_inr"] == "125000.00"


def test_india_vehicle_defaults_to_european_fund_or_direct_account() -> None:
    rules = {"instruments": [_row("india", ter="0.65")], "selection": {}}
    rules["instruments"][0]["exposure"] = "india_equity"
    assert (
        pick_india_vehicle(has_indian_nri_share_account=False, broker="nordnet", rules=rules)
        == "ucits_india_etf"
    )
    assert (
        pick_india_vehicle(has_indian_nri_share_account=True, broker="nordnet", rules=rules)
        == "in_equity_direct_core"
    )


def test_us_stock_picks_respect_opt_in_filters_and_code_caps() -> None:
    ideas = [
        {
            "security_id": "one",
            "symbol": "AAA",
            "market": "US",
            "dossier_status": "APPROVED",
            "quant_decile": 10,
            "name": "Company One",
            "isin": "US00TEST0001",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "security_id": "two",
            "symbol": "BBB",
            "market": "US",
            "dossier_status": "PENDING",
            "quant_decile": 10,
            "name": "Company Two",
            "isin": "US00TEST0002",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
    ]
    picks = approved_us_stock_buys(
        ideas,
        investable_wealth_dkk=Decimal("100000"),
        available_dkk=Decimal("50000"),
        satellite_pct=Decimal("25"),
    )
    assert len(picks) == 1
    assert picks[0][1] == Decimal("5000")
    assert (
        approved_us_stock_buys(
            ideas,
            investable_wealth_dkk=Decimal("100000"),
            available_dkk=Decimal("1000"),
            satellite_pct=Decimal("10"),
        )
        == []
    )


def _dk_member(member_id: str = "member", **pref_updates: object) -> MemberProfile:
    return MemberProfile(
        id=member_id,
        name="Member",
        relationship="self",
        is_self=True,
        residency=Residency(status="NRI", country="DK"),
        tax=MemberTax(),
        investing=InvestingPrefs.model_validate(pref_updates),
    )


def _global_action() -> Action:
    return Action(
        rank=1,
        kind="BUY_LUMPSUM",
        sleeve="equity_global",
        instrument_category="global_all_world",
        amount=Decimal("125000"),
        reason_code="target_gap",
        member_id="member",
        params={"account_type": "ORDINARY"},
    )


def _global_buy(*, amount_inr: str, account_type: str) -> Action:
    return _global_action().model_copy(
        update={
            "amount": Decimal(amount_inr),
            "params": {"account_type": account_type},
        }
    )


def _run_abroad(actions: list[Action], member: MemberProfile, ask_room: Decimal):
    context = SimpleNamespace(
        member=member,
        snapshot=SimpleNamespace(net_worth=Decimal("10000000")),
        limits_room={"ask_deposit_dkk": ask_room},
    )
    market = MarketContext(
        fx_rates={"DKK/INR": Decimal("12.5")},
        debt_yields={},
        index_valuation_percentiles={},
    )
    return _abroad_action_rows(actions, [member], [context], [], market, None)


def test_ordinary_action_yields_only_ordinary_lines() -> None:
    member = _dk_member()
    actions, lines, carry, _notes = _run_abroad(
        [_global_buy(amount_inr="125000", account_type="ORDINARY")],
        member,
        Decimal("50000"),
    )
    assert len(lines) == 1
    assert lines[0]["amount_dkk"] == "10000"
    assert lines[0]["amount_inr"] == "125000.00"
    assert lines[0]["account"] == "ordinary"
    assert actions[0].kind == "BUY_ABROAD"
    assert actions[0].params["broker"] == "nordnet"
    assert actions[0].params["account_type"] == "ORDINARY"
    assert carry == 0


def test_two_actions_for_one_member_never_use_more_than_remaining_ask_room() -> None:
    actions, lines, _carry, _notes = _run_abroad(
        [
            _global_buy(amount_inr="75000", account_type="ASK"),
            _global_buy(amount_inr="75000", account_type="ASK"),
        ],
        _dk_member(),
        Decimal("8000"),
    )
    ask_lines = [line for line in lines if line["account"] == "ask"]
    assert sum(Decimal(str(line["amount_dkk"])) for line in ask_lines) == Decimal("8000")
    assert [line["account"] for line in lines] == ["ask", "ask", "ordinary"]
    assert all(
        action.params["account_type"] == action.params["abroad_buy_list"][0]["account"].upper()
        for action in actions
    )


def test_ask_action_larger_than_running_room_splits_ask_and_ordinary() -> None:
    actions, lines, _carry, _notes = _run_abroad(
        [_global_buy(amount_inr="125000", account_type="ASK")],
        _dk_member(),
        Decimal("6000"),
    )
    assert [(line["account"], line["amount_dkk"]) for line in lines] == [
        ("ask", "6000"),
        ("ordinary", "4000"),
    ]
    assert actions[0].params["account_type"] == "ASK"


def test_unconfirmed_broker_availability_is_flagged_on_the_buy_line(
    monkeypatch,
) -> None:
    instrument = {
        "name": "World fund",
        "isin": "IE00TEST0001",
        "ask_eligible": True,
        "factsheet_url": "https://example.test/factsheet",
        "abis_source_url": "https://example.test/abis",
        "ask_source_url": "https://example.test/ask",
    }
    pick = EtfPick(instrument, "Xetra", "WLDX", "EUR")

    def ranked(_exposure, _broker, *, ask_only=False, broker_filter=True, rules=None):
        del ask_only, rules
        return [] if broker_filter else [pick]

    monkeypatch.setattr(personalise, "ranked_etfs", ranked)
    member = _dk_member(broker="other")
    _actions, lines, _carry, _notes = _run_abroad(
        [_global_buy(amount_inr="125000", account_type="ORDINARY")], member, Decimal(0)
    )
    assert "broker_availability_unchecked" in lines[0]["reason_codes"]


def test_ask_display_rounds_down_and_split_dust_is_merged() -> None:
    pick = EtfPick(
        instrument={
            "name": "World fund",
            "isin": "IE00TEST0001",
            "ask_eligible": True,
            "factsheet_url": "https://example.invalid/factsheet.pdf",
            "abis_source_url": "https://example.invalid/abis.xlsx",
            "ask_source_url": "https://example.invalid/ask",
        },
        exchange="Xetra",
        ticker="WLDX",
        currency="EUR",
    )
    lines = account_lines(
        pick,
        member_id="member",
        broker="nordnet",
        amount_dkk=Decimal("1050.5"),
        ask_room_dkk=Decimal("1050.5"),
        inr_per_dkk=Decimal("12.5"),
    )
    assert lines[0]["amount_dkk"] == "1000"
    assert lines[0]["_amount_dkk_unrounded"] == Decimal("1050.5")
    assert Decimal(str(lines[0]["amount_dkk"])) <= Decimal("1050.5")
    dust_lines = account_lines(
        pick,
        member_id="member",
        broker="nordnet",
        amount_dkk=Decimal("10.5"),
        ask_room_dkk=Decimal("10"),
        inr_per_dkk=Decimal("12.5"),
    )
    assert len(dust_lines) == 1 and dust_lines[0]["account"] == "ask"
    assert dust_lines[0]["_amount_dkk_unrounded"] == Decimal("10.5")
    assert dust_lines[0]["amount_dkk"] == "10"
    ask_dust_lines = account_lines(
        pick,
        member_id="member",
        broker="nordnet",
        amount_dkk=Decimal("100"),
        ask_room_dkk=Decimal("0.5"),
        inr_per_dkk=Decimal("12.5"),
    )
    assert [line["account"] for line in ask_dust_lines] == ["ordinary"]


def test_lines_over_the_member_limit_shrink_the_action_and_carry_forward() -> None:
    actions, lines, carry, _notes = _run_abroad(
        [
            _global_buy(amount_inr="25000", account_type="ASK"),
            _global_buy(amount_inr="25000", account_type="ORDINARY"),
            _global_buy(amount_inr="125000", account_type="ASK"),
        ],
        _dk_member(),
        Decimal("7000"),
    )
    assert [(line["account"], line["amount_dkk"]) for line in lines] == [
        ("ask", "2000"),
        ("ordinary", "2000"),
        ("ask", "5000"),
    ]
    assert carry == Decimal("62500.00")
    assert [action.amount for action in actions] == [
        Decimal("25000"),
        Decimal("25000"),
        Decimal("62500.00"),
    ]


def test_resident_indian_member_plan_is_unchanged() -> None:
    member = MemberProfile(
        id="member",
        name="Member",
        relationship="self",
        is_self=True,
        residency=Residency(status="RI", country="IN"),
    )
    action = _global_action()
    actions, lines, carry, _notes = _run_abroad([action], member, Decimal(0))
    assert actions == [action]
    assert lines == []
    assert carry == 0
