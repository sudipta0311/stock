from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wealth.api.personalisation import _idea_ownership
from wealth.policy import personalise
from wealth.policy.actions import Action, SatelliteIdea
from wealth.policy.denmark import (
    account_lines,
    ask_deposit_room_dkk,
    ask_rule_for_year,
    dkk_from_inr,
    inr_from_dkk,
    round_dkk_for_display,
)
from wealth.policy.denmark_picks import (
    EtfPick,
    approved_us_stock_buys,
    global_fund_ideas,
    pick_india_vehicle,
    plain_fund_name,
    ranked_etfs,
)
from wealth.policy.household import MemberProfile
from wealth.policy.personalise import (
    HoldingInput,
    MarketContext,
    PolicyProfile,
    ResearchPack,
    UserData,
    _abroad_action_rows,
    _share_buy_account,
)
from wealth.schemas.profile import FinancialContext, InvestingPrefs, MemberTax, Residency


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


def test_ask_rules_fall_back_to_the_latest_earlier_year() -> None:
    rules = {"denmark": {"ask": {"deposit_cap_dkk": {2026: "174200"}}}}

    assert ask_rule_for_year(rules, "deposit_cap_dkk", 2026) == ("174200", True)
    assert ask_rule_for_year(rules, "deposit_cap_dkk", 2027) == ("174200", False)


def test_source_uses_year_maps_for_ask_rules() -> None:
    source_root = Path(__file__).parents[2] / "src" / "wealth"
    year_suffixed_ask_key = re.compile(r"(?:deposit_cap_dkk|tax_rate_pct)_20\d{2}")

    for source_path in source_root.rglob("*.py"):
        assert year_suffixed_ask_key.search(source_path.read_text(encoding="utf-8")) is None


def test_displayed_fund_name_uses_plain_language() -> None:
    assert plain_fund_name("SPDR MSCI World UCITS ETF USD (Acc)") == "SPDR MSCI World fund"
    assert plain_fund_name("Example fund USD") == "Example fund"


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


def test_global_fund_ideas_follow_rules_order_and_mark_the_plan_fund() -> None:
    ideas = global_fund_ideas("DK", "nordnet")

    assert [(idea.exposure, idea.role, idea.market) for idea in ideas] == [
        ("global_all_world", "plan_main", None),
        ("global_sp500", "option", "US"),
        ("global_nasdaq100", "option", "US"),
    ]
    assert all(idea.id and idea.name and idea.currency and idea.brokers for idea in ideas)
    assert all(idea.ask_eligible for idea in ideas)
    assert global_fund_ideas("IN", "nordnet") == ()


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


@pytest.mark.parametrize(
    ("through_fund_inr", "expected_stock_dkk"),
    [
        (Decimal("0"), Decimal("3500")),
        (Decimal("43750"), None),
        (Decimal("12500"), Decimal("2500")),
    ],
)
def test_denmark_us_stock_room_counts_direct_and_fund_holdings(
    through_fund_inr: Decimal,
    expected_stock_dkk: Decimal | None,
) -> None:
    member = _dk_member(us_stock_picks_opt_in=True)
    context = SimpleNamespace(
        member=member,
        snapshot=SimpleNamespace(net_worth=Decimal("1250000"), investable_total=Decimal("1250000")),
        limits_room={"ask_deposit_dkk": Decimal(0)},
    )
    market = MarketContext(
        fx_rates={"DKK/INR": Decimal("12.5"), "USD/INR": Decimal("12.5")},
        debt_yields={},
        index_valuation_percentiles={},
    )
    idea = SatelliteIdea(
        security_id="synthetic-us-idea",
        symbol="SYNUS",
        market="US",
        sector="Industrials",
        price=Decimal("100"),
        quant_decile=10,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("35"),
        confidence="HIGH",
        name="Synthetic US Company",
        isin="US0000000001",
        exchange="NASDAQ",
        currency="USD",
    )
    direct = HoldingInput(
        security_id="direct-us",
        symbol=None,
        market="US",
        currency="USD",
        security_type="equity",
        sleeve="equity_global",
        category="equity:us",
        quantity=Decimal("15"),
        price=Decimal("100"),
        member_id=member.id,
        isin="US0000000001",
    )
    action = _global_buy(amount_inr="1000000", account_type="ORDINARY")
    action = action.model_copy(update={"member_id": member.id})
    actions, lines, _carry, _notes = _abroad_action_rows(
        [action],
        [member],
        [context],
        [direct],
        market,
        ResearchPack(week="2026-W45", ideas=(idea,)),
        {(member.id, idea.security_id): through_fund_inr} if through_fund_inr else {},
        {("US", "SYNUS"): ("US0000000001",)},
    )
    stock_lines = [line for line in lines if line["instrument"].get("ticker") == "SYNUS"]

    assert (
        Decimal(str(stock_lines[0]["amount_dkk"])) if stock_lines else None
    ) == expected_stock_dkk
    assert actions[0].kind == "BUY_ABROAD"


@pytest.mark.parametrize(
    ("held_dkk", "expected_status", "has_plan_step"),
    [
        (Decimal("3100"), "enough", False),
        (Decimal("3000"), "some", True),
    ],
)
def test_ideas_enough_status_matches_denmark_plan_minimum_room(
    held_dkk: Decimal,
    expected_status: str,
    has_plan_step: bool,
) -> None:
    member = _dk_member(us_stock_picks_opt_in=True, us_stock_picks_pct=Decimal("10"))
    idea = SatelliteIdea(
        security_id="synthetic-us-room",
        symbol="ROOMUS",
        market="US",
        sector="Technology",
        price=Decimal("100"),
        quant_decile=10,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("35"),
        confidence="HIGH",
        name="Synthetic US room company",
        isin="US0000000010",
        exchange="NASDAQ",
        currency="USD",
    )
    held_inr = held_dkk
    user_data = UserData(
        profile=PolicyProfile(),
        financial_context=FinancialContext(monthly_expenses=Decimal("10000")),
        goals=(),
        valued_assets=(),
        holdings=(),
        investing=member.investing,
        members=(member,),
        through_fund_inr={idea.security_id: held_inr},
        through_fund_by_member_security={(member.id, idea.security_id): held_inr},
    )
    context = SimpleNamespace(
        member_id=member.id,
        member=member,
        residency=member.residency,
        snapshot=SimpleNamespace(investable_total=Decimal("100000")),
    )
    market = MarketContext(
        fx_rates={"DKK/INR": Decimal(1)},
        debt_yields={},
        index_valuation_percentiles={},
    )
    ownership = _idea_ownership(
        idea,
        user_data,
        market,
        SimpleNamespace(allocation={}),
        {member.id: context},
    )
    plan_picks = approved_us_stock_buys(
        [idea],
        investable_wealth_dkk=Decimal("100000"),
        available_dkk=Decimal("5000"),
        satellite_pct=Decimal("10"),
        held_dkk_by_security={idea.security_id: held_dkk},
    )

    assert ownership.status == expected_status
    assert bool(plan_picks) is has_plan_step
    assert (ownership.status == "enough") is (not bool(plan_picks))


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


def _approved_stock_idea(security_id: str, symbol: str) -> SatelliteIdea:
    return SatelliteIdea(
        security_id=security_id,
        symbol=symbol,
        market="US",
        sector="Industrials",
        price=Decimal("100"),
        quant_decile=10,
        dossier_status="APPROVED",
        valuation_percentile=Decimal("35"),
        confidence="HIGH",
        name=f"Synthetic {symbol} Company",
        isin="US0000000001" if symbol.endswith("A") else "US0000000002",
        exchange="NASDAQ",
        currency="USD",
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


def _run_abroad(
    actions: list[Action],
    member: MemberProfile,
    ask_room: Decimal,
    *,
    research: ResearchPack | None = None,
    investable_total_inr: Decimal = Decimal("1250000"),
) -> tuple[list[Action], list[dict[str, Any]], Decimal, list[str]]:
    context = SimpleNamespace(
        member=member,
        snapshot=SimpleNamespace(
            net_worth=investable_total_inr, investable_total=investable_total_inr
        ),
        limits_room={"ask_deposit_dkk": ask_room},
    )
    market = MarketContext(
        fx_rates={"DKK/INR": Decimal("12.5"), "USD/INR": Decimal("12.5")},
        debt_yields={},
        index_valuation_percentiles={},
    )
    return _abroad_action_rows(actions, [member], [context], [], market, research, {})


def test_approved_share_uses_room_left_after_the_etf_and_marks_unverified_rules() -> None:
    member = _dk_member(us_stock_picks_opt_in=True, us_stock_picks_pct="25")
    idea = _approved_stock_idea("share-a", "SYNTHA")

    actions, lines, _carry, _notes = _run_abroad(
        [_global_buy(amount_inr="250000", account_type="ASK")],
        member,
        Decimal("20000"),
        research=ResearchPack(week="2026-W45", ideas=(idea,)),
    )

    assert [(line["account"], line["amount_dkk"]) for line in lines] == [
        ("ask", "15000"),
        ("ask", "5000"),
    ]
    share_line = lines[1]
    assert share_line["reason_codes"] == ["approved_us_stock_idea", "ask_room_remaining"]
    assert share_line["details"]["ask_unverified"] is True
    assert actions[0].params["account_type"] == "ASK"


def test_two_share_buys_use_remaining_room_in_order_after_the_etf() -> None:
    member = _dk_member(us_stock_picks_opt_in=True, us_stock_picks_pct="25")
    ideas = (_approved_stock_idea("share-a", "SYNTHA"), _approved_stock_idea("share-b", "SYNTHB"))

    actions, lines, _carry, _notes = _run_abroad(
        [_global_buy(amount_inr="300000", account_type="ASK")],
        member,
        Decimal("22000"),
        research=ResearchPack(week="2026-W45", ideas=ideas),
        investable_total_inr=Decimal("750000"),
    )

    assert [(line["account"], line["amount_dkk"]) for line in lines] == [
        ("ask", "18000"),
        ("ask", "3000"),
        ("ordinary", "3000"),
    ]
    assert lines[2]["reason_codes"] == ["approved_us_stock_idea", "ordinary_after_ask"]
    assert actions[0].params["account_type"] == "ASK"


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


def test_share_buys_use_only_room_left_after_the_etf_line() -> None:
    account, reasons, room_left = _share_buy_account(
        "US",
        Decimal("5000"),
        Decimal("8000"),
        ask_eligible_markets=["US"],
        may_use_ask=True,
        requested_account="ASK",
    )

    assert account == "ask"
    assert reasons == ["approved_us_stock_idea", "ask_room_remaining"]
    assert room_left == Decimal("3000")


def test_share_buy_uses_ordinary_account_instead_of_splitting_when_room_is_short() -> None:
    account, reasons, room_left = _share_buy_account(
        "US",
        Decimal("5000"),
        Decimal("2000"),
        ask_eligible_markets=["US"],
        may_use_ask=True,
        requested_account="ASK",
    )

    assert account == "ordinary"
    assert reasons == ["approved_us_stock_idea", "ordinary_after_ask"]
    assert room_left == Decimal("2000")


def test_ordinary_request_keeps_every_share_buy_in_ordinary_account() -> None:
    account, reasons, room_left = _share_buy_account(
        "US",
        Decimal("3000"),
        Decimal("5000"),
        ask_eligible_markets=["US"],
        may_use_ask=True,
        requested_account="ORDINARY",
    )

    assert account == "ordinary"
    assert reasons == ["approved_us_stock_idea"]
    assert room_left == Decimal("5000")


def test_member_who_cannot_use_ask_gets_no_after_ask_reason() -> None:
    account, reasons, room_left = _share_buy_account(
        "US",
        Decimal("3000"),
        Decimal("0"),
        ask_eligible_markets=["US"],
        may_use_ask=False,
        requested_account="ASK",
    )

    assert account == "ordinary"
    assert reasons == ["approved_us_stock_idea"]
    assert room_left == Decimal("0")


def test_share_buys_use_room_in_order_without_splitting() -> None:
    first = _share_buy_account(
        "US",
        Decimal("3000"),
        Decimal("4000"),
        ask_eligible_markets=["US"],
        may_use_ask=True,
        requested_account="ASK",
    )
    second = _share_buy_account(
        "US",
        Decimal("3000"),
        first[2],
        ask_eligible_markets=["US"],
        may_use_ask=True,
        requested_account="ASK",
    )

    assert (first[0], first[2]) == ("ask", Decimal("1000"))
    assert second == (
        "ordinary",
        ["approved_us_stock_idea", "ordinary_after_ask"],
        Decimal("1000"),
    )


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

    def ranked(_exposure, _broker, *, ask_first=False, rules=None):
        del ask_first, rules
        return [pick], True

    monkeypatch.setattr(personalise, "ranked_etfs_for_member", ranked)
    member = _dk_member(broker="other")
    _actions, lines, _carry, _notes = _run_abroad(
        [_global_buy(amount_inr="125000", account_type="ORDINARY")], member, Decimal(0)
    )
    assert "broker_availability_unchecked" in lines[0]["reason_codes"]


def test_index_fund_idea_matches_the_personalise_pick_for_the_same_member() -> None:
    member = _dk_member(broker="nordnet")
    world_fund = global_fund_ideas("DK", member.investing.broker)[0]

    _actions, lines, _carry, _notes = _run_abroad(
        [_global_buy(amount_inr="125000", account_type="ASK")],
        member,
        Decimal("10000"),
    )

    assert lines[0]["instrument"]["name"] == world_fund.name


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
