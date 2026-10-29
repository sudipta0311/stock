"""Deterministic Indian-fund checks with synthetic scheme data."""

from __future__ import annotations

from calendar import monthrange
from datetime import date
from decimal import Decimal

import pytest

import wealth.policy.fund_check as fund_check_policy
from wealth.marketdata.sources.amfi import classify_amfi_scheme_name, classify_amfi_section
from wealth.policy.actions import Sip, WeekInput
from wealth.policy.allocation import AllocationPlan
from wealth.policy.debt_picker import DebtPick
from wealth.policy.foundation import FoundationItem, FoundationReport
from wealth.policy.fund_check import (
    FundHolding,
    FundMember,
    FundRebalanceContext,
    FundReference,
    FundTransactionInput,
    _FundSaleLedger,
    check_funds,
    detect_monthly_sips,
    ordered_fund_rebalance_plan,
)
from wealth.policy.rebalance import band_breaches
from wealth.policy.routes import RouteDecision
from wealth.policy.tax import Lot, TaxContext, TaxEstimate

AS_OF = date(2026, 10, 31)
MEMBER = "member-ri"
SPOUSE = "member-spouse"


def _holding(
    security_id: str,
    *,
    name: str | None = None,
    sub_category: str = "Large Cap",
    value: str = "100000",
    member_id: str = MEMBER,
    plan_type: str | None = "direct",
    option_type: str | None = "growth",
    category: str = "mf:equity_india",
    ter: str | None = None,
    tax_lots: tuple[Lot, ...] = (),
    complete: bool = False,
    price: str | None = None,
    exit_load_pct: str | None = None,
) -> FundHolding:
    amount = Decimal(value)
    return FundHolding(
        security_id=security_id,
        name=name or f"Synthetic {security_id} Direct Plan Growth",
        category=category,
        sub_category=sub_category,
        plan_type=plan_type,
        option_type=option_type,
        ter=Decimal(ter) if ter is not None else None,
        member_id=member_id,
        value=amount,
        quantity=amount / Decimal(price) if price else amount,
        price=Decimal(price) if price else None,
        tax_lots=tax_lots,
        tax_lots_complete=complete,
        exit_load_pct=Decimal(exit_load_pct) if exit_load_pct is not None else None,
    )


def _reference(
    security_id: str,
    *,
    name: str | None = None,
    sub_category: str = "Large Cap",
    plan_type: str | None = "direct",
    option_type: str | None = "growth",
    category: str = "mf:equity_india",
    ter: str | None = None,
) -> FundReference:
    return FundReference(
        security_id=security_id,
        name=name or f"Synthetic {security_id} Direct Plan Growth",
        category=category,
        sub_category=sub_category,
        plan_type=plan_type,
        option_type=option_type,
        ter=Decimal(ter) if ter is not None else None,
    )


def _transactions(
    security_id: str,
    amounts: tuple[str, ...] = ("5000", "5000", "5000"),
    dates: tuple[date, ...] = (date(2026, 8, 5), date(2026, 9, 5), date(2026, 10, 5)),
    member_id: str = MEMBER,
) -> list[FundTransactionInput]:
    return [
        FundTransactionInput(
            member_id=member_id,
            security_id=security_id,
            transaction_date=on,
            kind="buy",
            amount=Decimal(amount),
            units=Decimal(amount) / Decimal(100),
        )
        for amount, on in zip(amounts, dates, strict=True)
    ]


def _check(
    holdings: list[FundHolding],
    *,
    references: list[FundReference] | None = None,
    histories: dict[str, list[tuple[date, Decimal]]] | None = None,
    transactions: list[FundTransactionInput] | None = None,
    members: list[FundMember] | None = None,
    previous: set[tuple[str, str]] | None = None,
    risk_profile: str = "growth",
    best_debt_option: dict[str, str] | None = None,
):
    return check_funds(
        holdings=holdings,
        references=references if references is not None else list(holdings),
        histories=histories or {},
        transactions=transactions or [],
        members=members or [FundMember(MEMBER)],
        as_of=AS_OF,
        risk_profile=risk_profile,
        previous_lagging=previous,
        best_debt_option=best_debt_option,
    )


def _history(start: str, finish: str, months: int = 60) -> list[tuple[date, Decimal]]:
    year, month = 2021, 10
    first = Decimal(start)
    last = Decimal(finish)
    rows = []
    for index in range(months + 1):
        day = date(year, month, monthrange(year, month)[1])
        nav = first + (last - first) * Decimal(index) / Decimal(months)
        rows.append((day, nav))
        month = month % 12 + 1
        if month == 1:
            year += 1
    return rows


def test_household_without_fund_holdings_has_only_the_upload_prompt() -> None:
    result = _check(
        [],
        members=[
            FundMember("member-denmark", residency_country="DK"),
            FundMember("member-india", residency_country="IN"),
        ],
    )

    assert result.status == "fine"
    assert result.summary_key == "fund_check.upload"
    assert result.checks == ()
    assert result.actions == ()


def _balanced_holdings(small_pct: str, flexi_pct: str) -> list[FundHolding]:
    equity = Decimal("750000")
    weights = {
        "Large Cap": Decimal("30"),
        "Mid Cap": Decimal("20"),
        "Small Cap": Decimal(small_pct),
        "Flexi Cap": Decimal(flexi_pct),
        "Sectoral/Thematic": Decimal("10"),
        "FoF Overseas": Decimal("5"),
    }
    result = [
        _holding(
            f"mix-{index}",
            sub_category=subcategory,
            value=format(equity * weight / Decimal(100), "f"),
            category="mf:equity_global" if subcategory == "FoF Overseas" else "mf:equity_india",
        )
        for index, (subcategory, weight) in enumerate(weights.items())
    ]
    result.extend(
        [
            _holding("mix-hybrid", sub_category="Aggressive Hybrid", value="100000"),
            _holding("mix-debt", sub_category="Liquid", value="150000", category="mf:debt"),
        ]
    )
    return result


def test_amfi_subcategory_and_plan_option_are_normalized() -> None:
    assert classify_amfi_section("Equity Scheme - Large Cap Fund").sub_category == "Large Cap"
    assert (
        classify_amfi_section("Equity Scheme - Large & Mid Cap Fund").sub_category
        == "Large & Mid Cap"
    )
    assert (
        classify_amfi_section("Hybrid Scheme - Aggressive Hybrid Fund").sub_category
        == "Aggressive Hybrid"
    )
    assert classify_amfi_section("Debt Scheme - Gilt Fund").sub_category == "Gilt"
    assert classify_amfi_section("FoF Overseas").category == "mf:equity_global"
    assert classify_amfi_scheme_name("Synthetic Top 100 Fund - Direct Plan - Growth") == (
        "direct",
        "growth",
    )
    assert classify_amfi_scheme_name("Synthetic Top 100 Fund - Regular Plan - IDCW Payout") == (
        "regular",
        "idcw",
    )


def test_sip_detection_accepts_one_missed_month_and_ignores_lump_sum() -> None:
    rows = _transactions(
        "fund",
        dates=(
            date(2026, 1, 5),
            date(2026, 2, 5),
            date(2026, 4, 5),
            date(2026, 5, 5),
        ),
        amounts=("10000", "10100", "9900", "10000"),
    )
    rows.append(
        FundTransactionInput(
            member_id=MEMBER,
            security_id="lump",
            transaction_date=date(2026, 10, 1),
            kind="buy",
            amount=Decimal("50000"),
            units=Decimal("500"),
        )
    )

    assert detect_monthly_sips(rows, AS_OF) == {(MEMBER, "fund"): Decimal("10000")}


def test_regular_plan_action_has_switch_tax_and_uses_ltcg_exemption() -> None:
    lot = Lot(
        "synthetic-lot",
        "equity_mf",
        Decimal("2000"),
        Decimal("900"),
        date(2024, 8, 1),
        market_price=Decimal("1000"),
    )
    regular = _holding(
        "regular",
        name="Synthetic Alpha Fund Regular Plan Growth",
        value="2000000",
        plan_type="regular",
        ter="1.50",
        tax_lots=(lot,),
        complete=True,
        price="1000",
        exit_load_pct="0",
    )
    direct = _reference(
        "direct",
        name="Synthetic Alpha Fund Direct Plan Growth",
        ter="0.50",
    )

    result = _check([regular], references=[regular, direct])

    check = next(item for item in result.checks if item["reason_code"] == "regular_plan")
    assert check["money_at_stake"] == "20000.00"
    assert check["details"]["tax_inr"] == "9750.00"
    assert check["details"]["ltcg_exemption_used_inr"] == "125000.00"
    assert check["details"]["ltcg_exemption_remaining_inr"] == "0.00"
    assert result.actions[0]["kind"] == "SWITCH_TO_DIRECT"
    assert result.actions[0]["member_id"] == MEMBER
    assert result.actions[0]["amount"] == "2000000.00"
    assert result.actions[0]["ltcg_exemption_used_inr"] == "125000.00"
    assert result.actions[0]["ltcg_exemption_remaining_inr"] == "0.00"


def test_fund_sell_suggestions_share_one_ltcg_exemption_per_member() -> None:
    holdings: list[FundHolding] = []
    references: list[FundReference] = []
    for suffix in ("first", "second"):
        regular = _holding(
            f"regular-{suffix}",
            name=f"Synthetic {suffix.title()} Fund Regular Plan Growth",
            value="5000000",
            plan_type="regular",
            ter="1.50",
            tax_lots=(
                Lot(
                    f"synthetic-{suffix}-lot",
                    "equity_mf",
                    Decimal("5000"),
                    Decimal("960"),
                    date(2024, 8, 1),
                ),
            ),
            complete=True,
            price="1000",
            exit_load_pct="0",
        )
        direct = _reference(
            f"direct-{suffix}",
            name=f"Synthetic {suffix.title()} Fund Direct Plan Growth",
            ter="0.50",
        )
        holdings.append(regular)
        references.extend((regular, direct))

    result = _check(holdings, references=references)
    sell_checks = {
        item["details"]["direct_security_id"]: item
        for item in result.checks
        if item["reason_code"] == "regular_plan"
    }

    assert sell_checks["direct-first"]["details"]["tax_inr"] == "9750.00"
    assert sell_checks["direct-first"]["details"]["ltcg_exemption_used_inr"] == "125000.00"
    assert sell_checks["direct-first"]["details"]["ltcg_exemption_remaining_inr"] == "0.00"
    assert sell_checks["direct-second"]["details"]["tax_inr"] == "26000.00"
    assert sell_checks["direct-second"]["details"]["ltcg_exemption_used_inr"] == "0.00"
    assert sell_checks["direct-second"]["details"]["ltcg_exemption_remaining_inr"] == "0.00"
    assert {action["kind"] for action in result.actions} == {"SWITCH_TO_DIRECT"}
    assert len(result.actions) == 2


def test_fund_sell_suggestions_keep_exemptions_separate_by_member() -> None:
    holdings: list[FundHolding] = []
    references: list[FundReference] = []
    members = [FundMember(MEMBER), FundMember(SPOUSE)]
    for member_id, suffix in ((MEMBER, "self"), (SPOUSE, "partner")):
        regular = _holding(
            f"regular-{suffix}",
            name=f"Synthetic {suffix.title()} Fund Regular Plan Growth",
            value="2000000",
            member_id=member_id,
            plan_type="regular",
            ter="1.50",
            tax_lots=(
                Lot(
                    f"synthetic-{suffix}-lot",
                    "equity_mf",
                    Decimal("2000"),
                    Decimal("900"),
                    date(2024, 8, 1),
                ),
            ),
            complete=True,
            price="1000",
            exit_load_pct="0",
        )
        direct = _reference(
            f"direct-{suffix}",
            name=f"Synthetic {suffix.title()} Fund Direct Plan Growth",
            ter="0.50",
        )
        holdings.append(regular)
        references.extend((regular, direct))

    result = _check(holdings, references=references, members=members)

    assert len(result.actions) == 2
    assert {action["member_id"] for action in result.actions} == {MEMBER, SPOUSE}
    for item in result.checks:
        if item["reason_code"] == "regular_plan":
            assert item["details"]["tax_inr"] == "9750.00"
            assert item["details"]["ltcg_exemption_used_inr"] == "125000.00"


def test_fund_sale_ledger_records_equity_stcg_in_member_context() -> None:
    member = FundMember(MEMBER)
    holding = _holding(
        "short-term-fund",
        value="200000",
        tax_lots=(
            Lot(
                "synthetic-short-term-lot",
                "equity_mf",
                Decimal("200"),
                Decimal("500"),
                date(2026, 5, 1),
            ),
        ),
        complete=True,
        price="1000",
        exit_load_pct="0",
    )
    sale_ledger = _FundSaleLedger({MEMBER: member})
    quote = sale_ledger.quote(member, sale_ledger.available_sales(holding), AS_OF)

    assert quote.realised_equity_stcg == Decimal("100000.00")
    sale_ledger.record(member, quote)
    updated_context = sale_ledger.context_for(member)
    assert updated_context.fy_realised_equity_stcg == Decimal("100000.00")
    assert updated_context.fy_realised_equity_ltcg == Decimal("0")


def test_sold_lots_are_removed_from_later_consolidation_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regular = _holding(
        "regular-alpha",
        name="Synthetic Alpha Fund Regular Plan Growth",
        sub_category="Large Cap",
        value="100000",
        plan_type="regular",
        ter="1.50",
        tax_lots=(
            Lot(
                "synthetic-regular-lot",
                "equity_mf",
                Decimal("100"),
                Decimal("900"),
                date(2024, 8, 1),
            ),
        ),
        complete=True,
        price="1000",
        exit_load_pct="0",
    )
    overlapping_candidate = _holding(
        "small-taxable",
        sub_category="Mid Cap",
        value="200000",
        tax_lots=(
            Lot(
                "synthetic-consolidation-lot",
                "equity_mf",
                Decimal("200"),
                Decimal("500"),
                date(2024, 8, 1),
            ),
        ),
        complete=True,
        price="1000",
        exit_load_pct="0",
    )
    holdings = [
        regular,
        overlapping_candidate,
        _holding("fund-large-1", sub_category="Small Cap", value="1000000"),
        _holding("fund-large-2", sub_category="Flexi Cap", value="900000"),
        _holding("fund-large-3", sub_category="Multi Cap", value="800000"),
        _holding("fund-large-4", sub_category="ELSS", value="700000"),
        _holding("fund-large-5", sub_category="Index Funds/ETFs", value="600000"),
        _holding("fund-large-6", sub_category="Sectoral/Thematic", value="500000"),
        _holding("fund-large-7", sub_category="Liquid", value="400000", category="mf:debt"),
    ]
    original_estimator = fund_check_policy.estimate_sale_tax
    estimated_lot_ids: list[tuple[str, ...]] = []

    def track_estimate(
        sales: list[tuple[Lot, Decimal, Decimal]], sale_date: date, ctx: TaxContext
    ) -> TaxEstimate:
        estimated_lot_ids.append(tuple(lot.lot_id for lot, _quantity, _price in sales))
        return original_estimator(sales, sale_date, ctx)

    monkeypatch.setattr(fund_check_policy, "estimate_sale_tax", track_estimate)
    result = _check(
        holdings,
        references=[
            regular,
            _reference(
                "direct-alpha",
                name="Synthetic Alpha Fund Direct Plan Growth",
                ter="0.50",
            ),
        ],
    )

    all_estimated_lots = [lot_id for estimate in estimated_lot_ids for lot_id in estimate]
    assert len(all_estimated_lots) == len(set(all_estimated_lots))
    assert estimated_lot_ids == [
        ("synthetic-regular-lot",),
        ("synthetic-consolidation-lot",),
    ]
    consolidation = next(
        action for action in result.actions if action["kind"] == "CONSOLIDATE_FUNDS"
    )
    assert consolidation["amount"] == "200000.00"
    assert consolidation["ltcg_exemption_used_inr"] == "100000.00"
    assert consolidation["ltcg_exemption_remaining_inr"] == "15000.00"


def test_regular_plan_payback_gate_and_summary_only_tax_note() -> None:
    lot = Lot(
        "synthetic-short-lot",
        "equity_mf",
        Decimal("2000"),
        Decimal("900"),
        date(2026, 5, 1),
        market_price=Decimal("1000"),
    )
    regular = _holding(
        "regular",
        name="Synthetic Alpha Fund Regular Plan Growth",
        value="2000000",
        plan_type="regular",
        ter="1.50",
        tax_lots=(lot,),
        complete=True,
        price="1000",
        exit_load_pct="2.00",
    )
    direct = _reference(
        "direct",
        name="Synthetic Alpha Fund Direct Plan Growth",
        ter="0.50",
    )
    gated = _check([regular], references=[regular, direct])
    check = next(item for item in gated.checks if item["reason_code"] == "regular_plan")
    assert check["details"]["payback_gate"] == "over_limit"
    assert not any(action["kind"] == "SWITCH_TO_DIRECT" for action in gated.actions)

    summary_only = _holding(
        "regular",
        name="Synthetic Alpha Fund Regular Plan Growth",
        value="2000000",
        plan_type="regular",
        ter="1.50",
    )
    skipped = _check([summary_only], references=[summary_only, direct])
    skipped_check = next(item for item in skipped.checks if item["reason_code"] == "regular_plan")
    assert "tax_estimate_unavailable" in skipped_check["details"]["notes"]
    assert not any(action["kind"] == "SWITCH_TO_DIRECT" for action in skipped.actions)


def test_lagging_fund_requires_two_consecutive_monthly_checks() -> None:
    lagging = _holding("lagging", value="250000")
    peers = [_reference(f"peer-{index}", sub_category="Large Cap") for index in range(3)]
    references = [lagging, *peers]
    histories = {
        "lagging": _history("100", "150"),
        "peer-0": _history("100", "200"),
        "peer-1": _history("100", "300"),
        "peer-2": _history("100", "450"),
    }

    first = _check([lagging], references=references, histories=histories)
    second = _check(
        [lagging],
        references=references,
        histories=histories,
        previous={(MEMBER, "lagging")},
    )

    assert not any(item["reason_code"] == "lagging_fund" for item in first.checks)
    lagging_check = next(item for item in second.checks if item["reason_code"] == "lagging_fund")
    assert lagging_check["details"]["outcome"] == "review"
    assert lagging_check["details"]["automatic_sell"] is False
    assert (
        next(action for action in second.actions if action["kind"] == "REVIEW_FUND")["member_id"]
        == MEMBER
    )


def test_too_many_funds_and_sector_heavy_checks_have_money_at_stake() -> None:
    categories = [
        "Large Cap",
        "Mid Cap",
        "Small Cap",
        "Flexi Cap",
        "Multi Cap",
        "ELSS",
        "Index Funds/ETFs",
        "Gilt",
        "Liquid",
    ]
    nine = [
        _holding(f"fund-{index}", sub_category=category, value=str((index + 1) * 1000))
        for index, category in enumerate(categories)
    ]
    too_many = _check(nine)
    too_many_check = next(
        item for item in too_many.checks if item["reason_code"] == "too_many_funds"
    )
    assert Decimal(too_many_check["money_at_stake"]) > 0
    assert too_many_check["action"] is None
    assert "tax_estimate_unavailable" in too_many_check["details"]["notes"]

    heavy = _check(
        [
            _holding("sector", sub_category="Sectoral/Thematic", value="200000"),
            _holding("large", sub_category="Large Cap", value="800000"),
        ]
    )
    sector_check = next(item for item in heavy.checks if item["reason_code"] == "sector_heavy")
    assert sector_check["money_at_stake"] == "100000.00"


def test_sector_heavy_check_keeps_the_member_that_is_over_limit() -> None:
    result = _check(
        [
            _holding("self-sector", sub_category="Sectoral/Thematic", value="200000"),
            _holding("self-large", sub_category="Large Cap", value="800000"),
            _holding(
                "spouse-sector",
                sub_category="Sectoral/Thematic",
                value="100000",
                member_id=SPOUSE,
            ),
            _holding("spouse-large", sub_category="Large Cap", value="900000", member_id=SPOUSE),
        ],
        members=[FundMember(MEMBER), FundMember(SPOUSE)],
    )

    sector_checks = [item for item in result.checks if item["reason_code"] == "sector_heavy"]
    assert len(sector_checks) == 1
    assert sector_checks[0]["member_id"] == MEMBER


def test_closet_index_compares_two_large_funds_and_debt_check_is_note_only() -> None:
    active_a = _holding("active-a", name="Synthetic Large Company A Direct Growth")
    active_b = _holding("active-b", name="Synthetic Large Company B Direct Growth")
    index = _reference(
        "index",
        name="Synthetic Nifty 50 Index Fund Direct Growth",
        sub_category="Index Funds/ETFs",
    )
    closet = _check(
        [active_a, active_b],
        references=[active_a, active_b, index],
        histories={
            "active-a": _history("100", "120"),
            "active-b": _history("100", "125"),
            "index": _history("100", "200"),
        },
    )
    closet_check = next(item for item in closet.checks if item["reason_code"] == "closet_index")
    assert closet_check["money_at_stake"] == "0.00"
    assert closet_check["action"] is None

    debt = _holding(
        "debt",
        sub_category="Liquid",
        category="mf:debt",
        tax_lots=(
            Lot("debt-lot", "debt_mf_post_2023", Decimal("1"), Decimal("100"), date(2024, 1, 1)),
        ),
    )
    debt_result = _check(
        [debt], best_debt_option={"instrument": "bank_fd", "after_tax_yield_pct": "5.50"}
    )
    debt_check = next(
        item for item in debt_result.checks if item["reason_code"] == "debt_fund_vs_fd"
    )
    assert debt_check["details"]["note_only"] is True
    assert debt_check["details"]["comparison"]["instrument"] == "bank_fd"
    assert debt_check["action"] is None


def test_denmark_member_gets_no_new_indian_fund_money_and_sip_redirects() -> None:
    regular = _holding(
        "dk-regular",
        name="Synthetic Alpha Fund Regular Plan Growth",
        member_id="member-dk",
        plan_type="regular",
        ter="1.50",
    )
    direct = _reference(
        "direct",
        name="Synthetic Alpha Fund Direct Plan Growth",
        ter="0.50",
    )
    dk = FundMember(
        "member-dk",
        residency_country="DK",
        residency_status="NRI",
        redirect_member_id=SPOUSE,
        redirect_member_name="RI spouse",
        clubbing_note="The RI spouse would hold redirected monthly investments.",
    )
    spouse = FundMember(SPOUSE)
    result = _check(
        [regular],
        references=[regular, direct],
        transactions=_transactions("dk-regular", member_id="member-dk"),
        members=[dk, spouse],
    )

    assert not any(item["reason_code"] == "regular_plan" for item in result.checks)
    residency_check = next(
        item
        for item in result.checks
        if item["reason_code"] == "dk_fund_tax_review"
        and "yearly_danish_tax_note" in item["details"]
    )
    assert residency_check["details"]["new_indian_money_allowed"] is False
    assert residency_check["details"]["yearly_danish_tax_note"]["verified"] is False
    redirect = next(
        action for action in result.actions if action["reason_code"] == "dk_fund_tax_review"
    )
    assert redirect["kind"] == "REDIRECT_SIP"
    assert redirect["member_id"] == SPOUSE
    assert redirect["source_member_id"] == "member-dk"


def test_projection_flags_now_after_five_months_and_not_when_balanced() -> None:
    current_breach = _balanced_holdings("35", "15")
    current = _check(
        current_breach,
        transactions=_transactions("mix-2"),
    )
    assert current.early_warning == {
        "bucket": "small",
        "months_until": 0,
        "direction": "over",
        "verified": False,
        "source_note": "Conservative monthly planning assumptions; not verified return forecasts.",
    }
    assert any(action["kind"] == "REDIRECT_SIP" for action in current.actions)
    assert not any(action["kind"].startswith("SELL") for action in current.actions)
    assert all(action["member_id"] == MEMBER for action in current.actions)

    near_limit = _balanced_holdings("17.5", "17.5")
    five_month = _check(
        near_limit,
        transactions=_transactions("mix-2", amounts=("2500", "2500", "2500")),
    )
    assert five_month.early_warning is not None
    assert five_month.early_warning["bucket"] == "small"
    assert five_month.early_warning["months_until"] == 5

    balanced = _check(_balanced_holdings("15", "20"))
    assert balanced.early_warning is None


def test_synthetic_household_fund_check_worked_example() -> None:
    spouse_holdings: list[FundHolding] = []
    references: list[FundReference] = []
    for suffix in ("alpha", "beta"):
        family = suffix.title()
        regular = _holding(
            f"ri-regular-{suffix}",
            name=f"Synthetic {family} Fund Regular Plan Growth",
            sub_category="Large Cap",
            value="500000",
            member_id=SPOUSE,
            plan_type="regular",
            ter="1.50",
            tax_lots=(
                Lot(
                    f"synthetic-{suffix}-lot",
                    "equity_mf",
                    Decimal("500"),
                    Decimal("600"),
                    date(2024, 8, 1),
                    market_price=Decimal("1000"),
                ),
            ),
            complete=True,
            price="1000",
            exit_load_pct="0",
        )
        direct = _reference(
            f"ri-direct-{suffix}",
            name=f"Synthetic {family} Fund Direct Plan Growth",
            sub_category="Large Cap",
            plan_type="direct",
            ter="0.50",
        )
        spouse_holdings.append(regular)
        references.extend((regular, direct))

    for security_id, category, value in (
        ("ri-small", "Small Cap", "583333"),
        ("ri-mid", "Mid Cap", "666667"),
        ("ri-flexi", "Flexi Cap", "583333"),
        ("ri-sector", "Sectoral/Thematic", "333333"),
        ("ri-global", "FoF Overseas", "166667"),
        ("ri-hybrid", "Aggressive Hybrid", "444444"),
        ("ri-debt", "Liquid", "666667"),
    ):
        tax_lots: tuple[Lot, ...] = ()
        complete = False
        price = None
        exit_load_pct = None
        if category in {"Sectoral/Thematic", "FoF Overseas"}:
            quantity = Decimal(value) / Decimal(100)
            tax_class = "intl_fof" if category == "FoF Overseas" else "equity_mf"
            tax_lots = (
                Lot(
                    f"synthetic-{security_id}-lot",
                    tax_class,
                    quantity,
                    Decimal("50"),
                    date(2024, 8, 1),
                    market_price=Decimal("100"),
                ),
            )
            complete = True
            price = "100"
            exit_load_pct = "0"
        spouse_holdings.append(
            _holding(
                security_id,
                sub_category=category,
                value=value,
                member_id=SPOUSE,
                category=(
                    "mf:debt"
                    if category == "Liquid"
                    else "mf:equity_global"
                    if category == "FoF Overseas"
                    else "mf:equity_india"
                ),
                tax_lots=tax_lots,
                complete=complete,
                price=price,
                exit_load_pct=exit_load_pct,
            )
        )
    denmark_holdings = [
        _holding("dk-one", member_id=MEMBER, value="150000"),
        _holding("dk-two", sub_category="Mid Cap", member_id=MEMBER, value="100000"),
    ]
    holdings = [*spouse_holdings, *denmark_holdings]
    transactions = [
        *_transactions(
            "ri-small",
            amounts=("11000", "11000", "11000"),
            member_id=SPOUSE,
        ),
        *_transactions("dk-one", member_id=MEMBER),
        *_transactions("dk-two", amounts=("3000", "3000", "3000"), member_id=MEMBER),
    ]
    members = [
        FundMember(SPOUSE),
        FundMember(
            MEMBER,
            residency_country="DK",
            residency_status="NRI",
            redirect_member_id=SPOUSE,
            redirect_member_name="RI spouse",
            clubbing_note="Confirm how redirected monthly investments are taxed.",
        ),
    ]
    family = _check(holdings, references=references, transactions=transactions, members=members)
    spouse = check_funds(
        holdings=holdings,
        references=references,
        histories={},
        transactions=transactions,
        members=members,
        as_of=AS_OF,
        risk_profile="growth",
        selected_member_id=SPOUSE,
    )
    denmark = check_funds(
        holdings=holdings,
        references=references,
        histories={},
        transactions=transactions,
        members=members,
        as_of=AS_OF,
        risk_profile="growth",
        selected_member_id=MEMBER,
    )

    assert len([item for item in spouse.details["funds"] if item["member_id"] == SPOUSE]) == 9
    assert sum(item["plan_type"] == "regular" for item in spouse.details["funds"]) == 2
    regular_checks = [item for item in spouse.checks if item["reason_code"] == "regular_plan"]
    assert len(regular_checks) == 1
    assert regular_checks[0]["details"]["yearly_saving_inr"] == "5000.00"
    assert regular_checks[0]["details"]["tax_inr"] == "9750.00"
    assert regular_checks[0]["details"]["ltcg_exemption_used_inr"] == "125000.00"
    assert regular_checks[0]["details"]["ltcg_exemption_remaining_inr"] == "0.00"
    assert {item["kind"] for item in spouse.actions} == {
        "CONSOLIDATE_FUNDS",
        "REDIRECT_SIP",
        "SWITCH_TO_DIRECT",
    }
    consolidation = next(item for item in spouse.actions if item["kind"] == "CONSOLIDATE_FUNDS")
    assert consolidation["amount"] == "500000.00"
    assert consolidation["tax_inr"] == "32500.00"
    assert consolidation["ltcg_exemption_used_inr"] == "0.00"
    assert consolidation["ltcg_exemption_remaining_inr"] == "0.00"
    assert spouse.early_warning is not None
    assert spouse.early_warning["bucket"] == "small"
    assert spouse.early_warning["months_until"] == 5
    assert len([item for item in denmark.details["funds"] if item["member_id"] == MEMBER]) == 2
    assert {item["kind"] for item in denmark.actions} == {"REDIRECT_SIP"}
    assert {item["amount"] for item in denmark.actions} == {"3000.00", "5000.00"}
    assert all(item["member_id"] == SPOUSE for item in denmark.actions)
    assert all(item["source_member_id"] == MEMBER for item in denmark.actions)
    assert family.status == "attention"


def test_sip_redirect_is_a_cash_action_with_no_tax_sale() -> None:
    result = _check(
        _balanced_holdings("35", "15"),
        transactions=_transactions("mix-2"),
    )
    redirects = [action for action in result.actions if action["kind"] == "REDIRECT_SIP"]

    assert redirects
    assert redirects[0]["amount"] == "5000.00"
    assert "tax_inr" not in redirects[0]
    assert not any(action["kind"].startswith("SELL") for action in result.actions)


def test_fund_rebalance_reuses_cash_sip_then_tax_aware_sale_order() -> None:
    target = {
        "equity_india": Decimal("5500000"),
        "equity_global": Decimal(0),
        "debt": Decimal("3500000"),
        "gold": Decimal("1000000"),
        "cash": Decimal(0),
    }
    current = {
        "equity_india": Decimal("6500000"),
        "equity_global": Decimal(0),
        "debt": Decimal("2500000"),
        "gold": Decimal("1000000"),
        "cash": Decimal(0),
    }
    plan = AllocationPlan(
        target_rupees=target,
        target_percent={key: value * 100 / Decimal("10000000") for key, value in target.items()},
        current_rupees=current,
        current_percent={key: value * 100 / Decimal("10000000") for key, value in current.items()},
        gap={key: target[key] - current[key] for key in target},
        buckets=(),
        investable_total=Decimal("10000000"),
    )
    foundation = FoundationReport(
        tuple(
            FoundationItem(name, "ok", Decimal(0), Decimal(0), Decimal(0))
            for name in ("high_interest_debt", "emergency_fund", "term_cover", "health_cover")
        )
    )
    week_input = WeekInput(
        plan=plan,
        foundation=foundation,
        new_money=Decimal("250000"),
        monthly_sip_capacity=Decimal("10000"),
        existing_sips=[
            Sip(
                sleeve="equity_india",
                instrument_category="mutual_fund",
                monthly_amount=Decimal("10000"),
            )
        ],
        route=RouteDecision(
            route="A",
            vehicle="in_international_index_fund",
            reasons=[],
            warnings=[],
            alternatives=[],
        ),
        debt_picks=[
            DebtPick(
                instrument="sweep_fd",
                amount=Decimal("1000"),
                after_tax_yield_pct=Decimal("4"),
                reason_codes=["instant"],
            )
        ],
        satellite_pct=Decimal("20"),
        week="2026-W45",
    )
    lot = Lot(
        "synthetic-fund-lot",
        "equity_mf",
        Decimal("5000"),
        Decimal("60"),
        date(2024, 1, 1),
        market_price=Decimal("100"),
    )

    result = ordered_fund_rebalance_plan(
        FundRebalanceContext(
            breaches=band_breaches(plan),
            week_input=week_input,
            lots_by_sleeve={"equity_india": [(lot, Decimal("100"))]},
            sale_date=AS_OF,
            tax_context=TaxContext(),
        )
    )

    assert result.cash_actions
    assert result.sip_redirect is not None
    assert len(result.sells) == 1
    assert result.sells[0].gross_proceeds == Decimal("350000.00")
    assert result.tax_estimate is not None
    assert result.tax_estimate.exemption_used == Decimal("125000.00")
    assert result.tax_estimate.total_tax == Decimal("1950.00")
