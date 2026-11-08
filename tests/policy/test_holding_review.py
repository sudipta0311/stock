from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

from wealth.policy.exits import HoldingFacts, build_exit_card
from wealth.policy.holding_review import (
    StockCheckFacts,
    holding_review_decision,
    thesis_status_from_revenue,
)
from wealth.policy.personalise import HoldingInput, HoldingLotInput, _holding_review_tax
from wealth.policy.tax import Lot, TaxContext

_TODAY = date(2026, 1, 10)


def _lot(*, quantity: str = "100", cost: str = "1000", acquired: date = date(2024, 1, 1)) -> Lot:
    return Lot(
        lot_id="synthetic-lot",
        tax_class="listed_equity",
        quantity=Decimal(quantity),
        cost_per_unit=Decimal(cost),
        acquired_on=acquired,
    )


def _facts(**changes: object) -> HoldingFacts:
    values: dict[str, object] = {
        "security_id": "synthetic-security",
        "symbol": "SYN",
        "sector": "industrials",
        "is_core": False,
        "lots": [_lot()],
        "price": Decimal("1000"),
        "weight_pct_of_equity": Decimal("4"),
        "target_weight_pct": Decimal("4"),
    }
    values.update(changes)
    return HoldingFacts(**values)  # type: ignore[arg-type]


def _check(**changes: object) -> StockCheckFacts:
    values: dict[str, object] = {
        "security_id": "synthetic-security",
        "week": "2026-W02",
        "as_of": _TODAY,
        "market": "IN",
        "quant_decile": 9,
        "pe_percentile": Decimal("60"),
        "pe_vs_sector": Decimal("1.1"),
        "price": Decimal("1000"),
    }
    values.update(changes)
    return StockCheckFacts(**values)  # type: ignore[arg-type]


def _decision(
    facts: HoldingFacts,
    check: StockCheckFacts,
    *,
    room: Decimal | None = Decimal("5000"),
    minimum: Decimal = Decimal("1000"),
    market: str = "IN",
):
    card = build_exit_card(facts, _TODAY, TaxContext())
    result = holding_review_decision(
        market=market,
        as_of=_TODAY,
        check=check,
        card=card,
        facts=facts,
        add_more_room=room,
        minimum_buy=minimum,
        position_size_known=facts.weight_pct_of_equity is not None,
    )
    return card, result


def test_fresh_strong_share_with_room_is_add_more() -> None:
    card, result = _decision(
        _facts(weight_pct_of_equity=Decimal("2"), target_weight_pct=Decimal("4")),
        _check(),
    )

    assert card.action == "KEEP"
    assert result.verdict == "add_more"
    assert result.reasons[0]["code"] == "add_more"
    assert "goal_timing" not in {row["key"] for row in result.checks}


def test_sector_multiple_can_support_add_more_without_own_history() -> None:
    _card, result = _decision(
        _facts(weight_pct_of_equity=Decimal("2"), target_weight_pct=Decimal("4")),
        _check(pe_percentile=None, pe_vs_sector=Decimal("1.1")),
    )

    assert result.verdict == "add_more"


def test_high_sector_multiple_keeps_share_without_own_history() -> None:
    _card, result = _decision(
        _facts(weight_pct_of_equity=Decimal("2"), target_weight_pct=Decimal("4")),
        _check(pe_percentile=None, pe_vs_sector=Decimal("1.3")),
    )

    assert result.verdict == "keep"


def test_sector_only_valuation_watch_uses_its_own_reason() -> None:
    _card, result = _decision(
        _facts(
            weight_pct_of_equity=Decimal("2"),
            target_weight_pct=Decimal("4"),
            pe_vs_sector_multiple=Decimal("1.6"),
        ),
        _check(pe_percentile=None, pe_vs_sector=Decimal("1.6")),
    )

    assert result.reasons[0]["code"] == "valuation_vs_sector"


def test_company_strength_uses_the_research_decile_alone() -> None:
    expected = ((9, "ok"), (5, "ok"), (2, "watch"), (None, "unknown"))

    for decile, status in expected:
        _card, result = _decision(_facts(), _check(quant_decile=decile))
        company_strength = next(
            row["status"] for row in result.checks if row["key"] == "company_strength"
        )
        assert company_strength == status


def test_weak_company_strength_adds_a_reason() -> None:
    _card, result = _decision(
        _facts(weight_pct_of_equity=Decimal("2"), target_weight_pct=Decimal("4")),
        _check(quant_decile=2),
    )

    assert result.reasons == ({"code": "company_strength_weak", "params": {}},)


def test_two_recent_revenue_drops_are_a_sell_verdict() -> None:
    revenue = (Decimal("-4"), Decimal("-2"))
    assert thesis_status_from_revenue(revenue) == "broken"
    card, result = _decision(
        _facts(
            revenue_yoy_pct_last_quarters=list(revenue),
            thesis_status=thesis_status_from_revenue(revenue),
        ),
        _check(revenue_yoy_pct=revenue),
    )

    assert card.action == "EXIT"
    assert result.verdict == "sell"
    assert result.reasons[0] == {"code": "revenue_down_quarters", "params": {"count": 2}}


def test_valuation_and_large_position_can_sell_some_while_tax_waits() -> None:
    facts = _facts(
        lots=[_lot(quantity="140", cost="500", acquired=date(2025, 1, 21))],
        weight_pct_of_equity=Decimal("7"),
        target_weight_pct=Decimal("5"),
        pe_own_band_percentile=Decimal("95"),
        pe_vs_sector_multiple=Decimal("1.6"),
    )
    card, result = _decision(
        facts,
        _check(pe_percentile=Decimal("95"), pe_vs_sector=Decimal("1.6")),
        room=Decimal(0),
    )

    assert card.action == "WAIT_FOR_LONG_TERM"
    assert card.params["wait_until"] == date(2026, 1, 22)
    assert result.verdict == "sell_some"


def test_one_revenue_drop_plus_price_risk_means_think_about_selling() -> None:
    revenue = (Decimal("-2"),)
    facts = _facts(
        lots=[_lot(quantity="10", cost="1000")],
        price=Decimal("700"),
        below_200dma=True,
        revenue_yoy_pct_last_quarters=list(revenue),
        thesis_status="watch",
    )
    card, result = _decision(
        facts,
        _check(price=Decimal("700"), below_dma200=True, revenue_yoy_pct=revenue),
        room=Decimal(0),
    )

    assert card.action == "REVIEW_THEN_EXIT"
    assert result.verdict == "review"
    assert {item["code"] for item in result.reasons} >= {
        "revenue_down_quarters",
        "risk_stop",
    }


def test_governance_event_is_a_critical_sell() -> None:
    facts = _facts(governance_events=["AUDITOR_RESIGNATION"])
    card, result = _decision(
        facts,
        _check(),
    )

    assert card.urgency == "CRITICAL"
    assert result.verdict == "sell"


def test_stale_or_european_holding_is_not_checked_and_cannot_sell() -> None:
    card, stale = _decision(
        _facts(governance_events=["AUDITOR_RESIGNATION"]),
        _check(as_of=_TODAY - timedelta(days=15)),
    )
    _, european = _decision(_facts(), _check(market="EU"), market="EU")

    assert card.action == "EXIT"
    assert stale.verdict == "not_checked"
    assert european.verdict == "not_checked"
    assert stale.checks == (
        {"key": "latest_price", "status": "ok"},
        {"key": "position_size", "status": "ok"},
    )
    assert european.checks == stale.checks


def test_share_savings_account_tax_never_includes_a_waiting_date() -> None:
    holding = HoldingInput(
        security_id="synthetic-security",
        symbol="SYN",
        market="US",
        currency="USD",
        security_type="equity",
        sleeve="equity_global",
        category="industrials",
        quantity=Decimal("10"),
        price=Decimal("100"),
        account_tax_wrapper="ask",
    )
    owner = SimpleNamespace(residency=SimpleNamespace(country="DK"))

    tax = _holding_review_tax(holding, owner, None)  # type: ignore[arg-type]

    assert tax == {"note_code": "ask_same_tax"}
    assert "wait_until" not in tax


def test_missing_buying_price_does_not_create_a_drawdown_sell() -> None:
    holding = HoldingInput(
        security_id="synthetic-security",
        symbol="SYN",
        market="IN",
        currency="INR",
        security_type="equity",
        sleeve="equity_india",
        category="industrials",
        quantity=Decimal("10"),
        price=Decimal("700"),
        lots=(
            HoldingLotInput(
                lot_id="synthetic-lot",
                quantity=Decimal("10"),
                cost_per_unit=None,
                acquired_on=date(2025, 1, 1),
            ),
        ),
    )
    facts = _facts(lots=[], price=Decimal("700"), below_200dma=True)
    card, result = _decision(
        facts,
        _check(price=Decimal("700"), below_dma200=True),
        room=Decimal(0),
    )
    tax = _holding_review_tax(
        holding,
        SimpleNamespace(residency=SimpleNamespace(country="IN")),
        card,  # type: ignore[arg-type]
    )

    assert card.action == "KEEP"
    assert result.verdict == "keep"
    assert tax == {"note_code": "tax_not_estimated"}
