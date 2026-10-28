from datetime import date
from decimal import Decimal

import pytest

from wealth.policy.exits import HoldingFacts, build_exit_card, load_exit_rules
from wealth.policy.tax import Lot, TaxClass, TaxContext


def _lot(
    quantity: str = "100",
    *,
    cost: str = "1000",
    acquired: date = date(2024, 1, 1),
    lot_id: str = "lot-1",
    tax_class: TaxClass = "listed_equity",
) -> Lot:
    return Lot(
        lot_id=lot_id,
        tax_class=tax_class,
        quantity=Decimal(quantity),
        cost_per_unit=Decimal(cost),
        acquired_on=acquired,
    )


def _facts(**overrides: object) -> HoldingFacts:
    values: dict[str, object] = {
        "security_id": "security-synthetic",
        "symbol": "SYN",
        "sector": "industrials",
        "is_core": False,
        "lots": [_lot()],
        "price": Decimal("1000"),
        "weight_pct_of_equity": Decimal("4"),
        "target_weight_pct": Decimal("4"),
    }
    values.update(overrides)
    return HoldingFacts(**values)  # type: ignore[arg-type]


TODAY = date(2026, 1, 10)


def test_exit_rules_match_appendix_defaults_and_watch_fraction() -> None:
    rules = load_exit_rules()

    assert rules["watch_fraction"] == "0.8"
    assert rules["satellite"]["thesis_break"]["revenue_yoy_negative_quarters"] == 2
    assert rules["satellite"]["governance_break"]["pledge_jump_pp"] == "10"
    assert rules["satellite"]["geopolitical_break"]["min_independent_sources"] == 2
    assert rules["satellite"]["valuation_stretch"]["own_band_percentile"] == "90"
    assert rules["satellite"]["risk_stop"]["drawdown_from_cost_pct"] == "25"
    assert rules["core"]["stop_loss"] is False


def test_governance_exit_is_critical_and_never_waits_for_long_term() -> None:
    facts = _facts(
        lots=[_lot(quantity="10", cost="500", acquired=date(2025, 1, 22))],
        governance_events=["AUDITOR_RESIGNATION"],
        pe_own_band_percentile=Decimal("99"),
        pe_vs_sector_multiple=Decimal("2"),
    )

    card = build_exit_card(facts, TODAY, TaxContext())

    assert card.action == "EXIT"
    assert card.urgency == "CRITICAL"
    assert card.quantity == Decimal("10")
    assert "wait_until" not in card.params


def test_two_consecutive_recent_revenue_declines_break_the_thesis() -> None:
    card = build_exit_card(
        _facts(revenue_yoy_pct_last_quarters=[Decimal("-2"), Decimal("-1"), Decimal("5")]),
        TODAY,
        TaxContext(),
    )

    assert card.triggers["thesis_break"].status == "triggered"
    assert card.action == "EXIT"
    assert card.urgency == "HIGH"


def test_one_recent_revenue_decline_is_watch_without_an_exit() -> None:
    card = build_exit_card(
        _facts(revenue_yoy_pct_last_quarters=[Decimal("-2"), Decimal("3")]),
        TODAY,
        TaxContext(),
    )

    assert card.triggers["thesis_break"].status == "watch"
    assert card.action == "KEEP"


def test_valuation_stretch_trims_to_target_quantity() -> None:
    facts = _facts(
        lots=[_lot(quantity="140")],
        weight_pct_of_equity=Decimal("7.0"),
        target_weight_pct=Decimal("5.0"),
        pe_own_band_percentile=Decimal("93"),
        pe_vs_sector_multiple=Decimal("1.6"),
    )

    card = build_exit_card(facts, TODAY, TaxContext())

    assert card.triggers["valuation_stretch"].status == "triggered"
    assert card.action == "TRIM_TO_TARGET"
    assert card.quantity == Decimal("40")


def test_nonurgent_trim_waits_until_selected_lot_turns_long_term() -> None:
    facts = _facts(
        lots=[_lot(quantity="100", cost="500", acquired=date(2025, 1, 21))],
        weight_pct_of_equity=Decimal("7"),
        target_weight_pct=Decimal("5"),
        pe_own_band_percentile=Decimal("93"),
        pe_vs_sector_multiple=Decimal("1.6"),
    )

    card = build_exit_card(facts, TODAY, TaxContext())

    assert card.action == "WAIT_FOR_LONG_TERM"
    assert card.params["wait_until"] == date(2026, 1, 22)
    assert card.params["tax_saved"] > Decimal("0")
    assert card.triggers["tax_wait"].status == "triggered"


def test_geopolitical_hit_needs_two_evidence_ids_to_trigger() -> None:
    one_source = build_exit_card(
        _facts(geo_revenue_hit_pct=Decimal("25"), geo_evidence_ids=["evidence-a"]),
        TODAY,
        TaxContext(),
    )
    two_sources = build_exit_card(
        _facts(geo_revenue_hit_pct=Decimal("25"), geo_evidence_ids=["evidence-a", "evidence-b"]),
        TODAY,
        TaxContext(),
    )

    assert one_source.triggers["geopolitical_break"].status == "watch"
    assert one_source.action == "KEEP"
    assert two_sources.action == "REVIEW_THEN_EXIT"


def test_risk_stop_requires_drawdown_dma_and_watch_thesis() -> None:
    risk_facts = _facts(
        lots=[_lot(quantity="10", cost="1000")],
        price=Decimal("700"),
        below_200dma=True,
        thesis_status="watch",
    )
    on_track_facts = _facts(
        lots=[_lot(quantity="10", cost="1000")],
        price=Decimal("700"),
        below_200dma=True,
        thesis_status="on_track",
    )

    risk_card = build_exit_card(risk_facts, TODAY, TaxContext())
    on_track_card = build_exit_card(on_track_facts, TODAY, TaxContext())

    assert risk_card.triggers["risk_stop"].status == "triggered"
    assert risk_card.action == "REVIEW_THEN_EXIT"
    assert on_track_card.action == "KEEP"


def test_missing_satellite_facts_leave_every_trigger_unknown() -> None:
    facts = _facts(
        lots=[],
        weight_pct_of_equity=None,
        target_weight_pct=None,
    )

    card = build_exit_card(facts, TODAY, TaxContext())

    assert all(trigger.status == "unknown" for trigger in card.triggers.values())
    assert card.action == "KEEP"


def test_core_holding_ignores_drawdown_and_broken_thesis() -> None:
    card = build_exit_card(
        _facts(
            is_core=True,
            lots=[_lot(quantity="10", cost="1000")],
            price=Decimal("600"),
            thesis_status="broken",
        ),
        TODAY,
        TaxContext(),
    )

    assert card.action == "KEEP"
    assert all(trigger.status == "ok" for key, trigger in card.triggers.items() if key != "drift")


def test_core_drift_rebalances_to_target_without_a_stop_loss() -> None:
    card = build_exit_card(
        _facts(
            is_core=True,
            lots=[_lot(quantity="160")],
            weight_pct_of_equity=Decimal("8"),
            target_weight_pct=Decimal("5"),
            price=Decimal("1000"),
        ),
        TODAY,
        TaxContext(),
    )

    assert card.action == "TRIM_TO_TARGET"
    assert card.quantity == Decimal("60")


@pytest.mark.parametrize(
    ("cost", "expected_action", "expected_tax"),
    [
        ("871.79485", "SWAP", "8000.00"),
        ("759.61535", "KEEP", "15000.00"),
    ],
)
def test_swap_must_clear_estimated_tax_cost(
    cost: str, expected_action: str, expected_tax: str
) -> None:
    card = build_exit_card(
        _facts(
            lots=[
                _lot(
                    quantity="200",
                    cost=cost,
                    acquired=date(2024, 3, 1),
                    tax_class="foreign_equity",
                )
            ],
            price=Decimal("1000"),
            better_candidate={
                "security_id": "candidate-synthetic",
                "z_advantage": Decimal("1.3"),
                "weeks_ahead": 5,
                "expected_edge_pct": Decimal("6"),
            },
        ),
        TODAY,
        TaxContext(),
    )

    assert card.tax_estimate.total_tax == Decimal(expected_tax)
    assert card.action == expected_action
    if expected_action == "KEEP":
        assert "swap_not_worth_tax" in card.reason_codes


def test_governance_has_priority_over_thesis_break() -> None:
    card = build_exit_card(
        _facts(
            governance_events=["RESTATEMENT"],
            revenue_yoy_pct_last_quarters=[Decimal("-2"), Decimal("-1")],
        ),
        TODAY,
        TaxContext(),
    )

    assert card.action == "EXIT"
    assert card.urgency == "CRITICAL"
    assert card.reason_codes[0] == "governance_break"


def test_goal_approach_recommends_portfolio_level_glide_without_selling_shares() -> None:
    card = build_exit_card(_facts(goal_within_years=2), TODAY, TaxContext())

    assert card.action == "GLIDE_TO_DEBT"
    assert card.quantity == Decimal("0")
    assert card.tax_estimate.total_tax == Decimal("0")
