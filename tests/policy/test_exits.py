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
    assert rules["satellite"]["no_reason_to_hold"]["max_quant_decile"] == 3
    assert "no_reason_to_hold" in rules["satellite"]["tax_wait"]["applies_to"]
    assert rules["satellite"]["risk_stop"]["drawdown_from_cost_pct"] == "25"
    assert "requires_thesis" not in rules["satellite"]["risk_stop"]
    assert rules["core"]["stop_loss"] is False


def test_weak_research_and_high_sector_multiple_exits() -> None:
    card = build_exit_card(
        _facts(quant_decile=2, pe_vs_sector_multiple=Decimal("1.6")), TODAY, TaxContext()
    )

    assert card.action == "EXIT"
    assert card.urgency == "MEDIUM"
    assert card.reason_codes == ("no_reason_to_hold",)
    assert card.triggers["no_reason_to_hold"].status == "triggered"


def test_weak_research_and_large_loss_below_average_exits() -> None:
    card = build_exit_card(
        _facts(
            quant_decile=3,
            lots=[],
            quantity_held=Decimal("12"),
            average_cost=Decimal("100"),
            price=Decimal("78"),
            below_200dma=True,
        ),
        TODAY,
        TaxContext(),
    )

    assert card.action == "EXIT"
    assert card.triggers["no_reason_to_hold"].status == "triggered"
    assert card.quantity == Decimal("12")


def test_weak_share_exit_precedes_risk_stop_when_both_fire() -> None:
    card = build_exit_card(
        _facts(
            quant_decile=2,
            pe_vs_sector_multiple=Decimal("1.6"),
            price=Decimal("700"),
            below_200dma=True,
        ),
        TODAY,
        TaxContext(),
    )

    assert card.triggers["no_reason_to_hold"].status == "triggered"
    assert card.triggers["risk_stop"].status == "triggered"
    assert card.action == "EXIT"
    assert card.urgency == "MEDIUM"
    assert card.reason_codes == ("no_reason_to_hold",)


def test_weak_research_without_expense_or_falling_loss_keeps() -> None:
    card = build_exit_card(
        _facts(
            quant_decile=2,
            pe_vs_sector_multiple=Decimal("1.2"),
            average_cost=Decimal("1000"),
            price=Decimal("1100"),
        ),
        TODAY,
        TaxContext(),
    )

    assert card.action == "KEEP"
    assert card.triggers["no_reason_to_hold"].status == "ok"


def test_strong_research_and_missing_decile_do_not_trigger_weak_share_exit() -> None:
    strong = build_exit_card(
        _facts(quant_decile=6, pe_vs_sector_multiple=Decimal("2.0")), TODAY, TaxContext()
    )
    unknown = build_exit_card(
        _facts(quant_decile=None, pe_vs_sector_multiple=Decimal("2.0")), TODAY, TaxContext()
    )

    assert strong.triggers["no_reason_to_hold"].status == "ok"
    assert unknown.triggers["no_reason_to_hold"].status == "unknown"
    assert strong.action == unknown.action == "KEEP"


def test_governance_exit_keeps_priority_over_weak_share_exit() -> None:
    card = build_exit_card(
        _facts(
            quant_decile=2,
            pe_vs_sector_multiple=Decimal("2.0"),
            governance_events=["RESTATEMENT"],
        ),
        TODAY,
        TaxContext(),
    )

    assert card.action == "EXIT"
    assert card.urgency == "CRITICAL"
    assert card.reason_codes == ("governance_break",)


def test_core_holding_does_not_apply_weak_share_exit() -> None:
    card = build_exit_card(
        _facts(is_core=True, quant_decile=2, pe_vs_sector_multiple=Decimal("2.0")),
        TODAY,
        TaxContext(),
    )

    assert card.triggers["no_reason_to_hold"].status == "ok"
    assert card.triggers["no_reason_to_hold"].threshold == "not applicable to core"
    assert card.action == "KEEP"


def test_weak_share_exit_can_wait_for_long_term_tax_saving() -> None:
    card = build_exit_card(
        _facts(
            lots=[_lot(quantity="10", cost="1000", acquired=date(2025, 1, 29))],
            quant_decile=2,
            pe_vs_sector_multiple=Decimal("1.6"),
            price=Decimal("1200"),
        ),
        TODAY,
        TaxContext(),
    )

    assert card.action == "WAIT_FOR_LONG_TERM"
    assert card.params["wait_until"] == date(2026, 1, 30)
    assert card.params["tax_saved"] > Decimal("0")


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


def test_risk_stop_uses_drawdown_and_dma_without_thesis_warning() -> None:
    risk_facts = _facts(
        lots=[_lot(quantity="10", cost="1000")],
        price=Decimal("700"),
        below_200dma=True,
        thesis_status="on_track",
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
    assert on_track_card.action == "REVIEW_THEN_EXIT"
    assert on_track_card.triggers["risk_stop"].status == "triggered"
    assert "thesis" not in on_track_card.triggers["risk_stop"].threshold


def test_risk_stop_uses_average_cost_only_when_no_lots_are_available() -> None:
    average_cost_only = _facts(
        lots=[],
        average_cost=Decimal("1000"),
        price=Decimal("700"),
        below_200dma=True,
        thesis_status="watch",
    )
    lot_cost_takes_precedence = _facts(
        lots=[_lot(quantity="10", cost="1000")],
        average_cost=Decimal("500"),
        price=Decimal("700"),
        below_200dma=True,
        thesis_status="watch",
    )

    average_cost_card = build_exit_card(average_cost_only, TODAY, TaxContext())
    lot_cost_card = build_exit_card(lot_cost_takes_precedence, TODAY, TaxContext())

    assert average_cost_card.triggers["risk_stop"].status == "triggered"
    assert lot_cost_card.triggers["risk_stop"].status == "triggered"


def test_risk_stop_watches_at_twenty_one_percent_loss_without_thesis_data() -> None:
    card = build_exit_card(
        _facts(
            lots=[],
            average_cost=Decimal("1000"),
            price=Decimal("790"),
            below_200dma=True,
            thesis_status="unknown",
        ),
        TODAY,
        TaxContext(),
    )

    assert card.triggers["risk_stop"].status == "watch"
    assert card.triggers["risk_stop"].value == "-21.00% from cost; below 200-DMA=True"
    assert "thesis" not in card.triggers["risk_stop"].threshold


def test_risk_stop_is_unknown_when_a_deep_loss_has_no_dma_status() -> None:
    card = build_exit_card(
        _facts(price=Decimal("700"), thesis_status="on_track"), TODAY, TaxContext()
    )

    assert card.triggers["risk_stop"].status == "unknown"


@pytest.mark.parametrize("decile", [0, 11, True, 2.0])
def test_quant_decile_rejects_values_outside_integer_range(decile: object) -> None:
    with pytest.raises(ValueError, match="quant_decile"):
        _facts(quant_decile=decile)


def test_known_risk_inputs_that_rule_out_a_watch_are_ok() -> None:
    cases = (
        _facts(price=Decimal("950")),
        _facts(price=Decimal("950"), below_200dma=True),
        _facts(price=Decimal("700"), below_200dma=False),
        _facts(price=Decimal("700"), below_200dma=False, thesis_status="on_track"),
    )

    assert [
        build_exit_card(facts, TODAY, TaxContext()).triggers["risk_stop"].status for facts in cases
    ] == ["ok", "ok", "ok", "ok"]


@pytest.mark.parametrize(("multiple", "expected"), [("1.6", "watch"), ("1.1", "ok")])
def test_sector_only_valuation_can_watch_but_never_triggers(multiple: str, expected: str) -> None:
    card = build_exit_card(_facts(pe_vs_sector_multiple=Decimal(multiple)), TODAY, TaxContext())

    assert card.triggers["valuation_stretch"].status == expected
    assert card.triggers["valuation_stretch"].status != "triggered"
    assert card.action == "KEEP"


def test_governance_is_ok_after_a_recent_quiet_watch() -> None:
    watched = build_exit_card(_facts(events_watched=True), TODAY, TaxContext())
    not_watched = build_exit_card(_facts(), TODAY, TaxContext())

    assert watched.triggers["governance_break"].status == "ok"
    assert not_watched.triggers["governance_break"].status == "unknown"


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
