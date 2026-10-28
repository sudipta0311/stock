from decimal import Decimal

from wealth.policy.routes import RouteContext, select_route


def _context(**overrides: object) -> RouteContext:
    values: dict[str, object] = {
        "monthly_global_new_money": "10000",
        "has_lrs_broker": False,
        "prefers_no_paperwork": False,
        "us_situs_holdings_usd": Decimal(0),
        "lrs_used_this_fy": "0",
    }
    values.update(overrides)
    return RouteContext(**values)


def test_small_global_sip_prefers_route_a_and_can_fall_back_to_route_b() -> None:
    """10,000 per month selects A; when subscriptions pause, a 0.4% ETF premium selects B."""
    assert select_route(_context()).route == "A"
    suspended = select_route(
        _context(amc_international_suspended=True, etf_inav_premium_pct=Decimal("0.4"))
    )
    assert suspended.route == "B"


def test_high_etf_premium_without_broker_waits_for_an_open_route() -> None:
    """A 2.5% iNAV premium exceeds the 1% route B limit; without an LRS broker the result waits."""
    decision = select_route(
        _context(amc_international_suspended=True, etf_inav_premium_pct=Decimal("2.5"))
    )
    assert decision.route == "WAIT"
    assert "no_open_route" in decision.reasons


def test_route_c_selects_ucits_at_estate_tax_threshold_and_warns_near_it() -> None:
    """Worked example: 600,000 annual investment with a broker selects route C.

    65,000 USD in US-situs holdings switches to UCITS; 50,000 USD triggers a warning.
    """
    route_c = select_route(_context(annual_global_new_money="600000", has_lrs_broker=True))
    ucits = select_route(
        _context(
            annual_global_new_money="600000",
            has_lrs_broker=True,
            us_situs_holdings_usd=Decimal("65000"),
        )
    )
    near = select_route(
        _context(
            annual_global_new_money="600000",
            has_lrs_broker=True,
            us_situs_holdings_usd=Decimal("50000"),
        )
    )

    assert route_c.route == "C" and route_c.vehicle == "us_etf"
    assert ucits.route == "C" and ucits.vehicle == "ucits_etf"
    assert "schedule_fa_reporting" in route_c.warnings
    assert "estate_tax_us_domiciled_assets" in route_c.warnings
    assert "us_estate_tax_threshold_near" in near.warnings
    assert "estate_tax_us_domiciled_assets" in near.warnings


def test_route_decision_reports_tcs_excess_and_amount() -> None:
    """Worked example: 600,000 used plus 600,000 new exceeds the 1,000,000 limit by 200,000.

    The 20% TCS is 40,000.
    """
    decision = select_route(
        _context(
            annual_global_new_money="600000",
            lrs_used_this_fy="600000",
            has_lrs_broker=True,
        )
    )
    assert "tcs_applies" in decision.warnings
    assert decision.warning_params["tcs_applies"] == {
        "excess_inr": Decimal("200000.00"),
        "tcs_inr": Decimal("40000"),
    }
