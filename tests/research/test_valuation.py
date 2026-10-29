from __future__ import annotations

from decimal import Decimal

from wealth.research.factors import FundamentalFactors
from wealth.research.rules import ResearchRules
from wealth.research.valuation import calculate_valuation_stats


def _factors(
    earnings_yield: str | None,
    fcf_yield: str | None,
) -> FundamentalFactors:
    return FundamentalFactors(
        roce=None,
        roe=None,
        revenue_yoy=None,
        earnings_yield=Decimal(earnings_yield) if earnings_yield is not None else None,
        fcf_yield=Decimal(fcf_yield) if fcf_yield is not None else None,
        debt_to_equity=None,
        accruals_ratio=None,
        pe_percentile_own_10y=None,
    )


def test_code_verdict_uses_sector_yield_and_reverse_dcf_math() -> None:
    result = calculate_valuation_stats(
        security_id="synthetic-security",
        market="IN",
        price=Decimal("12.5"),
        factors=_factors("0.05", "0.04"),
        sector_earnings_yields=[Decimal("0.04")],
        historical_pe=[],
        rules=ResearchRules.load(),
    )

    assert result.verdict == "CHEAP"
    assert result.earnings_yield_vs_sector == Decimal("1.25")
    assert result.reverse_dcf_implied_growth_pct == Decimal("8.00")
    assert result.discount_rate_pct == Decimal("12")


def test_own_history_p_e_uses_three_year_monthly_sample() -> None:
    result = calculate_valuation_stats(
        security_id="synthetic-security",
        market="US",
        price=Decimal("20"),
        factors=_factors("0.05", "0.02"),
        sector_earnings_yields=[],
        historical_pe=[Decimal("30")] * 36,
        rules=ResearchRules.load(),
    )

    assert result.pe_percentile == Decimal("0")
    assert result.verdict == "CHEAP"
    assert result.discount_rate_pct == Decimal("9")


def test_missing_valuation_inputs_stay_unknown() -> None:
    result = calculate_valuation_stats(
        security_id="synthetic-security",
        market="IN",
        price=None,
        factors=_factors(None, None),
        sector_earnings_yields=[],
        historical_pe=[],
        rules=ResearchRules.load(),
    )

    assert result.verdict == "UNKNOWN"
    assert result.reverse_dcf_implied_growth_pct is None
