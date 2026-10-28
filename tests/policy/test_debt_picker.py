from collections.abc import Callable
from decimal import Decimal

from wealth.policy.debt_picker import DebtNeed, load_debt_rules, pick_debt
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import PPFAsset
from wealth.schemas.common import LiquidityClass
from wealth.schemas.profile import FinancialContext
from wealth.valuation.base import Valuation


def _need(
    horizon: str,
    amount: str = "100000",
    liquidity_need: LiquidityClass = LiquidityClass.LOCKED,
) -> DebtNeed:
    return DebtNeed(
        amount=amount,
        horizon_years=Decimal(horizon),
        liquidity_need=liquidity_need,
    )


def _yields() -> dict[str, Decimal]:
    return {
        "bank_fd": Decimal("7.0"),
        "arbitrage_fund": Decimal("6.5"),
        "liquid_fund": Decimal("6.0"),
        "ppf": Decimal("7.1"),
        "scss": Decimal("8.2"),
    }


def _snapshot(
    ppf_contribution: str | None = None, valuation: Valuation | None = None
) -> WealthSnapshot:
    if ppf_contribution is None:
        return WealthSnapshot()
    asset = PPFAsset(
        asset_type="ppf",
        balance="100000",
        opened_fy_start_year=2020,
        yearly_contribution=ppf_contribution,
    )
    assert valuation is not None
    return WealthSnapshot(valued_assets=[(asset, valuation)])


def test_one_year_after_tax_yield_ranking_changes_with_tax_slab() -> None:
    """Worked example: at 30%, 6.5% arbitrage nets 5.20%, ahead of a 7.0% FD at 4.90%.

    At 10%, the FD nets 6.30%, ahead of arbitrage at 5.20%.
    """
    high_tax = pick_debt(
        _need("1", "200000", LiquidityClass.WITHIN_1Y),
        FinancialContext(monthly_expenses="0"),
        Decimal("30"),
        40,
        _yields(),
        WealthSnapshot(),
    )
    low_tax = pick_debt(
        _need("1", "200000", LiquidityClass.WITHIN_1Y),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        40,
        _yields(),
        WealthSnapshot(),
    )

    assert high_tax[0].instrument == "arbitrage_fund"
    assert high_tax[0].after_tax_yield_pct == Decimal("5.20")
    high_tax_fd = next(pick for pick in high_tax if pick.instrument == "bank_fd")
    assert high_tax_fd.after_tax_yield_pct == Decimal("4.90")
    assert high_tax_fd.reason_params["suggested_tenure"]["years"] == Decimal("1")
    assert low_tax[0].instrument == "bank_fd"
    assert low_tax[0].after_tax_yield_pct == Decimal("6.30")
    low_tax_arbitrage = next(pick for pick in low_tax if pick.instrument == "arbitrage_fund")
    assert low_tax_arbitrage.after_tax_yield_pct == Decimal("5.20")


def test_one_year_horizon_excludes_long_lock_in_products() -> None:
    """A one-year, within-1y need cannot select PPF, NSC, SCSS, or RBI FRSB lock-ins."""
    picks = pick_debt(
        _need("1", "200000", LiquidityClass.WITHIN_1Y),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        60,
        {
            "bank_fd": Decimal("7.0"),
            "arbitrage_fund": Decimal("6.5"),
            "liquid_fund": Decimal("6.0"),
            "ppf": Decimal("7.1"),
            "nsc": Decimal("7.0"),
            "scss": Decimal("8.2"),
            "rbi_frsb": Decimal("7.0"),
        },
        WealthSnapshot(),
    )

    selected = {pick.instrument for pick in picks}
    assert not selected.intersection({"ppf", "nsc", "scss", "rbi_frsb"})


def test_five_year_locked_need_can_use_nsc_and_five_year_post_office_td() -> None:
    """A five-year locked need admits NSC and the five-year Post Office Time Deposit."""
    picks = pick_debt(
        _need("5"),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        40,
        {"nsc": Decimal("7.0"), "po_td_5y": Decimal("7.1")},
        WealthSnapshot(),
    )

    assert {pick.instrument for pick in picks} == {"nsc", "po_td_5y"}


def test_emergency_horizon_only_returns_instant_access_instruments() -> None:
    """At horizon zero and instant liquidity, only the liquid fund and sweep FD are eligible."""
    yields = {item["yield_key"]: Decimal("5.0") for item in load_debt_rules()["instruments"]}
    picks = pick_debt(
        _need("0", "200000", LiquidityClass.INSTANT),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        40,
        yields,
        WealthSnapshot(),
    )

    assert {pick.instrument for pick in picks} == {"liquid_fund", "sweep_fd"}


def test_flexible_fd_tenure_is_capped_at_ten_years() -> None:
    """A 15-year need can use a flexible FD with a suggested tenure capped at 10 years."""
    picks = pick_debt(
        _need("15"),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        40,
        {"bank_fd": Decimal("7.0")},
        WealthSnapshot(),
    )

    assert picks[0].instrument == "bank_fd"
    assert picks[0].reason_params["suggested_tenure"]["years"] == Decimal("10")


def test_flexible_fd_requires_seven_days_before_the_horizon() -> None:
    """A 6.935-day horizon is too short for an FD, while a 7.3-day horizon qualifies."""
    too_short = pick_debt(
        _need("0.019", liquidity_need=LiquidityClass.WITHIN_1Y),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        40,
        {"bank_fd": Decimal("7.0")},
        WealthSnapshot(),
    )
    long_enough = pick_debt(
        _need("0.02", liquidity_need=LiquidityClass.WITHIN_1Y),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        40,
        {"bank_fd": Decimal("7.0")},
        WealthSnapshot(),
    )

    assert all(pick.instrument != "bank_fd" for pick in too_short)
    assert long_enough[0].instrument == "bank_fd"


def test_ppf_ranks_first_when_long_term_and_contribution_room_is_available() -> None:
    """Worked example: over 20 years PPF yields 7.10% exempt.

    That ranks ahead of SCSS at 5.74% after a 30% slab tax.
    """
    picks = pick_debt(
        _need("20"),
        FinancialContext(monthly_expenses="0", tax_regime="old"),
        Decimal("30"),
        60,
        _yields(),
        WealthSnapshot(),
    )
    ppf = picks[0]
    assert ppf.instrument == "ppf"
    assert ppf.after_tax_yield_pct == Decimal("7.10")
    assert "80c_deduction_available" in ppf.reason_codes


def test_ppf_is_excluded_when_current_fy_room_is_zero(
    make_valuation: Callable[[str], Valuation],
) -> None:
    """A full 150,000 PPF contribution leaves zero of its 150,000 annual room, so PPF is omitted."""
    picks = pick_debt(
        _need("20"),
        FinancialContext(monthly_expenses="0"),
        Decimal("30"),
        60,
        _yields(),
        _snapshot("150000", make_valuation("100000")),
    )
    assert all(pick.instrument != "ppf" for pick in picks)


def test_scss_minimum_age_and_fixed_deposit_bank_split() -> None:
    """Worked example: SCSS is excluded at 59 and eligible at 60.

    A 1,200,000 bank FD needs three banks for 500,000 cover per bank.
    """
    age_59 = pick_debt(
        _need("5"),
        FinancialContext(monthly_expenses="0"),
        Decimal("30"),
        59,
        _yields(),
        WealthSnapshot(),
    )
    age_60 = pick_debt(
        _need("5"),
        FinancialContext(monthly_expenses="0"),
        Decimal("30"),
        60,
        _yields(),
        WealthSnapshot(),
    )
    fd_pick = pick_debt(
        _need("1", "1200000"),
        FinancialContext(monthly_expenses="0"),
        Decimal("10"),
        40,
        {"bank_fd": Decimal("7.0")},
        WealthSnapshot(),
    )[0]

    assert all(pick.instrument != "scss" for pick in age_59)
    assert any(pick.instrument == "scss" for pick in age_60)
    assert "split_across_banks" in fd_pick.reason_codes
    assert fd_pick.reason_params["split_across_banks"]["bank_count"] == 3
