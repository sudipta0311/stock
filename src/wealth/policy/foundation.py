"""Foundation checks for debt, reserves, and insurance cover."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from wealth.core.money import round_paise
from wealth.policy.rules import AllocationRules, decimal_rule, integer_rule, resolve_rules
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import HealthInsuranceAsset, LoanAsset, TermInsuranceAsset
from wealth.schemas.common import LiquidityClass, Sleeve
from wealth.schemas.profile import FinancialContext

FoundationStatus = Literal["ok", "gap", "unknown"]


@dataclass(frozen=True)
class FoundationItem:
    """One quantified foundation check."""

    key: str
    status: FoundationStatus
    required: Decimal | None
    have: Decimal | None
    gap: Decimal | None


@dataclass(frozen=True)
class FoundationReport:
    """The four deterministic foundation checks in display order."""

    items: tuple[FoundationItem, ...]

    def __getitem__(self, key: str) -> FoundationItem:
        """Look up a check by its stable key."""
        for item in self.items:
            if item.key == key:
                return item
        raise KeyError(key)

    @property
    def by_key(self) -> dict[str, FoundationItem]:
        """Return the report as a key-indexed dictionary."""
        return {item.key: item for item in self.items}


def foundation_check(
    snapshot: WealthSnapshot,
    ctx: FinancialContext,
    rules: AllocationRules | None = None,
    *,
    emergency_months_override: int | None = None,
) -> FoundationReport:
    """Compare high-interest debt, liquid reserves, and cover with rule targets."""
    selected_rules = resolve_rules(rules)
    foundation = selected_rules["foundation"]
    high_interest_threshold = decimal_rule(
        foundation["high_interest_debt_rate_pct"], "foundation.high_interest_debt_rate_pct"
    )
    high_interest_debt = round_paise(
        sum(
            (
                asset.outstanding
                for asset, _ in snapshot.valued_assets
                if isinstance(asset, LoanAsset) and asset.rate_pct >= high_interest_threshold
            ),
            Decimal(0),
        )
    )
    debt_item = FoundationItem(
        key="high_interest_debt",
        status="gap" if high_interest_debt > 0 else "ok",
        required=high_interest_debt,
        have=Decimal("0.00"),
        gap=high_interest_debt,
    )

    emergency_months = (
        integer_rule(emergency_months_override, "emergency_months_override")
        if emergency_months_override is not None
        else integer_rule(
            foundation["emergency_months"][ctx.employment_type],
            f"foundation.emergency_months.{ctx.employment_type}",
        )
    )
    emergency_required = round_paise(ctx.monthly_expenses * emergency_months)
    emergency_have = round_paise(
        snapshot.amount_for(Sleeve.CASH, LiquidityClass.INSTANT)
        + snapshot.amount_for(Sleeve.DEBT, LiquidityClass.INSTANT)
    )
    emergency_gap = max(Decimal("0.00"), emergency_required - emergency_have)
    emergency_item = FoundationItem(
        key="emergency_fund",
        status="ok" if emergency_gap == 0 else "gap",
        required=emergency_required,
        have=emergency_have,
        gap=emergency_gap,
    )

    term_covers = round_paise(
        sum(
            (
                asset.cover
                for asset, _ in snapshot.valued_assets
                if isinstance(asset, TermInsuranceAsset)
            ),
            Decimal(0),
        )
    )
    if ctx.dependants == 0:
        term_required: Decimal | None = Decimal("0.00")
        term_gap: Decimal | None = Decimal("0.00")
        term_status: FoundationStatus = "ok"
    elif ctx.annual_income is None:
        term_required = None
        term_gap = None
        term_status = "unknown"
    else:
        target_multiple = integer_rule(
            foundation["term_cover_multiple_target"], "foundation.term_cover_multiple_target"
        )
        required_before_floor = (
            Decimal(target_multiple) * ctx.annual_income
            + snapshot.liabilities_total
            - _liquid_cash_and_debt(snapshot)
        )
        term_required = round_paise(max(Decimal(0), required_before_floor))
        term_gap = max(Decimal("0.00"), term_required - term_covers)
        term_status = "ok" if term_gap == 0 else "gap"
    term_item = FoundationItem(
        key="term_cover",
        status=term_status,
        required=term_required,
        have=term_covers,
        gap=term_gap,
    )

    health_rules = foundation["health_cover_min"]
    tier = (
        ctx.city_tier
        if ctx.city_tier is not None
        else integer_rule(foundation["default_city_tier"], "foundation.default_city_tier")
    )
    tier_key = f"tier{tier}"
    health_required = decimal_rule(
        health_rules[tier_key], f"foundation.health_cover_min.{tier_key}"
    )
    health_covers = round_paise(
        sum(
            (
                asset.cover + (asset.top_up_cover or Decimal(0))
                for asset, _ in snapshot.valued_assets
                if isinstance(asset, HealthInsuranceAsset)
            ),
            Decimal(0),
        )
    )
    health_required = round_paise(health_required)
    health_gap = max(Decimal("0.00"), health_required - health_covers)
    health_item = FoundationItem(
        key="health_cover",
        status="ok" if health_gap == 0 else "gap",
        required=health_required,
        have=health_covers,
        gap=health_gap,
    )
    return FoundationReport((debt_item, emergency_item, term_item, health_item))


def _liquid_cash_and_debt(snapshot: WealthSnapshot) -> Decimal:
    return round_paise(
        snapshot.amount_for(Sleeve.CASH, LiquidityClass.INSTANT)
        + snapshot.amount_for(Sleeve.DEBT, LiquidityClass.INSTANT)
    )
