"""Compare traditional life-policy outcomes without making a surrender recommendation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

from wealth.core.money import round_paise
from wealth.schemas.assets import TraditionalPolicyAsset
from wealth.valuation.returns import policy_irr


@dataclass(frozen=True)
class RateComparison:
    """A policy return compared with one supplied benchmark rate."""

    benchmark_pct: Decimal
    difference_pct: Decimal | None
    policy_outperforms: bool | None


@dataclass(frozen=True)
class PolicyReview:
    """Numbers for continue and paid-up choices plus an insurer-provided value."""

    irr_pct: Decimal | None
    continue_value: Decimal | None
    paid_up_value: Decimal
    surrender_value: Decimal | None
    comparison: Mapping[str, RateComparison]
    reason_codes: tuple[str, ...]
    recommendation_kind: Literal["compare_options"] = "compare_options"


def review_policy(
    policy: TraditionalPolicyAsset,
    as_of: date,
    compare_rates: Mapping[str, Decimal],
) -> PolicyReview:
    """Calculate illustration IRR and paid-up value, then compare with PPF and FD rates."""
    if not isinstance(policy, TraditionalPolicyAsset):
        raise ValueError("policy must be a TraditionalPolicyAsset")
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    if not isinstance(compare_rates, Mapping) or not {"ppf", "fd"} <= set(compare_rates):
        raise ValueError("compare_rates must contain ppf and fd percentages")
    checked_rates: dict[str, Decimal] = {}
    for key in ("ppf", "fd"):
        value = compare_rates[key]
        if (
            not isinstance(value, Decimal)
            or not value.is_finite()
            or not Decimal(0) <= value <= 100
        ):
            raise ValueError(f"compare_rates.{key} must be a Decimal percentage from 0 to 100")
        checked_rates[key] = value

    premiums_payable = min(policy.premium_paying_term or policy.term_years, policy.term_years)
    premiums_paid = min(max(0, as_of.year - policy.start_year), premiums_payable)
    bonus = policy.bonus_accrued or Decimal(0)
    paid_up_value = round_paise(
        policy.sum_assured * Decimal(premiums_paid) / Decimal(premiums_payable) + bonus
    )
    xirr_fraction = policy_irr(policy)
    irr_pct = None if xirr_fraction is None else xirr_fraction * Decimal(100)

    comparison: dict[str, RateComparison] = {}
    for key, rate in checked_rates.items():
        difference = None if irr_pct is None else irr_pct - rate
        comparison[key] = RateComparison(
            benchmark_pct=rate,
            difference_pct=difference,
            policy_outperforms=None if difference is None else difference > 0,
        )

    reasons: list[str] = []
    if policy.surrender_value is None:
        reasons.append("ask_insurer_for_surrender_value")
    if irr_pct is not None:
        if irr_pct < checked_rates["ppf"]:
            reasons.append("low_return_vs_ppf")
        if irr_pct < checked_rates["fd"]:
            reasons.append("low_return_vs_fd")
    if policy.sum_assured < Decimal(10) * policy.annual_premium:
        reasons.append("cover_low_vs_premium")

    return PolicyReview(
        irr_pct=irr_pct,
        continue_value=policy.maturity_illustration,
        paid_up_value=paid_up_value,
        surrender_value=policy.surrender_value,
        comparison=MappingProxyType(comparison),
        reason_codes=tuple(reasons),
    )
