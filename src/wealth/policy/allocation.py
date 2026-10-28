"""Build deterministic goal buckets and whole-wealth target allocations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Literal

from wealth.core.money import round_paise
from wealth.policy.rules import (
    AllocationRules,
    decimal_rule,
    integer_rule,
    resolve_rules,
)
from wealth.policy.snapshot import POLICY_SLEEVES, PolicySleeve, WealthSnapshot
from wealth.schemas.profile import FinancialContext, Goal

BucketKind = Literal["emergency", "short_goal", "medium_goal", "long_goal", "long_term"]
EquityRegion = Literal["equity_india", "equity_global"]
PlanSleeve = PolicySleeve | EquityRegion


@dataclass(frozen=True)
class AllocationBucket:
    """One reserve, with the sleeve split used to produce its target."""

    key: str
    kind: BucketKind
    amount: Decimal
    target_by_sleeve: Mapping[PolicySleeve, Decimal]
    goal_name: str | None = None
    goal_year: int | None = None
    horizon_years: int | None = None
    profile_band: str | None = None
    goal_currency: str | None = None
    goal_target_rupees: Decimal | None = None
    fx_usd_inr: Decimal | None = None
    fx_rate_to_inr: Decimal | None = None


@dataclass(frozen=True)
class AllocationPlan:
    """Target/current amounts and percentages for the policy sleeves."""

    target_rupees: Mapping[PlanSleeve, Decimal]
    target_percent: Mapping[PlanSleeve, Decimal]
    current_rupees: Mapping[PlanSleeve, Decimal]
    current_percent: Mapping[PlanSleeve, Decimal]
    gap: Mapping[PlanSleeve, Decimal]
    buckets: tuple[AllocationBucket, ...]
    investable_total: Decimal
    fx_usd_inr: Decimal | None = None
    current_equity_rupees: Mapping[EquityRegion, Decimal] | None = None
    new_money_only: bool = False
    notes: tuple[str, ...] = ()

    @property
    def target_by_sleeve(self) -> Mapping[PlanSleeve, Decimal]:
        """Alias for the target rupee amounts, keyed by policy sleeve."""
        return self.target_rupees

    @property
    def target_pct(self) -> Mapping[PlanSleeve, Decimal]:
        """Alias for target percentages expressed from 0 to 100."""
        return self.target_percent

    @property
    def current_by_sleeve(self) -> Mapping[PlanSleeve, Decimal]:
        """Alias for current rupee amounts, keyed by policy sleeve."""
        return self.current_rupees

    @property
    def current_pct(self) -> Mapping[PlanSleeve, Decimal]:
        """Alias for current percentages expressed from 0 to 100."""
        return self.current_percent

    @property
    def gaps(self) -> Mapping[PlanSleeve, Decimal]:
        """Alias for target minus current rupees for each sleeve."""
        return self.gap


def build_targets(
    snapshot: WealthSnapshot,
    ctx: FinancialContext,
    goals: Sequence[Goal],
    profile_band: str,
    as_of: date,
    rules: AllocationRules | None = None,
    *,
    fx_usd_inr: Decimal | None = None,
    fx_rates: Mapping[str, Decimal] | None = None,
    emergency_months_override: int | None = None,
) -> AllocationPlan:
    """Reserve wealth by goal horizon, converting foreign targets with supplied FX rates."""
    selected_rules = resolve_rules(rules)
    selected_fx = dict(fx_rates or {})
    if fx_usd_inr is not None:
        selected_fx.setdefault("USD/INR", fx_usd_inr)
    if fx_usd_inr is not None and (
        not isinstance(fx_usd_inr, Decimal) or not fx_usd_inr.is_finite() or fx_usd_inr <= 0
    ):
        raise ValueError("fx_usd_inr must be a positive finite Decimal")
    for pair, rate in selected_fx.items():
        if not isinstance(rate, Decimal) or not rate.is_finite() or rate <= 0:
            raise ValueError(f"{pair} must be a positive finite Decimal")
    missing_fx = tuple(
        dict.fromkeys(
            f"goal_fx_missing:{goal.currency}"
            for goal in goals
            if goal.currency != "INR" and f"{goal.currency}/INR" not in selected_fx
        )
    )
    effective_fx_usd_inr = (
        selected_fx.get("USD/INR") if any(goal.currency == "USD" for goal in goals) else None
    )
    if not isinstance(as_of, date):
        raise ValueError("as_of must be a date")
    profile_band = getattr(profile_band, "value", profile_band)
    profiles = selected_rules["profiles"]
    if profile_band not in profiles:
        raise ValueError(f"unknown profile band: {profile_band}")

    scale = decimal_rule(selected_rules["percent_scale"], "percent_scale")
    percent_display_places = integer_rule(
        selected_rules["allocation"]["percent_display_places"],
        "allocation.percent_display_places",
    )
    profile_mix = _read_mix(profiles[profile_band], f"profiles.{profile_band}", scale)
    emergency_mix = _read_mix(
        selected_rules["allocation"]["emergency_mix"], "allocation.emergency_mix", scale
    )
    short_mix = _read_mix(selected_rules["horizon"]["short_mix"], "horizon.short_mix", scale)
    short_years = integer_rule(selected_rules["horizon"]["short_years"], "horizon.short_years")
    medium_years = integer_rule(selected_rules["horizon"]["medium_years"], "horizon.medium_years")
    full_years = integer_rule(selected_rules["glide"]["full_years"], "glide.full_years")
    start_years = integer_rule(selected_rules["glide"]["start_years"], "glide.start_years")
    factor_bounds = selected_rules["glide"]["factor_bounds"]
    factor_minimum = decimal_rule(factor_bounds["minimum"], "glide.factor_bounds.minimum")
    factor_maximum = decimal_rule(factor_bounds["maximum"], "glide.factor_bounds.maximum")
    if start_years <= full_years:
        raise ValueError("glide.start_years must be greater than glide.full_years")
    if factor_minimum > factor_maximum:
        raise ValueError("glide factor minimum must not exceed its maximum")

    target = {sleeve: Decimal("0.00") for sleeve in POLICY_SLEEVES}
    remaining = snapshot.investable_total
    buckets: list[AllocationBucket] = []
    months = (
        integer_rule(emergency_months_override, "emergency_months_override")
        if emergency_months_override is not None
        else integer_rule(
            selected_rules["foundation"]["emergency_months"][ctx.employment_type],
            f"foundation.emergency_months.{ctx.employment_type}",
        )
    )
    emergency_required = round_paise(ctx.monthly_expenses * months)
    emergency_amount = min(emergency_required, remaining)
    emergency_split = _split_amount(emergency_amount, emergency_mix, scale)
    _add_split(target, emergency_split)
    remaining = round_paise(remaining - emergency_amount)
    buckets.append(
        AllocationBucket(
            key="emergency_fund",
            kind="emergency",
            amount=emergency_amount,
            target_by_sleeve=_freeze(emergency_split),
        )
    )

    def reserve_goals(
        selected_goals: Sequence[Goal],
        kind: Literal["short_goal", "medium_goal", "long_goal"],
    ) -> None:
        nonlocal remaining
        for goal in sorted(
            selected_goals, key=lambda item: (item.priority, item.target_year, item.name)
        ):
            horizon_years = goal.target_year - as_of.year
            pair = f"{goal.currency}/INR"
            if goal.currency != "INR" and pair not in selected_fx:
                continue
            goal_target_rupees = _goal_target_rupees(goal, selected_fx.get(pair))
            amount = min(goal_target_rupees, remaining)
            if kind == "short_goal":
                mix = short_mix
            elif kind == "medium_goal":
                capped_band = _profile_capped_at_moderate(profile_band, selected_rules)
                base_mix = _read_mix(profiles[capped_band], f"profiles.{capped_band}", scale)
                mix = _glide_mix(
                    base_mix,
                    horizon_years,
                    full_years,
                    start_years,
                    factor_minimum,
                    factor_maximum,
                )
            else:
                mix = profile_mix
            if kind == "short_goal":
                bucket_profile_band = None
            elif kind == "medium_goal":
                bucket_profile_band = capped_band
            else:
                bucket_profile_band = profile_band
            split = _split_amount(amount, mix, scale)
            _add_split(target, split)
            remaining = round_paise(remaining - amount)
            buckets.append(
                AllocationBucket(
                    key=f"{kind}:{goal.name}",
                    kind=kind,
                    amount=amount,
                    target_by_sleeve=_freeze(split),
                    goal_name=goal.name,
                    goal_year=goal.target_year,
                    horizon_years=horizon_years,
                    profile_band=bucket_profile_band,
                    goal_currency=goal.currency,
                    goal_target_rupees=goal_target_rupees,
                    fx_usd_inr=(selected_fx.get(pair) if goal.currency == "USD" else None),
                    fx_rate_to_inr=(selected_fx.get(pair) if goal.currency != "INR" else None),
                )
            )

    short_goals = [goal for goal in goals if goal.target_year - as_of.year < short_years]
    medium_goals = [
        goal for goal in goals if short_years <= goal.target_year - as_of.year <= medium_years
    ]
    long_goals = [goal for goal in goals if goal.target_year - as_of.year > medium_years]
    reserve_goals(short_goals, "short_goal")
    reserve_goals(medium_goals, "medium_goal")
    reserve_goals(long_goals, "long_goal")

    if remaining > 0:
        remainder_split = _split_amount(remaining, profile_mix, scale)
        _add_split(target, remainder_split)
        buckets.append(
            AllocationBucket(
                key="long_term_remainder",
                kind="long_term",
                amount=remaining,
                target_by_sleeve=_freeze(remainder_split),
                profile_band=profile_band,
            )
        )

    current = {
        sleeve: round_paise(snapshot.investable_totals_by_sleeve[sleeve])
        for sleeve in POLICY_SLEEVES
    }
    target = {sleeve: round_paise(target[sleeve]) for sleeve in POLICY_SLEEVES}
    gaps = {sleeve: round_paise(target[sleeve] - current[sleeve]) for sleeve in POLICY_SLEEVES}
    return AllocationPlan(
        target_rupees=_freeze(target),
        target_percent=_percentages(
            target, snapshot.investable_total, scale, percent_display_places
        ),
        current_rupees=_freeze(current),
        current_percent=_percentages(
            current, snapshot.investable_total, scale, percent_display_places
        ),
        gap=_freeze(gaps),
        buckets=tuple(buckets),
        investable_total=snapshot.investable_total,
        fx_usd_inr=effective_fx_usd_inr,
        current_equity_rupees=MappingProxyType(
            {
                "equity_india": round_paise(snapshot.totals_by_sleeve["equity_india"]),
                "equity_global": round_paise(snapshot.totals_by_sleeve["equity_global"]),
            }
        ),
        notes=missing_fx,
    )


def _goal_target_rupees(goal: Goal, fx_to_inr: Decimal | None) -> Decimal:
    if goal.currency != "INR":
        if fx_to_inr is None:
            raise ValueError(f"{goal.currency}/INR FX rate is required for goal conversion")
        return round_paise(goal.target_amount * fx_to_inr)
    return goal.target_amount


def _profile_capped_at_moderate(profile_band: str, rules: AllocationRules) -> str:
    bands = rules["risk"]["bands"]
    if profile_band not in bands or "moderate" not in bands:
        raise ValueError("risk bands must include the selected profile and moderate")
    return (
        profile_band
        if integer_rule(bands[profile_band], f"risk.bands.{profile_band}")
        <= integer_rule(bands["moderate"], "risk.bands.moderate")
        else "moderate"
    )


def _read_mix(
    raw_mix: Mapping[str, object], name: str, scale: Decimal
) -> dict[PolicySleeve, Decimal]:
    mix = {sleeve: decimal_rule(raw_mix[sleeve], f"{name}.{sleeve}") for sleeve in POLICY_SLEEVES}
    if sum(mix.values(), Decimal(0)) != scale:
        raise ValueError(f"{name} must sum to {scale}")
    return mix


def _glide_mix(
    base_mix: Mapping[PolicySleeve, Decimal],
    horizon_years: int,
    full_years: int,
    start_years: int,
    factor_minimum: Decimal,
    factor_maximum: Decimal,
) -> dict[PolicySleeve, Decimal]:
    factor = Decimal(horizon_years - full_years) / Decimal(start_years - full_years)
    factor = min(factor_maximum, max(factor_minimum, factor))
    equity = base_mix["equity"] * factor
    return {
        "equity": equity,
        "debt": base_mix["debt"] + base_mix["equity"] - equity,
        "gold": base_mix["gold"],
        "cash": base_mix["cash"],
    }


def _split_amount(
    amount: Decimal,
    mix: Mapping[PolicySleeve, Decimal],
    scale: Decimal,
) -> dict[PolicySleeve, Decimal]:
    if amount == 0:
        return {sleeve: Decimal("0.00") for sleeve in POLICY_SLEEVES}
    residual_sleeve = max(POLICY_SLEEVES, key=mix.__getitem__)
    split: dict[PolicySleeve, Decimal] = {}
    allocated = Decimal(0)
    for sleeve in POLICY_SLEEVES:
        if sleeve == residual_sleeve:
            continue
        part = round_paise(amount * mix[sleeve] / scale)
        split[sleeve] = part
        allocated += part
    split[residual_sleeve] = round_paise(amount - allocated)
    return split


def _add_split(total: dict[PolicySleeve, Decimal], split: Mapping[PolicySleeve, Decimal]) -> None:
    for sleeve in POLICY_SLEEVES:
        total[sleeve] += split[sleeve]


def _percentages(
    amounts: Mapping[PolicySleeve, Decimal],
    total: Decimal,
    scale: Decimal,
    display_places: int,
) -> Mapping[PolicySleeve, Decimal]:
    if total <= 0:
        return _freeze({sleeve: Decimal("0.00") for sleeve in POLICY_SLEEVES})
    precision = Decimal(1).scaleb(-display_places)
    percentages = {
        sleeve: (amounts[sleeve] * scale / total).quantize(precision, rounding=ROUND_HALF_UP)
        for sleeve in POLICY_SLEEVES
    }
    residual_sleeve = max(POLICY_SLEEVES, key=amounts.__getitem__)
    percentages[residual_sleeve] = scale - sum(
        (percentages[sleeve] for sleeve in POLICY_SLEEVES if sleeve != residual_sleeve), Decimal(0)
    )
    return _freeze(percentages)


def _freeze(values: Mapping[PolicySleeve, Decimal]) -> Mapping[PolicySleeve, Decimal]:
    return MappingProxyType(dict(values))
