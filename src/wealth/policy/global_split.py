"""Split equity targets between India and global holdings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from wealth.core.money import round_paise
from wealth.policy.allocation import AllocationPlan, PlanSleeve
from wealth.policy.rules import decimal_rule, integer_rule
from wealth.schemas.profile import Goal, Residency

GlobalSplitRules = Mapping[str, Any]

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "global.yaml"


def _as_decimal(value: object) -> Decimal:
    return decimal_rule(value, "tilt")


_STRICT_DECIMAL = Annotated[Decimal, BeforeValidator(_as_decimal)]


class GeoTilts(BaseModel):
    """Evidence-backed bounded global and gold allocation tilts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    global_pp: _STRICT_DECIMAL = Decimal(0)
    gold_pp: _STRICT_DECIMAL = Decimal(0)
    evidence_ids: list[str] = Field(default_factory=list)
    as_of: date

    @model_validator(mode="after")
    def validate_evidence_and_bounds(self) -> GeoTilts:
        bounds = load_global_rules()["tilt_bounds"]
        global_bound = decimal_rule(bounds["global_pp"], "tilt_bounds.global_pp")
        gold_bound = decimal_rule(bounds["gold_pp"], "tilt_bounds.gold_pp")
        if abs(self.global_pp) > global_bound:
            raise ValueError("global_pp exceeds the configured tilt bound")
        if abs(self.gold_pp) > gold_bound:
            raise ValueError("gold_pp exceeds the configured tilt bound")
        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids must be non-empty text")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")
        if (self.global_pp != 0 or self.gold_pp != 0) and len(self.evidence_ids) < 2:
            raise ValueError("non-zero tilts require at least two evidence_ids")
        return self


def load_global_rules() -> dict[str, Any]:
    """Load the checked-in global allocation rules."""
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        document = yaml.safe_load(rules_file)
    if not isinstance(document, dict):
        raise ValueError("global rules must contain a YAML mapping")
    validate_global_rules(document)
    return document


def validate_global_rules(rules: Mapping[str, Any]) -> None:
    """Validate the global share rules and their bounded adjustments."""
    if integer_rule(rules.get("version"), "version") != 1:
        raise ValueError("unsupported global rules version")
    base = rules.get("base_global_share_of_equity_pct")
    profiles = ("conservative", "moderate", "growth", "aggressive")
    if not isinstance(base, Mapping) or not set(profiles).issubset(base):
        raise ValueError("base global shares must define all profile bands")
    for profile in profiles:
        value = decimal_rule(base[profile], f"base_global_share_of_equity_pct.{profile}")
        if not Decimal(0) <= value <= Decimal(60):
            raise ValueError("base global shares must be between 0 and 60")
    bonuses = rules.get("foreign_goal_bonus")
    if not isinstance(bonuses, list):
        raise ValueError("foreign_goal_bonus must be a list")
    for bonus in bonuses:
        if not isinstance(bonus, Mapping):
            raise ValueError("each foreign goal bonus must be a mapping")
        minimum = decimal_rule(bonus.get("min_share_pct"), "foreign_goal_bonus.min_share_pct")
        bonus_pp = decimal_rule(bonus.get("bonus_pp"), "foreign_goal_bonus.bonus_pp")
        if minimum < 0 or bonus_pp < 0:
            raise ValueError("foreign goal thresholds and bonuses must be non-negative")
    bounds = rules.get("tilt_bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("tilt_bounds must be a mapping")
    for key in ("global_pp", "gold_pp"):
        if decimal_rule(bounds.get(key), f"tilt_bounds.{key}") < 0:
            raise ValueError("tilt bounds must be non-negative")
    residency = rules.get("residency", {})
    if residency:
        share_table_names = {
            name
            for name, table in residency.items()
            if name.startswith("india_share_")
            and name != "india_share_by_country"
            and isinstance(table, Mapping)
        }
        if "india_share_abroad_pct" not in share_table_names:
            raise ValueError("residency.india_share_abroad_pct must be defined")
        for table_name in share_table_names:
            table = residency[table_name]
            if not isinstance(table, Mapping) or not set(profiles).issubset(table):
                raise ValueError(f"residency.{table_name} must define all profile bands")
            for profile in profiles:
                value = decimal_rule(table[profile], f"residency.{table_name}.{profile}")
                if not Decimal(0) <= value <= Decimal(100):
                    raise ValueError(f"residency.{table_name}.{profile} must be 0 to 100")
        country_tables = residency.get("india_share_by_country", {})
        if not isinstance(country_tables, Mapping):
            raise ValueError("residency.india_share_by_country must be a mapping")
        if any(table_name not in share_table_names for table_name in country_tables.values()):
            raise ValueError("residency country shares must reference a configured share table")


def split_equity(
    plan: AllocationPlan,
    goals: Sequence[Goal],
    profile_band: str,
    tilts: GeoTilts | None = None,
    rules: GlobalSplitRules | None = None,
    *,
    residency: Residency | None = None,
    fx_rates: Mapping[str, Decimal] | None = None,
    as_of: date | None = None,
) -> AllocationPlan:
    """Return an allocation plan with India/global equity sleeves and bounded tilts."""
    selected_rules = load_global_rules() if rules is None else rules
    validate_global_rules(selected_rules)
    profile_band = getattr(profile_band, "value", profile_band)
    base_shares = selected_rules["base_global_share_of_equity_pct"]
    if profile_band not in base_shares:
        raise ValueError(f"unknown profile band: {profile_band}")

    selected_residency = residency or Residency()
    rates = dict(fx_rates or {})
    if plan.fx_usd_inr is not None:
        rates.setdefault("USD/INR", plan.fx_usd_inr)
    foreign_rupees = Decimal("0.00")
    for goal in goals:
        if goal.goal_type == "education_abroad" or goal.currency != "INR":
            amount = goal.target_amount
            if goal.currency != "INR":
                fx_rate = rates.get(f"{goal.currency}/INR")
                if fx_rate is None:
                    continue
                amount = round_paise(amount * fx_rate)
            foreign_rupees += amount
    foreign_share_pct = (
        foreign_rupees * Decimal(100) / plan.investable_total
        if plan.investable_total > 0
        else Decimal(0)
    )
    goal_bonus = Decimal(0)
    for bonus in sorted(
        selected_rules["foreign_goal_bonus"],
        key=lambda item: decimal_rule(item["min_share_pct"], "foreign_goal_bonus.min_share_pct"),
        reverse=True,
    ):
        minimum = decimal_rule(bonus["min_share_pct"], "foreign_goal_bonus.min_share_pct")
        if foreign_share_pct >= minimum:
            goal_bonus = decimal_rule(bonus["bonus_pp"], "foreign_goal_bonus.bonus_pp")
            break

    global_tilt = tilts.global_pp if tilts is not None else Decimal(0)
    gold_tilt = tilts.gold_pp if tilts is not None else Decimal(0)
    bounds = selected_rules["tilt_bounds"]
    if abs(global_tilt) > decimal_rule(bounds["global_pp"], "tilt_bounds.global_pp"):
        raise ValueError("global tilt exceeds the configured rule bound")
    if abs(gold_tilt) > decimal_rule(bounds["gold_pp"], "tilt_bounds.gold_pp"):
        raise ValueError("gold tilt exceeds the configured rule bound")

    base_global = decimal_rule(
        base_shares[profile_band], f"base_global_share_of_equity_pct.{profile_band}"
    )
    residency_rules = selected_rules.get("residency", {})
    is_nri = selected_residency.status == "NRI"
    if is_nri and residency_rules:
        country_tables = residency_rules.get("india_share_by_country", {})
        table_name = country_tables.get(selected_residency.country, "india_share_abroad_pct")
        return_year = selected_residency.planned_return_year
        if selected_residency.settle_plan == "india" and return_year is None:
            india_share = Decimal(100) - base_global
        else:
            india_share = decimal_rule(
                residency_rules[table_name][profile_band],
                f"residency.{table_name}.{profile_band}",
            )
        if return_year is not None and as_of is not None:
            years_to_return = return_year - as_of.year
            glide_years = integer_rule(residency_rules["return_glide_years"], "return_glide_years")
            if 0 <= years_to_return <= glide_years and glide_years > 0:
                ri_india_share = Decimal(100) - base_global
                progress_to_ri = Decimal(glide_years - years_to_return) / Decimal(glide_years)
                india_share += (ri_india_share - india_share) * progress_to_ri
        global_share = Decimal(100) - india_share + goal_bonus + global_tilt
        maximum_global = decimal_rule(
            residency_rules["maximum_global_share_pct"], "residency.maximum_global_share_pct"
        )
        global_share = min(maximum_global, max(Decimal(0), global_share))
    else:
        global_share = min(Decimal(60), max(Decimal(0), base_global + goal_bonus + global_tilt))
    if "equity" not in plan.target_rupees:
        raise ValueError("plan must contain the unsplit equity sleeve")

    equity_target = plan.target_rupees["equity"]
    equity_global = round_paise(equity_target * global_share / Decimal(100))
    equity_india = round_paise(equity_target - equity_global)
    gold_delta = round_paise(plan.investable_total * gold_tilt / Decimal(100))
    gold_target = round_paise(plan.target_rupees["gold"] + gold_delta)
    debt_target = round_paise(plan.target_rupees["debt"] - gold_delta)
    if gold_target < 0 or debt_target < 0:
        raise ValueError("gold tilt would make a target sleeve negative")

    target: dict[PlanSleeve, Decimal] = {
        "equity_india": equity_india,
        "equity_global": equity_global,
        "debt": debt_target,
        "gold": gold_target,
        "cash": plan.target_rupees["cash"],
    }
    if plan.current_equity_rupees is None:
        current_india = plan.current_rupees["equity"]
        current_global = Decimal("0.00")
    else:
        current_india = round_paise(plan.current_equity_rupees["equity_india"])
        current_global = round_paise(plan.current_equity_rupees["equity_global"])
    current: dict[PlanSleeve, Decimal] = {
        "equity_india": current_india,
        "equity_global": current_global,
        "debt": plan.current_rupees["debt"],
        "gold": plan.current_rupees["gold"],
        "cash": plan.current_rupees["cash"],
    }
    target_percent = _percentages(target, plan.investable_total)
    current_percent = _percentages(current, plan.investable_total)
    gaps = {sleeve: round_paise(target[sleeve] - current[sleeve]) for sleeve in target}
    equity_current = MappingProxyType(
        {"equity_india": current_india, "equity_global": current_global}
    )
    return AllocationPlan(
        target_rupees=MappingProxyType(target),
        target_percent=target_percent,
        current_rupees=MappingProxyType(current),
        current_percent=current_percent,
        gap=MappingProxyType(gaps),
        buckets=plan.buckets,
        investable_total=plan.investable_total,
        fx_usd_inr=plan.fx_usd_inr,
        current_equity_rupees=equity_current,
        new_money_only=plan.new_money_only or global_tilt != 0 or gold_tilt != 0,
        notes=plan.notes,
    )


def _percentages(
    amounts: Mapping[PlanSleeve, Decimal], total: Decimal
) -> Mapping[PlanSleeve, Decimal]:
    if total <= 0:
        return MappingProxyType({sleeve: Decimal("0.00") for sleeve in amounts})
    precision = Decimal("0.01")
    result = {
        sleeve: (amount * Decimal(100) / total).quantize(precision, rounding=ROUND_HALF_UP)
        for sleeve, amount in amounts.items()
    }
    residual_sleeve = max(amounts, key=amounts.__getitem__)
    result[residual_sleeve] += Decimal(100) - sum(result.values(), Decimal(0))
    return MappingProxyType(result)
