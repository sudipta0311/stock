"""Deterministic selector for routes to global investments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from wealth.core.money import round_paise
from wealth.policy.rules import decimal_rule, integer_rule
from wealth.schemas.common import Rupees
from wealth.schemas.profile import Residency

RouteCode = Literal["A", "B", "C", "D", "WAIT", "LOCAL"]
RouteRules = Mapping[str, Any]
_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "routes.yaml"


def _strict_decimal(value: object) -> Decimal:
    return decimal_rule(value, "decimal")


_USD_AMOUNT = Annotated[Decimal, BeforeValidator(_strict_decimal), Field(ge=Decimal(0))]
_PERCENT = Annotated[Decimal, BeforeValidator(_strict_decimal), Field(ge=Decimal(0))]


class RouteContext(BaseModel):
    """Values that determine route fit, cost, and reporting warnings."""

    model_config = ConfigDict(extra="forbid")

    monthly_global_new_money: Rupees
    annual_global_new_money: Rupees | None = None
    has_lrs_broker: bool
    prefers_no_paperwork: bool
    us_situs_holdings_usd: _USD_AMOUNT = Decimal(0)
    lrs_used_this_fy: Rupees = Decimal(0)
    amc_international_suspended: bool = False
    etf_inav_premium_pct: _PERCENT | None = None
    residency: Residency = Field(default_factory=Residency)

    @model_validator(mode="after")
    def derive_annual_amount(self) -> RouteContext:
        if self.annual_global_new_money is None:
            self.annual_global_new_money = round_paise(self.monthly_global_new_money * 12)
        return self


@dataclass(frozen=True)
class RouteDecision:
    """Selected route and machine-readable explanations and warnings."""

    route: RouteCode
    vehicle: str
    reasons: list[str]
    warnings: list[str]
    alternatives: list[RouteCode]
    warning_params: Mapping[str, Mapping[str, Decimal | str]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def load_route_rules() -> dict[str, Any]:
    """Load and validate the checked-in global route selector rules."""
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        document = yaml.safe_load(rules_file)
    if not isinstance(document, dict):
        raise ValueError("route rules must contain a YAML mapping")
    validate_route_rules(document)
    return document


def validate_route_rules(rules: Mapping[str, Any]) -> None:
    """Check versioned thresholds and route-selector switches."""
    if integer_rule(rules.get("version"), "version") != 1:
        raise ValueError("unsupported route rules version")
    defaults = rules.get("defaults")
    if not isinstance(defaults, Mapping):
        raise ValueError("route defaults must be a mapping")
    for key in (
        "route_a_max_monthly_inr",
        "route_c_min_annual_inr",
        "switch_to_ucits_above_usd",
        "etf_max_inav_premium_pct",
        "lrs_tcs_threshold_inr",
    ):
        if decimal_rule(defaults.get(key), f"defaults.{key}") < 0:
            raise ValueError(f"defaults.{key} must be non-negative")
    note = rules.get("fx_markup_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("fx_markup_note must be non-empty text")
    checks = rules.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("route checks must be a mapping")
    for key in (
        "pause_if_amc_subscription_suspended",
        "require_schedule_fa_reminder_for_route_c",
    ):
        if not isinstance(checks.get(key), bool):
            raise ValueError(f"checks.{key} must be boolean")


def select_route(ctx: RouteContext, rules: RouteRules | None = None) -> RouteDecision:
    """Choose the simplest eligible global route and report applicable warnings."""
    selected_rules = load_route_rules() if rules is None else rules
    validate_route_rules(selected_rules)
    defaults = selected_rules["defaults"]
    route_a_max_monthly = decimal_rule(
        defaults["route_a_max_monthly_inr"], "defaults.route_a_max_monthly_inr"
    )
    route_c_min_annual = decimal_rule(
        defaults["route_c_min_annual_inr"], "defaults.route_c_min_annual_inr"
    )
    ucits_threshold = decimal_rule(
        defaults["switch_to_ucits_above_usd"], "defaults.switch_to_ucits_above_usd"
    )
    maximum_premium = decimal_rule(
        defaults["etf_max_inav_premium_pct"], "defaults.etf_max_inav_premium_pct"
    )
    tcs_threshold = decimal_rule(
        defaults["lrs_tcs_threshold_inr"], "defaults.lrs_tcs_threshold_inr"
    )
    checks = selected_rules["checks"]
    amc_suspended = (
        ctx.amc_international_suspended and checks["pause_if_amc_subscription_suspended"]
    )
    annual_new_money = ctx.annual_global_new_money
    if annual_new_money is None:
        annual_new_money = round_paise(ctx.monthly_global_new_money * 12)

    if ctx.residency.status in {"NRI", "RNOR"}:
        is_us_person = ctx.residency.us_person
        vehicle = "us_broad_market_etf" if is_us_person else "ucits_global_etf"
        warnings = (
            ["estate_tax_us_domiciled_assets"]
            if not is_us_person
            and (ctx.us_situs_holdings_usd > 0 or vehicle in {"us_etf", "us_broad_market_etf"})
            else []
        )
        return RouteDecision(
            route="LOCAL",
            vehicle=vehicle,
            reasons=["residency_local_route"],
            warnings=warnings,
            alternatives=[],
        )

    prefers_simple_route = ctx.prefers_no_paperwork or annual_new_money < route_c_min_annual
    etf_is_eligible = (
        ctx.etf_inav_premium_pct is not None and ctx.etf_inav_premium_pct <= maximum_premium
    )
    if prefers_simple_route:
        if not amc_suspended:
            route: RouteCode = "A"
            reasons = (
                ["prefers_no_paperwork"]
                if ctx.prefers_no_paperwork
                else (
                    ["below_route_a_max_monthly_inr"]
                    if ctx.monthly_global_new_money <= route_a_max_monthly
                    else ["below_route_c_min_annual_inr"]
                )
            )
        elif etf_is_eligible:
            route = "B"
            reasons = ["amc_international_suspended", "etf_premium_acceptable"]
        elif ctx.has_lrs_broker:
            route = "C"
            reasons = ["amc_international_suspended", "route_b_unavailable"]
        else:
            route = "WAIT"
            reasons = ["no_open_route"]
    elif ctx.has_lrs_broker:
        route = "C"
        reasons = ["annual_global_new_money_meets_route_c_threshold"]
    elif not amc_suspended:
        route = "A"
        reasons = ["open_lrs_broker_to_save_costs"]
    elif etf_is_eligible:
        route = "B"
        reasons = ["amc_international_suspended", "etf_premium_acceptable"]
    else:
        route = "WAIT"
        reasons = ["no_open_route"]

    if route == "A":
        vehicle = "in_international_index_fund"
    elif route == "B":
        vehicle = "in_listed_international_etf"
    elif route == "C":
        vehicle = "ucits_etf" if ctx.us_situs_holdings_usd >= ucits_threshold else "us_etf"
    else:
        vehicle = "no_open_route"

    warnings: list[str] = []
    warning_params: dict[str, Mapping[str, Decimal | str]] = {}
    if not ctx.residency.us_person and (
        ctx.us_situs_holdings_usd > 0 or vehicle in {"us_etf", "us_broad_market_etf"}
    ):
        warnings.append("estate_tax_us_domiciled_assets")
    lrs_total = ctx.lrs_used_this_fy + annual_new_money
    if lrs_total > tcs_threshold:
        excess = round_paise(lrs_total - tcs_threshold)
        tcs_amount = (excess * Decimal("0.20")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        warnings.append("tcs_applies")
        warning_params["tcs_applies"] = MappingProxyType(
            {"excess_inr": excess, "tcs_inr": tcs_amount}
        )
    if route in {"C", "D"} and checks["require_schedule_fa_reminder_for_route_c"]:
        warnings.append("schedule_fa_reporting")
    if route == "C":
        note = selected_rules["fx_markup_note"]
        warnings.append("fx_markup_note")
        warning_params["fx_markup_note"] = MappingProxyType({"note": note})
        if (
            not ctx.residency.us_person
            and vehicle == "us_etf"
            and ctx.us_situs_holdings_usd >= ucits_threshold * Decimal("0.80")
        ):
            warnings.append("us_estate_tax_threshold_near")

    available_routes: list[RouteCode] = []
    if not amc_suspended:
        available_routes.append("A")
    if etf_is_eligible:
        available_routes.append("B")
    if ctx.has_lrs_broker:
        available_routes.append("C")
    alternatives = [candidate for candidate in available_routes if candidate != route]
    return RouteDecision(
        route=route,
        vehicle=vehicle,
        reasons=reasons,
        warnings=warnings,
        alternatives=alternatives,
        warning_params=MappingProxyType(warning_params),
    )
