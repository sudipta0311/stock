"""Select eligible debt instruments by deterministic after-tax yield."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from wealth.core.money import round_paise
from wealth.policy.residency import eligible_for_new_money
from wealth.policy.rules import decimal_rule, integer_rule
from wealth.policy.snapshot import WealthSnapshot
from wealth.schemas.assets import PPFAsset
from wealth.schemas.common import LiquidityClass, Rupees
from wealth.schemas.profile import FinancialContext, Residency, TaxRegime

DebtRules = Mapping[str, Any]
DebtTax = Literal["slab", "equity", "exempt", "slab_deferred", "local_unmodeled"]
_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "debt_instruments.yaml"
_LIQUIDITY_RANK: Mapping[LiquidityClass, int] = {
    LiquidityClass.INSTANT: 0,
    LiquidityClass.WITHIN_1Y: 1,
    LiquidityClass.LOCKED: 2,
}


class DebtNeed(BaseModel):
    """Amount, horizon, and liquidity limit for one debt allocation need."""

    model_config = ConfigDict(extra="forbid")

    amount: Rupees
    horizon_years: Decimal = Field(ge=Decimal(0))
    liquidity_need: LiquidityClass


@dataclass(frozen=True)
class DebtPick:
    """An eligible instrument ranked for the requested need."""

    instrument: str
    amount: Decimal
    after_tax_yield_pct: Decimal
    reason_codes: list[str]
    reason_params: Mapping[str, Mapping[str, int | Decimal]] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class UnrankedDebtOption:
    """Eligible debt instrument omitted from ranking because its yield is unavailable."""

    instrument: str
    reason: Literal["yield_unavailable"] = "yield_unavailable"


@dataclass(frozen=True)
class DebtPickResult:
    """Ranked instruments and eligible options that lack a usable yield."""

    ranked_picks: tuple[DebtPick, ...]
    unranked_options: tuple[UnrankedDebtOption, ...]


def load_debt_rules() -> dict[str, Any]:
    """Load and validate the checked-in debt instrument catalogue."""
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        document = yaml.safe_load(rules_file)
    if not isinstance(document, dict):
        raise ValueError("debt rules must contain a YAML mapping")
    validate_debt_rules(document)
    return document


def validate_debt_rules(rules: Mapping[str, Any]) -> None:
    """Validate instrument order, eligibility fields, and tax treatment."""
    if integer_rule(rules.get("version"), "version") != 1:
        raise ValueError("unsupported debt rules version")
    if decimal_rule(rules.get("fd_split_limit_inr"), "fd_split_limit_inr") <= 0:
        raise ValueError("fd_split_limit_inr must be positive")
    instruments = rules.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        raise ValueError("instruments must be a non-empty ordered list")
    keys: set[str] = set()
    tax_types = {"slab", "equity", "exempt", "slab_deferred", "local_unmodeled"}
    for item in instruments:
        if not isinstance(item, Mapping):
            raise ValueError("each debt instrument must be a mapping")
        key = item.get("key")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("instrument keys must be non-empty and unique")
        keys.add(key)
        minimum = decimal_rule(item.get("min_horizon_years"), f"{key}.min_horizon_years")
        maximum_raw = item.get("max_horizon_years")
        maximum = (
            None if maximum_raw is None else decimal_rule(maximum_raw, f"{key}.max_horizon_years")
        )
        if minimum < 0 or (maximum is not None and maximum < minimum):
            raise ValueError(f"{key} has an invalid horizon range")
        try:
            LiquidityClass(item.get("liquidity"))
        except ValueError as error:
            raise ValueError(f"{key}.liquidity is unknown") from error
        if item.get("tax") not in tax_types:
            raise ValueError(f"{key}.tax is unknown")
        is_flexible = item.get("tenure_flexible") is True
        has_lock_in = "lock_in_years" in item
        if is_flexible:
            if has_lock_in:
                raise ValueError(f"{key} cannot define both flexible tenure and lock-in")
            minimum_days = integer_rule(item.get("min_tenure_days"), f"{key}.min_tenure_days")
            if minimum_days < 0:
                raise ValueError(f"{key}.min_tenure_days must be non-negative")
        else:
            if not has_lock_in:
                raise ValueError(f"{key} must define lock_in_years or flexible tenure")
            lock_in_years = decimal_rule(item["lock_in_years"], f"{key}.lock_in_years")
            if lock_in_years < 0 or "min_tenure_days" in item:
                raise ValueError(f"{key} has invalid lock-in terms")
        eligibility = item.get("eligibility")
        if not isinstance(eligibility, Mapping):
            raise ValueError(f"{key}.eligibility must be a mapping")
        for name in ("min_age", "max_amount", "annual_limit"):
            value = eligibility.get(name)
            if value is not None:
                if name == "min_age":
                    if integer_rule(value, f"{key}.eligibility.min_age") < 0:
                        raise ValueError(f"{key}.eligibility.min_age must be non-negative")
                elif decimal_rule(value, f"{key}.eligibility.{name}") < 0:
                    raise ValueError(f"{key}.eligibility.{name} must be non-negative")
        yield_key = item.get("yield_key")
        if not isinstance(yield_key, str) or not yield_key:
            raise ValueError(f"{key}.yield_key must be non-empty text")


def pick_debt(
    need: DebtNeed,
    ctx: FinancialContext,
    profile_tax_slab_pct: Decimal,
    age: int,
    yields: Mapping[str, Decimal],
    existing: WealthSnapshot,
    rules: DebtRules | None = None,
    residency: Residency | None = None,
) -> list[DebtPick]:
    """Return up to three eligible debt instruments by after-tax yield."""
    result = pick_debt_with_unranked(
        need,
        ctx,
        profile_tax_slab_pct,
        age,
        yields,
        existing,
        rules=rules,
        residency=residency,
    )
    return list(result.ranked_picks)


def pick_debt_with_unranked(
    need: DebtNeed,
    ctx: FinancialContext,
    profile_tax_slab_pct: Decimal,
    age: int,
    yields: Mapping[str, Decimal],
    existing: WealthSnapshot,
    rules: DebtRules | None = None,
    residency: Residency | None = None,
) -> DebtPickResult:
    """Rank priced instruments and report eligible options whose yield is missing."""
    selected_rules = load_debt_rules() if rules is None else rules
    validate_debt_rules(selected_rules)
    selected_residency = residency or Residency()
    slab = decimal_rule(profile_tax_slab_pct, "profile_tax_slab_pct")
    if not Decimal(0) <= slab <= Decimal(30):
        raise ValueError("profile_tax_slab_pct must be from 0 to 30")
    if not isinstance(age, int) or isinstance(age, bool) or age < 0:
        raise ValueError("age must be a non-negative integer")
    rate_values = {key: _yield_value(value, f"yields.{key}") for key, value in yields.items()}

    ppf_contributions = sum(
        (
            asset.yearly_contribution or Decimal(0)
            for asset, _valuation in existing.valued_assets
            if isinstance(asset, PPFAsset)
        ),
        Decimal(0),
    )
    candidates: list[tuple[Decimal, int, int, DebtPick]] = []
    unranked_options: list[UnrankedDebtOption] = []
    for order, instrument in enumerate(selected_rules["instruments"]):
        key = instrument["key"]
        if not eligible_for_new_money(key, selected_residency):
            continue
        minimum_horizon = decimal_rule(instrument["min_horizon_years"], f"{key}.min_horizon_years")
        maximum_horizon = (
            None
            if instrument["max_horizon_years"] is None
            else decimal_rule(instrument["max_horizon_years"], f"{key}.max_horizon_years")
        )
        if need.horizon_years < minimum_horizon or (
            maximum_horizon is not None and need.horizon_years > maximum_horizon
        ):
            continue

        liquidity = LiquidityClass(instrument["liquidity"])
        liquidity_eligible = _LIQUIDITY_RANK[liquidity] <= _LIQUIDITY_RANK[need.liquidity_need]
        tenure_flexible = instrument.get("tenure_flexible") is True
        flexible_eligible = False
        if tenure_flexible:
            minimum_days = integer_rule(instrument["min_tenure_days"], f"{key}.min_tenure_days")
            # DebtNeed horizons are years, so compare flexible tenures on a 365-day year.
            flexible_eligible = Decimal(minimum_days) <= need.horizon_years * Decimal(365)
        lock_in_raw = instrument.get("lock_in_years")
        lock_in_eligible = (
            lock_in_raw is not None
            and decimal_rule(lock_in_raw, f"{key}.lock_in_years") <= need.horizon_years
        )
        if not (liquidity_eligible or flexible_eligible or lock_in_eligible):
            continue
        eligibility = instrument["eligibility"]
        minimum_age = eligibility["min_age"]
        if minimum_age is not None and age < integer_rule(minimum_age, f"{key}.min_age"):
            continue
        yield_key = instrument["yield_key"]
        if yield_key not in rate_values:
            unranked_options.append(UnrankedDebtOption(instrument=key))
            continue

        amount = need.amount
        reasons = ["after_tax_yield"]
        params: dict[str, Mapping[str, int | Decimal]] = {}
        if tenure_flexible:
            reasons.append("suggested_tenure")
            params["suggested_tenure"] = MappingProxyType(
                {"years": min(need.horizon_years, Decimal(10))}
            )
        maximum_amount_raw = eligibility["max_amount"]
        if maximum_amount_raw is not None:
            maximum_amount = decimal_rule(maximum_amount_raw, f"{key}.max_amount")
            if amount > maximum_amount:
                amount = maximum_amount
                reasons.append("amount_limited_by_instrument_limit")
        annual_limit_raw = eligibility["annual_limit"]
        if annual_limit_raw is not None:
            annual_limit = decimal_rule(annual_limit_raw, f"{key}.annual_limit")
            if key == "ppf":
                remaining_room = max(Decimal(0), annual_limit - ppf_contributions)
                if amount > remaining_room:
                    amount = remaining_room
                    reasons.append("amount_limited_by_annual_room")
        if amount <= 0:
            continue

        yield_pct = rate_values[yield_key]
        tax_type: DebtTax = instrument["tax"]
        after_tax = _after_tax_yield(yield_pct, slab, tax_type, need.horizon_years)
        if key == "nre_fd":
            reasons.append("nre_interest_exempt_in_india")
        elif key == "fcnr_b":
            reasons.extend(["fcnr_interest_exempt_in_india", "fcnr_currency_check"])
        elif key == "nro_fd":
            reasons.append("nro_interest_tds")
        elif key == "local_bank_deposit":
            reasons.append("local_deposit_tax_review")
        if selected_residency.status != "RI":
            reasons.append("tax_residence_country_may_apply")
        if selected_residency.us_person and key in {"nre_fd", "fcnr_b"}:
            reasons.append("us_tax_applies")
        if key == "ppf" and ctx.tax_regime == TaxRegime.OLD:
            reasons.append("80c_deduction_available")
        if key == "bank_fd" and amount > Decimal("500000"):
            bank_count = int((amount / Decimal("500000")).to_integral_value(rounding=ROUND_CEILING))
            reasons.append("split_across_banks")
            params["split_across_banks"] = MappingProxyType({"bank_count": bank_count})

        pick = DebtPick(
            instrument=key,
            amount=round_paise(amount),
            after_tax_yield_pct=after_tax.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            reason_codes=reasons,
            reason_params=MappingProxyType(params),
        )
        candidates.append((after_tax, _LIQUIDITY_RANK[liquidity], order, pick))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return DebtPickResult(
        ranked_picks=tuple(item[3] for item in candidates[:3]),
        unranked_options=tuple(unranked_options),
    )


def _yield_value(value: Decimal, name: str) -> Decimal:
    result = decimal_rule(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _after_tax_yield(
    yield_pct: Decimal,
    slab_pct: Decimal,
    tax_type: DebtTax,
    horizon_years: Decimal,
) -> Decimal:
    if tax_type == "exempt":
        return yield_pct
    if tax_type == "local_unmodeled":
        return yield_pct
    if tax_type in {"slab", "slab_deferred"}:
        tax_pct = slab_pct
    elif horizon_years <= 1:
        tax_pct = Decimal("20")
    else:
        tax_pct = Decimal("12.5")
    return yield_pct * (Decimal(1) - tax_pct / Decimal(100))
