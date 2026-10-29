"""Deterministic loan tax relief, debt-load, and spare-money decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from wealth.core.money import round_paise
from wealth.policy.rules import decimal_rule, integer_rule
from wealth.schemas.assets import LoanAsset

LoanDecisionKind = Literal["PAY_EXTRA", "INVEST", "BUILD_SAFETY_FIRST"]
DebtLoadLevel = Literal["fine", "attention", "high", "unknown"]
_RULES_DIR = Path(__file__).resolve().parents[3] / "rules"
_TAX_RULES_PATH = _RULES_DIR / "tax" / "loan_relief.yaml"
_LOAN_RULES_PATH = _RULES_DIR / "loans.yaml"


@dataclass(frozen=True)
class LoanCandidate:
    """A loan and its current household-currency amounts for policy ranking."""

    id: str
    loan: LoanAsset
    outstanding_inr: Decimal
    emi_inr: Decimal | None
    tax_regime: Literal["old", "new"] | None = None
    tax_slab_pct: Decimal | None = None
    country: str = "IN"
    member_id: str | None = None
    never_pays_off: bool = False


@dataclass(frozen=True)
class LoanPlanContext:
    """The spare-money and rate inputs needed for one loan decision."""

    spare_money_inr: Decimal
    emergency_fund_gap_inr: Decimal
    monthly_income_inr: Decimal | None
    expected_safe_return_pct: Decimal
    high_interest_threshold_pct: Decimal
    tax_rules: Mapping[str, Any]
    debt_load_rules: Mapping[str, Any]
    danish_interest_rule: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LoanDecision:
    action: LoanDecisionKind
    loan_id: str | None
    suggested_amount_inr: Decimal
    effective_rate_pct: Decimal | None
    safe_return_pct: Decimal
    reason_codes: tuple[str, ...]
    answer: str
    monthly_emis_inr: Decimal
    debt_load_pct: Decimal | None
    debt_load_level: DebtLoadLevel
    tax_source_urls: tuple[str, ...]


@lru_cache(maxsize=1)
def load_tax_relief_rules() -> dict[str, Any]:
    """Load the explicitly sourced, unverified India loan relief rules."""
    with _TAX_RULES_PATH.open(encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict) or integer_rule(document.get("version"), "version") != 1:
        raise ValueError("unsupported loan tax relief rules")
    rules = document.get("rules")
    india = rules.get("india") if isinstance(rules, Mapping) else None
    if not isinstance(india, Mapping) or not india:
        raise ValueError("tax rules must define India loan relief")
    for relief, regimes in india.items():
        if not isinstance(relief, str) or not isinstance(regimes, Mapping):
            raise ValueError("tax relief rules must use named regimes")
        for regime in ("old", "new"):
            rule = regimes.get(regime)
            if not isinstance(rule, Mapping) or not isinstance(rule.get("eligible"), bool):
                raise ValueError(f"tax relief {relief}.{regime} is invalid")
            url = rule.get("source_url")
            if (
                not isinstance(url, str)
                or not url.startswith("https://www.incometax.gov.in/")
                or rule.get("verified") is not False
            ):
                raise ValueError(
                    f"tax relief {relief}.{regime} must have an unverified official source"
                )
            if (
                rule.get("interest_cap_inr") is not None
                and decimal_rule(rule["interest_cap_inr"], f"{relief}.{regime}.interest_cap_inr")
                < 0
            ):
                raise ValueError("interest cap must be non-negative")
            if (
                rule.get("loss_setoff_cap_inr") is not None
                and decimal_rule(
                    rule["loss_setoff_cap_inr"], f"{relief}.{regime}.loss_setoff_cap_inr"
                )
                < 0
            ):
                raise ValueError("loss set-off cap must be non-negative")
    return document


@lru_cache(maxsize=1)
def load_loan_policy_rules() -> dict[str, Any]:
    """Load debt-load bands from checked-in rule data."""
    with _LOAN_RULES_PATH.open(encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict) or integer_rule(document.get("version"), "version") != 1:
        raise ValueError("unsupported loan policy rules")
    bands = document.get("debt_load")
    if not isinstance(bands, Mapping):
        raise ValueError("loan rules must define debt_load")
    fine = decimal_rule(bands.get("fine_below_pct"), "debt_load.fine_below_pct")
    attention = decimal_rule(bands.get("attention_through_pct"), "debt_load.attention_through_pct")
    if fine <= 0 or attention <= fine or attention > 100:
        raise ValueError("loan debt-load bands are invalid")
    return document


def effective_loan_rate(
    candidate: LoanCandidate,
    tax_rules: Mapping[str, Any],
    danish_interest_rule: Mapping[str, Any] | None = None,
) -> tuple[Decimal, tuple[str, ...], tuple[str, ...]]:
    """Return an estimated annual rate after only the selected sourced relief."""
    nominal = candidate.loan.rate_pct
    relief = candidate.loan.tax_relief
    if relief == "none":
        return nominal, (), ()
    if relief == "dk_interest_deduction":
        if (
            candidate.country == "DK"
            and danish_interest_rule is not None
            and danish_interest_rule.get("verified") is False
            and isinstance(danish_interest_rule.get("source_url"), str)
        ):
            return nominal, ("tax_relief_needs_review",), (str(danish_interest_rule["source_url"]),)
        return nominal, ("tax_relief_rule_missing",), ()
    if candidate.country != "IN" or relief not in {
        "in_home_self_occupied",
        "in_home_let_out",
        "in_education",
    }:
        return nominal, ("tax_relief_not_applicable",), ()
    expected_kind = "education" if relief == "in_education" else "home"
    if candidate.loan.loan_kind != expected_kind or candidate.tax_regime is None:
        return nominal, ("tax_relief_not_applicable",), ()
    india = tax_rules.get("rules", {}).get("india", {})
    by_regime = india.get(relief, {}) if isinstance(india, Mapping) else {}
    rule = by_regime.get(candidate.tax_regime) if isinstance(by_regime, Mapping) else None
    if not isinstance(rule, Mapping) or rule.get("eligible") is not True:
        return nominal, ("tax_relief_not_available",), ()
    source_url = rule.get("source_url")
    if not isinstance(source_url, str) or rule.get("verified") is not False:
        return nominal, ("tax_relief_rule_missing",), ()
    if relief == "in_home_let_out" and rule.get("other_head_loss_setoff") is False:
        return nominal, ("tax_relief_needs_review",), (source_url,)
    if candidate.tax_slab_pct is None or candidate.outstanding_inr <= 0:
        return nominal, ("tax_rate_needs_review", "tax_relief_needs_review"), (source_url,)
    slab = _rate(candidate.tax_slab_pct, "tax_slab_pct")
    if slab > 100:
        raise ValueError("tax_slab_pct must be at most 100")
    annual_interest = candidate.outstanding_inr * nominal / Decimal(100)
    eligible_interest = annual_interest
    interest_cap = rule.get("interest_cap_inr")
    if interest_cap is not None:
        eligible_interest = min(eligible_interest, decimal_rule(interest_cap, "interest_cap_inr"))
    setoff_cap = rule.get("loss_setoff_cap_inr")
    if setoff_cap is not None:
        eligible_interest = min(eligible_interest, decimal_rule(setoff_cap, "loss_setoff_cap_inr"))
    relief_amount = eligible_interest * slab / Decimal(100)
    rate_saving = relief_amount * Decimal(100) / candidate.outstanding_inr
    return max(Decimal(0), nominal - rate_saving), ("tax_relief_needs_review",), (source_url,)


def pay_extra_or_invest(
    loans: Sequence[LoanCandidate], plan_context: LoanPlanContext
) -> LoanDecision:
    """Choose safety, extra repayment, or investment using sourced rate inputs."""
    spare = _non_negative(plan_context.spare_money_inr, "spare_money_inr")
    safety_gap = _non_negative(plan_context.emergency_fund_gap_inr, "emergency_fund_gap_inr")
    safe_return = _rate(plan_context.expected_safe_return_pct, "expected_safe_return_pct")
    threshold = _rate(plan_context.high_interest_threshold_pct, "high_interest_threshold_pct")
    emi_values = [candidate.emi_inr for candidate in loans]
    monthly_emis = round_paise(sum((value or Decimal(0) for value in emi_values), Decimal(0)))
    income = plan_context.monthly_income_inr
    debt_pct: Decimal | None = None
    level: DebtLoadLevel = "unknown"
    if income is not None and income > 0 and all(value is not None for value in emi_values):
        debt_pct = monthly_emis * Decimal(100) / income
        bands = plan_context.debt_load_rules["debt_load"]
        fine = decimal_rule(bands["fine_below_pct"], "debt_load.fine_below_pct")
        attention = decimal_rule(bands["attention_through_pct"], "debt_load.attention_through_pct")
        level = "fine" if debt_pct < fine else "attention" if debt_pct <= attention else "high"

    if safety_gap > 0:
        return LoanDecision(
            action="BUILD_SAFETY_FIRST",
            loan_id=None,
            suggested_amount_inr=round_paise(min(spare, safety_gap)),
            effective_rate_pct=None,
            safe_return_pct=safe_return,
            reason_codes=("emergency_fund_below_target",),
            answer="Build your emergency fund before paying extra or investing.",
            monthly_emis_inr=monthly_emis,
            debt_load_pct=debt_pct,
            debt_load_level=level,
            tax_source_urls=(),
        )
    if not loans:
        return LoanDecision(
            action="INVEST",
            loan_id=None,
            suggested_amount_inr=spare,
            effective_rate_pct=None,
            safe_return_pct=safe_return,
            reason_codes=("no_active_loans",),
            answer="Invest spare money in line with your plan.",
            monthly_emis_inr=monthly_emis,
            debt_load_pct=debt_pct,
            debt_load_level=level,
            tax_source_urls=(),
        )

    ranked: list[tuple[Decimal, LoanCandidate, tuple[str, ...], tuple[str, ...]]] = []
    for candidate in loans:
        effective, reasons, sources = effective_loan_rate(
            candidate, plan_context.tax_rules, plan_context.danish_interest_rule
        )
        ranked.append((effective, candidate, reasons, sources))
    ranked.sort(
        key=lambda item: (
            0 if item[1].loan.loan_kind == "credit_card" else 1,
            -item[0],
            item[1].id,
        )
    )
    high_interest = [
        item for item in ranked if item[1].loan.loan_kind == "credit_card" or item[0] >= threshold
    ]
    selected = high_interest[0] if high_interest else ranked[0]
    effective, candidate, tax_reasons, sources = selected
    if high_interest or effective > safe_return:
        forced = bool(high_interest)
        return LoanDecision(
            action="PAY_EXTRA",
            loan_id=candidate.id,
            suggested_amount_inr=round_paise(min(spare, candidate.outstanding_inr)),
            effective_rate_pct=effective,
            safe_return_pct=safe_return,
            reason_codes=tuple(
                dict.fromkeys(
                    (
                        "high_interest_debt_first" if forced else "loan_cost_above_safe_return",
                        *tax_reasons,
                    )
                )
            ),
            answer=(
                f"Pay extra toward your {candidate.loan.loan_kind.replace('_', ' ')} loan first."
                if forced
                else (
                    "This loan costs more than safe money is expected to earn. Pay extra toward it."
                )
            ),
            monthly_emis_inr=monthly_emis,
            debt_load_pct=debt_pct,
            debt_load_level=level,
            tax_source_urls=sources,
        )
    return LoanDecision(
        action="INVEST",
        loan_id=candidate.id,
        suggested_amount_inr=spare,
        effective_rate_pct=effective,
        safe_return_pct=safe_return,
        reason_codes=tuple(dict.fromkeys(("safe_money_return_covers_loan_cost", *tax_reasons))),
        answer="Safe money is expected to earn at least this loan costs. Invest the spare money.",
        monthly_emis_inr=monthly_emis,
        debt_load_pct=debt_pct,
        debt_load_level=level,
        tax_source_urls=sources,
    )


def _non_negative(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a non-negative finite Decimal")
    return value


def _rate(value: Decimal, name: str) -> Decimal:
    amount = _non_negative(value, name)
    if amount > 100:
        raise ValueError(f"{name} must not exceed 100")
    return amount
