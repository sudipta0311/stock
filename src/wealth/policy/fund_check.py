"""Deterministic checks and twelve-month projections for Indian mutual funds."""

from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Literal

import yaml

from wealth.core.money import round_paise
from wealth.policy.actions import WeekInput
from wealth.policy.rebalance import (
    BandRules,
    Breach,
    RebalancePlan,
    band_breaches,
    rebalance_plan,
)
from wealth.policy.tax import Lot, TaxContext, TaxEstimate, estimate_sale_tax, load_tax_rules

_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "fund_check.yaml"
_MONEY_ZERO = Decimal("0.00")
_INDIA_EQUITY_BUCKETS = (
    "large",
    "mid",
    "small",
    "flexi_multi",
    "sector_thematic",
    "international",
)
_RECURRING_KINDS = {"buy", "switch_in"}
_EQUITY_TAX_START = date(2024, 7, 23)
_EQUITY_TAX_CLASSES = {"listed_equity", "equity_mf"}


@dataclass(frozen=True, kw_only=True)
class FundReference:
    security_id: str
    name: str
    category: str | None
    sub_category: str | None
    plan_type: str | None
    option_type: str | None
    ter: Decimal | None = None


@dataclass(frozen=True)
class FundHolding(FundReference):
    member_id: str
    value: Decimal
    quantity: Decimal
    price: Decimal | None
    tax_lots: tuple[Lot, ...] = ()
    tax_lots_complete: bool = False
    exit_load_pct: Decimal | None = None


@dataclass(frozen=True)
class FundTransactionInput:
    member_id: str
    security_id: str
    transaction_date: date
    kind: str
    amount: Decimal
    units: Decimal


@dataclass(frozen=True)
class FundMember:
    member_id: str
    residency_country: str = "IN"
    residency_status: str = "RI"
    tax_context: TaxContext = field(default_factory=TaxContext)
    redirect_member_id: str | None = None
    redirect_member_name: str | None = None
    clubbing_note: str | None = None
    sip_overrides: Mapping[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class FundCheckResult:
    status: Literal["fine", "attention"]
    summary_key: str
    checks: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    early_warning: dict[str, Any] | None
    details: dict[str, Any]
    lagging_candidates: tuple[tuple[str, str, bool], ...]
    sip_amounts: tuple[tuple[str, str, Decimal], ...]


_SaleLine = tuple[Lot, Decimal, Decimal]


@dataclass(frozen=True)
class _FundSaleQuote:
    sales: tuple[_SaleLine, ...]
    estimate: TaxEstimate
    realised_equity_ltcg: Decimal
    realised_equity_stcg: Decimal
    ltcg_exemption_remaining: Decimal


class _FundSaleLedger:
    """Track member-level realized gains and lots across ordered sell suggestions."""

    def __init__(self, members: Mapping[str, FundMember]) -> None:
        self._contexts = {member_id: member.tax_context for member_id, member in members.items()}
        self._sold_quantities: dict[tuple[str, str], Decimal] = {}
        exemption = load_tax_rules().get("equity_ltcg_exemption")
        if not isinstance(exemption, dict):
            raise ValueError("Indian tax rules need an equity LTCG exemption")
        self._ltcg_exemption_limit = Decimal(str(exemption["amount_inr"]))

    def context_for(self, member: FundMember) -> TaxContext:
        return self._contexts.get(member.member_id, member.tax_context)

    def has_unsold_lots(self, holding: FundHolding) -> bool:
        if not holding.tax_lots:
            return True
        return any(
            lot.quantity > self._sold_quantities.get((holding.member_id, lot.lot_id), Decimal(0))
            for lot in holding.tax_lots
        )

    def available_sales(self, holding: FundHolding) -> tuple[_SaleLine, ...]:
        if holding.price is None:
            return ()
        available: list[_SaleLine] = []
        for lot in holding.tax_lots:
            remaining = lot.quantity - self._sold_quantities.get(
                (holding.member_id, lot.lot_id), Decimal(0)
            )
            if remaining > 0:
                available.append((lot, remaining, holding.price))
        return tuple(available)

    def remaining_value(self, holding: FundHolding, current_value: Decimal) -> Decimal:
        if not holding.tax_lots:
            return current_value
        original_quantity = sum((lot.quantity for lot in holding.tax_lots), Decimal(0))
        remaining_quantity = sum(
            (
                max(
                    Decimal(0),
                    lot.quantity
                    - self._sold_quantities.get((holding.member_id, lot.lot_id), Decimal(0)),
                )
                for lot in holding.tax_lots
            ),
            Decimal(0),
        )
        if remaining_quantity == original_quantity:
            return current_value
        if holding.price is not None:
            return round_paise(remaining_quantity * holding.price)
        if original_quantity <= 0:
            return Decimal(0)
        return round_paise(current_value * remaining_quantity / original_quantity)

    def quote(self, member: FundMember, sales: Sequence[_SaleLine], as_of: date) -> _FundSaleQuote:
        selected: list[_SaleLine] = []
        seen_lots: set[str] = set()
        for sale in sales:
            lot, quantity, _price = sale
            if lot.lot_id in seen_lots:
                continue
            available_quantity = lot.quantity - self._sold_quantities.get(
                (member.member_id, lot.lot_id), Decimal(0)
            )
            if quantity > available_quantity:
                raise ValueError("fund sale estimate reuses an already-sold lot")
            seen_lots.add(lot.lot_id)
            selected.append(sale)
        selected_sales = tuple(selected)
        context = self.context_for(member)
        estimate = estimate_sale_tax(list(selected_sales), as_of, context)
        lots_by_id = {lot.lot_id: lot for lot, _quantity, _price in selected_sales}
        realised_ltcg = Decimal(0)
        realised_stcg = Decimal(0)
        for line in estimate.lines:
            lot = lots_by_id[line.lot_id]
            if lot.tax_class not in _EQUITY_TAX_CLASSES:
                continue
            if line.holding_period == "LT":
                realised_ltcg += line.gain_after_loss_setoff
            else:
                realised_stcg += line.gain_after_loss_setoff
        realised_ltcg = round_paise(realised_ltcg)
        realised_stcg = round_paise(realised_stcg)
        remaining_exemption = max(
            Decimal(0),
            self._ltcg_exemption_limit - context.fy_realised_equity_ltcg - realised_ltcg,
        )
        return _FundSaleQuote(
            sales=selected_sales,
            estimate=estimate,
            realised_equity_ltcg=realised_ltcg,
            realised_equity_stcg=realised_stcg,
            ltcg_exemption_remaining=round_paise(remaining_exemption),
        )

    def record(self, member: FundMember, quote: _FundSaleQuote) -> None:
        context = self.context_for(member)
        self._contexts[member.member_id] = replace(
            context,
            fy_realised_equity_ltcg=round_paise(
                context.fy_realised_equity_ltcg + quote.realised_equity_ltcg
            ),
            fy_realised_equity_stcg=round_paise(
                context.fy_realised_equity_stcg + quote.realised_equity_stcg
            ),
        )
        for lot, quantity, _price in quote.sales:
            key = (member.member_id, lot.lot_id)
            sold = self._sold_quantities.get(key, Decimal(0)) + quantity
            if sold > lot.quantity:
                raise ValueError("fund sale suggestions exceed the available quantity")
            self._sold_quantities[key] = sold


@dataclass(frozen=True)
class FundRebalanceContext:
    breaches: Sequence[Breach]
    week_input: WeekInput
    lots_by_sleeve: Mapping[str, object]
    sale_date: date
    tax_context: TaxContext


def ordered_fund_rebalance_plan(context: FundRebalanceContext) -> RebalancePlan:
    """Reuse the policy's cash, SIP, then tax-aware sale order for fund sleeve fixes."""
    return rebalance_plan(
        context.breaches,
        context.week_input,
        context.lots_by_sleeve,
        context.sale_date,
        context.tax_context,
    )


@lru_cache(maxsize=1)
def _rules() -> dict[str, Any]:
    with _RULES_PATH.open(encoding="utf-8") as rules_file:
        rules = yaml.safe_load(rules_file)
    if not isinstance(rules, dict) or rules.get("version") != 1:
        raise ValueError("fund-check rules need version 1")
    targets = rules.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("fund-check rules need profile targets")
    for profile, target in targets.items():
        if not isinstance(target, dict):
            raise ValueError(f"fund-check target {profile} must be a mapping")
        for key in ("asset_mix_pct", "indian_equity_pct"):
            values = target.get(key)
            if not isinstance(values, dict) or not values:
                raise ValueError(f"fund-check target {profile}.{key} must be a mapping")
            percentages = [Decimal(str(value)) for value in values.values()]
            if any(not value.is_finite() or value < 0 for value in percentages):
                raise ValueError(f"fund-check target {profile}.{key} has invalid percentages")
            if sum(percentages, Decimal(0)) != Decimal(100):
                raise ValueError(f"fund-check target {profile}.{key} must total 100")
    return rules


def load_fund_check_rules() -> dict[str, Any]:
    """Return a caller-owned copy of the checked-in fund-check rules."""
    return deepcopy(_rules())


def classify_fund_bucket(
    category: str | None, sub_category: str | None, name: str = ""
) -> str | None:
    """Map an AMFI category to a compact mix bucket used by deterministic policy."""
    category_value = (category or "").casefold()
    sub_value = (sub_category or "").casefold()
    name_value = name.casefold()
    if "gold" in sub_value or "gold" in category_value:
        return "gold"
    if "arbitrage" in sub_value or "arbitrage" in category_value:
        return "debt"
    if "debt" in category_value or sub_value in {"liquid", "gilt", "other debt"}:
        return "debt"
    if "hybrid" in sub_value or "hybrid" in category_value:
        return "hybrid"
    if "overseas" in sub_value or "international" in category_value or "global" in category_value:
        return "international"
    if "sectoral" in sub_value or "thematic" in sub_value:
        return "sector_thematic"
    if "large" in sub_value or "nifty 50" in name_value or "large cap" in name_value:
        return "large"
    if "mid" in sub_value or "mid cap" in name_value:
        return "mid"
    if "small" in sub_value or "small cap" in name_value:
        return "small"
    if any(term in sub_value for term in ("flexi", "multi", "elss", "index")):
        return "flexi_multi"
    if category_value in {"mf:equity_india", "mf:equity_global"}:
        return "flexi_multi"
    return None


def detect_monthly_sips(
    transactions: Sequence[FundTransactionInput],
    as_of: date,
    overrides: Mapping[tuple[str, str], Decimal] | None = None,
) -> dict[tuple[str, str], Decimal]:
    """Find similar recurring purchases, allowing one skipped month in the series."""
    by_fund: dict[tuple[str, str], dict[tuple[date, str, Decimal, Decimal], None]] = defaultdict(
        dict
    )
    for item in transactions:
        if item.kind not in _RECURRING_KINDS or item.amount <= 0 or item.transaction_date > as_of:
            continue
        if (as_of - item.transaction_date).days > 370:
            continue
        key = (item.member_id, item.security_id)
        signature = (item.transaction_date, item.kind, item.amount, item.units)
        by_fund[key][signature] = None

    detected: dict[tuple[str, str], Decimal] = {}
    for key, signatures in by_fund.items():
        purchase_by_date = {signature[0]: signature[2] for signature in signatures}
        purchase_dates = sorted(purchase_by_date)
        if len(purchase_dates) < 3:
            continue
        amounts = [purchase_by_date[day] for day in purchase_dates]
        typical = Decimal(str(median(amounts)))
        if typical <= 0 or max(amounts) - min(amounts) > typical * Decimal("0.15"):
            continue
        month_numbers = [_month_index(day) for day in purchase_dates]
        gaps = [right - left for left, right in zip(month_numbers, month_numbers[1:], strict=False)]
        if not gaps or any(gap not in {1, 2} for gap in gaps):
            continue
        detected[key] = typical

    for key, amount in (overrides or {}).items():
        if amount.is_finite() and amount >= 0:
            detected[key] = amount
    return {key: amount for key, amount in detected.items() if amount > 0}


def check_funds(
    *,
    holdings: Sequence[FundHolding],
    references: Sequence[FundReference],
    histories: Mapping[str, Sequence[tuple[date, Decimal]]],
    transactions: Sequence[FundTransactionInput],
    members: Sequence[FundMember],
    as_of: date,
    risk_profile: str | None,
    previous_lagging: set[tuple[str, str]] | None = None,
    band_rules: BandRules | None = None,
    selected_member_id: str | None = None,
    indian_sleeve_target: Decimal | None = None,
    best_debt_option: Mapping[str, str] | None = None,
) -> FundCheckResult:
    """Evaluate current fund costs, diversification, tax notes, SIPs, and projected bands."""
    rules = _rules()
    limits = rules["limits"]
    profile_key = _profile_key(risk_profile, rules)
    target = rules["targets"][profile_key]
    selected_holdings = tuple(
        item
        for item in holdings
        if selected_member_id is None or item.member_id == selected_member_id
    )
    selected_members = tuple(
        member
        for member in members
        if selected_member_id is None or member.member_id == selected_member_id
    )
    member_by_id = {member.member_id: member for member in selected_members}
    sale_ledger = _FundSaleLedger(member_by_id)
    overrides = {
        (member.member_id, security_id): amount
        for member in selected_members
        for security_id, amount in member.sip_overrides.items()
    }
    sips = detect_monthly_sips(transactions, as_of, overrides)
    if selected_member_id is not None:
        sips = {key: value for key, value in sips.items() if key[0] == selected_member_id}

    values_by_bucket: dict[str, Decimal] = defaultdict(Decimal)
    values_by_fund: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    holdings_by_fund: dict[tuple[str, str], FundHolding] = {}
    for item in selected_holdings:
        bucket = classify_fund_bucket(item.category, item.sub_category, item.name)
        if bucket is None:
            continue
        values_by_bucket[bucket] += item.value
        key = (item.member_id, item.security_id)
        values_by_fund[key] += item.value
        holdings_by_fund[key] = item

    all_reference_by_id = {item.security_id: item for item in references}
    all_reference_by_id.update({item.security_id: item for item in selected_holdings})
    returns = _returns_by_security(all_reference_by_id, histories, as_of)
    medians = _subcategory_medians(all_reference_by_id, returns)
    bottom_quartiles = _subcategory_bottom_quartiles(all_reference_by_id, returns)
    checks: list[dict[str, Any]] = []
    lagging_candidates: list[tuple[str, str, bool]] = []
    previous = previous_lagging or set()

    direct_by_family = {
        _scheme_family(reference.name): reference
        for reference in references
        if reference.plan_type == "direct"
    }
    for key, holding in holdings_by_fund.items():
        member = member_by_id.get(holding.member_id, FundMember(holding.member_id))
        if holding.plan_type == "regular" and not _is_denmark_resident(member):
            direct = direct_by_family.get(_scheme_family(holding.name))
            if direct is not None:
                _regular_plan_check(
                    checks, holding, direct, member, values_by_fund[key], as_of, sale_ledger
                )
        if holding.option_type == "idcw":
            _idcw_check(
                checks,
                holding,
                member,
                values_by_fund[key],
                sips.get(key, Decimal(0)),
                as_of,
                Decimal(str(limits["idcw_switch_cost_limit_inr"])),
                sale_ledger,
            )

        result_3y = returns.get(holding.security_id, {}).get(36)
        result_5y = returns.get(holding.security_id, {}).get(60)
        bottom_3y = bottom_quartiles.get((holding.sub_category or "", 36))
        bottom_5y = bottom_quartiles.get((holding.sub_category or "", 60))
        is_bottom = (
            result_3y is not None
            and result_5y is not None
            and bottom_3y is not None
            and bottom_5y is not None
            and result_3y <= bottom_3y
            and result_5y <= bottom_5y
        )
        lagging_candidates.append((holding.member_id, holding.security_id, is_bottom))
        if is_bottom and (holding.member_id, holding.security_id) in previous:
            checks.append(
                _check(
                    "lagging_fund",
                    holding.member_id,
                    values_by_fund[key],
                    {
                        "security_id": holding.security_id,
                        "outcome": "review",
                        "automatic_sell": False,
                        "return_3y_pct": _percentage_text(result_3y),
                        "return_5y_pct": _percentage_text(result_5y),
                    },
                    _action(
                        "REVIEW_FUND",
                        holding.member_id,
                        holding.security_id,
                        values_by_fund[key],
                        "lagging_fund",
                    ),
                )
            )

    if selected_holdings:
        _too_many_checks(
            checks, selected_holdings, values_by_fund, limits, member_by_id, as_of, sale_ledger
        )
        _closet_index_check(checks, selected_holdings, references, returns, sips, member_by_id)
        _sector_heavy_check(checks, selected_holdings, limits)
        _debt_note_check(checks, selected_holdings, best_debt_option)

    early_warning = (
        _project_warning(
            values_by_bucket,
            sips,
            selected_holdings,
            member_by_id,
            target,
            rules,
            band_rules or BandRules(),
            as_of,
        )
        if selected_holdings
        else None
    )
    if (
        early_warning is not None
        and early_warning["direction"] == "over"
        and early_warning["bucket"] in _INDIA_EQUITY_BUCKETS
    ):
        bucket = str(early_warning["bucket"])
        if any(
            classify_fund_bucket(item.category, item.sub_category, item.name) == bucket
            and sips.get((item.member_id, item.security_id), Decimal(0)) > 0
            for item in selected_holdings
        ):
            target_bucket = _redirect_bucket(values_by_bucket, target["indian_equity_pct"])
            target_fund = _best_fund_for_bucket(selected_holdings, values_by_fund, target_bucket)
            for (member_id, security_id), amount in sips.items():
                holding = holdings_by_fund.get((member_id, security_id))
                if (
                    holding is None
                    or classify_fund_bucket(holding.category, holding.sub_category, holding.name)
                    != bucket
                ):
                    continue
                member = member_by_id.get(member_id, FundMember(member_id))
                redirected_to = (
                    member.redirect_member_id if _is_denmark_resident(member) else member_id
                )
                if redirected_to is None:
                    continue
                clubbing_note = member.clubbing_note if _is_denmark_resident(member) else None
                action = _action(
                    "REDIRECT_SIP",
                    redirected_to,
                    security_id,
                    amount,
                    "projected_band_breach",
                    target_security_id=target_fund,
                    source_member_id=member_id,
                    clubbing_note=clubbing_note,
                )
                checks.append(
                    _check(
                        "projected_band_breach",
                        member_id,
                        amount,
                        {"bucket": bucket, "redirect_to_bucket": target_bucket},
                        action,
                    )
                )

    if selected_holdings:
        _residency_details(checks, selected_members, sips, selected_holdings)
    ordered_checks = sorted(
        checks,
        key=lambda item: (
            -Decimal(item["money_at_stake"]),
            item["reason_code"],
            item.get("member_id") or "",
        ),
    )
    actions = [item["action"] for item in ordered_checks if item.get("action") is not None]
    unique_actions: list[dict[str, Any]] = []
    seen_actions: set[tuple[str, str | None, str | None]] = set()
    for action in actions:
        signature = (action["kind"], action.get("member_id"), action.get("security_id"))
        if signature not in seen_actions and Decimal(action["amount"]) > 0:
            seen_actions.add(signature)
            unique_actions.append(action)

    fund_details = []
    for holding in selected_holdings:
        result = returns.get(holding.security_id, {})
        median_by_period = {
            months: medians.get((holding.sub_category or "", months)) for months in (36, 60)
        }
        fund_details.append(
            {
                "security_id": holding.security_id,
                "member_id": holding.member_id,
                "name": holding.name,
                "sub_category": holding.sub_category,
                "plan_type": holding.plan_type,
                "option": holding.option_type,
                "value_inr": _money_text(values_by_fund[(holding.member_id, holding.security_id)]),
                "ter_pct": _percentage_text(holding.ter),
                "ter_status": "available" if holding.ter is not None else "ter_unavailable",
                "return_3y_pct": _percentage_text(result.get(36)),
                "median_3y_pct": _percentage_text(median_by_period[36]),
                "return_5y_pct": _percentage_text(result.get(60)),
                "median_5y_pct": _percentage_text(median_by_period[60]),
                "tax_estimate_available": holding.tax_lots_complete,
            }
        )
    mix = _mix_details(values_by_bucket, target, indian_sleeve_target)
    has_holdings = bool(selected_holdings)
    # A check without an action (e.g. "keep existing funds, review local tax") is a
    # note, not a change; calling it "One change suggested" misled in a real walkthrough.
    needs_change = any(check.get("action") is not None for check in ordered_checks)
    summary_key = (
        "fund_check.attention"
        if needs_change
        else "fund_check.note"
        if ordered_checks
        else "fund_check.good"
        if has_holdings
        else "fund_check.upload"
    )
    return FundCheckResult(
        status="attention" if needs_change else "fine",
        summary_key=summary_key,
        checks=tuple(ordered_checks[:3]),
        actions=tuple(unique_actions[:3]),
        early_warning=early_warning,
        details={
            "funds": fund_details,
            "mix": mix,
            "notes": list(
                dict.fromkeys(
                    note for check in ordered_checks for note in check["details"].get("notes", [])
                )
            ),
            "sips": [
                {
                    "member_id": member_id,
                    "security_id": security_id,
                    "monthly_amount_inr": _money_text(amount),
                }
                for (member_id, security_id), amount in sorted(sips.items())
            ],
            "projection_verified": False,
            "projection_source": rules["source_note"],
            "denmark_note_verified": False
            if any(_is_denmark_resident(member) for member in selected_members)
            else None,
        },
        lagging_candidates=tuple(lagging_candidates),
        sip_amounts=tuple(
            (member_id, security_id, amount) for (member_id, security_id), amount in sips.items()
        ),
    )


def _regular_plan_check(
    checks: list[dict[str, Any]],
    holding: FundHolding,
    direct: FundReference,
    member: FundMember,
    value: Decimal,
    as_of: date,
    sale_ledger: _FundSaleLedger,
) -> None:
    details: dict[str, Any] = {"direct_security_id": direct.security_id}
    action = None
    yearly_saving = _MONEY_ZERO
    switch_cost = None
    payback = None
    if holding.ter is None or direct.ter is None:
        details["notes"] = ["ter_unavailable"]
    else:
        yearly_saving = round_paise(value * (holding.ter - direct.ter) / Decimal(100))
        details["yearly_saving_inr"] = _money_text(yearly_saving)
        if yearly_saving <= 0:
            return
        quote, estimate_note = _switch_cost(holding, member, as_of, sale_ledger)
        if quote is None:
            details["notes"] = [estimate_note or "tax_estimate_unavailable"]
        else:
            estimate = quote.estimate
            switch_cost = estimate.total_tax + _exit_load(value, holding.exit_load_pct)
            payback = switch_cost / yearly_saving if yearly_saving else None
            details.update(
                {
                    "switch_cost_inr": _money_text(switch_cost),
                    "tax_inr": _money_text(estimate.total_tax),
                    "ltcg_exemption_used_inr": _money_text(estimate.exemption_used),
                    "ltcg_exemption_remaining_inr": _money_text(quote.ltcg_exemption_remaining),
                    "payback_years": _percentage_text(payback),
                    "exit_load_inr": _money_text(_exit_load(value, holding.exit_load_pct)),
                }
            )
            max_payback = Decimal(str(_rules()["limits"]["regular_plan_max_payback_years"]))
            if payback is not None and payback <= max_payback:
                action = _action(
                    "SWITCH_TO_DIRECT",
                    member.member_id,
                    holding.security_id,
                    value,
                    "regular_plan",
                    target_security_id=direct.security_id,
                    yearly_saving_inr=_money_text(yearly_saving),
                    tax_inr=_money_text(estimate.total_tax),
                    exemption_used_inr=_money_text(estimate.exemption_used),
                    ltcg_exemption_used_inr=_money_text(estimate.exemption_used),
                    ltcg_exemption_remaining_inr=_money_text(quote.ltcg_exemption_remaining),
                )
                sale_ledger.record(member, quote)
    if action is None and switch_cost is not None and payback is not None:
        details["payback_gate"] = "over_limit"
    checks.append(_check("regular_plan", member.member_id, yearly_saving, details, action))


def _idcw_check(
    checks: list[dict[str, Any]],
    holding: FundHolding,
    member: FundMember,
    value: Decimal,
    sip_amount: Decimal,
    as_of: date,
    switch_limit: Decimal,
    sale_ledger: _FundSaleLedger,
) -> None:
    details: dict[str, Any] = {
        "recommendation": "new_money_not_allowed"
        if _is_denmark_resident(member)
        else "growth_for_new_money"
    }
    action = None
    quote, estimate_note = _switch_cost(holding, member, as_of, sale_ledger)
    if quote is None:
        details["notes"] = [estimate_note or "tax_estimate_unavailable"]
        switch_cost = None
    else:
        tax_estimate = quote.estimate
        switch_cost = tax_estimate.total_tax + _exit_load(value, holding.exit_load_pct)
        details["switch_cost_inr"] = _money_text(switch_cost)
        details["ltcg_exemption_used_inr"] = _money_text(tax_estimate.exemption_used)
        details["ltcg_exemption_remaining_inr"] = _money_text(quote.ltcg_exemption_remaining)
        details["switch_allowed"] = switch_cost <= switch_limit
    if sip_amount > 0 and not _is_denmark_resident(member):
        action = _action(
            "REDIRECT_SIP",
            member.member_id,
            holding.security_id,
            sip_amount,
            "idcw_option",
            switch_allowed=switch_cost is not None and switch_cost <= switch_limit,
        )
    checks.append(_check("idcw_option", member.member_id, sip_amount or value, details, action))


def _switch_cost(
    holding: FundHolding,
    member: FundMember,
    as_of: date,
    sale_ledger: _FundSaleLedger,
) -> tuple[_FundSaleQuote | None, str | None]:
    if not holding.tax_lots_complete or not holding.tax_lots or holding.price is None:
        return None, "tax_estimate_unavailable"
    if holding.exit_load_pct is None:
        return None, "exit_load_unavailable"
    if as_of < _EQUITY_TAX_START:
        return None, "tax_rule_unverified_for_sale_date"
    sales = sale_ledger.available_sales(holding)
    if not sales:
        return None, "tax_estimate_unavailable"
    try:
        return sale_ledger.quote(member, sales, as_of), None
    except ValueError:
        return None, "tax_estimate_unavailable"


def _too_many_checks(
    checks: list[dict[str, Any]],
    holdings: Sequence[FundHolding],
    values_by_fund: Mapping[tuple[str, str], Decimal],
    limits: Mapping[str, Any],
    members: Mapping[str, FundMember],
    as_of: date,
    sale_ledger: _FundSaleLedger,
) -> None:
    by_member: dict[str, list[FundHolding]] = defaultdict(list)
    for item in holdings:
        by_member[item.member_id].append(item)
    for member_id, member_holdings in by_member.items():
        if _is_denmark_resident(members.get(member_id, FundMember(member_id))):
            continue
        groups: dict[str, list[FundHolding]] = defaultdict(list)
        for holding in member_holdings:
            if holding.sub_category:
                groups[holding.sub_category].append(holding)
        over_groups = [
            (subcategory, group)
            for subcategory, group in groups.items()
            if len(group) > int(limits["max_funds_per_sub_category"])
        ]
        if len(member_holdings) > int(limits["max_funds_total"]):
            group = sorted(
                member_holdings,
                key=lambda item: (
                    values_by_fund[(item.member_id, item.security_id)],
                    item.security_id,
                ),
            )
            over_groups.append(("total", group))
        for subcategory, group in over_groups:
            ordered = sorted(
                group,
                key=lambda item: (
                    -values_by_fund[(item.member_id, item.security_id)],
                    item.security_id,
                ),
            )
            redundant_candidates = (
                ordered[int(limits["max_funds_per_sub_category"]) :]
                if subcategory != "total"
                else ordered[int(limits["max_funds_total"]) :]
            )
            redundant = [
                holding for holding in redundant_candidates if sale_ledger.has_unsold_lots(holding)
            ]
            if not redundant:
                continue
            amount = sum(
                (
                    sale_ledger.remaining_value(
                        item, values_by_fund[(item.member_id, item.security_id)]
                    )
                    for item in redundant
                ),
                Decimal(0),
            )
            details: dict[str, Any] = {
                "sub_category": None if subcategory == "total" else subcategory,
                "fund_count": len(group),
                "limit": int(limits["max_funds_total"])
                if subcategory == "total"
                else int(limits["max_funds_per_sub_category"]),
                "keep_security_id": ordered[0].security_id,
            }
            member = members.get(member_id, FundMember(member_id))
            quote, exit_load, note = _consolidation_cost(redundant, member, as_of, sale_ledger)
            action = None
            if quote is None:
                details["notes"] = [note or "tax_estimate_unavailable"]
            else:
                estimate = quote.estimate
                switch_cost = round_paise(estimate.total_tax + exit_load)
                details.update(
                    {
                        "tax_inr": _money_text(estimate.total_tax),
                        "exit_load_inr": _money_text(exit_load),
                        "switch_cost_inr": _money_text(switch_cost),
                        "ltcg_exemption_used_inr": _money_text(estimate.exemption_used),
                        "ltcg_exemption_remaining_inr": _money_text(quote.ltcg_exemption_remaining),
                    }
                )
                action = _action(
                    "CONSOLIDATE_FUNDS",
                    member_id,
                    redundant[0].security_id,
                    amount,
                    "too_many_funds",
                    target_security_id=ordered[0].security_id,
                    tax_inr=_money_text(estimate.total_tax),
                    switch_cost_inr=_money_text(switch_cost),
                    exemption_used_inr=_money_text(estimate.exemption_used),
                    ltcg_exemption_used_inr=_money_text(estimate.exemption_used),
                    ltcg_exemption_remaining_inr=_money_text(quote.ltcg_exemption_remaining),
                )
                sale_ledger.record(member, quote)
            checks.append(
                _check(
                    "too_many_funds",
                    member_id,
                    amount,
                    details,
                    action,
                )
            )


def _consolidation_cost(
    holdings: Sequence[FundHolding],
    member: FundMember,
    as_of: date,
    sale_ledger: _FundSaleLedger,
) -> tuple[_FundSaleQuote | None, Decimal, str | None]:
    if any(
        not holding.tax_lots_complete or not holding.tax_lots or holding.price is None
        for holding in holdings
    ):
        return None, Decimal(0), "tax_estimate_unavailable"
    if any(holding.exit_load_pct is None for holding in holdings):
        return None, Decimal(0), "exit_load_unavailable"
    if as_of < _EQUITY_TAX_START:
        return None, Decimal(0), "tax_rule_unverified_for_sale_date"
    sales = [sale for holding in holdings for sale in sale_ledger.available_sales(holding)]
    if not sales:
        return None, Decimal(0), "tax_estimate_unavailable"
    try:
        quote = sale_ledger.quote(member, sales, as_of)
    except ValueError:
        return None, Decimal(0), "tax_estimate_unavailable"
    exit_load = sum(
        (
            _exit_load(
                sale_ledger.remaining_value(holding, holding.value),
                holding.exit_load_pct,
            )
            for holding in holdings
            if holding.exit_load_pct is not None
        ),
        Decimal(0),
    )
    return quote, round_paise(exit_load), None


def _closet_index_check(
    checks: list[dict[str, Any]],
    holdings: Sequence[FundHolding],
    references: Sequence[FundReference],
    returns: Mapping[str, Mapping[int, Decimal]],
    sips: Mapping[tuple[str, str], Decimal],
    members: Mapping[str, FundMember],
) -> None:
    indexes = [
        item
        for item in references
        if item.sub_category == "Index Funds/ETFs" and "nifty 50" in item.name.casefold()
    ]
    index = next((item for item in indexes if 36 in returns.get(item.security_id, {})), None)
    if index is None:
        return
    index_return = returns[index.security_id][36]
    by_member: dict[str, list[FundHolding]] = defaultdict(list)
    for holding in holdings:
        if holding.sub_category == "Large Cap" and holding.plan_type in {"direct", "regular"}:
            by_member[holding.member_id].append(holding)
    for member_id, active_funds in by_member.items():
        trailing = [
            item
            for item in active_funds
            if 36 in returns.get(item.security_id, {})
            and returns[item.security_id][36] < index_return
            and item.security_id
            not in {
                reference.security_id
                for reference in references
                if "index" in (reference.sub_category or "").casefold()
            }
        ]
        if len(trailing) < 2:
            continue
        amount = sum(
            (sips.get((member_id, item.security_id), Decimal(0)) for item in trailing), Decimal(0)
        )
        is_denmark = _is_denmark_resident(members.get(member_id, FundMember(member_id)))
        checks.append(
            _check(
                "closet_index",
                member_id,
                amount,
                {
                    "fund_security_ids": [item.security_id for item in trailing],
                    "index_security_id": index.security_id,
                    "fund_return_3y_pct": [
                        _percentage_text(returns[item.security_id][36]) for item in trailing
                    ],
                    "index_return_3y_pct": _percentage_text(index_return),
                },
                _action(
                    "REDIRECT_SIP",
                    member_id,
                    trailing[0].security_id,
                    amount,
                    "closet_index",
                    target_security_id=index.security_id,
                )
                if amount > 0 and not is_denmark
                else None,
            )
        )


def _sector_heavy_check(
    checks: list[dict[str, Any]],
    holdings: Sequence[FundHolding],
    limits: Mapping[str, Any],
) -> None:
    limit_pct = Decimal(str(limits["sector_thematic_max_pct_of_indian_equity"]))
    member_values: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for holding in holdings:
        bucket = classify_fund_bucket(holding.category, holding.sub_category, holding.name)
        if bucket in _INDIA_EQUITY_BUCKETS:
            member_values[holding.member_id][bucket] += holding.value

    for member_id, values in member_values.items():
        domestic_equity = sum(values.values(), Decimal(0))
        sector_value = values.get("sector_thematic", Decimal(0))
        excess = sector_value - domestic_equity * limit_pct / Decimal(100)
        if domestic_equity <= 0 or excess <= 0:
            continue
        checks.append(
            _check(
                "sector_heavy",
                member_id,
                excess,
                {
                    "sector_thematic_pct": _percentage_text(sector_value * 100 / domestic_equity),
                    "limit_pct": _percentage_text(limit_pct),
                },
                None,
            )
        )


def _debt_note_check(
    checks: list[dict[str, Any]],
    holdings: Sequence[FundHolding],
    best_debt_option: Mapping[str, str] | None,
) -> None:
    post_2023 = [
        item
        for item in holdings
        if classify_fund_bucket(item.category, item.sub_category, item.name) == "debt"
        and any(lot.acquired_on >= date(2023, 4, 1) for lot in item.tax_lots)
    ]
    if not post_2023:
        return
    checks.append(
        _check(
            "debt_fund_vs_fd",
            post_2023[0].member_id,
            Decimal(0),
            {
                "tax_treatment": "slab_rate",
                "comparison": dict(best_debt_option or {}),
                "note_only": True,
            },
            None,
        )
    )


def _residency_details(
    checks: list[dict[str, Any]],
    members: Sequence[FundMember],
    sips: Mapping[tuple[str, str], Decimal],
    holdings: Sequence[FundHolding],
) -> None:
    holdings_by_member: dict[str, list[FundHolding]] = defaultdict(list)
    for holding in holdings:
        holdings_by_member[holding.member_id].append(holding)
    for member in members:
        if not _is_denmark_resident(member):
            continue
        existing_value = sum(
            (holding.value for holding in holdings_by_member.get(member.member_id, [])), Decimal(0)
        )
        sip_amount = sum(
            (
                amount
                for (member_id, _security_id), amount in sips.items()
                if member_id == member.member_id
            ),
            Decimal(0),
        )
        checks.append(
            _check(
                "dk_fund_tax_review",
                member.member_id,
                Decimal(0),
                {
                    "existing_funds": "hold_by_default",
                    "yearly_danish_tax_note": {"verified": False},
                    "new_indian_money_allowed": False,
                    "redirect_member_id": member.redirect_member_id,
                    "redirected_monthly_sip_inr": _money_text(sip_amount)
                    if member.redirect_member_id
                    else None,
                    "existing_value_inr": _money_text(existing_value),
                    "clubbing_note": member.clubbing_note,
                },
                None,
            )
        )
        if member.redirect_member_id is not None:
            for (source_member_id, security_id), amount in sips.items():
                if source_member_id != member.member_id:
                    continue
                checks.append(
                    _check(
                        "dk_fund_tax_review",
                        member.member_id,
                        amount,
                        {
                            "new_indian_money_allowed": False,
                            "redirect_member_id": member.redirect_member_id,
                            "clubbing_note": member.clubbing_note,
                        },
                        _action(
                            "REDIRECT_SIP",
                            member.redirect_member_id,
                            security_id,
                            amount,
                            "dk_fund_tax_review",
                            source_member_id=member.member_id,
                            clubbing_note=member.clubbing_note,
                        ),
                    )
                )


def _project_warning(
    current: Mapping[str, Decimal],
    sips: Mapping[tuple[str, str], Decimal],
    holdings: Sequence[FundHolding],
    members: Mapping[str, FundMember],
    target: Mapping[str, Any],
    rules: Mapping[str, Any],
    band_rules: BandRules,
    as_of: date,
) -> dict[str, Any] | None:
    values = {bucket: current.get(bucket, Decimal(0)) for bucket in rules["monthly_return_pct"]}
    contribution_by_bucket: dict[str, Decimal] = defaultdict(Decimal)
    for (member_id, security_id), amount in sips.items():
        member = members.get(member_id)
        if member is not None and _is_denmark_resident(member):
            continue
        holding = next(
            (
                item
                for item in holdings
                if item.member_id == member_id and item.security_id == security_id
            ),
            None,
        )
        bucket = (
            classify_fund_bucket(holding.category, holding.sub_category, holding.name)
            if holding is not None
            else None
        )
        if bucket is not None:
            contribution_by_bucket[bucket] += amount
    warning = _warning_for_values(values, target, band_rules, month=0)
    if warning is not None:
        return {**warning, "verified": False, "source_note": rules["source_note"]}
    for month in range(1, int(rules["limits"]["projection_months"]) + 1):
        for bucket, monthly_pct in rules["monthly_return_pct"].items():
            value = values.get(bucket, Decimal(0))
            values[bucket] = value * (
                Decimal(1) + Decimal(str(monthly_pct)) / Decimal(100)
            ) + contribution_by_bucket.get(bucket, Decimal(0))
        warning = _warning_for_values(values, target, band_rules, month=month)
        if warning is not None:
            return {**warning, "verified": False, "source_note": rules["source_note"]}
    return None


def _warning_for_values(
    values: Mapping[str, Decimal], target: Mapping[str, Any], rules: BandRules, month: int
) -> dict[str, Any] | None:
    equity_values = {bucket: values.get(bucket, Decimal(0)) for bucket in _INDIA_EQUITY_BUCKETS}
    equity_total = sum(equity_values.values(), Decimal(0))
    asset_values = {
        "equity": equity_total,
        "hybrid": values.get("hybrid", Decimal(0)),
        "debt": values.get("debt", Decimal(0)),
    }
    candidates: list[tuple[int, str, str]] = []
    for target_key, actual_values, prefix in (
        ("asset_mix_pct", asset_values, "asset_"),
        ("indian_equity_pct", equity_values, ""),
    ):
        total = sum(actual_values.values(), Decimal(0))
        if total <= 0:
            continue
        target_pcts = {key: Decimal(str(value)) for key, value in target[target_key].items()}
        current_pcts = {key: value * 100 / total for key, value in actual_values.items()}
        snapshot = SimpleNamespace(
            investable_total=total,
            target_percent=target_pcts,
            current_percent=current_pcts,
            target_rupees={key: total * value / 100 for key, value in target_pcts.items()},
            current_rupees=dict(actual_values),
        )
        for breach in band_breaches(snapshot, rules):
            candidates.append((breach.excess_amount, f"{prefix}{breach.sleeve}", breach.direction))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _amount, bucket, direction = candidates[0]
    return {"bucket": bucket.removeprefix("asset_"), "months_until": month, "direction": direction}


def _mix_details(
    values: Mapping[str, Decimal], target: Mapping[str, Any], indian_sleeve_target: Decimal | None
) -> dict[str, Any]:
    totals = {
        "equity": sum(
            (values.get(bucket, Decimal(0)) for bucket in _INDIA_EQUITY_BUCKETS), Decimal(0)
        ),
        "hybrid": values.get("hybrid", Decimal(0)),
        "debt": values.get("debt", Decimal(0)),
    }
    total = sum(totals.values(), Decimal(0))
    return {
        "current_inr": {key: _money_text(value) for key, value in totals.items()},
        "current_pct": {
            key: _percentage_text(value * 100 / total if total else Decimal(0))
            for key, value in totals.items()
        },
        "target_pct": {key: str(value) for key, value in target["asset_mix_pct"].items()},
        "indian_equity_current_inr": {
            bucket: _money_text(values.get(bucket, Decimal(0))) for bucket in _INDIA_EQUITY_BUCKETS
        },
        "indian_equity_target_pct": {
            key: str(value) for key, value in target["indian_equity_pct"].items()
        },
        "indian_sleeve_target_inr": _money_text(indian_sleeve_target)
        if indian_sleeve_target is not None
        else None,
        "indian_sleeve_target_source": "existing_plan"
        if indian_sleeve_target is not None
        else "current_funds",
    }


def _returns_by_security(
    references: Mapping[str, FundReference],
    histories: Mapping[str, Sequence[tuple[date, Decimal]]],
    as_of: date,
) -> dict[str, dict[int, Decimal]]:
    returns: dict[str, dict[int, Decimal]] = {}
    for security_id in references:
        points = sorted((day, nav) for day, nav in histories.get(security_id, ()) if day <= as_of)
        if len(points) < 37:
            continue
        latest_date, latest_nav = points[-1]
        by_period: dict[int, Decimal] = {}
        for months in (36, 60):
            anchor = _add_months(latest_date, -months)
            earlier = [nav for day, nav in points if day <= anchor]
            if len(earlier) < 1:
                continue
            start_nav = earlier[-1]
            if start_nav > 0:
                by_period[months] = (latest_nav / start_nav - 1) * 100
        if by_period:
            returns[security_id] = by_period
    return returns


def _subcategory_medians(
    references: Mapping[str, FundReference], returns: Mapping[str, Mapping[int, Decimal]]
) -> dict[tuple[str, int], Decimal]:
    grouped: dict[tuple[str, int], list[Decimal]] = defaultdict(list)
    for security_id, reference in references.items():
        if not reference.sub_category:
            continue
        for months, value in returns.get(security_id, {}).items():
            grouped[(reference.sub_category, months)].append(value)
    return {key: Decimal(str(median(values))) for key, values in grouped.items() if values}


def _subcategory_bottom_quartiles(
    references: Mapping[str, FundReference], returns: Mapping[str, Mapping[int, Decimal]]
) -> dict[tuple[str, int], Decimal]:
    grouped: dict[tuple[str, int], list[Decimal]] = defaultdict(list)
    for security_id, reference in references.items():
        if not reference.sub_category:
            continue
        for months, value in returns.get(security_id, {}).items():
            grouped[(reference.sub_category, months)].append(value)

    quartiles: dict[tuple[str, int], Decimal] = {}
    for key, values in grouped.items():
        if len(values) < 4:
            continue
        ordered = sorted(values)
        position = Decimal(len(ordered) - 1) / Decimal(4)
        lower_index = int(position)
        fraction = position - lower_index
        lower = ordered[lower_index]
        upper = ordered[lower_index + 1]
        quartiles[key] = lower + (upper - lower) * fraction
    return quartiles


def _check(
    reason_code: str,
    member_id: str,
    money_at_stake: Decimal,
    details: dict[str, Any],
    action: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "reason_code": reason_code,
        "member_id": member_id,
        "money_at_stake": _money_text(money_at_stake),
        "details": details,
        "action": action,
    }


def _action(
    kind: str,
    member_id: str,
    security_id: str,
    amount: Decimal,
    reason_code: str,
    **details: Any,
) -> dict[str, Any]:
    rounded = (amount / Decimal(100)).quantize(Decimal(1), rounding=ROUND_HALF_UP) * Decimal(100)
    return {
        "kind": kind,
        "member_id": member_id,
        "security_id": security_id,
        "amount": _money_text(rounded),
        "reason_code": reason_code,
        **details,
    }


def _exit_load(value: Decimal, exit_load_pct: Decimal | None) -> Decimal:
    return round_paise(value * exit_load_pct / 100) if exit_load_pct is not None else Decimal(0)


def _profile_key(profile: str | None, rules: Mapping[str, Any]) -> str:
    value = (profile or "moderate").casefold()
    if value in rules["targets"]:
        return value
    aliases = {"balanced": "moderate", "growth": "growth", "aggressive": "aggressive"}
    return aliases.get(value, "moderate")


def _scheme_family(name: str) -> str:
    value = name.casefold()
    for term in (
        "direct",
        "regular",
        "plan",
        "growth",
        "idcw",
        "dividend",
        "payout",
        "reinvestment",
        "option",
    ):
        value = value.replace(term, " ")
    return " ".join(value.replace("-", " ").split())


def _redirect_bucket(values: Mapping[str, Decimal], targets: Mapping[str, Any]) -> str:
    return min(
        _INDIA_EQUITY_BUCKETS,
        key=lambda bucket: (
            values.get(bucket, Decimal(0)) / max(Decimal(str(targets[bucket])), Decimal("0.01")),
            bucket,
        ),
    )


def _best_fund_for_bucket(
    holdings: Sequence[FundHolding],
    values_by_fund: Mapping[tuple[str, str], Decimal],
    bucket: str,
) -> str | None:
    eligible = [
        item
        for item in holdings
        if classify_fund_bucket(item.category, item.sub_category, item.name) == bucket
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (values_by_fund[(item.member_id, item.security_id)], item.security_id),
    ).security_id


def _is_denmark_resident(member: FundMember) -> bool:
    return member.residency_country == "DK"


def _month_index(value: date) -> int:
    return value.year * 12 + value.month


def _add_months(value: date, offset: int) -> date:
    month_index = _month_index(value) - 1 + offset
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    day = min(value.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    return monthrange(year, month)[1]


def _money_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(round_paise(value), "f")


def _percentage_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")
