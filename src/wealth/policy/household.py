"""Per-member tax, eligibility, limits, and whole-household wealth views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

from wealth.core.money import round_paise
from wealth.policy.debt_picker import load_debt_rules
from wealth.policy.denmark import ask_deposit_room_dkk
from wealth.policy.residency import (
    eligibility_rule,
    load_residency_rules,
    residency_note_codes,
)
from wealth.policy.snapshot import WealthSnapshot
from wealth.policy.tax import TaxContext
from wealth.schemas.assets import AnyAsset, PostOfficeAsset, PPFAsset
from wealth.schemas.common import LiquidityClass
from wealth.schemas.profile import (
    FinancialContext,
    InvestingPrefs,
    MemberTax,
    Residency,
)
from wealth.valuation.base import Valuation

Relationship = Literal["self", "spouse", "child", "parent", "other"]
EligibilityStatus = Literal["allowed", "conditional", "not_allowed"]
ValuedAsset = tuple[AnyAsset, Valuation]


@dataclass(frozen=True)
class MemberProfile:
    """Minimal persisted member inputs passed into deterministic policy code."""

    id: str
    name: str
    relationship: Relationship
    is_self: bool
    residency: Residency = field(default_factory=Residency)
    tax: MemberTax = field(default_factory=MemberTax)
    investing: InvestingPrefs = field(default_factory=InvestingPrefs)
    realised_equity_ltcg: Decimal = Decimal(0)
    realised_equity_stcg: Decimal = Decimal(0)
    ask_used_dkk: Decimal | None = None
    ask_deposit_fx_missing: bool = False
    has_prior_ask_activity: bool = False


@dataclass(frozen=True)
class MemberBreakdown:
    """Valued wealth totals attributed to a member's primary ownership."""

    member_id: str
    name: str
    relationship: Relationship
    snapshot: WealthSnapshot

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.member_id,
            "name": self.name,
            "relationship": self.relationship,
            "net_worth": format(self.snapshot.net_worth, "f"),
            "sleeves": {
                sleeve.value: format(value, "f")
                for sleeve, value in self.snapshot.totals_by_sleeve.items()
            },
            "liquidity": {
                liquidity.value: format(value, "f")
                for liquidity, value in self.snapshot.totals_by_liquidity.items()
            },
            "locked": {
                sleeve.value: format(value, "f")
                for sleeve, value in self.snapshot.locked_by_sleeve.items()
            },
        }


@dataclass(frozen=True)
class HouseholdSnapshot:
    """The whole-household C11 snapshot with the same totals split by owner."""

    wealth: WealthSnapshot
    members: tuple[MemberBreakdown, ...]

    @property
    def net_worth(self) -> Decimal:
        return self.wealth.net_worth

    def member(self, member_id: str) -> MemberBreakdown | None:
        return next((item for item in self.members if item.member_id == member_id), None)


@dataclass(frozen=True)
class MemberContext:
    """Policy-ready facts for one member, including their own tax and limit room."""

    member: MemberProfile
    snapshot: WealthSnapshot
    tax_context: TaxContext
    financial_context: FinancialContext
    eligibility: Mapping[str, bool]
    eligibility_status: Mapping[str, EligibilityStatus]
    eligibility_conditions: Mapping[str, str]
    limits_room: Mapping[str, Decimal | None]
    notes: tuple[str, ...] = ()

    @property
    def member_id(self) -> str:
        return self.member.id

    @property
    def name(self) -> str:
        return self.member.name

    @property
    def relationship(self) -> Relationship:
        return self.member.relationship

    @property
    def residency(self) -> Residency:
        return self.member.residency

    @property
    def new_money(self) -> Decimal:
        return self.member.investing.new_money_this_week

    @property
    def monthly_sip_capacity(self) -> Decimal:
        return self.member.investing.monthly_sip_capacity


def build_household_snapshot(
    valued_assets: Iterable[ValuedAsset],
    market_holdings: Iterable[tuple[str, Decimal, LiquidityClass]],
    member_holdings: Iterable[tuple[str, str, Decimal, LiquidityClass]],
    members: Iterable[MemberProfile],
    *,
    self_member_id: str,
) -> HouseholdSnapshot:
    """Build whole-household and per-member C11 snapshots from the same facts."""
    assets = tuple(valued_assets)
    all_market = tuple(market_holdings)
    per_member_market = tuple(member_holdings)
    profiles = tuple(members)
    wealth = WealthSnapshot.from_assets(assets, all_market)
    breakdowns: list[MemberBreakdown] = []
    for member in profiles:
        member_assets = tuple(
            item for item in assets if _asset_member_id(item[0], self_member_id) == member.id
        )
        member_market = tuple(
            (sleeve, value, liquidity)
            for owner_id, sleeve, value, liquidity in per_member_market
            if owner_id == member.id
        )
        breakdowns.append(
            MemberBreakdown(
                member_id=member.id,
                name=member.name,
                relationship=member.relationship,
                snapshot=WealthSnapshot.from_assets(member_assets, member_market),
            )
        )
    return HouseholdSnapshot(wealth=wealth, members=tuple(breakdowns))


def build_member_contexts(
    members: Iterable[MemberProfile],
    household: HouseholdSnapshot,
    valued_assets: Iterable[ValuedAsset],
    *,
    as_of_year: int,
    fx_rates: Mapping[str, Decimal],
    fallback_tax_slab_pct: Decimal | None = None,
    fallback_tax_regime: str | None = None,
    fallback_city_tier: int | None = None,
    has_indian_funds_by_member: Mapping[str, bool] | None = None,
) -> tuple[MemberContext, ...]:
    """Build per-member tax contexts, eligibility flags, and annual room remaining."""
    assets = tuple(valued_assets)
    funding_notes = has_indian_funds_by_member or {}
    selected_members = tuple(members)
    if not selected_members:
        return ()
    self_member = next((member for member in selected_members if member.is_self), None)
    fallback_member_id = self_member.id if self_member is not None else selected_members[0].id
    contexts: list[MemberContext] = []
    debt_rules = load_debt_rules()
    ppf_annual_limit = _limit_for(debt_rules, "ppf", "annual_limit", Decimal("150000"))
    scss_cap = _limit_for(debt_rules, "scss", "max_amount", Decimal("3000000"))
    pomis_cap = _limit_for(debt_rules, "pomis", "max_amount", Decimal("900000"))

    for member in selected_members:
        breakdown = household.member(member.id)
        snapshot = breakdown.snapshot if breakdown is not None else WealthSnapshot()
        tax_slab = member.tax.tax_slab_pct
        if tax_slab is None:
            tax_slab = (
                fallback_tax_slab_pct
                if member.is_self and fallback_tax_slab_pct is not None
                else Decimal("30")
            )
        tax_regime = member.tax.tax_regime
        if tax_regime is None and member.is_self:
            tax_regime = fallback_tax_regime
        annual_income = _income_inr(member, fx_rates)
        member_financial = FinancialContext(
            monthly_expenses=Decimal(0),
            annual_income=annual_income,
            tax_regime=tax_regime,
            employment_type=member.tax.employment_type,
            dependants=0,
        )
        age = (
            max(0, as_of_year - (member.tax.birth_year or member.investing.birth_year))
            if (member.tax.birth_year or member.investing.birth_year)
            else None
        )
        eligibility_status: dict[str, EligibilityStatus] = {}
        eligibility_conditions: dict[str, str] = {}
        instrument_rules = load_residency_rules()["instruments"]
        for instrument in instrument_rules:
            status, condition = _eligibility_for_member(instrument, member.residency)
            eligibility_status[instrument] = status
            if condition is not None:
                eligibility_conditions[instrument] = condition
        gold_status, gold_condition = _eligibility_for_member("gold_etf", member.residency)
        eligibility_status["gold_etf"] = gold_status
        if gold_condition is not None:
            eligibility_conditions["gold_etf"] = gold_condition
        eligibility_status["lrs"] = "allowed" if member.residency.status == "RI" else "not_allowed"
        eligibility_status["ask"] = "allowed" if member.residency.country == "DK" else "not_allowed"
        if age is None or age < 60:
            eligibility_status["scss"] = "not_allowed"
            eligibility_conditions.pop("scss", None)

        ppf_contributions = sum(
            (
                asset.yearly_contribution or Decimal(0)
                for asset, _ in assets
                if _asset_member_id(asset, fallback_member_id) == member.id
                and isinstance(asset, PPFAsset)
            ),
            Decimal(0),
        )
        scss_principal = sum(
            (
                asset.amount
                for asset, _ in assets
                if _asset_member_id(asset, fallback_member_id) == member.id
                and isinstance(asset, PostOfficeAsset)
                and asset.scheme == "scss"
            ),
            Decimal(0),
        )
        pomis_principal = sum(
            (
                asset.amount
                for asset, _ in assets
                if _asset_member_id(asset, fallback_member_id) == member.id
                and isinstance(asset, PostOfficeAsset)
                and asset.scheme == "mis"
            ),
            Decimal(0),
        )
        ssy_contributions = sum(
            (
                asset.annual_contribution or Decimal(0)
                for asset, _ in assets
                if _asset_member_id(asset, fallback_member_id) == member.id
                and isinstance(asset, PostOfficeAsset)
                and asset.scheme == "ssy"
            ),
            Decimal(0),
        )
        rules = load_residency_rules()
        ask_cap = Decimal(str(rules["denmark"]["ask"]["deposit_cap_dkk_2026"]))
        lrs_cap = Decimal(str(rules["limits"]["lrs_annual_usd"]))
        lrs_used_usd = Decimal(0)
        usd_inr = fx_rates.get("USD/INR")
        if usd_inr is not None and usd_inr > 0:
            lrs_used_usd = member.investing.lrs_used_this_fy / usd_inr
        ppf_room = (
            max(Decimal(0), ppf_annual_limit - ppf_contributions)
            if eligibility_status["ppf"] == "allowed"
            else Decimal(0)
        )
        ssy_room = (
            max(Decimal(0), Decimal(str(rules["limits"]["ssy_annual_inr"])) - ssy_contributions)
            if eligibility_status["ssy"] == "allowed"
            else Decimal(0)
        )
        scss_room = (
            max(Decimal(0), scss_cap - scss_principal)
            if eligibility_status["scss"] == "allowed"
            else Decimal(0)
        )
        pomis_room = (
            max(Decimal(0), pomis_cap - pomis_principal)
            if eligibility_status["pomis"] == "allowed"
            else Decimal(0)
        )
        nsc_room: Decimal | None = None if eligibility_status["nsc"] == "allowed" else Decimal(0)
        ask_start_value = member.investing.ask_value_start_of_year_dkk
        ask_start_value_missing = (
            member.residency.country == "DK"
            and member.has_prior_ask_activity
            and ask_start_value is None
        )
        ask_net_deposits = (
            member.ask_used_dkk
            if member.ask_used_dkk is not None
            else member.investing.ask_used_this_fy_dkk
        )
        ask_room = (
            Decimal(0)
            if member.ask_deposit_fx_missing or ask_start_value_missing
            else ask_deposit_room_dkk(
                ask_cap,
                ask_start_value or Decimal(0),
                ask_net_deposits,
            )
        )
        limits_room: dict[str, Decimal | None] = {
            "ppf_annual_inr": ppf_room,
            "lrs_annual_usd": max(Decimal(0), lrs_cap - lrs_used_usd)
            if eligibility_status["lrs"] == "allowed" and usd_inr is not None and usd_inr > 0
            else Decimal(0),
            "lrs_tcs_threshold_inr": Decimal(str(rules["limits"]["lrs_tcs_threshold_inr"])),
            "lrs_tcs_pct": Decimal(str(rules["limits"]["lrs_tcs_pct"])),
            "ask_deposit_dkk": ask_room
            if eligibility_status.get("ask") == "allowed"
            else Decimal(0),
            "scss_inr": scss_room,
            "pomis_inr": pomis_room,
            "ssy_annual_inr": ssy_room,
            "nsc_inr": nsc_room,
        }
        eligibility_status["lrs"] = "allowed" if member.residency.status == "RI" else "not_allowed"
        if member.residency.status == "RI" and limits_room["lrs_annual_usd"] == 0:
            eligibility_status["lrs"] = "not_allowed"
        eligibility_status["ask"] = "allowed" if member.residency.country == "DK" else "not_allowed"
        if member.residency.country == "DK" and limits_room["ask_deposit_dkk"] == 0:
            eligibility_status["ask"] = "not_allowed"
        for instrument, room_key in (
            ("ppf", "ppf_annual_inr"),
            ("ssy", "ssy_annual_inr"),
            ("scss", "scss_inr"),
            ("pomis", "pomis_inr"),
        ):
            room = limits_room[room_key]
            if room is not None and room <= 0:
                eligibility_status[instrument] = "not_allowed"
                eligibility_conditions.pop(instrument, None)
        if nsc_room == 0:
            eligibility_status["nsc"] = "not_allowed"
            eligibility_conditions.pop("nsc", None)
        eligibility: dict[str, bool] = {
            instrument: status == "allowed" for instrument, status in eligibility_status.items()
        }

        notes = residency_note_codes(
            member.residency,
            has_indian_funds=funding_notes.get(member.id, False),
        )
        if member.ask_deposit_fx_missing:
            notes = (*notes, "ask_deposit_fx_missing")
        if ask_start_value_missing:
            notes = (*notes, "ask_start_value_missing")
        contexts.append(
            MemberContext(
                member=member,
                snapshot=snapshot,
                tax_context=TaxContext(
                    slab_pct=tax_slab,
                    surcharge_pct=member.tax.surcharge_pct,
                    fy_realised_equity_ltcg=member.realised_equity_ltcg,
                    fy_realised_equity_stcg=member.realised_equity_stcg,
                    residency_status=member.residency.status,
                    us_person=member.residency.us_person,
                ),
                financial_context=member_financial,
                eligibility=MappingProxyType(eligibility),
                eligibility_status=MappingProxyType(eligibility_status),
                eligibility_conditions=MappingProxyType(eligibility_conditions),
                limits_room=MappingProxyType(
                    {
                        key: round_paise(value) if value is not None else None
                        for key, value in limits_room.items()
                    }
                ),
                notes=tuple(notes),
            )
        )
    return tuple(contexts)


def _asset_member_id(asset: AnyAsset, fallback: str) -> str:
    return asset.member_id or fallback


def _eligibility_for_member(
    instrument: str, residency: Residency
) -> tuple[EligibilityStatus, str | None]:
    """Return rulebook status while honoring the US-person pooled-fund exclusion."""
    if instrument == "gold_etf":
        if residency.country == "DK":
            return "not_allowed", "dk_fund_tax_review"
        return "allowed", None
    rule = eligibility_rule(instrument, residency)
    if rule is None:
        return "not_allowed", None
    key = instrument
    if key in {"in_mf", "in_etf"} and residency.us_person:
        return "not_allowed", None
    status = rule["new"]
    if status not in {"allowed", "conditional", "not_allowed"}:
        return "not_allowed", None
    rule_document = load_residency_rules()["instruments"].get(key, {})
    condition = rule_document.get("note_key") if status == "conditional" else None
    if instrument == "ucits_global_etf" and residency.status == "RI":
        condition = "ucits_via_lrs_broker"
    return status, condition


def _limit_for(
    rules: Mapping[str, object], instrument_key: str, limit_name: str, default: Decimal
) -> Decimal:
    instruments = rules.get("instruments")
    if not isinstance(instruments, list):
        return default
    for entry in instruments:
        if not isinstance(entry, Mapping) or entry.get("key") != instrument_key:
            continue
        eligibility = entry.get("eligibility")
        if isinstance(eligibility, Mapping) and eligibility.get(limit_name) is not None:
            return Decimal(str(eligibility[limit_name]))
    return default


def _income_inr(member: MemberProfile, fx_rates: Mapping[str, Decimal]) -> Decimal | None:
    amount = member.tax.annual_income
    if amount is None:
        return None
    currency = member.tax.annual_income_currency
    if currency == "INR":
        return amount
    rate = fx_rates.get(f"{currency}/INR")
    return round_paise(amount * rate) if rate is not None else None
