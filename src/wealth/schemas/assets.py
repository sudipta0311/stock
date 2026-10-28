"""Validated manual asset input models and their default classifications."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from wealth.schemas.common import (
    Compounding,
    LiquidityClass,
    LoanRatePct,
    MonthYear,
    PayoutType,
    Percentage,
    RatePct,
    Rupees,
    Sleeve,
)


class AssetBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    member_id: str | None = None
    joint: bool = False
    label: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=500)


class SavingsAsset(AssetBase):
    asset_type: Literal["savings"]
    bank: str = Field(min_length=1, max_length=120)
    balance: Rupees


class FixedDepositAsset(AssetBase):
    asset_type: Literal["fd"]
    bank: str = Field(min_length=1, max_length=120)
    principal: Rupees
    rate_pct: RatePct
    maturity: MonthYear
    start: MonthYear | None = None
    compounding: Compounding = Compounding.QUARTERLY
    payout: PayoutType = PayoutType.CUMULATIVE


class RecurringDepositAsset(AssetBase):
    asset_type: Literal["rd"]
    institution: str = Field(min_length=1, max_length=120)
    monthly_amount: Rupees
    rate_pct: RatePct
    maturity: MonthYear
    start: MonthYear | None = None


class PPFAsset(AssetBase):
    asset_type: Literal["ppf"]
    balance: Rupees
    opened_fy_start_year: int = Field(ge=1900, le=9999)
    yearly_contribution: Rupees | None = None
    balance_as_of: MonthYear | None = None
    held_at: str | None = Field(default=None, max_length=120)


class EPFAsset(AssetBase):
    asset_type: Literal["epf"]
    balance: Rupees
    monthly_employee: Rupees | None = None
    monthly_employer: Rupees | None = None
    vpf_monthly: Rupees | None = None
    balance_as_of: MonthYear | None = None


class NPSAsset(AssetBase):
    asset_type: Literal["nps"]
    tier1_balance: Rupees
    tier2_balance: Rupees | None = None
    equity_pct: Percentage | None = None
    corporate_debt_pct: Percentage | None = None
    govt_debt_pct: Percentage | None = None
    alternate_pct: Percentage | None = None

    @model_validator(mode="after")
    def validate_scheme_mix(self) -> NPSAsset:
        mix = (
            self.equity_pct,
            self.corporate_debt_pct,
            self.govt_debt_pct,
            self.alternate_pct,
        )
        provided = [value for value in mix if value is not None]
        if provided and (len(provided) != len(mix) or sum(provided, Decimal(0)) != Decimal(100)):
            raise ValueError("all NPS scheme percentages must be provided and sum to 100")
        return self


class PostOfficeAsset(AssetBase):
    asset_type: Literal["po_scheme"]
    scheme: Literal["nsc", "kvp", "scss", "mis", "td", "rd", "ssy"]
    amount: Rupees
    purchase: MonthYear
    balance_as_of: MonthYear | None = None
    annual_contribution: Rupees | None = None
    td_years: Literal[1, 2, 3, 5] | None = None
    rate_pct: RatePct | None = None

    @model_validator(mode="after")
    def validate_td_years(self) -> PostOfficeAsset:
        if self.scheme == "td" and self.td_years is None:
            raise ValueError("td_years is required when scheme is td")
        if self.scheme != "td" and self.td_years is not None:
            raise ValueError("td_years is only valid when scheme is td")
        return self


class BondAsset(AssetBase):
    asset_type: Literal["bond"]
    kind: Literal["gsec", "tbill", "rbi_frsb", "corporate", "tax_free", "bond_54ec"]
    face_value: Rupees
    maturity: MonthYear
    coupon_pct: RatePct | None = None
    coupon_frequency: Literal[1, 2] | None = None
    isin: str | None = Field(default=None, min_length=1, max_length=12)
    issuer: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def set_default_coupon_frequency(self) -> BondAsset:
        if self.coupon_frequency is None:
            self.coupon_frequency = 2 if self.kind in {"gsec", "rbi_frsb"} else 1
        return self


class GoldAsset(AssetBase):
    asset_type: Literal["gold"]
    kind: Literal["sgb", "etf", "mf", "physical", "digital"]
    units: Decimal | None = Field(default=None, gt=Decimal("0"))
    grams: Decimal | None = Field(default=None, gt=Decimal("0"))
    purity: Literal[24, 22, 18] | None = None
    series: str | None = Field(default=None, max_length=80)
    issue: MonthYear | None = None
    issue_price_per_gram: Rupees | None = None
    purpose: Literal["jewellery", "investment"] | None = None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> GoldAsset:
        if self.kind == "physical":
            if self.grams is None or self.purity is None:
                raise ValueError("physical gold requires grams and purity")
            if (
                self.units is not None
                or self.series is not None
                or self.issue is not None
                or self.issue_price_per_gram is not None
            ):
                raise ValueError("physical gold does not accept units, series, or issue fields")
        elif self.kind == "digital":
            if self.grams is None:
                raise ValueError("digital gold requires grams")
            if (
                self.purity is not None
                or self.units is not None
                or self.series is not None
                or self.issue is not None
                or self.issue_price_per_gram is not None
                or self.purpose is not None
            ):
                raise ValueError("digital gold accepts grams only")
        else:
            if self.grams is not None or self.purity is not None or self.purpose is not None:
                raise ValueError("grams, purity, and purpose are only valid for physical gold")
            if self.kind not in ("sgb", "etf", "mf") and self.units is not None:
                raise ValueError("units are only valid for SGB, ETF, or MF gold")
            if self.kind not in {"sgb", "etf", "mf"} and self.series is not None:
                raise ValueError("series is only valid for SGB, ETF, or MF gold")
            if self.kind != "sgb" and self.issue is not None:
                raise ValueError("issue is only valid for SGB gold")
            if self.kind != "sgb" and self.issue_price_per_gram is not None:
                raise ValueError("issue_price_per_gram is only valid for SGB gold")
        return self


class TermInsuranceAsset(AssetBase):
    asset_type: Literal["term_insurance"]
    insurer: str = Field(min_length=1, max_length=120)
    cover: Rupees
    ends_year: int = Field(ge=1900, le=9999)
    premium: Rupees | None = None
    premium_frequency: Literal["monthly", "quarterly", "half_yearly", "yearly"] | None = None


class HealthInsuranceAsset(AssetBase):
    asset_type: Literal["health_insurance"]
    insurer: str = Field(min_length=1, max_length=120)
    cover: Rupees
    renewal_month: int = Field(ge=1, le=12)
    members_covered: list[str] | None = None
    top_up_cover: Rupees | None = None


class TraditionalPolicyAsset(AssetBase):
    asset_type: Literal["traditional_policy"]
    insurer: str = Field(min_length=1, max_length=120)
    plan: str = Field(min_length=1, max_length=120)
    policy_kind: Literal["endowment", "money_back", "whole_life"]
    sum_assured: Rupees
    annual_premium: Rupees
    start_year: int = Field(ge=1900, le=9999)
    term_years: int = Field(gt=0, le=100)
    premium_paying_term: int | None = Field(default=None, gt=0, le=100)
    bonus_accrued: Rupees | None = None
    surrender_value: Rupees | None = None
    maturity_illustration: Rupees | None = None


class ULIPAsset(AssetBase):
    asset_type: Literal["ulip"]
    insurer: str = Field(min_length=1, max_length=120)
    fund_value: Rupees
    annual_premium: Rupees
    equity_pct: Percentage | None = None
    start_year: int | None = Field(default=None, ge=1900, le=9999)


class PropertyAsset(AssetBase):
    asset_type: Literal["property"]
    property_kind: str = Field(min_length=1, max_length=80)
    estimated_value: Rupees
    linked_loan_id: str | None = None
    monthly_rent: Rupees | None = None


class LoanAsset(AssetBase):
    asset_type: Literal["loan"]
    loan_kind: Literal["home", "car", "personal", "education", "credit_card", "gold", "other"]
    outstanding: Rupees
    rate_pct: LoanRatePct
    emi: Rupees | None = None
    end: MonthYear | None = None


type AnyAsset = Annotated[
    SavingsAsset
    | FixedDepositAsset
    | RecurringDepositAsset
    | PPFAsset
    | EPFAsset
    | NPSAsset
    | PostOfficeAsset
    | BondAsset
    | GoldAsset
    | TermInsuranceAsset
    | HealthInsuranceAsset
    | TraditionalPolicyAsset
    | ULIPAsset
    | PropertyAsset
    | LoanAsset,
    Field(discriminator="asset_type"),
]

ASSET_MODELS: dict[str, type[AssetBase]] = {
    "savings": SavingsAsset,
    "fd": FixedDepositAsset,
    "rd": RecurringDepositAsset,
    "ppf": PPFAsset,
    "epf": EPFAsset,
    "nps": NPSAsset,
    "po_scheme": PostOfficeAsset,
    "bond": BondAsset,
    "gold": GoldAsset,
    "term_insurance": TermInsuranceAsset,
    "health_insurance": HealthInsuranceAsset,
    "traditional_policy": TraditionalPolicyAsset,
    "ulip": ULIPAsset,
    "property": PropertyAsset,
    "loan": LoanAsset,
}
_ASSET_ADAPTER = TypeAdapter(AnyAsset)


def parse_asset(data: dict[str, object]) -> AnyAsset:
    """Validate a plain mapping against the asset type discriminator."""
    return _ASSET_ADAPTER.validate_python(data)


def default_sleeves(asset: AnyAsset) -> dict[Sleeve, Decimal]:
    """Return the default portfolio sleeve fractions for a manual asset."""
    asset_type = asset.asset_type
    if asset_type == "loan":
        return {}
    if asset_type == "savings":
        return {Sleeve.CASH: Decimal(1)}
    if asset_type in {"fd", "rd", "ppf", "epf", "po_scheme", "bond", "traditional_policy"}:
        return {Sleeve.DEBT: Decimal(1)}
    if asset_type in {"term_insurance", "health_insurance"}:
        return {Sleeve.PROTECTION: Decimal(1)}
    if asset_type == "gold":
        return {Sleeve.GOLD: Decimal(1)}
    if asset_type == "property":
        return {Sleeve.REAL_ESTATE: Decimal(1)}
    if asset_type == "nps":
        equity = asset.equity_pct if asset.equity_pct is not None else Decimal(50)
        debt = Decimal(100) - equity
        return _equity_debt_split(equity, debt)
    if asset_type == "ulip":
        equity = asset.equity_pct if asset.equity_pct is not None else Decimal(50)
        return _equity_debt_split(equity, Decimal(100) - equity)
    raise ValueError(f"Unsupported asset type: {asset_type}")


def _equity_debt_split(equity: Decimal, debt: Decimal) -> dict[Sleeve, Decimal]:
    result: dict[Sleeve, Decimal] = {}
    if equity:
        result[Sleeve.EQUITY_INDIA] = equity / Decimal(100)
    if debt:
        result[Sleeve.DEBT] = debt / Decimal(100)
    return result


def default_liquidity(asset: AnyAsset) -> LiquidityClass:
    """Return a coarse liquidity class based on the asset catalogue."""
    asset_type = asset.asset_type
    if asset_type == "savings":
        return LiquidityClass.INSTANT
    if asset_type == "gold":
        if asset.kind in {"etf", "mf", "digital"}:
            return LiquidityClass.WITHIN_1Y
        return LiquidityClass.LOCKED
    if asset_type == "bond":
        return LiquidityClass.WITHIN_1Y
    if asset_type in {"term_insurance", "health_insurance", "property", "loan"}:
        return LiquidityClass.LOCKED
    return LiquidityClass.LOCKED
