"""Household profile and goal input models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    model_validator,
)

from wealth.schemas.common import DanishKroner, RatePct, Rupees

_SATELLITE_PCT_SCHEMA = {
    "type": "string",
    "pattern": r"^(0|[1-9]|1[0-9]|2[0-4])(\.[0-9]{1,4})?$|^25(\.0{1,4})?$",
    "format": "decimal",
    "x-minimum": "0",
    "x-maximum": "25",
}


def _satellite_pct_decimal(value: Any) -> Decimal:
    if isinstance(value, (float, bool)):
        raise ValueError("satellite_pct must be a decimal value")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("satellite_pct must be a decimal value") from error
    if not result.is_finite():
        raise ValueError("satellite_pct must be finite")
    return Decimal(format(result, "f"))


def _validate_satellite_pct(value: Decimal) -> Decimal:
    if value < 0 or value > 25:
        raise ValueError("satellite_pct must be between 0 and 25")
    if max(0, -value.as_tuple().exponent) > 4:
        raise ValueError("satellite_pct cannot have more than 4 decimal places")
    return value


SatellitePct = Annotated[
    Decimal,
    BeforeValidator(_satellite_pct_decimal),
    AfterValidator(_validate_satellite_pct),
    Field(ge=Decimal("0"), le=Decimal("25")),
    PlainSerializer(lambda value: format(value, "f"), return_type=str, when_used="json"),
    WithJsonSchema(_SATELLITE_PCT_SCHEMA, mode="validation"),
    WithJsonSchema(_SATELLITE_PCT_SCHEMA, mode="serialization"),
]


class InvestingSip(BaseModel):
    """One existing monthly investment commitment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sleeve: Literal["equity_india", "equity_global", "debt", "gold", "cash"]
    instrument_category: str = Field(min_length=1, max_length=80)
    monthly_amount: Rupees


class InvestingPrefs(BaseModel):
    """User-controlled investing constraints used by the deterministic policy engine."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    monthly_sip_capacity: Rupees = Decimal("0")
    existing_sips: list[InvestingSip] = Field(default_factory=list)
    new_money_this_week: Rupees = Decimal("0")
    has_lrs_broker: bool = False
    prefers_no_paperwork: bool = False
    satellite_pct: SatellitePct = Decimal("20")
    birth_year: int | None = Field(default=None, ge=1900, le=9999)
    lrs_used_this_fy: Rupees = Decimal("0")
    ask_used_this_fy_dkk: DanishKroner = Decimal("0")
    ask_value_start_of_year_dkk: DanishKroner | None = None
    broker: Literal["nordnet", "saxo", "danske_bank", "other"] = "nordnet"
    has_indian_nri_share_account: bool = False
    has_danish_pension_account: bool = False
    us_stock_picks_opt_in: bool = False
    us_stock_picks_pct: SatellitePct = Decimal("10")


class TaxRegime(StrEnum):
    OLD = "old"
    NEW = "new"


class GoalType(StrEnum):
    EMERGENCY = "emergency"
    EDUCATION_INDIA = "education_india"
    EDUCATION_ABROAD = "education_abroad"
    HOME = "home"
    RETIREMENT = "retirement"
    WEALTH = "wealth"
    OTHER = "other"


class FinancialContext(BaseModel):
    """Household details used by deterministic foundation and risk checks."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    monthly_expenses: Rupees
    annual_income: Rupees | None = None
    tax_regime: TaxRegime | None = None
    dependants: int = Field(default=0, ge=0)
    employment_type: Literal["salaried", "self_employed", "business", "retired", "other"] = (
        "salaried"
    )
    income_stability: Literal["high", "medium", "low"] = "medium"
    city_tier: Literal[1, 2, 3] | None = None


class ProfileBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


ResidencyCountry = Literal["IN", "DK", "OTHER"]
SelectableResidencyCountry = Literal["IN", "DK"]
GoalCurrency = Literal["INR", "DKK", "EUR", "USD"]
SUPPORTED_GOAL_CURRENCIES: tuple[str, ...] = ("INR", "DKK", "EUR", "USD")


class Residency(ProfileBase):
    """Tax and country profile used by residency-aware policy decisions."""

    status: Literal["RI", "NRI", "RNOR"] = "RI"
    country: ResidencyCountry = "IN"
    us_person: bool = False
    planned_return_year: int | None = Field(default=None, ge=1900, le=9999)
    settle_plan: Literal["india", "abroad", "undecided"] = "undecided"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_country(cls, value: Any) -> Any:
        """Normalize country values saved before the current residency choices."""
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        status = data.get("status", "RI")
        if not isinstance(status, str):
            return data
        supported_country = {"RI": "IN", "NRI": "DK", "RNOR": "IN"}.get(status)
        if supported_country is None:
            return data
        country = data.get("country")
        if status == "RI" and country is None:
            data["country"] = "IN"
        elif country != supported_country and country != "OTHER":
            data["country"] = "OTHER"
        return data

    @model_validator(mode="after")
    def validate_residency_fields(self) -> Residency:
        valid_countries = {
            "RI": {"IN", "OTHER"},
            "NRI": {"DK", "OTHER"},
            "RNOR": {"IN", "OTHER"},
        }
        if self.country not in valid_countries[self.status]:
            expected = {"RI": "IN", "NRI": "DK", "RNOR": "IN"}[self.status]
            raise ValueError(f"{self.status} residency country must be {expected}")
        if self.planned_return_year is not None and self.planned_return_year < date.today().year:
            raise ValueError("planned_return_year cannot be earlier than the current year")
        return self


class ResidencyInput(ProfileBase):
    """Strict API input limited to the currently supported residence choices."""

    status: Literal["RI", "NRI", "RNOR"] = "RI"
    country: SelectableResidencyCountry = "IN"
    planned_return_year: int | None = Field(default=None, ge=1900, le=9999)
    settle_plan: Literal["india", "abroad", "undecided"] = "undecided"

    @model_validator(mode="after")
    def validate_selectable_country(self) -> ResidencyInput:
        expected = {"RI": "IN", "NRI": "DK", "RNOR": "IN"}[self.status]
        if self.country != expected:
            raise ValueError(f"{self.status} residency country must be {expected}")
        Residency.model_validate(self.model_dump())
        return self


class MemberTax(ProfileBase):
    """Tax and earning facts attached to one household member."""

    tax_regime: TaxRegime | None = None
    tax_slab_pct: RatePct | None = None
    surcharge_pct: RatePct = Decimal("0")
    birth_year: int | None = Field(default=None, ge=1900, le=9999)
    annual_income: Rupees | None = None
    annual_income_currency: Literal["INR", "DKK", "EUR", "USD"] = "INR"
    fy_realised_equity_ltcg: Rupees = Decimal("0")
    fy_realised_equity_stcg: Rupees = Decimal("0")
    employment_type: Literal["salaried", "self_employed", "business", "retired", "other"] = "other"


def residency_from_stored(value: Mapping[str, Any] | None) -> Residency:
    """Load persisted residency and map countries outside current scope to OTHER."""
    return Residency.model_validate(dict(value or {}))


class Goal(ProfileBase):
    name: str = Field(min_length=1, max_length=120)
    goal_type: GoalType
    target_amount: Rupees
    target_year: int = Field(ge=1900, le=9999)
    currency: GoalCurrency
    priority: int = Field(ge=1, le=5)
    member_id: UUID | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_goal_currency(cls, value: Any) -> Any:
        if isinstance(value, dict):
            currency = value.get("currency")
            if isinstance(currency, str) and currency not in SUPPORTED_GOAL_CURRENCIES:
                raise ValueError(f"Unsupported goal currency {currency}: no ECB reference series")
        return value


class HouseholdMember(ProfileBase):
    id: UUID
    display_name: str = Field(min_length=1, max_length=40)
    relationship: Literal["self", "spouse", "child", "parent", "other"]
    is_self: bool = False
    residency: Residency = Field(default_factory=Residency)
    tax: MemberTax = Field(default_factory=MemberTax)
    investing: InvestingPrefs = Field(default_factory=InvestingPrefs)


class Profile(ProfileBase):
    tax_regime: TaxRegime
    tax_slab_pct: RatePct | None = None
    city_tier: Literal[1, 2, 3] | None = None
    timezone: str = "Asia/Kolkata"
    language: str = "en"
    residency: Residency = Field(default_factory=Residency)
