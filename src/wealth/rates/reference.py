"""Validated loaders for versioned, human-reviewed scheme-rate reference data."""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from wealth.schemas.common import Compounding, RatePct

_SCHEME_NAMES = {
    "nsc",
    "kvp",
    "td_1y",
    "td_2y",
    "td_3y",
    "td_5y",
    "rd_5y",
    "ppf",
    "scss",
    "mis",
    "ssy",
}
_FINANCIAL_YEAR = re.compile(r"^\d{4}-\d{2}$")


class RateUnavailable(LookupError):
    """Raised when no reference-rate period covers the requested date or year."""

    def __init__(self, scheme: str, on: date | str) -> None:
        self.scheme = scheme
        self.on = on
        super().__init__(f"No reference rate for {scheme!r} on {on}")


class SchemeRate(BaseModel):
    """A single rate selected from the reference data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate_pct: RatePct
    maturity_months: PositiveInt | None = None
    verified: bool
    source_url: str = Field(min_length=1)


class RatePeriod(SchemeRate):
    """A rate and its inclusive date range in the YAML reference file."""

    effective_from: date = Field(alias="from")
    effective_to: date | None = Field(default=None, alias="to")

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def validate_period(self) -> Self:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("rate period 'to' must be on or after 'from'")
        return self


class SchemeReference(BaseModel):
    """Compounding and rate periods for one small-savings scheme."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compounding: Compounding
    tenure_months: PositiveInt | None = None
    periods: tuple[RatePeriod, ...] = ()

    @model_validator(mode="after")
    def validate_period_order(self) -> Self:
        ordered = sorted(self.periods, key=lambda period: period.effective_from)
        if tuple(ordered) != self.periods:
            raise ValueError("rate periods must be ordered by their 'from' date")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.effective_to is None or previous.effective_to >= current.effective_from:
                raise ValueError("rate periods must not overlap")
        return self


class EpfRatePeriod(BaseModel):
    """An EPF rate keyed to one Indian financial year."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fy: str
    rate_pct: RatePct
    source_url: str = Field(min_length=1)
    verified: bool

    @model_validator(mode="after")
    def validate_financial_year(self) -> Self:
        if not _FINANCIAL_YEAR.fullmatch(self.fy):
            raise ValueError("fy must use YYYY-YY format")
        start_year = int(self.fy[:4])
        if self.fy[-2:] != f"{(start_year + 1) % 100:02d}":
            raise ValueError("fy must identify consecutive calendar years")
        return self


class EpfReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    periods: tuple[EpfRatePeriod, ...] = ()

    @model_validator(mode="after")
    def validate_unique_years(self) -> Self:
        years = [period.fy for period in self.periods]
        if len(years) != len(set(years)):
            raise ValueError("EPF periods must have unique financial years")
        return self


class RateBook(BaseModel):
    """Validated rate references with date-based lookups."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: PositiveInt
    note: str = Field(min_length=1)
    schemes: dict[str, SchemeReference]
    epf: EpfReference

    @model_validator(mode="after")
    def validate_scheme_keys(self) -> Self:
        if set(self.schemes) != _SCHEME_NAMES:
            missing = sorted(_SCHEME_NAMES - set(self.schemes))
            unexpected = sorted(set(self.schemes) - _SCHEME_NAMES)
            raise ValueError(f"invalid scheme keys; missing={missing}, unexpected={unexpected}")
        if any(period.maturity_months is None for period in self.schemes["kvp"].periods):
            raise ValueError("each KVP rate period must include maturity_months")
        if any(
            period.maturity_months is not None
            for name, reference in self.schemes.items()
            if name != "kvp"
            for period in reference.periods
        ):
            raise ValueError("maturity_months is only valid for KVP rate periods")
        return self

    def rate_for(self, scheme: str, on: date) -> SchemeRate:
        """Return the period covering ``on``; both date bounds are inclusive."""
        reference = self.schemes.get(scheme)
        if reference is None:
            raise RateUnavailable(scheme, on)
        for period in reference.periods:
            if period.effective_from <= on and (
                period.effective_to is None or on <= period.effective_to
            ):
                return SchemeRate(
                    rate_pct=period.rate_pct,
                    maturity_months=period.maturity_months,
                    verified=period.verified,
                    source_url=period.source_url,
                )
        raise RateUnavailable(scheme, on)

    def epf_rate(self, fy: str) -> SchemeRate:
        """Return the rate recorded for one Indian financial year."""
        for period in self.epf.periods:
            if period.fy == fy:
                return SchemeRate(
                    rate_pct=period.rate_pct,
                    verified=period.verified,
                    source_url=period.source_url,
                )
        raise RateUnavailable("epf", fy)


@lru_cache(maxsize=8)
def load_rates(path: Path | None = None) -> RateBook:
    """Load and validate the scheme-rate YAML, defaulting to the repo reference."""
    source_path = path or (
        Path(__file__).resolve().parents[3] / "rules" / "schemes" / "small_savings.yaml"
    )
    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    return RateBook.model_validate(data)
