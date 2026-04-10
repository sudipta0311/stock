from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class SignalRecord:
    family: str
    sector: str
    conviction: str
    score: float
    source: str
    horizon: str
    detail: str
    as_of_date: str
    signal_key: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioInput:
    mutual_funds: list[dict[str, Any]] = field(default_factory=list)
    etfs: list[dict[str, Any]] = field(default_factory=list)
    direct_equities: list[dict[str, Any]] = field(default_factory=list)
    macro_thesis: str = ""
    investable_surplus: float = 0.0
    direct_equity_corpus: float = 0.0


@dataclass(slots=True)
class BuyRequest:
    index_name: str
    horizon_months: int
    risk_profile: str
    top_n: int


@dataclass(slots=True)
class RecommendationRecord:
    symbol: str
    company_name: str
    sector: str
    action: str
    score: float
    confidence_band: str
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MonitoringAction:
    symbol: str
    action: str
    severity: str
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

