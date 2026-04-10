from __future__ import annotations

from typing import Any, TypedDict


class SignalState(TypedDict, total=False):
    trigger: str
    macro_thesis: str
    geo_signals: list[dict[str, Any]]
    policy_signals: list[dict[str, Any]]
    flow_signals: list[dict[str, Any]]
    contrarian_signals: list[dict[str, Any]]
    unified_signals: list[dict[str, Any]]
    run_summary: dict[str, Any]


class PortfolioState(TypedDict, total=False):
    payload: dict[str, Any]
    mutual_fund_exposure: list[dict[str, Any]]
    etf_exposure: list[dict[str, Any]]
    normalized_exposure: list[dict[str, Any]]
    overlap_scores: list[dict[str, Any]]
    identified_gaps: list[dict[str, Any]]
    run_summary: dict[str, Any]


class BuyState(TypedDict, total=False):
    request: dict[str, Any]
    portfolio_context: dict[str, Any]
    universe: list[dict[str, Any]]
    preferred_industries: list[dict[str, Any]]
    industry_narrative: str
    candidates: list[dict[str, Any]]
    scored_candidates: list[dict[str, Any]]
    risk_filtered_candidates: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]
    differentiated_shortlist: list[dict[str, Any]]
    timing_assessments: list[dict[str, Any]]
    allocations: list[dict[str, Any]]
    tax_assessment: dict[str, Any]
    confidence: dict[str, Any]
    recommendations: list[dict[str, Any]]
    run_summary: dict[str, Any]


class MonitoringState(TypedDict, total=False):
    request: dict[str, Any]
    portfolio_context: dict[str, Any]
    industry_reviews: list[dict[str, Any]]
    stock_reviews: list[dict[str, Any]]
    quant_scores: list[dict[str, Any]]
    thesis_reviews: list[dict[str, Any]]
    drawdown_alerts: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    behavioural_flags: list[dict[str, Any]]
    replacement_prompt: dict[str, Any]
    run_summary: dict[str, Any]

