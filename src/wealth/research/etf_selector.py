"""Deterministic fund matching and scoring; the model may only explain the result."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from wealth.core.plain_language import find_banned_terms, sentence_word_counts
from wealth.db.models import FeedItem, IndexValuation, NavDaily, PriceEod, Security
from wealth.research.errors import BudgetExceeded
from wealth.research.router import FakeRouter, ModelRouter, RunContext
from wealth.research.store import save_pack_fields

_RULES_DIR = Path(__file__).resolve().parents[3] / "rules"
_TWO = Decimal("2")
_HUNDRED = Decimal("100")
_YEAR_DAYS = 365


@dataclass(frozen=True)
class CategoryRule:
    id: str
    label: str
    market: str
    security_types: tuple[str, ...]
    index_name: str | None
    include_any: tuple[str, ...]
    exclude_any: tuple[str, ...]


@dataclass(frozen=True)
class SelectorRules:
    categories: tuple[CategoryRule, ...]
    history_days: int
    minimum_history_days: int
    liquidity_floor_inr: Decimal
    liquidity_full_score_inr: Decimal
    premium_disqualify_pct: Decimal
    weights: dict[str, Decimal]

    @classmethod
    def load(
        cls, categories_path: Path | None = None, scoring_path: Path | None = None
    ) -> SelectorRules:
        category_file = categories_path or _RULES_DIR / "etf_categories.yaml"
        scoring_file = scoring_path or _RULES_DIR / "etf_selector.yaml"
        category_raw = yaml.safe_load(category_file.read_text(encoding="utf-8"))
        scoring_raw = yaml.safe_load(scoring_file.read_text(encoding="utf-8"))
        categories = tuple(
            CategoryRule(
                id=str(item["id"]),
                label=str(item["label"]),
                market=str(item["market"]),
                security_types=tuple(str(value) for value in item["security_types"]),
                index_name=str(item["index_name"]) if item.get("index_name") else None,
                include_any=tuple(str(value) for value in item["include_any"]),
                exclude_any=tuple(str(value) for value in item.get("exclude_any", [])),
            )
            for item in category_raw["categories"]
        )
        rules = cls(
            categories=categories,
            history_days=int(scoring_raw.get("history_days", 90)),
            minimum_history_days=int(scoring_raw.get("minimum_history_days", 20)),
            liquidity_floor_inr=Decimal(str(scoring_raw["liquidity_floor_inr"])),
            liquidity_full_score_inr=Decimal(str(scoring_raw["liquidity_full_score_inr"])),
            premium_disqualify_pct=Decimal(str(scoring_raw["premium_disqualify_pct"])),
            weights={key: Decimal(str(value)) for key, value in scoring_raw["weights"].items()},
        )
        if len({item.id for item in rules.categories}) != len(rules.categories):
            raise ValueError("ETF category identifiers must be unique")
        if rules.history_days < 1 or rules.minimum_history_days < 1:
            raise ValueError("ETF selector history windows must be positive")
        if (
            rules.liquidity_floor_inr <= 0
            or rules.liquidity_full_score_inr <= rules.liquidity_floor_inr
        ):
            raise ValueError("ETF liquidity score thresholds are invalid")
        if sum(rules.weights.values(), Decimal(0)) != _HUNDRED:
            raise ValueError("ETF score weights must total 100")
        return rules


@dataclass(frozen=True)
class CandidateMetrics:
    liquidity_median_3m_inr: Decimal | None
    premium_discount_pct: Decimal | None
    tracking_difference_1y_pct: Decimal | None
    expense_ratio_pct: Decimal | None
    assets_inr: Decimal | None


@dataclass(frozen=True)
class Candidate:
    security: Security
    category: CategoryRule
    metrics: CandidateMetrics
    score: Decimal
    reason_codes: tuple[str, ...]


class PickExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category_id: str
    rank: int = Field(ge=1, le=2)
    text: str = Field(max_length=220)


class ExplanationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    explanations: list[PickExplanation] = Field(default_factory=list, max_length=16)


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def matches_category(security: Security, category: CategoryRule) -> bool:
    """Match a securities-master record using the reviewed category keywords."""
    if security.market != category.market or security.security_type not in category.security_types:
        return False
    text = _normalise(
        " ".join(
            value
            for value in (
                security.name,
                security.symbol,
                security.nse_symbol,
                security.category,
                security.sub_category,
            )
            if value
        )
    )
    return any(_normalise(term) in text for term in category.include_any) and not any(
        _normalise(term) in text for term in category.exclude_any
    )


@lru_cache(maxsize=1)
def _not_plain_trackers() -> tuple[str, ...]:
    """Lock-in, factor and active variants; shared with market data in the rules file."""
    raw = yaml.safe_load((_RULES_DIR / "etf_categories.yaml").read_text(encoding="utf-8"))
    return tuple(_normalise(str(term)) for term in raw.get("not_plain_trackers", []))


def eligible_core_fund(security: Security) -> bool:
    text = _normalise(security.name or "")
    if any(term in text for term in _not_plain_trackers()):
        return False
    if security.security_type == "mf":
        if (security.plan_type or "").casefold() == "regular":
            return False
        if (security.option or "").casefold() == "idcw":
            return False
    return True


def has_measured_metric(metrics: CandidateMetrics) -> bool:
    """Missing metrics never boost a score; with none at all there is nothing to rank by."""
    return any(
        value is not None
        for value in (
            metrics.liquidity_median_3m_inr,
            metrics.premium_discount_pct,
            metrics.tracking_difference_1y_pct,
            metrics.expense_ratio_pct,
            metrics.assets_inr,
        )
    )


def _median(values: Sequence[Decimal]) -> Decimal | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / _TWO


def _returns(
    security_prices: Mapping[date, Decimal], index_prices: Mapping[date, Decimal]
) -> Decimal | None:
    common_dates = sorted(set(security_prices) & set(index_prices))
    if len(common_dates) < 2:
        return None
    end_date = common_dates[-1]
    eligible_starts = [
        item for item in common_dates if item <= end_date - timedelta(days=_YEAR_DAYS)
    ]
    if not eligible_starts:
        return None
    start_date = eligible_starts[-1]
    if not 350 <= (end_date - start_date).days <= 380:
        return None
    security_return = (security_prices[end_date] / security_prices[start_date] - 1) * _HUNDRED
    index_return = (index_prices[end_date] / index_prices[start_date] - 1) * _HUNDRED
    return security_return - index_return


def _score(metrics: CandidateMetrics, rules: SelectorRules, *, is_etf: bool) -> Decimal:
    components: dict[str, Decimal | None] = {
        "liquidity": None,
        "premium": None,
        "tracking_difference": None,
        "expense_ratio": None,
        "assets": None,
    }
    liquidity = metrics.liquidity_median_3m_inr
    if is_etf and liquidity is not None:
        components["liquidity"] = min(
            _HUNDRED,
            max(
                Decimal(0),
                (liquidity - rules.liquidity_floor_inr)
                / (rules.liquidity_full_score_inr - rules.liquidity_floor_inr)
                * _HUNDRED,
            ),
        )
    premium = metrics.premium_discount_pct
    if premium is not None:
        components["premium"] = max(Decimal(0), _HUNDRED - max(Decimal(0), premium) * _HUNDRED)
    tracking = metrics.tracking_difference_1y_pct
    if tracking is not None:
        components["tracking_difference"] = max(Decimal(0), _HUNDRED - abs(tracking) * Decimal(10))
    expense = metrics.expense_ratio_pct
    if expense is not None:
        components["expense_ratio"] = max(Decimal(0), _HUNDRED - expense * _HUNDRED)
    assets = metrics.assets_inr
    if assets is not None:
        components["assets"] = min(
            _HUNDRED, max(Decimal(0), assets / Decimal("1000000000") * _HUNDRED)
        )
    return sum(
        (
            rules.weights[name] * component / _HUNDRED
            for name, component in components.items()
            if component is not None
        ),
        Decimal(0),
    )


def _price_activity(rows: Iterable[PriceEod]) -> dict[UUID, list[PriceEod]]:
    activity: dict[UUID, list[PriceEod]] = defaultdict(list)
    for row in rows:
        activity[row.security_id].append(row)
    return activity


def _candidate_metrics(
    security: Security,
    category: CategoryRule,
    *,
    as_of: date,
    rules: SelectorRules,
    activity: Mapping[UUID, Sequence[PriceEod]],
    navs: Mapping[UUID, Sequence[NavDaily]],
    index_history: Mapping[str, Mapping[date, Decimal]],
) -> CandidateMetrics:
    security_activity = list(activity.get(security.id, ()))
    daily_values: list[Decimal] = []
    for row in security_activity:
        if row.as_of < as_of - timedelta(days=rules.history_days - 1):
            continue
        traded_value = row.traded_value
        if traded_value is None and row.volume is not None:
            traded_value = row.close * Decimal(row.volume)
        if traded_value is not None:
            daily_values.append(Decimal(traded_value))
    liquidity = (
        _median(daily_values)
        if security.security_type == "etf" and len(daily_values) >= rules.minimum_history_days
        else None
    )

    current_navs = sorted(
        (item for item in navs.get(security.id, ()) if item.as_of <= as_of),
        key=lambda item: item.as_of,
    )
    nav = current_navs[-1] if current_navs else None
    current_prices = sorted(
        (item for item in security_activity if item.as_of <= as_of), key=lambda item: item.as_of
    )
    latest_price = current_prices[-1] if current_prices else None
    premium: Decimal | None = None
    if (
        security.security_type == "etf"
        and latest_price is not None
        and nav is not None
        and nav.nav > 0
        and abs((latest_price.as_of - nav.as_of).days) <= 5
    ):
        premium = (latest_price.close - nav.nav) / nav.nav * _HUNDRED

    tracking: Decimal | None = None
    if category.index_name is not None:
        index_prices = index_history.get(category.index_name, {})
        if security.security_type == "mf":
            fund_prices = {item.as_of: item.nav for item in current_navs}
        else:
            fund_prices = {item.as_of: item.close for item in current_prices if item.as_of <= as_of}
        tracking = _returns(fund_prices, index_prices)

    return CandidateMetrics(
        liquidity_median_3m_inr=liquidity,
        premium_discount_pct=premium,
        tracking_difference_1y_pct=tracking,
        # AMFI's published file does not grant automated, non-commercial reuse.
        expense_ratio_pct=None,
        assets_inr=None,
    )


def _candidate_reason_codes(
    metrics: CandidateMetrics, rules: SelectorRules, *, is_etf: bool
) -> tuple[str, ...]:
    reasons: list[str] = []
    if is_etf and metrics.liquidity_median_3m_inr is None:
        reasons.append("liquidity_unavailable")
    if (
        is_etf
        and metrics.liquidity_median_3m_inr is not None
        and metrics.liquidity_median_3m_inr < rules.liquidity_floor_inr
    ):
        reasons.append("liquidity_below_floor")
    if metrics.premium_discount_pct is None:
        reasons.append("premium_unavailable")
    elif metrics.premium_discount_pct > rules.premium_disqualify_pct:
        reasons.append("premium_over_1pct")
    if metrics.tracking_difference_1y_pct is None:
        reasons.append("tracking_difference_unavailable")
    if metrics.expense_ratio_pct is None:
        reasons.append("ter_unavailable")
    if metrics.assets_inr is None:
        reasons.append("assets_unavailable")
    return tuple(reasons)


def _serialise_metrics(metrics: CandidateMetrics) -> dict[str, str | None]:
    return {
        name: format(value, "f") if value is not None else None
        for name, value in (
            ("liquidity_median_3m_inr", metrics.liquidity_median_3m_inr),
            ("premium_discount_pct", metrics.premium_discount_pct),
            ("tracking_difference_1y_pct", metrics.tracking_difference_1y_pct),
            ("expense_ratio_pct", metrics.expense_ratio_pct),
            ("assets_inr", metrics.assets_inr),
        )
    }


def _fallback_explanation(pick: Mapping[str, Any]) -> str:
    if pick["metrics"]["liquidity_median_3m_inr"] is None:
        return "Recent trading data is limited. Check the latest fund details before choosing."
    if pick["rank"] == 1:
        return (
            "This fund ranked first using the available checks. "
            "Missing details did not add to its score."
        )
    return (
        "This fund ranked among the top two using the available checks. "
        "Missing details did not add to its score."
    )


def _valid_explanation(text: str) -> bool:
    sentences = [
        value.strip() for value in re.split(r"(?<=[.!?])\s+", text.strip()) if value.strip()
    ]
    return (
        1 <= len(sentences) <= 2
        and all(len(sentence.split()) <= 20 for sentence in sentences)
        and not any(char.isdigit() for char in text)
        and not find_banned_terms(text)
        and all(count <= 20 for count in sentence_word_counts(text))
    )


def _amc_international_suspension(items: Iterable[FeedItem], *, as_of: date) -> bool | None:
    """Flag only explicit AMC notices pausing international-fund subscriptions."""
    subscription_pause = re.compile(
        r"\b(?:suspend(?:s|ed|ing)?|pause(?:s|d|ing)?|halt(?:s|ed|ing)?)\b.{0,60}"
        r"\bsubscriptions?\b|\bsubscriptions?\b.{0,60}"
        r"\b(?:suspend(?:s|ed|ing)?|pause(?:s|d|ing)?|halt(?:s|ed|ing)?)\b",
        re.IGNORECASE,
    )
    for item in items:
        if item.source != "pib" or item.published_at is None:
            continue
        if not as_of - timedelta(days=365) <= item.published_at.date() <= as_of:
            continue
        text = f"{item.title} {item.summary}"
        if not subscription_pause.search(text):
            continue
        if re.search(r"\b(?:international|overseas|foreign)\b", text, re.IGNORECASE):
            return True
    return None


def select_core_picks(
    engine: Engine,
    *,
    as_of: date,
    rules: SelectorRules | None = None,
    router: ModelRouter | FakeRouter | None = None,
    run_ctx: RunContext | None = None,
) -> dict[str, Any]:
    """Score matching securities from stored prices, NAVs, and NSE index closes."""
    selected_rules = rules or SelectorRules.load()
    with Session(engine) as session:
        securities = list(
            session.scalars(
                select(Security).where(
                    Security.market == "IN",
                    Security.security_type.in_(("etf", "mf")),
                )
            )
        )
        security_ids = [row.id for row in securities]
        if security_ids:
            activity_rows = list(
                session.scalars(
                    select(PriceEod).where(
                        PriceEod.security_id.in_(security_ids),
                        PriceEod.as_of >= as_of - timedelta(days=400),
                        PriceEod.as_of <= as_of,
                    )
                )
            )
            nav_rows = list(
                session.scalars(
                    select(NavDaily).where(
                        NavDaily.security_id.in_(security_ids),
                        NavDaily.as_of >= as_of - timedelta(days=400),
                        NavDaily.as_of <= as_of,
                    )
                )
            )
        else:
            activity_rows = []
            nav_rows = []
        index_names = {item.index_name for item in selected_rules.categories if item.index_name}
        index_rows = list(
            session.scalars(
                select(IndexValuation).where(
                    IndexValuation.index_name.in_(index_names or {"NIFTY 50"}),
                    IndexValuation.close.is_not(None),
                    IndexValuation.as_of <= as_of,
                    IndexValuation.as_of >= as_of - timedelta(days=400),
                )
            )
        )
        feed_rows = list(
            session.scalars(
                select(FeedItem)
                .where(FeedItem.source == "pib")
                .order_by(FeedItem.published_at.desc().nullslast())
                .limit(80)
            )
        )
    activity = _price_activity(activity_rows)
    navs: dict[UUID, list[NavDaily]] = defaultdict(list)
    for item in nav_rows:
        navs[item.security_id].append(item)
    index_history: dict[str, dict[date, Decimal]] = defaultdict(dict)
    for item in index_rows:
        if item.close is not None:
            index_history[item.index_name][item.as_of] = Decimal(item.close)

    output: dict[str, list[dict[str, Any]]] = {}
    inav: dict[str, str | None] = {}
    all_picks: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}
    disqualified_counts: dict[str, int] = {}
    for category in selected_rules.categories:
        candidates: list[Candidate] = []
        disqualified = 0
        for security in securities:
            if not matches_category(security, category) or not eligible_core_fund(security):
                continue
            metrics = _candidate_metrics(
                security,
                category,
                as_of=as_of,
                rules=selected_rules,
                activity=activity,
                navs=navs,
                index_history=index_history,
            )
            reasons = _candidate_reason_codes(
                metrics, selected_rules, is_etf=security.security_type == "etf"
            )
            if "liquidity_below_floor" in reasons or "premium_over_1pct" in reasons:
                disqualified += 1
                continue
            # An alphabetical tie between unmeasured funds is not a pick.
            if not has_measured_metric(metrics):
                continue
            candidates.append(
                Candidate(
                    security=security,
                    category=category,
                    metrics=metrics,
                    score=_score(metrics, selected_rules, is_etf=security.security_type == "etf"),
                    reason_codes=reasons,
                )
            )
        candidates.sort(
            key=lambda item: (-item.score, item.security.name.casefold(), item.security.id.hex)
        )
        candidate_counts[category.id] = len(candidates)
        disqualified_counts[category.id] = disqualified
        category_picks: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates[:2], start=1):
            pick: dict[str, Any] = {
                "category_id": category.id,
                "category_label": category.label,
                "rank": rank,
                "security_id": str(candidate.security.id),
                "symbol": candidate.security.symbol or candidate.security.nse_symbol,
                "name": candidate.security.name,
                "security_type": candidate.security.security_type,
                "score": format(
                    candidate.score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f"
                ),
                "metrics": _serialise_metrics(candidate.metrics),
                "reason_codes": list(candidate.reason_codes),
                "explanation": "",
            }
            category_picks.append(pick)
            all_picks.append(pick)
        output[category.id] = category_picks
        inav[category.id] = next(
            (
                pick["metrics"]["premium_discount_pct"]
                for pick in category_picks
                if pick["metrics"]["premium_discount_pct"] is not None
            ),
            None,
        )

    if router is not None and all_picks and run_ctx is not None:
        payload = [
            {
                "category_id": item["category_id"],
                "rank": item["rank"],
                "name": item["name"],
                "available_metrics": [
                    key for key, value in item["metrics"].items() if value is not None
                ],
            }
            for item in all_picks
        ]
        try:
            response = router.complete(
                "nano",
                [
                    {
                        "role": "system",
                        "content": (
                            "Write at most two short plain-language sentences per fund. "
                            "Explain only "
                            "which supplied checks are available. Do not add numbers, make a pick, "
                            "or mention missing facts as strengths."
                        ),
                    },
                    {"role": "user", "content": yaml.safe_dump(payload, sort_keys=True)},
                ],
                schema=ExplanationBatch,
                max_output_tokens=500,
                run_ctx=run_ctx,
            )
            parsed = (
                response.parsed
                if isinstance(response.parsed, ExplanationBatch)
                else ExplanationBatch.model_validate(response.parsed)
            )
            known = {(item["category_id"], item["rank"]): item for item in all_picks}
            for explanation in parsed.explanations:
                item = known.get((explanation.category_id, explanation.rank))
                if item is not None and _valid_explanation(explanation.text):
                    item["explanation"] = explanation.text
        except (BudgetExceeded, ValueError, TypeError):
            pass

    for pick in all_picks:
        if not pick["explanation"]:
            pick["explanation"] = _fallback_explanation(pick)
    return {
        "core_picks": output,
        "inav_premium_pct": inav,
        "amc_international_suspended": _amc_international_suspension(feed_rows, as_of=as_of),
        "selector_meta": {
            "as_of": as_of.isoformat(),
            "eligible_count_by_category": candidate_counts,
            "disqualified_count_by_category": disqualified_counts,
            "ter_note": "ter_unavailable",
        },
    }


def save_core_picks(engine: Engine, week: str, fields: Mapping[str, Any]) -> None:
    """Merge selector output into the existing weekly research pack."""
    save_pack_fields(engine, week, dict(fields))
