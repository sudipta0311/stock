"""Read-only checks for the data used by the scheduled wealth workflows."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection

from wealth.db.session import normalize_database_url
from wealth.policy.denmark import ask_rule_for_year
from wealth.policy.residency import load_residency_rules

# Two missed weekday sessions are tolerated before a price feed is called stale.
PRICE_STALE_TRADING_DAYS = 2
# Volume is compared with the ten earlier stored dates to spot empty refreshes.
PRICE_HISTORY_DATES = 10
# A newest-day row count below half its normal median signals a partial price load.
PRICE_MINIMUM_MEDIAN_RATIO = 0.5
# FX sources publish daily, with weekends and short publication delays included.
FX_MAX_AGE_DAYS = 4
# Weekly source rows must be refreshed within the preceding eight calendar days.
WEEKLY_DATA_MAX_AGE_DAYS = 8
# The failed-run window spans the latest scheduled refresh and a late retry.
FAILED_RUN_MAX_AGE_DAYS = 2
# Daily market data is scheduled at 13:30 UTC and may start five hours late.
DAILY_DATA_READY_UTC = time(18, 30)
# Overseas market data is scheduled at 18:00 UTC and may start five hours late.
ABROAD_DATA_READY_UTC = time(23, 0)
# Saturday research ends at 20:00 UTC; midnight allows four hours for its last run.
RESEARCH_PACK_READY_UTC = time(0, 0)
# Weekly company data is checked from Sunday after its Friday refresh window.
WEEKLY_DATA_READY_UTC = time(0, 0)
# Sunday plan and email runs have completed before Monday UTC begins.
WEEKLY_USER_DATA_READY_UTC = time(0, 0)

_FX_PAIRS = ("DKK/INR", "USD/INR", "EUR/INR")
_WEEKLY_SOURCE_COUNT_KEYS: dict[str, tuple[str, ...]] = {
    "fund_baskets": ("baskets_stored",),
    "universes": ("universes_loaded",),
    "nse_history": ("prices_eod", "sessions_requested"),
    "edgar": ("symbols",),
    "us_index_pe": ("index_valuations",),
    "nse_fundamentals": ("symbols",),
}
_ALL_MARKETDATA_SOURCES = frozenset(
    {
        "nse",
        "amfi",
        "fx",
        "gold",
        "master",
        "amfi_history",
        "nse_history",
        "fund_baskets",
        "eu_etfs",
        "us_stocks",
        "universes",
        "edgar",
        "us_index_pe",
        "nse_fundamentals",
        "index_pe",
        "ust",
        "feeds",
        "sentinel",
    }
)
_SOURCE_WORKFLOWS = {
    "nse": "Daily market data",
    "amfi": "Daily market data",
    "fx": "Daily market data",
    "gold": "Daily market data",
    "master": "Daily market data",
    "index_pe": "Daily market data",
    "ust": "Daily market data",
    "feeds": "Daily market data",
    "amfi_history": "Monthly fund history",
    "fund_baskets": "Weekly company data",
    "eu_etfs": "Overseas market data",
    "us_stocks": "Overseas market data",
    "universes": "Weekly company data",
    "nse_history": "Weekly company data",
    "edgar": "Weekly company data",
    "us_index_pe": "Weekly company data",
    "nse_fundamentals": "Weekly company data",
    "sentinel": "Daily safety checks and reminders",
}


@dataclass(frozen=True)
class DayRowCount:
    on: date
    rows: int


@dataclass(frozen=True)
class MarketdataRunFact:
    started_at: datetime
    status: str
    sources: tuple[str, ...]
    source_counts: Mapping[str, Mapping[str, int]]
    error_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class FundBasketHealthFact:
    basket_id: str
    as_of: date
    members_total: int
    members_found: int


@dataclass(frozen=True)
class HealthFacts:
    india_prices: tuple[DayRowCount, ...]
    india_fund_prices: tuple[DayRowCount, ...]
    abroad_prices: tuple[DayRowCount, ...]
    fx_latest: Mapping[str, date | None]
    marketdata_runs: tuple[MarketdataRunFact, ...]
    research_pack_week: str
    research_pack_idea_counts: Mapping[str, int] | None
    weekly_plan_week: str
    eligible_plan_count: int
    missing_plan_count: int
    eligible_email_count: int
    missing_email_count: int
    email_sending_configured: bool
    fund_baskets: tuple[FundBasketHealthFact, ...] = ()
    stock_checks_as_of: date | None = None
    held_listed_count: int = 0
    held_listed_fresh_count: int = 0


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _iso_week(value: date) -> str:
    iso = value.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def _most_recent_saturday(now: datetime) -> date:
    days_since_saturday = (now.weekday() - 5) % 7
    return now.date() - timedelta(days=days_since_saturday)


def _most_recent_completed_sunday(now: datetime) -> date:
    days_since_sunday = (now.weekday() + 1) % 7
    if days_since_sunday == 0:
        days_since_sunday = 7
    return now.date() - timedelta(days=days_since_sunday)


def _next_sunday_midnight(now: datetime, ready_at: time) -> datetime:
    days_until_sunday = (6 - now.weekday()) % 7
    due_date = now.date() + timedelta(days=days_until_sunday)
    return datetime.combine(due_date, ready_at, UTC)


def _next_monday_midnight(now: datetime) -> datetime:
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    due_date = now.date() + timedelta(days=days_until_monday)
    return datetime.combine(due_date, WEEKLY_USER_DATA_READY_UTC, UTC)


def weekday_sessions_oldest_first(latest: date, expected: date) -> int:
    """Count weekdays after a stored date through the expected session date."""
    if latest >= expected:
        return 0
    return sum(
        1
        for offset in range(1, (expected - latest).days + 1)
        if (latest + timedelta(days=offset)).weekday() < 5
    )


def _last_completed_weekday(now: datetime, ready_at: time) -> date:
    candidate = now.date()
    if candidate.weekday() >= 5 or (
        candidate.weekday() < 5 and now.timetz().replace(tzinfo=None) < ready_at
    ):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _check(
    key: str,
    level: str,
    message: str,
    facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "level": level,
        "message": message,
        "facts": dict(facts or {}),
    }


def _check_price(
    key: str,
    label: str,
    workflow: str,
    stored_days: Sequence[DayRowCount],
    expected_date: date,
) -> dict[str, Any]:
    ordered = sorted(stored_days, key=lambda item: item.on, reverse=True)
    latest = ordered[0] if ordered else None
    previous = ordered[1 : PRICE_HISTORY_DATES + 1]
    median_rows = (
        statistics.median([item.rows for item in previous])
        if len(previous) == PRICE_HISTORY_DATES
        else None
    )
    old_by = weekday_sessions_oldest_first(latest.on, expected_date) if latest is not None else None
    stale = old_by is None or old_by > PRICE_STALE_TRADING_DAYS
    low_volume = (
        latest is not None
        and median_rows is not None
        and latest.rows < median_rows * PRICE_MINIMUM_MEDIAN_RATIO
    )
    facts: dict[str, Any] = {
        "expected_date": expected_date.isoformat(),
        "latest_date": latest.on.isoformat() if latest is not None else None,
        "trading_days_old": old_by,
        "latest_rows": latest.rows if latest is not None else 0,
        "previous_dates": len(previous),
    }
    if median_rows is not None:
        facts["previous_median_rows"] = median_rows

    sentence_label = label[0].upper() + label[1:]
    if not stale and not low_volume:
        return _check(key, "ok", f"{sentence_label} are up to date. No action is needed.", facts)
    if latest is None:
        message = f"No {label} are stored. Rerun {workflow}."
    elif stale and low_volume:
        message = (
            f"{sentence_label} are {old_by} trading days old and unusually few. Rerun {workflow}."
        )
    elif stale:
        message = f"{sentence_label} are {old_by} trading days old. Rerun {workflow}."
    else:
        message = f"{sentence_label} stored fewer than half the usual rows. Rerun {workflow}."
    return _check(key, "problem", message, facts)


def _check_fx(now: datetime, latest_by_pair: Mapping[str, date | None]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for pair in _FX_PAIRS:
        latest = latest_by_pair.get(pair)
        age = (now.date() - latest).days if latest is not None else None
        facts = {
            "latest_date": latest.isoformat() if latest is not None else None,
            "age_days": age,
        }
        if age is not None and age <= FX_MAX_AGE_DAYS:
            checks.append(_check("fx", "ok", f"{pair} has a recent rate.", facts))
        elif latest is None:
            checks.append(
                _check(
                    "fx",
                    "problem",
                    f"{pair} has no recent rate. Rerun Daily market data.",
                    facts,
                )
            )
        else:
            checks.append(
                _check(
                    "fx",
                    "problem",
                    f"{pair} data is {age} days old. Rerun Daily market data.",
                    facts,
                )
            )
    return checks


def _source_has_rows(source: str, counts: Mapping[str, int]) -> bool:
    if source == "nse_history":
        stored = counts.get("prices_eod", 0)
        requested = counts.get("sessions_requested")
        # A complete history has no missing sessions to request, so zero stored rows is normal.
        if requested == 0:
            return True
        return stored > 0
    if source in {"edgar", "nse_fundamentals"}:
        # A source can process symbols without new filings, so its symbol count is the signal.
        return counts.get("symbols", 0) > 0
    count_key = _WEEKLY_SOURCE_COUNT_KEYS[source][0]
    return counts.get(count_key, 0) > 0


def _check_weekly_data(now: datetime, runs: Sequence[MarketdataRunFact]) -> dict[str, Any]:
    if now.weekday() == 5:
        return _check(
            "weekly_data",
            "ok",
            "Weekly company data is not due yet.",
            {"not_due_until": _iso_utc(_next_sunday_midnight(now, WEEKLY_DATA_READY_UTC))},
        )

    cutoff = now - timedelta(days=WEEKLY_DATA_MAX_AGE_DAYS)
    missing_sources: list[str] = []
    empty_sources: list[str] = []
    for source in _WEEKLY_SOURCE_COUNT_KEYS:
        source_runs = [run for run in runs if source in run.sources and run.started_at <= now]
        recent_successes = [
            run
            for run in source_runs
            if run.started_at >= cutoff and run.status in {"succeeded", "partial"}
        ]
        if not recent_successes:
            missing_sources.append(source)
            continue
        latest = max(source_runs, key=lambda item: item.started_at)
        if latest.status in {"succeeded", "partial"} and not _source_has_rows(
            source, latest.source_counts.get(source, {})
        ):
            empty_sources.append(source)

    affected = sorted(set(missing_sources + empty_sources))
    facts = {
        "sources": affected,
        "missing_sources": sorted(missing_sources),
        "empty_sources": sorted(empty_sources),
        "window_days": WEEKLY_DATA_MAX_AGE_DAYS,
    }
    if not affected:
        return _check(
            "weekly_data", "ok", "Weekly company data is current and contains rows.", facts
        )
    if empty_sources:
        message = (
            f"Weekly company data has no rows for {', '.join(affected)}. Rerun Weekly company data."
        )
    else:
        message = (
            f"Weekly company data is missing for {', '.join(affected)}. Rerun Weekly company data."
        )
    return _check("weekly_data", "problem", message, facts)


def _sources_for_run_status(run: MarketdataRunFact) -> tuple[str, ...]:
    if run.status in {"failed", "partial"} and run.error_sources:
        return run.error_sources
    return run.sources


def _group_sources(sources: set[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for source in sorted(sources):
        workflow = _SOURCE_WORKFLOWS.get(source, "Daily market data")
        grouped.setdefault(workflow, []).append(source)
    return grouped


def _check_failed_runs(now: datetime, runs: Sequence[MarketdataRunFact]) -> list[dict[str, Any]]:
    cutoff = now - timedelta(days=FAILED_RUN_MAX_AGE_DAYS)
    recent = [run for run in runs if cutoff <= run.started_at <= now]
    failed_sources = {
        source
        for run in recent
        if run.status == "failed"
        for source in (_sources_for_run_status(run) or ("unknown",))
    }
    partial_sources = {
        source
        for run in recent
        if run.status == "partial"
        for source in (_sources_for_run_status(run) or ("unknown",))
    }
    common_facts = {
        "failed_sources": sorted(failed_sources),
        "partial_sources": sorted(partial_sources),
        "failed_runs": sum(1 for run in recent if run.status == "failed"),
        "partial_runs": sum(1 for run in recent if run.status == "partial"),
        "window_days": FAILED_RUN_MAX_AGE_DAYS,
    }
    checks: list[dict[str, Any]] = []
    for level, sources in (("problem", failed_sources), ("warning", partial_sources)):
        for workflow, grouped_sources in _group_sources(sources).items():
            joined_sources = ", ".join(grouped_sources)
            verb = "failed" if level == "problem" else "were partial"
            message = f"Market data sources {verb}: {joined_sources}. Rerun {workflow}."
            checks.append(_check("failed_runs", level, message, common_facts))
    if not checks:
        checks.append(
            _check(
                "failed_runs", "ok", "No market data runs failed in the last 2 days.", common_facts
            )
        )
    return checks


def _check_research_pack(now: datetime, facts: HealthFacts) -> dict[str, Any]:
    if now.weekday() == 5:
        due = _next_sunday_midnight(now, RESEARCH_PACK_READY_UTC)
        return _check(
            "research_pack",
            "ok",
            "Weekly research has time to finish.",
            {"week": facts.research_pack_week, "not_due_until": _iso_utc(due)},
        )
    counts = facts.research_pack_idea_counts
    if counts is None:
        return _check(
            "research_pack",
            "problem",
            "No research pack is ready. Rerun Weekly research.",
            {"week": facts.research_pack_week},
        )
    missing_markets = [market for market in ("IN", "US") if counts.get(market, 0) <= 0]
    output_facts = {
        "week": facts.research_pack_week,
        "idea_counts": {market: counts.get(market, 0) for market in ("IN", "US")},
    }
    if not missing_markets:
        return _check(
            "research_pack", "ok", "The research pack has ideas for both markets.", output_facts
        )
    market_name = "India" if missing_markets[0] == "IN" else "US"
    message = f"The research pack has no {market_name} idea. Rerun Weekly research."
    return _check("research_pack", "warning", message, output_facts)


def _check_weekly_plans(now: datetime, facts: HealthFacts) -> dict[str, Any]:
    if now.weekday() in {5, 6}:
        return _check(
            "weekly_plans",
            "ok",
            "Weekly plans are not due yet.",
            {"week": facts.weekly_plan_week, "not_due_until": _iso_utc(_next_monday_midnight(now))},
        )
    result_facts = {
        "week": facts.weekly_plan_week,
        "eligible_count": facts.eligible_plan_count,
        "missing_count": facts.missing_plan_count,
    }
    if facts.missing_plan_count == 0:
        return _check("weekly_plans", "ok", "All households have a weekly plan.", result_facts)
    count = facts.missing_plan_count
    household_word = "household" if count == 1 else "households"
    message = f"{count} {household_word} have no weekly plan. Rerun Weekly household plans."
    if count == 1:
        message = "1 household has no weekly plan. Rerun Weekly household plans."
    return _check("weekly_plans", "problem", message, result_facts)


def _check_weekly_email(now: datetime, facts: HealthFacts) -> dict[str, Any]:
    if not facts.email_sending_configured:
        return _check(
            "weekly_email",
            "ok",
            "Weekly email sending is not configured, so this check is skipped.",
        )
    if now.weekday() in {5, 6}:
        return _check(
            "weekly_email",
            "ok",
            "Weekly email is not due yet.",
            {"week": facts.weekly_plan_week, "not_due_until": _iso_utc(_next_monday_midnight(now))},
        )
    result_facts = {
        "week": facts.weekly_plan_week,
        "eligible_count": facts.eligible_email_count,
        "missing_count": facts.missing_email_count,
    }
    if facts.missing_email_count == 0:
        return _check(
            "weekly_email", "ok", "All opted-in users received a weekly brief.", result_facts
        )
    count = facts.missing_email_count
    user_word = "user" if count == 1 else "users"
    message = f"{count} opted-in {user_word} have no weekly brief. Rerun Weekly local-time email."
    return _check("weekly_email", "problem", message, result_facts)


def _check_fund_lookthrough(facts: HealthFacts, now: datetime) -> dict[str, Any]:
    from wealth.policy.lookthrough import load_lookthrough_rules

    rules = load_lookthrough_rules()
    mapped = set(rules.get("baskets", {}))
    stored = {item.basket_id: item for item in facts.fund_baskets}
    counts = {
        "stored": len(stored),
        "mapped": len(mapped),
        "missing": sorted(mapped - stored.keys()),
        "old": [],
        "low_coverage": [],
        "max_age_days": int(rules.get("max_age_days", 21)),
    }
    if not stored:
        return _check(
            "fund_lookthrough",
            "problem",
            "No fund holdings data is stored. Rerun Weekly company data.",
            counts,
        )
    as_of = now.astimezone(UTC).date()
    counts["old"] = sorted(
        basket_id
        for basket_id in mapped & stored.keys()
        if (as_of - stored[basket_id].as_of).days < 0
        or (as_of - stored[basket_id].as_of).days > counts["max_age_days"]
    )
    for basket_id in ("nifty50", "nifty100"):
        item = stored.get(basket_id)
        if item is not None and (
            item.members_total == 0
            or Decimal(item.members_found) / Decimal(item.members_total) < Decimal("0.9")
        ):
            counts["low_coverage"].append(basket_id)
    if counts["missing"] or counts["old"]:
        return _check(
            "fund_lookthrough",
            "warning",
            "Some fund holdings data is missing or old. Rerun Weekly company data.",
            counts,
        )
    if counts["low_coverage"]:
        return _check(
            "fund_lookthrough",
            "warning",
            "Some India index lists have limited matches. Rerun Weekly company data.",
            counts,
        )
    return _check(
        "fund_lookthrough",
        "ok",
        "Fund holdings data is current.",
        counts,
    )


def _check_stock_checks(facts: HealthFacts, now: datetime) -> dict[str, Any]:
    latest = facts.stock_checks_as_of
    age = (now.date() - latest).days if latest is not None else None
    coverage = (
        Decimal(facts.held_listed_fresh_count) / Decimal(facts.held_listed_count)
        if facts.held_listed_count
        else Decimal(1)
    )
    output = {
        "latest_date": latest.isoformat() if latest is not None else None,
        "age_days": age,
        "held_in_lists": facts.held_listed_count,
        "held_with_fresh_checks": facts.held_listed_fresh_count,
        "held_coverage_pct": format(coverage * Decimal(100), ".2f"),
    }
    stale = age is None or age < 0 or age > 9
    low_coverage = facts.held_listed_count > 0 and coverage < Decimal("0.9")
    if not stale and not low_coverage:
        return _check("stock_checks", "ok", "Weekly share checks are current.", output)
    messages: list[str] = []
    if stale:
        messages.append(
            "No weekly share checks are stored. Rerun Weekly research."
            if age is None
            else f"Weekly share checks are {age} days old. Rerun Weekly research."
        )
    if low_coverage:
        messages.append(
            f"Only {facts.held_listed_fresh_count} of {facts.held_listed_count} held shares in the "
            "main lists have fresh checks."
        )
    return _check("stock_checks", "warning", " ".join(messages), output)


def _check_ask_rules(now: datetime) -> dict[str, Any]:
    """Warn when the current year has no exact share savings account cap rule."""
    try:
        _cap, exact = ask_rule_for_year(
            load_residency_rules(), "deposit_cap_dkk", now.astimezone(UTC).year
        )
    except (KeyError, TypeError):
        exact = False
    if not exact:
        return _check(
            "ask_rules",
            "warning",
            "ASK deposit_cap_dkk has no exact entry for the current year.",
            {"rule_key": "deposit_cap_dkk", "exact": False},
        )
    return _check(
        "ask_rules",
        "ok",
        "ASK deposit_cap_dkk has an exact entry for the current year.",
        {"rule_key": "deposit_cap_dkk", "exact": True},
    )


def evaluate_health(facts: HealthFacts, now: datetime) -> dict[str, Any]:
    """Turn aggregate database facts into ordered, user-safe health checks."""
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ValueError("now must be a timezone-aware UTC datetime")
    current = now.astimezone(UTC)
    india_expected = _last_completed_weekday(current, DAILY_DATA_READY_UTC)
    abroad_expected = _last_completed_weekday(current, ABROAD_DATA_READY_UTC)
    checks: list[dict[str, Any]] = [
        _check_price(
            "india_prices",
            "India share prices",
            "Daily market data",
            facts.india_prices,
            india_expected,
        ),
        _check_price(
            "india_fund_prices",
            "Indian fund prices",
            "Daily market data",
            facts.india_fund_prices,
            india_expected,
        ),
        _check_price(
            "abroad_prices",
            "overseas share prices",
            "Overseas market data",
            facts.abroad_prices,
            abroad_expected,
        ),
    ]
    checks.extend(_check_fx(current, facts.fx_latest))
    checks.append(_check_weekly_data(current, facts.marketdata_runs))
    checks.extend(_check_failed_runs(current, facts.marketdata_runs))
    checks.append(_check_research_pack(current, facts))
    checks.append(_check_weekly_plans(current, facts))
    checks.append(_check_weekly_email(current, facts))
    checks.append(_check_fund_lookthrough(facts, current))
    checks.append(_check_stock_checks(facts, current))
    checks.append(_check_ask_rules(current))
    return {
        "ok": all(item["level"] != "problem" for item in checks),
        "checked_at": _iso_utc(current),
        "checks": checks,
    }


@contextmanager
def read_only_transaction(engine: Engine) -> Iterator[Connection]:
    """Run all health queries in one explicitly read-only database transaction."""
    with engine.connect() as connection, connection.begin():
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        yield connection


def _date_row_counts(
    connection: Connection,
    *,
    table: str,
    market_clause: str,
    parameters: Mapping[str, Any],
) -> tuple[DayRowCount, ...]:
    statement = text(
        f"""
        SELECT data.as_of, count(*)::int AS rows
        FROM {table} AS data
        JOIN securities AS security ON security.id = data.security_id
        WHERE {market_clause} AND data.as_of <= :as_of
        GROUP BY data.as_of
        ORDER BY data.as_of DESC
        LIMIT :limit
        """
    )
    rows = connection.execute(statement, {**parameters, "limit": PRICE_HISTORY_DATES + 1}).all()
    return tuple(DayRowCount(on=row.as_of, rows=int(row.rows)) for row in rows)


def _safe_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _run_fact(
    started_at: datetime,
    status: str,
    raw_counts: Any,
    raw_errors: Any,
) -> MarketdataRunFact:
    counts = raw_counts if isinstance(raw_counts, Mapping) else {}
    errors = raw_errors if isinstance(raw_errors, Mapping) else {}
    source_names = {str(key) for key in counts if key in _ALL_MARKETDATA_SOURCES}
    error_sources = {
        source
        for error_key in errors
        if (source := str(error_key).split(":", maxsplit=1)[0]) in _ALL_MARKETDATA_SOURCES
    }
    source_names.update(error_sources)
    nested_counts: dict[str, Mapping[str, int]] = {}
    for source in source_names:
        source_value = counts.get(source)
        source_value = source_value if isinstance(source_value, Mapping) else {}
        safe_values = {
            str(key): number
            for key, value in source_value.items()
            if (number := _safe_count(value)) is not None
        }
        nested_counts[source] = safe_values
    return MarketdataRunFact(
        started_at=started_at.astimezone(UTC),
        status=str(status),
        sources=tuple(sorted(source_names)),
        source_counts=nested_counts,
        error_sources=tuple(sorted(error_sources)),
    )


def _idea_counts(payload: Any) -> Mapping[str, int]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {"IN": 0, "US": 0}
    if not isinstance(payload, Mapping):
        return {"IN": 0, "US": 0}
    counts: dict[str, int] = {"IN": 0, "US": 0}
    markets = payload.get("markets")
    if isinstance(markets, Mapping):
        for market in counts:
            data = markets.get(market)
            if isinstance(data, Mapping):
                count = _safe_count(data.get("idea_count"))
                counts[market] = count or 0
        return counts
    overall = payload.get("idea_count")
    if isinstance(overall, Mapping):
        for market in counts:
            counts[market] = _safe_count(overall.get(market)) or 0
        return counts
    ideas = payload.get("ideas")
    if isinstance(ideas, list):
        for idea in ideas:
            if not isinstance(idea, Mapping):
                continue
            market = idea.get("market")
            security = idea.get("security")
            if market is None and isinstance(security, Mapping):
                market = security.get("market")
            if market in counts:
                counts[str(market)] += 1
    return counts


def _weekly_plan_counts(connection: Connection, week: str) -> tuple[int, int]:
    row = connection.execute(
        text(
            """
            WITH active_users AS (
                SELECT id FROM users WHERE deleted_at IS NULL
            ),
            active_partners AS (
                SELECT DISTINCT grantee_user_id AS id
                FROM household_access
                WHERE role = 'partner' AND revoked_at IS NULL
            ),
            users_with_data AS (
                SELECT user_id FROM profiles
                UNION SELECT user_id FROM goals
                UNION SELECT user_id FROM accounts
                UNION SELECT user_id FROM manual_assets
                UNION SELECT user_id FROM holdings_lots
                UNION SELECT user_id FROM import_jobs
            ),
            eligible AS (
                SELECT active_users.id
                FROM active_users
                WHERE NOT EXISTS (
                    SELECT 1 FROM active_partners WHERE active_partners.id = active_users.id
                ) OR EXISTS (
                    SELECT 1 FROM users_with_data WHERE users_with_data.user_id = active_users.id
                )
            )
            SELECT count(*)::int AS eligible_count,
                   count(*) FILTER (
                       WHERE NOT EXISTS (
                           SELECT 1 FROM weekly_plans
                           WHERE weekly_plans.user_id = eligible.id
                             AND weekly_plans.week = :week
                       )
                   )::int AS missing_count
            FROM eligible
            """
        ),
        {"week": week},
    ).one()
    return int(row.eligible_count), int(row.missing_count)


def _weekly_email_counts(connection: Connection, week: str) -> tuple[int, int]:
    # Mirrors recipient eligibility and the existing-alert lookup in alerts/weekly.py.
    row = connection.execute(
        text(
            """
            WITH latest_consent AS (
                SELECT DISTINCT ON (user_id) user_id, withdrawn_at
                FROM consents
                WHERE purpose = 'weekly_email'
                ORDER BY user_id, granted_at DESC, id DESC
            ),
            users_with_data AS (
                SELECT user_id FROM profiles
                UNION SELECT user_id FROM goals
                UNION SELECT user_id FROM accounts
                UNION SELECT user_id FROM manual_assets
                UNION SELECT user_id FROM holdings_lots
                UNION SELECT user_id FROM import_jobs
            ),
            resolved_owners AS (
                SELECT users.id AS login_user_id,
                       CASE
                           WHEN users_with_data.user_id IS NOT NULL THEN users.id
                           ELSE COALESCE(partner.owner_user_id, users.id)
                       END AS owner_user_id
                FROM users
                LEFT JOIN users_with_data ON users_with_data.user_id = users.id
                LEFT JOIN LATERAL (
                    SELECT household_access.owner_user_id
                    FROM household_access
                    WHERE household_access.grantee_user_id = users.id
                      AND household_access.role = 'partner'
                      AND household_access.revoked_at IS NULL
                    ORDER BY household_access.created_at, household_access.owner_user_id
                    LIMIT 1
                ) AS partner ON TRUE
                WHERE users.deleted_at IS NULL
            ),
            eligible AS (
                SELECT resolved_owners.login_user_id, resolved_owners.owner_user_id
                FROM resolved_owners
                JOIN latest_consent
                  ON latest_consent.user_id = resolved_owners.login_user_id
                JOIN profiles
                  ON profiles.user_id = resolved_owners.login_user_id
                JOIN weekly_plans
                  ON weekly_plans.user_id = resolved_owners.owner_user_id
                 AND weekly_plans.week = :week
                WHERE latest_consent.withdrawn_at IS NULL
            )
            SELECT count(*)::int AS eligible_count,
                   count(*) FILTER (
                       WHERE NOT EXISTS (
                           SELECT 1 FROM alerts_log
                           WHERE alerts_log.user_id = eligible.owner_user_id
                             AND alerts_log.dedupe_key =
                                 'weekly:' || :week || ':' || eligible.login_user_id::text
                       )
                   )::int AS missing_count
            FROM eligible
            """
        ),
        {"week": week},
    ).one()
    return int(row.eligible_count), int(row.missing_count)


def load_health_facts(
    engine: Engine,
    now: datetime,
    *,
    email_sending_configured: bool,
) -> HealthFacts:
    """Load aggregate source facts without reading personal fields or writing rows."""
    current = now.astimezone(UTC)
    research_week = _iso_week(_most_recent_saturday(current))
    plan_week = _iso_week(_most_recent_completed_sunday(current))
    with read_only_transaction(engine) as connection:
        india_prices = _date_row_counts(
            connection,
            table="prices_eod",
            market_clause="security.market = 'IN' AND data.source = 'NSE'",
            parameters={"as_of": current.date()},
        )
        india_fund_prices = _date_row_counts(
            connection,
            table="nav_daily",
            market_clause="security.market = 'IN' AND security.security_type = 'mf'",
            parameters={"as_of": current.date()},
        )
        abroad_prices = _date_row_counts(
            connection,
            table="prices_eod",
            market_clause=(
                "security.market IN ('US', 'EU', 'GB') AND data.source = 'AlphaVantage'"
            ),
            parameters={"as_of": current.date()},
        )
        fx_rows = connection.execute(
            text(
                "SELECT pair, max(as_of) AS as_of FROM fx_rates "
                "WHERE pair = ANY(:pairs) AND as_of <= :as_of GROUP BY pair"
            ),
            {"pairs": list(_FX_PAIRS), "as_of": current.date()},
        ).all()
        fx_latest = {pair: None for pair in _FX_PAIRS}
        fx_latest.update({str(row.pair): row.as_of for row in fx_rows})
        basket_rows = connection.execute(
            text("SELECT id, as_of, members_total, members_found FROM fund_baskets ORDER BY id")
        ).all()
        fund_baskets = tuple(
            FundBasketHealthFact(
                basket_id=str(row.id),
                as_of=row.as_of,
                members_total=int(row.members_total),
                members_found=int(row.members_found),
            )
            for row in basket_rows
        )
        stock_checks_as_of = connection.execute(
            text("SELECT max(as_of) FROM stock_checks WHERE as_of <= :as_of"),
            {"as_of": current.date()},
        ).scalar_one()
        held_listed = connection.execute(
            text(
                """
                WITH held AS (
                    SELECT security_id
                    FROM holdings_lots
                    WHERE quantity > 0
                    UNION
                    SELECT security_id
                    FROM recorded_buys
                    WHERE status = 'waiting'
                      AND security_type = 'equity'
                      AND security_id IS NOT NULL
                ), listed AS (
                    SELECT DISTINCT held.security_id
                    FROM held
                    JOIN securities ON securities.id = held.security_id
                    JOIN universe_members ON universe_members.security_id = held.security_id
                    WHERE universe_members.removed_on IS NULL
                      AND ((securities.market = 'IN' AND universe_members.universe = 'nifty500')
                        OR (securities.market = 'US' AND universe_members.universe = 'sp500'))
                )
                SELECT count(*)::int AS total,
                       count(*) FILTER (
                           WHERE latest.as_of >= :fresh_after
                             AND latest.as_of <= :as_of
                       )::int AS fresh
                FROM listed
                LEFT JOIN LATERAL (
                    SELECT stock_checks.as_of
                    FROM stock_checks
                    WHERE stock_checks.security_id = listed.security_id
                    ORDER BY stock_checks.as_of DESC
                    LIMIT 1
                ) AS latest ON TRUE
                """
            ),
            {"as_of": current.date(), "fresh_after": current.date() - timedelta(days=14)},
        ).one()
        run_rows = connection.execute(
            text(
                "SELECT started_at, status, counts, errors FROM marketdata_runs "
                "WHERE started_at >= :cutoff AND started_at <= :now ORDER BY started_at"
            ),
            {"cutoff": current - timedelta(days=WEEKLY_DATA_MAX_AGE_DAYS), "now": current},
        ).all()
        marketdata_runs = tuple(
            _run_fact(row.started_at, row.status, row.counts, row.errors) for row in run_rows
        )
        pack_row = connection.execute(
            text("SELECT pack FROM research_packs WHERE week = :week"),
            {"week": research_week},
        ).one_or_none()
        pack_idea_counts = _idea_counts(pack_row.pack) if pack_row is not None else None
        eligible_plans, missing_plans = _weekly_plan_counts(connection, plan_week)
        if email_sending_configured:
            eligible_emails, missing_emails = _weekly_email_counts(connection, plan_week)
        else:
            eligible_emails, missing_emails = 0, 0
    return HealthFacts(
        india_prices=india_prices,
        india_fund_prices=india_fund_prices,
        abroad_prices=abroad_prices,
        fx_latest=fx_latest,
        marketdata_runs=marketdata_runs,
        research_pack_week=research_week,
        research_pack_idea_counts=pack_idea_counts,
        weekly_plan_week=plan_week,
        eligible_plan_count=eligible_plans,
        missing_plan_count=missing_plans,
        eligible_email_count=eligible_emails,
        missing_email_count=missing_emails,
        email_sending_configured=email_sending_configured,
        fund_baskets=fund_baskets,
        stock_checks_as_of=stock_checks_as_of,
        held_listed_count=int(held_listed.total),
        held_listed_fresh_count=int(held_listed.fresh),
    )


def _database_url() -> str | None:
    return os.getenv("OWNER_DATABASE_URL") or os.getenv("TEST_DATABASE_URL") or None


def _email_sending_configured() -> bool:
    provider = os.getenv("EMAIL_PROVIDER", "dry_run").strip().casefold()
    return provider in {"brevo", "resend"}


def _github_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def emit_github_outputs(
    result: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
) -> None:
    """Write workflow annotations and the optional step summary."""
    values = environ if environ is not None else os.environ
    if values.get("GITHUB_ACTIONS", "").casefold() != "true":
        return
    output = stdout if stdout is not None else sys.stdout
    checks = result.get("checks", [])
    for item in checks:
        level = item.get("level")
        if level not in {"problem", "warning"}:
            continue
        command = "error" if level == "problem" else "warning"
        key = _github_escape(str(item.get("key", "health")))
        message = _github_escape(str(item.get("message", "Data health check needs attention.")))
        print(f"::{command} title={key}::{message}", file=output)
    summary_path = values.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = ["| Check | Result | Message |", "| --- | --- | --- |"]
    for item in checks:
        key = str(item.get("key", "health")).replace("|", "\\|")
        level = str(item.get("level", "ok")).replace("|", "\\|")
        message = str(item.get("message", "")).replace("|", "\\|")
        lines.append(f"| {key} | {level} | {message} |")
    with Path(summary_path).open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def _parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--now must be an ISO datetime in UTC") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError("--now must be an ISO datetime in UTC")
    return parsed.astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check stored data used by scheduled workflows")
    parser.add_argument("--now", help="fixed ISO datetime in UTC for tests and reruns")
    arguments = parser.parse_args(argv)
    try:
        current = _parse_now(arguments.now)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    database_url = _database_url()
    if not database_url:
        print(
            "Set OWNER_DATABASE_URL or TEST_DATABASE_URL to run the data health check.",
            file=sys.stderr,
        )
        return 2
    try:
        engine = create_engine(normalize_database_url(database_url), pool_pre_ping=False)
        try:
            facts = load_health_facts(
                engine,
                current,
                email_sending_configured=_email_sending_configured(),
            )
        finally:
            engine.dispose()
    except Exception:
        print("Data health check could not read the database.", file=sys.stderr)
        return 2

    result = evaluate_health(facts, current)
    try:
        emit_github_outputs(result)
    except OSError:
        print("Data health check could not write the workflow summary.", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
