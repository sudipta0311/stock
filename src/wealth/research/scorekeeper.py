"""Paper positions and honest benchmark tracking for approved weekly ideas.

Worked example (all values are synthetic): idea A returns 20% against a 10% benchmark,
idea B returns -10% against -5%, and idea C returns 10% against 8%. Their excess returns
are +10, -5, and +2 percentage points: 2 are ahead and average excess return is 2.333333%.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from wealth.core.plain_language import find_banned_terms, sentence_word_counts
from wealth.db.models import PaperPosition, PriceEod, Security, TrackRecord
from wealth.research.errors import BudgetExceeded
from wealth.research.router import FakeRouter, ModelRouter, RunContext

_PERCENT_QUANTUM = Decimal("0.000001")
_CURRENCY_NOTIONAL = {"IN": ("INR", Decimal("100000")), "US": ("USD", Decimal("1000"))}
_BENCHMARKS = {"IN": ("NIFTYBEES",), "US": ("SPY", "VOO", "IVV")}


class ChecklistOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[str] = Field(default_factory=list, max_length=3)


def return_pct(entry: Decimal, current: Decimal) -> Decimal:
    """Return a percentage from two positive, exact prices."""
    if not entry.is_finite() or not current.is_finite() or entry <= 0 or current <= 0:
        raise ValueError("Prices must be finite and positive")
    return (current / entry - 1) * Decimal(100)


def excess_return_pct(
    entry: Decimal,
    current: Decimal,
    benchmark_entry: Decimal,
    benchmark_current: Decimal,
) -> Decimal:
    """Subtract benchmark return from idea return, in percentage points."""
    return return_pct(entry, current) - return_pct(benchmark_entry, benchmark_current)


def _week_end(week: str) -> date:
    year_text, week_text = week.split("-W", maxsplit=1)
    return date.fromisocalendar(int(year_text), int(week_text), 5)


def _week_distance(start: str, end: str) -> int:
    start_year, start_week = (int(value) for value in start.split("-W", maxsplit=1))
    end_year, end_week = (int(value) for value in end.split("-W", maxsplit=1))
    start_monday = date.fromisocalendar(start_year, start_week, 1)
    end_monday = date.fromisocalendar(end_year, end_week, 1)
    return (end_monday - start_monday).days // 7


def _benchmark_security(session: Session, market: str) -> Security | None:
    for symbol in _BENCHMARKS[market]:
        row = session.scalar(
            select(Security).where(Security.market == market, Security.symbol == symbol)
        )
        if row is not None:
            return row
    return None


def _price_on_or_before(session: Session, security_id: UUID | None, on: date) -> PriceEod | None:
    if security_id is None:
        return None
    return session.scalar(
        select(PriceEod)
        .where(PriceEod.security_id == security_id, PriceEod.as_of <= on)
        .order_by(PriceEod.as_of.desc())
        .limit(1)
    )


def _price_after(session: Session, security_id: UUID, after: date) -> PriceEod | None:
    return session.scalar(
        select(PriceEod)
        .where(
            PriceEod.security_id == security_id,
            PriceEod.as_of > after,
            PriceEod.as_of <= after + timedelta(days=10),
        )
        .order_by(PriceEod.as_of)
        .limit(1)
    )


def _percent(value: Decimal | None) -> Decimal | None:
    return value.quantize(_PERCENT_QUANTUM, rounding=ROUND_HALF_UP) if value is not None else None


def _outcome_statuses(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for item in outcomes:
        status = str(item.get("status", "")).upper()
        if item.get("dossier_id"):
            statuses[f"dossier:{item['dossier_id']}"] = status
        if item.get("security_id"):
            statuses[f"security:{item['security_id']}"] = status
        if item.get("market") and item.get("symbol"):
            statuses[f"symbol:{item['market']}:{item['symbol']}"] = status
    return statuses


def _refresh_position(
    session: Session,
    position: PaperPosition,
    as_of: date,
    status_by_id: Mapping[str, str],
    week: str,
) -> None:
    instrument = _price_on_or_before(session, position.security_id, as_of)
    if instrument is None:
        return
    end_date = instrument.as_of
    if position.status == "closed" and position.exit_date is not None:
        end_date = min(end_date, position.exit_date)
        instrument = _price_on_or_before(session, position.security_id, end_date)
        if instrument is None:
            return
    position.current_price = instrument.close
    position.return_pct = _percent(return_pct(position.entry_price, instrument.close))
    if position.benchmark_status == "available" and position.benchmark_entry_price is not None:
        benchmark = session.scalar(
            select(Security).where(
                Security.market == position.market,
                Security.symbol == position.benchmark_symbol,
            )
        )
        benchmark_price = (
            _price_on_or_before(session, benchmark.id, end_date) if benchmark is not None else None
        )
        if benchmark_price is not None:
            position.benchmark_current_price = benchmark_price.close
            position.benchmark_return_pct = _percent(
                return_pct(position.benchmark_entry_price, benchmark_price.close)
            )
            position.excess_return_pct = _percent(
                position.return_pct - position.benchmark_return_pct
            )
    if position.status == "open" and week > position.week:
        idea_status = (
            status_by_id.get(f"dossier:{position.dossier_id}")
            or status_by_id.get(f"security:{position.security_id}")
            or status_by_id.get(f"symbol:{position.market}:{position.symbol}")
        )
        reason = idea_status.lower() if idea_status in {"VETOED", "UNRESOLVED"} else None
        if reason is None and _week_distance(position.week, week) >= 52:
            reason = "52_weeks"
        if reason is not None:
            position.status = "closed"
            position.exit_date = instrument.as_of
            position.exit_reason = reason


def _falsifiers_by_id(dossiers: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in dossiers:
        dossier_id = str(item.get("dossier_id", item.get("dossierId", "")))
        verdicts = item.get("verdicts", {})
        red_team = verdicts.get("red_team", {}) if isinstance(verdicts, Mapping) else {}
        falsifier = red_team.get("falsifier") if isinstance(red_team, Mapping) else None
        result[dossier_id] = (
            [str(falsifier)] if isinstance(falsifier, str) and falsifier.strip() else []
        )
    return result


def _add_new_positions(
    session: Session,
    week: str,
    outcomes: Sequence[Mapping[str, Any]],
    falsifiers: Mapping[str, list[str]],
) -> int:
    friday = _week_end(week)
    inserted = 0
    for outcome in outcomes:
        if str(outcome.get("status", "")).upper() != "APPROVED":
            continue
        if outcome.get("market") not in _CURRENCY_NOTIONAL:
            continue
        try:
            security_id = UUID(str(outcome["security_id"]))
        except (KeyError, ValueError):
            continue
        dossier_id = str(outcome.get("dossier_id", ""))
        if not dossier_id:
            continue
        if (
            session.scalar(
                select(PaperPosition.id).where(
                    PaperPosition.week == week, PaperPosition.dossier_id == dossier_id
                )
            )
            is not None
        ):
            continue
        security = session.get(Security, security_id)
        if security is None:
            continue
        entry = _price_after(session, security_id, friday)
        if entry is None or entry.close <= 0:
            continue
        market = str(outcome["market"])
        currency, notional = _CURRENCY_NOTIONAL[market]
        benchmark = _benchmark_security(session, market)
        benchmark_entry = (
            session.scalar(
                select(PriceEod).where(
                    PriceEod.security_id == benchmark.id,
                    PriceEod.as_of == entry.as_of,
                )
            )
            if benchmark is not None
            else None
        )
        session.add(
            PaperPosition(
                week=week,
                market=market,
                dossier_id=dossier_id,
                security_id=security.id,
                symbol=security.symbol or str(outcome.get("symbol", "")),
                confidence=str(outcome.get("confidence", "LOW")),
                currency=currency,
                notional=notional,
                entry_date=entry.as_of,
                entry_price=entry.close,
                benchmark_symbol=benchmark.symbol if benchmark is not None else None,
                benchmark_status="available"
                if benchmark_entry is not None
                else "benchmark_unavailable",
                benchmark_entry_price=benchmark_entry.close
                if benchmark_entry is not None
                else None,
                benchmark_entry_date=benchmark_entry.as_of if benchmark_entry is not None else None,
                current_price=entry.close,
                benchmark_current_price=benchmark_entry.close
                if benchmark_entry is not None
                else None,
                red_team_falsifiers=falsifiers.get(dossier_id, []),
            )
        )
        inserted += 1
    return inserted


def _valid_checklist(items: Sequence[str]) -> list[str]:
    checked: list[str] = []
    for item in items[:3]:
        text = item.strip()
        if (
            not text
            or len(text.split()) > 20
            or re.search(r"\d", text)
            or find_banned_terms(text)
            or any(count > 20 for count in sentence_word_counts(text))
        ):
            continue
        checked.append(text)
    return checked


def _error_checklist(
    worst_ideas: Sequence[Mapping[str, Any]],
    *,
    router: ModelRouter | FakeRouter | None,
    run_ctx: RunContext | None,
) -> list[str]:
    falsifiers = [
        str(item["red_team_falsifier"]) for item in worst_ideas if item.get("red_team_falsifier")
    ]
    if router is not None and run_ctx is not None and falsifiers:
        try:
            response = router.complete(
                "mini",
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarise recurring research mistakes as up to three short "
                            "checklist items. Use only the supplied falsifiers. Do not include "
                            "numbers, decisions, or company names."
                        ),
                    },
                    {"role": "user", "content": "\n".join(falsifiers[:5])},
                ],
                schema=ChecklistOutput,
                max_output_tokens=220,
                run_ctx=run_ctx,
            )
            result = (
                response.parsed
                if isinstance(response.parsed, ChecklistOutput)
                else ChecklistOutput.model_validate(response.parsed)
            )
            accepted = _valid_checklist(result.items)
            if accepted:
                return accepted
        except (BudgetExceeded, ValueError, TypeError):
            pass
    return (
        ["Review the failed assumptions behind the weakest ideas before approving new work."]
        if worst_ideas
        else []
    )


def _summary_text(
    market: str, benchmark: str, start: date, avg: Decimal, count: int, ahead: int
) -> str:
    value = avg.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    prefix = "+" if value > 0 else ""
    market_name = "Stock" if market == "IN" else "U.S. stock"
    return (
        f"{market_name} ideas since {start.isoformat()}: {prefix}{value}% vs {benchmark} "
        f"({count} ideas, {ahead} ahead)."
    )


def _track_snapshot(
    positions: Sequence[PaperPosition],
    week: str,
    *,
    router: ModelRouter | FakeRouter | None,
    run_ctx: RunContext | None,
) -> dict[str, Any]:
    by_market = {
        market: [item for item in positions if item.market == market] for market in ("IN", "US")
    }
    market_details: dict[str, dict[str, Any]] = {}
    full_details: dict[str, Any] = {}
    for market, rows in by_market.items():
        comparable = [item for item in rows if item.excess_return_pct is not None]
        ahead = sum(1 for item in comparable if item.excess_return_pct > 0)
        average = (
            sum((Decimal(item.excess_return_pct) for item in comparable), Decimal(0))
            / len(comparable)
            if comparable
            else None
        )
        benchmark_name = "Nifty 50" if market == "IN" else "S&P 500"
        start = min((item.entry_date for item in rows), default=None)
        text = (
            _summary_text(market, benchmark_name, start, average, len(comparable), ahead)
            if start is not None and average is not None
            else None
        )
        calibration: dict[str, dict[str, Any]] = {}
        for band in ("HIGH", "MEDIUM", "LOW"):
            sample = [item for item in comparable if item.confidence == band]
            calibration[band] = {
                "status": "available" if len(sample) >= 10 else "insufficient_data",
                "sample_count": len(sample),
                "hit_rate_pct": (
                    format(
                        Decimal(sum(item.excess_return_pct > 0 for item in sample))
                        / len(sample)
                        * 100,
                        ".1f",
                    )
                    if len(sample) >= 10
                    else None
                ),
            }
        market_details[market] = {
            "status": "available" if comparable else "insufficient_data",
            "benchmark": benchmark_name,
            "started_on": start.isoformat() if start else None,
            "summary": text,
            "ideas_count": len(rows),
            "benchmarked_count": len(comparable),
            "ahead_count": ahead,
            "avg_excess_return_pct": format(average, "f") if average is not None else None,
            "benchmark_unavailable_count": len(rows) - len(comparable),
            "calibration": calibration,
        }
        full_details[market] = [
            {
                "dossier_id": item.dossier_id,
                "symbol": item.symbol,
                "confidence": item.confidence,
                "entry_date": item.entry_date.isoformat(),
                "entry_price": format(item.entry_price, "f"),
                "current_price": format(item.current_price, "f")
                if item.current_price is not None
                else None,
                "return_pct": format(item.return_pct, "f") if item.return_pct is not None else None,
                "benchmark_return_pct": format(item.benchmark_return_pct, "f")
                if item.benchmark_return_pct is not None
                else None,
                "excess_return_pct": format(item.excess_return_pct, "f")
                if item.excess_return_pct is not None
                else None,
                "benchmark_status": item.benchmark_status,
                "status": item.status,
                "exit_reason": item.exit_reason,
                "red_team_falsifiers": item.red_team_falsifiers,
            }
            for item in rows
        ]
    all_comparable = [item for item in positions if item.excess_return_pct is not None]
    worst = sorted(all_comparable, key=lambda item: (item.excess_return_pct, item.entry_date))[:5]
    worst_ideas = [
        {
            "dossier_id": item.dossier_id,
            "symbol": item.symbol,
            "excess_return_pct": format(item.excess_return_pct, "f"),
            "red_team_falsifier": next(iter(item.red_team_falsifiers), None),
        }
        for item in worst
    ]
    checklist = _error_checklist(worst_ideas, router=router, run_ctx=run_ctx)
    return {
        "status": "available" if all_comparable else "insufficient_data",
        "as_of_week": week,
        "markets": market_details,
        "calibration": {
            market: details["calibration"] for market, details in market_details.items()
        },
        "worst_ideas": worst_ideas,
        "checklist": checklist,
        "details": full_details,
    }


def run_scorekeeper(
    engine: Engine,
    *,
    week: str,
    idea_outcomes: Sequence[Mapping[str, Any]],
    dossiers: Sequence[Mapping[str, Any]] = (),
    router: ModelRouter | FakeRouter | None = None,
    run_ctx: RunContext | None = None,
) -> dict[str, Any]:
    """Open next-day positions, refresh returns, close vetoes, and save weekly outcomes."""
    as_of = _week_end(week)
    status_by_id = _outcome_statuses(idea_outcomes)
    falsifiers = _falsifiers_by_id(dossiers)
    for outcome in idea_outcomes:
        dossier_id = str(outcome.get("dossier_id", ""))
        falsifier = outcome.get("red_team_falsifier")
        if dossier_id and isinstance(falsifier, str) and falsifier.strip():
            falsifiers[dossier_id] = [falsifier]
    with Session(engine) as session, session.begin():
        positions = list(
            session.scalars(
                select(PaperPosition).order_by(PaperPosition.week, PaperPosition.dossier_id)
            )
        )
        for position in positions:
            _refresh_position(session, position, as_of, status_by_id, week)
        _add_new_positions(session, week, idea_outcomes, falsifiers)
        session.flush()
        positions = list(
            session.scalars(
                select(PaperPosition).order_by(PaperPosition.week, PaperPosition.dossier_id)
            )
        )
        track = _track_snapshot(positions, week, router=router, run_ctx=run_ctx)
        for market in ("IN", "US"):
            data = track["markets"][market]
            row = session.get(TrackRecord, (week, market))
            if row is None:
                row = TrackRecord(
                    week=week, market=market, ideas_count=0, ahead_count=0, details={}
                )
                session.add(row)
            row.ideas_count = data["ideas_count"]
            row.ahead_count = data["ahead_count"]
            row.avg_excess_return_pct = (
                Decimal(data["avg_excess_return_pct"]) if data["avg_excess_return_pct"] else None
            )
            row.details = {
                "status": data["status"],
                "started_on": data["started_on"],
                "benchmarked_count": data["benchmarked_count"],
                "benchmark_unavailable_count": data["benchmark_unavailable_count"],
                "calibration": data["calibration"],
                "worst_ideas": track["worst_ideas"],
            }
    return {"track_record": track, "cio_checklist": track["checklist"]}
