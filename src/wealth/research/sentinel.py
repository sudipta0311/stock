"""Rules-first daily filing Sentinel with a tightly budgeted optional nano review."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from wealth.db.models import Security, SentinelEvent
from wealth.db.session import make_engine, system_session
from wealth.marketdata.sources.edgar import (
    EdgarCompany,
    parse_company_tickers,
    ticker_lookup,
)
from wealth.marketdata.sources.edgar_filings import extract_8k_items
from wealth.research.dossier import DossierBudget
from wealth.research.errors import BudgetExceeded
from wealth.research.router import ModelRouter, RunBudget, RunContext
from wealth.research.tools import ToolRegistry, ToolResult, evidence_message

_LOGGER = logging.getLogger("wealth.research.sentinel")
_INDIA = ZoneInfo("Asia/Kolkata")
_RULES_PATH = Path(__file__).resolve().parents[3] / "rules" / "sentinel.yaml"
_ITEM_EVENTS = {
    "4.01": "AUDITOR_RESIGNATION",
    "4.02": "RESTATEMENT",
    "5.02": "KEY_PERSON_EXIT",
    "1.03": "BANKRUPTCY",
    "2.06": "IMPAIRMENT",
}
_GOVERNANCE_EVENTS = frozenset(
    {"AUDITOR_RESIGNATION", "RESTATEMENT", "REGULATOR_ENFORCEMENT", "BANKRUPTCY"}
)
_SEC_FORMS = {"8-K", "NT 10-Q", "NT 10-K"}
_MAX_NANO_CALLS = 30


@dataclass(frozen=True)
class SentinelTarget:
    id: UUID
    market: str
    symbol: str | None


class EventCode(StrEnum):
    AUDITOR_RESIGNATION = "AUDITOR_RESIGNATION"
    RESTATEMENT = "RESTATEMENT"
    REGULATOR_ENFORCEMENT = "REGULATOR_ENFORCEMENT"
    BANKRUPTCY = "BANKRUPTCY"
    KEY_PERSON_EXIT = "KEY_PERSON_EXIT"
    IMPAIRMENT = "IMPAIRMENT"
    LATE_FILING = "LATE_FILING"
    OTHER_MATERIAL = "OTHER_MATERIAL"
    NONE = "NONE"


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SentinelClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: bool
    event_code: EventCode
    confidence: Confidence
    quote: str = Field(max_length=300)


def rule_events_for_filing(form: str, text_content: str = "") -> list[str]:
    """Map EDGAR forms and recognized 8-K item identifiers without a model call."""
    normalized_form = " ".join(form.upper().split())
    if normalized_form in {"NT 10-Q", "NT 10-K"}:
        return ["LATE_FILING"]
    if normalized_form != "8-K":
        return []
    items = extract_8k_items(text_content)
    return list(dict.fromkeys(_ITEM_EVENTS[item] for item in sorted(items) if item in _ITEM_EVENTS))


def _rules(path: Path = _RULES_PATH) -> tuple[list[tuple[str, str]], frozenset[str]]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    categories = [
        (str(row["contains"]).casefold(), str(row["event_code"]))
        for row in raw.get("india_categories", [])
    ]
    governance = frozenset(str(value) for value in raw.get("governance_events", []))
    return categories, governance


def _india_rule_event(category: str, text_content: str, rules: list[tuple[str, str]]) -> str | None:
    searchable = f"{category} {text_content}".casefold()
    return next((event_code for needle, event_code in rules if needle in searchable), None)


def _classification_text(result: ToolResult, evidence: Any) -> str:
    scoped = ToolResult(
        tool_call_id=result.tool_call_id,
        result=evidence,
        source_url=result.source_url,
        retrieved_at=result.retrieved_at,
    )
    return evidence_message(scoped)


def _nano_event(
    router: Any | None,
    evidence_result: ToolResult,
    evidence: Any,
    evidence_text: str,
    run_ctx: RunContext,
) -> str | None:
    if router is None or not evidence_text.strip():
        return None
    messages = [
        {
            "role": "system",
            "content": (
                "Classify only material company events supported by the cited evidence. "
                "Use NONE when the text does not establish a material event. The quote must "
                "be an exact substring of the evidence, at most 300 characters. Do not infer "
                "amounts, rates, dates, or facts absent from the evidence."
            ),
        },
        {
            "role": "user",
            "content": _classification_text(evidence_result, evidence),
        },
    ]
    try:
        response = router.complete(
            "nano",
            messages,
            schema=SentinelClassification,
            max_output_tokens=240,
            run_ctx=run_ctx,
        )
    except BudgetExceeded:
        raise
    parsed = getattr(response, "parsed", response)
    classification = (
        parsed
        if isinstance(parsed, SentinelClassification)
        else SentinelClassification.model_validate(parsed)
    )
    if not classification.material or classification.event_code == EventCode.NONE:
        return None
    quote = classification.quote
    if not quote or len(quote) > 300 or quote not in evidence_text:
        _LOGGER.info("Rejected Sentinel classification with a quote outside the evidence.")
        return None
    return classification.event_code.value


def _store_event(
    engine: Engine,
    *,
    security_id: UUID,
    event_code: str,
    source: str,
    evidence_tool_call_id: str,
    method: str,
    now: datetime,
    governance_events: frozenset[str],
) -> UUID:
    event_id = UUID(
        bytes=hashlib.sha256(f"{security_id}:{event_code}:{source}".encode()).digest()[:16]
    )
    statement = pg_insert(SentinelEvent).values(
        id=event_id,
        security_id=security_id,
        event_code=event_code,
        source=source,
        evidence_tool_call_id=UUID(evidence_tool_call_id),
        detected_at=now,
        method=method,
        urgent=event_code in governance_events,
    )
    statement = statement.on_conflict_do_nothing(constraint="uq_sentinel_event_source")
    with system_session(engine) as session:
        session.execute(statement)
        saved_id = session.scalar(
            select(SentinelEvent.id).where(
                SentinelEvent.security_id == security_id,
                SentinelEvent.event_code == event_code,
                SentinelEvent.source == source,
            )
        )
    if saved_id is None:
        raise RuntimeError("Sentinel event was not saved")
    return saved_id


def _targets(engine: Engine, week: str) -> list[SentinelTarget]:
    with system_session(engine) as session:
        held_ids = set(
            session.scalars(
                text("SELECT DISTINCT security_id FROM holdings_lots WHERE quantity > 0")
            )
        )
        shortlisted_ids = set(
            session.scalars(
                text(
                    "SELECT security_id FROM dossiers "
                    "WHERE week = :week AND stage <> 'QUEUED' AND security_id IS NOT NULL"
                ),
                {"week": week},
            )
        )
        security_ids = held_ids | shortlisted_ids
        if not security_ids:
            return []
        rows = session.execute(
            select(Security.id, Security.market, Security.symbol)
            .where(Security.id.in_(security_ids), Security.security_type == "equity")
            .order_by(Security.market, Security.symbol)
        ).all()
        return [SentinelTarget(id=row.id, market=row.market, symbol=row.symbol) for row in rows]


def _company_map(
    registry: ToolRegistry, symbols: list[str], run_ctx: RunContext
) -> dict[str, EdgarCompany]:
    result = registry.call("edgar_company_tickers", {"tickers": symbols}, run_ctx)
    companies = parse_company_tickers(
        {
            str(index): {"ticker": row["ticker"], "cik_str": int(row["cik"]), "title": row["name"]}
            for index, row in enumerate(result.result)
        }
    )
    return companies


def _process_edgar(
    engine: Engine,
    registry: ToolRegistry,
    targets: list[SentinelTarget],
    *,
    days: int,
    as_of: date,
    now: datetime,
    router: Any | None,
    run_ctx: RunContext,
    governance_events: frozenset[str],
) -> dict[str, int | str]:
    symbols = [
        security.symbol for security in targets if security.market == "US" and security.symbol
    ]
    if not symbols:
        return {"status": "no_targets", "filings": 0, "events": 0, "nano_calls": 0}
    try:
        companies = _company_map(registry, symbols, run_ctx)
    except Exception as error:
        _LOGGER.info("EDGAR unavailable (%s).", type(error).__name__)
        return {
            "status": "unavailable",
            "reason": type(error).__name__,
            "filings": 0,
            "events": 0,
            "nano_calls": 0,
        }
    by_symbol: dict[str, EdgarCompany] = {}
    for security in targets:
        if security.market != "US" or not security.symbol:
            continue
        company = ticker_lookup(companies, security.symbol)
        if company is not None:
            by_symbol[security.symbol] = company

    event_count = filing_count = 0
    nano_errors = 0
    for security in targets:
        if security.market != "US" or not security.symbol:
            continue
        company = by_symbol.get(security.symbol)
        if company is None:
            continue
        try:
            filings_result = registry.call(
                "edgar_filings",
                {
                    "cik": company.cik,
                    "forms": sorted(_SEC_FORMS),
                    "days": days,
                    "as_of": as_of.isoformat(),
                },
                run_ctx,
            )
        except Exception as error:
            _LOGGER.info("EDGAR unavailable for one issuer (%s).", type(error).__name__)
            continue
        for filing in filings_result.result:
            filing_count += 1
            form = str(filing["form"])
            filing_url = str(filing["document_url"])
            evidence_id = filings_result.tool_call_id
            document_text = ""
            if form == "8-K":
                try:
                    text_result = registry.call(
                        "edgar_filing_text",
                        {"accession": filing["accession"], "max_chars": 20_000},
                        run_ctx,
                    )
                except Exception as error:
                    _LOGGER.info("EDGAR filing text unavailable (%s).", type(error).__name__)
                    continue
                evidence_id = text_result.tool_call_id
                document_text = str(text_result.result.get("text", ""))
            known_events = rule_events_for_filing(form, document_text)
            for event_code in known_events:
                _store_event(
                    engine,
                    security_id=security.id,
                    event_code=event_code,
                    source=filing_url,
                    evidence_tool_call_id=evidence_id,
                    method="rule",
                    now=now,
                    governance_events=governance_events,
                )
                event_count += 1
            if form == "8-K" and not known_events and router is not None:
                try:
                    event_code = _nano_event(
                        router,
                        text_result,
                        text_result.result,
                        document_text,
                        run_ctx,
                    )
                except BudgetExceeded:
                    return {
                        "status": "complete",
                        "filings": filing_count,
                        "events": event_count,
                        "nano_calls": run_ctx.budget.calls,
                        "nano_budget_exhausted": 1,
                    }
                except Exception as error:
                    _LOGGER.info("Nano classification skipped (%s).", type(error).__name__)
                    nano_errors += 1
                    event_code = None
                if event_code is not None:
                    _store_event(
                        engine,
                        security_id=security.id,
                        event_code=event_code,
                        source=filing_url,
                        evidence_tool_call_id=evidence_id,
                        method="llm",
                        now=now,
                        governance_events=governance_events,
                    )
                    event_count += 1
    return {
        "status": "available",
        "filings": filing_count,
        "events": event_count,
        "nano_calls": run_ctx.budget.calls,
        "nano_errors": nano_errors,
    }


def _process_india(
    engine: Engine,
    registry: ToolRegistry,
    targets: list[SentinelTarget],
    *,
    days: int,
    as_of: date,
    now: datetime,
    router: Any | None,
    run_ctx: RunContext,
    category_rules: list[tuple[str, str]],
    governance_events: frozenset[str],
) -> dict[str, int | str]:
    indian_targets = [
        security for security in targets if security.market == "IN" and security.symbol
    ]
    if not indian_targets:
        return {"status": "no_targets", "announcements": 0, "events": 0, "nano_calls": 0}
    announcement_count = event_count = nano_errors = 0
    for security in indian_targets:
        try:
            result = registry.call(
                "nse_announcements",
                {"symbol": security.symbol, "days": days, "as_of": as_of.isoformat()},
                run_ctx,
            )
        except Exception as error:
            _LOGGER.info("NSE announcements unavailable (%s).", type(error).__name__)
            return {
                "status": "unavailable",
                "reason": type(error).__name__,
                "announcements": announcement_count,
                "events": event_count,
                "nano_calls": run_ctx.budget.calls,
            }
        payload = result.result
        if payload.get("status") == "unavailable":
            _LOGGER.info("NSE announcements unavailable for this run.")
            return {
                "status": "unavailable",
                "announcements": announcement_count,
                "events": event_count,
                "nano_calls": run_ctx.budget.calls,
            }
        for announcement in payload.get("announcements", []):
            announcement_count += 1
            category = str(announcement.get("category", ""))
            text_content = str(announcement.get("text", ""))
            event_code = _india_rule_event(category, text_content, category_rules)
            method = "rule"
            evidence_id = result.tool_call_id
            if event_code is None and router is not None:
                try:
                    event_code = _nano_event(
                        router,
                        result,
                        announcement,
                        text_content,
                        run_ctx,
                    )
                except BudgetExceeded:
                    return {
                        "status": "available",
                        "announcements": announcement_count,
                        "events": event_count,
                        "nano_calls": run_ctx.budget.calls,
                        "nano_budget_exhausted": 1,
                    }
                except Exception as error:
                    _LOGGER.info("Nano classification skipped (%s).", type(error).__name__)
                    nano_errors += 1
                method = "llm"
            if event_code is None:
                continue
            identity = "|".join(
                (security.symbol or "", str(announcement.get("date", "")), category, text_content)
            )
            source = f"NSE:{security.symbol}:{hashlib.sha256(identity.encode()).hexdigest()}"
            _store_event(
                engine,
                security_id=security.id,
                event_code=event_code,
                source=source,
                evidence_tool_call_id=evidence_id,
                method=method,
                now=now,
                governance_events=governance_events,
            )
            event_count += 1
    return {
        "status": "available",
        "announcements": announcement_count,
        "events": event_count,
        "nano_calls": run_ctx.budget.calls,
        "nano_errors": nano_errors,
    }


def run_sentinel(
    engine: Engine,
    *,
    days: int = 2,
    now: datetime | None = None,
    router: Any | None = None,
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Check current holders and this week's shortlist, using rules before nano review."""
    if not 1 <= days <= 30:
        raise ValueError("days must be between 1 and 30")
    current = now or datetime.now(UTC)
    current_day = current.astimezone(_INDIA).date()
    iso = current_day.isocalendar()
    week = f"{iso.year:04d}-W{iso.week:02d}"
    targets = _targets(engine, week)
    counts = {"targets": len(targets), "IN": 0, "US": 0}
    for security in targets:
        counts[security.market] = counts.get(security.market, 0) + 1
    if not targets:
        if not os.getenv("SEC_USER_AGENT", "").strip():
            _LOGGER.info("EDGAR unavailable: SEC_USER_AGENT is not configured.")
            edgar_status = {
                "status": "unavailable",
                "reason": "SEC_USER_AGENT is not configured",
            }
        else:
            edgar_status = {"status": "not_needed"}
        return {
            "status": "no_targets",
            "edgar": edgar_status,
            "nse": {"status": "not_needed"},
            **counts,
        }

    model_router = router
    model_status = "available" if model_router is not None else "unavailable"
    if model_router is None and os.getenv("FOUNDRY_ENDPOINT") and os.getenv("FOUNDRY_API_KEY"):
        try:
            model_router = ModelRouter(owner_engine=engine)
            model_status = "available"
        except Exception as error:
            _LOGGER.info("Nano classification unavailable (%s).", type(error).__name__)
            model_status = "unavailable"
    if model_router is None:
        _LOGGER.info("Foundry is unavailable; rule events remain enabled.")

    budget = RunBudget(max_usd=Decimal("5"), max_calls=_MAX_NANO_CALLS)
    run_ctx = RunContext(
        budget=budget,
        mode="daily",
        agent="sentinel",
        owner_engine=engine,
        dossier_budget=DossierBudget(max_calls=10000, max_usd=Decimal("5")),
    )
    selected_registry = registry or ToolRegistry(engine)
    try:
        categories, governance_events = _rules()
        as_of = current_day

        sec_user_agent = os.getenv("SEC_USER_AGENT", "").strip()
        us_targets = [item for item in targets if item.market == "US"]
        if us_targets and not sec_user_agent:
            edgar_result: dict[str, Any] = {
                "status": "unavailable",
                "reason": "SEC_USER_AGENT is not configured",
                "filings": 0,
                "events": 0,
            }
            _LOGGER.info("EDGAR unavailable: SEC_USER_AGENT is not configured.")
        elif us_targets:
            edgar_result = _process_edgar(
                engine,
                selected_registry,
                targets,
                days=days,
                as_of=as_of,
                now=current,
                router=model_router,
                run_ctx=run_ctx,
                governance_events=governance_events,
            )
        else:
            edgar_result = {"status": "not_needed", "filings": 0, "events": 0}

        in_targets = [item for item in targets if item.market == "IN"]
        nse_result = _process_india(
            engine,
            selected_registry,
            in_targets,
            days=days,
            as_of=as_of,
            now=current,
            router=model_router,
            run_ctx=run_ctx,
            category_rules=categories,
            governance_events=governance_events,
        )
        return {
            "status": "complete",
            "week": week,
            "days": days,
            "models": {"nano": model_status, "max_calls": _MAX_NANO_CALLS, "calls": budget.calls},
            "edgar": edgar_result,
            "nse": nse_result,
            **counts,
        }
    finally:
        if registry is None:
            selected_registry.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check held and shortlisted securities for filings"
    )
    parser.add_argument("--days", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.days <= 30:
        parser.error("--days must be between 1 and 30")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    database_url = os.getenv("OWNER_DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
    if not database_url:
        print(json.dumps({"status": "unavailable", "database": "not configured"}))
        return 2
    engine = make_engine(database_url)
    try:
        result = run_sentinel(engine, days=args.days)
    except Exception as error:
        _LOGGER.info("Sentinel unavailable (%s).", type(error).__name__)
        print(json.dumps({"status": "unavailable", "reason": type(error).__name__}))
        return 1
    else:
        print(json.dumps(result, ensure_ascii=False))
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
