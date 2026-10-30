"""Bounded, evidence-checked macro and geopolitics research agent."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from wealth.db.models import FxRate
from wealth.marketdata import store
from wealth.marketdata.sources.macro_geo import FEDERAL_REGISTER_API_URL, FEED_URLS
from wealth.policy.global_split import GeoTilts, load_global_rules
from wealth.policy.rules import decimal_rule
from wealth.research.agents import (
    EVIDENCE_SYSTEM_PROMPT,
    load_prompt_version,
    plain_language_failures,
)
from wealth.research.dossier import (
    Claim,
    Dossier,
    DossierSecurity,
    Evidence,
    Stage,
    validate_grounding,
)
from wealth.research.errors import BudgetExceeded
from wealth.research.router import FakeRouter, ModelRouter, RunContext
from wealth.research.tools import ToolRegistry, ToolResult, evidence_message

MacroTopic = Literal[
    "oil",
    "rupee",
    "us_india_trade",
    "global_risk",
    "rates",
    "india_valuation",
    "us_valuation",
]
MacroLevel = Literal["calm", "elevated", "high"]
_INDEX_NAMES = {
    "nifty 50": "NIFTY 50",
    "nifty next 50": "NIFTY NEXT 50",
    "nifty midcap 150": "NIFTY MIDCAP 150",
    "nifty 500": "NIFTY 500",
}
_ALLOWED_FEED_SOURCES = {"pib", "rbi", "ustr", "federal_register"}
_FEED_SOURCE_URLS = {**FEED_URLS, "federal_register": FEDERAL_REGISTER_API_URL}
_TILT_DECIMAL = Annotated[Decimal, BeforeValidator(lambda value: decimal_rule(value, "tilt"))]
_TOPIC_LABELS = {
    "oil": "oil prices",
    "rupee": "the rupee",
    "us_india_trade": "trade between India and the US",
    "global_risk": "global markets",
    "rates": "interest rates",
    "india_valuation": "Indian share prices",
    "us_valuation": "US share prices",
}
_TOOL_LIMIT = 8
_FRESH_EVIDENCE_DAYS = 7
_DECAY_FLOOR_PP = Decimal("0.5")


class MacroRegimeNote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: MacroTopic
    level: MacroLevel
    text: str = Field(max_length=200)
    evidence: list[str] = Field(min_length=1, max_length=8)


class ProposedGeoTilts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    global_pp: _TILT_DECIMAL = Decimal(0)
    gold_pp: _TILT_DECIMAL = Decimal(0)
    rationale: str = Field(default="", max_length=300)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)


class MacroGeoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    regime_notes: list[MacroRegimeNote] = Field(default_factory=list, max_length=4)
    proposed_tilts: ProposedGeoTilts = Field(default_factory=ProposedGeoTilts)


@dataclass(frozen=True)
class MacroGeoResult:
    as_of: date
    geo_tilts: GeoTilts
    regime_notes: tuple[MacroRegimeNote, ...]
    rationale: str
    tilt_reason: str
    agent_status: str
    index_valuation_percentiles: dict[str, str | None]
    index_valuation_history: dict[str, dict[str, str | int | None]]

    def pack_fields(self) -> dict[str, Any]:
        """Return top-level keys merged into this week's single Research Pack row."""
        return {
            "geo_tilts": self.geo_tilts.model_dump(mode="json"),
            "macro": {
                "as_of": self.as_of.isoformat(),
                "regime_notes": [note.model_dump(mode="json") for note in self.regime_notes],
                "rationale": self.rationale,
                "tilt_reason": self.tilt_reason,
                "agent_status": self.agent_status,
                "index_valuation_history": self.index_valuation_history,
            },
            "index_valuation_percentiles": self.index_valuation_percentiles,
        }


class MacroGeoToolbox:
    """Read-only market-data tools exposed to the macro agent."""

    def __init__(self, engine: Engine | None, as_of: date) -> None:
        self.engine = engine
        self.as_of = as_of

    def register(self, registry: ToolRegistry) -> None:
        registry.register("index_valuation", self.index_valuation)
        registry.register("macro_series", self.macro_series)
        registry.register("fx_trend", self.fx_trend)
        registry.register("feed_search", self.feed_search)

    def _engine(self) -> Engine:
        if self.engine is None:
            raise RuntimeError("Macro market-data tools need an owner database connection")
        return self.engine

    def _index_data(self, index: str) -> dict[str, Any] | None:
        canonical = _INDEX_NAMES.get(index.casefold().strip())
        if canonical is None:
            raise ValueError("Unsupported index")
        with Session(self._engine()) as session:
            value = store.latest_index_valuation_percentile(session, canonical, self.as_of)
        if value is None:
            return None
        return {
            "index_name": value.index_name,
            "as_of": value.as_of.isoformat(),
            "pe": _decimal_json(value.pe),
            "pb": _decimal_json(value.pb),
            "div_yield": _decimal_json(value.div_yield),
            "pe_percentile": _decimal_json(value.percentile),
            "history_label": value.history_label,
            "history_count": value.history_count,
            "source": value.source,
            "source_url": value.source_url,
        }

    def index_valuation(self, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        index = str(args.get("index", ""))
        result = self._index_data(index)
        source_url = str(result["source_url"]) if result is not None else None
        return result or {"index": index, "status": "unavailable"}, source_url

    def macro_series(self, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        series = str(args.get("series", "")).upper()
        if series not in {"UST_10Y", "UST_3M"}:
            raise ValueError("Unsupported Treasury series")
        days = _bounded_integer(args.get("days"), minimum=1, maximum=400, name="days")
        with Session(self._engine()) as session:
            rows = store.macro_series_history(session, series, self.as_of, days=days)
        observations = [
            {"as_of": row.as_of.isoformat(), "value": format(row.value, "f")} for row in rows
        ]
        source_url = rows[-1].source_url if rows else None
        return {
            "series": series,
            "observations": observations,
            "as_of": rows[-1].as_of.isoformat() if rows else None,
            "source_urls": list(dict.fromkeys(row.source_url for row in rows)),
        }, source_url

    def fx_trend(self, _args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        with Session(self._engine()) as session:
            current = session.scalar(
                select(FxRate)
                .where(FxRate.pair == "USD/INR", FxRate.as_of <= self.as_of)
                .order_by(FxRate.as_of.desc())
                .limit(1)
            )
            if current is None:
                return {"pair": "USD/INR", "status": "unavailable"}, None

            changes: dict[str, str | None] = {}
            source_urls = [current.source_url]
            source_dates = [{"url": current.source_url, "as_of": current.as_of.isoformat()}]
            for label, days in (("change_3m_pct", 90), ("change_12m_pct", 365)):
                baseline = session.scalar(
                    select(FxRate)
                    .where(
                        FxRate.pair == "USD/INR",
                        FxRate.as_of <= self.as_of - timedelta(days=days),
                    )
                    .order_by(FxRate.as_of.desc())
                    .limit(1)
                )
                if baseline is None or baseline.rate == 0:
                    changes[label] = None
                    continue
                change = (current.rate / baseline.rate - Decimal(1)) * Decimal(100)
                changes[label] = format(change.quantize(Decimal("0.01")), "f")
                source_urls.append(baseline.source_url)
                source_dates.append(
                    {"url": baseline.source_url, "as_of": baseline.as_of.isoformat()}
                )
        return {
            "pair": "USD/INR",
            "as_of": current.as_of.isoformat(),
            "rate": format(current.rate, "f"),
            **changes,
            "source_urls": list(dict.fromkeys(source_urls)),
            "source_dates": source_dates,
        }, current.source_url

    def feed_search(self, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        query = str(args.get("query", "")).strip()
        days = _bounded_integer(args.get("days"), minimum=1, maximum=365, name="days")
        raw_sources = args.get("sources", [])
        if not isinstance(raw_sources, list) or any(
            not isinstance(item, str) for item in raw_sources
        ):
            raise ValueError("sources must be a list")
        sources = list(dict.fromkeys(item.casefold() for item in raw_sources))
        if not set(sources) <= _ALLOWED_FEED_SOURCES:
            raise ValueError("Unsupported feed source")
        with Session(self._engine()) as session:
            rows = store.search_feed_items(
                session,
                query,
                self.as_of,
                days=days,
                sources=sources or None,
                limit=16,
            )
        items = [
            {
                "source": row.source,
                "source_url": _FEED_SOURCE_URLS[row.source],
                "guid": row.guid,
                "title": row.title,
                "url": row.url,
                "published_at": row.published_at.isoformat() if row.published_at else None,
                "summary": row.summary,
            }
            for row in rows
        ]
        source_urls = list(dict.fromkeys(item["source_url"] for item in items))
        if not source_urls and len(sources) == 1:
            source_urls = [_FEED_SOURCE_URLS[sources[0]]]
        result_source_url = source_urls[0] if len(source_urls) == 1 else None
        return {"query": query[:200], "items": items, "source_urls": source_urls}, result_source_url

    def policy_index_percentiles(
        self,
    ) -> tuple[dict[str, str | None], dict[str, dict[str, str | int | None]]]:
        percentiles: dict[str, str | None] = {"IN": None, "US": None}
        history: dict[str, dict[str, str | int | None]] = {
            "IN": {"index": "NIFTY 50", "history_label": "unavailable", "as_of": None},
            "US": {"index": None, "history_label": "unavailable", "as_of": None},
        }
        if self.engine is None:
            return percentiles, history
        with Session(self.engine) as session:
            value = store.latest_index_valuation_percentile(session, "NIFTY 50", self.as_of)
        if value is not None:
            percentiles["IN"] = _decimal_json(value.percentile)
            history["IN"] = {
                "index": value.index_name,
                "history_label": value.history_label,
                "as_of": value.as_of.isoformat(),
                "history_count": value.history_count,
            }
        return percentiles, history


class MacroGeoAgent:
    """Ask the mini tier for grounded context and let code own the tilt decision."""

    def __init__(
        self,
        router: ModelRouter | FakeRouter,
        tools: ToolRegistry,
        run_ctx: RunContext,
        *,
        as_of: date,
        week: str,
        toolbox: MacroGeoToolbox | None = None,
    ) -> None:
        self.router = router
        self.tools = tools
        self.run_ctx = run_ctx
        self.as_of = as_of
        self.week = week
        self.toolbox = toolbox

    def run(self, previous: GeoTilts | None = None) -> MacroGeoResult:
        prompt = load_prompt_version("macro_geo")
        self.run_ctx.agent = "macro_geo_agent"
        self.run_ctx.dossier_id = None
        self.run_ctx.prompt_version_sha256 = prompt.sha256
        evidence: dict[str, ToolResult] = {}
        output: MacroGeoOutput | None = None
        agent_status = "completed"
        try:
            messages = self._initial_messages(prompt.text, previous)
            self._call_tools(messages, evidence)
            response = self.router.complete(
                "mini",
                messages,
                schema=MacroGeoOutput,
                max_output_tokens=850,
                run_ctx=self.run_ctx,
            )
            if isinstance(response.parsed, MacroGeoOutput):
                output = response.parsed
            else:
                output = MacroGeoOutput.model_validate(response.parsed)
        except BudgetExceeded:
            agent_status = "budget_exceeded"
        except Exception:
            agent_status = "unavailable"

        notes = self._ground_regime_notes(output, evidence)
        proposal = output.proposed_tilts if output is not None else None
        tilts, tilt_reason, accepted_ids = self._apply_tilt_policy(proposal, evidence, previous)
        rationale = self._plain_rationale(proposal, tilt_reason, evidence, accepted_ids)
        index_percentiles, index_history = (
            self.toolbox.policy_index_percentiles()
            if self.toolbox is not None
            else (
                {"IN": None, "US": None},
                {
                    "IN": {
                        "index": "NIFTY 50",
                        "history_label": "unavailable",
                        "as_of": None,
                    },
                    "US": {"index": None, "history_label": "unavailable", "as_of": None},
                },
            )
        )
        return MacroGeoResult(
            as_of=self.as_of,
            geo_tilts=tilts,
            regime_notes=tuple(notes),
            rationale=rationale,
            tilt_reason=tilt_reason,
            agent_status=agent_status,
            index_valuation_percentiles=index_percentiles,
            index_valuation_history=index_history,
        )

    def _initial_messages(
        self, prompt_text: str, previous: GeoTilts | None
    ) -> list[dict[str, Any]]:
        prior = previous.model_dump(mode="json") if previous is not None else None
        return [
            {
                "role": "system",
                "content": f"{EVIDENCE_SYSTEM_PROMPT}\n{prompt_text}",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "as_of": self.as_of.isoformat(),
                        "allowed_topics": list(MacroTopic.__args__),
                        "previous_week_tilts": prior,
                        "output_schema": MacroGeoOutput.model_json_schema(),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]

    def _call_tools(self, messages: list[dict[str, Any]], evidence: dict[str, ToolResult]) -> None:
        calls_used = 0
        while calls_used < _TOOL_LIMIT:
            response = self.router.complete(
                "mini",
                messages,
                schema=None,
                max_output_tokens=650,
                run_ctx=self.run_ctx,
                tools=_TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            requested_calls = list(response.tool_calls[: _TOOL_LIMIT - calls_used])
            if not requested_calls:
                if response.raw_text:
                    messages.append({"role": "assistant", "content": response.raw_text})
                return
            messages.append(
                {
                    "role": "assistant",
                    "content": response.raw_text or None,
                    "tool_calls": requested_calls,
                }
            )
            for requested in requested_calls:
                calls_used += 1
                call_id = str(requested.get("id", ""))
                function = requested.get("function", {})
                name = str(function.get("name", ""))
                try:
                    raw_arguments = json.loads(str(function.get("arguments", "{}")))
                    if not isinstance(raw_arguments, dict):
                        raise ValueError("Tool arguments must be an object")
                    _validate_tool_arguments(name, raw_arguments)
                    result = self.tools.call(name, raw_arguments, self.run_ctx)
                    evidence[result.tool_call_id] = result
                    content = evidence_message(result)
                except BudgetExceeded:
                    raise
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    content = json.dumps({"error": "tool_request_rejected"})
                except Exception:
                    content = json.dumps({"error": "tool_unavailable"})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def _ground_regime_notes(
        self,
        output: MacroGeoOutput | None,
        evidence: Mapping[str, ToolResult],
    ) -> list[MacroRegimeNote]:
        if output is None:
            return []
        known_ids = set(evidence)
        accepted: list[MacroRegimeNote] = []
        for note in output.regime_notes[:4]:
            citations = _known_evidence_ids(note.evidence, known_ids)
            if not citations:
                continue
            text = note.text.strip()
            if not text:
                continue
            if plain_language_failures(text):
                text = (
                    "Official data did not support a clear view of "
                    f"{_TOPIC_LABELS[note.topic]} this week."
                )
            grounded = _grounded_text(text, citations, evidence, self.week)
            if grounded:
                accepted.append(note.model_copy(update={"text": text, "evidence": citations}))
        return accepted

    def _apply_tilt_policy(
        self,
        proposal: ProposedGeoTilts | None,
        evidence: Mapping[str, ToolResult],
        previous: GeoTilts | None,
    ) -> tuple[GeoTilts, str, list[str]]:
        if proposal is None:
            return self._decay(previous)
        citations = _known_evidence_ids(proposal.evidence_ids, set(evidence))
        hosts = set().union(*(_result_hosts(evidence[item]) for item in citations))
        fresh_hosts = set().union(*(_fresh_hosts(evidence[item], self.as_of) for item in citations))
        if len(citations) < 2 or len(hosts) < 2:
            decayed, _decay_reason, _prior_ids = self._decay(previous)
            return decayed, "insufficient_independent_evidence", []
        if len(fresh_hosts) < 2:
            decayed, _decay_reason, _prior_ids = self._decay(previous)
            return decayed, "stale_evidence", []

        bounds = load_global_rules()["tilt_bounds"]
        global_bound = decimal_rule(bounds["global_pp"], "tilt_bounds.global_pp")
        gold_bound = decimal_rule(bounds["gold_pp"], "tilt_bounds.gold_pp")
        global_pp = max(-global_bound, min(global_bound, proposal.global_pp))
        gold_pp = max(-gold_bound, min(gold_bound, proposal.gold_pp))
        return (
            GeoTilts(
                global_pp=global_pp,
                gold_pp=gold_pp,
                evidence_ids=citations,
                as_of=self.as_of,
            ),
            "reconfirmed" if global_pp or gold_pp else "confirmed_no_tilt",
            citations,
        )

    def _decay(self, previous: GeoTilts | None) -> tuple[GeoTilts, str, list[str]]:
        if previous is None:
            return (
                GeoTilts(
                    global_pp=Decimal(0), gold_pp=Decimal(0), evidence_ids=[], as_of=self.as_of
                ),
                "no_fresh_proposal",
                [],
            )
        if previous.as_of > self.as_of:
            return (
                GeoTilts(
                    global_pp=Decimal(0), gold_pp=Decimal(0), evidence_ids=[], as_of=self.as_of
                ),
                "prior_tilt_is_future_dated",
                [],
            )
        previous_week = previous.as_of - timedelta(days=previous.as_of.weekday())
        current_week = self.as_of - timedelta(days=self.as_of.weekday())
        elapsed_weeks = max(1, (current_week - previous_week).days // 7)
        divisor = Decimal(2**elapsed_weeks)
        global_pp = previous.global_pp / divisor
        gold_pp = previous.gold_pp / divisor
        if abs(global_pp) < _DECAY_FLOOR_PP:
            global_pp = Decimal(0)
        if abs(gold_pp) < _DECAY_FLOOR_PP:
            gold_pp = Decimal(0)
        retained_ids = previous.evidence_ids if global_pp or gold_pp else []
        reason = "decayed" if global_pp or gold_pp else "decayed_to_zero"
        return (
            GeoTilts(
                global_pp=global_pp,
                gold_pp=gold_pp,
                evidence_ids=retained_ids,
                as_of=self.as_of,
            ),
            reason,
            retained_ids,
        )

    def _plain_rationale(
        self,
        proposal: ProposedGeoTilts | None,
        reason: str,
        evidence: Mapping[str, ToolResult],
        citations: list[str],
    ) -> str:
        fallback = (
            "Current evidence does not support a change to this week's targets."
            if reason not in {"reconfirmed", "confirmed_no_tilt"}
            else "Current evidence supports this week's target settings."
        )
        if proposal is None:
            return fallback
        text = proposal.rationale.strip()
        if (
            not text
            or len(text) > 300
            or re.search(r"\d", text)
            or plain_language_failures(text)
            or not citations
            or any(identifier not in evidence for identifier in citations)
        ):
            return fallback
        return text


def register_macro_geo_tools(
    registry: ToolRegistry,
    engine: Engine,
    as_of: date,
) -> MacroGeoToolbox:
    toolbox = MacroGeoToolbox(engine, as_of)
    toolbox.register(registry)
    return toolbox


def _grounded_text(
    text: str,
    citations: list[str],
    evidence: Mapping[str, ToolResult],
    week: str,
) -> bool:
    if plain_language_failures(text):
        return False
    claim = Claim(
        kind="FACT",
        text=text,
        evidence=[
            Evidence(
                tool_call_id=identifier,
                source_url=evidence[identifier].source_url,
                retrieved_at=evidence[identifier].retrieved_at,
            )
            for identifier in citations
        ],
    )
    dossier = Dossier(
        dossier_id=f"{week}/IN/MACRO",
        security=DossierSecurity(id="IN:MACRO", market="IN", type="EQUITY"),
        stage=Stage.SCREENED,
        claims=[claim],
    )
    return not validate_grounding(dossier, evidence)


def _known_evidence_ids(values: Sequence[str], known: set[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value in known))


def _fresh_hosts(result: ToolResult, as_of: date) -> set[str]:
    payload = result.result
    host_dates: dict[str, date] = {}

    def add(url: object, raw_date: object) -> None:
        if not isinstance(url, str) or not isinstance(raw_date, str):
            return
        host = urlparse(url).hostname
        observed = _observation_date(raw_date)
        if host and observed is not None:
            host_dates[host.casefold()] = max(observed, host_dates.get(host.casefold(), observed))

    if isinstance(payload, Mapping):
        items = payload.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, Mapping):
                    add(item.get("source_url"), item.get("published_at"))
        observations = payload.get("observations")
        if isinstance(observations, list):
            urls = payload.get("source_urls", [])
            if isinstance(urls, list):
                for item in observations:
                    if isinstance(item, Mapping):
                        for url in urls:
                            add(url, item.get("as_of"))
        source_dates = payload.get("source_dates")
        if isinstance(source_dates, list):
            for item in source_dates:
                if isinstance(item, Mapping):
                    add(item.get("url"), item.get("as_of"))
        raw_as_of = payload.get("as_of")
        if result.source_url:
            add(result.source_url, raw_as_of)
        source_urls = payload.get("source_urls", [])
        if isinstance(source_urls, list):
            for url in source_urls:
                add(url, raw_as_of)

    return {
        host
        for host, observed in host_dates.items()
        if 0 <= (as_of - observed).days <= _FRESH_EVIDENCE_DAYS
    }


def _result_hosts(result: ToolResult) -> set[str]:
    hosts: set[str] = set()

    def add(value: object) -> None:
        if isinstance(value, str):
            host = urlparse(value).hostname
            if host:
                hosts.add(host.casefold())

    add(result.source_url)
    if isinstance(result.result, Mapping):
        for key in ("url", "source_url"):
            add(result.result.get(key))
        for key in ("source_urls",):
            urls = result.result.get(key)
            if isinstance(urls, list):
                for value in urls:
                    add(value)
        for key in ("items", "source_dates"):
            rows = result.result.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, Mapping):
                        add(row.get("source_url"))
    return hosts


def _observation_date(value: str) -> date | None:
    try:
        if "T" in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _decimal_json(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _bounded_integer(value: Any, *, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(result) != str(value) or not minimum <= result <= maximum:
        raise ValueError(f"{name} is out of range")
    return result


def _validate_tool_arguments(name: str, arguments: dict[str, Any]) -> None:
    if name == "index_valuation":
        if not isinstance(arguments.get("index"), str):
            raise ValueError("index must be text")
        if arguments["index"].casefold().strip() not in _INDEX_NAMES:
            raise ValueError("Unsupported index")
        return
    if name == "macro_series":
        if str(arguments.get("series", "")).upper() not in {"UST_10Y", "UST_3M"}:
            raise ValueError("Unsupported Treasury series")
        _bounded_integer(arguments.get("days"), minimum=1, maximum=400, name="days")
        return
    if name == "fx_trend":
        if arguments:
            raise ValueError("fx_trend takes no arguments")
        return
    if name == "feed_search":
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            raise ValueError("query must be 1 to 200 characters")
        _bounded_integer(arguments.get("days"), minimum=1, maximum=365, name="days")
        sources = arguments.get("sources", [])
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            raise ValueError("sources must be a list")
        if not {item.casefold() for item in sources} <= _ALLOWED_FEED_SOURCES:
            raise ValueError("Unsupported feed source")
        return
    raise ValueError("Tool is not allowlisted")


_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "index_valuation",
            "description": "Read the latest stored valuation and P/E percentile for one NSE index.",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "string", "enum": list(_INDEX_NAMES.values())}},
                "required": ["index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "macro_series",
            "description": "Read recent official US Treasury yield observations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series": {"type": "string", "enum": ["UST_10Y", "UST_3M"]},
                    "days": {"type": "integer", "minimum": 1, "maximum": 400},
                },
                "required": ["series", "days"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fx_trend",
            "description": "Read the 3-month and 12-month USD/INR change from stored FX rates.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "feed_search",
            "description": "Search recent official press-release and Federal Register summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 200},
                    "days": {"type": "integer", "minimum": 1, "maximum": 365},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["pib", "rbi", "ustr", "federal_register"],
                        },
                    },
                },
                "required": ["query", "days", "sources"],
                "additionalProperties": False,
            },
        },
    },
]
