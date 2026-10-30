"""Typed research agents with code-owned gates and quarantined evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wealth.core.plain_language import find_banned_terms, sentence_word_counts
from wealth.research.dossier import (
    CitedReason,
    Claim,
    Dossier,
    Evidence,
    GeoExposure,
    RedTeamVerdict,
    Stage,
    validate_grounding,
)
from wealth.research.errors import BudgetExceeded, SchemaError
from wealth.research.router import FakeRouter, ModelRouter, RunContext
from wealth.research.rules import ResearchRules
from wealth.research.tools import EVIDENCE_SYSTEM_PROMPT, ToolRegistry, ToolResult, evidence_message

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
_PROMPT_FILES = {
    "analyst": "analyst.md",
    "valuation": "valuation.md",
    "red_team": "red_team.md",
    "compliance": "compliance.md",
    "cio": "cio.md",
    "macro_geo": "macro_geo.md",
}
_PROHIBITED_LANGUAGE = (
    re.compile(r"\bguarantee(?:d|s)?\b", re.IGNORECASE),
    re.compile(r"\bassured\s+returns?\b", re.IGNORECASE),
    re.compile(r"\bsure[\s-]?shot\b", re.IGNORECASE),
    re.compile(r"\bmultibagger\b", re.IGNORECASE),
    re.compile(r"\b(?:can['’]?t|cannot)\s+lose\b", re.IGNORECASE),
    re.compile(r"\b100\s*%(?!\w)", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*%\s*returns?\b", re.IGNORECASE),
)
_ANALYST_METRICS = {
    "revenue",
    "eps_diluted",
    "net_income",
    "operating_income",
    "operating_cash_flow",
    "capex",
    "shares_outstanding",
    "total_assets",
    "total_equity",
    "total_debt",
    "current_liabilities",
}
_TOOL_PROPERTIES: dict[str, dict[str, Any]] = {
    "factor_snapshot": {
        "security_id": {"type": "string"},
        "as_of": {"type": "string", "format": "date"},
    },
    "fundamentals_series": {
        "security_id": {"type": "string"},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "quarters": {"type": "integer", "minimum": 1, "maximum": 40},
        "as_of": {"type": "string", "format": "date"},
    },
    "price_stats": {
        "security_id": {"type": "string"},
        "as_of": {"type": "string", "format": "date"},
    },
    "sector_peers": {
        "security_id": {"type": "string"},
        "metric": {"type": "string", "enum": ["earnings_yield", "fcf_yield", "roce", "roe"]},
        "as_of": {"type": "string", "format": "date"},
    },
    "nse_announcements": {
        "symbol": {"type": "string"},
        "days": {"type": "integer", "minimum": 1, "maximum": 90},
        "as_of": {"type": "string", "format": "date"},
    },
    "edgar_company_tickers": {"tickers": {"type": "array", "items": {"type": "string"}}},
    "edgar_filings": {
        "cik": {"type": "string"},
        "forms": {"type": "array", "items": {"type": "string"}},
        "days": {"type": "integer", "minimum": 1, "maximum": 90},
        "as_of": {"type": "string", "format": "date"},
    },
    "edgar_filing_text": {
        "accession": {"type": "string"},
        "max_chars": {"type": "integer", "minimum": 100, "maximum": 20000},
    },
}
_TOOL_DESCRIPTIONS = {
    "factor_snapshot": "Read code-calculated factors for this company.",
    "fundamentals_series": "Read point-in-time company metrics.",
    "price_stats": "Read dated price statistics for this company.",
    "sector_peers": "Compare a code-calculated metric with sector peers.",
    "nse_announcements": "Read recent exchange announcements for this India company.",
    "edgar_company_tickers": "Resolve the named US ticker to a filing identifier.",
    "edgar_filings": "Read recent filings for the resolved US company.",
    "edgar_filing_text": "Read text from a filing returned by the filings tool.",
}
_MARKET_TOOLS = {
    "IN": (
        "factor_snapshot",
        "fundamentals_series",
        "price_stats",
        "sector_peers",
        "nse_announcements",
    ),
    "US": (
        "factor_snapshot",
        "fundamentals_series",
        "price_stats",
        "sector_peers",
        "edgar_company_tickers",
        "edgar_filings",
        "edgar_filing_text",
    ),
}
_CIO_GENERAL_WORDS = {
    "a",
    "an",
    "after",
    "and",
    "are",
    "as",
    "be",
    "been",
    "before",
    "but",
    "by",
    "case",
    "checks",
    "company",
    "could",
    "data",
    "did",
    "does",
    "evidence",
    "for",
    "from",
    "found",
    "has",
    "have",
    "higher",
    "in",
    "is",
    "it",
    "its",
    "lower",
    "may",
    "might",
    "no",
    "not",
    "of",
    "on",
    "or",
    "passed",
    "review",
    "selected",
    "source",
    "strong",
    "the",
    "this",
    "to",
    "week",
    "were",
    "with",
}


class AgentClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FACT", "INFERENCE", "OPINION"]
    text: str = Field(min_length=1, max_length=240)
    evidence: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def fact_requires_tool_evidence(self) -> AgentClaim:
        if self.kind == "FACT" and not self.evidence:
            raise ValueError("FACT claims require cited tool evidence")
        return self


class AgentGeoExposure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    us_revenue_pct: Decimal | None = Field(default=None, ge=0, le=100)
    tariff_sensitive: bool | None = None
    evidence: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def cited_when_known(self) -> AgentGeoExposure:
        if (self.us_revenue_pct is not None or self.tariff_sensitive is not None) and not (
            self.evidence
        ):
            raise ValueError("Known geographic exposure needs a cited tool call")
        return self


class AnalystOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[AgentClaim] = Field(default_factory=list, max_length=12)
    drivers: list[str] = Field(default_factory=list, max_length=3)
    risks: list[str] = Field(default_factory=list, max_length=3)
    geo_exposure: AgentGeoExposure
    fundamental_verdict: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]
    open_questions: list[str] = Field(default_factory=list, max_length=8)


# Used when the model's wording can't pass the plain-language rules; no numbers to ground.
_VERDICT_SENTENCES = {
    "CHEAP": "Our valuation checks find the price low compared with its usual range.",
    "FAIR": "Our valuation checks find the price about fair.",
    "EXPENSIVE": "Our valuation checks find the price high compared with its usual range.",
    "UNKNOWN": "Our valuation checks don't have enough data yet.",
}


class ValuationExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    explanation: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(min_length=1, max_length=4)


class AgentReason(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=300)
    evidence: list[str] = Field(default_factory=list, max_length=8)


class RedTeamOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strongest_bear_case: str = Field(max_length=300)
    falsifier: str = Field(min_length=1, max_length=240)
    downgrade: Literal["NONE", "WATCH", "VETO"]
    reasons: list[AgentReason] = Field(default_factory=list, max_length=8)


class ComplianceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    has_personalised_advice_or_amount: bool


class CIOOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_dossier_ids: list[str] = Field(default_factory=list, max_length=10)
    citations: list[str] = Field(default_factory=list, max_length=20)
    summary: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class PromptVersion:
    text: str
    sha256: str


def load_prompt_version(name: str) -> PromptVersion:
    """Read one versioned prompt and return its exact source hash."""
    filename = _PROMPT_FILES.get(name)
    if filename is None:
        raise ValueError(f"Unknown prompt name: {name}")
    payload = (_PROMPT_DIR / filename).read_bytes()
    return PromptVersion(
        text=payload.decode("utf-8"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def prohibited_language_matches(value: str) -> list[str]:
    """Return the prohibited wording patterns present in a user-facing string."""
    return [pattern.pattern for pattern in _PROHIBITED_LANGUAGE if pattern.search(value)]


def plain_language_failures(value: str, *, security_names: Sequence[str] = ()) -> list[str]:
    """Check shared banned words and the C28 sentence-length rule."""
    scrubbed = value
    for name in sorted(
        {item.strip() for item in security_names if item.strip()}, key=len, reverse=True
    ):
        scrubbed = re.sub(re.escape(name), " ", scrubbed, flags=re.IGNORECASE)
    failures = [f"banned term: {term}" for term in find_banned_terms(scrubbed)]
    if any(count > 20 for count in sentence_word_counts(scrubbed)):
        failures.append("sentence exceeds 20 words")
    return failures


def normalise_red_team_output(
    output: RedTeamOutput, known_evidence_ids: set[str]
) -> tuple[RedTeamOutput, bool]:
    """Downgrade an uncited VETO to WATCH and retain only known evidence references."""
    reasons = [
        reason.model_copy(
            update={"evidence": [item for item in reason.evidence if item in known_evidence_ids]}
        )
        for reason in output.reasons
    ]
    cited = any(reason.evidence for reason in reasons)
    uncited_veto = output.downgrade == "VETO" and not cited
    downgrade = "WATCH" if uncited_veto else output.downgrade
    return output.model_copy(update={"downgrade": downgrade, "reasons": reasons}), uncited_veto


def validate_cio_selection(
    output: CIOOutput,
    *,
    shortlist_ids: set[str],
    eligible_ids: set[str],
) -> None:
    """Reject any sign-off that names a security outside the code-approved candidate set."""
    selected = set(output.selected_dossier_ids)
    citations = set(output.citations)
    if len(selected) != len(output.selected_dossier_ids):
        raise SchemaError("CIO selection contains duplicate dossier IDs")
    if not selected <= shortlist_ids or not selected <= eligible_ids:
        raise SchemaError("CIO selection contains a dossier outside the eligible shortlist")
    if not citations <= shortlist_ids or not citations <= eligible_ids:
        raise SchemaError("CIO citations contain a dossier outside the eligible shortlist")
    if not selected <= citations:
        raise SchemaError("Every selected dossier needs a CIO summary citation")
    if any(identifier not in output.summary for identifier in citations):
        raise SchemaError("The CIO summary must cite each listed dossier ID")


class ResearchTeam:
    """Run the five agent roles while code owns every status and numeric field."""

    def __init__(
        self,
        router: ModelRouter | FakeRouter,
        tools: ToolRegistry,
        run_ctx: RunContext,
        *,
        rules: ResearchRules | None = None,
    ) -> None:
        self.router = router
        self.tools = tools
        self.run_ctx = run_ctx
        self.rules = rules or ResearchRules.load()
        self._active_tool_results: dict[str, dict[str, dict[str, Any]]] = {}

    def analyse(
        self,
        candidate: Mapping[str, Any],
        dossier: Dossier,
        tool_results: dict[str, dict[str, Any]],
        *,
        feedback: Sequence[str] = (),
    ) -> tuple[Dossier, dict[str, dict[str, Any]]]:
        version = load_prompt_version("analyst")
        self._set_call_context(dossier.dossier_id, "fundamental_analyst", version.sha256)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": f"{EVIDENCE_SYSTEM_PROMPT}\n{version.text}",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "shortlisted_security": self._candidate_prompt(candidate),
                        "prior_claims": [claim.model_dump(mode="json") for claim in dossier.claims],
                        "compliance_feedback": list(feedback),
                        "required_output": AnalystOutput.model_json_schema(),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        for result in self._ordered_tool_results(tool_results):
            messages.append({"role": "user", "content": evidence_message(result)})

        allowed_tools = self._tool_definitions(str(candidate["market"]))
        call_count = 0
        analyst: AnalystOutput | None = None
        while allowed_tools and call_count < 6:
            response = self.router.complete(
                "mini",
                messages,
                schema=None,
                max_output_tokens=900,
                run_ctx=self.run_ctx,
                tools=allowed_tools,
                tool_choice="auto",
            )
            if not response.tool_calls:
                analyst = self._parse_analyst(response.raw_text, response.parsed)
                break
            accepted_calls = list(response.tool_calls[: 6 - call_count])
            messages.append(
                {
                    "role": "assistant",
                    "content": response.raw_text or None,
                    "tool_calls": accepted_calls,
                }
            )
            for requested in accepted_calls:
                call_count += 1
                call_id = str(requested.get("id", ""))
                function = requested.get("function", {})
                name = str(function.get("name", ""))
                try:
                    arguments = json.loads(str(function.get("arguments", "{}")))
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    self._validate_tool_args(name, arguments, candidate, tool_results)
                    result = self.tools.call(name, arguments, self.run_ctx)
                    tool_results[result.tool_call_id] = self._result_record(result)
                    tool_content = evidence_message(result)
                except BudgetExceeded:
                    raise
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    tool_content = json.dumps(
                        {"error": "tool_request_rejected"},
                        ensure_ascii=False,
                    )
                except Exception:
                    tool_content = json.dumps({"error": "tool_unavailable"}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_content})

        if analyst is None:
            final = self.router.complete(
                "mini",
                messages,
                schema=AnalystOutput,
                max_output_tokens=900,
                run_ctx=self.run_ctx,
            )
            analyst = final.parsed

        facts = [
            Claim(
                kind=claim.kind,
                text=claim.text,
                evidence=[Evidence(tool_call_id=identifier) for identifier in claim.evidence],
            )
            for claim in analyst.claims
        ]
        previous_ids = set(dossier.analyst_claim_ids)
        base_claims = [claim for claim in dossier.claims if claim.id not in previous_ids]
        new_ids = [claim.id for claim in facts]
        geo = GeoExposure(
            us_revenue_pct=analyst.geo_exposure.us_revenue_pct,
            tariff_sensitive=analyst.geo_exposure.tariff_sensitive,
            evidence=analyst.geo_exposure.evidence,
        )
        updated = dossier.model_copy(
            update={
                "stage": Stage.ANALYSED,
                "claims": [*base_claims, *facts],
                "analyst_claim_ids": new_ids,
                "drivers": list(analyst.drivers),
                "risks": list(analyst.risks),
                "geo_exposure": geo,
                "verdicts": dossier.verdicts.model_copy(
                    update={"fundamental": analyst.fundamental_verdict}
                ),
                "open_questions": list(analyst.open_questions),
                "compliance_failures": [],
            }
        )
        self._active_tool_results[dossier.dossier_id] = tool_results
        return updated, tool_results

    def value(
        self,
        candidate: Mapping[str, Any],
        dossier: Dossier,
        tool_results: dict[str, dict[str, Any]],
    ) -> tuple[Dossier, dict[str, dict[str, Any]]]:
        result = self.tools.call(
            "valuation_stats",
            {"security_id": str(candidate["security_id"]), "as_of": str(candidate["as_of"])},
            self.run_ctx,
        )
        tool_results[result.tool_call_id] = self._result_record(result)
        self._active_tool_results[dossier.dossier_id] = tool_results
        stats = result.result
        if not isinstance(stats, dict) or str(stats.get("security_id")) != str(
            candidate["security_id"]
        ):
            raise SchemaError("Valuation tool returned data for a different security")
        verdict = str(stats.get("verdict"))
        if verdict not in {"CHEAP", "FAIR", "EXPENSIVE", "UNKNOWN"}:
            raise SchemaError("Valuation tool returned an unknown verdict")

        version = load_prompt_version("valuation")
        self._set_call_context(dossier.dossier_id, "valuation_agent", version.sha256)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{EVIDENCE_SYSTEM_PROMPT}\n{version.text}"},
            {
                "role": "user",
                "content": json.dumps(
                    {"verdict": verdict, "statistics": stats},
                    ensure_ascii=False,
                    default=str,
                ),
            },
            {"role": "user", "content": evidence_message(result)},
        ]
        explanation_text = ""
        # Compliance hands failures back to the Analyst, who doesn't write this text, so
        # the wording is fixed here: one rewrite, then a sentence written by code.
        for _attempt in range(2):
            response = self.router.complete(
                "nano",
                messages,
                schema=ValuationExplanation,
                max_output_tokens=240,
                run_ctx=self.run_ctx,
            )
            candidate_text = response.parsed.explanation
            problems = plain_language_failures(
                candidate_text, security_names=[str(candidate.get("name", ""))]
            )
            if len(sentence_word_counts(candidate_text)) > 2:
                problems.append("more than two sentences")
            if not problems:
                explanation_text = candidate_text
                break
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "Rewrite it: at most two sentences, each 20 words or fewer, "
                    "in plain words. Problems: " + "; ".join(problems),
                },
            ]
        if not explanation_text:
            explanation_text = _VERDICT_SENTENCES[verdict]
        percentile = stats.get("pe_percentile")
        if percentile is None:
            percentile = {"CHEAP": "25", "FAIR": "50", "EXPENSIVE": "75", "UNKNOWN": "50"}[verdict]
        old_valuation_ids = set(dossier.valuation_claim_ids)
        base_claims = [claim for claim in dossier.claims if claim.id not in old_valuation_ids]
        # The explanation describes this one tool result. In the live smoke the model
        # wrote prose where the evidence id belongs, so code attaches the id itself.
        evidence_ids = [str(result.tool_call_id)]
        claim = Claim(
            kind="FACT",
            text=explanation_text,
            evidence=[Evidence(tool_call_id=identifier) for identifier in evidence_ids],
        )
        updated = dossier.model_copy(
            update={
                "stage": Stage.VALUED,
                "claims": [*base_claims, claim],
                "valuation_claim_ids": [claim.id],
                "valuation_stats": stats,
                "valuation_explanation": explanation_text,
                "valuation_evidence_ids": evidence_ids,
                "valuation_percentile": Decimal(str(percentile)),
                "verdicts": dossier.verdicts.model_copy(update={"valuation": verdict}),
            }
        )
        return updated, tool_results

    def red_team(
        self,
        candidate: Mapping[str, Any],
        dossier: Dossier,
        tool_results: dict[str, dict[str, Any]],
    ) -> tuple[Dossier, dict[str, dict[str, Any]]]:
        version = load_prompt_version("red_team")
        self._set_call_context(dossier.dossier_id, "red_team_agent", version.sha256)
        results = self._ordered_tool_results(tool_results)
        response = self.router.complete(
            "redteam",
            [
                {"role": "system", "content": f"{EVIDENCE_SYSTEM_PROMPT}\n{version.text}"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "security": self._candidate_prompt(candidate),
                            "claims": [claim.model_dump(mode="json") for claim in dossier.claims],
                            "fundamental_verdict": dossier.verdicts.fundamental,
                            "valuation_verdict": dossier.verdicts.valuation,
                            "required_output": RedTeamOutput.model_json_schema(),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
                *[{"role": "user", "content": evidence_message(item)} for item in results],
            ],
            schema=RedTeamOutput,
            max_output_tokens=500,
            run_ctx=self.run_ctx,
        )
        output, uncited_veto = normalise_red_team_output(response.parsed, set(tool_results))
        reasons = [
            CitedReason(text=reason.text, evidence=reason.evidence) for reason in output.reasons
        ]
        red_team = RedTeamVerdict(
            strongest_bear_case=output.strongest_bear_case,
            falsifier=output.falsifier,
            downgrade=output.downgrade,
            reasons=reasons,
        )
        flags = list(dossier.red_team_flags)
        if uncited_veto:
            flags.append("uncited_veto_downgraded_to_watch")
        status = "VETOED" if output.downgrade == "VETO" else dossier.status
        updated = dossier.model_copy(
            update={
                "stage": Stage.RED_TEAM_DONE,
                "status": status,
                "status_reason": "red_team_veto" if status == "VETOED" else dossier.status_reason,
                "red_team_flags": flags,
                "verdicts": dossier.verdicts.model_copy(update={"red_team": red_team}),
            }
        )
        self._active_tool_results[dossier.dossier_id] = tool_results
        return updated, tool_results

    def compliance_answer(self, dossier: Dossier) -> ComplianceOutput:
        version = load_prompt_version("compliance")
        self._set_call_context(dossier.dossier_id, "compliance_agent", version.sha256)
        user_text = "\n".join(
            [
                *[claim.text for claim in dossier.claims],
                *dossier.drivers,
                *dossier.risks,
                dossier.valuation_explanation or "",
                dossier.verdicts.red_team.strongest_bear_case
                if dossier.verdicts.red_team is not None
                else "",
            ]
        )
        response = self.router.complete(
            "mini",
            [
                {"role": "system", "content": f"{EVIDENCE_SYSTEM_PROMPT}\n{version.text}"},
                {"role": "user", "content": user_text},
            ],
            schema=ComplianceOutput,
            max_output_tokens=64,
            run_ctx=self.run_ctx,
        )
        return response.parsed

    def cio_signoff(
        self,
        dossiers: Sequence[Dossier],
        candidates_by_id: Mapping[str, Mapping[str, Any]],
        shortlist_ids: set[str],
    ) -> tuple[CIOOutput, list[str]]:
        per_market: dict[str, list[Dossier]] = {"IN": [], "US": []}
        for dossier in dossiers:
            if self._cio_eligible(dossier):
                per_market[dossier.security.market].append(dossier)
        eligible: list[Dossier] = []
        for items in per_market.values():
            items.sort(
                key=lambda dossier: (
                    -Decimal(str(candidates_by_id[dossier.dossier_id]["composite"])),
                    dossier.security.id,
                )
            )
            eligible.extend(items[: self.rules.max_ideas_per_market])
        eligible_ids = {dossier.dossier_id for dossier in eligible}
        if not eligible:
            return CIOOutput(summary="No ideas passed the checks this week."), []
        # Code decides: the top eligible ideas per market are approved. The model only
        # writes the summary, and a summary that fails its checks is replaced, not fatal.
        approved_ids = [dossier.dossier_id for dossier in eligible]
        try:
            output = self._cio_summary(eligible, candidates_by_id, shortlist_ids, eligible_ids)
        except (SchemaError, ValueError):
            count = len(approved_ids)
            output = CIOOutput(
                summary=(
                    "One idea passed every check this week."
                    if count == 1
                    else f"{count} ideas passed every check this week."
                )
                + " Open an idea to see why."
            )
        return output, approved_ids

    def _cio_summary(
        self,
        eligible: Sequence[Dossier],
        candidates_by_id: Mapping[str, Mapping[str, Any]],
        shortlist_ids: set[str],
        eligible_ids: set[str],
    ) -> CIOOutput:

        version = load_prompt_version("cio")
        self.run_ctx.dossier_budget = None
        self._set_call_context(None, "cio_agent", version.sha256)
        prompt_candidates = [
            {
                "dossier_id": dossier.dossier_id,
                "market": dossier.security.market,
                "symbol": candidates_by_id[dossier.dossier_id]["symbol"],
                "fundamental_verdict": dossier.verdicts.fundamental,
                "valuation_verdict": dossier.verdicts.valuation,
                "red_team_downgrade": (
                    dossier.verdicts.red_team.downgrade
                    if dossier.verdicts.red_team is not None
                    else "UNKNOWN"
                ),
                "claims": [claim.text for claim in dossier.claims[:4]],
            }
            for dossier in eligible
        ]
        response = self.router.complete(
            "strong",
            [
                {"role": "system", "content": f"{EVIDENCE_SYSTEM_PROMPT}\n{version.text}"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"eligible_dossiers": prompt_candidates},
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            schema=CIOOutput,
            max_output_tokens=480,
            run_ctx=self.run_ctx,
        )
        output: CIOOutput = response.parsed
        validate_cio_selection(
            output,
            shortlist_ids=shortlist_ids,
            eligible_ids=eligible_ids,
        )
        if len(output.summary.split()) > 120:
            raise SchemaError("CIO summary exceeds 120 words")
        names = [str(candidates_by_id[item.dossier_id].get("name", "")) for item in eligible]
        if plain_language_failures(output.summary, security_names=names):
            raise SchemaError("CIO summary failed the plain-language check")
        if prohibited_language_matches(output.summary):
            raise SchemaError("CIO summary contains prohibited language")
        cited_dossiers = [
            dossier for dossier in eligible if dossier.dossier_id in set(output.citations)
        ]
        self._validate_cio_summary_claim_words(
            output,
            cited_dossiers,
            candidates_by_id,
        )
        if not output.citations and re.search(r"\d", output.summary):
            raise SchemaError("CIO summary has a number without a dossier citation")
        summary_for_grounding = output.summary
        for identifier in output.citations:
            summary_for_grounding = summary_for_grounding.replace(identifier, " ")
        if cited_dossiers:
            summary_claim = Claim(
                kind="FACT",
                text=summary_for_grounding,
                evidence=[
                    Evidence(tool_call_id=evidence_id)
                    for dossier in cited_dossiers
                    for claim in dossier.claims
                    for evidence_id in [item.tool_call_id for item in claim.evidence]
                ],
            )
            union_tools = {
                evidence_id: result
                for dossier in cited_dossiers
                for evidence_id, result in self._active_tool_results.get(
                    dossier.dossier_id, {}
                ).items()
            }
            grounding_dossier = Dossier(
                dossier_id=eligible[0].dossier_id,
                security=eligible[0].security,
                stage=eligible[0].stage,
                claims=[summary_claim],
            )
            if validate_grounding(grounding_dossier, union_tools):
                raise SchemaError("CIO summary contains unsupported numbers")
        return output

    @staticmethod
    def _validate_cio_summary_claim_words(
        output: CIOOutput,
        cited_dossiers: Sequence[Dossier],
        candidates_by_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        allowed_words = set(_CIO_GENERAL_WORDS)
        for dossier in cited_dossiers:
            for claim in dossier.claims:
                allowed_words.update(re.findall(r"[a-z]+", claim.text.casefold()))
            symbol = str(candidates_by_id[dossier.dossier_id].get("symbol", ""))
            allowed_words.update(re.findall(r"[a-z]+", symbol.casefold()))
        summary = output.summary
        for dossier_id in output.citations:
            summary = summary.replace(dossier_id, " ")
        unknown = sorted(set(re.findall(r"[a-z]+", summary.casefold())) - allowed_words)
        if unknown:
            raise SchemaError("CIO summary contains words unsupported by its cited dossiers")

    def _set_call_context(self, dossier_id: str | None, agent: str, prompt_sha256: str) -> None:
        self.run_ctx.dossier_id = dossier_id
        self.run_ctx.agent = agent
        self.run_ctx.prompt_version_sha256 = prompt_sha256

    @staticmethod
    def _candidate_prompt(candidate: Mapping[str, Any]) -> dict[str, Any]:
        allowed = ("dossier_id", "security_id", "symbol", "market", "sector", "as_of")
        return {key: candidate[key] for key in allowed if key in candidate}

    def _tool_definitions(self, market: str) -> list[dict[str, Any]]:
        names = [
            name
            for name in _MARKET_TOOLS.get(market.upper(), ())
            if name in self.tools.registered_names
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _TOOL_DESCRIPTIONS[name],
                    "parameters": {
                        "type": "object",
                        "properties": _TOOL_PROPERTIES[name],
                        "required": list(_TOOL_PROPERTIES[name]),
                        "additionalProperties": False,
                    },
                },
            }
            for name in names
        ]

    @staticmethod
    def _result_record(result: ToolResult) -> dict[str, Any]:
        return {
            "tool_call_id": result.tool_call_id,
            "result": result.result,
            "source_url": result.source_url,
            "retrieved_at": result.retrieved_at.isoformat(),
        }

    @staticmethod
    def _ordered_tool_results(results: Mapping[str, Mapping[str, Any]]) -> list[ToolResult]:
        output: list[ToolResult] = []
        for key, row in results.items():
            retrieved_at = row.get("retrieved_at")
            if isinstance(retrieved_at, str):
                retrieved_at = datetime.fromisoformat(retrieved_at)
            if not isinstance(retrieved_at, datetime):
                retrieved_at = datetime.now(UTC)
            output.append(
                ToolResult(
                    tool_call_id=str(row.get("tool_call_id", key)),
                    result=row.get("result"),
                    source_url=row.get("source_url"),
                    retrieved_at=retrieved_at,
                )
            )
        return output

    @staticmethod
    def _parse_analyst(raw_text: str, parsed: Any) -> AnalystOutput:
        if isinstance(parsed, AnalystOutput):
            return parsed
        if not raw_text:
            raise SchemaError("Analyst returned no structured response")
        return AnalystOutput.model_validate_json(raw_text)

    @staticmethod
    def _validate_tool_args(
        name: str,
        args: dict[str, Any],
        candidate: Mapping[str, Any],
        tool_results: Mapping[str, Mapping[str, Any]],
    ) -> None:
        security_bound_tools = {
            "factor_snapshot",
            "fundamentals_series",
            "price_stats",
            "sector_peers",
        }
        if name in security_bound_tools and str(args.get("security_id")) != str(
            candidate["security_id"]
        ):
            raise ValueError("Tool requests must stay on the shortlisted security")
        if name == "fundamentals_series":
            metrics = args.get("metrics")
            if not isinstance(metrics, list) or not set(metrics) <= _ANALYST_METRICS:
                raise ValueError("Fundamental metrics are outside the allowlist")
        if (
            name == "nse_announcements"
            and str(args.get("symbol", "")).upper() != str(candidate["symbol"]).upper()
        ):
            raise ValueError("Exchange requests must stay on the shortlisted security")
        if name == "edgar_company_tickers":
            tickers = args.get("tickers")
            if not isinstance(tickers, list) or any(
                str(ticker).upper() != str(candidate["symbol"]).upper() for ticker in tickers
            ):
                raise ValueError("Filing lookup must stay on the shortlisted ticker")
        if name == "edgar_filings":
            known_ciks = {
                str(row.get("cik"))
                for record in tool_results.values()
                for row in (record.get("result") if isinstance(record.get("result"), list) else [])
                if isinstance(row, dict) and row.get("cik")
            }
            if str(args.get("cik")) not in known_ciks:
                raise ValueError("Filing lookup must use a resolved shortlisted ticker")
        if name == "edgar_filing_text":
            known_accessions = {
                str(row.get("accession"))
                for record in tool_results.values()
                for row in (record.get("result") if isinstance(record.get("result"), list) else [])
                if isinstance(row, dict) and row.get("accession")
            }
            if str(args.get("accession")) not in known_accessions:
                raise ValueError("Filing text must come from a logged shortlist filing")

    @staticmethod
    def _cio_eligible(dossier: Dossier) -> bool:
        return (
            dossier.status == "PENDING"
            and dossier.stage == Stage.COMPLIANCE_DONE
            and dossier.verdicts.fundamental in {"POSITIVE", "NEUTRAL"}
            and dossier.verdicts.valuation in {"CHEAP", "FAIR"}
            and dossier.verdicts.red_team is not None
            and dossier.verdicts.red_team.downgrade != "VETO"
        )


def dossier_compliance_failures(
    dossier: Dossier,
    tool_results: Mapping[str, Mapping[str, Any]],
    *,
    security_names: Sequence[str] = (),
) -> list[str]:
    """Run deterministic grounding, required-field, language, and advice checks."""
    failures: list[str] = []
    failures.extend(validate_grounding(dossier, tool_results))
    analyst_claims = [
        claim for claim in dossier.claims if claim.id in set(dossier.analyst_claim_ids)
    ]
    if not any(claim.kind == "FACT" for claim in analyst_claims):
        failures.append("a cited analyst fact is required")
    if dossier.verdicts.fundamental not in {"POSITIVE", "NEUTRAL", "NEGATIVE"}:
        failures.append("fundamental verdict is required")
    if dossier.verdicts.valuation not in {"CHEAP", "FAIR", "EXPENSIVE", "UNKNOWN"}:
        failures.append("valuation verdict is required")
    if dossier.verdicts.red_team is None or not dossier.verdicts.red_team.falsifier:
        failures.append("red-team falsifier is required")
    if (
        dossier.geo_exposure is not None
        and (
            dossier.geo_exposure.us_revenue_pct is not None
            or dossier.geo_exposure.tariff_sensitive is not None
        )
        and (
            not dossier.geo_exposure.evidence
            or not set(dossier.geo_exposure.evidence) <= set(tool_results)
        )
    ):
        failures.append("geographic exposure needs valid evidence")
    red_team_text: list[str] = []
    if dossier.verdicts.red_team is not None:
        red_team_text = [
            dossier.verdicts.red_team.strongest_bear_case,
            dossier.verdicts.red_team.falsifier,
            *[reason.text for reason in dossier.verdicts.red_team.reasons],
        ]
    user_text = [
        *(claim.text for claim in dossier.claims),
        *dossier.drivers,
        *dossier.risks,
        dossier.valuation_explanation or "",
        *red_team_text,
    ]
    for value in user_text:
        if prohibited_language_matches(value):
            failures.append("prohibited investment promise")
    if dossier.valuation_explanation:
        failures.extend(
            f"valuation explanation: {issue}"
            for issue in plain_language_failures(
                dossier.valuation_explanation,
                security_names=security_names,
            )
        )
        if len(sentence_word_counts(dossier.valuation_explanation)) > 2:
            failures.append("valuation explanation exceeds two sentences")
    return list(dict.fromkeys(failures))
