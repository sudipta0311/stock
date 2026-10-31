"""Official rate and rulebook watchers with exact-quote checks and owner review."""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from wealth.db.models import FeedItem, RuleChangeProposal
from wealth.db.session import make_engine
from wealth.research.errors import BudgetExceeded
from wealth.research.router import FakeRouter, ModelRouter, RunBudget, RunContext
from wealth.research.store import create_run, update_run

INDIA_POST_RATES_URL = "https://www.indiapost.gov.in/banking-services/savings"
DEA_SMALL_SAVINGS_URL = "https://dea.gov.in/budget-division/475"
EPFO_INTEREST_URL = "https://www.epfindia.gov.in/site_en/For_Employees.php?id=sm1_index"
WATCHER_USER_AGENT = "Wealth Copilot scheduled rates watcher/1.0"
_RULES_DIR = Path(__file__).resolve().parents[3] / "rules"
_SCHEME_ALIASES: dict[str, tuple[str, ...]] = {
    "nsc": ("national savings certificate", "nsc"),
    "kvp": ("kisan vikas patra", "kvp"),
    "td_1y": ("1 year time deposit", "1-year time deposit", "one year time deposit"),
    "td_2y": ("2 year time deposit", "2-year time deposit", "two year time deposit"),
    "td_3y": ("3 year time deposit", "3-year time deposit", "three year time deposit"),
    "td_5y": ("5 year time deposit", "5-year time deposit", "five year time deposit"),
    "rd_5y": ("recurring deposit", "5 year recurring deposit"),
    "ppf": ("public provident fund", "ppf"),
    "scss": ("senior citizens savings scheme", "scss"),
    "mis": ("monthly income scheme", "mis"),
    "ssy": ("sukanya samriddhi", "ssa", "ssy"),
    "epf": ("employees provident fund", "employees' provident fund", "epf"),
}


class QuoteList(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quotes: list[str] = Field(default_factory=list, max_length=40)


class FeedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: Literal["tax_rate", "lrs_tcs", "small_savings", "epf", "irrelevant"]
    quote: str = Field(max_length=1000)


class FeedFindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: list[FeedFinding] = Field(default_factory=list, max_length=40)


@dataclass(frozen=True)
class RateDocument:
    kind: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class RateQuote:
    scheme: str
    rate_pct: Decimal
    period_from: date
    period_to: date
    quote: str
    source_url: str


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._link_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "div", "tr", "li", "h1", "h2", "h3", "br", "td", "th"}:
            self.text_parts.append("\n")
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._link_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((" ".join(self._link_parts).strip(), self._href))
            self._href = None
            self._link_parts = []
        if tag in {"p", "div", "tr", "li", "h1", "h2", "h3", "td", "th"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if clean:
            self.text_parts.append(clean + " ")
            if self._href is not None:
                self._link_parts.append(clean)


def _page(content: bytes) -> tuple[str, list[tuple[str, str]]]:
    parser = _PageParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    return " ".join(" ".join(parser.text_parts).split()), parser.links


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)[:80_000]


def _get(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url)
    response.raise_for_status()
    if len(response.content) > 4_000_000:
        raise ValueError("Official rate document exceeded the size limit")
    return response


def _latest_pdf_link(
    links: Sequence[tuple[str, str]], terms: tuple[str, ...], base_url: str
) -> str | None:
    for title, href in links:
        joined = urljoin(base_url, href)
        host = (urlparse(joined).hostname or "").casefold()
        if not joined.casefold().split("?")[0].endswith(".pdf"):
            continue
        if not any(term in title.casefold() for term in terms):
            continue
        if host not in {
            "dea.gov.in",
            "www.dea.gov.in",
            "epfindia.gov.in",
            "www.epfindia.gov.in",
            "epfo.gov.in",
            "www.epfo.gov.in",
        }:
            continue
        return joined
    return None


def fetch_official_rate_documents(
    client: httpx.Client | None = None,
    *,
    pause: Any = time.sleep,
) -> list[RateDocument]:
    """Fetch a small set of official pages sequentially with a descriptive User-Agent.

    Each page is optional: India Post often times out, and one unreachable site must not
    hide the others from the rate check.
    """
    owned_client = client is None
    active_client = client or httpx.Client(
        timeout=httpx.Timeout(20.0, connect=8.0),
        follow_redirects=True,
        headers={"User-Agent": WATCHER_USER_AGENT, "Accept": "text/html,application/pdf"},
    )
    documents: list[RateDocument] = []

    def fetch(url: str) -> httpx.Response | None:
        try:
            return _get(active_client, url)
        except (httpx.HTTPError, ValueError):
            return None

    try:
        india_post = fetch(INDIA_POST_RATES_URL)
        if india_post is not None:
            india_text, _india_links = _page(india_post.content)
            documents.append(
                RateDocument(
                    "india_post",
                    "Post office savings rates",
                    INDIA_POST_RATES_URL,
                    india_text[:30_000],
                )
            )
        pause(1.0)

        dea_page = fetch(DEA_SMALL_SAVINGS_URL)
        if dea_page is not None:
            dea_text, dea_links = _page(dea_page.content)
            documents.append(
                RateDocument(
                    "dea_page", "Small-savings notices", DEA_SMALL_SAVINGS_URL, dea_text[:15_000]
                )
            )
            dea_pdf = _latest_pdf_link(
                dea_links, ("small savings", "small saving"), DEA_SMALL_SAVINGS_URL
            )
            if dea_pdf:
                pause(1.0)
                response = fetch(dea_pdf)
                if response is not None:
                    documents.append(
                        RateDocument(
                            "dea_notice",
                            "Latest small-savings notice",
                            dea_pdf,
                            _pdf_text(response.content),
                        )
                    )
        pause(1.0)

        epfo_page = fetch(EPFO_INTEREST_URL)
        if epfo_page is not None:
            epfo_text, epfo_links = _page(epfo_page.content)
            documents.append(
                RateDocument(
                    "epfo_page",
                    "Provident-fund interest notices",
                    EPFO_INTEREST_URL,
                    epfo_text[:15_000],
                )
            )
            epfo_pdf = _latest_pdf_link(
                epfo_links,
                ("declaration", "interest rate", "interest of", "roi"),
                EPFO_INTEREST_URL,
            )
            if epfo_pdf:
                pause(1.0)
                response = fetch(epfo_pdf)
                if response is not None:
                    documents.append(
                        RateDocument(
                            "epfo_notice",
                            "Provident-fund rate notice",
                            epfo_pdf,
                            _pdf_text(response.content),
                        )
                    )
        return documents
    finally:
        if owned_client:
            active_client.close()


def _quarter_period(value: str) -> tuple[date, date] | None:
    match = re.search(
        r"\bQ([1-4])\s*(?:of\s*)?FY\s*(20\d{2})\s*[-\u2010-\u2015\u2212/]\s*(\d{2}|20\d{2})",
        value,
        re.I,
    )
    if match is None:
        return None
    quarter = int(match.group(1))
    year = int(match.group(2))
    start_month = (quarter - 1) * 3 + 4
    if start_month > 12:
        start_month -= 12
        year += 1
    start = date(year, start_month, 1)
    end_month = start_month + 2
    end_year = year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    end_day = (
        date(end_year + (end_month == 12), 1 if end_month == 12 else end_month + 1, 1)
        - date.resolution
    ).day
    return start, date(end_year, end_month, end_day)


def _fy_period(value: str) -> tuple[date, date] | None:
    match = re.search(r"\bFY\s*(20\d{2})\s*[-\u2010-\u2015\u2212/]\s*(\d{2}|20\d{2})", value, re.I)
    if match is None:
        return None
    start_year = int(match.group(1))
    end_year_text = match.group(2)
    end_year = (
        int(end_year_text)
        if len(end_year_text) == 4
        else start_year // 100 * 100 + int(end_year_text)
    )
    if end_year < start_year:
        end_year += 100
    return date(start_year, 4, 1), date(end_year, 3, 31)


def _period_from_text(value: str, *, epf: bool = False) -> tuple[date, date] | None:
    if epf:
        fiscal = _fy_period(value)
        if fiscal is not None:
            return fiscal
    quarter = _quarter_period(value)
    if quarter is not None:
        return quarter
    iso = re.search(
        r"(20\d{2}-\d{2}-\d{2})\s*(?:to|through|[-\u2010-\u2015\u2212])\s*(20\d{2}-\d{2}-\d{2})",
        value,
        re.I,
    )
    if iso:
        try:
            start, end = date.fromisoformat(iso.group(1)), date.fromisoformat(iso.group(2))
        except ValueError:
            return None
        return (start, end) if start <= end else None
    return None


def _scheme_for_quote(value: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    matches: list[str] = []
    for scheme, aliases in _SCHEME_ALIASES.items():
        for alias in aliases:
            normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
            if re.search(rf"\b{re.escape(normalized_alias)}\b", normalized):
                matches.append(scheme)
                break
    return matches[0] if len(matches) == 1 else None


def parse_rate_quotes(documents: Sequence[RateDocument], quotes: Iterable[str]) -> list[RateQuote]:
    """Accept only exact source substrings, then parse all numbers and dates in code."""
    period_by_kind: dict[str, tuple[date, date]] = {}
    for document in documents:
        if document.kind in {"dea_notice", "dea_page"}:
            period = _period_from_text(document.text)
            if period:
                period_by_kind["small_savings"] = period
        if document.kind.startswith("epfo"):
            period = _period_from_text(document.text, epf=True)
            if period:
                period_by_kind["epf"] = period
    results: list[RateQuote] = []
    seen: set[tuple[str, Decimal, date, date]] = set()
    for raw_quote in quotes:
        quote = raw_quote.strip()
        if not quote:
            continue
        document = next((item for item in documents if quote in item.text), None)
        if document is None:
            continue
        scheme = _scheme_for_quote(quote)
        rate_matches = re.findall(r"(?<![\w.])(\d{1,2}(?:\.\d{1,4})?)\s*%", quote)
        if scheme is None or len(rate_matches) != 1:
            continue
        try:
            rate = Decimal(rate_matches[0])
        except InvalidOperation:
            continue
        if not rate.is_finite() or not Decimal(0) <= rate <= Decimal(30):
            continue
        epf = scheme == "epf"
        period = _period_from_text(quote, epf=epf) or period_by_kind.get(
            "epf" if epf else "small_savings"
        )
        if period is None:
            continue
        result = RateQuote(scheme, rate, period[0], period[1], quote, document.url)
        unique = (scheme, rate, period[0], period[1])
        if unique not in seen:
            seen.add(unique)
            results.append(result)
    return results


def _rule_file(rules_dir: Path, path: str) -> tuple[Path, str]:
    file_part, separator, yaml_path = path.partition("#")
    if not separator:
        raise ValueError("Proposal path must include a YAML pointer")
    relative = Path(file_part)
    if relative.parts and relative.parts[0] == "rules":
        relative = Path(*relative.parts[1:])
    target = (rules_dir / relative).resolve()
    root = rules_dir.resolve()
    if root not in target.parents or not target.is_file():
        raise ValueError("Proposal path is outside the rules directory or missing")
    return target, yaml_path.strip("/")


def _current_rate(rules_dir: Path, scheme: str, period_from: date, period_to: date) -> Any:
    target, pointer = _rule_file(rules_dir, "rules/schemes/small_savings.yaml#schemes")
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    if scheme == "epf":
        items = document.get("epf", {}).get("periods", [])
        fy = f"{period_from.year}-{str(period_to.year)[-2:]}"
        return next((item for item in items if item.get("fy") == fy), None)
    items = document.get("schemes", {}).get(scheme, {}).get("periods", [])
    start_text, end_text = period_from.isoformat(), period_to.isoformat()
    return next(
        (item for item in items if item.get("from") == start_text and item.get("to") == end_text),
        None,
    )


def build_rate_proposals(
    documents: Sequence[RateDocument], quotes: Iterable[str], *, rules_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Build pending proposals from exact quotes and compare with the active YAML period."""
    base = rules_dir or _RULES_DIR
    proposals: list[dict[str, Any]] = []
    for item in parse_rate_quotes(documents, quotes):
        period = {"from": item.period_from.isoformat(), "to": item.period_to.isoformat()}
        if item.scheme == "epf":
            fy = f"{item.period_from.year}-{str(item.period_to.year)[-2:]}"
            path = "rules/schemes/small_savings.yaml#/epf/periods"
            proposed: dict[str, Any] = {
                "fy": fy,
                "rate_pct": format(item.rate_pct, "f"),
                "source_url": item.source_url,
            }
            kind = "epf_rate"
        else:
            path = f"rules/schemes/small_savings.yaml#/schemes/{item.scheme}/periods"
            proposed = {
                **period,
                "rate_pct": format(item.rate_pct, "f"),
                "source_url": item.source_url,
            }
            kind = "small_savings_rate"
        current = _current_rate(base, item.scheme, item.period_from, item.period_to)
        if current is not None and str(current.get("rate_pct")) == proposed["rate_pct"]:
            continue
        proposals.append(
            {
                "kind": kind,
                "path": path,
                "current_value": current,
                "proposed_value": proposed,
                "evidence": {"quote": item.quote, "source_url": item.source_url},
            }
        )
    return proposals


def _exact_rate_quotes(
    router: ModelRouter | FakeRouter, documents: Sequence[RateDocument], run_ctx: RunContext
) -> list[str]:
    text = "\n\n".join(
        f"SOURCE {index + 1}: {item.title}\n{item.text}" for index, item in enumerate(documents)
    )
    if not text.strip():
        return []
    try:
        response = router.complete(
            "nano",
            [
                {
                    "role": "system",
                    "content": (
                        "Return only exact sentences that state a current small-savings or "
                        "provident-fund interest rate. Copy each sentence exactly. Do not "
                        "calculate rates or add dates."
                    ),
                },
                {"role": "user", "content": text[:48_000]},
            ],
            schema=QuoteList,
            max_output_tokens=1000,
            run_ctx=run_ctx,
        )
        result = (
            response.parsed
            if isinstance(response.parsed, QuoteList)
            else QuoteList.model_validate(response.parsed)
        )
    except (BudgetExceeded, ValueError, TypeError):
        return []
    return [quote for quote in result.quotes if any(quote in doc.text for doc in documents)]


def _insert_proposals(engine: Engine, proposals: Sequence[dict[str, Any]]) -> int:
    inserted = 0
    with Session(engine) as session, session.begin():
        for item in proposals:
            existing = session.scalar(
                select(RuleChangeProposal.id).where(
                    RuleChangeProposal.kind == item["kind"],
                    RuleChangeProposal.path == item["path"],
                    RuleChangeProposal.proposed_value == item["proposed_value"],
                    RuleChangeProposal.status == "pending",
                )
            )
            if existing is not None:
                continue
            session.add(RuleChangeProposal(**item))
            inserted += 1
    return inserted


def run_rates_watcher(
    engine: Engine,
    *,
    router: ModelRouter | FakeRouter,
    run_ctx: RunContext,
    documents: Sequence[RateDocument] | None = None,
    rules_dir: Path | None = None,
) -> dict[str, Any]:
    sources = list(documents) if documents is not None else fetch_official_rate_documents()
    quotes = _exact_rate_quotes(router, sources, run_ctx)
    proposals = build_rate_proposals(sources, quotes, rules_dir=rules_dir)
    return {
        "documents": len(sources),
        "quotes": len(quotes),
        "proposals": _insert_proposals(engine, proposals),
    }


def _rulebook_target(topic: str) -> str | None:
    return {
        "tax_rate": "rules/tax/india.yaml#/watcher_proposals",
        "lrs_tcs": "rules/routes.yaml#/watcher_proposals",
        "small_savings": "rules/schemes/small_savings.yaml#/watcher_proposals",
        "epf": "rules/schemes/small_savings.yaml#/watcher_proposals",
    }.get(topic)


def _quoted_numbers(quote: str) -> list[str]:
    return re.findall(
        r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)\s*(?:%|per cent|percent)", quote, re.I
    )


def build_rulebook_proposals(
    items: Sequence[FeedItem], findings: Sequence[FeedFinding]
) -> list[dict[str, Any]]:
    """Reject quotes absent from C16a feed text; code extracts any numeric values."""
    by_quote: dict[str, FeedItem] = {}
    for item in items:
        source_text = f"{item.title}\n{item.summary}"
        for finding in findings:
            if (
                finding.topic == "irrelevant"
                or not finding.quote
                or finding.quote not in source_text
            ):
                continue
            by_quote.setdefault(finding.quote, item)
    proposals: list[dict[str, Any]] = []
    for quote, item in by_quote.items():
        finding = next((candidate for candidate in findings if candidate.quote == quote), None)
        if finding is None or finding.topic == "irrelevant":
            continue
        path = _rulebook_target(finding.topic)
        if path is None:
            continue
        values = _quoted_numbers(quote)
        proposed: dict[str, Any] | None = None
        if len(values) == 1:
            proposed = {"quoted_rate_pct": values[0], "source_url": item.url}
        proposals.append(
            {
                "kind": "rulebook",
                "path": path,
                "current_value": None,
                "proposed_value": proposed,
                "evidence": {
                    "quote": quote,
                    "source": item.source,
                    "source_url": item.url,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                },
            }
        )
    return proposals


def run_rulebook_watcher(
    engine: Engine,
    *,
    router: ModelRouter | FakeRouter,
    run_ctx: RunContext,
    items: Sequence[FeedItem] | None = None,
) -> dict[str, Any]:
    if items is None:
        with Session(engine) as session:
            items = list(
                session.scalars(
                    select(FeedItem)
                    .order_by(
                        FeedItem.published_at.desc().nullslast(), FeedItem.source, FeedItem.guid
                    )
                    .limit(40)
                )
            )
    if not items:
        return {"feed_items": 0, "proposals": 0}
    payload = "\n\n".join(f"{item.title}\n{item.summary}" for item in items)
    try:
        response = router.complete(
            "mini",
            [
                {
                    "role": "system",
                    "content": (
                        "Classify only material about tax rates, LRS/TCS, small savings, or "
                        "EPF. For each relevant item, copy one exact sentence from the "
                        "supplied text. Do not invent values or citations."
                    ),
                },
                {"role": "user", "content": payload[:48_000]},
            ],
            schema=FeedFindings,
            max_output_tokens=1400,
            run_ctx=run_ctx,
        )
        result = (
            response.parsed
            if isinstance(response.parsed, FeedFindings)
            else FeedFindings.model_validate(response.parsed)
        )
    except (BudgetExceeded, ValueError, TypeError):
        return {"feed_items": len(items), "proposals": 0, "status": "model_unavailable"}
    proposals = build_rulebook_proposals(items, result.findings)
    return {"feed_items": len(items), "proposals": _insert_proposals(engine, proposals)}


def _run_all(args: argparse.Namespace) -> int:
    engine: Engine | None = None
    run_id = uuid4()
    week = (
        args.week or f"{date.today().isocalendar().year:04d}-W{date.today().isocalendar().week:02d}"
    )
    try:
        database_url = args.database_url
        if not database_url:
            import os

            database_url = os.getenv("OWNER_DATABASE_URL")
        if not database_url:
            raise RuntimeError("Owner database is unavailable")
        engine = make_engine(database_url)
        create_run(engine, run_id=run_id, week=week, mode="watchers", budget_usd=Decimal("0.25"))
        ctx = RunContext(
            budget=RunBudget(max_usd=Decimal("0.25"), max_calls=20),
            run_id=run_id,
            mode="weekly",
            owner_engine=engine,
        )
        router = ModelRouter(owner_engine=engine)
        rates = run_rates_watcher(engine, router=router, run_ctx=ctx)
        rulebook = run_rulebook_watcher(engine, router=router, run_ctx=ctx)
        update_run(
            engine,
            run_id,
            stage="WATCHERS_DONE",
            status="completed",
            spent_usd=ctx.budget.spent_usd,
            finished=True,
        )
        print(
            json.dumps(
                {"status": "completed", "rates": rates, "rulebook": rulebook}, sort_keys=True
            )
        )
        return 0
    except Exception as error:
        if engine is not None:
            update_run(engine, run_id, status="failed", finished=True)
        print(json.dumps({"status": "failed", "error": type(error).__name__}, sort_keys=True))
        return 3
    finally:
        if engine is not None:
            engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wealth.research.watchers")
    parser.add_argument("--week")
    parser.add_argument("--database-url", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return _run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
