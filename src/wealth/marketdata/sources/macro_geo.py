"""Official macro data and news-feed adapters for the macro research agent."""

from __future__ import annotations

import csv
import html
import io
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin
from xml.etree import ElementTree

import httpx

from wealth.marketdata.sources.http import get

MACRO_USER_AGENT = "WealthCopilotMacroData/1.0"
NSE_INDEX_ARCHIVE_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{date}.csv"
NSE_INDEX_REPORT_URL = "https://www.nseindia.com/all-reports"
TREASURY_XML_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
)
TREASURY_SOURCE = "US Treasury"
FEDERAL_REGISTER_API_URL = "https://www.federalregister.gov/api/v1/documents.json"
PIB_FEED_URL = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=2"
RBI_FEED_URL = "https://rbi.org.in/pressreleases_rss.xml"
USTR_FEED_URL = "https://ustr.gov/archive/Meta_Content/RSS/ustr_press_releases_10475.xml"
FEED_URLS = {
    "pib": PIB_FEED_URL,
    "rbi": RBI_FEED_URL,
    "ustr": USTR_FEED_URL,
}
FEED_ITEM_LIMIT = 20
FEDERAL_REGISTER_TERM_LIMIT = 25
# 60 days covers the weekly macro view; 120 would take up to ~90 MB of Neon's free 0.5 GB.
FEED_RETENTION_DAYS = 60
MACRO_SERIES_RETENTION_DAYS = 400

_INDEX_NAMES = {
    "nifty50": "NIFTY 50",
    "niftynext50": "NIFTY NEXT 50",
    "niftymidcap150": "NIFTY MIDCAP 150",
    "nifty500": "NIFTY 500",
}
_TREASURY_SERIES = {"BC_10YEAR": "UST_10Y", "BC_3MONTH": "UST_3M"}
_XML_ACCEPT = "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"


@dataclass(frozen=True)
class IndexValuationObservation:
    index_name: str
    as_of: date
    pe: Decimal | None
    pb: Decimal | None
    div_yield: Decimal | None
    source: str
    source_url: str
    close: Decimal | None = None


@dataclass(frozen=True)
class MacroSeriesObservation:
    series: str
    as_of: date
    value: Decimal
    source: str
    source_url: str


@dataclass(frozen=True)
class FeedItemObservation:
    source: str
    guid: str
    title: str
    url: str
    published_at: datetime | None
    summary: str
    fetched_at: datetime


@dataclass(frozen=True)
class MacroSeriesBatch:
    observations: list[MacroSeriesObservation]
    errors: dict[str, str]


@dataclass(frozen=True)
class FeedBatch:
    items: list[FeedItemObservation]
    errors: dict[str, str]


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _request(
    url: str,
    *,
    client: httpx.Client | None = None,
    params: Mapping[str, Any] | None = None,
) -> httpx.Response:
    headers = {"User-Agent": MACRO_USER_AGENT, "Accept": _XML_ACCEPT}
    if client is not None:
        response = client.get(url, params=params, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response
    request_url = str(httpx.URL(url, params=params)) if params else url
    return get(request_url, headers=headers, timeout=30.0)


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _parse_date(value: str) -> date:
    cleaned = value.strip()
    for pattern in (
        "%d-%m-%Y",
        "%d-%b-%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(cleaned[:19], pattern).date()
        except ValueError:
            continue
    if len(cleaned) >= 10:
        try:
            return date.fromisoformat(cleaned[:10])
        except ValueError:
            pass
    try:
        parsed = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("unrecognized publication date") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).date()


def _parse_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned in {"-", "—", "NA", "N/A"}:
        return None
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError("invalid numeric value in official data") from error
    if not parsed.is_finite():
        raise ValueError("non-finite numeric value in official data")
    return parsed


def parse_nse_index_valuations(
    content: str | bytes,
    *,
    source_url: str,
    as_of_max: date | None = None,
) -> list[IndexValuationObservation]:
    """Parse NSE's daily all-indices CSV and keep the four supported equity indices."""
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("NSE index file has no header")
    field_names = {_normalise_header(name): name for name in reader.fieldnames if name}
    required = {"indexname", "indexdate", "pe", "pb", "divyield"}
    if not required <= set(field_names):
        raise ValueError("NSE index file is missing valuation columns")

    observations: dict[tuple[str, date], IndexValuationObservation] = {}
    for row in reader:
        raw_name = row.get(field_names["indexname"], "") or ""
        index_name = _INDEX_NAMES.get(_normalise_header(raw_name))
        if index_name is None:
            continue
        try:
            as_of = _parse_date(row.get(field_names["indexdate"], "") or "")
        except ValueError:
            continue
        if as_of_max is not None and as_of > as_of_max:
            continue
        close_field = next(
            (
                field_names[key]
                for key in (
                    "closingindexvalue",
                    "closingindex",
                    "closeindexvalue",
                    "closeindex",
                )
                if key in field_names
            ),
            None,
        )
        close = _parse_decimal(row.get(close_field or ""))
        observation = IndexValuationObservation(
            index_name=index_name,
            as_of=as_of,
            pe=_parse_decimal(row.get(field_names["pe"])),
            pb=_parse_decimal(row.get(field_names["pb"])),
            div_yield=_parse_decimal(row.get(field_names["divyield"])),
            source="NSE",
            source_url=source_url,
            close=close if close is not None and close > 0 else None,
        )
        observations[(index_name, as_of)] = observation
    if not observations:
        raise ValueError("NSE index file contained no supported valuation rows")
    return sorted(observations.values(), key=lambda item: (item.index_name, item.as_of))


def fetch_nse_index_valuations(
    on: date,
    *,
    client: httpx.Client | None = None,
    lookback_days: int = 7,
) -> list[IndexValuationObservation]:
    """Fetch the latest available NSE daily all-indices archive within one week."""
    for offset in range(lookback_days + 1):
        file_date = on - timedelta(days=offset)
        source_url = NSE_INDEX_ARCHIVE_URL.format(date=file_date.strftime("%d%m%Y"))
        try:
            response = _request(source_url, client=client)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                continue
            raise
        try:
            records = parse_nse_index_valuations(
                response.content,
                source_url=source_url,
                as_of_max=on,
            )
        except ValueError:
            continue
        if records:
            return records
    raise ValueError("NSE index archive was unavailable for the last seven days")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def parse_treasury_yield_curve(
    content: str | bytes,
    *,
    source_url: str,
    as_of_max: date | None = None,
) -> list[MacroSeriesObservation]:
    """Parse Treasury's Atom/OData yield curve XML into 10-year and 3-month series."""
    payload = content.encode("utf-8") if isinstance(content, str) else content
    root = ElementTree.fromstring(payload)
    observations: list[MacroSeriesObservation] = []
    entries = [node for node in root.iter() if _local_name(node.tag) == "entry"]
    for entry in entries:
        fields = {
            _local_name(node.tag): (node.text or "").strip()
            for node in entry.iter()
            if node is not entry
        }
        date_value = fields.get("NEW_DATE")
        if not date_value:
            continue
        try:
            as_of = _parse_date(date_value)
        except ValueError:
            continue
        if as_of_max is not None and as_of > as_of_max:
            continue
        for field_name, series in _TREASURY_SERIES.items():
            value = _parse_decimal(fields.get(field_name))
            if value is not None:
                observations.append(
                    MacroSeriesObservation(
                        series=series,
                        as_of=as_of,
                        value=value,
                        source=TREASURY_SOURCE,
                        source_url=source_url,
                    )
                )
    return observations


def fetch_treasury_yields(
    on: date,
    *,
    client: httpx.Client | None = None,
    retention_days: int = MACRO_SERIES_RETENTION_DAYS,
) -> MacroSeriesBatch:
    """Fetch current and prior-year par curves, keeping a compact 400-day history."""
    observations: dict[tuple[str, date], MacroSeriesObservation] = {}
    errors: dict[str, str] = {}
    for year in (on.year - 1, on.year):
        source_url = (
            f"{TREASURY_XML_URL}?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
        )
        try:
            response = _request(source_url, client=client)
            rows = parse_treasury_yield_curve(
                response.content,
                source_url=source_url,
                as_of_max=on,
            )
            for row in rows:
                if row.as_of >= on - timedelta(days=retention_days):
                    observations[(row.series, row.as_of)] = row
            if not rows:
                errors[str(year)] = "EmptyFeed"
        except Exception as error:
            errors[str(year)] = type(error).__name__
    if not observations and not errors:
        errors["feed"] = "NoData"
    return MacroSeriesBatch(
        sorted(observations.values(), key=lambda item: (item.series, item.as_of)), errors
    )


def _plain_text(value: str) -> str:
    parser = _PlainText()
    parser.feed(value)
    return " ".join(html.unescape(" ".join(parser.parts)).split())


def _capped_text(value: str, *, maximum_bytes: int = 1024) -> str:
    cleaned = " ".join(value.split())
    encoded = cleaned.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return cleaned
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore").rstrip()


def _child_text(element: ElementTree.Element, names: set[str]) -> str:
    for child in element:
        if _local_name(child.tag).casefold() in names and child.text:
            return child.text.strip()
    return ""


def _element_link(element: ElementTree.Element, *, base_url: str) -> str:
    links = [child for child in element if _local_name(child.tag).casefold() == "link"]
    for child in sorted(
        links,
        key=lambda item: 0 if item.attrib.get("rel", "alternate").casefold() == "alternate" else 1,
    ):
        value = (child.text or "").strip() or child.attrib.get("href", "").strip()
        if value:
            return urljoin(base_url, value)
    return ""


def _published_at(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value.strip())
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_rss_atom_feed(
    content: str | bytes,
    *,
    source: str,
    feed_url: str,
    fetched_at: datetime,
) -> list[FeedItemObservation]:
    """Parse RSS 0.91/2.0 or Atom entries without following linked pages."""
    payload = content.encode("utf-8") if isinstance(content, str) else content
    root = ElementTree.fromstring(payload)
    elements = [
        node for node in root.iter() if _local_name(node.tag).casefold() in {"item", "entry"}
    ]
    results: list[FeedItemObservation] = []
    for element in elements:
        title = _capped_text(_plain_text(_child_text(element, {"title"})))
        link = _element_link(element, base_url=feed_url)
        if len(link.encode("utf-8")) > 2048:
            continue
        guid = _child_text(element, {"guid", "id"}) or link
        summary = _capped_text(
            _plain_text(_child_text(element, {"description", "summary", "content"}))
        )
        published_text = _child_text(element, {"pubdate", "published", "updated", "date"})
        published_at = _published_at(published_text)
        if not title or not guid or not link:
            continue
        results.append(
            FeedItemObservation(
                source=source,
                guid=_capped_text(guid, maximum_bytes=512),
                title=title,
                url=link,
                published_at=published_at,
                summary=summary,
                fetched_at=fetched_at,
            )
        )
    return results


def parse_federal_register_results(
    payload: Mapping[str, Any], *, fetched_at: datetime
) -> list[FeedItemObservation]:
    """Parse the Federal Register search API response for tariff/sanctions records."""
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Federal Register response has no results list")
    parsed: list[FeedItemObservation] = []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        title = row.get("title")
        url = row.get("html_url")
        guid = row.get("document_number")
        if not isinstance(title, str) or not isinstance(url, str) or not isinstance(guid, str):
            continue
        if len(url.encode("utf-8")) > 2048:
            continue
        publication = row.get("publication_date")
        published_at = None
        if isinstance(publication, str):
            published_day = _parse_date(publication)
            published_at = datetime.combine(published_day, datetime.min.time(), tzinfo=UTC)
        summary = row.get("abstract")
        parsed.append(
            FeedItemObservation(
                source="federal_register",
                guid=_capped_text(guid, maximum_bytes=512),
                title=_capped_text(_plain_text(title)),
                url=url,
                published_at=published_at,
                summary=_capped_text(_plain_text(summary if isinstance(summary, str) else "")),
                fetched_at=fetched_at,
            )
        )
    return parsed


def _federal_register_params(term: str, on: date) -> dict[str, Any]:
    return {
        "conditions[term]": term,
        "conditions[publication_date][gte]": (on - timedelta(days=FEED_RETENTION_DAYS)).isoformat(),
        "conditions[publication_date][lte]": on.isoformat(),
        "fields[]": ["title", "document_number", "html_url", "publication_date", "abstract"],
        "per_page": FEDERAL_REGISTER_TERM_LIMIT,
        "order": "newest",
    }


def fetch_official_feeds(
    on: date,
    *,
    client: httpx.Client | None = None,
    fetched_at: datetime | None = None,
) -> FeedBatch:
    """Fetch only official RSS/API summaries, continuing after any individual source fails."""
    retrieved_at = fetched_at or datetime.now(UTC)
    cutoff = on - timedelta(days=FEED_RETENTION_DAYS)
    items: dict[tuple[str, str], FeedItemObservation] = {}
    errors: dict[str, str] = {}

    for source, feed_url in FEED_URLS.items():
        try:
            response = _request(feed_url, client=client)
            parsed = parse_rss_atom_feed(
                response.content,
                source=source,
                feed_url=str(response.url),
                fetched_at=retrieved_at,
            )
            selected = [
                item
                for item in parsed
                if item.published_at is None
                or cutoff <= item.published_at.date() <= on + timedelta(days=1)
            ][:FEED_ITEM_LIMIT]
            if not selected:
                errors[source] = "StaleFeed" if parsed else "EmptyFeed"
            for item in selected:
                items[(item.source, item.guid)] = item
        except Exception as error:
            errors[source] = type(error).__name__

    for term in ("tariffs", "sanctions"):
        try:
            response = _request(
                FEDERAL_REGISTER_API_URL,
                client=client,
                params=_federal_register_params(term, on),
            )
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("Federal Register response is not an object")
            parsed = parse_federal_register_results(payload, fetched_at=retrieved_at)
            for item in parsed:
                if item.published_at is None or item.published_at.date() >= cutoff:
                    items[(item.source, item.guid)] = item
        except Exception as error:
            errors[f"federal_register:{term}"] = type(error).__name__

    if not any(source == "federal_register" for source, _guid in items) and not any(
        key.startswith("federal_register:") for key in errors
    ):
        errors["federal_register"] = "EmptyFeed"
    return FeedBatch(
        sorted(
            items.values(),
            key=lambda item: (
                item.published_at or item.fetched_at,
                item.source,
                item.guid,
            ),
            reverse=True,
        ),
        errors,
    )
