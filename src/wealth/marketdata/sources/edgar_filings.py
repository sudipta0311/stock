"""Official SEC submissions and filing-text access for the daily Sentinel."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from wealth.marketdata.sources.edgar import (
    SEC_COMPANY_TICKERS_URL,
    EdgarCompany,
    EdgarConfigurationError,
    parse_company_tickers,
)

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_SOURCE_URL = "https://www.sec.gov/edgar/search/"
_EMAIL_PATTERN = re.compile(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+")
_ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_DOCUMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,160}$")


@dataclass(frozen=True)
class EdgarFiling:
    cik: str
    accession: str
    form: str
    filed_at: date
    document_url: str
    primary_document: str


class SecFilingsClient:
    """Read SEC's public JSON and archive endpoints at a maximum of ten requests/second."""

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        supplied = user_agent if user_agent is not None else os.getenv("SEC_USER_AGENT", "")
        self.user_agent = supplied.strip()
        if not self.user_agent or not _EMAIL_PATTERN.search(self.user_agent):
            raise EdgarConfigurationError(
                "Set SEC_USER_AGENT to an identifying client string with a contact email."
            )
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept": "application/json,text/html,*/*"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._sleep = sleep
        self._last_request_at: float | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SecFilingsClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _get(self, url: str) -> httpx.Response:
        if self._last_request_at is not None:
            remaining = 0.1 - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = time.monotonic()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def company_tickers(self) -> dict[str, EdgarCompany]:
        payload = self._get(SEC_COMPANY_TICKERS_URL).json()
        if not isinstance(payload, dict):
            raise ValueError("SEC company tickers response must be an object")
        return parse_company_tickers(payload)

    def filings(
        self,
        cik: str,
        *,
        forms: set[str],
        days: int,
        as_of: date,
    ) -> list[EdgarFiling]:
        normalized = cik.strip().removeprefix("CIK").zfill(10)
        if len(normalized) != 10 or not normalized.isdigit():
            raise ValueError("CIK must contain at most ten digits")
        if not 1 <= days <= 90:
            raise ValueError("days must be between 1 and 90")
        payload = self._get(SEC_SUBMISSIONS_URL.format(cik=normalized)).json()
        recent = payload.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            return []
        fields = [
            recent.get(key, [])
            for key in ("form", "filingDate", "accessionNumber", "primaryDocument")
        ]
        if not all(isinstance(field, list) for field in fields):
            return []
        forms_list, filed_list, accessions, documents = fields
        result: list[EdgarFiling] = []
        cutoff = as_of - timedelta(days=days)
        for form, filed, accession, document in zip(
            forms_list, filed_list, accessions, documents, strict=False
        ):
            if not isinstance(form, str) or form not in forms:
                continue
            if not isinstance(filed, str) or not isinstance(accession, str):
                continue
            if not isinstance(document, str) or not _DOCUMENT_PATTERN.fullmatch(document):
                continue
            try:
                filed_at = date.fromisoformat(filed)
            except ValueError:
                continue
            if filed_at < cutoff or filed_at > as_of or not _ACCESSION_PATTERN.fullmatch(accession):
                continue
            cik_path = str(int(normalized))
            accession_path = accession.replace("-", "")
            url = f"{SEC_ARCHIVE_URL}/{cik_path}/{accession_path}/{quote(document, safe='._-')}"
            result.append(EdgarFiling(normalized, accession, form, filed_at, url, document))
        return sorted(result, key=lambda item: (item.filed_at, item.accession), reverse=True)

    def filing_text(self, filing: EdgarFiling, *, max_chars: int) -> str:
        if not 1 <= max_chars <= 100_000:
            raise ValueError("max_chars must be between 1 and 100000")
        return self._get(filing.document_url).text[:max_chars]


def extract_8k_items(text: str) -> set[str]:
    """Extract recognized 8-K item identifiers from plain or HTML filing text."""
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = plain.replace("\xa0", " ")
    return {match.group(1) for match in re.finditer(r"\bITEM\s+([1-9]\.\d{2})\b", plain, re.I)}
