from __future__ import annotations

from datetime import date

import httpx
import pytest

from wealth.marketdata.sources.edgar import EdgarConfigurationError
from wealth.marketdata.sources.edgar_filings import (
    SEC_COMPANY_TICKERS_URL,
    SEC_SUBMISSIONS_URL,
    SecFilingsClient,
    extract_8k_items,
)
from wealth.marketdata.sources.nse_announcements import (
    NSE_ANNOUNCEMENTS_API_URL,
    NSE_ANNOUNCEMENTS_PAGE_URL,
    NseAnnouncementsClient,
)


def test_sec_filings_client_requires_contact_user_agent() -> None:
    with pytest.raises(EdgarConfigurationError, match="SEC_USER_AGENT"):
        SecFilingsClient(user_agent="")


def test_sec_filings_client_reads_official_json_and_filing_text() -> None:
    accession = "0000000001-26-000001"
    user_agents: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        user_agents.append(request.headers["User-Agent"])
        if str(request.url) == SEC_COMPANY_TICKERS_URL:
            return httpx.Response(
                200,
                json={"0": {"ticker": "SYNTH", "cik_str": 1, "title": "Synthetic Issuer"}},
                request=request,
            )
        if str(request.url) == SEC_SUBMISSIONS_URL.format(cik="0000000001"):
            return httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {
                            "form": ["8-K", "10-Q"],
                            "filingDate": ["2026-10-29", "2026-10-29"],
                            "accessionNumber": [accession, "0000000001-26-000002"],
                            "primaryDocument": ["synthetic.htm", "quarterly.htm"],
                        }
                    }
                },
                request=request,
            )
        return httpx.Response(200, text="<p>ITEM 4.01 auditor change</p>", request=request)

    transport = httpx.MockTransport(handle)
    with SecFilingsClient(
        "Sentinel test contact=sentinel@example.test", transport=transport, sleep=lambda _s: None
    ) as client:
        companies = client.company_tickers()
        filings = client.filings("1", forms={"8-K"}, days=2, as_of=date(2026, 10, 29))
        text_content = client.filing_text(filings[0], max_chars=100)

    assert companies["SYNTH"].cik == "0000000001"
    assert len(filings) == 1
    assert filings[0].accession == accession
    assert text_content == "<p>ITEM 4.01 auditor change</p>"
    assert all(agent == "Sentinel test contact=sentinel@example.test" for agent in user_agents)
    assert "sec.gov" in filings[0].document_url


def test_8k_item_extractor_handles_html_and_case() -> None:
    assert extract_8k_items("<div>item 4.01</div><p>ITEM 5.02</p>") == {"4.01", "5.02"}


def test_nse_official_api_403_is_visible_as_a_blocked_response() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if str(request.url) == NSE_ANNOUNCEMENTS_PAGE_URL:
            return httpx.Response(200, text="NSE page", request=request)
        assert str(request.url).startswith(NSE_ANNOUNCEMENTS_API_URL)
        return httpx.Response(403, request=request)

    with (
        NseAnnouncementsClient(transport=httpx.MockTransport(handle)) as client,
        pytest.raises(httpx.HTTPStatusError) as error,
    ):
        client.announcements("SYNTH", days=2, as_of=date(2026, 10, 29))

    assert error.value.response.status_code == 403
