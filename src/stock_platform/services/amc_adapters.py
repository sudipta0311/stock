from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests


def normalize_scheme_name(value: str) -> str:
    normalized = value.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def symbol_from_company_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", name).upper()
    return cleaned[:18] if cleaned else "UNKNOWN"


@dataclass(slots=True)
class AdapterResult:
    holdings: dict[str, float]
    source: str
    url: str


class BaseAMCAdapter:
    def __init__(self, session: requests.Session) -> None:
        self.session = session

    def resolve(self, fund_name: str, month: str | None = None) -> AdapterResult | None:
        raise NotImplementedError


class MiraeAssetAdapter(BaseAMCAdapter):
    """Parses official Mirae scheme pages that expose jsonHoldings in the page source."""

    SCHEME_URLS = {
        normalize_scheme_name("Mirae Asset Large Cap Fund - Direct Plan"): "https://www.miraeassetmf.co.in/mutual-fund-scheme/equity-fund/mirae-asset-large-cap-fund",
        normalize_scheme_name("Mirae Asset Equity Savings Fund - Direct Plan"): "https://www.miraeassetmf.co.in/mutual-fund-scheme/hybrid-fund/mirae-asset-equity-savings-fund",
    }

    def resolve(self, fund_name: str, month: str | None = None) -> AdapterResult | None:
        url = self.SCHEME_URLS.get(normalize_scheme_name(fund_name))
        if not url:
            return None
        response = self.session.get(url, timeout=25)
        response.raise_for_status()
        match = re.search(r"var jsonHoldings = (\[.*?\]);", response.text, re.S)
        if not match:
            return None
        rows = json.loads(match.group(1))
        holdings: dict[str, float] = {}
        for row in rows:
            name = str(row.get("Name") or "").strip()
            pct_text = str(row.get("Percentage") or "").strip()
            if not name or not pct_text or name.lower() == "other equities":
                continue
            try:
                pct = float(pct_text.replace("%", "").replace(",", ""))
            except ValueError:
                continue
            symbol = symbol_from_company_name(name)
            holdings[symbol] = pct / 100.0
        if not holdings:
            return None
        return AdapterResult(holdings=holdings, source="official:miraeasset", url=url)


class OfficialAMCResolver:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "stock-langgraph-platform/0.1"})
        self.adapters: list[BaseAMCAdapter] = [
            MiraeAssetAdapter(self.session),
        ]

    def resolve(self, fund_name: str, month: str | None = None) -> AdapterResult | None:
        for adapter in self.adapters:
            try:
                result = adapter.resolve(fund_name, month=month)
            except Exception:
                result = None
            if result:
                return result
        return None

