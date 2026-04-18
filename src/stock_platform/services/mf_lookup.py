from __future__ import annotations

import hashlib
import re
import time
from typing import Any

import requests

from stock_platform.services.amc_adapters import OfficialAMCResolver
from stock_platform.config import AppConfig

# Maps truncated company names (cleaned[:18]) produced when an API response
# lacks a proper NSE symbol, to the canonical NSE ticker.
_MF_NAME_TO_NSE: dict[str, str] = {
    "HDFCBANKLIMITED":      "HDFCBANK",
    "ICICIBANKLIMITED":     "ICICIBANK",
    "RELIANCEINDUSTRIES":   "RELIANCE",
    "AXISBANKLIMITED":      "AXISBANK",
    "TATACONSULTANCYSER":   "TCS",
    "INFOSYSLIMITED":       "INFY",
    "BHARTIAIRTELLIMITE":   "BHARTIARTL",
    "STATEBANKOFINDIA":     "SBIN",
    "SUNPHARMACEUTICALI":   "SUNPHARMA",
    "HCLTECHNOLOGIESLIM":   "HCLTECH",
    "BIOCONLIMITED":        "BIOCON",
    "SBILIFEINSURANCECO":   "SBILIFE",
    "95MUTHOOTFINANCELI":   "MUTHOOTFIN",
}


class MutualFundHoldingsClient:
    def __init__(self, config: AppConfig, repo: Any | None = None) -> None:
        self.config = config
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "stock-langgraph-platform/0.1"})
        self.official_resolver = OfficialAMCResolver()
        self.scheme_cache: dict[str, dict[str, Any] | None] = {}
        self.holdings_cache: dict[tuple[str, str], dict[str, float] | None] = {}
        self.source_cache: dict[tuple[str, str], str] = {}

    def _cache_key(self, prefix: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def _load_repo_cache(self, key: str) -> Any:
        if self.repo is None:
            return None
        return self.repo.get_cache(key, default=None)

    def _persist_repo_cache(self, key: str, payload: Any, ttl_seconds: int = 86400) -> None:
        if self.repo is not None and payload is not None:
            self.repo.set_cache(key, payload, ttl_seconds=ttl_seconds)

    def resolve_holdings(self, fund_name: str, month: str | None = None) -> dict[str, float] | None:
        official = self.official_resolver.resolve(fund_name, month=month)
        if official:
            cache_key = (self._normalize_name(fund_name), month or "latest")
            self.holdings_cache[cache_key] = official.holdings
            self.source_cache[cache_key] = official.source
            return official.holdings
        scheme = self.find_best_scheme(fund_name)
        if not scheme:
            return None
        family_id = self._extract_family_id(scheme)
        if not family_id:
            details = self.fetch_scheme_details(str(scheme.get("scheme_code") or scheme.get("amfi_code") or ""))
            family_id = self._extract_family_id(details or {})
        if not family_id:
            return None
        cache_key = (str(family_id), month or "latest")
        if cache_key in self.holdings_cache:
            return self.holdings_cache[cache_key]
        holdings = self.fetch_family_holdings(str(family_id), month=month)
        self.holdings_cache[cache_key] = holdings
        if holdings:
            self.source_cache[cache_key] = "fallback:mfdata"
        return holdings

    def resolve_with_meta(self, fund_name: str, month: str | None = None) -> tuple[dict[str, float] | None, str]:
        official = self.official_resolver.resolve(fund_name, month=month)
        if official:
            return official.holdings, official.source
        holdings = self.resolve_holdings(fund_name, month=month)
        if not holdings:
            return None, "unresolved"
        return holdings, "fallback:mfdata"

    def find_best_scheme(self, fund_name: str) -> dict[str, Any] | None:
        normalized = self._normalize_name(fund_name)
        if normalized in self.scheme_cache:
            return self.scheme_cache[normalized]
        cache_key = self._cache_key("mf_scheme_search", normalized)
        cached = self._load_repo_cache(cache_key)
        if isinstance(cached, dict):
            self.scheme_cache[normalized] = cached
            return cached
        url = f"{self.config.mf_api_base_url.rstrip('/')}/search"
        candidates = self._request_json(url, params={"q": normalized})
        if not isinstance(candidates, list):
            self.scheme_cache[normalized] = None
            return None
        best = self._pick_best_match(normalized, candidates)
        self.scheme_cache[normalized] = best
        self._persist_repo_cache(cache_key, best, ttl_seconds=43200)
        return best

    def fetch_scheme_details(self, scheme_code: str) -> dict[str, Any] | None:
        if not scheme_code:
            return None
        url = f"{self.config.mf_api_base_url.rstrip('/')}/schemes/{scheme_code}"
        details = self._request_json(url)
        return details if isinstance(details, dict) else None

    def fetch_family_holdings(self, family_id: str, month: str | None = None) -> dict[str, float] | None:
        params = {"month": month} if month else None
        url = f"{self.config.mf_api_base_url.rstrip('/')}/families/{family_id}/holdings"
        payload = self._request_json(url, params=params)
        if not isinstance(payload, dict):
            return None
        equities = payload.get("equity_holdings") or payload.get("holdings") or []
        holdings: dict[str, float] = {}
        for row in equities:
            if not isinstance(row, dict):
                continue
            name = row.get("stock_name") or row.get("company_name") or row.get("security_name") or row.get("symbol")
            weight = row.get("weight_pct") or row.get("weight") or row.get("percent")
            if not name or weight is None:
                continue
            symbol = self._symbol_from_name(name, row.get("symbol"))
            holdings[symbol] = float(weight) / 100.0
        return holdings or None

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        cache_identity = f"{url}?{repr(sorted((params or {}).items()))}"
        cache_key = self._cache_key("mf_http", cache_identity)
        cached = self._load_repo_cache(cache_key)
        if cached is not None:
            return cached
        for attempt in range(2):
            try:
                response = self.session.get(url, params=params, timeout=self.config.mf_holdings_timeout_seconds)
                if response.status_code >= 500:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                response.raise_for_status()
                body = response.json()
                payload = body.get("data")
                self._persist_repo_cache(cache_key, payload, ttl_seconds=86400)
                return payload
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        return None

    def _pick_best_match(self, normalized_query: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        query_tokens = set(normalized_query.split())
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in candidates[:20]:
            scheme_name = self._normalize_name(str(item.get("scheme_name") or item.get("name") or ""))
            scheme_tokens = set(scheme_name.split())
            overlap = len(query_tokens & scheme_tokens)
            score = overlap / max(len(query_tokens), 1)
            if "direct" in query_tokens and "direct" not in scheme_tokens:
                score -= 0.15
            if "regular" in query_tokens and "regular" not in scheme_tokens:
                score -= 0.15
            if "growth" in query_tokens and "growth" not in scheme_tokens:
                score -= 0.05
            if "etf" in query_tokens and "etf" not in scheme_tokens:
                score -= 0.2
            ranked.append((score, item))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.45:
            return None
        return ranked[0][1]

    def _extract_family_id(self, payload: dict[str, Any]) -> str | None:
        for key in ("family_id", "familyId", "portfolio_family_id", "scheme_family_id"):
            value = payload.get(key)
            if value not in (None, ""):
                return str(value)
        family = payload.get("family")
        if isinstance(family, dict):
            for key in ("id", "family_id"):
                value = family.get(key)
                if value not in (None, ""):
                    return str(value)
        return None

    def _normalize_name(self, value: str) -> str:
        normalized = value.lower()
        normalized = normalized.replace("&", " and ")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _symbol_from_name(self, name: str, provided_symbol: str | None = None) -> str:
        symbol = (provided_symbol or "").replace(".NSE", "").replace(".BSE", "").replace(" ", "").upper()
        if symbol:
            return symbol
        cleaned = re.sub(r"[^A-Za-z0-9]+", "", name).upper()
        raw = cleaned[:18] if cleaned else "UNKNOWN"
        return _MF_NAME_TO_NSE.get(raw, raw)
