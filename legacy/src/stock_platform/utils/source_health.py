from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

import requests
import yfinance as yf

_log = logging.getLogger(__name__)

_HEALTH_CACHE_KEY = "source_health_v1"
_HEALTH_CACHE_TTL_SECONDS = 1800  # 30 minutes

# 10 large-cap NSE symbols that reliably exist across all three data sources.
_PROBE_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "ITC", "LT", "SBIN", "BHARTIARTL", "KOTAKBANK",
]

_SCREENER_BASE = "https://www.screener.in/company/{symbol}/consolidated/"
_NSE_CSV_URL   = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
_REQUEST_TIMEOUT = 8  # seconds per probe

_SCREENER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
    )
}

_STATUS_HEALTHY  = "HEALTHY"
_STATUS_DEGRADED = "DEGRADED"
_STATUS_FAILED   = "FAILED"

_THRESHOLD_HEALTHY  = 0.80
_THRESHOLD_DEGRADED = 0.50


def _rate_to_status(success_rate: float) -> str:
    if success_rate >= _THRESHOLD_HEALTHY:
        return _STATUS_HEALTHY
    if success_rate >= _THRESHOLD_DEGRADED:
        return _STATUS_DEGRADED
    return _STATUS_FAILED


class SourceHealthChecker:
    """
    Pre-flight probe for the three external data sources used by FLOW 2 and FLOW 3.

    Results are cached via the repository's cache_entries table (30-min TTL) to
    avoid hammering external services on every run.
    """

    def __init__(self, repo: Any) -> None:
        self.repo = repo
        self._session = requests.Session()
        self._session.headers.update(_SCREENER_HEADERS)

    # ── per-source probes ──────────────────────────────────────────────────────

    def check_screener(self) -> tuple[float, list[str]]:
        """
        Probe Screener.in for each symbol; success = HTTP 200 with non-trivial body.
        Returns (success_rate, failed_symbols).
        """
        failed: list[str] = []

        def _probe(symbol: str) -> bool:
            try:
                url = _SCREENER_BASE.format(symbol=symbol)
                resp = self._session.get(url, timeout=_REQUEST_TIMEOUT, allow_redirects=True)
                # A redirect to login page means the symbol 404'd; body < 5 kB means no data.
                return resp.status_code == 200 and len(resp.content) > 5_000
            except Exception as exc:
                _log.debug("Screener probe %s: %r", symbol, exc)
                return False

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_probe, sym): sym for sym in _PROBE_SYMBOLS}
            for fut in as_completed(futures):
                sym = futures[fut]
                if not fut.result():
                    failed.append(sym)

        success_rate = round(1.0 - len(failed) / len(_PROBE_SYMBOLS), 3)
        _log.info("Screener health: %.0f%% (%d failed)", success_rate * 100, len(failed))
        return success_rate, failed

    def check_yfinance(self) -> tuple[float, list[str]]:
        """
        Probe yfinance for each symbol via fast_info.last_price.
        Returns (success_rate, failed_symbols).
        """
        failed: list[str] = []

        def _probe(symbol: str) -> bool:
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                price = ticker.fast_info.last_price
                return price is not None and float(price) > 0
            except Exception as exc:
                _log.debug("yfinance probe %s: %r", symbol, exc)
                return False

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_probe, sym): sym for sym in _PROBE_SYMBOLS}
            for fut in as_completed(futures):
                sym = futures[fut]
                if not fut.result():
                    failed.append(sym)

        success_rate = round(1.0 - len(failed) / len(_PROBE_SYMBOLS), 3)
        _log.info("yfinance health: %.0f%% (%d failed)", success_rate * 100, len(failed))
        return success_rate, failed

    def check_nse_index(self) -> tuple[bool, str | None]:
        """
        Probe the NSE NIFTY50 CSV endpoint.
        Returns (ok, error_message_or_None).
        """
        try:
            resp = self._session.get(_NSE_CSV_URL, timeout=_REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            # Expect at least 10 data rows — a valid NIFTY50 CSV has 50.
            row_count = resp.text.count("\n")
            if row_count < 10:
                return False, f"CSV too short ({row_count} lines)"
            return True, None
        except Exception as exc:
            msg = str(exc)
            _log.debug("NSE CSV probe: %r", exc)
            return False, msg

    # ── orchestrator ──────────────────────────────────────────────────────────

    def run_health_check(self) -> dict[str, Any]:
        """
        Run all three source probes and return a structured health report.

        Schema::

            {
                "overall_status": "HEALTHY" | "DEGRADED" | "FAILED",
                "timestamp": "<ISO-8601 UTC>",
                "sources": {
                    "screener": {"status": ..., "success_rate": 0.9, "failed": [...]},
                    "yfinance":  {"status": ..., "success_rate": 1.0, "failed": []},
                    "nse_index": {"status": ..., "ok": True, "error": None},
                },
            }
        """
        timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        screener_rate, screener_failed = self.check_screener()
        yfinance_rate, yfinance_failed = self.check_yfinance()
        nse_ok, nse_error            = self.check_nse_index()

        screener_status = _rate_to_status(screener_rate)
        yfinance_status = _rate_to_status(yfinance_rate)
        nse_status      = _STATUS_HEALTHY if nse_ok else _STATUS_FAILED

        # Overall: worst of all three source statuses.
        _rank = {_STATUS_HEALTHY: 0, _STATUS_DEGRADED: 1, _STATUS_FAILED: 2}
        worst = max(screener_status, yfinance_status, nse_status, key=lambda s: _rank[s])

        report: dict[str, Any] = {
            "overall_status": worst,
            "timestamp": timestamp,
            "sources": {
                "screener": {
                    "status": screener_status,
                    "success_rate": screener_rate,
                    "failed": screener_failed,
                },
                "yfinance": {
                    "status": yfinance_status,
                    "success_rate": yfinance_rate,
                    "failed": yfinance_failed,
                },
                "nse_index": {
                    "status": nse_status,
                    "ok": nse_ok,
                    "error": nse_error,
                },
            },
        }
        _log.info(
            "Source health overall=%s  screener=%.0f%%  yfinance=%.0f%%  nse=%s",
            worst, screener_rate * 100, yfinance_rate * 100,
            "OK" if nse_ok else f"FAIL({nse_error})",
        )
        return report


# ── module-level helpers used by agents ───────────────────────────────────────

def get_source_health(repo: Any) -> dict[str, Any]:
    """
    Return a cached health report, running a fresh check only on cache miss.

    Reads from ``cache_entries`` (key ``source_health_v1``, TTL 30 min) so
    repeated calls within the same half-hour window hit no external services.
    """
    cached = repo.get_cache(_HEALTH_CACHE_KEY)
    if cached:
        _log.debug("Source health: cache hit (status=%s)", cached.get("overall_status"))
        return cached

    checker = SourceHealthChecker(repo)
    report  = checker.run_health_check()
    repo.set_cache(_HEALTH_CACHE_KEY, report, ttl_seconds=_HEALTH_CACHE_TTL_SECONDS)
    return report


def assert_source_health(repo: Any) -> dict[str, Any] | None:
    """
    Run (or retrieve) the health check and raise ValueError if sources FAILED.

    Returns the report dict for HEALTHY/DEGRADED so callers can attach warnings.
    Raises ValueError (aborts the LangGraph node) on FAILED.
    """
    report = get_source_health(repo)
    status = report.get("overall_status", _STATUS_FAILED)

    if status == _STATUS_FAILED:
        sources = report.get("sources", {})
        details = "; ".join(
            f"{src}={info.get('status', '?')}"
            for src, info in sources.items()
        )
        raise ValueError(
            f"Data source health check FAILED — aborting run to avoid bad recommendations. "
            f"Details: {details}. "
            f"Retry in 30 minutes or check network/API availability."
        )

    return report
