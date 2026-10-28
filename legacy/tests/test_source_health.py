from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.utils.source_health import (
    SourceHealthChecker,
    _PROBE_SYMBOLS,
    _STATUS_DEGRADED,
    _STATUS_FAILED,
    _STATUS_HEALTHY,
    _rate_to_status,
    assert_source_health,
    get_source_health,
)


class TestRateToStatus(unittest.TestCase):
    def test_healthy_threshold(self):
        self.assertEqual(_rate_to_status(1.0), _STATUS_HEALTHY)
        self.assertEqual(_rate_to_status(0.80), _STATUS_HEALTHY)

    def test_degraded_threshold(self):
        self.assertEqual(_rate_to_status(0.79), _STATUS_DEGRADED)
        self.assertEqual(_rate_to_status(0.50), _STATUS_DEGRADED)

    def test_failed_threshold(self):
        self.assertEqual(_rate_to_status(0.49), _STATUS_FAILED)
        self.assertEqual(_rate_to_status(0.0), _STATUS_FAILED)


class _FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"x" * 6000, text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text or "\n".join(["row"] * 55)


class TestSourceHealthChecker(unittest.TestCase):
    def _make_checker(self):
        repo = MagicMock()
        checker = SourceHealthChecker(repo)
        return checker

    # ── check_screener ─────────────────────────────────────────────────────

    def test_check_screener_all_ok(self):
        checker = self._make_checker()
        with patch.object(checker._session, "get", return_value=_FakeResponse(200, b"x" * 6000)):
            rate, failed = checker.check_screener()
        self.assertAlmostEqual(rate, 1.0)
        self.assertEqual(failed, [])

    def test_check_screener_all_fail(self):
        checker = self._make_checker()
        with patch.object(checker._session, "get", return_value=_FakeResponse(404, b"")):
            rate, failed = checker.check_screener()
        self.assertAlmostEqual(rate, 0.0)
        self.assertEqual(len(failed), len(_PROBE_SYMBOLS))

    def test_check_screener_partial(self):
        """60 % of symbols succeed → DEGRADED."""
        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 6:
                return _FakeResponse(200, b"x" * 6000)
            return _FakeResponse(404, b"")

        checker = self._make_checker()
        with patch.object(checker._session, "get", side_effect=_side_effect):
            rate, failed = checker.check_screener()
        self.assertGreaterEqual(rate, 0.50)
        self.assertLess(rate, 0.80)

    # ── check_yfinance ─────────────────────────────────────────────────────

    def test_check_yfinance_all_ok(self):
        checker = self._make_checker()
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 1500.0
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        with patch("stock_platform.utils.source_health.yf.Ticker", return_value=mock_ticker):
            rate, failed = checker.check_yfinance()
        self.assertAlmostEqual(rate, 1.0)
        self.assertEqual(failed, [])

    def test_check_yfinance_all_fail(self):
        checker = self._make_checker()
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = None
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        with patch("stock_platform.utils.source_health.yf.Ticker", return_value=mock_ticker):
            rate, failed = checker.check_yfinance()
        self.assertAlmostEqual(rate, 0.0)
        self.assertEqual(len(failed), len(_PROBE_SYMBOLS))

    # ── check_nse_index ────────────────────────────────────────────────────

    def test_check_nse_index_ok(self):
        checker = self._make_checker()
        fake = _FakeResponse(200, b"", "\n".join(["row"] * 55))
        with patch.object(checker._session, "get", return_value=fake):
            ok, err = checker.check_nse_index()
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_check_nse_index_bad_status(self):
        checker = self._make_checker()
        with patch.object(checker._session, "get", return_value=_FakeResponse(503)):
            ok, err = checker.check_nse_index()
        self.assertFalse(ok)
        self.assertIn("503", err)

    def test_check_nse_index_network_error(self):
        checker = self._make_checker()
        with patch.object(checker._session, "get", side_effect=Exception("timeout")):
            ok, err = checker.check_nse_index()
        self.assertFalse(ok)
        self.assertIsNotNone(err)

    # ── run_health_check ───────────────────────────────────────────────────

    def test_run_health_check_all_healthy(self):
        checker = self._make_checker()
        with (
            patch.object(checker, "check_screener", return_value=(1.0, [])),
            patch.object(checker, "check_yfinance", return_value=(1.0, [])),
            patch.object(checker, "check_nse_index", return_value=(True, None)),
        ):
            report = checker.run_health_check()
        self.assertEqual(report["overall_status"], _STATUS_HEALTHY)
        self.assertIn("timestamp", report)
        self.assertIn("sources", report)

    def test_run_health_check_one_source_degraded(self):
        checker = self._make_checker()
        with (
            patch.object(checker, "check_screener", return_value=(0.60, ["A", "B", "C", "D"])),
            patch.object(checker, "check_yfinance", return_value=(1.0, [])),
            patch.object(checker, "check_nse_index", return_value=(True, None)),
        ):
            report = checker.run_health_check()
        self.assertEqual(report["overall_status"], _STATUS_DEGRADED)

    def test_run_health_check_nse_failure_causes_failed(self):
        checker = self._make_checker()
        with (
            patch.object(checker, "check_screener", return_value=(1.0, [])),
            patch.object(checker, "check_yfinance", return_value=(1.0, [])),
            patch.object(checker, "check_nse_index", return_value=(False, "timeout")),
        ):
            report = checker.run_health_check()
        self.assertEqual(report["overall_status"], _STATUS_FAILED)


class TestGetSourceHealth(unittest.TestCase):
    def _make_repo(self, cached_value=None):
        repo = MagicMock()
        repo.get_cache.return_value = cached_value
        return repo

    def test_returns_cached_result(self):
        cached = {"overall_status": _STATUS_HEALTHY, "timestamp": "T", "sources": {}}
        repo = self._make_repo(cached_value=cached)
        result = get_source_health(repo)
        self.assertEqual(result, cached)
        repo.set_cache.assert_not_called()

    def test_runs_check_on_miss_and_caches(self):
        repo = self._make_repo(cached_value=None)
        healthy_report = {
            "overall_status": _STATUS_HEALTHY,
            "timestamp": "T",
            "sources": {"screener": {}, "yfinance": {}, "nse_index": {}},
        }
        with patch(
            "stock_platform.utils.source_health.SourceHealthChecker.run_health_check",
            return_value=healthy_report,
        ):
            result = get_source_health(repo)
        self.assertEqual(result["overall_status"], _STATUS_HEALTHY)
        repo.set_cache.assert_called_once()


class TestAssertSourceHealth(unittest.TestCase):
    def test_raises_on_failed(self):
        repo = MagicMock()
        failed_report = {
            "overall_status": _STATUS_FAILED,
            "timestamp": "T",
            "sources": {
                "screener": {"status": _STATUS_FAILED},
                "yfinance": {"status": _STATUS_HEALTHY},
                "nse_index": {"status": _STATUS_FAILED},
            },
        }
        with patch("stock_platform.utils.source_health.get_source_health", return_value=failed_report):
            with self.assertRaises(ValueError) as ctx:
                assert_source_health(repo)
        self.assertIn("FAILED", str(ctx.exception))

    def test_returns_report_on_degraded(self):
        repo = MagicMock()
        degraded_report = {
            "overall_status": _STATUS_DEGRADED,
            "timestamp": "T",
            "sources": {},
        }
        with patch("stock_platform.utils.source_health.get_source_health", return_value=degraded_report):
            result = assert_source_health(repo)
        self.assertEqual(result["overall_status"], _STATUS_DEGRADED)

    def test_returns_report_on_healthy(self):
        repo = MagicMock()
        healthy_report = {"overall_status": _STATUS_HEALTHY, "timestamp": "T", "sources": {}}
        with patch("stock_platform.utils.source_health.get_source_health", return_value=healthy_report):
            result = assert_source_health(repo)
        self.assertEqual(result["overall_status"], _STATUS_HEALTHY)


if __name__ == "__main__":
    unittest.main()
