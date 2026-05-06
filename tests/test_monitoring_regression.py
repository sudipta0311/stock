"""
Regression tests for monitoring agent pure functions.
Covers all historical failure modes so the fix loop cannot repeat them.
"""
from __future__ import annotations

import sys
import types
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path (mirrors test_monitoring_tax_logic.py pattern).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Import the pure functions under test — no I/O, no network, no DB.
# ---------------------------------------------------------------------------
from stock_platform.agents.monitor_agents import (
    _earnings_blackout_active,
    _compute_action_for_winner,
    _enforce_field_consistency,
    _rebuild_rationale,
    _valuation_floor_active,
    _ltcg_guard,
    _get_next_earnings_date,
    _check_data_quality_flags,
)


# ===========================================================================
# 1. _earnings_blackout_active — future date within 7 days → must block
# ===========================================================================
def test_earnings_blackout_active_blocks_within_7d():
    d = {"next_earnings_date": date.today() + timedelta(days=3)}
    assert _earnings_blackout_active(d) is True


# ===========================================================================
# 2. _earnings_blackout_active — date already past → must NOT block
#    (regression: last_result+91d gave a past date → returned False → BUY MORE passed)
# ===========================================================================
def test_earnings_blackout_does_not_block_past_date():
    d = {"next_earnings_date": date.today() - timedelta(days=5)}
    assert _earnings_blackout_active(d) is False


# ===========================================================================
# 3. _get_next_earnings_date — all sources fail → (None, "unavailable")
#    (regression: silent-fail returned (past_date, ...) → blackout missed)
# ===========================================================================
def test_get_next_earnings_date_fail_closed():
    # Patch the source list directly — the list captures function references at
    # import time, so patching individual function names has no effect.
    with patch(
        "stock_platform.agents.monitor_agents._EARNINGS_DATE_SOURCES",
        [("stub_1", lambda _: None), ("stub_2", lambda _: None), ("stub_3", lambda _: None)],
    ):
        d, source = _get_next_earnings_date("TESTSTOCK")
    assert d is None
    assert source == "unavailable"


# ===========================================================================
# 4. _compute_action_for_winner — missing PE data → HOLD + degraded flag
#    (regression: None PE caused AttributeError or silent wrong TRIM)
# ===========================================================================
def test_compute_action_for_winner_missing_pe_degrades_to_hold():
    result = _compute_action_for_winner({
        "symbol": "TEST",
        "current_pe": None,
        "pe_5yr_avg": None,
        "pct_from_52w_high": -5.0,
        "quant_score": 0.6,
        "thesis_status": "INTACT",
    })
    assert result == "HOLD"


# ===========================================================================
# 5. _enforce_field_consistency Rule 3 — CRITICAL+HOLD → REVIEW
#    (regression: LLM downgraded severity to LOW so rule never fired)
# ===========================================================================
def test_enforce_consistency_critical_hold_escalates_to_review():
    data = {
        "action": "HOLD",
        "severity": "CRITICAL",
        "rationale": "Thesis intact (quant 0.60)",
        "quant_score": 0.60,
        "thesis_status": "INTACT",
        "pnl_pct": -5.0,
    }
    result = _enforce_field_consistency(data)
    assert result["action"] == "REVIEW"
    # Rationale must be rebuilt — must NOT contain stale "Thesis intact" literal
    # while also containing the AUTO escalation notice.
    assert "[AUTO:" in result["rationale"]


# ===========================================================================
# 6. _enforce_field_consistency Rule 1 — quant<0.45+INTACT → WEAKENED + rebuilt rationale
#    (regression: string-append left "Thesis intact" in rationale after downgrade)
# ===========================================================================
def test_enforce_consistency_weak_quant_rebuilds_rationale():
    data = {
        "action": "HOLD",
        "severity": "MEDIUM",
        "rationale": "Thesis intact (quant 0.40) | P&L: +5.0%",
        "quant_score": 0.40,
        "thesis_status": "INTACT",
        "pnl_pct": 5.0,
        "held_days": 120,
    }
    result = _enforce_field_consistency(data)
    assert result["thesis_status"] == "WEAKENED"
    # The rebuilt rationale must reflect the corrected thesis status.
    assert "weakened" in result["rationale"].lower()
    # The old stale prefix must NOT survive verbatim in the rebuilt rationale.
    assert "Thesis intact (quant 0.40)" not in result["rationale"]


# ===========================================================================
# 7. _check_data_quality_flags — disagree_excluded source → LOW quality
#    (regression: ICICIBANK/KOTAKBANK/JIOFIN received confident HOLD verdicts
#    despite YoY data disputes flagged in financial layer — 2026-05-06)
# ===========================================================================
class TestMonitoringDataQualityGate:

    def test_low_confidence_data_is_detected(self):
        """When data layer flags YoY as unresolved, is_low_quality must be True."""
        financials = {
            "symbol": "ICICIBANK",
            "current_price": 1351,
            "yoy_confidence": "LOW",
            "yoy_source": "disagree_excluded",
            "exclude_from_recommendations": True,
            "revenue_yoy_pct": -19.2,
        }
        is_low, yoy_conf = _check_data_quality_flags(financials)
        assert is_low is True, (
            "ICICIBANK: data quality flag not detected — same architectural bug as 2026-05-06"
        )
        assert yoy_conf == "LOW"

    def test_high_confidence_data_passes_through(self):
        """Clean data must not be flagged as low quality."""
        financials = {
            "symbol": "TESTCO",
            "current_price": 100,
            "yoy_confidence": "HIGH",
            "yoy_source": "BSE_FILING",
            "exclude_from_recommendations": False,
            "revenue_yoy_pct": 12,
        }
        is_low, yoy_conf = _check_data_quality_flags(financials)
        assert is_low is False
        assert yoy_conf == "HIGH"

    def test_exclude_flag_alone_triggers_low_quality(self):
        """exclude_from_recommendations=True is sufficient even without yoy_confidence."""
        financials = {"exclude_from_recommendations": True}
        is_low, _ = _check_data_quality_flags(financials)
        assert is_low is True

    def test_standalone_only_source_triggers_low_quality(self):
        """standalone_only is an unreliable source for holding companies."""
        financials = {
            "yoy_source": "standalone_only",
            "yoy_confidence": "LOW",
            "exclude_from_recommendations": True,
        }
        is_low, _ = _check_data_quality_flags(financials)
        assert is_low is True

    def test_no_data_source_triggers_low_quality(self):
        """no_data means we have nothing — must flag as low quality."""
        financials = {
            "yoy_source": "no_data",
            "yoy_confidence": "NONE",
            "exclude_from_recommendations": True,
        }
        is_low, _ = _check_data_quality_flags(financials)
        assert is_low is True

    def test_empty_financials_does_not_flag_low_quality(self):
        """Missing flags default to clean — do not block when data provider is absent."""
        is_low, yoy_conf = _check_data_quality_flags({})
        assert is_low is False
        assert yoy_conf == ""

    @pytest.mark.parametrize("symbol", ["ICICIBANK", "KOTAKBANK", "JIOFIN"])
    def test_real_world_bank_holdings_with_data_disputes(self, symbol):
        """
        The three holdings that triggered this fix on 2026-05-06.
        When the financial layer marks YoY as disputed, the data quality
        gate must detect it — preventing a confident HOLD on bad numbers.
        """
        financials = {
            "symbol": symbol,
            "yoy_confidence": "LOW",
            "yoy_source": "disagree_excluded",
            "exclude_from_recommendations": True,
        }
        is_low, _ = _check_data_quality_flags(financials)
        assert is_low is True, (
            f"{symbol}: data quality gate failed to detect disputed YoY data — "
            "same architectural bug as 2026-05-06 (ICICIBANK/KOTAKBANK/JIOFIN "
            "received confident HOLD verdicts despite flagged data)"
        )
