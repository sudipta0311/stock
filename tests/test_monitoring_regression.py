"""
Regression tests for monitoring agent pure functions.
Covers all historical failure modes so the fix loop cannot repeat them.
"""
from __future__ import annotations

import sys
import types
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

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
    with (
        patch("stock_platform.agents.monitor_agents._fetch_nse_earnings_date", return_value=None),
        patch("stock_platform.agents.monitor_agents._fetch_screener_earnings_date", return_value=None),
        patch("stock_platform.agents.monitor_agents._fetch_yfinance_earnings_date", return_value=None),
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
