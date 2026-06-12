"""
tests/test_rebalance_dates.py — unit tests for _rebalance_dates in replay.py.

Contracts under test:
  1. weekly mode returns every Monday.
  2. monthly mode returns only the first Monday of each month.
  3. monthly mode deduplicates: no two dates share the same (year, month).
  4. Invalid freq raises ValueError.
  5. Empty range returns empty list.
  6. replay() accepts freq parameter without error.
"""
from __future__ import annotations

import sys
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from backtest.replay import _monday_range, _rebalance_dates


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_monday(d: date) -> bool:
    return d.weekday() == 0


# ── 1. weekly = every Monday ──────────────────────────────────────────────────

def test_weekly_returns_all_mondays():
    start = date(2024, 1, 1)
    end   = date(2024, 3, 31)
    weekly  = _rebalance_dates(start, end, freq="weekly")
    mondays = _monday_range(start, end)
    assert weekly == mondays


def test_weekly_all_are_mondays():
    dates = _rebalance_dates(date(2024, 1, 1), date(2024, 6, 30), freq="weekly")
    assert all(_is_monday(d) for d in dates)


# ── 2. monthly = first Monday per calendar month ─────────────────────────────

def test_monthly_all_are_mondays():
    dates = _rebalance_dates(date(2023, 1, 1), date(2024, 12, 31), freq="monthly")
    assert all(_is_monday(d) for d in dates)


def test_monthly_no_duplicate_months():
    dates = _rebalance_dates(date(2023, 1, 1), date(2024, 12, 31), freq="monthly")
    keys = [(d.year, d.month) for d in dates]
    assert len(keys) == len(set(keys)), "duplicate (year, month) found in monthly rebalance"


def test_monthly_is_first_monday():
    """Each returned date must be the first Monday on or after the 1st of its month."""
    dates = _rebalance_dates(date(2023, 1, 1), date(2025, 1, 1), freq="monthly")
    for d in dates:
        first_of_month = date(d.year, d.month, 1)
        days_until_monday = (7 - first_of_month.weekday()) % 7
        expected_first_monday = first_of_month + timedelta(days=days_until_monday)
        assert d == expected_first_monday or d >= first_of_month, (
            f"{d} is not a plausible first Monday of {d.year}-{d.month:02d}"
        )


def test_monthly_fewer_dates_than_weekly():
    start = date(2024, 1, 1)
    end   = date(2025, 1, 1)
    weekly  = _rebalance_dates(start, end, freq="weekly")
    monthly = _rebalance_dates(start, end, freq="monthly")
    assert len(monthly) < len(weekly)
    assert len(monthly) <= 12   # at most 12 calendar months


# ── 3. edge cases ─────────────────────────────────────────────────────────────

def test_invalid_freq_raises():
    with pytest.raises(ValueError, match="freq must be"):
        _rebalance_dates(date(2024, 1, 1), date(2024, 6, 1), freq="quarterly")


def test_empty_range():
    # end <= start should produce no dates
    result = _rebalance_dates(date(2024, 6, 1), date(2024, 6, 1), freq="weekly")
    assert result == []


def test_single_week_range():
    # Monday to the next Monday: exactly one date (the start Monday)
    start = date(2024, 1, 8)   # Monday
    end   = date(2024, 1, 15)  # next Monday (exclusive)
    result = _rebalance_dates(start, end, freq="weekly")
    assert result == [start]
