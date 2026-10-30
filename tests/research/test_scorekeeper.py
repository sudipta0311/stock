from __future__ import annotations

from datetime import date
from decimal import Decimal

from wealth.db.models import PaperPosition
from wealth.research.scorekeeper import (
    _summary_text,
    _track_snapshot,
    excess_return_pct,
)


def _position(
    dossier_id: str,
    *,
    confidence: str,
    excess: Decimal,
) -> PaperPosition:
    return PaperPosition(
        week="2026-W44",
        market="IN",
        dossier_id=dossier_id,
        security_id=None,
        symbol=dossier_id,
        confidence=confidence,
        currency="INR",
        notional=Decimal("100000"),
        entry_date=date(2026, 10, 1),
        entry_price=Decimal("100"),
        current_price=Decimal("100"),
        return_pct=Decimal("0"),
        benchmark_symbol="NIFTYBEES",
        benchmark_status="available",
        benchmark_entry_price=Decimal("100"),
        benchmark_current_price=Decimal("100"),
        benchmark_return_pct=Decimal("0"),
        excess_return_pct=excess,
        red_team_falsifiers=[],
    )


def test_worked_example_tracks_hand_computed_excess_returns() -> None:
    excess = [
        excess_return_pct(Decimal("100"), Decimal("120"), Decimal("200"), Decimal("220")),
        excess_return_pct(Decimal("100"), Decimal("90"), Decimal("200"), Decimal("190")),
        excess_return_pct(Decimal("100"), Decimal("110"), Decimal("200"), Decimal("216")),
    ]
    positions = [
        _position(f"synthetic-{index}", confidence="MEDIUM", excess=value)
        for index, value in enumerate(excess)
    ]

    snapshot = _track_snapshot(positions, "2026-W44", router=None, run_ctx=None)

    india = snapshot["markets"]["IN"]
    assert excess == [Decimal("10.00"), Decimal("-5.00"), Decimal("2.00")]
    assert india["ahead_count"] == 2
    assert india["avg_excess_return_pct"] == "2.333333333333333333333333333"
    assert india["summary"] == (
        "Stock ideas since 2026-10-01: +2.3% vs Nifty 50 (3 ideas, 2 ahead)."
    )


def test_track_record_calibration_is_insufficient_before_ten_ideas() -> None:
    positions = [
        _position(f"synthetic-{index}", confidence="LOW", excess=Decimal("1")) for index in range(9)
    ]

    snapshot = _track_snapshot(positions, "2026-W44", router=None, run_ctx=None)

    low = snapshot["calibration"]["IN"]["LOW"]
    assert low == {
        "status": "insufficient_data",
        "sample_count": 9,
        "hit_rate_pct": None,
    }


def test_track_record_summary_format_uses_paise_independent_decimal_rounding() -> None:
    assert (
        _summary_text(
            "IN",
            "Nifty 50",
            date(2026, 1, 2),
            Decimal("-0.05"),
            3,
            1,
        )
        == "Stock ideas since 2026-01-02: -0.1% vs Nifty 50 (3 ideas, 1 ahead)."
    )
