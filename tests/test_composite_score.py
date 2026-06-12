"""
tests/test_composite_score.py — unit tests for the cross-sectional composite scorer.

Contracts under test:
  1. _pct_rank returns [0,1] values; None inputs get 0.5.
  2. With a clear rank order, higher-value stocks get higher percentile.
  3. Sector-neutral ranking ranks within sector when ≥5 peers; falls back to universe.
  4. compute_composite_scores augments every candidate with required keys.
  5. When PE data is absent (<30% coverage), valuation weight is redistributed.
  6. Single-candidate universe returns 0.5 for all pct components.
  7. evidence_scoring tech_score uses momentum-positive logic (far from low = good).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_platform.agents.composite_score import (
    _pct_rank,
    _sector_neutral_pct,
    compute_composite_scores,
)


# ── 1. _pct_rank basic contract ───────────────────────────────────────────────

def test_pct_rank_none_gets_neutral():
    ranks = _pct_rank([None, None, None])
    assert all(r == 0.5 for r in ranks)


def test_pct_rank_order():
    """Higher values get higher percentile ranks."""
    ranks = _pct_rank([10.0, 50.0, 90.0])
    assert ranks[0] < ranks[1] < ranks[2]
    assert ranks[0] == pytest.approx(0.0)
    assert ranks[2] == pytest.approx(1.0)


def test_pct_rank_none_mixed():
    """None entries get 0.5; non-None entries are ranked among themselves."""
    ranks = _pct_rank([None, 10.0, 90.0])
    assert ranks[0] == pytest.approx(0.5)
    assert ranks[1] == pytest.approx(0.0)
    assert ranks[2] == pytest.approx(1.0)


def test_pct_rank_ties_get_average():
    """Tied values get the same average rank."""
    ranks = _pct_rank([5.0, 5.0, 10.0])
    assert ranks[0] == ranks[1]
    assert ranks[2] > ranks[0]


def test_pct_rank_winsorisation():
    """Extreme outlier should be pulled in by winsorisation without distorting the others."""
    normal_ranks  = _pct_rank([1.0, 2.0, 3.0, 4.0, 5.0])
    extreme_ranks = _pct_rank([1.0, 2.0, 3.0, 4.0, 1_000_000.0])
    # After winsorisation, ranks of the normal values should remain similar.
    for i in range(4):
        assert abs(extreme_ranks[i] - normal_ranks[i]) < 0.25


# ── 2. _sector_neutral_pct ────────────────────────────────────────────────────

def test_sector_neutral_uses_universe_when_small_sector():
    """Sector with <5 members uses universe rank."""
    values  = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    sectors = ["A", "A", "A", "B", "B", "B"]   # 3 each — below threshold
    pcts = _sector_neutral_pct(values, sectors)
    # All ranked together, so first < last
    assert pcts[0] < pcts[-1]


def test_sector_neutral_ranks_within_sector():
    """Sector with ≥5 members: ranking is within sector, not universe."""
    # 5 stocks in sector "X" with values 1-5 and 1 stock in "Y" with value 100.
    # If sector-neutral, the "Y" stock ranks alone (universe rank) = 0.5.
    # Within sector X: stock with value 5 gets rank 1.0, value 1 gets 0.0.
    values  = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    sectors = ["X", "X", "X", "X", "X", "Y"]
    pcts = _sector_neutral_pct(values, sectors)
    # Sector X highest (index 4) should be 1.0
    assert pcts[4] == pytest.approx(1.0)
    # Sector X lowest (index 0) should be 0.0
    assert pcts[0] == pytest.approx(0.0)
    # "Y" stock (index 5) is alone — universe rank of one = 0.5
    assert pcts[5] == pytest.approx(0.5)


# ── 3. compute_composite_scores ───────────────────────────────────────────────

def _make_candidates(specs: list[dict]) -> list[dict]:
    return [
        {
            "symbol":     s.get("symbol", f"SYM{i}"),
            "financials": s.get("fin", {}),
            "sector":     s.get("sector", "Unknown"),
        }
        for i, s in enumerate(specs)
    ]


def test_compute_composite_required_keys():
    """compute_composite_scores adds the four required keys to every candidate."""
    candidates = _make_candidates([
        {"fin": {"roce_pct": 15.0, "debt_to_equity": 0.5, "revenue_growth_pct": 10.0}},
        {"fin": {"roce_pct": 5.0,  "debt_to_equity": 2.0, "revenue_growth_pct": -5.0}},
    ])
    result = compute_composite_scores(candidates)
    for r in result:
        assert "composite_score"  in r
        assert "quality_pct"      in r
        assert "valuation_pct"    in r
        assert "momentum_pct"     in r
        assert 0.0 <= r["composite_score"] <= 1.0
        assert 0.0 <= r["quality_pct"]     <= 1.0
        assert 0.0 <= r["valuation_pct"]   <= 1.0
        assert 0.0 <= r["momentum_pct"]    <= 1.0


def test_compute_composite_better_stock_ranks_higher():
    """A fundamentally stronger stock should receive a higher composite score."""
    candidates = _make_candidates([
        {"symbol": "GOOD", "fin": {"roce_pct": 25.0, "debt_to_equity": 0.2, "revenue_growth_pct": 20.0}},
        {"symbol": "BAD",  "fin": {"roce_pct": 2.0,  "debt_to_equity": 3.0, "revenue_growth_pct": -10.0}},
    ])
    result = compute_composite_scores(candidates)
    by_sym = {r["symbol"]: r for r in result}
    assert by_sym["GOOD"]["composite_score"] > by_sym["BAD"]["composite_score"]


def test_compute_composite_empty_input():
    assert compute_composite_scores([]) == []


def test_compute_composite_single_candidate():
    """Single candidate — all pct fields should be 0.5 (neutral)."""
    candidates = _make_candidates([
        {"fin": {"roce_pct": 20.0, "debt_to_equity": 0.3, "revenue_growth_pct": 12.0}},
    ])
    result = compute_composite_scores(candidates)
    assert len(result) == 1
    assert result[0]["quality_pct"]  == pytest.approx(0.5)
    assert result[0]["momentum_pct"] == pytest.approx(0.5)


def test_compute_composite_no_pe_data_redistributes_weight():
    """When no PE data is present, valuation weight is zero but composite still works."""
    candidates = _make_candidates([
        {"fin": {"roce_pct": 20.0, "debt_to_equity": 0.3, "revenue_growth_pct": 10.0}},
        {"fin": {"roce_pct": 5.0,  "debt_to_equity": 2.0, "revenue_growth_pct": -5.0}},
    ])
    # No pe_trailing in either candidate — valuation_pct should both be 0.5 (neutral)
    result = compute_composite_scores(candidates)
    for r in result:
        assert r["valuation_pct"] == pytest.approx(0.5)
    # But quality and momentum should still differentiate
    by_sym = {r["symbol"]: r for r in result}
    assert by_sym["SYM0"]["quality_pct"] > by_sym["SYM1"]["quality_pct"]


def test_compute_composite_preserves_original_fields():
    """compute_composite_scores should not drop any pre-existing fields."""
    candidates = [
        {
            "symbol":        "TESTCO",
            "quality_score": 0.8,
            "financials":    {"roce_pct": 20.0},
            "sector":        "IT",
            "extra_field":   "kept",
        }
    ]
    result = compute_composite_scores(candidates)
    assert result[0]["extra_field"] == "kept"
    assert result[0]["quality_score"] == pytest.approx(0.8)


# ── 4. evidence_scoring tech_score fix ───────────────────────────────────────

def test_evidence_tech_score_momentum_positive():
    """pct_from_52w_low > 60 should get highest tech_score (0.10)."""
    from stock_platform.utils.evidence_scoring import compute_evidence_strength

    result_near_high = compute_evidence_strength(
        fin_data={},
        pe_context={},
        val_reliability={},
        tech_signals={"pct_from_52w_low": 70.0},
        sector_signal={},
    )
    result_near_low = compute_evidence_strength(
        fin_data={},
        pe_context={},
        val_reliability={},
        tech_signals={"pct_from_52w_low": 10.0},
        sector_signal={},
    )
    # Far from low (70%) should score higher than near low (10%)
    assert result_near_high["components"]["technical"] > result_near_low["components"]["technical"]
    assert result_near_high["components"]["technical"] == pytest.approx(0.10)
    assert result_near_low["components"]["technical"]  == pytest.approx(0.02)


def test_evidence_tech_score_moderate():
    """pct_from_52w_low between 30-60 should get 0.05."""
    from stock_platform.utils.evidence_scoring import compute_evidence_strength
    result = compute_evidence_strength(
        fin_data={}, pe_context={}, val_reliability={},
        tech_signals={"pct_from_52w_low": 45.0},
        sector_signal={},
    )
    assert result["components"]["technical"] == pytest.approx(0.05)
