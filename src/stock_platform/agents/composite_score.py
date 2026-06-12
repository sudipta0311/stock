"""
composite_score.py — cross-sectional composite score (Quality + Valuation + Momentum).

Replaces the pure quality_score + sector-multiplier ranking used in the original
replay.py. Computed cross-sectionally over the full replay universe each week so
each stock is ranked relative to its peers rather than against absolute thresholds.

Pipeline per week
-----------------
1. Extract per-factor raw values from each candidate's financials.
2. Winsorise at ±3σ across the universe.
3. Convert to [0,1] percentile rank — sector-neutral if ≥5 peers share a sector.
4. Combine quality_pct, valuation_pct, momentum_pct with YAML-loaded weights.
   When PE data coverage is <30%, valuation weight is redistributed to quality.

Weight file: rules/composite_weights.yaml (falls back to 50/25/25 defaults).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_WEIGHT_DEFAULTS: dict[str, float] = {
    "quality":   0.50,
    "valuation": 0.25,
    "momentum":  0.25,
}

_SECTOR_NEUTRAL_MIN = 5  # min peers in sector to use sector-neutral ranking


def _load_composite_weights() -> dict[str, float]:
    yaml_path = Path(__file__).resolve().parents[4] / "rules" / "composite_weights.yaml"
    if not yaml_path.exists():
        return dict(_WEIGHT_DEFAULTS)
    try:
        import yaml
        with yaml_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        w = doc.get("weights", {})
        loaded = {
            "quality":   float(w.get("quality",   _WEIGHT_DEFAULTS["quality"])),
            "valuation": float(w.get("valuation", _WEIGHT_DEFAULTS["valuation"])),
            "momentum":  float(w.get("momentum",  _WEIGHT_DEFAULTS["momentum"])),
        }
        _log.info(
            "composite_weights loaded: version=%s  weights=%s",
            doc.get("version", "?"), loaded,
        )
        return loaded
    except Exception as exc:
        _log.warning("composite_weights: failed to load (%r) — using defaults", exc)
        return dict(_WEIGHT_DEFAULTS)


_COMPOSITE_WEIGHTS = _load_composite_weights()


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if f != f else f  # reject NaN
    except (TypeError, ValueError):
        return None


def _pct_rank(values: list[float | None]) -> list[float]:
    """
    Compute [0,1] percentile ranks for a list with optional None entries.
    None entries receive 0.5 (neutral/unknown).
    Ties are resolved by average rank.
    Values are winsorised at ±3σ before ranking.
    """
    n = len(values)
    present_idx = [i for i, v in enumerate(values) if v is not None]
    present_vals = [values[i] for i in present_idx]  # type: ignore[index]

    ranks = [0.5] * n
    m = len(present_vals)
    if m < 2:
        return ranks

    # Winsorise at ±3σ
    mean = sum(present_vals) / m
    variance = sum((v - mean) ** 2 for v in present_vals) / m
    std = variance ** 0.5 or 1.0
    lo, hi = mean - 3 * std, mean + 3 * std
    winsorised = [max(lo, min(hi, v)) for v in present_vals]

    # Sort-based rank with average tie-breaking
    order = sorted(range(m), key=lambda i: winsorised[i])
    rank_vals = [0.0] * m
    i = 0
    while i < m:
        j = i
        while j < m - 1 and winsorised[order[j]] == winsorised[order[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2
        for k in range(i, j + 1):
            rank_vals[order[k]] = avg_rank
        i = j + 1

    denom = m - 1
    for local_idx, global_idx in enumerate(present_idx):
        ranks[global_idx] = round(rank_vals[local_idx] / denom, 4) if denom > 0 else 0.5

    return ranks


def _sector_neutral_pct(
    values: list[float | None],
    sectors: list[str],
) -> list[float]:
    """
    Rank within sector when sector has ≥ _SECTOR_NEUTRAL_MIN members,
    otherwise use universe-wide rank.
    """
    n = len(values)

    sector_indices: dict[str, list[int]] = {}
    for i, s in enumerate(sectors):
        sector_indices.setdefault(s, []).append(i)

    # Track which indices were assigned sector-neutral ranks
    sector_assigned: set[int] = set()
    results = [0.5] * n

    for sector, idxs in sector_indices.items():
        if len(idxs) < _SECTOR_NEUTRAL_MIN:
            continue
        sector_vals = [values[i] for i in idxs]
        pcts = _pct_rank(sector_vals)
        for i, p in zip(idxs, pcts):
            results[i] = p
            sector_assigned.add(i)

    # Universe rank for the remainder
    univ_indices = [i for i in range(n) if i not in sector_assigned]
    if univ_indices:
        univ_vals = [values[i] for i in univ_indices]
        pcts = _pct_rank(univ_vals)
        for i, p in zip(univ_indices, pcts):
            results[i] = p

    return results


def compute_composite_scores(
    candidates: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Augment each candidate with composite_score, quality_pct, valuation_pct,
    and momentum_pct.

    Each candidate must have:
      - "financials" (or "live_financials"): dict with fundamental fields
      - "sector": str (optional — defaults to "Unknown")

    Factor definitions
    ------------------
    Quality   : weighted blend of ROCE (60%) and inverse-D/E (40%).
    Valuation : negative PE trailing (lower PE → higher rank); absent → 0.5 neutral.
    Momentum  : revenue_growth_pct; absent → 0.5 neutral.

    When PE data covers <30% of candidates, valuation weight is redistributed
    to quality automatically so the score stays meaningful in backtest mode.

    Returns a new list with the original dict augmented by four new keys.
    """
    if not candidates:
        return candidates

    w = dict(weights or _COMPOSITE_WEIGHTS)
    w_total = sum(w.values())
    if w_total <= 0:
        w = dict(_WEIGHT_DEFAULTS)
        w_total = sum(w.values())
    w = {k: v / w_total for k, v in w.items()}

    sectors: list[str] = []
    q_raws: list[float | None] = []
    v_raws: list[float | None] = []
    m_raws: list[float | None] = []

    for c in candidates:
        fin = c.get("financials") or c.get("live_financials") or {}
        sectors.append(str(c.get("sector") or "Unknown"))

        # Quality raw: ROCE × 0.6 + (1 / (D/E + 0.01)) × 0.4
        roce = _safe_float(fin.get("roce_pct"))
        de   = _safe_float(fin.get("debt_to_equity"))
        de_inv = (1.0 / (de + 0.01)) if de is not None else None
        if roce is not None and de_inv is not None:
            q_raws.append(roce * 0.6 + de_inv * 0.4)
        elif roce is not None:
            q_raws.append(roce)
        elif de_inv is not None:
            q_raws.append(de_inv)
        else:
            q_raws.append(None)

        # Valuation raw: −PE (lower PE = better = higher rank)
        pe = _safe_float(fin.get("pe_trailing"))
        v_raws.append(-pe if (pe is not None and pe > 0) else None)

        # Momentum raw: revenue growth %
        m_raws.append(_safe_float(fin.get("revenue_growth_pct")))

    # Redistribute valuation weight when coverage is sparse (backtest mode)
    v_present = sum(1 for v in v_raws if v is not None)
    if v_present < len(v_raws) * 0.30:
        w_q = w.get("quality", 0.5) + w.get("valuation", 0.25)
        w_v = 0.0
        w_m = w.get("momentum", 0.25)
    else:
        w_q = w.get("quality", 0.5)
        w_v = w.get("valuation", 0.25)
        w_m = w.get("momentum", 0.25)

    w_sum = w_q + w_v + w_m or 1.0
    w_q /= w_sum
    w_v /= w_sum
    w_m /= w_sum

    q_pct = _sector_neutral_pct(q_raws, sectors)
    v_pct = _sector_neutral_pct(v_raws, sectors)
    m_pct = _sector_neutral_pct(m_raws, sectors)

    result: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        composite = round(w_q * q_pct[i] + w_v * v_pct[i] + w_m * m_pct[i], 4)
        result.append({
            **c,
            "composite_score": composite,
            "quality_pct":     round(q_pct[i], 4),
            "valuation_pct":   round(v_pct[i], 4),
            "momentum_pct":    round(m_pct[i], 4),
        })

    return result
