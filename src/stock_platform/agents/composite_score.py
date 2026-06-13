"""
composite_score.py — cross-sectional composite score (Quality + Valuation + Momentum).

Replaces the pure quality_score + sector-multiplier ranking used in the original
replay.py. Computed cross-sectionally over the full replay universe each week so
each stock is ranked relative to its peers rather than against absolute thresholds.

Factor formulas (Task 1 spec)
------------------------------
Quality   = 0.30*zs(roce_pct) + 0.25*zs(revenue_growth_pct)
            + 0.20*zs(−debt_to_equity) + 0.15*zs(promoter_holding)
            + 0.10*(1 if eps > 0 else 0)
            Missing terms: drop and renormalize remaining weights for that symbol.
            All terms missing → quality_pct = 0.50 (neutral).

Valuation = 0.60*zs(earnings_yield) + 0.40*pe_discount_score
            earnings_yield    = eps_ttm / price  (skipped if eps_ttm <= 0)
            pe_discount_score = clamp((pe_5y_median − pe_current) / pe_5y_median, −1, +1)
                                (0.0 when no PE history; field: financials["pe_5y_median"])
            When earnings_yield coverage < 30%, valuation weight redistributed to quality.

Momentum  = financials["price_momentum_6m"]  (pre-computed by caller as
            price(t−21 trading days) / price(t−126 trading days) − 1).
            Absent → momentum_pct = 0.50 (neutral).

Each factor's raw value is cross-sectionally percentile-ranked (sector-neutral when ≥ 5
peers share a sector, otherwise universe-wide).  Rankings use average-rank tie-breaking
and ±3σ winsorization.

Weight file: rules/composite_weights.yaml (fallback 0.40 / 0.30 / 0.30).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_WEIGHT_DEFAULTS: dict[str, float] = {
    "quality":   0.40,
    "valuation": 0.30,
    "momentum":  0.30,
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


def _zs_cross_sectional(values: list[float | None]) -> list[float | None]:
    """
    Cross-sectional z-score with ±3σ winsorization.
    None inputs → None output.
    With fewer than 2 non-None values, present items receive 0.0 (no variation to measure).
    """
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    result: list[float | None] = [None] * len(values)

    if not present:
        return result
    if len(present) == 1:
        result[present[0][0]] = 0.0
        return result

    vals = [v for _, v in present]
    mean = sum(vals) / len(vals)
    var  = sum((v - mean) ** 2 for v in vals) / len(vals)
    std  = var ** 0.5 or 1.0

    lo, hi = mean - 3 * std, mean + 3 * std
    winsorized = [max(lo, min(hi, v)) for v in vals]

    wmean = sum(winsorized) / len(winsorized)
    wvar  = sum((v - wmean) ** 2 for v in winsorized) / len(winsorized)
    wstd  = wvar ** 0.5 or 1.0

    zs = [(v - wmean) / wstd for v in winsorized]
    for local_i, (global_i, _) in enumerate(present):
        result[global_i] = zs[local_i]
    return result


def _pct_rank(values: list[float | None]) -> list[float]:
    """
    Compute [0,1] percentile ranks for a list with optional None entries.
    None entries receive 0.5 (neutral/unknown).
    Ties are resolved by average rank.
    Values are winsorised at ±3σ before ranking.
    """
    n = len(values)
    present_idx  = [i for i, v in enumerate(values) if v is not None]
    present_vals = [values[i] for i in present_idx]  # type: ignore[index]

    ranks = [0.5] * n
    m = len(present_vals)
    if m < 2:
        return ranks

    mean = sum(present_vals) / m
    variance = sum((v - mean) ** 2 for v in present_vals) / m
    std = variance ** 0.5 or 1.0
    lo, hi = mean - 3 * std, mean + 3 * std
    winsorised = [max(lo, min(hi, v)) for v in present_vals]

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

    Each candidate must supply:
      - "financials" (or "live_financials"): dict with fundamental/price fields
      - "sector": str (optional — defaults to "Unknown")

    See module docstring for factor formulas.
    """
    if not candidates:
        return candidates

    w = dict(weights or _COMPOSITE_WEIGHTS)
    w_total = sum(w.values())
    if w_total <= 0:
        w = dict(_WEIGHT_DEFAULTS)
        w_total = sum(w.values())
    w = {k: v / w_total for k, v in w.items()}

    n = len(candidates)
    fin_all = [c.get("financials") or c.get("live_financials") or {} for c in candidates]
    sectors  = [str(c.get("sector") or "Unknown") for c in candidates]

    # ── Quality: 5-component cross-sectional z-score blend ───────────────────
    # Higher is better: roce, revenue_growth, promoter_holding, eps>0
    # Lower is better:  debt_to_equity (inverted via negation before z-scoring)
    roce_zs = _zs_cross_sectional([_safe_float(f.get("roce_pct")) for f in fin_all])
    revg_zs = _zs_cross_sectional([_safe_float(f.get("revenue_growth_pct")) for f in fin_all])
    de_vals  = [_safe_float(f.get("debt_to_equity")) for f in fin_all]
    de_zs    = _zs_cross_sectional([-v if v is not None else None for v in de_vals])
    prom_zs  = _zs_cross_sectional([_safe_float(f.get("promoter_holding")) for f in fin_all])

    q_raws: list[float | None] = []
    for i, fin in enumerate(fin_all):
        terms: list[tuple[float, float]] = []
        if roce_zs[i]  is not None: terms.append((0.30, roce_zs[i]))   # type: ignore[arg-type]
        if revg_zs[i]  is not None: terms.append((0.25, revg_zs[i]))   # type: ignore[arg-type]
        if de_zs[i]    is not None: terms.append((0.20, de_zs[i]))     # type: ignore[arg-type]
        if prom_zs[i]  is not None: terms.append((0.15, prom_zs[i]))   # type: ignore[arg-type]
        eps_val = _safe_float(fin.get("eps"))
        if eps_val is not None:
            terms.append((0.10, 1.0 if eps_val > 0 else 0.0))
        if not terms:
            q_raws.append(None)
        else:
            total_w = sum(wt for wt, _ in terms)
            q_raws.append(sum(wt / total_w * v for wt, v in terms))

    # ── Valuation: earnings_yield (z-scored) + pe_discount_score (clamped) ──
    ey_raws:      list[float | None] = []
    pe_disc_raws: list[float]        = []

    for fin in fin_all:
        eps_ttm = _safe_float(fin.get("eps"))
        price   = _safe_float(fin.get("current_price") or fin.get("currentPrice"))
        if eps_ttm is not None and eps_ttm > 0 and price is not None and price > 0:
            ey_raws.append(eps_ttm / price)
        else:
            ey_raws.append(None)

        # PE discount: positive = stock is cheaper than its historical median
        pe_5y  = _safe_float(fin.get("pe_5y_median"))
        pe_cur = _safe_float(fin.get("pe_current") or fin.get("pe_trailing"))
        if pe_5y is not None and pe_cur is not None and pe_5y > 0:
            raw_disc = (pe_5y - pe_cur) / pe_5y
            pe_disc_raws.append(max(-1.0, min(1.0, raw_disc)))
        else:
            pe_disc_raws.append(0.0)  # neutral when no 5y PE history

    ey_zs = _zs_cross_sectional(ey_raws)

    v_raws: list[float | None] = []
    for i in range(n):
        terms_v: list[tuple[float, float]] = []
        if ey_zs[i] is not None:
            terms_v.append((0.60, ey_zs[i]))  # type: ignore[arg-type]
        terms_v.append((0.40, pe_disc_raws[i]))  # always present (may be 0.0)
        total_v = sum(wt for wt, _ in terms_v)
        v_raws.append(sum(wt / total_v * v for wt, v in terms_v))

    # ── Momentum: price-based return pre-computed by caller ──────────────────
    # Caller sets financials["price_momentum_6m"] = price(t-21d)/price(t-126d)-1.
    # Absent → raw = None → percentile = 0.50 neutral.
    m_raws: list[float | None] = [
        _safe_float(fin.get("price_momentum_6m")) for fin in fin_all
    ]

    # Redistribute valuation weight to quality when earnings_yield is too sparse
    ey_coverage = sum(1 for v in ey_raws if v is not None) / max(n, 1)
    if ey_coverage < 0.30:
        w_q_eff = w.get("quality", 0.40) + w.get("valuation", 0.30)
        w_v_eff = 0.0
        w_m_eff = w.get("momentum", 0.30)
    else:
        w_q_eff = w.get("quality", 0.40)
        w_v_eff = w.get("valuation", 0.30)
        w_m_eff = w.get("momentum", 0.30)

    w_sum = (w_q_eff + w_v_eff + w_m_eff) or 1.0
    w_q_eff /= w_sum
    w_v_eff /= w_sum
    w_m_eff /= w_sum

    q_pct = _sector_neutral_pct(q_raws, sectors)
    v_pct = _sector_neutral_pct(v_raws, sectors)
    m_pct = _sector_neutral_pct(m_raws, sectors)

    result: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        composite = round(w_q_eff * q_pct[i] + w_v_eff * v_pct[i] + w_m_eff * m_pct[i], 4)
        result.append({
            **c,
            "composite_score": composite,
            "quality_pct":     round(q_pct[i], 4),
            "valuation_pct":   round(v_pct[i], 4),
            "momentum_pct":    round(m_pct[i], 4),
        })

    return result
