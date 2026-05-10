from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from stock_platform.utils.risk_profiles import get_risk_config
from stock_platform.utils.rules import clamp

_log = logging.getLogger(__name__)

# The five fields that feed into compute_quality_score; used for provenance counting.
_SCORED_FIELDS = ("roce_pct", "eps", "revenue_growth_pct", "promoter_holding", "debt_to_equity")

# ── Calibrated weights (loaded from rules/quality_weights.yaml at import time) ──
_DEFAULT_WEIGHTS = {
    "roce":           0.25,
    "eps":            0.25,
    "revenue_growth": 0.20,
    "promoter":       0.15,
    "debt_equity":    0.15,
}

def _load_weights() -> dict[str, float]:
    """
    Load quality-score weights from rules/quality_weights.yaml (repo root).
    Falls back to hardcoded defaults if the file is absent or malformed.
    Logs the version string on first successful load.
    """
    yaml_path = Path(__file__).resolve().parents[4] / "rules" / "quality_weights.yaml"
    if not yaml_path.exists():
        return dict(_DEFAULT_WEIGHTS)
    try:
        import yaml
        with yaml_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        w = doc.get("weights", {})
        loaded = {
            "roce":           float(w.get("roce",           _DEFAULT_WEIGHTS["roce"])),
            "eps":            float(w.get("eps",            _DEFAULT_WEIGHTS["eps"])),
            "revenue_growth": float(w.get("revenue_growth", _DEFAULT_WEIGHTS["revenue_growth"])),
            "promoter":       float(w.get("promoter",       _DEFAULT_WEIGHTS["promoter"])),
            "debt_equity":    float(w.get("debt_equity",    _DEFAULT_WEIGHTS["debt_equity"])),
        }
        _log.info(
            "quality_weights loaded: version=%s  weights=%s",
            doc.get("version", "?"), loaded,
        )
        return loaded
    except Exception as exc:
        _log.warning("quality_weights: failed to load %s (%r) — using defaults", yaml_path, exc)
        return dict(_DEFAULT_WEIGHTS)

_WEIGHTS = _load_weights()


def _data_quality_label(fin_data: dict[str, Any]) -> str:
    """
    Derive a data-quality label from the _data_provenance sub-dict embedded by
    screener_fetcher.  Falls back to field-presence heuristic when provenance is absent.

    Returns "CLEAN" (0 defaults), "PARTIAL" (1-2 defaults), or "DEGRADED" (3+ defaults).
    """
    provenance: dict[str, str] = fin_data.get("_data_provenance") or {}

    if provenance:
        defaults = sum(
            1 for f in _SCORED_FIELDS if provenance.get(f, "DEFAULT") == "DEFAULT"
        )
    else:
        # No provenance dict — treat absent fields as DEFAULT.
        defaults = sum(1 for f in _SCORED_FIELDS if fin_data.get(f) is None)

    if defaults == 0:
        return "CLEAN"
    if defaults <= 2:
        return "PARTIAL"
    return "DEGRADED"


def compute_quality_score(
    symbol: str,
    fin_data: dict[str, Any],
    _unused: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """
    Quality score based on standardised fundamentals.

    Returns ``(score, data_quality)`` where:

    * score       — float in [0.0, 1.0] (0.5 for fully unknown data)
    * data_quality — ``"CLEAN"`` (all 5 metrics fetched), ``"PARTIAL"`` (1-2
                     defaults), or ``"DEGRADED"`` (3+ defaults / no data).

    The ``_data_provenance`` sub-dict inside *fin_data* (populated by
    ``screener_fetcher.fetch_screener_data``) drives the quality label; field
    presence is used as a fallback when provenance is absent.
    """
    if not fin_data:
        _log.warning("No data for %s - score=0.5 (unknown)", symbol)
        return 0.5, "DEGRADED"

    scores: list[float] = []
    weights: list[float] = []

    roce = fin_data.get("roce_pct")
    if roce is not None:
        scores.append(
            1.0 if roce > 18 else
            0.6 if roce > 10 else
            0.0 if roce > 0 else
            -0.5
        )
        weights.append(_WEIGHTS["roce"])

    eps = fin_data.get("eps")
    if eps is not None:
        scores.append(1.0 if eps > 0 else 0.0)
        weights.append(_WEIGHTS["eps"])

    revenue_growth = fin_data.get("revenue_growth_pct")
    if revenue_growth is not None:
        scores.append(
            1.0 if revenue_growth > 15 else
            0.7 if revenue_growth > 8 else
            0.3 if revenue_growth > 0 else
            0.0
        )
        weights.append(_WEIGHTS["revenue_growth"])

    promoter = fin_data.get("promoter_holding")
    if promoter is not None:
        scores.append(
            1.0 if promoter > 50 else
            0.7 if promoter > 35 else
            0.3
        )
        weights.append(_WEIGHTS["promoter"])

    debt_to_equity = fin_data.get("debt_to_equity")
    if debt_to_equity is not None:
        scores.append(
            1.0 if debt_to_equity < 0.5 else
            0.5 if debt_to_equity < 1.0 else
            0.1 if debt_to_equity < 2.0 else
            0.0
        )
        weights.append(_WEIGHTS["debt_equity"])

    if not scores:
        _log.warning("No usable data for %s - score=0.5 (unknown)", symbol)
        return 0.5, "DEGRADED"

    raw_score  = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    final_score = round(clamp(raw_score, 0.0, 1.0), 2)

    if eps is not None and eps < 0:
        final_score = min(final_score, 0.35)
        _log.debug("%s: negative EPS — capped quality at 0.35", symbol)

    critical_fields = sum(
        1
        for key in ("roce_pct", "revenue_growth_pct", "debt_to_equity")
        if fin_data.get(key) is not None
    )
    if fin_data.get("symbol_mapped") and critical_fields <= 1:
        final_score = min(final_score, 0.45)
        _log.debug(
            "%s: mapped to %s with sparse post-demerger metrics — capped at 0.45",
            symbol, fin_data.get("symbol"),
        )

    quality_label = _data_quality_label(fin_data)
    return round(final_score, 2), quality_label


# ── Entry signal vocabulary used in assess_timing() ──────────────────────────
_BULLISH_ENTRY_SIGNALS = {"STRONG ENTER", "ACCUMULATE", "SMALL INITIAL"}
_SOFTEN_MAP = {"STRONG ENTER": "ACCUMULATE", "ACCUMULATE": "SMALL INITIAL"}

# Rank for quant signals — higher = more bullish. Used to cap downward only.
_QUANT_RANK = {
    "STRONG ENTER":  4,
    "ACCUMULATE":    3,
    "SMALL INITIAL": 2,
    "WAIT":          1,
    "DO NOT ENTER":  0,
}


def apply_freshness_cap(
    entry_signal: str,
    fin_data: dict[str, Any],
    risk_profile: str = "Balanced",
) -> str:
    """
    Caps quant entry signal based on data freshness and the investor's risk profile.

    Conservative/Balanced: no result date -> WAIT (hard stop).
    Aggressive: no result date -> SMALL INITIAL (data uncertainty tolerated).

    entry_signal uses assess_timing() vocabulary:
        STRONG ENTER | ACCUMULATE | SMALL INITIAL | WAIT | DO NOT ENTER
    """
    if entry_signal not in _BULLISH_ENTRY_SIGNALS:
        return entry_signal

    config      = get_risk_config(risk_profile)
    result_date = fin_data.get("last_result_date")
    days_stale  = fin_data.get("result_days_stale", 999)
    w52_quality = fin_data.get("52w_data_quality", "")

    # No earnings anchor — apply profile cap (may still allow SMALL INITIAL for Aggressive)
    if not result_date or str(result_date).lower() in ("none", ""):
        cap_signal = config.get("quant_cap_no_result_signal", "WAIT")
        if _QUANT_RANK.get(entry_signal, 0) > _QUANT_RANK.get(cap_signal, 0):
            _log.debug(
                "Freshness cap [%s]: no result date — capping '%s' to '%s'",
                risk_profile, entry_signal, cap_signal,
            )
            return cap_signal
        return entry_signal

    # Staleness cap: result is too old for this profile -> soften one tier
    staleness_cap = int(config.get("staleness_cap_days", 90))
    if isinstance(days_stale, (int, float)) and days_stale > staleness_cap:
        softened = _SOFTEN_MAP.get(entry_signal, entry_signal)
        if softened != entry_signal:
            _log.debug(
                "Freshness cap [%s]: %dd > %dd — softening '%s' to '%s'",
                risk_profile, days_stale, staleness_cap, entry_signal, softened,
            )
        return softened

    # Corrupt price data — soften one tier regardless of profile
    if w52_quality in ("DATA_CORRUPT", "UNAVAILABLE"):
        softened = _SOFTEN_MAP.get(entry_signal, entry_signal)
        if softened != entry_signal:
            _log.debug(
                "Freshness cap: 52W %s — softening '%s' to '%s'",
                w52_quality, entry_signal, softened,
            )
        return softened

    return entry_signal
