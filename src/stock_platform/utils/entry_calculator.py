from __future__ import annotations

from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_week52_low(fin_data: dict[str, Any]) -> float | None:
    for key in ("week52_low", "52_week_low", "fiftyTwoWeekLow", "yearLow"):
        value = _as_float(fin_data.get(key))
        if value and value > 0:
            return value
    return None


def calculate_entry_levels(
    symbol: str,
    current_price: float,
    analyst_target: float,
    signal: str,
    quant_score: float,
    fin_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate entry price, entry zone, stop loss, and tranche guidance.
    """
    del symbol  # Reserved for future symbol-specific rules.

    current_price_value = _as_float(current_price)
    if current_price_value is None or current_price_value <= 0:
        return {}

    analyst_target_value = _as_float(analyst_target)
    if analyst_target_value is None or analyst_target_value <= 0:
        analyst_target_value = round(current_price_value * 1.15, 1)

    signal_key = str(signal or "").strip().upper().replace("_", " ")
    quality_score = _as_float(quant_score) or 0.0
    fundamentals = fin_data or {}

    if signal_key in {"STRONG BUY", "STRONG ENTER"}:
        entry_price = round(current_price_value, 1)
        entry_zone_low = round(current_price_value * 0.98, 1)
        entry_zone_high = round(current_price_value * 1.01, 1)
        entry_note = "Enter at market price now."
        discount_pct = 0.0
    elif signal_key in {"BUY", "SMALL INITIAL"}:
        discount_pct = 0.03
        entry_price = round(current_price_value * (1 - discount_pct), 1)
        entry_zone_low = round(current_price_value * 0.96, 1)
        entry_zone_high = round(current_price_value * 1.00, 1)
        entry_note = f"Set a limit order near ₹{entry_price:.0f} (3% below current price)."
    elif signal_key == "ACCUMULATE":
        discount_pct = 0.05
        entry_price = round(current_price_value * (1 - discount_pct), 1)
        entry_zone_low = round(current_price_value * 0.93, 1)
        entry_zone_high = round(current_price_value * 0.97, 1)
        entry_note = (
            f"Wait for a pullback toward ₹{entry_price:.0f}. "
            f"Preferred buy zone: ₹{entry_zone_low:.0f}-₹{entry_zone_high:.0f}."
        )
    else:
        week52_low = _pick_week52_low(fundamentals)
        if week52_low and week52_low > current_price_value * 0.5:
            entry_price = round((current_price_value + week52_low) / 2, 1)
        else:
            entry_price = round(current_price_value * 0.85, 1)
        entry_zone_low = round(entry_price * 0.97, 1)
        entry_zone_high = round(entry_price * 1.03, 1)
        discount_pct = max(0.0, (current_price_value - entry_price) / current_price_value)
        entry_note = (
            f"Do not buy yet. Wait for ₹{entry_price:.0f} "
            f"({discount_pct * 100:.0f}% below current price) and set a price alert."
        )

    if quality_score >= 0.75:
        stop_loss_pct = 0.15
    elif quality_score >= 0.50:
        stop_loss_pct = 0.12
    else:
        stop_loss_pct = 0.08

    stop_loss = round(entry_price * (1 - stop_loss_pct), 1)
    reward = analyst_target_value - entry_price
    risk = entry_price - stop_loss
    rr_ratio = round(reward / risk, 1) if reward > 0 and risk > 0 else 0.0

    if signal_key in {"ACCUMULATE", "BUY", "SMALL INITIAL"}:
        tranche_1_pct = 40
        tranche_2_pct = 35
        tranche_3_pct = 25
        tranche_2_price = round(entry_price * 0.95, 1)
        tranche_3_trigger = "After the next quarterly result confirms the thesis."
    elif signal_key in {"STRONG BUY", "STRONG ENTER"}:
        tranche_1_pct = 60
        tranche_2_pct = 40
        tranche_3_pct = 0
        tranche_2_price = round(current_price_value * 0.97, 1)
        tranche_3_trigger = ""
    else:
        tranche_1_pct = 100
        tranche_2_pct = 0
        tranche_3_pct = 0
        tranche_2_price = entry_price
        tranche_3_trigger = ""

    return {
        "current_price": round(current_price_value, 1),
        "entry_price": entry_price,
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "entry_note": entry_note,
        "stop_loss": stop_loss,
        "stop_loss_pct": round(stop_loss_pct * 100, 0),
        "analyst_target": round(analyst_target_value, 1),
        "upside_from_entry": round(((analyst_target_value - entry_price) / entry_price) * 100, 1),
        "risk_reward": rr_ratio,
        "tranche_1_pct": tranche_1_pct,
        "tranche_2_pct": tranche_2_pct,
        "tranche_2_price": tranche_2_price,
        "tranche_3_pct": tranche_3_pct,
        "tranche_3_trigger": tranche_3_trigger,
        "signal": signal,
        "discount_from_current": round(((current_price_value - entry_price) / current_price_value) * 100, 1),
    }
