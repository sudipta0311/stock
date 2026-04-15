from __future__ import annotations

from typing import Any


KNOWN_ANALYST_TARGETS = {
    "SUZLON": 74.0,
    "HAVELLS": 1583.0,
    "DIXON": 12617.0,
    "LGEINDIA": 1752.0,
    "BEL": 488.0,
    "HAL": 4875.0,
}


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


def fetch_analyst_consensus_target(symbol: str, current_price: float) -> float:
    """
    Fetch a real analyst consensus target, trying multiple sources in order.
    """
    current_price_value = _as_float(current_price) or 0.0
    if current_price_value <= 0:
        return 0.0

    # Require analyst target to be above current price — stale targets (below current
    # after a market correction) fall through to the 15% fallback.
    min_valid_target = current_price_value
    clean_symbol = str(symbol or "").upper().strip()

    import logging
    _log = logging.getLogger(__name__)

    try:
        from utils.screener_fetcher import fetch_screener_data

        data = fetch_screener_data(clean_symbol) or {}
        target = data.get("target_price") or data.get("target_mean_price")
        target_value = _as_float(target)
        if target_value is not None and target_value > min_valid_target:
            _log.info("%s target from Screener: Rs.%.0f", clean_symbol, target_value)
            return float(target_value)
    except Exception:
        pass

    try:
        import yfinance as yf
        from utils.symbol_resolver import resolve_nse_symbol

        info = yf.Ticker(resolve_nse_symbol(clean_symbol)).info or {}
        target = info.get("targetMeanPrice")
        target_value = _as_float(target)
        if target_value is not None and target_value > min_valid_target:
            _log.info("%s target from yfinance: Rs.%.0f", clean_symbol, target_value)
            return float(target_value)
    except Exception:
        pass

    if clean_symbol in KNOWN_ANALYST_TARGETS:
        target_value = float(KNOWN_ANALYST_TARGETS[clean_symbol])
        if target_value > min_valid_target:
            _log.info("%s target from known targets: Rs.%.0f", clean_symbol, target_value)
            return target_value

    fallback = current_price_value * 1.15
    _log.info("%s: no valid target above current price - using Rs.%.0f (15%% fallback)", clean_symbol, fallback)
    return float(fallback)


def apply_momentum_override(
    signal: str,
    recent_results: dict[str, Any],
    current_price: float,
    week52_low: float | None,
) -> str:
    """
    Upgrade a WAIT signal when revenue momentum is strong and the stock is still
    trading relatively close to its 52-week low.
    """
    if not recent_results:
        return signal

    growth = _as_float(recent_results.get("revenue_yoy_growth_pct")) or 0.0
    momentum = str(recent_results.get("momentum") or "").upper()
    current_price_value = _as_float(current_price)
    week52_low_value = _as_float(week52_low)

    near_low = False
    if (
        current_price_value is not None
        and current_price_value > 0
        and week52_low_value is not None
        and week52_low_value > 0
    ):
        pct_from_low = ((current_price_value - week52_low_value) / week52_low_value) * 100.0
        near_low = pct_from_low < 30.0

    signal_key = str(signal or "").strip().upper().replace("_", " ")
    if signal_key == "WAIT" and momentum == "STRONG" and growth > 30.0 and near_low:
        print(
            "Momentum override: upgrading WAIT -> ACCUMULATE "
            f"(revenue growth {growth:.1f}% + near 52w low)"
        )
        return "ACCUMULATE"

    return signal


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
