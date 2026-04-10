from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    clean = value.replace("Z", "+00:00")
    return datetime.fromisoformat(clean)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def conviction_from_score(score: float) -> str:
    if score >= 0.75:
        return "STRONG_BUY"
    if score >= 0.55:
        return "BUY"
    if score >= 0.35:
        return "NEUTRAL"
    if score >= 0.15:
        return "AVOID"
    return "STRONG_AVOID"

