"""
Stock validation safety gate.

Every stock must pass validate_stock() before entering the pipeline.
Unresolvable stocks (demerged, delisted, no Screener data, no price)
are dropped silently from the candidate list and logged to SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from stock_platform.data.db import database_connection


class ValidationStatus(Enum):
    VALID         = "VALID"
    NOT_FOUND     = "NOT_FOUND"       # 404 / empty response from data source
    NO_DATA       = "NO_DATA"         # response present but all key fields None
    DEMERGED      = "DEMERGED"        # known corporate action, symbol remapped
    DELISTED      = "DELISTED"        # confirmed delisted
    STALE_DATA    = "STALE_DATA"      # data older than 90 days
    PRICE_MISSING = "PRICE_MISSING"   # cannot determine current price


@dataclass
class ValidationResult:
    symbol:          str
    status:          ValidationStatus
    resolved_symbol: str | None
    reason:          str
    can_recommend:   bool
    fin_data:        dict[str, Any]


# Fields that must have at least 2 populated for data to be considered usable.
_KEY_FIELDS = ("roce_pct", "eps", "debt_to_equity", "pe_ratio")

RECENTLY_LISTED_STOCKS = {
    # Symbol: IPO date (YYYY-MM-DD)
    "LGEINDIA": "2025-10-01",
    "TMPV": "2025-10-01",
    "TMCV": "2025-10-01",
}

LOCK_IN_WARNING_MONTHS = 12


def check_recently_listed(symbol: str, current_date: date | None = None) -> dict[str, Any]:
    if current_date is None:
        current_date = date.today()

    ipo_date_str = RECENTLY_LISTED_STOCKS.get(str(symbol).upper())
    if not ipo_date_str:
        return {
            "recently_listed": False,
            "months_since_ipo": None,
            "warning": None,
            "recommendation": None,
        }

    ipo_date = datetime.strptime(ipo_date_str, "%Y-%m-%d").date()
    months_since_ipo = (current_date - ipo_date).days / 30

    if months_since_ipo < LOCK_IN_WARNING_MONTHS:
        months_remaining = max(0.0, LOCK_IN_WARNING_MONTHS - months_since_ipo)
        return {
            "recently_listed": True,
            "months_since_ipo": round(months_since_ipo, 1),
            "warning": (
                f"{symbol.upper()} listed only "
                f"{months_since_ipo:.0f} months ago. "
                "Lock-in expiry overhang risk - "
                "large shareholders can now sell freely. "
                f"Recommend waiting until {months_remaining:.0f} more months "
                "post-IPO before building position."
            ),
            "recommendation": "WAIT - lock-in risk",
        }

    return {
        "recently_listed": False,
        "months_since_ipo": round(months_since_ipo, 1),
        "warning": None,
        "recommendation": None,
    }


def validate_stock(
    symbol: str,
    fin_data: dict[str, Any],
    current_price: float | None,
) -> ValidationResult:
    """
    Single entry point for all stock validation.

    Gates (checked in order):
      1. No fin_data at all          → NOT_FOUND
      2. All key fields are None     → NO_DATA
      3. Price missing or zero       → PRICE_MISSING
      4. Fewer than 2 key fields     → NO_DATA

    Returns ValidationResult with can_recommend=True only when all gates pass.
    """
    from stock_platform.utils.symbol_resolver import resolve_nse_symbol

    resolved = resolve_nse_symbol(symbol)

    # Gate 1: No financial data at all.
    if not fin_data:
        return ValidationResult(
            symbol=symbol,
            status=ValidationStatus.NOT_FOUND,
            resolved_symbol=None,
            reason=(
                f"{symbol} returned no data from Screener.in — "
                "may be demerged, delisted, or symbol incorrect. "
                "Check NSE_SYMBOL_MAP in utils/symbol_resolver.py."
            ),
            can_recommend=False,
            fin_data={},
        )

    # Gate 2: Data present but every key field is None.
    populated = [f for f in _KEY_FIELDS if fin_data.get(f) is not None]
    if len(populated) == 0:
        return ValidationResult(
            symbol=symbol,
            status=ValidationStatus.NO_DATA,
            resolved_symbol=resolved,
            reason=(
                f"{symbol} data fetched but all financial fields are None — "
                "cannot compute quality score reliably."
            ),
            can_recommend=False,
            fin_data=fin_data,
        )

    # Gate 3: Current price missing.
    if current_price is None or current_price <= 0:
        return ValidationResult(
            symbol=symbol,
            status=ValidationStatus.PRICE_MISSING,
            resolved_symbol=resolved,
            reason=(
                f"{symbol} current price unavailable — "
                "cannot compute allocation or net return."
            ),
            can_recommend=False,
            fin_data=fin_data,
        )

    # Gate 4: Fewer than 2 key fields populated — too uncertain to score reliably.
    if len(populated) < 2:
        return ValidationResult(
            symbol=symbol,
            status=ValidationStatus.NO_DATA,
            resolved_symbol=resolved,
            reason=(
                f"{symbol} has only {len(populated)}/{len(_KEY_FIELDS)} key fields — "
                "insufficient data for quality scoring."
            ),
            can_recommend=False,
            fin_data=fin_data,
        )

    # All gates passed.
    return ValidationResult(
        symbol=symbol,
        status=ValidationStatus.VALID,
        resolved_symbol=resolved,
        reason="All validation gates passed.",
        can_recommend=True,
        fin_data=fin_data,
    )


def log_skipped_stock(
    db_path: str,
    result: ValidationResult,
    run_id: str,
    *,
    neon_database_url: str = "",
    # Legacy Turso params — accepted for backward compat, not used.
    turso_database_url: str = "",
    turso_auth_token: str = "",
    turso_sync_interval_seconds: int | None = None,
) -> None:
    """Persist a skipped stock to the skipped_stocks table for audit transparency."""
    try:
        with database_connection(
            db_path,
            neon_url=neon_database_url or None,
        ) as conn:
            conn.execute(
                """
                INSERT INTO skipped_stocks (run_id, symbol, status, reason, skipped_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.symbol,
                    result.status.value,
                    result.reason,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception as exc:
        print(f"WARNING: could not log skipped stock {result.symbol} to DB: {exc}")


def filter_valid_candidates(
    candidates: list[dict[str, Any]],
    db_path: str,
    run_id: str,
    *,
    neon_database_url: str = "",
    # Legacy Turso params — accepted for backward compat, not used.
    turso_database_url: str = "",
    turso_auth_token: str = "",
    turso_sync_interval_seconds: int | None = None,
) -> tuple[list[dict[str, Any]], list[ValidationResult]]:
    """
    Split candidates into valid (can proceed) and skipped (blocked).

    Each candidate dict must have:
      - "symbol"        : str
      - "fin_data"      : dict  (from get_financials)
      - "current_price" : float | None

    Returns (valid_candidates, skipped_results).
    """
    valid: list[dict[str, Any]] = []
    skipped: list[ValidationResult] = []

    for stock in candidates:
        symbol        = stock.get("symbol", "")
        fin_data      = stock.get("fin_data") or {}
        current_price = stock.get("current_price")

        result = validate_stock(symbol, fin_data, current_price)

        if result.can_recommend:
            valid.append(stock)
        else:
            skipped.append(result)
            log_skipped_stock(
                db_path,
                result,
                run_id,
                neon_database_url=neon_database_url,
            )
            print(f"SKIPPED {symbol}: {result.reason}")

    return valid, skipped


def get_candidates_with_fallback(
    sector: str,
    needed: int,
    already_skipped: list[str],
    all_scored_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return top-N valid candidates for a sector, excluding known-bad symbols.

    When the highest-scored candidate in a sector is skipped, the next-best
    automatically fills the slot because they are sorted by quality_score desc.
    """
    sector_candidates = [
        c for c in all_scored_candidates
        if c.get("sector") == sector
        and c.get("symbol") not in already_skipped
    ]
    sector_candidates.sort(key=lambda x: x.get("quality_score", 0.0), reverse=True)
    return sector_candidates[:needed]
