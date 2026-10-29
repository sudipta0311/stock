"""Pure DKK account ordering and presentation for Denmark-resident plans."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any

from wealth.core.money import round_paise
from wealth.policy.denmark_picks import EtfPick, comparison_for_etf, plain_fund_name


def round_dkk_for_display(amount: Decimal) -> Decimal:
    """Present amounts to whole kroner, or the nearest 100 above DKK 1,000."""
    quantum = Decimal("1E2") if amount >= Decimal("1000") else Decimal("1")
    return amount.quantize(quantum, rounding=ROUND_HALF_UP)


def inr_from_dkk(amount_dkk: Decimal, inr_per_dkk: Decimal) -> Decimal:
    """Convert DKK into INR using the FX table's INR-per-DKK rate."""
    if inr_per_dkk <= 0:
        raise ValueError("DKK/INR must be positive")
    return round_paise(amount_dkk * inr_per_dkk)


def dkk_from_inr(amount_inr: Decimal, inr_per_dkk: Decimal) -> Decimal:
    """Convert INR to DKK using the FX table's INR-per-DKK rate."""
    if inr_per_dkk <= 0:
        raise ValueError("DKK/INR must be positive")
    return amount_inr / inr_per_dkk


def ask_deposit_room_dkk(
    cap_dkk: Decimal,
    value_start_of_year_dkk: Decimal,
    net_deposits_this_year_dkk: Decimal,
) -> Decimal:
    """Return remaining room under the ASK calendar-year net-deposit cap."""
    # https://skat.dk/borger/aktier-og-andre-vaerdipapirer/aktiesparekonto
    return max(Decimal(0), cap_dkk - value_start_of_year_dkk - net_deposits_this_year_dkk)


def account_lines(
    pick: EtfPick,
    *,
    member_id: str,
    broker: str,
    amount_dkk: Decimal,
    ask_room_dkk: Decimal,
    inr_per_dkk: Decimal,
    details: Mapping[str, Any] | None = None,
    broker_availability_unchecked: bool = False,
) -> list[dict[str, Any]]:
    """Allocate an eligible fund to the share savings account first, then ordinary."""
    if amount_dkk <= 0:
        return []
    total = (
        min(amount_dkk, max(Decimal(0), ask_room_dkk))
        if pick.instrument.get("ask_eligible")
        else Decimal(0)
    )
    ordinary = amount_dkk - total
    if Decimal(0) < total < Decimal(1) and ordinary > 0:
        total, ordinary = Decimal(0), amount_dkk
    elif Decimal(0) < ordinary < Decimal(1) and total > 0:
        # Paise rounding in placement leaves sub-krone dust; ASK display rounds down, so
        # folding it into the ASK line never shows more than the room.
        total, ordinary = amount_dkk, Decimal(0)
    if amount_dkk < 1:
        return []
    chunks: list[tuple[str, Decimal]] = []
    if total > 0:
        chunks.append(("ask", total))
    if ordinary > 0:
        chunks.append(("ordinary", ordinary))

    lines: list[dict[str, Any]] = []
    for account, amount in chunks:
        displayed_amount = round_dkk_for_display(amount)
        if account == "ask":
            quantum = Decimal("1E2") if amount >= Decimal("1000") else Decimal("1")
            displayed_amount = amount.quantize(quantum, rounding=ROUND_DOWN)
        line = {
            "member_id": member_id,
            "account": account,
            "broker": broker,
            "instrument": {
                "name": plain_fund_name(str(pick.instrument["name"])),
                "ticker": pick.ticker,
                "exchange": pick.exchange,
                "isin": pick.instrument["isin"],
                "currency": pick.currency,
            },
            "amount_dkk": format(displayed_amount, "f"),
            "amount_inr": format(inr_from_dkk(amount, inr_per_dkk), ".2f"),
            "_amount_dkk_unrounded": amount,
            "reason_codes": (
                ["ask_first", "ask_room_remaining"] if account == "ask" else ["ordinary_after_ask"]
            ),
            "how_to_key": f"how_to.{broker}",
            "details": {
                "alternatives_considered": [
                    {
                        "name": row["name"],
                        "reason": "higher_fee_or_lower_rank",
                    }
                    for row in details.get("alternatives", [])
                ]
                if details
                else [],
                "us_comparison": comparison_for_etf(pick),
                "tax_verified": False,
                "held_us_etf_notes": list(details.get("held_us_etf_notes", [])) if details else [],
                "pension_note": bool(details.get("pension_note")) if details else False,
                "sources": [
                    {"label": "factsheet", "url": pick.instrument["factsheet_url"]},
                    {"label": "tax_list", "url": pick.instrument["abis_source_url"]},
                    {"label": "share_savings_account", "url": pick.instrument["ask_source_url"]},
                ],
            },
        }
        if broker_availability_unchecked:
            line["reason_codes"].append("broker_availability_unchecked")
        lines.append(line)
    return lines


def buy_abroad_action_params(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the deterministic action metadata consumed by email and PWA clients."""
    return {
        "abroad_buy_list": lines,
        "broker": lines[0]["broker"] if lines else "",
        "account_type": lines[0]["account"].upper() if lines else "",
    }
