from __future__ import annotations

from datetime import date, datetime
from typing import Any


STCG_RATE = 0.20
LTCG_RATE = 0.125
LTCG_EXEMPTION = 125000


def calculate_pnl(
    symbol: str,
    avg_buy_price: float,
    current_price: float,
    quantity: float,
    buy_date_str: str,
) -> dict[str, Any]:
    invested = avg_buy_price * quantity
    current_val = current_price * quantity
    gross_pnl = current_val - invested
    pnl_pct = (gross_pnl / invested * 100) if invested > 0 else 0
    try:
        buy_date = datetime.strptime(buy_date_str, "%Y-%m-%d").date()
        days_held = (date.today() - buy_date).days
    except Exception:
        days_held = None
    is_ltcg = days_held is not None and days_held >= 365
    tax_rate = LTCG_RATE if is_ltcg else STCG_RATE
    taxable = max(0, gross_pnl - (LTCG_EXEMPTION if is_ltcg else 0))
    tax_amount = taxable * tax_rate if gross_pnl > 0 else 0
    net_pnl = gross_pnl - tax_amount
    days_to_ltcg = max(0, 365 - days_held) if days_held is not None and days_held < 365 else 0
    return {
        "symbol": symbol,
        "quantity": quantity,
        "avg_buy_price": avg_buy_price,
        "current_price": current_price,
        "invested": round(invested, 2),
        "current_value": round(current_val, 2),
        "gross_pnl": round(gross_pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "days_held": days_held,
        "is_ltcg": is_ltcg,
        "tax_rate_pct": tax_rate * 100,
        "tax_amount": round(tax_amount, 2),
        "net_pnl": round(net_pnl, 2),
        "days_to_ltcg": days_to_ltcg,
        "tax_type": "LTCG" if is_ltcg else "STCG",
    }


def should_exit(pnl: dict[str, Any], analyst_target: float | None, current_price: float) -> dict[str, Any]:
    gross_pnl = pnl["gross_pnl"]
    days_to_ltcg = pnl["days_to_ltcg"]
    is_ltcg = pnl["is_ltcg"]
    tax_amount = pnl["tax_amount"]
    remaining_upside = (
        ((analyst_target - current_price) / current_price * 100)
        if analyst_target and current_price > 0
        else 0
    )

    if not is_ltcg and gross_pnl > 0:
        ltcg_tax = max(0, gross_pnl - LTCG_EXEMPTION) * LTCG_RATE
        stcg_tax = gross_pnl * STCG_RATE
        tax_saving_from_wait = stcg_tax - ltcg_tax
    else:
        tax_saving_from_wait = 0

    if gross_pnl < 0:
        return {
            "exit_recommendation": "EXIT — cut loss",
            "reasoning": (
                f"Down Rs{abs(gross_pnl):,.0f} ({pnl['pnl_pct']:.1f}%). "
                "No tax on losses — exit if thesis broken."
            ),
            "tax_note": "No capital gains tax on loss",
            "urgency": "HIGH",
        }
    if remaining_upside < 10 and gross_pnl > 0:
        if days_to_ltcg > 0 and days_to_ltcg <= 30 and tax_saving_from_wait > 5000:
            return {
                "exit_recommendation": f"WAIT {days_to_ltcg} days then EXIT",
                "reasoning": (
                    f"Only {remaining_upside:.1f}% upside remaining. "
                    f"Waiting {days_to_ltcg} days saves Rs{tax_saving_from_wait:,.0f} in tax."
                ),
                "tax_note": (
                    f"STCG now: Rs{tax_amount:,.0f} -> "
                    f"LTCG after wait: Rs{max(0, gross_pnl - LTCG_EXEMPTION) * LTCG_RATE:,.0f}"
                ),
                "urgency": "MEDIUM",
            }
        return {
            "exit_recommendation": "EXIT — limited upside",
            "reasoning": (
                f"In profit Rs{gross_pnl:,.0f} ({pnl['pnl_pct']:.1f}%) but only "
                f"{remaining_upside:.1f}% upside remaining."
            ),
            "tax_note": f"{pnl['tax_type']}: Rs{tax_amount:,.0f}. Net proceeds: Rs{pnl['net_pnl']:,.0f}",
            "urgency": "MEDIUM",
        }
    return {
        "exit_recommendation": "HOLD",
        "reasoning": (
            f"In profit Rs{gross_pnl:,.0f} ({pnl['pnl_pct']:.1f}%) with "
            f"{remaining_upside:.1f}% upside remaining."
        ),
        "tax_note": f"If exited today: {pnl['tax_type']} Rs{tax_amount:,.0f}",
        "urgency": "LOW",
    }
