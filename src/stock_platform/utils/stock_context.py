from __future__ import annotations

from typing import Any


def build_factual_snapshot(
        symbol: str,
        fin_data: dict,
        current_price: float,
        pe_context: dict,
        tech_signals: dict,
        portfolio_overlap: float,
        sector_gap: dict) -> dict:
    """
    Build verified factual snapshot passed to both analysts.
    Clearly marks what is measured vs derived vs inferred.
    """
    # EPS — prefer explicit eps_ttm, fall back to generic eps / trailingEps
    eps = fin_data.get("eps_ttm") or fin_data.get("eps") or fin_data.get("trailingEps") or 0

    # Revenue growth — screener returns %, yfinance returns fraction
    revenue = (
        fin_data.get("revenue_growth_pct")
        or fin_data.get("revenueGrowth")
        or fin_data.get("revenue_growth")
    )
    if revenue is not None:
        try:
            rv = float(revenue)
            # fraction → percentage when |val| < 2.0 and val != 0
            if rv != 0 and abs(rv) < 2.0:
                revenue = round(rv * 100, 1)
            else:
                revenue = round(rv, 1)
        except (TypeError, ValueError):
            revenue = None

    # ROCE — screener %, yfinance fraction
    roce = (
        fin_data.get("roce_pct")
        or fin_data.get("roce_ttm")
        or fin_data.get("returnOnCapitalEmployed")
        or fin_data.get("roce")
    )
    if roce is not None:
        try:
            rv = float(roce)
            if rv != 0 and abs(rv) < 1.5:
                roce = round(rv * 100, 1)
            else:
                roce = round(rv, 1)
        except (TypeError, ValueError):
            roce = None

    de = fin_data.get("debt_to_equity") or 0
    qtr_rev = (fin_data.get("recent_results") or {}).get("revenue_yoy_growth_pct")

    return {
        # MEASURED — from latest filing/price
        "price":             current_price,
        "eps_ttm":           eps,
        "pe_current":        pe_context.get("pe_current"),
        "revenue_growth":    revenue,
        "roce":              roce,
        "debt_to_equity":    de,
        "week52_low":        fin_data.get("week52_low") or fin_data.get("fiftyTwoWeekLow"),
        "week52_high":       fin_data.get("week52_high") or fin_data.get("fiftyTwoWeekHigh"),
        "52w_data_quality":  fin_data.get("52w_data_quality", "UNKNOWN"),
        "latest_qtr_growth": qtr_rev,

        # DERIVED — computed from measured data
        "pe_vs_history":     pe_context.get("pe_vs_median_pct"),
        "pe_signal":         pe_context.get("pe_signal"),
        "pe_assessment":     pe_context.get("pe_assessment"),
        "pct_from_52w_low":  tech_signals.get("pct_from_52w_low"),
        "above_200dma":      tech_signals.get("above_200dma"),

        # PORTFOLIO CONTEXT
        "overlap_pct":       portfolio_overlap,
        "sector_gap":        sector_gap.get("conviction", ""),
        "gap_target_pct":    sector_gap.get("target_pct", 0),

        # DATA FRESHNESS
        "data_source":       fin_data.get("source", "screener"),
        "last_result_date":  fin_data.get("last_result_date"),
        "data_age_days":     fin_data.get("data_age_days", 99),
    }


def _n(val: Any, fmt: str, fallback: str = "N/A") -> str:
    """Format a value safely, returning fallback if None/zero."""
    if val is None:
        return fallback
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def format_snapshot_for_prompt(snapshot: dict) -> str:
    """
    Format factual snapshot as prompt text.
    Clearly labels measured vs derived vs inferred.
    """
    s = snapshot
    age = s.get("data_age_days", 99)
    try:
        age = int(age)
    except (TypeError, ValueError):
        age = 99

    freshness = (
        "FRESH (<7 days)"                         if age < 7  else
        "RECENT (7-30 days)"                      if age < 30 else
        "STALE (>30 days — treat with caution)"
    )

    price    = s.get("price") or 0
    eps      = s.get("eps_ttm") or 0
    pe       = s.get("pe_current")
    rev      = s.get("revenue_growth")
    roce     = s.get("roce")
    de       = s.get("debt_to_equity") or 0
    w52_low    = s.get("week52_low") or 0
    w52_high   = s.get("week52_high") or 0
    w52_quality = s.get("52w_data_quality", "UNKNOWN")

    qtr_growth = s.get("latest_qtr_growth")
    latest_qtr = f"{float(qtr_growth):.1f}% YoY" if qtr_growth is not None else "not available"

    pe_line   = f"{float(pe):.1f}x"  if pe  is not None else "N/A"
    rev_line  = f"{float(rev):.1f}% YoY" if rev is not None else "N/A"
    roce_line = f"{float(roce):.1f}%" if roce is not None else "N/A"

    pe_vs   = s.get("pe_vs_history")
    pe_vs_l = f"{float(pe_vs):+.0f}% vs 5yr median" if pe_vs is not None else "N/A"

    pct_low   = s.get("pct_from_52w_low")
    pct_low_l = f"{float(pct_low):.1f}% above 52W Low" if pct_low is not None else "N/A"

    # Build 52W range lines — suppress or warn based on data quality
    _bad_52w = w52_quality in ("DATA_CORRUPT", "UNAVAILABLE", "RANGE_MISMATCH")
    if _bad_52w:
        _w52_block = (
            f"  52W Low:          DATA {w52_quality} — do NOT use price-range-relative signals\n"
            f"  52W High:         DATA {w52_quality} — omit all 52W high/low references from entry rationale\n"
        )
        pct_low_l = f"SUPPRESSED (52W data {w52_quality})"
    elif w52_quality == "COMPUTED_FROM_HISTORY":
        _w52_block = (
            f"  52W Low:          ₹{float(w52_low):,.0f} [computed from 1Y history]\n"
            f"  52W High:         ₹{float(w52_high):,.0f} [computed from 1Y history]\n"
        )
    else:
        _w52_block = (
            f"  52W Low:          ₹{float(w52_low):,.0f}\n"
            f"  52W High:         ₹{float(w52_high):,.0f}\n"
        )

    above_200   = s.get("above_200dma")
    above_200_l = ("Yes" if above_200 else "No") if above_200 is not None else "N/A"

    val_rel  = s.get("val_reliability", "")
    val_note = s.get("val_reliability_note", "")
    val_flags = s.get("val_reliability_flags", [])
    val_section = ""
    if val_rel:
        val_section = (
            f"\nVALUATION RELIABILITY: {val_rel}\n"
            f"  Note: {val_note}\n"
        )
        if val_flags:
            val_section += "  Flags: " + "; ".join(val_flags) + "\n"

    return (
        "\n═══ VERIFIED FACTUAL SNAPSHOT ═══\n"
        "(Both analysts must reference these same facts.\n"
        " Do NOT use other numbers unless clearly labelled\n"
        " as your own estimate.)\n"
        "\n"
        "MEASURED (from filings/prices):\n"
        f"  Current Price:    ₹{float(price):,.0f}\n"
        f"  EPS (TTM):        ₹{float(eps):.2f}\n"
        f"  Current PE:       {pe_line}\n"
        f"  Revenue Growth:   {rev_line}\n"
        f"  Latest Qtr Rev:   {latest_qtr}\n"
        f"  ROCE:             {roce_line}\n"
        f"  D/E Ratio:        {float(de):.2f}\n"
        + _w52_block
        + "\n"
        "DERIVED (computed from above):\n"
        f"  PE vs 5yr Median: {pe_vs_l}\n"
        f"  PE Signal:        {s.get('pe_signal', 'UNKNOWN')}\n"
        f"  PE Context:       {s.get('pe_assessment', '')}\n"
        f"  % Above 52W Low:  {pct_low_l}\n"
        f"  Above 200 DMA:    {above_200_l}\n"
        + val_section
        + "\n"
        "PORTFOLIO CONTEXT:\n"
        f"  Existing Overlap: {float(s.get('overlap_pct', 0)):.1f}%\n"
        f"  Sector Gap:       {s.get('sector_gap', '')}\n"
        f"  Gap Target:       {float(s.get('gap_target_pct', 0)):.1f}%\n"
        "\n"
        "DATA QUALITY:\n"
        f"  Source:           {s.get('data_source', 'unknown')}\n"
        f"  Last Result:      {s.get('last_result_date', 'unknown')}\n"
        f"  Freshness:        {freshness}\n"
        "═════════════════════════════════\n"
    )
