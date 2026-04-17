from __future__ import annotations

from typing import Any


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalise_pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw != 0 and abs(raw) < 2.0:
        return round(raw * 100, 1)
    return round(raw, 1)


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
    del symbol

    eps = fin_data.get("eps_ttm") or fin_data.get("eps") or fin_data.get("trailingEps") or 0

    recent_results = fin_data.get("recent_results") or {}
    revenue_ttm = _normalise_pct(_first_present(
        fin_data.get("revenue_growth_ttm"),
        fin_data.get("revenue_growth_pct"),
        fin_data.get("revenueGrowth"),
        fin_data.get("revenue_growth"),
    ))
    revenue_latest_qtr = _normalise_pct(_first_present(
        fin_data.get("revenue_growth_latest_qtr"),
        recent_results.get("revenue_yoy_growth_pct"),
    ))
    revenue_latest_qtr_label = _first_present(
        fin_data.get("revenue_growth_latest_qtr_label"),
        (fin_data.get("revenue_momentum") or {}).get("period"),
        recent_results.get("comparison_label"),
    )
    pat_block = fin_data.get("pat_momentum") or {}
    pat_signal = _first_present(
        pat_block.get("pat_momentum"),
        recent_results.get("pat_momentum"),
    )
    pat_growth = _normalise_pct(_first_present(
        pat_block.get("pat_growth_pct"),
        recent_results.get("pat_growth_pct"),
    ))
    pat_period = _first_present(
        pat_block.get("period"),
        recent_results.get("period"),
        revenue_latest_qtr_label,
    )
    rev_pat_divergence = bool(_first_present(
        pat_block.get("rev_pat_divergence"),
        recent_results.get("rev_pat_divergence"),
        False,
    ))

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

    return {
        "price": current_price,
        "eps_ttm": eps,
        "pe_current": pe_context.get("pe_current"),
        "revenue_growth": revenue_ttm,
        "revenue_growth_ttm": revenue_ttm,
        "revenue_growth_latest_qtr": revenue_latest_qtr,
        "latest_qtr_growth": revenue_latest_qtr,
        "latest_qtr_comparison_label": revenue_latest_qtr_label,
        "pat_momentum": pat_signal,
        "pat_growth_pct": pat_growth,
        "pat_period": pat_period,
        "rev_pat_divergence": rev_pat_divergence,
        "roce": roce,
        "debt_to_equity": de,
        "week52_low": fin_data.get("week52_low") or fin_data.get("fiftyTwoWeekLow"),
        "week52_high": fin_data.get("week52_high") or fin_data.get("fiftyTwoWeekHigh"),
        "52w_data_quality": fin_data.get("52w_data_quality", "UNKNOWN"),
        "pe_vs_history": pe_context.get("pe_vs_median_pct"),
        "pe_signal": pe_context.get("pe_signal"),
        "pe_assessment": pe_context.get("pe_assessment"),
        "pct_from_52w_low": tech_signals.get("pct_from_52w_low"),
        "above_200dma": tech_signals.get("above_200dma"),
        "overlap_pct": portfolio_overlap,
        "sector_gap": sector_gap.get("conviction", ""),
        "gap_target_pct": sector_gap.get("target_pct", 0),
        "pledge_pct": fin_data.get("pledge_pct"),
        "pledge_trend": fin_data.get("pledge_trend"),
        "promoter_holding": fin_data.get("promoter_holding"),
        "data_source": fin_data.get("source", "screener"),
        "last_result_date": fin_data.get("last_result_date"),
        "data_age_days": fin_data.get("data_age_days", 99),
    }


def _n(val: Any, fmt: str, fallback: str = "N/A") -> str:
    """Format a value safely, returning fallback if None/zero."""
    if val is None:
        return fallback
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def _format_pledge_block(s: dict) -> str:
    pledge_pct = s.get("pledge_pct")
    pledge_trend = s.get("pledge_trend")
    promoter_holding = s.get("promoter_holding")
    if pledge_pct is None and promoter_holding is None:
        return ""
    lines = ["\nGOVERNANCE:\n"]
    if promoter_holding is not None:
        lines.append(f"  Promoter Holding: {float(promoter_holding):.1f}%\n")
    if pledge_pct is not None:
        risk_flag = " [HIGH - margin call overhang risk]" if pledge_pct > 30 else ""
        lines.append(f"  Promoter Pledge:  {float(pledge_pct):.1f}%{risk_flag}\n")
        if pledge_trend:
            lines.append(f"  Pledge Trend:     {pledge_trend}\n")
    return "".join(lines)


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

    if age == 99:
        freshness_label = "NO_RESULT_DATE"
        freshness_days_str = "unknown"
        freshness_prompt = "Data freshness: UNKNOWN - no result date available. Do NOT make conviction calls on timing."
    elif age <= 30:
        freshness_label = "FRESH"
        freshness_days_str = f"{age} days ago"
        freshness_prompt = f"Data freshness: FRESH - last quarterly result {age} days ago. Conviction calls permitted."
    elif age <= 60:
        freshness_label = "ACCEPTABLE"
        freshness_days_str = f"{age} days ago"
        freshness_prompt = f"Data freshness: ACCEPTABLE - last quarterly result {age} days ago. Reduce conviction slightly."
    elif age <= 90:
        freshness_label = "STALE"
        freshness_days_str = f"{age} days ago"
        freshness_prompt = f"Data freshness: STALE - last quarterly result {age} days ago. Flag this. Lower conviction."
    else:
        freshness_label = "VERY_STALE"
        freshness_days_str = f"{age} days ago"
        freshness_prompt = f"Data freshness: VERY_STALE - last quarterly result {age} days ago. Do NOT recommend entry. Downgrade to WATCHLIST."

    freshness = f"{freshness_label} ({freshness_days_str})"

    price = s.get("price") or 0
    eps = s.get("eps_ttm") or 0
    pe = s.get("pe_current")
    rev_ttm = _first_present(s.get("revenue_growth_ttm"), s.get("revenue_growth"))
    roce = s.get("roce")
    de = s.get("debt_to_equity") or 0
    w52_low = s.get("week52_low") or 0
    w52_high = s.get("week52_high") or 0
    w52_quality = s.get("52w_data_quality", "UNKNOWN")

    qtr_growth = _first_present(s.get("revenue_growth_latest_qtr"), s.get("latest_qtr_growth"))
    qtr_label = s.get("latest_qtr_comparison_label") or "latest quarter vs same quarter last year"
    pat_signal = s.get("pat_momentum") or "UNKNOWN"
    pat_growth = s.get("pat_growth_pct")
    pat_period = s.get("pat_period") or qtr_label
    pat_line = (
        f"{pat_signal} ({float(pat_growth):+.1f}% YoY)"
        if pat_growth is not None
        else pat_signal
    )

    pe_line = f"{float(pe):.1f}x" if pe is not None else "N/A"
    rev_line = f"{float(rev_ttm):.1f}%" if rev_ttm is not None else "N/A"
    qtr_line = f"{float(qtr_growth):.1f}%" if qtr_growth is not None else "N/A"
    roce_line = f"{float(roce):.1f}%" if roce is not None else "N/A"

    pe_vs = s.get("pe_vs_history")
    pe_vs_l = f"{float(pe_vs):+.0f}% vs 5yr median" if pe_vs is not None else "N/A"

    pct_low = s.get("pct_from_52w_low")
    pct_low_l = f"{float(pct_low):.1f}% above 52W Low" if pct_low is not None else "N/A"

    bad_52w = w52_quality in ("DATA_CORRUPT", "UNAVAILABLE", "RANGE_MISMATCH")
    if bad_52w:
        w52_block = (
            f"  52W Low:          DATA {w52_quality} - do NOT use price-range-relative signals\n"
            f"  52W High:         DATA {w52_quality} - omit all 52W high/low references from entry rationale\n"
        )
        pct_low_l = f"SUPPRESSED (52W data {w52_quality})"
    elif w52_quality == "COMPUTED_FROM_HISTORY":
        w52_block = (
            f"  52W Low:          Rs.{float(w52_low):,.0f} [computed from 1Y history]\n"
            f"  52W High:         Rs.{float(w52_high):,.0f} [computed from 1Y history]\n"
        )
    else:
        w52_block = (
            f"  52W Low:          Rs.{float(w52_low):,.0f}\n"
            f"  52W High:         Rs.{float(w52_high):,.0f}\n"
        )

    above_200 = s.get("above_200dma")
    above_200_l = ("Yes" if above_200 else "No") if above_200 is not None else "N/A"

    val_rel = s.get("val_reliability", "")
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
        "\n=== VERIFIED FACTUAL SNAPSHOT ===\n"
        "(Both analysts must reference these same facts.\n"
        " Do NOT use other numbers unless clearly labelled\n"
        " as your own estimate.)\n"
        "\n"
        "MEASURED (from filings/prices):\n"
        f"  Current Price:    Rs.{float(price):,.0f}\n"
        f"  EPS (TTM):        Rs.{float(eps):.2f}\n"
        f"  Current PE:       {pe_line}\n"
        f"  Revenue growth (TTM YoY): {rev_line}\n"
        f"  Revenue growth ({qtr_label}): {qtr_line}\n"
        "  Use quarterly figure for momentum assessment, TTM figure for trend assessment.\n"
        f"  PAT momentum ({pat_period}): {pat_line}\n"
        f"  ROCE:             {roce_line}\n"
        f"  D/E Ratio:        {float(de):.2f}\n"
        + w52_block
        + "\n"
        "DERIVED (computed from above):\n"
        f"  PE vs 5yr Median: {pe_vs_l}\n"
        f"  PE Signal:        {s.get('pe_signal', 'UNKNOWN')}\n"
        f"  PE Context:       {s.get('pe_assessment', '')}\n"
        f"  % Above 52W Low:  {pct_low_l}\n"
        f"  Above 200 DMA:    {above_200_l}\n"
        + (
            "  Revenue/PAT divergence: YES - revenue is growing while PAT is deteriorating. "
            "Identify whether this is one-time or structural before recommending entry.\n"
            if s.get("rev_pat_divergence")
            else ""
        )
        + val_section
        + "\n"
        "PORTFOLIO CONTEXT:\n"
        f"  Existing Overlap: {float(s.get('overlap_pct', 0)):.1f}%\n"
        f"  Sector Gap:       {s.get('sector_gap', '')}\n"
        f"  Gap Target:       {float(s.get('gap_target_pct', 0)):.1f}%\n"
        + _format_pledge_block(s)
        + "\n"
        "DATA QUALITY:\n"
        f"  Source:           {s.get('data_source', 'unknown')}\n"
        f"  Last Result:      {s.get('last_result_date', 'unknown')}\n"
        f"  Freshness:        {freshness}\n"
        f"  {freshness_prompt}\n"
        "  Adjust conviction accordingly.\n"
        "=================================\n"
    )
