from __future__ import annotations

import csv
import hashlib
import html as _html
import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.services.engine import PlatformEngine


st.set_page_config(
    page_title="Stock Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f3f4ee;
        --panel: rgba(255,255,255,0.92);
        --panel-strong: #ffffff;
        --text: #122c24;
        --muted: #627267;
        --line: rgba(18, 44, 36, 0.10);
        --teal: #0f766e;
        --teal-dark: #134e4a;
        --gold: #b45309;
        --green-soft: rgba(15,118,110,0.10);
        --gold-soft: rgba(180,83,9,0.11);
        --shadow: 0 16px 40px rgba(18, 44, 36, 0.08);
        --shadow-soft: 0 8px 18px rgba(18, 44, 36, 0.05);
        --radius-xl: 26px;
        --radius-lg: 20px;
        --radius-md: 16px;
    }
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(15,118,110,0.18), transparent 28%),
            radial-gradient(circle at 100% 0%, rgba(180,83,9,0.12), transparent 24%),
            linear-gradient(180deg, #f7f8f3 0%, #eef2eb 100%);
        color: var(--text);
    }
    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    div[data-testid="stTabs"] button {
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
    }
    div[data-testid="stDataFrame"] {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--line);
        box-shadow: var(--shadow-soft);
        background: var(--panel-strong);
    }
    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: var(--shadow-soft);
    }
    .hero-shell {
        background:
            linear-gradient(135deg, rgba(15,118,110,0.96), rgba(19,78,74,0.94) 62%, rgba(180,83,9,0.88));
        color: white;
        border-radius: var(--radius-xl);
        padding: 1.25rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 24px 54px rgba(18, 44, 36, 0.14);
        overflow: hidden;
        position: relative;
        margin-bottom: 1rem;
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -10% -30% auto;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.16), transparent 64%);
        pointer-events: none;
    }
    .hero-title {
        margin: 0 0 0.35rem 0;
        font-size: clamp(1.7rem, 4vw, 2.8rem);
        line-height: 1.03;
        letter-spacing: -0.03em;
    }
    .hero-copy {
        margin: 0;
        color: rgba(255,255,255,0.84);
        max-width: 900px;
        font-size: 0.98rem;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.9rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        color: #f9fffd;
        font-size: 0.76rem;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .metric-card, .glass-card, .model-box, .qa-card, .info-tile, .rec-card {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        box-shadow: var(--shadow-soft);
    }
    .metric-card, .glass-card {
        border-radius: var(--radius-lg);
        padding: 1rem;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.45rem;
    }
    .metric-value {
        color: var(--text);
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.0;
        margin-bottom: 0.28rem;
    }
    .metric-subtle, .mini-note {
        color: var(--muted);
        font-size: 0.88rem;
    }
    .section-chip, .pill, .provider-badge, .status-ok, .status-warn {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        font-size: 0.74rem;
    }
    .section-chip, .pill {
        padding: 0.28rem 0.68rem;
        background: var(--green-soft);
        color: var(--teal-dark);
        font-weight: 600;
    }
    .provider-badge {
        padding: 0.22rem 0.58rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .provider-badge-anthropic {
        background: var(--gold-soft);
        color: var(--gold);
    }
    .provider-badge-openai {
        background: rgba(16,163,127,0.12);
        color: #0f8a6c;
    }
    .status-ok, .status-warn {
        padding: 0.18rem 0.56rem;
        font-weight: 700;
    }
    .status-ok {
        background: rgba(22,163,74,0.12);
        color: #166534;
    }
    .status-warn {
        background: rgba(217,119,6,0.12);
        color: #92400e;
    }
    .info-tile, .model-box, .qa-card, .rec-card {
        border-radius: var(--radius-md);
        padding: 0.9rem 1rem;
    }
    .rec-card {
        border-left: 5px solid var(--teal);
        margin-bottom: 0.9rem;
    }
    .rec-card-anthropic { border-left-color: #b45309; }
    .rec-card-openai { border-left-color: #10a37f; }
    .model-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.83rem;
    }
    .model-chip {
        background: rgba(18,44,36,0.06);
        border-radius: 8px;
        padding: 0.13rem 0.45rem;
        font-family: Consolas, monospace;
        font-size: 0.78rem;
        color: var(--text);
    }
    .empty-panel {
        background: rgba(255,255,255,0.7);
        border: 1px dashed rgba(18,44,36,0.2);
        border-radius: var(--radius-lg);
        padding: 1.1rem;
        color: var(--muted);
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        .hero-shell {
            border-radius: 20px;
            padding: 1rem;
        }
        .metric-value {
            font-size: 1.45rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> PlatformEngine:
    return PlatformEngine()


def push_notice(message: str, level: str = "info") -> None:
    st.session_state["portal_notice"] = {"message": message, "level": level}


def render_pending_notice() -> None:
    notice = st.session_state.pop("portal_notice", None)
    if not notice:
        return
    message = notice["message"]
    level = notice["level"]
    icon = {"success": "✅", "warning": "⚠️", "error": "❌", "info": "ℹ️"}.get(level, "ℹ️")
    if hasattr(st, "toast"):
        st.toast(message, icon=icon)
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def build_upload_signature(uploaded_file, pdf_password: str) -> str:
    raw = uploaded_file.getvalue()
    digest = hashlib.sha256(raw + pdf_password.encode("utf-8")).hexdigest()
    return f"{uploaded_file.name}:{digest}"


def run_ingestion_workflow(engine: PlatformEngine, payload: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    if hasattr(st, "toast"):
        st.toast("Ingestion in progress...", icon="⏳")
    with st.status(f"Ingestion in progress for {source_label}...", expanded=True) as status:
        status.write("Reading holdings and refreshing portfolio context.")
        result = engine.ingest_portfolio(payload)
        status.write(f"Built {len(result['normalized_exposure'])} normalized positions.")
        status.update(label="Ingestion complete", state="complete", expanded=False)
    return result


def run_buy_workflow(
    engine: PlatformEngine,
    buy_request: dict[str, Any],
    *,
    provider_choice: str,
) -> dict[str, Any] | None:
    if provider_choice == "Compare Both":
        if hasattr(st, "toast"):
            st.toast("Recommendation comparison in progress...", icon="⏳")
        with st.status("Generating recommendations from both providers...", expanded=True) as status:
            comp = engine.run_buy_analysis_comparison(buy_request)
            status.write("Anthropic and OpenAI runs completed.")
            status.update(label="Recommendations ready", state="complete", expanded=False)
        return comp

    llm_provider = "anthropic" if provider_choice == "Anthropic Claude" else "openai"
    if hasattr(st, "toast"):
        st.toast("Recommendation run in progress...", icon="⏳")
    with st.status(f"Generating recommendations with {provider_choice}...", expanded=True) as status:
        engine.run_buy_analysis(buy_request, llm_provider=llm_provider)
        status.write("Recommendation cards are ready to review.")
        status.update(label="Recommendations ready", state="complete", expanded=False)
    return None


def _df(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def parse_upload(uploaded_file, pdf_password: str, engine: PlatformEngine) -> dict[str, Any] | None:
    if not uploaded_file:
        return None
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".json":
        return json.loads(uploaded_file.getvalue().decode("utf-8"))
    if suffix == ".csv":
        text = uploaded_file.getvalue().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        payload: dict[str, Any] = {"mutual_funds": [], "etfs": [], "direct_equities": []}
        for row in reader:
            holding_type = (row.get("holding_type") or "").strip().lower()
            record = {
                "instrument_name": row.get("instrument_name") or row.get("symbol"),
                "symbol": row.get("symbol") or None,
                "market_value": float(row.get("market_value") or 0),
                "quantity": float(row.get("quantity") or 0),
            }
            if holding_type == "mutual_fund":
                payload["mutual_funds"].append(record)
            elif holding_type == "etf":
                payload["etfs"].append(record)
            elif holding_type == "direct_equity":
                payload["direct_equities"].append(record)
        return payload
    if suffix == ".pdf":
        if not pdf_password:
            st.info("PDF detected. Enter the PDF password above to auto-parse the statement.")
            return None
        try:
            parsed = engine.pdf_parser.parse_bytes(uploaded_file.getvalue(), password=pdf_password).to_payload()
            st.success(
                f"PDF parsed: {len(parsed['mutual_funds'])} mutual funds, "
                f"{len(parsed['etfs'])} ETFs, "
                f"{len(parsed['direct_equities'])} direct equities."
            )
            return parsed
        except Exception as exc:
            st.error(f"PDF parse failed: {exc}")
            return None
    st.warning("Unsupported upload type. Use JSON, CSV, or PDF.")
    return None


def render_metric(label: str, value: str, subtle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{_html.escape(label)}</div>
            <div class="metric-value">{_html.escape(value)}</div>
            <div class="metric-subtle">{_html.escape(subtle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_tile(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="info-tile">
            <div class="metric-label">{_html.escape(title)}</div>
            <div style="font-size:1rem;font-weight:700;color:#122c24;">{_html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_panel(message: str) -> None:
    st.markdown(f'<div class="empty-panel">{_html.escape(message)}</div>', unsafe_allow_html=True)


def render_provider_model_box(label: str, fast_model: str, reasoning_model: str, enabled: bool, provider: str) -> None:
    status_badge = '<span class="status-ok">Configured</span>' if enabled else '<span class="status-warn">API key missing</span>'
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    hint = "" if enabled else f"Add {key_env} to .env to enable this provider."
    st.markdown(
        f"""
        <div class="model-box">
            <h5 style="margin:0 0 0.35rem 0;">{_html.escape(label)}</h5>
            <div class="model-row">
                <span>Fast <span class="model-chip">{_html.escape(fast_model)}</span></span>
                <span>Reasoning <span class="model-chip">{_html.escape(reasoning_model)}</span></span>
                <span>{status_badge}</span>
            </div>
            {"<div class='mini-note' style='margin-top:0.35rem;'>" + _html.escape(hint) + "</div>" if hint else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(item: dict[str, Any], provider: str = "") -> None:
    payload = item.get("payload", {})
    company = str(item.get("company_name", ""))
    symbol = str(item.get("symbol", ""))
    sector = str(item.get("sector", ""))
    action = str(item.get("action", ""))
    rationale = str(item.get("rationale", ""))
    confidence_band = str(item.get("confidence_band", ""))
    why_for_portfolio = str(payload.get("why_for_portfolio", ""))
    validation_reasoning = str(payload.get("validation_reasoning", ""))
    overlap = payload.get("overlap_pct", 0)
    allocation = payload.get("allocation_pct", 0)
    net_return = payload.get("net_of_tax_return_projection", 0)

    provider_label = ""
    if provider == "anthropic":
        provider_label = "Anthropic Claude"
    elif provider == "openai":
        provider_label = "OpenAI GPT"

    with st.container(border=True):
        head_col, action_col = st.columns([4, 1])
        with head_col:
            if provider_label:
                st.caption(provider_label)
            st.markdown(f"#### {company} ({symbol})")
            if sector:
                st.caption(sector)
        with action_col:
            st.markdown(f"**{action}**")

        if rationale:
            st.write(rationale)

        summary_bits = [
            f"Overlap {overlap}%",
            f"Allocation {allocation}%",
            f"Confidence {confidence_band}",
            f"Net Return {net_return}%",
        ]
        st.caption(" | ".join(summary_bits))

        if why_for_portfolio:
            st.write(why_for_portfolio)

        if validation_reasoning:
            st.caption(validation_reasoning)


def render_check_card(title: str, passed: bool, detail: str) -> None:
    status = '<span class="status-ok">Pass</span>' if passed else '<span class="status-warn">Attention</span>'
    st.markdown(
        f"""
        <div class="qa-card">
            <h5 style="margin:0 0 0.35rem 0;">{_html.escape(title)}</h5>
            <div style="margin-bottom:0.35rem;">{status}</div>
            <div class="mini-note">{_html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_render_checks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    portfolio = snapshot["portfolio"]
    signals = snapshot["signals"]
    return [
        {
            "title": "Snapshot Shape",
            "passed": all(key in snapshot for key in ("portfolio", "recommendations", "monitoring_actions", "signals", "llm")),
            "detail": "Top-level snapshot keys are available for the portal shell.",
        },
        {
            "title": "Portfolio Tables",
            "passed": all(key in portfolio for key in ("normalized_exposure", "identified_gaps", "raw_holdings")),
            "detail": f"{len(portfolio.get('normalized_exposure', []))} exposure rows and {len(portfolio.get('identified_gaps', []))} gap rows available.",
        },
        {
            "title": "Signal Frames",
            "passed": all(family in signals for family in ("unified", "geo", "policy", "flow", "contrarian")),
            "detail": f"Unified signals count: {len(signals.get('unified', []))}.",
        },
        {
            "title": "Recommendation Cards",
            "passed": isinstance(snapshot.get("recommendations", []), list),
            "detail": f"{len(snapshot.get('recommendations', []))} recommendation card payloads ready for rendering.",
        },
        {
            "title": "Monitoring Grid",
            "passed": isinstance(snapshot.get("monitoring_actions", []), list),
            "detail": f"{len(snapshot.get('monitoring_actions', []))} monitoring action rows ready for rendering.",
        },
        {
            "title": "LLM Status Banner",
            "passed": all(key in snapshot["llm"] for key in ("anthropic_enabled", "openai_enabled", "anthropic_fast_model", "openai_fast_model")),
            "detail": "Provider availability and model names are present for the portal controls.",
        },
    ]


engine = get_engine()
render_pending_notice()
snapshot = engine.get_dashboard_snapshot()
portfolio = snapshot["portfolio"]
signals = snapshot["signals"]
recommendations = snapshot["recommendations"]
monitoring_actions = snapshot["monitoring_actions"]
run_meta = snapshot.get("run_meta", {})
llm_status = snapshot["llm"]
portfolio_updated_at = portfolio.get("portfolio_meta", {}).get("portfolio_last_updated", "")
monitor_run_meta = run_meta.get("monitoring", {})
monitor_created_at = monitor_run_meta.get("created_at", "")
monitoring_is_stale = bool(portfolio_updated_at) and (
    not monitor_created_at or monitor_created_at < portfolio_updated_at
)

llm_providers_on = []
if llm_status["anthropic_enabled"]:
    llm_providers_on.append("Anthropic")
if llm_status["openai_enabled"]:
    llm_providers_on.append("OpenAI")
llm_display = " + ".join(llm_providers_on) if llm_providers_on else "Fallback"

if "monitoring_llm_provider" not in st.session_state:
    st.session_state["monitoring_llm_provider"] = (
        "anthropic" if llm_status["anthropic_enabled"] else
        "openai" if llm_status["openai_enabled"] else
        "anthropic"
    )

st.markdown(
    f"""
    <div class="hero-shell">
        <h1 class="hero-title">Your Portfolio Assistant</h1>
        <p class="hero-copy">
            Upload your statement, let the app ingest it automatically, then review simple buy ideas
            and direct-stock monitoring without digging through technical screens.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">1. Upload PDF</span>
            <span class="hero-badge">2. Auto Ingest</span>
            <span class="hero-badge">3. Review Buy Ideas</span>
            <span class="hero-badge">4. Monitor Direct Stocks</span>
            <span class="hero-badge">LLM: {_html.escape(llm_display)}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_bar = st.columns([1, 1, 1])
with top_bar[0]:
    if st.button("Seed Demo Portfolio", use_container_width=True):
        try:
            engine.seed_demo_data()
            push_notice("Sample portfolio loaded successfully.", "success")
            st.rerun()
        except ModuleNotFoundError as exc:
            st.error(f"Dependency missing: {exc}. Install project dependencies first.")
with top_bar[1]:
    if st.button("Refresh Signals", use_container_width=True):
        try:
            engine.run_signal_refresh(trigger="manual")
            push_notice("Signals refreshed.", "success")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
with top_bar[2]:
    if st.button("Run Monitoring", use_container_width=True):
        try:
            provider_for_mon = st.session_state.get("monitoring_llm_provider", "anthropic")
            provider_label = "Anthropic Claude" if provider_for_mon == "anthropic" else "OpenAI GPT"
            if hasattr(st, "toast"):
                st.toast("Monitoring in progress...", icon="⏳")
            with st.status(f"Monitoring your direct holdings with {provider_label}...", expanded=True) as status:
                engine.run_monitoring(llm_provider=provider_for_mon)
                status.write("Monitoring actions are ready.")
                status.update(label="Monitoring complete", state="complete", expanded=False)
            push_notice("Monitoring complete.", "success")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric("Normalized Holdings", str(len(portfolio["normalized_exposure"])), "Look-through after MF / ETF expansion")
with metric_cols[1]:
    render_metric("Unified Signals", str(len(signals["unified"])), "Geo, policy, flow, contrarian blend")
with metric_cols[2]:
    render_metric("Open Recommendations", str(len(recommendations)), "Latest personalized buy run")
with metric_cols[3]:
    render_metric("Portal LLM", llm_display, "Rendering rationale providers")

info_cols = st.columns(3)
with info_cols[0]:
    render_info_tile("Portfolio State", "Ready" if portfolio["normalized_exposure"] else "Awaiting ingestion")
with info_cols[1]:
    render_info_tile("AMC Look-Through", "Official + fallback" if portfolio["normalized_exposure"] else "No portfolio context")
with info_cols[2]:
    _direct_count = len([r for r in portfolio["raw_holdings"] if r["holding_type"] == "direct_equity"])
    _watch_count = len(portfolio.get("watchlist", []))
    render_info_tile("Monitoring Scope", f"{_direct_count} direct + {_watch_count} watchlist")

tabs = st.tabs(["Home", "Upload Portfolio", "Buy Ideas", "Monitoring", "Signals", "Diagnostics"])

with tabs[0]:
    st.markdown('<span class="section-chip">Quick Snapshot</span>', unsafe_allow_html=True)
    st.info("Start with `Upload Portfolio`. PDF uploads will parse and ingest automatically.")
    top_cols = st.columns([1.15, 0.85])
    with top_cols[0]:
        st.subheader("Portfolio Exposure")
        exposure_frame = _df(portfolio["normalized_exposure"], ["symbol", "company_name", "sector", "total_weight"])
        if len(exposure_frame):
            st.dataframe(exposure_frame, use_container_width=True, hide_index=True)
        else:
            render_empty_panel("No normalized exposure yet. Seed demo data or ingest a statement to populate this view.")
    with top_cols[1]:
        st.subheader("Sector Gaps")
        gap_frame = _df(portfolio["identified_gaps"], ["sector", "underweight_pct", "conviction", "score", "reason"])
        if len(gap_frame):
            st.dataframe(gap_frame, use_container_width=True, hide_index=True)
        else:
            render_empty_panel("Gap analysis will appear after portfolio ingestion and signal refresh.")

    st.markdown('<span class="section-chip">Latest Buy Ideas</span>', unsafe_allow_html=True)
    st.subheader("Latest Recommendation Run")
    if recommendations:
        for item in recommendations:
            render_recommendation_card(item)
    else:
        render_empty_panel("No recommendation run yet. Move to Buy Studio after ingesting a portfolio.")

with tabs[1]:
    st.markdown('<span class="section-chip">Upload And Ingest</span>', unsafe_allow_html=True)
    st.subheader("Upload Your Portfolio")
    st.caption("Upload JSON, CSV, or an encrypted NSDL / CAS PDF. PDF uploads start ingestion automatically after parsing.")

    pdf_password = st.text_input(
        "PDF password",
        type="password",
        help="Required for encrypted NSDL / CAS statements.",
    )
    uploaded = st.file_uploader("Upload portfolio file", type=["json", "csv", "pdf"])
    uploaded_payload = parse_upload(uploaded, pdf_password, engine)

    if uploaded and uploaded_payload and Path(uploaded.name).suffix.lower() == ".pdf":
        upload_signature = build_upload_signature(uploaded, pdf_password)
        if st.session_state.get("auto_ingested_pdf_signature") != upload_signature:
            try:
                result = run_ingestion_workflow(engine, uploaded_payload, source_label=uploaded.name)
                st.session_state["auto_ingested_pdf_signature"] = upload_signature
                push_notice(
                    f"Ingestion complete. {len(result['normalized_exposure'])} normalized positions are ready.",
                    "success",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"PDF ingestion failed: {exc}")
                push_notice(f"PDF ingestion failed: {exc}", "error")
        else:
            st.caption("This PDF is already ingested. You can upload a new file or edit the portfolio manually below.")

    existing_prefs = portfolio["user_preferences"]
    default_payload = uploaded_payload or {
        "macro_thesis": existing_prefs.get("macro_thesis", ""),
        "investable_surplus": existing_prefs.get("investable_surplus", 0),
        "direct_equity_corpus": existing_prefs.get("direct_equity_corpus", 0),
        "mutual_funds": [row["payload"] for row in portfolio["raw_holdings"] if row["holding_type"] == "mutual_fund"],
        "etfs": [row["payload"] for row in portfolio["raw_holdings"] if row["holding_type"] == "etf"],
        "direct_equities": [row["payload"] for row in portfolio["raw_holdings"] if row["holding_type"] == "direct_equity"],
    }

    with st.expander("Edit portfolio manually", expanded=not bool(uploaded and Path(uploaded.name).suffix.lower() == ".pdf")):
        with st.form("portfolio-form"):
            thesis = st.text_area("Macro thesis override", value=default_payload.get("macro_thesis", ""), height=90)
            a, b = st.columns(2)
            with a:
                surplus = st.number_input("Investable surplus", min_value=0.0, value=float(default_payload.get("investable_surplus", 0.0)))
            with b:
                corpus = st.number_input("Direct equity corpus", min_value=0.0, value=float(default_payload.get("direct_equity_corpus", 0.0)))

            st.caption("Mutual funds")
            mf_df = st.data_editor(
                _df(default_payload.get("mutual_funds", []), ["instrument_name", "market_value"]),
                num_rows="dynamic",
                use_container_width=True,
                key="mf_editor",
            )
            st.caption("ETFs")
            etf_df = st.data_editor(
                _df(default_payload.get("etfs", []), ["instrument_name", "market_value"]),
                num_rows="dynamic",
                use_container_width=True,
                key="etf_editor",
            )
            st.caption("Direct equities")
            direct_df = st.data_editor(
                _df(default_payload.get("direct_equities", []), ["instrument_name", "symbol", "quantity", "market_value"]),
                num_rows="dynamic",
                use_container_width=True,
                key="direct_editor",
            )
            submitted = st.form_submit_button("Save And Ingest Portfolio", use_container_width=True)
            if submitted:
                ingest_payload = {
                    "macro_thesis": thesis,
                    "investable_surplus": surplus,
                    "direct_equity_corpus": corpus,
                    "mutual_funds": mf_df.fillna("").to_dict("records"),
                    "etfs": etf_df.fillna("").to_dict("records"),
                    "direct_equities": direct_df.fillna("").to_dict("records"),
                }
                try:
                    result = run_ingestion_workflow(engine, ingest_payload, source_label="manual portfolio")
                    push_notice(
                        f"Ingestion complete. {len(result['normalized_exposure'])} normalized positions are ready.",
                        "success",
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

with tabs[2]:
    st.markdown('<span class="section-chip">Buy Ideas</span>', unsafe_allow_html=True)
    st.subheader("Generate Recommendations")

    st.markdown("#### LLM Provider")
    mp_col1, mp_col2 = st.columns([1.4, 2.3])
    with mp_col1:
        provider_choice = st.radio(
            "Select LLM provider for recommendations",
            options=["Anthropic Claude", "OpenAI GPT", "Compare Both"],
            index=0,
            label_visibility="collapsed",
            key="provider_radio",
        )
        st.session_state["global_llm_provider"] = (
            "anthropic" if provider_choice == "Anthropic Claude" else
            "openai" if provider_choice == "OpenAI GPT" else
            "anthropic"
        )
    with mp_col2:
        if provider_choice in ("Anthropic Claude", "Compare Both"):
            render_provider_model_box(
                "Anthropic Claude",
                llm_status["anthropic_fast_model"],
                llm_status["anthropic_reasoning_model"],
                llm_status["anthropic_enabled"],
                "anthropic",
            )
        if provider_choice in ("OpenAI GPT", "Compare Both"):
            render_provider_model_box(
                "OpenAI GPT",
                llm_status["openai_fast_model"],
                llm_status["openai_reasoning_model"],
                llm_status["openai_enabled"],
                "openai",
            )

    with st.form("buy-form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            index_name = st.selectbox("Index", ["NIFTY50", "NIFTYNEXT50"])
        with c2:
            horizon_months = st.slider("Horizon (months)", 6, 36, 18)
        with c3:
            risk_profile = st.selectbox("Risk", ["Balanced", "Moderate", "Aggressive"])
        with c4:
            top_n = st.slider("Top N", 1, 4, 3)

        btn_label = {
            "Anthropic Claude": "Generate Buy Ideas",
            "OpenAI GPT": "Generate Buy Ideas",
            "Compare Both": "Compare Both Providers",
        }[provider_choice]
        run_buy = st.form_submit_button(btn_label, use_container_width=True)

        if run_buy:
            buy_request = {
                "index_name": index_name,
                "horizon_months": horizon_months,
                "risk_profile": risk_profile,
                "top_n": top_n,
            }
            try:
                comp = run_buy_workflow(engine, buy_request, provider_choice=provider_choice)
                if provider_choice == "Compare Both":
                    st.session_state["comparison_result"] = comp
                    push_notice("Recommendation comparison complete.", "success")
                else:
                    st.session_state.pop("comparison_result", None)
                    push_notice("Recommendation run complete.", "success")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if "comparison_result" in st.session_state:
        st.subheader("Provider Comparison")
        comp: dict[str, Any] = st.session_state["comparison_result"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<span class="provider-badge provider-badge-anthropic">Anthropic Claude</span>', unsafe_allow_html=True)
            a_data = comp.get("anthropic", {})
            if a_data.get("error"):
                st.warning(a_data["error"])
            elif not a_data.get("recommendations"):
                render_empty_panel("No Anthropic recommendations were generated.")
            else:
                for item in a_data["recommendations"]:
                    render_recommendation_card(item, provider="anthropic")
        with col_b:
            st.markdown('<span class="provider-badge provider-badge-openai">OpenAI GPT</span>', unsafe_allow_html=True)
            o_data = comp.get("openai", {})
            if o_data.get("error"):
                st.warning(o_data["error"])
            elif not o_data.get("recommendations"):
                render_empty_panel("No OpenAI recommendations were generated.")
            else:
                for item in o_data["recommendations"]:
                    render_recommendation_card(item, provider="openai")
        if st.button("Clear comparison view"):
            st.session_state.pop("comparison_result", None)
            st.rerun()
    else:
        st.subheader("Latest Recommendation Run")
        if recommendations:
            for item in recommendations:
                render_recommendation_card(item)
        else:
            render_empty_panel("Run the buy workflow to populate the recommendation feed.")

with tabs[3]:
    st.markdown('<span class="section-chip">Actions + Behaviour</span>', unsafe_allow_html=True)
    st.subheader("Monitoring")

    st.markdown("#### Monitoring LLM")
    mon_choice = st.radio(
        "Select LLM provider for monitoring",
        options=["Anthropic Claude", "OpenAI GPT"],
        index=0 if st.session_state.get("monitoring_llm_provider", "anthropic") == "anthropic" else 1,
        horizontal=True,
        key="monitoring_provider_radio",
    )
    st.session_state["monitoring_llm_provider"] = "anthropic" if mon_choice == "Anthropic Claude" else "openai"

    if st.session_state["monitoring_llm_provider"] == "anthropic":
        render_provider_model_box(
            "Monitoring uses Anthropic Claude",
            llm_status["anthropic_fast_model"],
            llm_status["anthropic_reasoning_model"],
            llm_status["anthropic_enabled"],
            "anthropic",
        )
    else:
        render_provider_model_box(
            "Monitoring uses OpenAI GPT",
            llm_status["openai_fast_model"],
            llm_status["openai_reasoning_model"],
            llm_status["openai_enabled"],
            "openai",
        )

    if st.button("Run Monitoring For Current Portfolio", use_container_width=True, key="monitoring_tab_run_btn"):
        try:
            provider_for_mon = st.session_state.get("monitoring_llm_provider", "anthropic")
            provider_label = "Anthropic Claude" if provider_for_mon == "anthropic" else "OpenAI GPT"
            if hasattr(st, "toast"):
                st.toast("Monitoring in progress...", icon="⏳")
            with st.status(f"Monitoring your direct holdings with {provider_label}...", expanded=True) as status:
                engine.run_monitoring(llm_provider=provider_for_mon)
                status.write("Monitoring actions are ready.")
                status.update(label="Monitoring complete", state="complete", expanded=False)
            push_notice(f"Monitoring complete using {provider_label}.", "success")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if portfolio_updated_at:
        freshness_bits = [f"Portfolio updated: {portfolio_updated_at}"]
        if monitor_created_at:
            freshness_bits.append(f"Last monitoring run: {monitor_created_at}")
        else:
            freshness_bits.append("Last monitoring run: not run yet for current portfolio")
        st.caption(" | ".join(freshness_bits))

    if monitoring_is_stale and portfolio["raw_holdings"]:
        st.warning(
            "Monitoring results are stale for the latest portfolio. "
            "Run Monitoring to refresh actions for the currently ingested holdings."
        )

    # ── Monitoring Scope ──────────────────────────────────────────────────────
    st.markdown("#### Monitoring Scope")
    st.caption(
        "Only your directly held stocks and any watchlist additions are monitored. "
        "Mutual fund & ETF holdings are managed by fund managers — they are excluded."
    )

    direct_holdings = [r for r in portfolio["raw_holdings"] if r["holding_type"] == "direct_equity"]
    watchlist = portfolio.get("watchlist", [])

    scope_cols = st.columns([1, 1])
    with scope_cols[0]:
        st.markdown(f"**Direct Holdings — {len(direct_holdings)} stock(s)**")
        if direct_holdings:
            st.dataframe(
                pd.DataFrame([
                    {
                        "symbol": r.get("symbol", ""),
                        "name": r["instrument_name"],
                        "market value": r["market_value"],
                    }
                    for r in direct_holdings
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_empty_panel("No direct equities found. Ingest a portfolio or NSDL PDF to populate.")

    with scope_cols[1]:
        st.markdown(f"**Watchlist — {len(watchlist)} stock(s)**")
        if watchlist:
            st.dataframe(
                pd.DataFrame([
                    {
                        "symbol": r["symbol"],
                        "name": r["company_name"],
                        "sector": r["sector"],
                        "note": r["note"],
                    }
                    for r in watchlist
                ]),
                use_container_width=True,
                hide_index=True,
            )
            remove_sym = st.selectbox(
                "Remove stock from watchlist",
                [r["symbol"] for r in watchlist],
                key="watchlist_remove_select",
            )
            if st.button("Remove from Watchlist", key="watchlist_remove_btn"):
                engine.remove_watchlist_stock(remove_sym)
                st.success(f"{remove_sym} removed from watchlist.")
                st.rerun()
        else:
            render_empty_panel("Watchlist is empty. Add stocks below to include them in monitoring.")

    with st.expander("Add stock to watchlist", expanded=not bool(watchlist)):
        with st.form("watchlist-add-form"):
            wl_c1, wl_c2, wl_c3 = st.columns(3)
            with wl_c1:
                wl_symbol = st.text_input("Symbol (e.g. INFY)")
            with wl_c2:
                wl_name = st.text_input("Company name")
            with wl_c3:
                wl_sector = st.text_input("Sector (optional)")
            wl_note = st.text_input("Note (optional)")
            if st.form_submit_button("Add to Watchlist", use_container_width=True):
                sym_clean = wl_symbol.strip().upper()
                name_clean = wl_name.strip()
                if not sym_clean or not name_clean:
                    st.warning("Symbol and company name are required.")
                else:
                    engine.add_watchlist_stock(
                        symbol=sym_clean,
                        company_name=name_clean,
                        sector=wl_sector.strip() or "Unknown",
                        note=wl_note.strip(),
                    )
                    st.success(f"{sym_clean} added to watchlist.")
                    st.rerun()

    st.divider()

    # ── Monitoring Results ────────────────────────────────────────────────────
    st.markdown("#### Latest Monitoring Results")
    if monitoring_actions:
        monitor_frame = pd.DataFrame(
            [
                {
                    "symbol": item["symbol"],
                    "action": item["action"],
                    "severity": item["severity"],
                    "rationale": item["rationale"],
                }
                for item in monitoring_actions
            ]
        )
        st.dataframe(monitor_frame, use_container_width=True, hide_index=True)
    else:
        render_empty_panel(
            "Monitoring has not been run for the current portfolio yet. Use the Run Monitoring button above."
        )

with tabs[4]:
    st.markdown('<span class="section-chip">Signal Intelligence Layer</span>', unsafe_allow_html=True)
    st.subheader("Signals")
    signal_family = st.radio(
        "Signal family",
        options=["unified", "geo", "policy", "flow", "contrarian"],
        horizontal=True,
        index=0,
    )
    signal_frame = _df(signals[signal_family], ["sector", "conviction", "score", "source", "horizon", "detail"])
    if len(signal_frame):
        st.dataframe(signal_frame, use_container_width=True, hide_index=True)
    else:
        render_empty_panel("No signals available yet. Refresh signals or seed demo data.")

with tabs[5]:
    st.markdown('<span class="section-chip">Render Diagnostics</span>', unsafe_allow_html=True)
    st.subheader("Portal QA")
    st.caption("This section helps confirm that the portal is receiving valid data and rendering its major blocks correctly.")

    checks = build_render_checks(snapshot)
    qa_cols = st.columns(2)
    for idx, check in enumerate(checks):
        with qa_cols[idx % 2]:
            render_check_card(check["title"], check["passed"], check["detail"])

    st.markdown("#### Visual Smoke Test")
    smoke_cols = st.columns(3)
    with smoke_cols[0]:
        render_metric("Smoke Metric", "12", "Metric cards render with theme styling")
    with smoke_cols[1]:
        render_info_tile("Smoke Tile", "Info tiles, spacing, and typography")
    with smoke_cols[2]:
        render_check_card("Card Styling", True, "QA cards, pills, badges, and visual hierarchy should all render cleanly.")

    st.markdown("#### Data Shape Preview")
    preview = {
        "portfolio_rows": len(portfolio["normalized_exposure"]),
        "gap_rows": len(portfolio["identified_gaps"]),
        "recommendation_rows": len(recommendations),
        "monitoring_rows": len(monitoring_actions),
        "signal_rows": {key: len(value) for key, value in signals.items()},
    }
    st.json(preview)

    st.markdown("#### Recommendation Card Preview")
    preview_item = recommendations[0] if recommendations else {
        "company_name": "Preview Stock",
        "symbol": "PRV",
        "sector": "Preview Sector",
        "action": "ACCUMULATE",
        "confidence_band": "GREEN",
        "rationale": "This preview card confirms card layout, typography, pills, and longer copy rendering.",
        "payload": {
            "overlap_pct": 0.8,
            "allocation_pct": 12,
            "net_of_tax_return_projection": 17.4,
            "why_for_portfolio": "Used to verify recommendation-card rendering even when the live feed is empty.",
        },
    }
    render_recommendation_card(preview_item, provider="openai" if llm_status["openai_enabled"] else "")
