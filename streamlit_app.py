from __future__ import annotations

import csv
import hashlib
import html as _html
import io
import json
import sys
import textwrap
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

# ── Push secrets into os.environ at module load time ─────────────────────────
# Two-pass approach: (1) st.secrets (Streamlit Cloud dashboard), then
# (2) .streamlit/secrets.toml on disk (committed to repo, always present).
# AppConfig is a frozen dataclass — reads os.getenv() once at construction.
import os as _os, tomllib as _tomllib

# Pass 1: Streamlit Cloud dashboard secrets
try:
    for _sk, _sv in st.secrets.items():
        if isinstance(_sv, str) and _sk not in _os.environ:
            _os.environ[_sk] = _sv
except Exception:
    pass

# Pass 2: Read .streamlit/secrets.toml directly (bypasses st.secrets parsing)
try:
    _secrets_toml = ROOT / ".streamlit" / "secrets.toml"
    if _secrets_toml.exists():
        with _secrets_toml.open("rb") as _sf:
            _toml_data = _tomllib.load(_sf)
        for _sk, _sv in _toml_data.items():
            if isinstance(_sv, str) and _sk not in _os.environ:
                _os.environ[_sk] = _sv
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

# Module-level state for the PE-cache background thread.
# Threads write here; Streamlit reruns read it.  Dict mutation is GIL-safe.
# Guard: Streamlit re-executes the script on every rerun in the SAME globals
# namespace.  Without the guard this line would reset the dict on every rerun,
# losing the "running" state that the button handler just wrote.
if "_PE_CACHE_JOB" not in globals():
    _PE_CACHE_JOB: dict[str, Any] = {
        "running": False, "done": False,
        "count": 0, "saved": 0, "skipped": 0, "failed": 0,
        "error": "",
    }
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_platform.config import AppConfig
from stock_platform.agents.buy_agents import MINIMUM_RR_RATIO
from stock_platform.agents.quant_model import apply_freshness_cap
from stock_platform.services.engine import PlatformEngine
from stock_platform.services.llm import PlatformLLM
from stock_platform.utils.index_config import DEFAULT_INDEX, INDEX_UNIVERSE, SELECTABLE_INDICES
from stock_platform.utils.fii_dii_fetcher import fetch_fii_dii_sector_flow
from stock_platform.utils.cache_init import ensure_cache_tables, get_cache_row_counts
from stock_platform.utils.pe_history_fetcher import prefetch_pe_history_for_universe
from stock_platform.utils.direct_equity_merge import apply_saved_buy_prices

# Ensure all three cache tables exist in Neon (or SQLite fallback) at startup.
# Guard prevents re-running on every Streamlit rerun within the same process.
if "_CACHE_TABLES_ENSURED" not in globals():
    try:
        ensure_cache_tables()
    except Exception:
        pass
    _CACHE_TABLES_ENSURED = True
from stock_platform.utils.recommendation_resolver import (
    extract_synthesis_verdict,
    resolve_final_recommendation,
)
from stock_platform.utils.sector_config import ELEVATED_GOVERNANCE_RISK
from utils.entry_calculator import calculate_entry_levels


st.set_page_config(
    page_title="Stock Intelligence Platform",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --font-sans: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
        --font-display: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        --bg-soft-2: rgba(250, 248, 241, 0.82);
        --line-strong: rgba(28, 49, 41, 0.16);
        --shadow-strong: 0 28px 64px rgba(26, 48, 40, 0.16);
    }
    html, body {
        background-color: #fbfaf6 !important;
        font-family: var(--font-sans) !important;
        height: auto !important;
        min-height: 100vh !important;
    }
    [data-testid="stAppViewContainer"],
    .stApp {
        height: auto !important;
        min-height: 100vh !important;
    }
    body, p, label, li, td, th, input, textarea, button {
        font-family: var(--font-sans) !important;
    }
    [data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"] *,
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-style: normal !important;
        font-weight: 400 !important;
        letter-spacing: normal !important;
        line-height: 1 !important;
        text-transform: none !important;
        white-space: nowrap !important;
    }
    h1, h2, h3, h4, h5, h6, .hero-title, .section-title {
        font-family: var(--font-display) !important;
        letter-spacing: -0.035em;
    }
    .stApp {
        position: relative;
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(182, 103, 26, 0.12), transparent 25%),
            linear-gradient(180deg, #fbfaf6 0%, #eef1e9 52%, #f6f4ee 100%) !important;
        overflow-x: hidden;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.18) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.18) 1px, transparent 1px);
        background-size: 32px 32px;
        mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.28), transparent 78%);
        opacity: 0.42;
        pointer-events: none;
    }
    header[data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: transparent !important;
    }
    .block-container {
        max-width: 980px;
        padding: 1.1rem 1.35rem 3rem;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        gap: 0.45rem;
        padding: 0.42rem;
        margin-bottom: 1.1rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.50);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(250, 247, 239, 0.82));
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
        overflow-x: auto;
        scrollbar-width: none;
    }
    div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
        display: none;
    }
    div[data-testid="stTabs"] button {
        min-height: 2.55rem;
        padding: 0.55rem 1rem;
        border: 1px solid transparent !important;
        font-size: 0.92rem;
        font-weight: 700;
        transition: background-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #134e4a !important;
        -webkit-text-fill-color: #134e4a !important;
        background: linear-gradient(180deg, rgba(15, 118, 110, 0.16), rgba(15, 118, 110, 0.08)) !important;
        border-color: rgba(15, 118, 110, 0.18) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
    }
    div[data-testid="stTabs"] button:hover {
        background-color: rgba(15, 118, 110, 0.07) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        border-radius: 18px;
        border: 1px solid rgba(18, 44, 36, 0.10);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(249, 247, 241, 0.94)) !important;
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 22px !important;
        border: 1px solid rgba(18, 44, 36, 0.10) !important;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(250, 248, 243, 0.96)) !important;
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
        overflow: hidden;
    }
    div[data-testid="stForm"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(250, 247, 239, 0.94)) !important;
        border: 1px solid rgba(18, 44, 36, 0.10) !important;
        border-radius: 22px;
        padding: 1.05rem 1rem 0.7rem;
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(18, 44, 36, 0.10) !important;
        border-radius: 22px !important;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(248, 246, 240, 0.86)) !important;
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
        overflow: hidden;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCaptionContainer"] p,
    .mini-note,
    .section-copy {
        color: #627267 !important;
        -webkit-text-fill-color: #627267 !important;
        line-height: 1.6;
    }
    div[data-testid="stMarkdownContainer"] .hero-copy,
    div[data-testid="stMarkdownContainer"] .hero-copy p,
    .hero-copy {
        color: rgba(255, 251, 245, 0.90) !important;
        -webkit-text-fill-color: rgba(255, 251, 245, 0.90) !important;
    }
    div[data-testid="stMarkdownContainer"] .hero-pill,
    div[data-testid="stMarkdownContainer"] .hero-kicker,
    div[data-testid="stMarkdownContainer"] .hero-stat,
    .hero-pill,
    .hero-kicker,
    .hero-stat {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        min-height: 2.9rem;
        border-radius: 16px;
        border: 1px solid rgba(15, 118, 110, 0.18) !important;
        background: linear-gradient(180deg, rgba(15, 118, 110, 0.98), rgba(19, 78, 74, 0.94)) !important;
        color: #f8fcfa !important;
        -webkit-text-fill-color: #f8fcfa !important;
        font-weight: 800;
        box-shadow: 0 14px 28px rgba(15, 118, 110, 0.18);
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }
    .stButton > button *,
    div[data-testid="stFormSubmitButton"] > button *,
    div[data-testid="stDownloadButton"] > button * {
        color: #f8fcfa !important;
        -webkit-text-fill-color: #f8fcfa !important;
    }
    .stButton > button [data-testid="stMarkdownContainer"] p,
    div[data-testid="stFormSubmitButton"] > button [data-testid="stMarkdownContainer"] p,
    div[data-testid="stDownloadButton"] > button [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(15, 118, 110, 0.22);
        filter: saturate(1.02);
    }
    .stButton > button:focus-visible,
    div[data-testid="stFormSubmitButton"] > button:focus-visible,
    div[data-testid="stDownloadButton"] > button:focus-visible {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > div {
        border-radius: 14px !important;
        border: 1px solid var(--line-strong) !important;
        background: rgba(255, 255, 255, 0.95) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 22px !important;
        border: 1.5px dashed rgba(15, 118, 110, 0.28) !important;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(241, 247, 244, 0.92)) !important;
        padding: 1.2rem !important;
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
    }
    div[data-testid="stRadio"] label {
        border-radius: 999px !important;
        border: 1px solid rgba(18, 44, 36, 0.10) !important;
        background: rgba(255, 255, 255, 0.72) !important;
        padding: 0.38rem 0.72rem !important;
        margin-right: 0.45rem;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        border-color: rgba(15, 118, 110, 0.20) !important;
        background: rgba(15, 118, 110, 0.10) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
    }
    div[data-baseweb="notification"] {
        border-radius: 18px !important;
        border: 1px solid rgba(18, 44, 36, 0.10) !important;
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 246, 240, 0.96)) !important;
    }
    .hero-shell {
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
        padding: 1.4rem 1.35rem 1.3rem;
        border-radius: 30px;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background:
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.20), transparent 30%),
            linear-gradient(135deg, rgba(15, 118, 110, 0.98), rgba(19, 78, 74, 0.95) 58%, rgba(182, 103, 26, 0.88));
        box-shadow: var(--shadow-strong);
    }
    .hero-shell::before {
        content: "";
        position: absolute;
        inset: auto -12% -34% auto;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.18), transparent 64%);
        pointer-events: none;
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(120deg, rgba(255, 255, 255, 0.04), transparent 35%),
            linear-gradient(180deg, transparent, rgba(8, 20, 16, 0.12));
        pointer-events: none;
    }
    .hero-shell-login {
        text-align: center;
        padding: 2.65rem 1.6rem;
    }
    .hero-kicker-row,
    .hero-stat-row {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .hero-kicker-row {
        margin-bottom: 0.8rem;
    }
    .hero-kicker,
    .hero-pill,
    .section-chip,
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.72rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .hero-kicker {
        background: rgba(255, 255, 255, 0.16);
        color: rgba(255, 255, 255, 0.92) !important;
    }
    .hero-pill {
        background: rgba(10, 20, 17, 0.18);
        color: rgba(255, 251, 245, 0.92) !important;
    }
    .hero-title {
        position: relative;
        z-index: 1;
        margin: 0 0 0.4rem;
        font-size: clamp(2rem, 5vw, 3rem);
        line-height: 0.98;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 44rem;
        margin: 0;
        color: rgba(255, 253, 248, 0.82) !important;
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .hero-stat-row {
        margin-top: 1rem;
    }
    .hero-stat {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.46rem 0.76rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: rgba(255, 253, 248, 0.98) !important;
        font-size: 0.82rem;
        backdrop-filter: blur(8px);
    }
    .hero-stat strong {
        color: white !important;
        font-size: 0.92rem;
    }
    .section-intro {
        margin: 0.3rem 0 0.95rem;
    }
    .section-chip,
    .pill {
        background: linear-gradient(180deg, rgba(15, 118, 110, 0.13), rgba(15, 118, 110, 0.08)) !important;
        color: #134e4a !important;
        -webkit-text-fill-color: #134e4a !important;
        border: 1px solid rgba(15, 118, 110, 0.12);
    }
    .section-title {
        margin: 0.55rem 0 0.2rem;
        font-size: clamp(1.45rem, 3vw, 2rem);
    }
    .provider-badge {
        padding: 0.28rem 0.68rem;
        margin-bottom: 0.55rem;
        border: 1px solid transparent;
        font-weight: 800;
        letter-spacing: 0.01em;
    }
    .provider-badge-anthropic {
        background-color: rgba(182, 103, 26, 0.12) !important;
        color: #9a5513 !important;
        -webkit-text-fill-color: #9a5513 !important;
        border-color: rgba(182, 103, 26, 0.14);
    }
    .provider-badge-openai {
        background-color: rgba(16, 163, 127, 0.12) !important;
        color: #0f8a6c !important;
        -webkit-text-fill-color: #0f8a6c !important;
        border-color: rgba(16, 163, 127, 0.14);
    }
    .status-ok,
    .status-warn {
        padding: 0.24rem 0.58rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .metric-card,
    .info-tile,
    .qa-card,
    .model-box,
    .empty-panel {
        box-shadow: 0 10px 24px rgba(26, 48, 40, 0.06);
    }
    .metric-card {
        min-height: 100%;
        padding: 0.95rem 1rem;
        border: 1px solid rgba(18, 44, 36, 0.10);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 246, 240, 0.95)) !important;
    }
    .metric-label {
        margin-bottom: 0.28rem;
        color: #627267 !important;
        -webkit-text-fill-color: #627267 !important;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-subtle {
        margin-top: 0.26rem;
        color: #627267 !important;
        -webkit-text-fill-color: #627267 !important;
        font-size: 0.82rem;
    }
    .info-tile,
    .qa-card,
    .model-box {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(18, 44, 36, 0.10);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), var(--bg-soft-2)) !important;
    }
    .empty-panel {
        padding: 1rem 1.05rem;
        border: 1px dashed rgba(28, 49, 41, 0.22);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.74), rgba(247, 244, 237, 0.82)) !important;
    }
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.55rem !important;
        }
        div[data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .block-container {
            padding: 0.7rem 0.65rem 2rem !important;
        }
        .hero-shell {
            padding: 1rem 0.92rem;
            border-radius: 20px;
            margin-bottom: 0.7rem;
        }
        .hero-shell-login {
            padding: 2rem 1rem;
        }
        .hero-title {
            font-size: 1.7rem;
        }
        .hero-copy {
            font-size: 0.88rem;
        }
        .hero-stat {
            width: 100%;
            justify-content: space-between;
        }
        div[data-testid="stTabs"] button {
            padding: 0.42rem 0.78rem !important;
            font-size: 0.82rem !important;
            white-space: nowrap;
        }
        div[data-testid="stForm"] {
            padding: 0.8rem 0.78rem 0.5rem !important;
        }
        .section-title {
            font-size: 1.42rem;
        }
        .metric-card,
        .qa-card,
        .empty-panel,
        .info-tile,
        .model-box {
            padding: 0.78rem 0.82rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _auth_configured() -> bool:
    """Return True only when Google OAuth secrets are fully and validly configured."""
    try:
        auth = st.secrets.get("auth", {})
        if not auth.get("google"):
            return False
        cookie = str(auth.get("cookie_secret", ""))
        # Reject placeholder values — a weak/example cookie_secret silently
        # breaks Streamlit's OAuth session signing and causes a blank screen.
        if not cookie or cookie == "replace-with-a-long-random-string" or len(cookie) < 16:
            return False
        return True
    except Exception:
        return False


@st.cache_resource(max_entries=50)
def get_engine(user_key: str = "local") -> PlatformEngine:
    """Return a PlatformEngine scoped to this user.

    Chinese wall: each authenticated user gets a dedicated SQLite database
    at data/users/<sha256(email)>.db — no data crosses between accounts.
    user_key is the full sha256 hex digest of the user's Google email,
    or 'local' for single-user / unauthenticated mode.
    """
    # os.environ already populated from st.secrets at module load time.
    # load_app_env() handles local .env / secrets.toml for dev environments.
    from stock_platform.config import load_app_env
    load_app_env()

    if user_key == "local":
        return PlatformEngine()
    users_dir = ROOT / "data" / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    return PlatformEngine(config=AppConfig(db_path=users_dir / f"{user_key}.db"))


@st.cache_data(ttl=60, show_spinner=False)
def get_openai_connection_status(
    openai_api_key: str,
    openai_fast_model: str,
    openai_reasoning_model: str,
) -> tuple[bool, str]:
    llm = PlatformLLM(
        AppConfig(
            openai_api_key=openai_api_key,
            openai_fast_model=openai_fast_model,
            openai_reasoning_model=openai_reasoning_model,
        ),
        provider="openai",
    )
    return llm.test_openai_connection()


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
        st.session_state.pop("comparison_result", None)
        st.session_state.pop("buy_blocked_reason", None)
        st.session_state["buy_skipped_stocks"] = []
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
        with st.status("Refreshing signals and generating recommendations...", expanded=True) as status:
            comp = engine.run_buy_analysis_comparison(buy_request)
            engine.repo.set_state("buy_comparison_result", comp)
            status.write("Signal refresh and both provider runs completed.")
            status.update(label="Recommendations ready", state="complete", expanded=False)
        return comp

    llm_provider = "anthropic" if provider_choice == "Anthropic Claude" else "openai"
    if hasattr(st, "toast"):
        st.toast("Recommendation run in progress...", icon="⏳")
    with st.status(f"Generating recommendations with {provider_choice}...", expanded=True) as status:
        result = engine.run_buy_analysis(buy_request, llm_provider=llm_provider)
        run_summary = result.get("run_summary", {}) if result else {}
        if run_summary.get("blocked_reason"):
            status.update(label="Market signals too weak — no recommendations", state="error", expanded=False)
        else:
            status.write("Recommendation cards are ready to review.")
            status.update(label="Recommendations ready", state="complete", expanded=False)
    # Persist skipped stocks and pipeline stats so the display section can show them.
    st.session_state["buy_skipped_stocks"] = (result or {}).get("skipped_stocks", [])
    st.session_state["buy_run_summary"] = run_summary
    engine.repo.set_state("buy_comparison_result", {})
    return {"run_summary": run_summary} if run_summary.get("blocked_reason") else None


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
            <div style="font-size:1rem;font-weight:700;color:#122c24;-webkit-text-fill-color:#122c24;">{_html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(chip: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <span class="section-chip">{_html.escape(chip)}</span>
            <h2 class="section-title">{_html.escape(title)}</h2>
            <p class="section-copy">{_html.escape(description)}</p>
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


ENTRY_SUMMARY_MARKER = "**ENTRY SUMMARY:**"


def should_show_entry_plan(fin_data: dict, risk_profile: str = "Balanced") -> dict:
    """Return whether to show entry plan given staleness and risk profile.

    Hard suppress cutoffs (days stale): Conservative=45, Balanced=90, Aggressive=150.
    Returns {"show": bool, "warning": str}.
    """
    _cutoffs = {"Conservative": 45, "Balanced": 90, "Aggressive": 150}
    cutoff = _cutoffs.get(risk_profile, 90)
    days_stale = fin_data.get("result_days_stale")
    result_date = fin_data.get("last_result_date")

    if not result_date or str(result_date).lower() in ("none", ""):
        return {"show": False, "warning": ""}

    try:
        days = int(days_stale) if days_stale is not None else 999
    except (TypeError, ValueError):
        days = 999

    if days > cutoff:
        return {"show": False, "warning": ""}

    warning = ""
    if days > 45:
        warning = (
            f"Note: Last result date is {days} days old. "
            f"Entry plan shown because {risk_profile} profile tolerates up to {cutoff} days of data staleness."
        )
    return {"show": True, "warning": warning}


def compute_data_badge(fin_data: dict) -> dict:
    """
    Three-state data quality badge based on actual fundamental availability.
    DATA UNKNOWN fires only when core ratios are missing, not merely when
    the result date is absent.
    """
    has_pe     = fin_data.get("pe_ratio") not in (None, 0) or fin_data.get("pe_trailing") not in (None, 0)
    has_roce   = fin_data.get("roce_pct") not in (None, 0) or fin_data.get("roce") not in (None, 0)
    has_rev    = (
        fin_data.get("revenue_growth_pct") not in (None, 0)
        or fin_data.get("revenueGrowth") not in (None, 0)
        or fin_data.get("revenue_growth") not in (None, 0)
    )
    has_result = fin_data.get("last_result_date") not in (None, "None", "")

    core_data_present = has_pe and has_roce and has_rev

    if not core_data_present:
        return {
            "badge":   "DATA UNKNOWN",
            "color":   "#888",
            "tooltip": "Core fundamental ratios unavailable",
        }
    elif not has_result:
        return {
            "badge":   "RESULT DATE UNKNOWN",
            "color":   "#e67e22",
            "tooltip": "Fundamentals present but earnings date unconfirmed - staleness risk",
        }
    else:
        return {
            "badge":   "DATA VERIFIED",
            "color":   "#27ae60",
            "tooltip": f"Last result: {fin_data['last_result_date']}",
        }


def get_agreement_badge(symbol: str, comparison_map: dict) -> tuple[str, str]:
    """Return (badge_text, streamlit_method_name) for a symbol's model agreement level."""
    if symbol in comparison_map.get("both_agree", set()):
        return "BOTH MODELS AGREE", "success"
    elif symbol in comparison_map.get("anthropic_only", set()):
        return "RISK ANALYST ONLY", "warning"
    else:
        return "CATALYST ANALYST ONLY", "info"


def render_entry_details(
    entry: dict[str, Any],
    target_source_label: str = "",
    is_degraded: bool = False,
) -> None:
    """
    Render entry details with custom HTML so prices never truncate.
    target_source_label: forwarded from payload for R/R caveat (improvement 6E).
    """
    if not entry:
        return

    rr_value  = float(entry.get("risk_reward", 0) or 0)
    rr_border = "#27ae60" if rr_value >= 2.0 else "#e67e22" if rr_value >= MINIMUM_RR_RATIO else "#e74c3c"
    if is_degraded:
        rr_status = "Indicative — not trade-ready"
        rr_symbol = "Model estimate"
    else:
        rr_status = "Strong setup" if rr_value >= 2.0 else "Watch threshold" if rr_value >= MINIMUM_RR_RATIO else "Below minimum threshold"
        rr_symbol = "OK" if rr_value >= 2.0 else "Watch" if rr_value >= MINIMUM_RR_RATIO else "Skip"

    # ── Improvement 6D: stop loss "(model-derived)" label ───────────────────
    stop_label = "model-derived"

    # ── Improvement 6E: R/R caveat based on target source ───────────────────
    _t_src_lower = (target_source_label or "").lower()
    if "model" in _t_src_lower or not target_source_label:
        rr_caveat = " <span style='color:#999;font-size:11px;'>(indicative — model target)</span>"
    else:
        rr_caveat = " <span style='color:#999;font-size:11px;'>(broker/screener target)</span>"

    html = textwrap.dedent(
        f"""
        <div style="
            background:#f8f9fa;
            border:1px solid #e0e0e0;
            border-radius:8px;
            padding:16px;
            margin:8px 0;
            font-family:sans-serif;
        ">
            <table style="width:100%; border-collapse:collapse; table-layout:fixed;">
                <tr style="border-bottom:1px solid #e0e0e0;">
                    <td style="padding:8px; color:#666; font-size:12px;">Current Price</td>
                    <td style="padding:8px; color:#666; font-size:12px;">Entry Price</td>
                    <td style="padding:8px; color:#666; font-size:12px;">Stop Loss</td>
                    <td style="padding:8px; color:#666; font-size:12px;">Target</td>
                </tr>
                <tr>
                    <td style="padding:8px; font-size:clamp(16px, 2vw, 20px); font-weight:600; color:#1a1a1a;">
                        &#8377;{entry['current_price']:,.0f}
                    </td>
                    <td style="padding:8px; font-size:clamp(16px, 2vw, 20px); font-weight:600; color:#1a1a1a;">
                        &#8377;{entry['entry_price']:,.0f}<br>
                        <span style="font-size:11px; color:#e67e22;">
                            &#9660; {entry['discount_from_current']:.1f}% from CMP
                        </span>
                    </td>
                    <td style="padding:8px; font-size:clamp(16px, 2vw, 20px); font-weight:600; color:#c0392b;">
                        &#8377;{entry['stop_loss']:,.0f}<br>
                        <span style="font-size:11px; color:#c0392b;">
                            &#9660; {entry['stop_loss_pct']:.0f}% from entry
                            ({stop_label})
                        </span>
                    </td>
                    <td style="padding:8px; font-size:clamp(16px, 2vw, 20px); font-weight:600; color:#27ae60;">
                        &#8377;{entry['analyst_target']:,.0f}<br>
                        <span style="font-size:11px; color:#27ae60;">
                            &#9650; {entry['upside_from_entry']:.1f}% from entry
                        </span>
                    </td>
                </tr>
            </table>
            <div style="
                margin-top:12px;
                padding:10px;
                background:#fff;
                border-radius:6px;
                border-left:4px solid {rr_border};
            ">
                <strong>Risk/Reward: {rr_value}x</strong>{rr_caveat}
                — For every &#8377;1 risked, potential gain is &#8377;{rr_value:.1f}
                <span style="color:{rr_border}; font-weight:600;">{rr_symbol}: {rr_status}</span>
            </div>
            <div style="margin-top:10px; font-size:13px; color:#555;">
                <strong>Entry Zone:</strong> &#8377;{entry['entry_zone_low']:,.0f} - &#8377;{entry['entry_zone_high']:,.0f}
            </div>
            <div style="
                margin-top:8px;
                padding:10px;
                background:#fffbf0;
                border-radius:6px;
                font-size:13px;
                color:#555;
            ">
                {_html.escape(str(entry['entry_note']))}
            </div>
        </div>
        """
    ).strip()
    st.markdown(html, unsafe_allow_html=True)

    if is_degraded:
        st.warning(
            f"Risk/Reward {rr_value}x — indicative only · model-derived · "
            "not trade-ready · do not treat as execution guidance"
        )
    elif rr_value < MINIMUM_RR_RATIO:
        st.warning(
            f"Risk/Reward {rr_value}x is below minimum threshold of "
            f"{MINIMUM_RR_RATIO}x. Consider skipping or waiting for better entry."
        )
    elif rr_value >= 2.0:
        st.success(f"Strong Risk/Reward {rr_value}x - favourable setup")

    schedule_lines = [
        f"- **Tranche 1 ({entry['tranche_1_pct']}%):** Buy at ₹{entry['entry_price']:,.0f}"
    ]
    if entry.get("tranche_2_pct", 0) > 0:
        schedule_lines.append(
            f"- **Tranche 2 ({entry['tranche_2_pct']}%):** "
            f"Buy at ₹{entry['tranche_2_price']:,.0f} on next dip"
        )
    if entry.get("tranche_3_pct", 0) > 0:
        schedule_lines.append(
            f"- **Tranche 3 ({entry['tranche_3_pct']}%):** "
            f"{entry['tranche_3_trigger']}"
        )
    st.markdown("**Deployment Schedule**")
    st.markdown("\n".join(schedule_lines))


def render_synthesis_with_entry_summary(synthesis_text: str) -> None:
    body, marker, summary = synthesis_text.partition(ENTRY_SUMMARY_MARKER)
    if body.strip():
        st.markdown(body.strip())
    if not marker:
        return

    parts = [segment.strip() for segment in summary.strip().split("|") if segment.strip()]
    cleaned_parts = [
        part.replace("Current", "CMP", 1).replace("Enter at", "Enter", 1)
        for part in parts
    ]
    if cleaned_parts:
        st.info(f"**Entry Summary** - {' | '.join(cleaned_parts)}")


def render_recommendation_card(
    item: dict[str, Any],
    provider: str = "",
    comparison_map: dict | None = None,
    synthesis_text: str = "",
) -> None:
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
    initial_pct = payload.get("initial_tranche_pct", payload.get("allocation_pct", 0))
    target_pct = payload.get("target_pct", 0)
    net_return = payload.get("net_of_tax_return_pct", payload.get("net_of_tax_return_projection", 0))
    entry = payload.get("entry_levels") or calculate_entry_levels(
        symbol=symbol,
        current_price=payload.get("current_price"),
        analyst_target=payload.get("analyst_target"),
        signal=str(payload.get("entry_signal") or action),
        quant_score=payload.get("quality_score"),
        fin_data=payload.get("fin_data") or {},
    )
    news_context = item.get("news_context")
    if not isinstance(news_context, dict):
        payload_news = payload.get("news_context")
        news_context = payload_news if isinstance(payload_news, dict) else {}

    provider_label = ""
    if provider == "anthropic":
        provider_label = "Anthropic Claude"
    elif provider == "openai":
        provider_label = "OpenAI GPT"

    # ── Governance: resolve final canonical state ─────────────────────────────
    _rec_risk_profile = payload.get("risk_profile", "Balanced")
    # Re-apply freshness cap with the current UI risk profile so that the
    # reconciliation badge compares the profile-correct quant signal, not
    # the signal that was capped during the original run (which may have used
    # a different profile).
    _original_entry = payload.get("original_entry_signal") or action
    _fin_data_for_cap = payload.get("fin_data") or {}
    _recapped_action = apply_freshness_cap(
        _original_entry, _fin_data_for_cap, _rec_risk_profile
    )
    gov = resolve_final_recommendation(
        quant_action=_recapped_action,
        anthropic_rationale=str(payload.get("anthropic_rationale") or rationale) if provider == "anthropic" else rationale,
        openai_rationale=str(payload.get("openai_rationale") or rationale) if provider == "openai" else rationale,
        synthesis_text=synthesis_text,
        payload=payload,
        provider=provider,
        risk_profile=_rec_risk_profile,
    )
    canonical_state  = gov["canonical_state"]
    actionability    = gov["actionability"]   # ACTIONABLE | NON_ACTIONABLE | DEGRADED
    show_trade_plan  = actionability != "NON_ACTIONABLE"
    is_degraded      = actionability == "DEGRADED"

    # Display verdict: use final_verdict (governing signal from reconciliation)
    # rather than canonical_state, which may have been silently downgraded by
    # safety gates (data freshness, critical failures) while the reconciliation
    # strip already shows the pre-downgrade agreement. This keeps the badge and
    # the reconciliation message consistent. Actionability / trade-plan logic
    # still uses canonical_state (the fully resolved conservative state).
    _display_verdict = gov.get("final_verdict") or canonical_state

    # ── Profile-aware entry plan visibility override ──────────────────────────
    # Only apply staleness override when canonical_state itself is actionable
    # (i.e. suppression is purely staleness-driven, not a hard AVOID/WATCHLIST/CONFIRMATION).
    _NON_ENTRY_STATES = {"AVOID", "WATCHLIST", "BUY ONLY AFTER CONFIRMATION"}
    _stale_warning = ""
    if not show_trade_plan and canonical_state not in _NON_ENTRY_STATES:
        _ep_result = should_show_entry_plan(payload.get("fin_data") or {}, _rec_risk_profile)
        if _ep_result["show"]:
            show_trade_plan = True
            _stale_warning = _ep_result.get("warning", "")

    # Badge colours for canonical state
    _state_badge_color: dict[str, str] = {
        "AVOID":                      "#c0392b",
        "WATCHLIST":                  "#e67e22",
        "BUY ONLY AFTER CONFIRMATION":"#8e44ad",
        "ACCUMULATE ON DIPS":         "#2980b9",
        "ACCUMULATE GRADUALLY":       "#27ae60",
        "ACTIONABLE BUY":             "#1a7a1a",
    }
    _badge_color = _state_badge_color.get(_display_verdict, "#555")

    with st.container(border=True):
        # ── Verdict stripe — colored top bar gives instant visual signal ──────
        _stripe_alpha = "1.0" if actionability == "ACTIONABLE" else "0.55"
        st.markdown(
            f'<div style="height:4px;background:{_badge_color};opacity:{_stripe_alpha};'
            f'border-radius:2px;margin-bottom:6px;"></div>',
            unsafe_allow_html=True,
        )
        head_col, action_col = st.columns([4, 1])
        with head_col:
            if provider_label:
                st.caption(provider_label)
            st.markdown(f"#### {company} ({symbol})")
            if sector:
                st.caption(sector)
            # ── Confidence / data-status warning badges ───────────────────────
            _conf_badge_color = {"HIGH": "#27ae60", "MODERATE": "#e67e22", "LOW": "#c0392b"}
            _data_badge_color = {"STALE": "#c0392b", "INCONSISTENT": "#c0392b", "UNKNOWN": "#888"}
            _hdr_badges: list[str] = []
            _hdr_cl = gov.get("confidence_level", "")
            _hdr_ds = gov.get("data_status", "")
            if _hdr_cl and _hdr_cl not in ("HIGH", "UNKNOWN"):
                _bc = _conf_badge_color.get(_hdr_cl, "#888")
                _hdr_badges.append(
                    f'<span style="background:{_bc};color:#fff;padding:2px 6px;'
                    f'border-radius:3px;font-size:11px;font-weight:600;">{_hdr_cl} CONFIDENCE</span>'
                )
            if _hdr_ds and _hdr_ds != "OK":
                if _hdr_ds == "UNKNOWN":
                    # Three-state: only show badge when fundamentals are actually missing
                    # or result date is absent — not as a catch-all for missing dates.
                    _db = compute_data_badge(payload.get("fin_data") or {})
                    if _db["badge"] != "DATA VERIFIED":
                        _hdr_badges.append(
                            f'<span style="background:{_db["color"]};color:#fff;padding:2px 6px;'
                            f'border-radius:3px;font-size:11px;font-weight:600;"'
                            f' title="{_db["tooltip"]}">{_db["badge"]}</span>'
                        )
                else:
                    _dc = _data_badge_color.get(_hdr_ds, "#888")
                    _hdr_badges.append(
                        f'<span style="background:{_dc};color:#fff;padding:2px 6px;'
                        f'border-radius:3px;font-size:11px;font-weight:600;">DATA {_hdr_ds}</span>'
                    )
            if _hdr_badges:
                st.markdown(" &nbsp; ".join(_hdr_badges), unsafe_allow_html=True)
        with action_col:
            # Verdict badge — block-level, muted when non-actionable
            _badge_opacity = "1.0" if actionability == "ACTIONABLE" else "0.72"
            st.markdown(
                f'<div style="background:{_badge_color};color:#fff;opacity:{_badge_opacity};'
                f'padding:6px 8px;border-radius:5px;font-size:13px;font-weight:700;'
                f'text-align:center;line-height:1.25;">'
                f'{_display_verdict}</div>',
                unsafe_allow_html=True,
            )

        # ── Governance strip — compact annotation, no verdict echo (badge shows it)
        _act_color = {"ACTIONABLE": "#27ae60", "DEGRADED": "#e67e22", "NON_ACTIONABLE": "#c0392b"}
        _act_label = {"ACTIONABLE": "Actionable", "DEGRADED": "Indicative", "NON_ACTIONABLE": "Not actionable"}
        _conf_reason = gov["confidence_reason"]
        st.markdown(
            f'<div style="border-left:3px solid {_act_color.get(actionability,"#ccc")};'
            f'padding:4px 10px;margin:4px 0 6px 0;font-size:11.5px;color:#666;">'
            f'<span style="color:{_act_color.get(actionability,"#999")};font-weight:600;">'
            f'{_act_label.get(actionability, actionability)}</span>'
            f'&ensp;·&ensp;{_html.escape(_conf_reason)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Consistency failure warnings
        for _fail in gov["consistency_failures"]:
            if _fail["severity"] == "CRITICAL":
                st.error(f"Data integrity: {_fail['message']}")
            else:
                st.warning(f"Note: {_fail['message']}")

        # 52W data quality banner — driven by validate_52w_range() result stored in fin_data
        _w52q = (payload.get("fin_data") or {}).get("52w_data_quality", "")
        if _w52q == "DATA_CORRUPT":
            st.warning("52W range data corrupted — technical entry signals suppressed")
        elif _w52q == "COMPUTED_FROM_HISTORY":
            st.info("52W range computed from 1Y price history (screener fallback via yfinance)")
        elif _w52q == "RANGE_MISMATCH":
            st.warning("Current price is outside stated 52W range — verify data before acting on range signals")
        elif _w52q == "UNAVAILABLE":
            st.warning("52W range data unavailable — range-relative signals omitted")

        # Non-actionable verdict block — styled with reason + upgrade trigger
        if not show_trade_plan:
            _reasons_text = " · ".join(
                gov.get("suppressed_reasons") or ["No actionable trade plan"]
            )
            _upgrade = gov.get("upgrade_trigger", "")
            st.markdown(
                f'<div style="background:#fff8f0;border-left:4px solid #e67e22;'
                f'padding:10px 14px;border-radius:6px;margin:8px 0;font-size:13px;">'
                f'<strong>No actionable trade plan</strong><br>'
                f'<span style="color:#555;">{_html.escape(_reasons_text)}</span>'
                + (
                    f'<br><em style="color:#888;font-size:12px;">'
                    f'Upgrade trigger: {_html.escape(_upgrade)}</em>'
                    if _upgrade else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        # Signal reconciliation badge — quant vs LLM verdict
        _rec = gov.get("reconciliation", {})
        _rec_status = _rec.get("status", "UNKNOWN")
        _rec_msg = _rec.get("message", "")
        _final_verdict = gov.get("final_verdict", "")
        if _rec_status == "ALIGNED":
            st.success(f"\u2705 {_rec_msg}")
        elif _rec_status == "MINOR_DIVERGENCE":
            st.info(f"\u2139\ufe0f {_rec_msg}")
        elif _rec_status == "CONFLICT":
            st.warning(f"\u26a0\ufe0f Signal conflict: {_rec_msg}")
            if _final_verdict:
                st.caption(f"Governing verdict: **{_final_verdict}**")
        # UNKNOWN / NO_SYNTHESIS — show nothing

        if comparison_map:
            badge_text, badge_type = get_agreement_badge(symbol, comparison_map)
            getattr(st, badge_type)(f"Model consensus: {badge_text}")

        if rationale:
            if not show_trade_plan:
                # Collapsed on NON_ACTIONABLE — verdict block above already explains the decision
                with st.expander("Analyst reasoning", expanded=False):
                    st.write(rationale)
            else:
                st.write(rationale)

        # ── Improvement 6A: remove false decimal precision from deploy% ──────
        if news_context.get("material_risks_found"):
            summary = str(news_context.get("summary") or "").strip()
            st.error("Recent News Risk Flags")
            if summary:
                st.caption(summary)
            for flag in news_context.get("flags", []):
                if not isinstance(flag, dict):
                    continue
                severity = str(flag.get("severity") or "LOW").upper()
                flag_type = str(flag.get("type") or "RISK")
                headline = str(flag.get("headline") or "").strip()
                if not headline:
                    continue
                severity_color = {
                    "HIGH": "error",
                    "MEDIUM": "warning",
                    "LOW": "info",
                }.get(severity, "info")
                getattr(st, severity_color)(f"[{flag_type}] {headline}")

            if item.get("news_override") or payload.get("news_override"):
                prior_verdict = (
                    item.get("preliminary_verdict_before_news")
                    or payload.get("preliminary_verdict_before_news")
                    or "ACCUMULATE GRADUALLY"
                )
                override_verdict = (
                    item.get("news_override_verdict")
                    or payload.get("news_override_verdict")
                    or "WATCHLIST"
                )
                st.warning(
                    f"Verdict downgraded from {prior_verdict} to {override_verdict} "
                    "due to material news risk."
                )

        initial_pct_display = f"~{round(float(initial_pct or 0))}%"

        # ── Improvement 6B: target label with source caveat ──────────────────
        target_source_label = str(payload.get("target_source_label") or "model estimate")
        analyst_target_val  = payload.get("analyst_target") or (entry or {}).get("analyst_target", 0)
        upside_from_entry   = (entry or {}).get("upside_from_entry", 0)
        if analyst_target_val and float(analyst_target_val) > 0:
            target_display = (
                f"Analyst target: ₹{float(analyst_target_val):,.0f}"
                f" (~{float(upside_from_entry or 0):.0f}% from entry)"
                f" [{target_source_label}]"
            )
        else:
            target_display = f"Target: {target_pct}%" if target_pct else ""

        # ── Improvement 6C: replace binary confidence band with evidence label
        evidence_label = (
            payload.get("evidence", {}).get("label")
            if payload.get("evidence")
            else ""
        )
        confidence_display = (
            f"{evidence_label} evidence" if evidence_label
            else f"Confidence {confidence_band}"
        )

        if _stale_warning:
            st.warning(_stale_warning)

        if show_trade_plan:
            sizing_label = (
                f"Position size guide: {initial_pct_display} | {target_display}"
                if target_display
                else f"Position size guide: {initial_pct_display}"
            )
            summary_bits = [
                f"Overlap {overlap}%",
                sizing_label,
                confidence_display,
                f"Net Return {net_return}%",
            ]
            st.caption(" | ".join(summary_bits))
            st.caption("_(Position size is model-derived — adjust to your risk tolerance)_")
        else:
            # Non-actionable: only overlap is meaningful — sizing/target/return don't apply
            if overlap:
                st.caption(f"Portfolio overlap: {overlap}%")
        lock_in_warning = str(payload.get("lock_in_warning") or "")
        if lock_in_warning:
            st.error(f"Recently listed stock - {lock_in_warning}")
        tariff_warning = str(payload.get("tariff_warning") or "")
        if tariff_warning:
            st.warning(f"US tariff risk: {tariff_warning}")
        # Momentum signals — always show; staleness is shown separately via DATA STALE badge.
        # Hiding momentum when data is stale defeats the purpose: momentum IS the freshness indicator.
        revenue_momentum = payload.get("revenue_momentum") or payload.get("recent_results") or {}
        momentum = str(
            revenue_momentum.get("momentum")
            or revenue_momentum.get("revenue_momentum")
            or ""
        )
        growth = revenue_momentum.get("growth_pct")
        if growth is None:
            growth = revenue_momentum.get("revenue_yoy_growth_pct")
        period = str(
            revenue_momentum.get("period")
            or revenue_momentum.get("comparison_label")
            or "latest quarter vs same quarter last year"
        )
        if momentum:
            momentum_level = {
                "STRONG": "success",
                "GOOD": "success",
                "MODERATE": "info",
                "WEAK": "warning",
                "DECLINING": "error",
            }.get(momentum, "info")
            message = (
                f"Revenue momentum: {momentum}"
                + (
                    f" - {float(growth):+.1f}% YoY ({period})"
                    if growth is not None
                    else ""
                )
            )
            getattr(st, momentum_level)(message)
        pat_momentum = payload.get("pat_momentum") or {}
        pat_signal = str(pat_momentum.get("pat_momentum") or "")
        pat_growth = pat_momentum.get("pat_growth_pct")
        pat_period = str(pat_momentum.get("period") or period)
        pat_qualifier = str(pat_momentum.get("qualifier") or "").strip()
        if pat_signal and pat_growth is not None:
            pat_level = {
                "STRONG": "success",
                "MODERATE": "info",
                "FLAT": "info",
                "DECLINING": "warning",
                "COLLAPSING": "error",
            }.get(pat_signal, "info")
            getattr(st, pat_level)(
                f"PAT momentum: {pat_signal} - {float(pat_growth):+.1f}% YoY ({pat_period})"
                + (f" {pat_qualifier}" if pat_qualifier else "")
            )
        if pat_momentum.get("rev_pat_divergence"):
            st.warning(
                "Critical divergence: revenue is growing but PAT is falling. "
                "Check for one-time charges, margin compression, or accounting changes before accumulating."
            )
        if show_trade_plan and payload.get("momentum_override_applied"):
            st.info(
                f"Momentum note: base signal {payload.get('original_entry_signal', 'WAIT')} "
                f"adjusted to {payload.get('entry_signal', 'ACCUMULATE')}"
            )

        # ── Trade mechanics: conditional on actionability ─────────────────────
        if show_trade_plan and entry:
            if is_degraded:
                st.markdown(
                    '<div style="background:#fffbf0;border:1px solid #f0c040;border-radius:6px;'
                    'padding:6px 12px;font-size:12px;color:#7a5c00;margin:4px 0;">'
                    '<strong>Indicative only</strong> · Model-derived · Not trade-ready · '
                    'Do not treat as execution guidance</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("**Entry Details**")
            render_entry_details(
                entry,
                target_source_label=str(payload.get("target_source_label") or ""),
                is_degraded=is_degraded,
            )
            entry = None
        elif not show_trade_plan:
            entry = None  # suppress fallback metrics block too

        # PE context and technical signals — input signals, not relevant when not trading
        if show_trade_plan:
            pe_ctx = payload.get("pe_context") or {}
            if pe_ctx.get("pe_current") is not None:
                _pe_color = {
                    "CHEAP_VS_HISTORY":     "success",
                    "FAIR_VS_HISTORY":      "success",
                    "SLIGHT_PREMIUM":       "warning",
                    "EXPENSIVE_VS_HISTORY": "error",
                    "NEUTRAL":              "info",
                }.get(pe_ctx.get("pe_signal", "NEUTRAL"), "info")
                getattr(st, _pe_color)(f"PE Context: {pe_ctx['pe_assessment']}")

            tech_signals = payload.get("technical_signals", [])
            if tech_signals:
                st.markdown("**Technical Signals**")
                _signal_color = {
                    "CONTRARIAN_BUY": "success",
                    "POSITIVE":       "success",
                    "WATCH":          "warning",
                    "CAUTION":        "warning",
                    "NEGATIVE":       "error",
                }
                for sig in tech_signals:
                    color = _signal_color.get(sig["signal"], "info")
                    getattr(st, color)(
                        f"{sig['type']}: {sig['value']} — {sig['note']}"
                    )

        # Fallback metrics block (only shown when primary render_entry_details was skipped
        # AND trade plan is allowed — i.e. entry_levels was absent but actionability permits it)
        if entry and show_trade_plan:
            if is_degraded:
                st.markdown(
                    '<div style="background:#fffbf0;border:1px solid #f0c040;border-radius:6px;'
                    'padding:6px 12px;font-size:12px;color:#7a5c00;margin:4px 0;">'
                    '<strong>Indicative only</strong> · Model-derived · Not trade-ready · '
                    'Do not treat as execution guidance</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("**Entry Details**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{entry['current_price']:,.0f}")
            with col2:
                st.metric(
                    "Entry Price",
                    f"₹{entry['entry_price']:,.0f}",
                    delta=f"-{entry['discount_from_current']:.1f}% from CMP",
                )
            with col3:
                st.metric(
                    "Stop Loss (model-derived)",
                    f"₹{entry['stop_loss']:,.0f}",
                    delta=f"-{entry['stop_loss_pct']:.0f}% from entry",
                )
            with col4:
                st.metric(
                    "Target",
                    f"₹{entry['analyst_target']:,.0f}",
                    delta=f"+{entry['upside_from_entry']:.1f}% from entry",
                )
            st.caption(f"Entry Zone: ₹{entry['entry_zone_low']:,.0f}-₹{entry['entry_zone_high']:,.0f}")
            st.info(entry["entry_note"])

            if entry["risk_reward"] > 0:
                if is_degraded:
                    st.warning(
                        f"Risk/Reward {entry['risk_reward']}x — indicative only · "
                        "model-derived · not trade-ready · do not treat as execution guidance"
                    )
                elif entry["risk_reward"] < MINIMUM_RR_RATIO:
                    st.warning(
                        f"Risk/Reward {entry['risk_reward']}x is below minimum threshold of "
                        f"{MINIMUM_RR_RATIO}x. Consider skipping or waiting for better entry."
                    )
                elif entry["risk_reward"] >= 2.0:
                    st.success(
                        f"Strong Risk/Reward {entry['risk_reward']}x — favourable setup"
                    )
                _rr_src = str(payload.get("target_source_label") or "")
                _rr_caveat = (
                    " _(indicative — model target)_"
                    if "model" in _rr_src.lower() or not _rr_src
                    else " _(broker/screener target)_"
                )
                st.markdown(
                    f"**Risk/Reward: {entry['risk_reward']}x**{_rr_caveat} | "
                    f"For every ₹1 risked, potential gain is about ₹{entry['risk_reward']:.1f}."
                )

            schedule_lines = [
                f"- **Tranche 1 ({entry['tranche_1_pct']}%):** Buy at ₹{entry['entry_price']:,.0f}"
                f"{' now' if entry['tranche_1_pct'] < 100 else ''}"
            ]
            if entry["tranche_2_pct"] > 0:
                schedule_lines.append(
                    f"- **Tranche 2 ({entry['tranche_2_pct']}%):** "
                    f"Buy at ₹{entry['tranche_2_price']:,.0f} on the next dip"
                )
            if entry["tranche_3_pct"] > 0:
                schedule_lines.append(
                    f"- **Tranche 3 ({entry['tranche_3_pct']}%):** {entry['tranche_3_trigger']}"
                )
            st.markdown("**Deployment Schedule**")
            st.markdown("\n".join(schedule_lines))

        if why_for_portfolio:
            st.write(why_for_portfolio)

        # Quant model detail — only shown when trade is being actively considered
        if show_trade_plan and validation_reasoning:
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


def _render_skipped_stocks(skipped: list[dict[str, Any]], run_summary: dict[str, Any] | None = None) -> None:
    """Show the skipped-stocks transparency panel below recommendation cards."""
    _INVESTMENT_CRITERIA_STATUSES = {
        "NEGATIVE_NET_RETURN", "LOW_RISK_REWARD", "GOVERNANCE_RISK",
        "WEAK_EVIDENCE", "NO_PRICE", "OVERLAP_FILTERED",
    }
    _DATA_VALIDATION_STATUSES = {
        "NOT_FOUND", "NO_DATA", "PRICE_MISSING", "DEMERGED", "DELISTED", "STALE_DATA",
    }

    group_deferred    = [s for s in skipped if s.get("status") == "GROUP_CONCENTRATION"]
    investment_gated  = [s for s in skipped if s.get("status") in _INVESTMENT_CRITERIA_STATUSES]
    data_failures     = [s for s in skipped if s.get("status") in _DATA_VALIDATION_STATUSES]
    other_skipped     = [s for s in skipped if s.get("status") not in
                         _INVESTMENT_CRITERIA_STATUSES | _DATA_VALIDATION_STATUSES | {"GROUP_CONCENTRATION"}]

    # Pipeline stats summary — always show if run_summary is available.
    if run_summary:
        stats = run_summary.get("pipeline_stats", {})
        rec_count = run_summary.get("recommendation_count", 0)
        shortlist_count = stats.get("shortlist_count", 0)
        parts = []
        if shortlist_count:
            parts.append(f"Shortlisted: **{shortlist_count}**")
        parts.append(f"Recommended: **{rec_count}**")
        if investment_gated:
            parts.append(f"Excluded by criteria: **{len(investment_gated)}**")
        if data_failures:
            parts.append(f"Data unavailable: **{len(data_failures)}**")
        if parts:
            st.caption("Pipeline: " + "  ·  ".join(parts))

    if group_deferred:
        with st.expander(f"Group concentration: {len(group_deferred)} stock(s) deferred", expanded=False):
            st.caption(
                "These stocks scored well but were deferred because another stock "
                "from the same promoter group already occupies a slot. "
                "The highest-scoring group member is kept."
            )
            for s in group_deferred:
                st.info(f"**{s['symbol']}** — {s['reason']}")

    if investment_gated:
        _status_labels = {
            "NEGATIVE_NET_RETURN": "Analyst target ≤ current price — no net-positive return",
            "LOW_RISK_REWARD":     "Risk/reward ratio below profile minimum",
            "GOVERNANCE_RISK":     "Governance / fraud risk block",
            "WEAK_EVIDENCE":       "Evidence too stale or thin for this risk profile",
            "NO_PRICE":            "Current price unavailable",
            "OVERLAP_FILTERED":    "Already >3% represented via MF/ETF look-through",
        }
        with st.expander(
            f"⚠️ {len(investment_gated)} stock(s) excluded by investment criteria", expanded=True
        ):
            st.caption(
                "These shortlisted stocks were eliminated by the investment-quality gates — "
                "not a data issue, but current prices / analyst targets / evidence don't meet "
                "the bar for this risk profile."
            )
            for sk in investment_gated:
                label = _status_labels.get(sk["status"], sk["status"])
                st.warning(f"**{sk['symbol']}** — {label}: {sk['reason']}")

    if data_failures or other_skipped:
        all_data = data_failures + other_skipped
        with st.expander(f"⚠️ {len(all_data)} stock(s) skipped — data unavailable", expanded=False):
            st.caption(
                "These stocks were considered but excluded because financial data "
                "could not be validated. They will not appear in recommendations "
                "until data is available."
            )
            for sk in all_data:
                st.warning(f"**{sk['symbol']}** — {sk['status']}: {sk['reason']}")
            st.caption(
                "To fix: update `NSE_SYMBOL_MAP` in "
                "`src/stock_platform/utils/symbol_resolver.py` with the correct current ticker."
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


# ── Auth gate ────────────────────────────────────────────────────────────────
_use_auth = _auth_configured()
if _use_auth and not st.user.is_logged_in:
    st.markdown(
        """
        <div class="hero-shell hero-shell-login">
            <div class="hero-kicker-row" style="justify-content:center;">
                <span class="hero-kicker">Secure access</span>
            </div>
            <h1 class="hero-title" style="color:#ffffff;">Portfolio Assistant</h1>
            <p class="hero-copy" style="margin:0.5rem auto 0;">
                Sign in with your Google account to access your personal portfolio,
                buy ideas and stock monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Sign in with Google", use_container_width=True, type="primary"):
            st.login("google")
    st.stop()

if _use_auth:
    _user_email: str = st.user.email
    _user_display: str = getattr(st.user, "name", None) or _user_email
    # Full SHA-256 hex digest (64 chars) — zero collision risk across users.
    # This is the DB filename: data/users/<digest>.db — one file per Google account.
    _user_key = hashlib.sha256(_user_email.encode()).hexdigest()
else:
    _user_email = "local"
    _user_display = "Local"
    _user_key = "local"

engine = get_engine(_user_key)
DB_PATH = engine.config.db_path

if not st.session_state.get("pe_prefetch_done"):
    _universe_members = engine.provider.get_index_members("NIFTY200")
    _universe_symbols = [m["symbol"] for m in _universe_members]
    threading.Thread(
        target=prefetch_pe_history_for_universe,
        args=(_universe_symbols, str(DB_PATH), engine.config.neon_database_url),
        daemon=True,
    ).start()
    st.session_state["pe_prefetch_done"] = True

render_pending_notice()
snapshot = engine.get_dashboard_snapshot()
persisted_comparison = snapshot.get("buy_comparison_result", {})
if persisted_comparison and "comparison_result" not in st.session_state:
    st.session_state["comparison_result"] = persisted_comparison
portfolio = snapshot["portfolio"]
signals = snapshot["signals"]
recommendations = snapshot["recommendations"]
monitoring_actions = snapshot["monitoring_actions"]
run_meta = snapshot.get("run_meta", {})
llm_status = snapshot["llm"]
portfolio_updated_at = portfolio.get("portfolio_meta", {}).get("portfolio_last_updated", "")
monitor_run_meta = run_meta.get("monitoring", {})
monitor_created_at = monitor_run_meta.get("created_at", "")
monitor_provider_label = monitor_run_meta.get("llm_label", "")
monitoring_is_stale = bool(portfolio_updated_at) and (
    not monitor_created_at or monitor_created_at < portfolio_updated_at
)

llm_providers_on = []
if llm_status["anthropic_enabled"]:
    llm_providers_on.append("Anthropic")
if llm_status["openai_enabled"]:
    llm_providers_on.append("OpenAI")
llm_display = " + ".join(llm_providers_on) if llm_providers_on else "Fallback"
openai_ok, openai_message = get_openai_connection_status(
    engine.config.openai_api_key,
    engine.config.openai_fast_model,
    engine.config.openai_reasoning_model,
)
if openai_ok:
    st.sidebar.success("OpenAI ✓")
else:
    st.sidebar.error(f"OpenAI ✗ {openai_message}")

# ── Cache health widget ───────────────────────────────────────────────────────
with st.sidebar.expander("Cache DB health", expanded=False):
    try:
        _cc = get_cache_row_counts()
        _be = _cc.get("backend", "?")
        _be_label = "Neon" if "postgresql" in str(_be) else "SQLite"
        st.caption(f"Backend: **{_be_label}**")
        st.caption(f"result_date_cache: {_cc.get('result_date_cache', '?')} rows")
        st.caption(f"pe_history_cache:  {_cc.get('pe_history_cache', '?')} rows")
        st.caption(f"fii_dii_cache:     {_cc.get('fii_dii_cache', '?')} rows")
    except Exception as _cce:
        st.caption(f"Cache check failed: {_cce}")

holding_count = len(portfolio["raw_holdings"])
gap_count = len(portfolio["identified_gaps"])
recommendation_count = len(recommendations)
signal_count = sum(len(value) for value in signals.values())

if "monitoring_llm_provider" not in st.session_state:
    st.session_state["monitoring_llm_provider"] = (
        "anthropic" if llm_status["anthropic_enabled"] else
        "openai" if llm_status["openai_enabled"] else
        "anthropic"
    )

_hero_col, _logout_col = st.columns([5, 1])
with _hero_col:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker-row">
                <span class="hero-kicker">Portfolio intelligence</span>
                <span class="hero-pill">LLM stack: {_html.escape(llm_display)}</span>
            </div>
            <h1 class="hero-title">Portfolio Assistant</h1>
            <p class="hero-copy">
                {_html.escape(_user_display)}
                {f'&nbsp;<span style="opacity:0.65;font-size:0.85em;">({_html.escape(_user_email)})</span>' if _use_auth else ''}
                &nbsp;·&nbsp; Upload statement · Buy ideas · Monitor stocks &nbsp;·&nbsp;
                LLM: {_html.escape(llm_display)}
            </p>
            <div class="hero-stat-row">
                <span class="hero-stat"><strong>{holding_count}</strong> tracked holdings</span>
                <span class="hero-stat"><strong>{signal_count}</strong> signal rows</span>
                <span class="hero-stat"><strong>{recommendation_count}</strong> current ideas</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with _logout_col:
    if _use_auth:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            st.logout()

summary_cols = st.columns(4)
with summary_cols[0]:
    render_metric("Tracked Holdings", str(holding_count), "All ingested portfolio rows")
with summary_cols[1]:
    render_metric("Sector Gaps", str(gap_count), "Detected from exposure analysis")
with summary_cols[2]:
    render_metric("Buy Ideas", str(recommendation_count), "Most recent recommendation feed")
with summary_cols[3]:
    render_metric("Signals", str(signal_count), "Across all signal families")

tabs = st.tabs(["Overview", "Portfolio", "Buy Ideas", "Monitoring", "Signals"])

with tabs[0]:
    render_section_header(
        "Quick Snapshot",
        "Dashboard Overview",
        "See the current portfolio shape, sector gaps, and pipeline readiness before you drill into each workflow.",
    )
    st.info("Upload your statement in the **Portfolio** tab — your data is saved for your account. Then go to **Buy Ideas** and **Monitoring**.")

    # ── DB backend status badge — direct psycopg2 probe ──────────────────────
    def _probe_neon() -> tuple[str, str]:
        """
        Directly test the Neon connection without going through AppConfig.
        Returns (status, detail) where status is 'ok', 'no_url', 'no_psycopg2', or 'error'.
        """
        # Try all secret sources in priority order
        _url = ""
        try:
            _url = st.secrets.get("NEON_DATABASE_URL", "") or ""
        except Exception:
            pass
        if not _url:
            _url = _os.environ.get("NEON_DATABASE_URL", "") or ""
        if not _url:
            try:
                _p = ROOT / ".streamlit" / "secrets.toml"
                if _p.exists():
                    with _p.open("rb") as _f:
                        _url = _tomllib.load(_f).get("NEON_DATABASE_URL", "") or ""
            except Exception:
                pass
        _url = _url.strip()
        if not _url:
            return "no_url", ""
        try:
            import psycopg2 as _pg2
        except ImportError as _ie:
            return "no_psycopg2", str(_ie)
        try:
            _c = _pg2.connect(_url)
            _c.close()
            return "ok", ""
        except Exception as _ce:
            return "error", str(_ce)

    _neon_status, _neon_detail = _probe_neon()
    if _neon_status == "ok":
        st.success("Database: **Neon PostgreSQL** — data persists across deployments.")
    elif _neon_status == "no_url":
        st.warning(
            "Database: **Local SQLite** (ephemeral) — "
            "`NEON_DATABASE_URL` not found in Streamlit Cloud secrets. "
            "Go to **Manage app → Settings → Secrets** and add it."
        )
    elif _neon_status == "no_psycopg2":
        st.warning(
            f"Database: **Local SQLite** — `psycopg2` not installed (`{_neon_detail}`). "
            "Add `psycopg2-binary` to requirements.txt."
        )
    else:
        st.warning(
            f"Database: **Local SQLite** — Neon connection failed: `{_neon_detail}`. "
            "Check the `NEON_DATABASE_URL` value in Streamlit Cloud secrets."
        )
    top_cols = st.columns([1.15, 0.85])
    with top_cols[0]:
        st.subheader("Portfolio Exposure")
        exposure_frame = _df(portfolio["normalized_exposure"], ["symbol", "company_name", "sector", "total_weight"])
        if len(exposure_frame):
            st.dataframe(exposure_frame, use_container_width=True, hide_index=True)
        else:
            render_empty_panel("No normalized exposure yet. Load the sample portfolio or upload a statement to populate this view.")
    with top_cols[1]:
        st.subheader("Sector Gaps")
        gap_frame = _df(portfolio["identified_gaps"], ["sector", "underweight_pct", "conviction", "score", "reason"])
        if len(gap_frame):
            st.dataframe(gap_frame, use_container_width=True, hide_index=True)
        else:
            render_empty_panel("Gap analysis will appear after portfolio ingestion and signal refresh.")

    with st.expander("Diagnostics", expanded=False):
        st.caption("Portal QA — confirms the app is receiving valid data from each pipeline stage.")
        checks = build_render_checks(snapshot)
        qa_cols = st.columns(2)
        for idx, check in enumerate(checks):
            with qa_cols[idx % 2]:
                render_check_card(check["title"], check["passed"], check["detail"])
        st.markdown("#### Data Shape")
        preview = {
            "portfolio_rows": len(portfolio["normalized_exposure"]),
            "gap_rows": len(portfolio["identified_gaps"]),
            "recommendation_rows": len(recommendations),
            "monitoring_rows": len(monitoring_actions),
            "signal_rows": {key: len(value) for key, value in signals.items()},
        }
        st.json(preview)

    # ── PE History Cache refresh — top-level expander so it is easy to find ──
    with st.expander("PE History Cache — Refresh All Indices", expanded=False):
        st.caption(
            "Pre-fetches historical PE ratio data (Screener.in → Wisesheets → yfinance) "
            "for every stock across all indices — NIFTY50 through NIFTYMIDSMALLCAP400. "
            "Cached for 7 days in Neon. Run once after deployment so the buy pipeline "
            "finds warm PE data immediately."
        )

        if _PE_CACHE_JOB["running"]:
            st.info(
                f"Refresh running in background — {_PE_CACHE_JOB['count']} symbols queued. "
                "You can continue using the app."
            )
            if st.button("Check Status", key="pe_cache_check_btn"):
                st.rerun()
        elif _PE_CACHE_JOB["done"]:
            if _PE_CACHE_JOB["error"]:
                st.warning(f"Refresh finished with error: {_PE_CACHE_JOB['error']}")
            else:
                _saved = _PE_CACHE_JOB.get("saved", 0)
                _skipped = _PE_CACHE_JOB.get("skipped", 0)
                _failed = _PE_CACHE_JOB.get("failed", 0)
                if _saved == 0 and _failed > 0:
                    st.warning(
                        f"Refresh complete but **0 symbols were saved** — "
                        f"{_failed} fetch attempts returned no data (all sources failed). "
                        f"{_skipped} were already cached. "
                        "Check Streamlit logs for per-symbol errors."
                    )
                else:
                    st.success(
                        f"Refresh complete — {_saved} saved, "
                        f"{_skipped} already cached, {_failed} failed "
                        f"(out of {_PE_CACHE_JOB['count']} unique symbols)."
                    )
            if st.button("Refresh Again", key="pe_cache_refresh_again_btn", use_container_width=True):
                _PE_CACHE_JOB.update({
                    "running": False, "done": False,
                    "count": 0, "saved": 0, "skipped": 0, "failed": 0,
                    "error": "",
                })
                st.rerun()
        else:
            if st.button(
                "Refresh PE Cache — All Indices",
                key="pe_cache_refresh_btn",
                use_container_width=True,
                help="Fetches historical PE data for every stock in all selectable indices.",
            ):
                _all_symbols: set[str] = set()
                for _idx_name, _idx_cfg in SELECTABLE_INDICES.items():
                    try:
                        _members = engine.provider.get_index_members(_idx_cfg["code"])
                        _all_symbols.update(m["symbol"] for m in _members)
                    except Exception as _exc:
                        st.warning(f"Could not load {_idx_name}: {_exc}")

                _symbol_list = sorted(_all_symbols)
                _PE_CACHE_JOB.update({"running": True, "done": False, "count": len(_symbol_list), "error": ""})

                def _run_cache_refresh(symbols: list[str], db_path: str, neon_database_url: str) -> None:
                    try:
                        stats = prefetch_pe_history_for_universe(symbols, db_path, neon_database_url)
                        _PE_CACHE_JOB.update({
                            "running": False, "done": True,
                            "saved": stats["saved"],
                            "skipped": stats["skipped"],
                            "failed": stats["failed"],
                        })
                    except Exception as exc:
                        _PE_CACHE_JOB.update({"running": False, "done": True, "error": str(exc)})

                threading.Thread(
                    target=_run_cache_refresh,
                    args=(_symbol_list, str(DB_PATH), engine.config.neon_database_url),
                    daemon=True,
                ).start()
                st.rerun()

with tabs[1]:
    render_section_header(
        "Upload And Ingest",
        "Portfolio Workspace",
        "Upload JSON, CSV, or an encrypted NSDL / CAS PDF. PDF uploads start ingestion automatically after parsing.",
    )

    pdf_password = st.text_input(
        "PDF password",
        type="password",
        help="Required for encrypted NSDL / CAS statements.",
    )
    uploaded = st.file_uploader("Upload portfolio file", type=["json", "csv", "pdf"])
    uploaded_payload = parse_upload(uploaded, pdf_password, engine)

    # Auto-extract buying prices from any CSV uploaded here (broker format detection).
    if uploaded and Path(uploaded.name).suffix.lower() == ".csv":
        _csv_sig = f"broker_autoimport_{uploaded.name}_{len(uploaded.getvalue())}"
        if st.session_state.get("broker_autoimport_sig") != _csv_sig:
            try:
                import os, tempfile
                from utils.broker_parser import parse_broker_file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as _tmp:
                    _tmp.write(uploaded.getvalue())
                    _tmp_path = _tmp.name
                try:
                    _broker_holdings = parse_broker_file(_tmp_path)
                finally:
                    os.unlink(_tmp_path)
                if _broker_holdings:
                    engine.repo.upsert_direct_equity_holdings(_broker_holdings)
                    st.session_state["broker_autoimport_sig"] = _csv_sig
                    st.session_state["broker_holdings_refreshed"] = True
                    # No st.rerun() here — a rerun before tabs[3] would swallow
                    # any Monitoring button click. The snapshot refreshes on the
                    # next natural user action instead.
            except Exception as _exc:
                pass  # Silent fail — broker parsing is best-effort

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

    st.subheader("Direct Equity Holdings — Broker Statement")
    st.caption(
        "Upload your broker holding statement to enable profit/loss tracking and tax-aware exit "
        "recommendations. Supported: Zerodha Console CSV, Groww Holdings CSV, ICICI Direct PDF, "
        "or any CSV with symbol + avg price columns."
    )
    broker_file = st.file_uploader(
        "Upload broker holding statement",
        type=["csv", "pdf"],
        key="broker_upload",
        help="Zerodha: Console -> Holdings -> Download CSV",
    )

    if st.session_state.pop("broker_holdings_refreshed", False):
        latest_direct_equity_holdings = engine.repo.list_direct_equity_holdings()
    else:
        latest_direct_equity_holdings = portfolio.get("direct_equity_holdings", [])

    if broker_file:
        import os
        import tempfile

        suffix = ".csv" if broker_file.name.lower().endswith(".csv") else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(broker_file.getvalue())
            tmp_path = tmp.name
        try:
            from utils.broker_parser import parse_broker_file

            holdings = parse_broker_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        if holdings:
            saved = engine.repo.upsert_direct_equity_holdings(holdings)
            latest_direct_equity_holdings = engine.repo.list_direct_equity_holdings()
            st.success(f"Saved {saved} holdings with buying prices")
            preview_df = pd.DataFrame(holdings)[["symbol", "quantity", "avg_buy_price", "buy_date"]]
            preview_df.columns = ["Symbol", "Qty", "Avg Buy Rs", "Buy Date"]
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not parse the broker statement. Check the detected columns below.")
            if broker_file.name.lower().endswith(".csv"):
                try:
                    df_debug = pd.read_csv(io.BytesIO(broker_file.getvalue()))
                    st.write("Columns found:", list(df_debug.columns))
                except Exception:
                    pass

    with st.expander("Enter buying price manually for a single stock"):
        st.caption(
            "For TCS, enter `TCS` plus your actual quantity, avg buy price, and buy date, "
            "then click Save holding."
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            m_symbol = st.text_input("Symbol e.g. INFY", key="manual_broker_symbol")
        with col2:
            m_qty = st.number_input("Quantity", min_value=1, value=1, key="manual_broker_qty")
        with col3:
            m_price = st.number_input("Avg buy price Rs", min_value=0.0, value=0.0, key="manual_broker_price")
        with col4:
            m_date = st.date_input("Buy date", key="manual_broker_date")
        if st.button("Save holding", key="manual_broker_save"):
            if m_symbol and m_price > 0:
                engine.repo.upsert_direct_equity_holdings(
                    [
                        {
                            "symbol": m_symbol.upper(),
                            "quantity": m_qty,
                            "avg_buy_price": m_price,
                            "current_price": None,
                            "buy_date": str(m_date),
                            "source": "manual",
                        }
                    ],
                )
                latest_direct_equity_holdings = engine.repo.list_direct_equity_holdings()
                st.success(f"Saved {m_symbol.upper()} at Rs{m_price}")

    deh = latest_direct_equity_holdings
    if deh:
        st.subheader("Saved Buying Prices")
        deh_df = pd.DataFrame(deh)[["symbol", "quantity", "avg_buy_price", "buy_date"]]
        deh_df.columns = ["Symbol", "Qty", "Avg Buy Rs", "Buy Date"]
        st.dataframe(deh_df, use_container_width=True, hide_index=True)

    existing_prefs = portfolio["user_preferences"]
    default_payload = uploaded_payload or {
        "macro_thesis": existing_prefs.get("macro_thesis", ""),
        "investable_surplus": existing_prefs.get("investable_surplus", 0),
        "direct_equity_corpus": existing_prefs.get("direct_equity_corpus", 0),
        "mutual_funds": [row["payload"] for row in portfolio["raw_holdings"] if row["holding_type"] == "mutual_fund"],
        "etfs": [row["payload"] for row in portfolio["raw_holdings"] if row["holding_type"] == "etf"],
        "direct_equities": [row["payload"] for row in portfolio["raw_holdings"] if row["holding_type"] == "direct_equity"],
    }
    default_payload["direct_equities"] = apply_saved_buy_prices(
        default_payload.get("direct_equities", []),
        latest_direct_equity_holdings,
    )

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
                _df(default_payload.get("direct_equities", []), ["instrument_name", "symbol", "quantity", "avg_buy_price", "market_value"]),
                num_rows="dynamic",
                use_container_width=True,
                key="direct_editor",
            )
            st.caption(
                "Click **Save And Ingest Portfolio** to run sector gap analysis and generate buy ideas. "
                "Your portfolio is already saved from the CSV upload — click this only to re-run the analysis "
                "after editing the tables above."
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
    render_section_header(
        "Buy Ideas",
        "Recommendation Studio",
        "Run the existing buy-idea flow with Anthropic, OpenAI, or both providers side by side.",
    )

    # ── Market Flow badge ─────────────────────────────────────────────────────
    try:
        _flow = fetch_fii_dii_sector_flow(
            neon_database_url=_os.environ.get("NEON_DATABASE_URL", ""),
        )
        _msig = _flow.get("market_signal", "UNKNOWN")
        _fii  = _flow.get("fii_net_5d_cr") or 0
        _dii  = _flow.get("dii_net_5d_cr") or 0
        _src  = _flow.get("source", "")
        _asof = _flow.get("as_of", "")
        _fii_str = f"+\u20b9{abs(_fii):,.0f}Cr" if _fii >= 0 else f"-\u20b9{abs(_fii):,.0f}Cr"
        _dii_str = f"+\u20b9{abs(_dii):,.0f}Cr" if _dii >= 0 else f"-\u20b9{abs(_dii):,.0f}Cr"
        _flow_msg = (
            f"Market Flow ({_asof}):  "
            f"FII {_fii_str}  |  DII {_dii_str}  |  "
            f"Signal: {_msig}  |  via {_src}"
        )
        _badge_fn = {
            "RISK_ON":  st.success,
            "NEUTRAL":  st.info,
            "CAUTIOUS": st.warning,
            "RISK_OFF": st.error,
        }.get(_msig, st.info)
        _badge_fn(_flow_msg)
    except Exception:
        pass
    # ─────────────────────────────────────────────────────────────────────────

    provider_choice = st.radio(
        "LLM provider",
        options=["Anthropic Claude", "OpenAI GPT", "Compare Both"],
        index=0,
        horizontal=True,
        key="provider_radio",
    )
    st.session_state["global_llm_provider"] = (
        "anthropic" if provider_choice == "Anthropic Claude" else
        "openai" if provider_choice == "OpenAI GPT" else
        "anthropic"
    )
    if provider_choice in ("Anthropic Claude", "Compare Both"):
        _a_status = "Configured" if llm_status["anthropic_enabled"] else "API key missing — add ANTHROPIC_API_KEY to secrets"
        st.caption(
            f"**Anthropic Claude** — Fast: `{llm_status['anthropic_fast_model']}` · "
            f"Reasoning: `{llm_status['anthropic_reasoning_model']}` · {_a_status}"
        )
    if provider_choice in ("OpenAI GPT", "Compare Both"):
        _o_status = "Configured" if llm_status["openai_enabled"] else "API key missing — add OPENAI_API_KEY to secrets"
        st.caption(
            f"**OpenAI GPT** — Fast: `{llm_status['openai_fast_model']}` · "
            f"Reasoning: `{llm_status['openai_reasoning_model']}` · {_o_status}"
        )

    with st.form("buy-form"):
        c1, c2 = st.columns(2)
        with c1:
            _selectable = list(SELECTABLE_INDICES.keys())
            selected_display = st.selectbox(
                "Index universe",
                options=_selectable,
                index=_selectable.index(DEFAULT_INDEX),
                help=(
                    "Larger index = more candidates = better gap-filling. "
                    "Use sectoral indices for targeted searches."
                ),
            )
            _sel_cfg = SELECTABLE_INDICES[selected_display]
            st.caption(f"{_sel_cfg['description']} · {_sel_cfg['count']} stocks in universe")
        with c2:
            risk_profile = st.selectbox("Risk Profile", ["Conservative", "Balanced", "Aggressive"])
        c3, c4 = st.columns(2)
        with c3:
            horizon_months = st.slider("Horizon (months)", 6, 36, 18)
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
                "index_name": _sel_cfg["code"],
                "horizon_months": horizon_months,
                "risk_profile": risk_profile,
                "top_n": top_n,
            }
            _buy_exc: Exception | None = None
            try:
                comp = run_buy_workflow(engine, buy_request, provider_choice=provider_choice)
                if provider_choice == "Compare Both":
                    st.session_state["comparison_result"] = comp
                    st.session_state.pop("buy_blocked_reason", None)
                    push_notice("Recommendation comparison complete.", "success")
                else:
                    st.session_state.pop("comparison_result", None)
                    blocked = (comp or {}).get("run_summary", {}).get("blocked_reason", "")
                    if blocked:
                        st.session_state["buy_blocked_reason"] = blocked
                        push_notice("Market signals too weak — no recommendations generated.", "warning")
                    else:
                        st.session_state.pop("buy_blocked_reason", None)
                        push_notice("Recommendation run complete.", "success")
            except Exception as exc:
                _buy_exc = exc
            if _buy_exc is not None:
                st.error(str(_buy_exc))
            else:
                st.rerun()

    if "comparison_result" in st.session_state:
        st.subheader("Provider Comparison")
        comp: dict[str, Any] = st.session_state["comparison_result"]
        agreement: dict = comp.get("agreement", {})

        # Hoist synthesis_map so card renderers can receive synthesis_text per symbol
        synthesis_map: dict[str, str] = comp.get("synthesis", {})

        # ── Candidate pipeline diagnostic ────────────────────────────────────
        _a_stats = comp.get("anthropic", {}).get("run_summary", {}).get("pipeline_stats") or {}
        _o_stats = comp.get("openai", {}).get("run_summary", {}).get("pipeline_stats") or {}
        _diag_stats = _a_stats or _o_stats
        if _diag_stats:
            _universe   = _diag_stats.get("universe_size", "?")
            _scored     = _diag_stats.get("post_scoring_count", "?")
            _excl       = _diag_stats.get("post_exclusion_count", "?")
            _a_short    = _a_stats.get("shortlist_count", "?")
            _o_short    = _o_stats.get("shortlist_count", "?")
            _top_n_req  = comp.get("anthropic", {}).get("run_summary", {}).get("recommendation_count", "?")
            with st.expander("🔍 Candidate Pipeline Diagnostic"):
                st.caption(
                    f"Universe: **{_universe}** stocks"
                    f" → After quant scoring: **{_scored}**"
                    f" → After hard exclusion: **{_excl}**"
                    f" → Anthropic pool: **{_a_short}**"
                    f" → OpenAI pool: **{_o_short}**"
                )
                if isinstance(_excl, int) and isinstance(_universe, int) and _excl < 10:
                    st.warning(
                        f"⚠️ Only **{_excl}** stocks survived hard exclusion filters "
                        f"(pledge >50%, D/E >5x, market cap <₹1000 Cr, illiquidity). "
                        "Consider using a larger index universe (e.g. NIFTY 500)."
                    )
                st.caption(
                    "Hard exclusions are universal (all profiles): illiquidity <₹5 Cr/day, "
                    "promoter pledge >50%, D/E >5x, market cap <₹1000 Cr. "
                    "Profile thresholds (ROCE, D/E preferred levels) are scoring modifiers only — "
                    "they never reduce the candidate pool."
                )

        # ── SYNTHESIS FIRST — final resolved verdicts dominate the view ──────
        if synthesis_map:
            st.markdown("#### Final Synthesis")
            st.caption(
                "Contrarian risk view (Anthropic) vs. momentum catalyst view (OpenAI) — "
                "synthesised by Claude Sonnet. This section reflects the final resolved verdict."
            )
            _synth_state_colors: dict[str, str] = {
                "AVOID":                      "#c0392b",
                "WATCHLIST":                  "#e67e22",
                "BUY ONLY AFTER CONFIRMATION":"#8e44ad",
                "ACCUMULATE ON DIPS":         "#2980b9",
                "ACCUMULATE GRADUALLY":       "#27ae60",
                "ACTIONABLE BUY":             "#1a7a1a",
            }
            conviction_order = agreement.get("conviction_order", list(synthesis_map.keys()))
            for symbol in conviction_order:
                synthesis_text = synthesis_map.get(symbol)
                if not synthesis_text:
                    continue
                synth_canon, synth_conf = extract_synthesis_verdict(synthesis_text)
                _sv_color = _synth_state_colors.get(synth_canon, "#555")
                if symbol in agreement.get("both_agree", set()):
                    agree_tag = "Both analysts agree"
                    agree_color = "#27ae60"
                elif symbol in agreement.get("anthropic_only", set()):
                    agree_tag = "Risk analyst only — catalyst analyst disagrees"
                    agree_color = "#e67e22"
                else:
                    agree_tag = "Catalyst analyst only — risk analyst disagrees"
                    agree_color = "#8e44ad"
                _conf_tag = f" · {synth_conf} confidence" if synth_conf else ""
                # st.expander doesn't support HTML — render the styled header
                # with unsafe_allow_html, then put content in a plain expander.
                st.markdown(
                    f'<div style="margin:6px 0 2px 0;">'
                    f'<strong>{_html.escape(symbol)}</strong>'
                    f' &mdash; '
                    f'<span style="background:{_sv_color};color:#fff;padding:2px 8px;'
                    f'border-radius:3px;font-size:12px;font-weight:600;">'
                    f'{_html.escape(synth_canon or "-")}</span>'
                    f'&ensp;<span style="color:{agree_color};font-size:12px;">'
                    f'{_html.escape(agree_tag)}{_html.escape(_conf_tag)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Show full synthesis", expanded=True):
                    render_synthesis_with_entry_summary(synthesis_text)
                    if symbol.upper().replace(".NS", "") in ELEVATED_GOVERNANCE_RISK:
                        st.caption(
                            "⚠️ Adani Group stocks carry elevated governance risk. "
                            "This recommendation requires your independent judgment "
                            "on group-level political and regulatory exposure."
                        )
            st.markdown("---")

        # ── Model agreement summary ───────────────────────────────────────────
        if agreement:
            n_both = len(agreement.get("both_agree", set()))
            n_a_only = len(agreement.get("anthropic_only", set()))
            n_o_only = len(agreement.get("openai_only", set()))
            st.info(
                f"**Model Agreement** — "
                f"Both agree: {n_both} stocks | "
                f"Risk analyst only: {n_a_only} stocks | "
                f"Catalyst analyst only: {n_o_only} stocks"
            )

        # ── Analyst detail view — inputs, not final truth ─────────────────────
        st.markdown("#### Analyst Detail View")
        st.caption("Individual analyst inputs — synthesis above reflects the final resolved verdict.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<span class="provider-badge provider-badge-anthropic">Anthropic Claude</span>', unsafe_allow_html=True)
            a_data = comp.get("anthropic", {})
            if a_data.get("error"):
                st.warning(a_data["error"])
            elif a_data.get("run_summary", {}).get("blocked_reason"):
                st.warning(a_data["run_summary"]["blocked_reason"])
            elif not a_data.get("recommendations"):
                render_empty_panel("No Anthropic recommendations were generated.")
            else:
                for item in a_data["recommendations"]:
                    render_recommendation_card(
                        item, provider="anthropic", comparison_map=agreement,
                        synthesis_text=synthesis_map.get(item.get("symbol", ""), ""),
                    )
        with col_b:
            st.markdown('<span class="provider-badge provider-badge-openai">OpenAI GPT</span>', unsafe_allow_html=True)
            o_data = comp.get("openai", {})
            if o_data.get("error"):
                st.warning(o_data["error"])
            elif o_data.get("run_summary", {}).get("blocked_reason"):
                st.warning(o_data["run_summary"]["blocked_reason"])
            elif not o_data.get("recommendations"):
                render_empty_panel("No OpenAI recommendations were generated.")
            else:
                for item in o_data["recommendations"]:
                    render_recommendation_card(
                        item, provider="openai", comparison_map=agreement,
                        synthesis_text=synthesis_map.get(item.get("symbol", ""), ""),
                    )
        _render_skipped_stocks(comp.get("skipped_stocks", []))
        if st.button("Clear comparison view"):
            st.session_state.pop("comparison_result", None)
            engine.repo.set_state("buy_comparison_result", {})
            st.rerun()
    else:
        _blocked = st.session_state.get("buy_blocked_reason", "")
        if _blocked:
            st.warning(f"**No recommendations — market signals too weak.**\n\n{_blocked}")
        elif recommendations:
            for item in recommendations:
                render_recommendation_card(item)
        else:
            render_empty_panel("Run the buy workflow to populate the recommendation feed.")
        _render_skipped_stocks(
            st.session_state.get("buy_skipped_stocks", []),
            run_summary=st.session_state.get("buy_run_summary"),
        )

with tabs[3]:
    render_section_header(
        "Actions + Behaviour",
        "Monitoring Desk",
        "Track direct holdings and watchlist names, then refresh monitoring actions for the latest ingested portfolio.",
    )

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
        _m_status = "Configured" if llm_status["anthropic_enabled"] else "API key missing — add ANTHROPIC_API_KEY"
        st.caption(
            f"**Anthropic Claude** — Fast: `{llm_status['anthropic_fast_model']}` · "
            f"Reasoning: `{llm_status['anthropic_reasoning_model']}` · {_m_status}"
        )
    else:
        _m_status = "Configured" if llm_status["openai_enabled"] else "API key missing — add OPENAI_API_KEY"
        st.caption(
            f"**OpenAI GPT** — Fast: `{llm_status['openai_fast_model']}` · "
            f"Reasoning: `{llm_status['openai_reasoning_model']}` · {_m_status}"
        )

    _overlap_rows = snapshot["portfolio"].get("overlap_scores", [])
    if not snapshot["portfolio"].get("raw_holdings"):
        st.warning(
            "Portfolio context missing. Please upload a CAMS statement first "
            "so overlap scores can be computed before monitoring runs."
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
            st.exception(exc)  # show full traceback so root cause is visible

    if portfolio_updated_at:
        freshness_bits = [f"Portfolio updated: {portfolio_updated_at}"]
        if monitor_created_at:
            freshness_bits.append(f"Last monitoring run: {monitor_created_at}")
            if monitor_provider_label:
                freshness_bits.append(f"LLM: {monitor_provider_label}")
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

    _all_raw = portfolio.get("raw_holdings", [])
    direct_holdings = [r for r in _all_raw if r.get("holding_type") == "direct_equity"]
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
            wl_c1, wl_c2 = st.columns(2)
            with wl_c1:
                wl_symbol = st.text_input("Symbol (e.g. INFY)")
            with wl_c2:
                wl_name = st.text_input("Company name")
            wl_c3, wl_c4 = st.columns(2)
            with wl_c3:
                wl_sector = st.text_input("Sector (optional)")
            with wl_c4:
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
    if monitor_provider_label:
        st.caption(f"Latest monitoring run used {monitor_provider_label}.")
    if monitoring_actions:
        monitor_frame = pd.DataFrame(
            [
                {
                    "llm": monitor_provider_label or "Unknown",
                    "symbol": item["symbol"],
                    "action": item["action"],
                    "severity": item["severity"],
                    "urgency": item.get("urgency", "LOW"),
                    "overlap_pct": item.get("overlap_pct", 0.0),
                    "rationale": item["rationale"],
                }
                for item in monitoring_actions
            ]
        )
        st.dataframe(
            monitor_frame,
            use_container_width=True,
            hide_index=True,
            height=min(600, len(monitor_frame) * 35 + 38),
        )
    else:
        render_empty_panel(
            "Monitoring has not been run for the current portfolio yet. Use the Run Monitoring button above."
        )

with tabs[4]:
    render_section_header(
        "Signal Intelligence Layer",
        "Macro Signal Feed",
        "Refresh and inspect the signal families that shape gap analysis, monitoring, and recommendation runs.",
    )
    if st.button("Refresh Signals", use_container_width=True):
        try:
            engine.run_signal_refresh(trigger="manual")
            push_notice("Signals refreshed.", "success")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
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
        render_empty_panel("No signals available yet. Refresh signals or load the sample portfolio.")

