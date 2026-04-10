# Stock LangGraph Platform

This project turns the `stock_platform_v5.puml` sequence diagram into a working Python scaffold with:

- `LangGraph` workflows for signal refresh, portfolio ingestion, new-buy recommendations, and monitoring
- `SQLite` persistence for signals, look-through exposure, overlap, gaps, recommendations, and monitoring actions
- `Streamlit` portal with a mobile-friendly layout
- A deterministic local demo data provider so the platform can run without live API keys

## Architecture

The implementation mirrors the major flows in the UML:

- `FLOW 0`: signal intelligence pipeline
- `FLOW 1`: portfolio ingestion and normalization
- `FLOW 2`: portfolio-personalized buy recommendation flow
- `FLOW 3`: monitoring, decisioning, and behavioural guardrails

Key modules:

- `src/stock_platform/agents/`: stage-specific agent logic
- `src/stock_platform/graphs/`: LangGraph workflow definitions
- `src/stock_platform/data/`: SQLite schema and repository
- `src/stock_platform/providers/demo.py`: deterministic local data source
- `streamlit_app.py`: Streamlit portal

### Monitoring scope

Buy recommendations consider **all** portfolio holdings — MF, ETF, and direct equities — so overlap and gap analysis reflects your full exposure.

Monitoring is intentionally scoped to a narrower universe:

| Source | Included in monitoring |
|---|---|
| Direct equities (from statement PDF) | Yes |
| Manual watchlist | Yes |
| Mutual fund look-through holdings | No — managed by the fund manager |
| ETF look-through holdings | No — managed by the fund manager |

The `monitoring_watchlist` table (`symbol`, `company_name`, `sector`, `note`) lets you add any stock you want to track even if it is not in your direct holdings.

## Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -e .
```

3. Configure one or both LLM providers in `.env`:

```env
# ── Anthropic Claude (recommended) ──────────────────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# Optional model overrides (defaults shown):
# LLM_FAST_MODEL=claude-haiku-4-5-20251001      # buy & monitoring rationale (cached)
# LLM_REASONING_MODEL=claude-sonnet-4-6         # industry, validation, thesis review

# ── OpenAI GPT (alternative / comparison) ───────────────────────────────────
OPENAI_API_KEY=your_openai_api_key_here
# Optional model overrides (defaults shown):
# OPENAI_FAST_MODEL=gpt-5.4-mini               # buy & monitoring rationale
# OPENAI_REASONING_MODEL=gpt-5.4               # industry, validation, thesis review
```

Both keys can be set simultaneously to enable the **Compare Both** feature in the Buy Flow tab.

4. Start the portal:

```bash
streamlit run streamlit_app.py
```

If the `.env` values are not populated, the platform still runs using deterministic fallbacks for rationale generation.

## Streamlit Cloud

This repo can be deployed on Streamlit Community Cloud with `streamlit_app.py` as the entrypoint.

Use these deployment settings:

- Repository: `sudipta0311/stock`
- Branch: `codex/streamlit-cloud-ready`
- Main file path: `streamlit_app.py`
- Python version: `3.11`

Add your runtime secrets in the app's `Settings -> Secrets` panel using [.streamlit/secrets.toml.example](c:/Project/App/.streamlit/secrets.toml.example) as the template. The app reads both local `.env` files and Streamlit secrets.

### Current deployment reference

- Active deployment branch: `codex/streamlit-cloud-ready`
- Latest Streamlit deployment branch commit: `2d2abc5`
- Latest `main` branch commit at the time of writing: `b6324be`
- App URL: `https://sudipta0311-stock-streamlit-app-codexstreamlit-cloud-rea-ltl7mo.streamlit.app/`

### Redeploy checklist

1. In Streamlit Cloud, confirm the app is pointing to branch `codex/streamlit-cloud-ready`.
2. Use `Reboot app` or `Redeploy latest commit` after pushing updates.
3. Re-check the app secrets after any delete/recreate cycle.
4. If portfolio monitoring looks empty after a PDF upload, re-upload the statement, let auto-ingestion complete, then run monitoring from the `Monitoring` tab.
5. The Monitoring tab now includes a visible LLM selector and shows the LLM used in the latest monitoring results.

## Portal Flow

1. Click `Seed Demo Portfolio` for a ready-made walkthrough.
2. Or go to `Upload Portfolio` and upload/import holdings.
3. For PDFs, enter the password and let auto-ingestion complete.
4. Open `Buy Ideas` to generate recommendations.
5. Open `Monitoring` to review direct holdings, choose the monitoring LLM, and run monitoring.

## Supported Inputs

The ingestion tab supports:

- Manual entry through editable tables
- `.json` upload with keys: `macro_thesis`, `investable_surplus`, `direct_equity_corpus`, `mutual_funds`, `etfs`, `direct_equities`
- `.csv` upload with columns: `holding_type`, `instrument_name`, `symbol`, `market_value`, `quantity`
- `.pdf` upload as a workflow trigger placeholder; the current scaffold still relies on manual table content for parsing

## Notes

- The app is built around a local demo provider to keep it runnable without network access.
- Replacing the demo provider with live data adapters is straightforward because the agent layer already consumes provider methods rather than hardcoding APIs.
- The **Buy Ideas** tab has a **Model Platform selector**: choose Anthropic Claude, OpenAI GPT, or **Compare Both** to run both providers side-by-side and view their recommendations in two columns.
- The **Monitoring** tab has its own LLM selector and the latest monitoring results show which LLM generated the run.
- Both providers follow the same fast/reasoning tier split. Anthropic adds system-prompt caching on high-volume calls.
- All LLM calls fall back gracefully to deterministic text when the relevant API key is absent or a call fails.
- **Monitoring scope** is limited to direct equities from the ingested statement and any manually added watchlist stocks. MF/ETF look-through holdings are excluded from monitoring because they are managed by fund managers.
- The watchlist persists in SQLite (`monitoring_watchlist` table) and survives app restarts.
