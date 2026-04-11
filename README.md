# Stock LangGraph Platform

This project turns the `stock_platform_v5.puml` sequence diagram into a working Python scaffold with:

- `LangGraph` workflows for signal refresh, portfolio ingestion, new-buy recommendations, and monitoring
- `SQLite` persistence for signals, look-through exposure, overlap, gaps, recommendations, and monitoring actions
- `Streamlit` portal with a mobile-friendly layout
- Live market data via NSE constituent files, `yfinance`, and official/fallback mutual-fund look-through adapters

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
- `src/stock_platform/providers/live.py`: live NSE and `yfinance` data source used by the runtime
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

If the `.env` values are not populated, the platform still runs using deterministic text fallbacks for rationale generation, but the market-data pipeline remains live.

## Streamlit Cloud

This repo can be deployed on Streamlit Community Cloud with `streamlit_app.py` as the entrypoint.

Use these deployment settings:

- Repository: `sudipta0311/stock`
- Branch: `codex/streamlit-cloud-ready`
- Main file path: `streamlit_app.py`
- Python version: `3.11`

Add your runtime secrets in the app's `Settings -> Secrets` panel using [.streamlit/secrets.toml.example](c:/Project/App/.streamlit/secrets.toml.example) as the template. The app reads both local `.env` files and Streamlit secrets.

For Google sign-in, the hosted secrets must include the OpenID Connect metadata URL:

```toml
[auth]
redirect_uri = "https://<your-app-subdomain>.streamlit.app/oauth2callback"
cookie_secret = "replace-with-a-long-random-string"

[auth.google]
client_id = "xxxxxxxxxxxx.apps.googleusercontent.com"
client_secret = "GOCSPX-xxxxxxxxxxxxxxxxxxxxxxxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

In Google Cloud Console, add these exact redirect URIs to the OAuth client that backs the app:

- `https://<your-app-subdomain>.streamlit.app/oauth2callback`
- `http://localhost:8501/oauth2callback` for local testing

If you update secrets or OAuth settings, reboot the Streamlit app before re-testing login.

### Current deployment reference

- Active deployment branch: `codex/streamlit-cloud-ready`
- Latest deployment branch commit verified in this workspace: `166ffae`
- App URL: `https://sudipta0311-stock-streamlit-app-codexstreamlit-cloud-rea-ynukc3.streamlit.app/`

### Redeploy checklist

1. In Streamlit Cloud, confirm the app is pointing to branch `codex/streamlit-cloud-ready`.
2. Use `Reboot app` or `Redeploy latest commit` after pushing updates.
3. Re-check the app secrets after any delete/recreate cycle.
4. If Google login fails immediately inside Streamlit, verify `server_metadata_url` is present in `[auth.google]` and the redirect URI matches Google Cloud exactly.
5. If Google login reaches Google but stops with `redirect_uri_mismatch`, add the exact callback URI shown in the browser to the Google OAuth client.
6. If portfolio monitoring looks empty after a PDF upload, re-upload the statement, let auto-ingestion complete, then run monitoring from the `Monitoring` tab.
7. The Monitoring tab now includes a visible LLM selector and shows the LLM used in the latest monitoring results.
8. Direct equity holdings entered manually are preserved across PDF re-ingestions — a PDF upload will no longer wipe manually saved direct equities.

## Portal Flow

1. Load the sample portfolio for a ready-made walkthrough if you want to test without a statement first.
2. Or go to `Upload Portfolio` and upload/import holdings.
3. For PDFs, enter the password and let auto-ingestion complete.
4. Open `Buy Ideas` to generate recommendations.
5. Open `Monitoring` to review direct holdings, choose the monitoring LLM, and run monitoring.

## Supported Inputs

The ingestion tab supports:

- Manual entry through editable tables
- `.json` upload with keys: `macro_thesis`, `investable_surplus`, `direct_equity_corpus`, `mutual_funds`, `etfs`, `direct_equities`
- `.csv` upload with columns: `holding_type`, `instrument_name`, `symbol`, `market_value`, `quantity`
- `.pdf` upload for encrypted NSDL CAS statements; the parser extracts holdings directly into the ingestion workflow

## Notes

- The runtime provider is now live: index membership comes from NSE archives, stock-level market/financial context comes from `yfinance`, and MF/ETF look-through continues to use official AMC pages plus the existing fallback resolver.
- `seed_demo_data()` remains available only as a sample portfolio helper for testing the UI quickly. It no longer powers runtime recommendations or monitoring.
- The **Buy Ideas** tab has a **Model Platform selector**: choose Anthropic Claude, OpenAI GPT, or **Compare Both** to run both providers side-by-side and view their recommendations in two columns.
- The **Monitoring** tab has its own LLM selector and the latest monitoring results show which LLM generated the run.
- Both providers follow the same fast/reasoning tier split. Anthropic adds system-prompt caching on high-volume calls.
- All LLM calls fall back gracefully to deterministic text when the relevant API key is absent or a call fails.
- **Monitoring scope** is limited to direct equities from the ingested statement and any manually added watchlist stocks. MF/ETF look-through holdings are excluded from monitoring because they are managed by fund managers.
- The watchlist persists in SQLite (`monitoring_watchlist` table) and survives app restarts.

## Recent fixes

| Area | What changed |
|---|---|
| **Stock validation safety gate** | Every candidate stock now passes a 4-gate `validate_stock()` check before scoring or any LLM call. Gates: (1) no Screener.in data → `NOT_FOUND`; (2) all key fields None → `NO_DATA`; (3) price unavailable → `PRICE_MISSING`; (4) fewer than 2 key fields → `NO_DATA`. Blocked stocks are logged to a `skipped_stocks` SQLite table and shown to the user in a **⚠️ Skipped stocks** expander below recommendation cards. |
| **Demerged / delisted symbol resolution** | `utils/symbol_resolver.py` maps old symbols to their current NSE tickers (e.g. `TATAMOTORS → TMCV`). Unresolvable symbols are blocked at the gate and never reach qualitative validation or the LLM rationale step. |
| **Screener.in data source** | Fundamentals (ROCE, D/E, EPS, promoter holding, revenue growth) are now fetched from Screener.in via `utils/screener_fetcher.py`, with consolidated/standalone fallback. `yfinance` is retained for current price and analyst target only. This fixes stale/missing data for Indian stocks that `yfinance` returns incorrectly. |
| **Quality score accuracy** | `agents/quant_model.py` scores on 5 rules using Screener field names (`roce_pct`, `eps`, `revenue_growth_pct`, `promoter_holding`, `debt_to_equity`). No-data returns `0.5` (never `1.0`). Negative EPS hard-caps quality at `0.35`. |
| **Compare Both — analyst differentiation** | Anthropic Claude uses a **contrarian risk analyst** prompt (4 bullets: RISK / WE ARE WRONG IF / VERDICT / SUPPORTING METRIC). OpenAI GPT uses a **momentum catalyst analyst** prompt (4 bullets: CATALYST / MARKET MISREAD / EXIT TRIGGER / VERDICT). The two rationales are structurally different and surface distinct information. |
| **Compare Both — synthesis** | When both providers return recommendations, a third **Analyst Synthesis** section appears below the two columns. Claude Sonnet synthesises the contrarian and catalyst views into 3 bullets: WHERE THEY AGREE / WHERE THEY DISAGREE / COMBINED VERDICT (ENTER NOW / ACCUMULATE GRADUALLY / WAIT FOR BETTER ENTRY). |
| **Live market-data runtime** | `PlatformEngine` now uses `LiveMarketDataProvider` instead of `DemoDataProvider`. Candidate discovery uses official NSE constituent files, stock snapshots/price context/financials come from live sources, and unknown live fields are left unknown rather than replaced with optimistic demo defaults. |
| **Direct holdings — Monitoring Desk** | `capture_user_portfolio` now skips blank data-editor rows (`float("")` no longer crashes ingestion). Direct equity rows are preserved on PDF re-ingestion — only replaced when the new payload explicitly provides at least one valid row. `normalize_exposure` no longer crashes on stocks with no symbol. |
| **Position sizing** | Buy recommendations show **"Deploy now: X% \| Target: Y% over 3 months"** using `compute_position_size(entry_signal, quality_score, corpus)`. Each stock gets a different initial tranche. Hard cap remains 30%. |
| **Sector gap conviction** | `identify_gaps` computes true sector exposure across all instruments (MF + ETF + direct equity). Sectors ≥ 6% total exposure → `AVOID`. Sectors 4–6% → `NEUTRAL` max. |
