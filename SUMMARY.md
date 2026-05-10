# Implementation Summary — Four Production Improvements

Generated: 2026-05-10

This document lists every file added or modified, every new environment variable,
every new database table, and the migration steps required for existing Neon deployments.

---

## Files Added (NEW)

| File | Purpose |
|---|---|
| `src/stock_platform/utils/source_health.py` | Pre-flight source health gate — probes Screener.in, yfinance, NSE CSV before FLOW 2/3 |
| `backtest/__init__.py` | Package marker |
| `backtest/snapshot.py` | Fetches and stores 2y weekly prices + quarterly fundamentals into new DB tables |
| `backtest/replay.py` | `HistoricalDataProvider` + per-Monday replay loop; `_NoOpLLM` stub |
| `backtest/scorer.py` | Forward-return computation, 3m/6m/12m hit rates, alpha vs NIFTY benchmark |
| `backtest/run_backtest.py` | CLI entry point (`--mode snapshot\|replay\|score\|full`); JSON logging; exit codes 0/1/2 |
| `backtest/calibrate.py` | Walk-forward weight calibration via itertools grid search; writes `rules/quality_weights.yaml` |
| `backtest/README.md` | Full backtest documentation |
| `rules/quality_weights.yaml` | Calibrated quality-score weights (YAML); read by `quant_model.py` at import time |
| `.github/workflows/backtest.yml` | GitHub Actions — weekly cron Sunday 02:00 UTC + `workflow_dispatch`; requires `NEON_DATABASE_URL` secret |
| `tests/test_source_health.py` | 15 unit tests for source health gate (in-memory SQLite, no live network calls) |
| `tests/test_backtest_scorer.py` | Tests for forward-return helpers, hit-rate computation, confidence-band breakdown, weight combinations |
| `SESSION_LOG.md` | Persistent session log for VM resume (human-readable progress tracker) |

---

## Files Modified (CHANGED)

| File | What changed |
|---|---|
| `src/stock_platform/agents/quant_model.py` | `compute_quality_score` now returns `tuple[float, str]` (score, data_quality). Added `_data_quality_label()`. Weights loaded from YAML at import (`_load_weights()`). Changed `print` → `logging`. |
| `src/stock_platform/utils/screener_fetcher.py` | Added `_data_provenance` dict (per-field FETCHED/DEFAULT tracking) included as `"_data_provenance"` key in returned financials dict. |
| `src/stock_platform/agents/buy_agents.py` | Added `assert_source_health` call in `load_portfolio_gate` (gated on `neon_enabled`). `score_quality`: unpacks tuple from `compute_quality_score`, stores `data_quality` in candidate. `filter_risk`: DEGRADED stocks routed to `degraded_watchlist`. `validate_qualitative` and `finalize_recommendation`: `skip_llm_nodes` short-circuit. `finalize_recommendation`: adds `data_quality` and `data_quality_warning` to payload. |
| `src/stock_platform/agents/monitor_agents.py` | Added `assert_source_health` call in `load_context` (gated on `neon_enabled`). Attaches DEGRADED warning to portfolio context. |
| `src/stock_platform/state.py` | Added `degraded_watchlist: list[dict[str, Any]]` to `BuyState`. |
| `src/stock_platform/data/schema.py` | Added 4 new tables to both `_DDL_SQLITE` and `_DDL_PG`: `historical_fundamentals`, `historical_prices`, `backtest_runs`, `backtest_recommendations`. Added 3 indexes. |
| `tests/test_buy_quality_and_prompts.py` | Updated 6 assertions to unpack `(score, quality_label)` tuple from `compute_quality_score`. Added provenance test. |
| `README.md` | Added sections 14 (Source Health), 15 (Backtesting), 16 (Weight Calibration). Updated QuantModel API docs. Updated file tree. Renumbered sections 15→18, 16→19, 17→20. |

---

## New Environment Variables

No new environment variables are required. The existing `NEON_DATABASE_URL` is used:
- As the gate for source health checks (`config.neon_enabled` — True when URL is set)
- As the production DB for the GitHub Actions backtest workflow

Optional new GitHub Actions secret (already used by the repo if deploying to Neon):

| Secret | Where | Purpose |
|---|---|---|
| `NEON_DATABASE_URL` | GitHub repo → Settings → Secrets | Required by `.github/workflows/backtest.yml` |

---

## New Database Tables

All four tables are added via `CREATE TABLE IF NOT EXISTS` in `data/schema.py`.
Migration requires only calling `repo.initialize()` — no destructive changes.

### `historical_fundamentals`

Stores quarterly fundamental snapshots used by the backtest harness.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER / BIGSERIAL | PK autoincrement |
| `symbol` | TEXT | NSE symbol |
| `as_of_date` | TEXT | ISO date of snapshot |
| `roce_pct` | REAL | Return on capital employed % |
| `eps` | REAL | Earnings per share |
| `revenue_growth_pct` | REAL | YoY revenue growth % |
| `promoter_holding` | REAL | Promoter holding % |
| `debt_to_equity` | REAL | D/E ratio |
| `created_at` | TEXT | ISO timestamp |

Unique constraint: `(symbol, as_of_date)`.

### `historical_prices`

Stores daily/weekly close prices including the NIFTY benchmark row.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER / BIGSERIAL | PK autoincrement |
| `symbol` | TEXT | NSE symbol or `"NIFTY"` |
| `date` | TEXT | ISO date |
| `close_price` | REAL | Close price |
| `created_at` | TEXT | ISO timestamp |

Unique constraint: `(symbol, date)`.

### `backtest_runs`

One row per replay run.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT | PK (UUID) |
| `start_date` | TEXT | Replay start (ISO) |
| `end_date` | TEXT | Replay end (ISO) |
| `weights_hash` | TEXT | MD5 of weight config |
| `hit_rate_3m` | REAL | 3-month hit rate |
| `hit_rate_6m` | REAL | 6-month hit rate |
| `hit_rate_12m` | REAL | 12-month hit rate |
| `total_recommendations` | INTEGER | Count of replayed recs |
| `created_at` | TEXT | ISO timestamp |

### `backtest_recommendations`

One row per symbol per replay date.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER / BIGSERIAL | PK autoincrement |
| `run_id` | TEXT | FK → `backtest_runs.run_id` |
| `symbol` | TEXT | NSE symbol |
| `recommendation_date` | TEXT | ISO date of replay |
| `action` | TEXT | ACCUMULATE / STRONG ENTER / etc. |
| `confidence_band` | TEXT | GREEN / YELLOW / RED |
| `quality_score` | REAL | Score at replay date |
| `forward_return_3m` | REAL | % return at 13 weeks |
| `forward_return_6m` | REAL | % return at 26 weeks |
| `forward_return_12m` | REAL | % return at 52 weeks |
| `nifty_return_3m` | REAL | NIFTY % return at 13 weeks |
| `nifty_return_6m` | REAL | NIFTY % return at 26 weeks |
| `nifty_return_12m` | REAL | NIFTY % return at 52 weeks |
| `hit` | INTEGER | 1 if beat NIFTY by >2% at 6m, 0 otherwise |
| `created_at` | TEXT | ISO timestamp |

---

## Migration Steps for Existing Neon DB

1. **No destructive changes** — all new tables use `CREATE TABLE IF NOT EXISTS`.

2. **Apply migration** by calling `repo.initialize()`:
   ```bash
   python -c "
   from stock_platform.config import AppConfig, load_app_env
   from stock_platform.data.repository import PlatformRepository
   load_app_env()
   cfg = AppConfig()
   repo = PlatformRepository(db_path=cfg.db_path, neon_database_url=cfg.neon_database_url)
   repo.initialize()
   print('Migration complete')
   "
   ```
   Or simply restart the Streamlit app — `PlatformEngine.__init__` calls `repo.initialize()`.

3. **Existing tables are untouched.** The `recommendations`, `signals`, `monitoring_actions`,
   `cache_entries`, and all other tables remain as-is.

4. **`compute_quality_score` return type changed** from `float` to `tuple[float, str]`.
   All internal call sites in `buy_agents.py` and `monitor_agents.py` have been updated.
   If any external code (outside this repo) calls `compute_quality_score` directly, it must
   be updated to unpack the tuple: `score, quality = compute_quality_score(symbol, fin_data)`.

5. **No new secrets required** for the core application. `NEON_DATABASE_URL` is already
   configured in production. For the GitHub Actions backtest workflow, confirm the secret
   exists in GitHub → Settings → Secrets → Actions.

---

## Test Coverage

```
tests/
├── test_source_health.py         15 tests — SourceHealthChecker, cache, status thresholds
├── test_backtest_scorer.py        8 tests — forward returns, hit rates, weight combinations
└── test_buy_quality_and_prompts.py  (updated) — tuple unpacking + provenance assertions
```

Run the full suite:
```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

All tests use in-memory SQLite; no network calls, no API keys required.
