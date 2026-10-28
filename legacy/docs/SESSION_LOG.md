# Session Log — Four-Task Implementation

**Repo:** `c:\Project\Stock` (branch: `streamlit-multi-agent`)
**Started:** 2026-05-10
**Model:** claude-sonnet-4-6

---

## STATUS SNAPSHOT (read this first on resume)

| Task | Status | Notes |
|------|--------|-------|
| Task 1 — Source Health Gate | ✅ COMPLETE | All files written, wired, tested |
| Task 2 — Validation-Before-Scoring | ✅ COMPLETE | tuple return, provenance dict, DEGRADED gate |
| Task 3 — Backtest Harness | ✅ COMPLETE | snapshot, replay, scorer, run_backtest, GH Actions |
| Task 4 — Weight Calibration | ✅ COMPLETE | calibrate.py, quality_weights.yaml, YAML loader |
| Unit tests | ✅ COMPLETE | test_source_health.py, test_backtest_scorer.py |
| README + SUMMARY.md | ✅ COMPLETE | 3 new sections in README; SUMMARY.md created |
| **Final test run** | ✅ COMPLETE | 148 passed, 2 skipped (MF API down), 0 failed |
| **GitHub Actions backtest** | ✅ WORKING | Neon connected, 5 recs/week verified locally |
| **Session 5 — Fundamentals quality** | ✅ COMPLETE | ROCE/D/E/RevGrowth fixed in snapshot.py; Screener enrichment; coverage report |

**Session 5 complete. Neon fundamentals quality vastly improved. See SESSION 5 section below.**

To re-run backtest locally (one week, diagnostic):
```
$env:NEON_DATABASE_URL="<your-neon-url>"
.venv\Scripts\python.exe -m backtest.run_backtest --mode replay --start-date 2024-10-07 --end-date 2024-10-14
```

---

---

## SESSION 5 — 2026-05-14: Fundamentals Data Quality Improvement

### Goal
Backtest harness was producing degenerate results because `snapshot.py` left ROCE, promoter_holding always NULL, used a single scalar for revenue_growth, and had a balance-sheet column alignment bug in D/E.

### Files created / modified

| File | Change |
|------|--------|
| `src/stock_platform/data/schema.py` | Added `fetched_source TEXT` column to `historical_fundamentals` DDL (both dialects). Added `_ensure_column` call in `initialize_schema()` for Neon migration. |
| `backtest/snapshot.py` | Full rewrite of `snapshot_fundamentals()`: ROCE from income+balance sheet, per-quarter YoY revenue growth, D/E balance-sheet column alignment, `fetched_source="yfinance"`. Added `get_coverage_counts()` and `report_coverage()`. |
| `backtest/screener_history_fetcher.py` | **NEW** — `fetch_screener_history()`: parses #quarterly-shp for promoter holding, #quarters P&L for ROCE estimate (OP×4/TotalAssets×100, tagged `screener_computed`), #top-ratios for current ROCE. `enrich_from_screener()`: first-non-null-wins UPDATE, 5 s/symbol rate limit. |
| `backtest/run_backtest.py` | Added `--enrich-screener` flag (fills NULLs after snapshot), `--mode coverage` (print coverage table only). Before/after coverage captured whenever snapshot runs. |

### Coverage results (Neon DB, 2026-05-14)

Three-pass summary — starting state was all five fields 100% NULL:

| Field | After yfinance snapshot | After Screener enrich | Final NULL% |
|---|---|---|---|
| ROCE | 24.8% | 14.4% | **14.4%** |
| EPS | 14.8% | 14.8% | **14.8%** |
| D/E | 5.6% | 5.6% | **5.6%** |
| RevGrowth | 81.1% | 81.1% | **81.1%** (window limit†) |
| Promoter | 100.0% | 6.7% | **6.7%** |

†RevGrowth requires same-quarter prior year (index i+4). With avg 5.4 quarters stored, most symbols only get 1 YoY pair. Fix: use `period="5y"` in snapshot to store 20 quarters.

**Remaining ROCE nulls (14.4%):** Banks and insurance companies (HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK, INDUSINDBK, BAJFINANCE, BAJAJFINSV, HDFCLIFE, SBILIFE) — yfinance has no "Operating Income" for financials; Screener P&L similarly has no traditional operating profit row. Known limitation.

**fetched_source values in DB:**
- `"yfinance"` — populated by snapshot.py
- `"screener"` — directly reported (promoter from quarterly-shp, current ROCE from top-ratios)
- `"screener_computed"` — ROCE estimated as (OP × 4) / Total_Assets × 100 from quarterly P&L

### Key design decisions enforced (amendments)

1. Targets Neon Postgres only — NeonWrapper used throughout
2. Screener ROCE for older quarters: computed from #quarters P&L table (operating_margin × asset_turnover approximation), tagged `screener_computed`
3. First-non-null-wins: Screener enrichment uses `WHERE field IS NULL` in every UPDATE
4. Rate limit: 5.0 s per symbol (hardcoded, non-negotiable)
5. Before/after coverage captured in every snapshot run for audit trail

### To run enrichment again (after fresh snapshot)
```
$env:NEON_DATABASE_URL="<url>"
.venv\Scripts\python.exe -m backtest.run_backtest --mode snapshot --enrich-screener
```

### To check current coverage without re-snapshotting
```
$env:NEON_DATABASE_URL="<url>"
.venv\Scripts\python.exe -m backtest.run_backtest --mode coverage
```

---

## SESSION 4 — 2026-05-12: GitHub Actions Backtest Fixes

### Problems solved this session

| Problem | Root cause | Fix |
|---------|-----------|-----|
| Workflow not visible in Actions UI | `backtest.yml` only on `streamlit-multi-agent`; GH requires workflow on default branch | Copied `backtest.yml` to `main` branch (only that file) |
| `403 Permission denied` on summary commit | GH Actions token is read-only by default | Added `permissions: contents: write` to job |
| All runs writing to local SQLite, not Neon | `psycopg2` not installed in runner; fell back to ephemeral SQLite | Added `psycopg2-binary` to `pyproject.toml` `[neon]` extra; workflow now runs `pip install -e ".[neon]"` |
| 0 candidates every replay week | `generate_candidates` filters by sector; `HistoricalDataProvider.get_index_members()` returns `sector="Unknown"` for all stocks | Added `"Unknown"` sector to `_SEED_PORTFOLIO_CONTEXT.identified_gaps` in `replay.py` |
| `ValueError: Portfolio data is missing` in replay | `load_portfolio_gate` always re-reads from DB (empty in CI), ignoring injected `portfolio_context` | Added early-return in `load_portfolio_gate`: if `portfolio_context` already in state with `normalized_exposure`, skip DB load |
| All stocks skipped (financials all None) | Default start date `24 months ago` = May 2024, but earliest fundamental snapshot is Sep 2024; `_get_fundamentals()` returns nothing for dates before first snapshot | Changed workflow default from `24 months ago` to `15 months ago` (always within yfinance 8-quarter window) |

### Verified working (local run, 2026-05-12)

```
Neon fundamentals range: 2024-09-30 → 2026-03-31  (270 rows)
Neon prices range:       2024-05-06 → 2026-05-04   (5355 rows)

Replay 2024-10-07 → 2024-10-14:
  universe=50  scored=22  degraded_watchlist=22
  replay bt-f4faa8d555: 2024-10-07 → 5 recs (confidence=GREEN)
```

### Files changed this session

| File | Change |
|------|--------|
| `.github/workflows/backtest.yml` | `permissions: contents: write`; `pip install -e ".[neon]"`; start date `24m→15m ago`; `ref: streamlit-multi-agent` checkout |
| `pyproject.toml` | Added `neon = ["psycopg2-binary>=2.9.0"]` optional dependency |
| `backtest/replay.py` | Added `"Unknown"` sector to `_SEED_PORTFOLIO_CONTEXT.identified_gaps` |
| `src/stock_platform/agents/buy_agents.py` | `load_portfolio_gate`: skip DB load when `portfolio_context` already injected |

### Important Neon DB facts (stock-agent project, production branch)

- **Fundamentals**: `2024-09-30` → `2026-03-31` (270 rows, 5 fields per quarter per stock)
- **Prices**: `2024-05-06` → `2026-05-04` (5355 rows, weekly)
- **Safe replay window**: `2024-10-01` → today (start before Sep 2024 = no fundamentals = all stocks skipped)
- **Safe score window**: replay ending >6m before today has complete 6m forward returns; >12m has complete 12m forward returns

---

## WHAT WAS DONE IN SESSION 3 (2026-05-10)

All four tasks are now fully implemented. Session 1 had Tasks 1–4 coded; this session
fixed residual bugs and produced documentation.

### Bugs fixed this session

| Bug | Root cause | Fix |
|-----|-----------|-----|
| `test_count_reasonable` assertion wrong | C(24,4)=10,626 not 1,820 — stars-and-bars formula | Changed upper bound to 15,000 |
| `UnboundLocalError: RecommendationRecord` at line 1593 of buy_agents.py | Duplicate `from stock_platform.models import RecommendationRecord` inside `if skip_llm_nodes` block shadowed module-level import; Python 3.14 raises UnboundLocalError when non-skipped path runs | Removed the redundant inner import (module-level import at line 134 already present) |
| 5 `test_monitoring_tax_logic.py` failures: `no such table: overlap_scores` | `data/platform.db` was not initialized — only had `fii_dii_cache` table | Called `repo.initialize()` once to create all tables in `data/platform.db` |

### Documentation added this session

- **`README.md`** — Added sections 14 (Source Health Monitoring), 15 (Backtesting),
  16 (Walk-Forward Weight Calibration). Updated QuantModel API docs (tuple return).
  Updated file tree with all new files. Renumbered §15→18, §16→19, §17→20.
- **`SUMMARY.md`** — New file at repo root: all added/modified files, new env vars,
  all 4 new DB tables with column definitions, migration steps for existing Neon DBs.

---

## FINAL TEST RESULTS (session 3, 2026-05-10)

**148 passed, 2 skipped, 0 failed — full suite clean.**

```
tests/test_backtest_scorer.py          8 pass
tests/test_buy_quality_and_prompts.py  55 pass (49 + 3 subtests)
tests/test_direct_equity_merge.py      3 pass
tests/test_engine_monitoring.py        4 pass, 2 skip (MF API unreachable)
tests/test_monitoring_tax_logic.py     19 pass
tests/test_source_health.py            15 pass
tests/test_ingest_pipeline.py          44 pass
```

### Fixes applied in session 3
| Fix | File | Root cause |
|-----|------|-----------|
| Patched `time.sleep` in synthesis test | `test_engine_monitoring.py` | 20s inter-call delay fired in test |
| Patched `fetch_critical_news` in synthesis test | `test_engine_monitoring.py` | Lazy Anthropic module import on first HTTP call |
| Added `_mf_api_reachable()` skip guard | `test_engine_monitoring.py` | mfdata.in unresponsive (TCP connects, no HTTP) |
| `pytest-timeout` installed | `pyproject.toml` / venv | No timeout = hang forever on network tests |

---

## TASK 1 — Source Health Gate ✅

### Files created/modified
| File | Change |
|------|--------|
| `src/stock_platform/utils/source_health.py` | **NEW** — `SourceHealthChecker`, `get_source_health`, `assert_source_health` |
| `src/stock_platform/agents/buy_agents.py` | Import + wire into `load_portfolio_gate` (gated on `neon_enabled`) |
| `src/stock_platform/agents/monitor_agents.py` | Import + wire into `load_context` (gated on `neon_enabled`) |
| `tests/test_source_health.py` | **NEW** — 15 unit tests, all passing |

### Key design decisions
- Cache key: `"source_health_v1"`, TTL 1800 s via `repo.set_cache` / `repo.get_cache`
- FAILED → `assert_source_health` raises `ValueError` (aborts LangGraph node = aborts run)
- DEGRADED → `context["source_health_warning"] = "DEGRADED"` stored on portfolio_context
- **Health check is gated on `config.neon_enabled`** — skipped in local SQLite mode (tests, local dev). Only active in production Neon deployments.
- Probe uses `ThreadPoolExecutor(max_workers=5)` for parallel symbol checks
- Thresholds: HEALTHY ≥ 80%, DEGRADED ≥ 50%, FAILED < 50%

---

## TASK 2 — Validation-Before-Scoring Guard ✅

### Files modified
| File | Change |
|------|--------|
| `src/stock_platform/agents/quant_model.py` | **REWRITTEN** — `compute_quality_score` now returns `tuple[float, str]`. Added `_data_quality_label()`. Added YAML weight loader `_load_weights()`. |
| `src/stock_platform/utils/screener_fetcher.py` | Added `_provenance` dict; included as `_data_provenance` in returned dict |
| `src/stock_platform/agents/buy_agents.py` | `score_quality`: unpack tuple; `filter_risk`: DEGRADED gate; `finalize_recommendation`: `data_quality` + `data_quality_warning` in payload; removed redundant import of `RecommendationRecord` inside `if skip_llm_nodes` |
| `src/stock_platform/state.py` | Added `degraded_watchlist: list[dict[str, Any]]` to `BuyState` |
| `tests/test_buy_quality_and_prompts.py` | Updated 6 assertions for tuple; added `test_data_quality_label_from_provenance` |

---

## TASK 3 — Backtest Harness ✅

### Files created
| File | Purpose |
|------|---------|
| `src/stock_platform/data/schema.py` | Added 4 tables + 3 indexes to both SQLite and PG DDL |
| `backtest/__init__.py` | Package marker |
| `backtest/snapshot.py` | 2y weekly price + quarterly fundamental snapshots |
| `backtest/replay.py` | `HistoricalDataProvider` + per-Monday replay loop; `_NoOpLLM` stub |
| `backtest/scorer.py` | Forward-return computation, 3m/6m/12m hit rates, alpha vs NIFTY |
| `backtest/run_backtest.py` | CLI `--mode snapshot|replay|score|full`; exit codes 0/1/2 |
| `backtest/README.md` | Full backtest documentation |
| `.github/workflows/backtest.yml` | Weekly cron (Sunday 02:00 UTC) + workflow_dispatch |
| `tests/test_backtest_scorer.py` | 8 unit tests (forward return, hit rate, confidence bands, weight combos) |

### `skip_llm_nodes` short-circuit
- `validate_qualitative`: if flag set, approve all candidates, skip LLM call
- `finalize_recommendation`: if flag set, return lightweight records immediately (no LLM rationale)

---

## TASK 4 — Weight Calibration ✅

### Files created/modified
| File | Change |
|------|--------|
| `rules/quality_weights.yaml` | **NEW** — v1 defaults (roce:0.25, eps:0.25, rev:0.20, promoter:0.15, de:0.15) |
| `backtest/calibrate.py` | **NEW** — `_weight_combinations()` (itertools, ~10,626 combos), `calibrate()`, `main()` |
| `src/stock_platform/agents/quant_model.py` | `_load_weights()` reads YAML at import time; falls back to defaults |

### Calibration details
- 18m train / 6m validate split
- Overfitting flag: val hit rate < train hit rate − 5pp
- Grid: C(24,4) = 10,626 combinations (5 weights, 0.05 step, sum = 1.0)
- Output: `rules/quality_weights.yaml` with version, hash, periods, hit rates

---

## RESUME INSTRUCTIONS

```
cd c:\Project\Stock

# 1. Quick sanity — everything except known hangers
.venv\Scripts\python.exe -m pytest tests/ --ignore=tests/test_engine_monitoring.py --ignore=tests/test_ingest_pipeline.py -q

# 2. Local backtest sanity (needs Neon URL, use start date >= 2024-10-01)
$env:NEON_DATABASE_URL="<neon-url>"
.venv\Scripts\python.exe -m backtest.run_backtest --mode replay --start-date 2024-10-07 --end-date 2024-10-14

# 3. Import sanity
.venv\Scripts\python.exe -c "from stock_platform.services.engine import PlatformEngine; print('OK')"
```

The session log is at `c:\Project\Stock\SESSION_LOG.md`.
The progress summary is at `c:\Project\Stock\SUMMARY.md`.
