# Backtest Harness

Replays FLOW 2 (buy recommendations) against historical fundamental and price
snapshots to measure how well the quant pipeline would have performed.

---

## Architecture

```
backtest/
├── __init__.py
├── snapshot.py      # Fetch & store 2y weekly prices + quarterly fundamentals
├── replay.py        # HistoricalDataProvider + per-Monday replay loop
├── scorer.py        # Forward-return computation, hit rate, alpha vs NIFTY
├── run_backtest.py  # CLI entry point (argparse, JSON logging)
├── calibrate.py     # Walk-forward weight calibration (grid search)
└── README.md        # This file
```

---

## Database tables (added to schema.py, both SQLite + Postgres)

| Table | Purpose |
|-------|---------|
| `historical_fundamentals` | Quarterly ROCE, EPS, D/E, revenue growth snapshots |
| `historical_prices` | Daily/weekly close prices (includes `NIFTY` benchmark) |
| `backtest_runs` | Summary per replay run (hit rates, alpha, weights hash) |
| `backtest_recommendations` | Per-symbol per-date recommendations with forward returns |

Migration: `initialize_schema()` uses `CREATE TABLE IF NOT EXISTS` — just restart
the engine and the tables appear automatically. No destructive migration needed.

---

## Running locally

```bash
# 1. Activate your virtualenv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Full pipeline (snapshot → replay → score)
python -m backtest.run_backtest \
  --mode full \
  --start-date 2024-05-09 \
  --end-date   2026-05-09

# 3. Individual phases
python -m backtest.run_backtest --mode snapshot            # fetch prices + fundamentals
python -m backtest.run_backtest --mode replay  --start-date 2024-05-09 --end-date 2026-05-09
python -m backtest.run_backtest --mode score   --run-id <run_id from replay output>

# 4. Walk-forward weight calibration (needs a completed full run first)
python -m backtest.calibrate --run-id <run_id>
```

Structured JSON lines are emitted to stdout — pipe to `jq` for readability:
```bash
python -m backtest.run_backtest --mode full 2>&1 | jq .
```

---

## GitHub Actions (`.github/workflows/backtest.yml`)

- **Trigger:** weekly cron Sunday 02:00 UTC, or `workflow_dispatch`
- **Secret required:** `NEON_DATABASE_URL` (production Neon DB)
- **Exit code 1** if 6-month hit rate < `--hit-rate-floor` (default 0.45)
- Summary markdown committed to `backtest/results/` on every run

---

## HistoricalDataProvider

`replay.py` defines `HistoricalDataProvider`, a drop-in replacement for
`LiveMarketDataProvider` that reads from the snapshot tables.

**Critical contract:**
```python
provider = HistoricalDataProvider(repo, replay_date=start_date)
for monday in mondays:
    provider.today = monday      # ← MUST update before each iteration
    provider._price_cache.clear()
    # ... invoke graph ...
```

Forgetting to update `today` per iteration means every replay step runs with
the same date — results look plausible but are garbage.

---

## LLM nodes are skipped in backtest mode

`request["skip_llm_nodes"] = True` short-circuits:
- `validate_qualitative` — all candidates approved with quality_score confidence
- `finalize_recommendation` — LLM rationale phase skipped; lightweight records written

This keeps token costs at zero and makes replay deterministic.
The quant pipeline (`score_quality`, `filter_risk`, `assess_timing`, `size_positions`) runs in full.

---

## Known wall-clock leaks

These functions in the buy path use `date.today()` instead of `provider.today`:

| Call site | Impact |
|-----------|--------|
| `stock_validator.check_recently_listed` | Listing-age check uses wall-clock; may over/under-cap IPO stocks |
| `result_date_fetcher` | Staleness calculation uses wall-clock; may affect freshness caps |

Both affect `apply_freshness_cap` in `assess_timing`. Since LLM nodes are skipped
and entry signals are not the primary output of the backtest (quality scores are),
the impact is limited and acceptable for v1. Fix in v2 by routing date through provider.

---

## Point-in-time (PIT) fundamentals

### The look-ahead leak (fixed in this branch)

Before this fix, `_get_fundamentals()` in `replay.py` queried:
```sql
WHERE symbol = ? AND snapshot_date <= ?
ORDER BY snapshot_date DESC LIMIT 1
```

`snapshot_date` is the **quarter-end date** (e.g. 2023-12-31).  
Indian companies publish quarterly results **30–60 days after quarter-end**.  
So using `snapshot_date <= replay_date` allowed the replay to use Q4 2023 results
while replaying a date of 2024-01-05 — data that was not publicly available then.

### The fix

Each row in `historical_fundamentals` now carries an `available_date` column:
```
available_date = snapshot_date + 60 days  (conservative publication lag)
```

`_get_fundamentals()` now queries:
```sql
WHERE symbol = ? AND available_date <= ?
ORDER BY available_date DESC LIMIT 1
```

This prevents any fundamental row from appearing in replay before it would
realistically have been available to investors.

### Migration for existing databases

If you have a `historical_fundamentals` table populated before this fix, run:
```bash
python -m backtest.run_backtest --mode migrate-pit
```

This backfills `available_date = snapshot_date + 60d` for all rows where
`available_date IS NULL`.

### Backward-incompatibility warning

**Backtest results produced before this fix are NOT comparable to results produced
after it.** The look-ahead leak inflated hit rates and alpha — particularly for the
most recent quarters in the snapshot window. Always re-run the full pipeline after
migrating.

---

## Hit rate definition

A recommendation is a **hit** if the stock beat NIFTY by more than **2 percentage points**
in the measurement window (3m / 6m / 12m).

```
hit = (stock_forward_return - nifty_forward_return) > 2.0%
```

Forward prices are fetched from `historical_prices` — the first available date
at or after `recommendation_date + window_weeks`.
