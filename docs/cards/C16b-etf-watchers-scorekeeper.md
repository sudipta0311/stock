# C16b: ETF & index-fund selector, Rates/Rulebook watchers with a human approval queue, Scorekeeper (paper portfolio + track record), CIO weekly agenda

**Plan:** `docs/UPGRADE_PLAN.md` §4.7 (ETF/fund selection is deterministic; the agent only explains and flags), §5.2 agents #8 (ETF selector), #9 (Rates watcher), #10 (Rulebook watcher), #14 (Scorekeeper), #1 (CIO agenda), §5.8 (bounded self-improvement), §9.2 section 7 ("How our calls are doing", honest, misses included), §3.5/C03 (scheme rates need human approval). Read `AGENTS.md` first. Builds on C10/C15a/C16a data, C14 router/tools, C15c pack.

## Scope

### 1. ETF & index-fund selector (`wealth/research/etf_selector.py`, code; `nano` explanation only)
- **Categories** (`rules/etf_categories.yaml`): `in_nifty50`, `in_nifty_next50`, `in_midcap150`, `in_gold`, `in_liquid`, `global_sp500`, `global_nasdaq100`, `global_all_world`. Each has a match rule over the securities master (NSE ETF list + AMFI index funds/FoFs) by name/category.
- **Metrics:**
  - liquidity (median 3-month traded value from `prices_eod` volume; ETFs only)
  - premium/discount to NAV = (close − NAV)/NAV, using AMFI NAV for the same ISIN where both exist
  - tracking difference: the 1-year return gap vs the index level when NSE index closes are available (C16a stores index levels; add them if missing). Else `null`.
  - expense ratio: from AMFI's published TER data if a stable free official file exists (investigate; otherwise `null` with the note `ter_unavailable`)
- **Scoring:** `rules/etf_selector.yaml` weights. Disqualify when liquidity is under the floor or the premium is > 1%. Missing metrics never boost a score.
- **Output:** the top 2 per category with metrics and `reason_codes`, stored in the Research Pack `core_picks`. The `nano` explanation (≤ 2 sentences) is grounded on the metrics. Also expose per-category `inav_premium_pct` for C12's route selector, and an `amc_international_suspended` flag when an AMC notice says subscriptions are paused (from the C16a PIB/AMC feed if available; else `null`).

### 2. Watchers + human approval queue
- **Rates watcher** (`nano` for extraction, code for the diff): fetch the official India Post small-savings page, the latest DEA notification, and the EPFO interest-declaration page. Extract `{scheme, rate_pct, from, to}`. Diff against `rules/schemes/small_savings.yaml`; differences become rows in `rule_change_proposals(kind, path, current_value, proposed_value, evidence JSON, status: pending|approved|rejected, created_at, decided_at, decided_by)`.
- **Rulebook watcher:** the same pattern for CBDT/RBI/SEBI press-release feeds (from C16a `feed_items`). The LLM only classifies relevance (tax rates, LRS/TCS, small savings, EPF) and quotes the exact sentence (substring-verified). It proposes **no** values unless the quote contains them.
- **Admin CLI** `python -m wealth.research.rules_admin list|show <id>|approve <id>|reject <id>`:
  - `approve` edits the YAML file in the working tree (keeps comments where feasible; at minimum, a valid YAML diff) and marks the proposal approved with `verified: true` for that period and `decided_by` from the env `ADMIN_NAME`.
  - **It never commits.** A human reviews the git diff.

### 3. Scorekeeper (`wealth/research/scorekeeper.py`, code)
- **Paper portfolio:**
  - Each APPROVED idea in a weekly pack is "bought" at the next trading day's close: a ₹1,00,000-equivalent notional, or $1,000 for US ideas.
  - It's held until the idea is VETOED/UNRESOLVED in a later pack or for 52 weeks.
  - Benchmarks: NIFTYBEES (India) and, for US, an S&P 500 proxy if prices exist; else marked `benchmark_unavailable`.
- Tables `paper_positions` and `track_record(week, market, ideas_count, ahead_count, avg_excess_return_pct, details JSON)`.
- **Track record summary** for emails/PWA: "Satellite ideas since launch: +x.x% vs NIFTY 50 (n ideas, m ahead)". It includes misses and states the start date.
- **Calibration:** hit rate by confidence band (HIGH/MEDIUM/LOW) once n ≥ 10 per band; before that `insufficient_data`.
- **Error analysis** for the CIO: list the 5 worst ideas with their dossier ids and red-team falsifiers (code). A `mini` call summarises recurring patterns into ≤ 3 checklist items, stored for next week's agenda. These are text only; **prompts and models are never auto-edited**.

### 4. CIO weekly agenda (graph `plan` node)
Reads the scorekeeper checklist + the C16a regime notes. Sets per-market dossier budgets and the max ideas (bounded by `rules/research.yaml`), records the agenda in the run. No status decisions.

### 5. Integration
- C17a reads `core_picks`, `inav_premium_pct`, tilts and track record from the pack.
- C17b's brief section 7 uses the track record (omitted when insufficient).
- The PWA Ideas tab shows `core_picks` (C22 feature-detects).

### 6. Tests (no network; FakeRouter)
- Selector: category matching, the premium disqualification, missing TER never boosts, a deterministic top-2.
- Watchers: changed rate → proposal; unchanged → none; an LLM quote not a substring → rejected; `approve` edits YAML and flips `verified` (run it on a temp copy of the rules dir).
- Scorekeeper: worked example (3 ideas, synthetic prices, hand-computed excess returns and ahead count in the docstring); veto closes a position; calibration `insufficient_data` below n=10.
- The CIO agenda is bounded.
- Track record text formatting.

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass, no network
2. `ruff` clean; migration(s) up/down
3. Live smoke (no LLM): the selector over the dev DB's real ETF list (report the top picks per category and any missing metrics)
