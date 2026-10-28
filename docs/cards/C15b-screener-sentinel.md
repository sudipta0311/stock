# C15b: Factor Screener (no LLM), filings Sentinel (rules first, nano LLM for free text), research tools, sector data

**Plan:** `docs/UPGRADE_PLAN.md` §4.7 (sector-neutral multi-factor composite; agents may veto or downgrade, never originate), §5.2 agents #3 (Screener) and #4 (Sentinel), §5.4 (daily sentinel → urgent path for governance breaks), §4.9 (governance events trigger exits). Read `AGENTS.md` first. Builds on C14 (`ModelRouter`, `ToolRegistry`, Dossier store, run CLI) and C15a (fundamentals_pit, price factors, universes).

## Goal
Each week, rank each universe with plain code and pick a short, explainable shortlist. Every day, catch material filings for held, watched and shortlisted companies, using deterministic rules wherever the filing format allows it.

## Scope

### 1. Sector data (migration: next free number)
- `securities.sector`, `securities.industry` (text). Load them from the NIFTY 500 constituents CSV (it has an industry column) and, for the US, from the universe file if it has sectors.
- Unknown sector → `"Unclassified"`. Each unclassified security forms its own peer group and is flagged.

### 2. Screener: `wealth/research/screener.py` (pure + DB reads; **no LLM**)
- **Factor groups:**
  - quality = mean z of [roce, roe, −accruals_ratio, −debt_to_equity]
  - value = mean z of [earnings_yield, fcf_yield]
  - momentum = z(momentum_12_1)
  - low_vol = z(−volatility_1y)
- z-scores are computed **within sector** (sector-neutral), winsorised at ±3. A group counts if ≥ 1 of its inputs exists.
- **Coverage rule:** composite = mean of available group scores. A security needs **≥ 3 of 4 groups**, or for markets flagged `price_only` (no fundamentals source, see C15a) momentum + low_vol, with `factor_coverage = "price_only"`. Otherwise it's excluded with reason `insufficient_data`.
- Deciles within universe; liquidity floor: median 3-month traded value ≥ ₹5 cr (India) / $20 M (US), computed from `prices_eod` volume. If the C10 price tables lack volume, add the column in this card's migration and backfill on the next refresh.
- **Output:** for each market, the top-decile shortlist capped at `max_per_market` (`rules/screener.yaml`, default 25), sorted by composite. Each item: `security_id, symbol, sector, composite, decile, groups, factor_coverage, reasons`.
- Persist as dossiers at stage `SCREENED`, with the factor table as FACT claims whose `tool_call_id` points to a `screener_snapshot` tool call that stores the exact inputs.
- **Determinism:** the same inputs give the same ranking. Ties are broken by symbol.

### 3. Research tools (register in `ToolRegistry`; all read-only, logged)
- `fundamentals_series(security_id, metrics, quarters)` → from `fundamentals_pit` as of the run date
- `factor_snapshot(security_id)`
- `price_stats(security_id)` (returns, drawdown, 200-DMA)
- `sector_peers(security_id, metric)` (median + percentile)
- `edgar_filings(cik, forms, days)` + `edgar_filing_text(accession, max_chars)` (US; the SEC User-Agent rule from C15a applies)
- `nse_announcements(symbol, days)`: best effort. If NSE needs session cookies or blocks automation, return `{"status": "unavailable"}` (never scrape third-party aggregators) and report it.

Fetched text is stored as evidence and passed to models only inside `<evidence id=...>` blocks (C14 rule).

### 4. Sentinel: `wealth/research/sentinel.py`
- **Inputs:** securities that are held by any user (`holdings_lots`), on any watchlist (future), or shortlisted this week.
- **Rules first (no LLM):**
  - US 8-K item codes → events:
    - 4.01 → `AUDITOR_RESIGNATION` (auditor change)
    - 4.02 → `RESTATEMENT`
    - 5.02 → `KEY_PERSON_EXIT`
    - 1.03 → `BANKRUPTCY`
    - 2.06 → `IMPAIRMENT`
  - Form `NT 10-Q`/`NT 10-K` → `LATE_FILING`.
  - India announcement categories (if available) mapped by a rules table in `rules/sentinel.yaml`.
- **LLM (tier `nano`) only for items the rules can't classify:**
  - schema: `{material: bool, event_code: enum | "NONE", confidence: "HIGH"|"MEDIUM"|"LOW", quote: str ≤ 300 chars}`
  - `quote` must be an exact substring of the evidence text; reject it otherwise.
- **Output:** `SentinelEvent{security_id, event_code, source, evidence_tool_call_id, detected_at, method: "rule"|"llm"}` in a new table `sentinel_events` (shared, owner-written).
  - Events in the governance set (`AUDITOR_RESIGNATION`, `RESTATEMENT`, `REGULATOR_ENFORCEMENT`, `BANKRUPTCY`) are flagged `urgent=true` for the alerts card (C17).
- **CLI:** `python -m wealth.research.sentinel --days 2`. Daily job; idempotent (same filing → same event id).

### 5. Tests (no network; FakeRouter)
- **Screener:** sector neutrality (a sector with uniformly high ROCE doesn't dominate); the coverage rule; the price_only path; the liquidity floor; deterministic ties; decile boundaries on a 20-security synthetic universe with hand-computed composites for 3 securities (put the arithmetic in the docstring).
- **Tools:** each returns a logged tool call with source_url/retrieved_at; `nse_announcements` returns `unavailable` cleanly on 401/403.
- **Sentinel:** 8-K fixtures with items 4.01 / 4.02 / 5.02 → rule events without any LLM call (assert the FakeRouter call count is 0); a free-text announcement → nano call → `quote` validated as a substring; an invented quote → rejected; idempotency.

## Out of scope
Analyst / Valuation / Red Team / Compliance / CIO agents (C15c); eval harness (C15d); alerts delivery (C17).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,importer,research,dev]"`, then `.venv\Scripts\python -m pytest` → all pass, no network
2. `.venv\Scripts\python -m ruff check src tests scripts` and `.venv\Scripts\python -m ruff format --check src tests scripts` → clean
3. Migration up/down works; the API still doesn't import `wealth.research`
