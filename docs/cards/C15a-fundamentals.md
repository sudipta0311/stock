# C15a: Point-in-time fundamentals: SEC EDGAR (US), NSE XBRL results (India, if a stable free source exists), price-derived factors, universe lists

**Plan:** `docs/UPGRADE_PLAN.md` §6.1–6.3 (point-in-time: store `period_end`, `filed_at`, `known_at`; no look-ahead), §4.7 (stock factors: quality, value, momentum, low volatility, governance), §5.2 agent #3 (the Screener needs no LLM). Read `AGENTS.md` first. Builds on C10 (market-data tables, refresh CLI, provenance conventions). No LLM in this card.

## Goal
Give the factor screener and the analyst agents trustworthy, dated company numbers, each known only from the date it was actually published.

## Scope

### 1. Migration (next free number)
Shared, no RLS, `SELECT` for `wealth_app`:
- `fundamentals_pit(security_id, metric, period_end, period_type ('Q'|'FY'), value NUMERIC, unit, filed_at, known_at, source, source_url, fetched_at)`, PK `(security_id, metric, period_end, period_type, source)`
- `universe_members(universe, security_id, added_on, removed_on, source, source_url)`

### 2. US: SEC EDGAR (`wealth/marketdata/sources/edgar.py`)
- `companyfacts` JSON (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`) with a declared `User-Agent` containing a contact email from env `SEC_USER_AGENT` (required by the SEC; refuse to run without it). Ticker→CIK from the SEC's `company_tickers.json`.
- Respect the SEC's 10 requests/s limit (sleep).
- Metrics, using us-gaap tags with documented fallbacks: `revenue`, `net_income`, `operating_income`, `total_assets`, `total_equity`, `total_debt`, `cash`, `shares_outstanding`, `eps_diluted`, `operating_cash_flow`, `capex`.
- `filed_at` from the `filed` field; `known_at` = `filed_at`. Keep both 10-Q and 10-K facts; dedupe restatements by keeping every filing (point-in-time) but marking `is_latest_for_period`.

### 3. India: investigate first, then implement or report
- Research during this run whether NSE (or BSE) publishes **free, machine-readable** quarterly financial results (e.g. XBRL "Integrated Filing – Financials") with filing dates, reachable without login.
- **If yes:** implement an adapter for the same metric set (`revenue`, `net_income`, `operating_income`, `eps_diluted`, plus `total_assets`/`total_equity`/`total_debt` when the half-year balance sheet is filed), with `filed_at` from the filing timestamp. Document the URLs and the format in the result file.
- **If no stable free source exists:** don't scrape third-party sites (Screener.in, Tickertape, etc.: terms of service). Leave Indian fundamentals unavailable, implement price-derived factors only (section 4), and explain clearly in the result file what a licensed vendor would need to supply.

### 4. Price-derived factors (both markets): `wealth/research/factors.py` (pure)
From `prices_eod`:
- `momentum_12_1` (12-month return excluding the last month)
- `volatility_1y` (annualised stdev of daily log returns)
- `max_drawdown_1y`
- `dist_from_200dma`
Each returns `None` when there's insufficient history (< 200 trading days for 1-year metrics).

### 5. Fundamental factors (where data exists), all **as of a date** using only rows with `known_at ≤ as_of`
- `roce` (operating income / (total assets − current liabilities, or equity + debt if missing))
- `roe`
- `revenue_yoy` for the last 4 quarters
- `earnings_yield` (TTM EPS / price)
- `fcf_yield`
- `debt_to_equity`
- `accruals_ratio`
- `pe_percentile_own_10y` (needs history; `None` until available)

### 6. Universes
- US: S&P 500 from a free, reliable list. Try the SEC/issuer sources first; if none is stable, accept a CSV in `rules/universes/sp500.csv` committed with `source_url` and `as_of`, explicitly labelled **current constituents only (survivorship bias)**.
- India: NIFTY 500 from the NSE archive CSV, with the same survivorship label. Store in `universe_members` (`added_on` = load date, `removed_on` null).
- **The label matters:** the backtest/scorekeeper must know these lists aren't point-in-time.

### 7. CLI
Extend `python -m wealth.marketdata.refresh` with `--only edgar,nse_fundamentals,universes` and a `--symbols` filter. Respect rate limits. Keep going on per-symbol errors.

### 8. Tests (no network; small committed public-data fixtures)
- EDGAR companyfacts excerpt for one large US company:
  - tag mapping and fallbacks
  - `filed_at` → `known_at`
  - a restatement keeps both values and picks the right one as of a date before vs after the restatement filing
- (If implemented) NSE results excerpt.
- Factor maths on synthetic price series:
  - known momentum
  - volatility of a constant series = 0
  - 200-DMA distance
  - `None` on short history
- **Look-ahead guard:** a factor as of 2026-02-01 must not use a filing made on 2026-02-10.
- Universe loaders with the survivorship label.

## Out of scope
Ranking/screening logic and agents (C15b/C15c). Licensed vendors.

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass, no network
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. Migration up/down works
4. The result file states clearly which Indian fundamentals source (if any) was implemented, or why none was. Live EDGAR smoke only if `SEC_USER_AGENT` is already set in your environment. Otherwise skip it and say so: the SEC requires a real contact email, so don't make one up.
