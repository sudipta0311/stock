# C10: Market data: securities master, NSE EOD prices, AMFI NAVs, USD/INR, gold; point-in-time store with provenance; DB-backed PriceProvider

**Plan:** `docs/UPGRADE_PLAN.md` §6 (data layer: official/licensed first, point-in-time, provenance on every number), §6.2 (sources), Appendix D (Neon free storage: keep only what's needed; GitHub Actions does scheduled refreshes). Read `AGENTS.md` first. Builds on C04 (`PriceProvider`), C05 (DB), C07/C08b (net worth, holdings, `securities`).

## Goal
Daily, provenance-tagged prices for what users actually hold, so net worth uses current values, not statement values. Gold and ETF assets stop showing "price needed".

## Decisions (follow them)
- Fetching runs **only** in a CLI job (`python -m wealth.marketdata.refresh`), never inside API requests. The API reads from the DB through `DbPriceProvider`. Jobs write with `system_session` (owner). New market tables are shared, without RLS, and `wealth_app` gets `SELECT` only.
- **Sources** (each behind a small adapter interface, so a licensed vendor can replace it later):
  - **NSE equity EOD:** the official daily bhavcopy CSV from NSE archives. Send a browser-like `User-Agent`, retry with backoff, and skip non-trading days.
  - **NSE securities master:** NSE's published equity list CSV (symbol, name, ISIN, series) + ETF list if available.
  - **Mutual funds:** AMFI's official `NAVAll.txt` (scheme code, ISINs for growth/reinvest, name, NAV, date). It doubles as the MF master. Category comes from the section headers in that file, e.g. "Open Ended Schemes(Debt Scheme - Liquid Fund)" → `mf:debt:liquid`.
  - **USD/INR:** FRED series `DEXINUS` via the public CSV endpoint (no key). Record its publication lag.
  - **Gold (24k ₹/g):** best-effort adapter for the IBJA daily rates page. If parsing fails, fall back to `rules/manual_prices.yaml` (`gold_24k_per_gram` with `as_of` and `source: manual`), which an admin can update. Never invent a price.
  - **US EOD:** out of scope here (C09b).
- **Provenance on every row:** `source`, `source_url`, `fetched_at`, `as_of` (the trading date the value refers to).
- **Storage discipline (Neon free tier):** keep daily prices for 400 days only for securities that any user holds (`holdings_lots`) plus index proxies (`NIFTYBEES`, `JUNIORBEES`, `GOLDBEES`). `prune` deletes older rows. The master tables keep everything (small).

## Scope

### 1. Migration `0006_market_data`
Shared tables (with `SELECT` grants to `wealth_app`, no RLS):
- `prices_eod(security_id, as_of, close, source, source_url, fetched_at)`, PK `(security_id, as_of)`
- `nav_daily(security_id, as_of, nav, source, fetched_at)`
- `fx_rates(pair, as_of, rate, source, fetched_at)`
- `gold_prices(as_of, inr_per_gram_24k, source, source_url, fetched_at)`
- `marketdata_runs(id, started_at, finished_at, status, counts JSONB, errors JSONB)`

Also `securities.amfi_scheme_code`, `securities.nse_symbol`, `securities.series` (nullable).

### 2. Package `wealth/marketdata/`
- `sources/nse.py` (bhavcopy + equity list parsers: pure functions on text, plus a `fetch` wrapper using `httpx` with timeouts)
- `sources/amfi.py`, `sources/fred.py`, `sources/ibja.py`, `sources/manual.py`
- `store.py`: upserts (idempotent), plus `latest_price(security_id, on)` → `(value, as_of, source)`, the latest row ≤ `on`
- `refresh.py` CLI: `--date YYYY-MM-DD` (default today IST), `--only nse,amfi,fx,gold,master`, `--prune`
  - Records a `marketdata_runs` row.
  - **Continues past a failing source**: one source down must not block the others. Exit code 0 when all succeeded, 2 when some failed.
- `provider.py`: `DbPriceProvider(session)` implementing the C04 `PriceProvider`:
  - `gold_24k_per_gram(on)`
  - `nav(key, on)` where key = ISIN, NSE symbol or AMFI code
  - `price(security_id, on)` for equities/ETFs
  - Each returns a value **and** staleness; raises `PriceUnavailable` when nothing exists within 10 days.

### 3. Wire into the API
- `value_asset` calls from the API get `prices=DbPriceProvider(session)`, so gold/ETF/SGB manual assets get values and notes such as "price as of 2026-10-27 (IBJA)".
- `/v1/networth` values holdings at the latest price (equity/ETF: `prices_eod`; MF: `nav_daily`), with each holding's `price_as_of` and `price_source`. If no price exists, fall back to `statement_value` with status `statement_value_used`.
- Add `market_data_as_of` to the networth response.

### 4. Classification refinement
After the AMFI master loads, `classify_holding` (C08b) prefers the AMFI category mapping (debt/liquid/gilt → debt; gold → gold; international/FoF overseas → equity_global; equity/hybrid-aggressive/index → equity_india; arbitrage → debt with tax note). Name keywords remain the fallback.

### 5. Fixtures and tests (no network in tests)
- Commit **small** real-format text excerpts: 20 lines of a bhavcopy, 15 lines of the equity list, a 3-section excerpt of NAVAll, 10 rows of the DEXINUS CSV, and a saved minimal IBJA HTML snippet. Public market data only, no personal data. Fetch them once during this run to get the exact current formats, then trim.
- Parser tests for each source, including holidays/empty files, malformed rows skipped with a warning, and category mapping for 8 AMFI section headers.
- Store idempotency; `latest_price` picks the right row; staleness; prune keeps held securities.
- Refresh CLI with all sources mocked: one source raising → exit code 2, the others stored, run recorded.
- API: a gold asset (20 g, 22k) valued from a `gold_prices` row; an MF holding valued from `nav_daily` instead of `statement_value`; fallback to statement value with status when no price exists.

### 6. One live smoke run (reviewer's machine, not a test)
After the code is done, run `python -m wealth.marketdata.refresh --only master,amfi,fx,gold` once against `TEST_DATABASE_URL`. Report the counts and any source failures in the result file. Do **not** commit fetched data.

## Out of scope
US prices (C09b), fundamentals / point-in-time financials for the research desk (C15), index constituents history (C15).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,importer,dev]"`, then `.venv\Scripts\python -m pytest`. All pass, with no network access in tests (patch `httpx`).
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. `alembic upgrade head` / `downgrade 0005_imports` work; OpenAPI refreshed
4. The live smoke run result is reported (source failures are acceptable if explained)
