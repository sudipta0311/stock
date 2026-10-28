# C09b: Importer part 2b (priority: Danish brokers Nordnet, Saxo, Danske Bank first; then IBKR; Vested/INDmoney last, used by Indian-resident members for LRS): US brokers (IBKR + mapping fallback for Vested/INDmoney), USD securities keyed by symbol, USD lots with acquisition FX, domicile for estate-tax checks

**Plan:** `docs/UPGRADE_PLAN.md` §3.3 (US broker CSVs), §4.5 (route C: US-situs threshold US$60k, UCITS preference), §4.10 (foreign equity: LTCG after 24 months; cost converted to INR at acquisition), §3.2 (US holdings count toward `equity_global`). Read `AGENTS.md` first. Builds on C09 (broker import engine, mapping fallback, FIFO, merge statuses), C10 (`fx_rates`, `DbPriceProvider`), C08b (`app_upsert_security`, import API).

## Goal
A resident Indian who invests through IBKR (or via Vested/INDmoney exports) can import US holdings and trades. These are valued in rupees at the current USD/INR rate, carry dollar cost plus the exchange rate on the purchase date (for Indian tax), and are flagged by domicile, so the app can warn when US-situs assets approach the US$60k estate-tax threshold.

## Scope

### 1. Migration (next free number; revise the current head)
- `securities`: add `symbol` (generic ticker), `exchange`, `domicile` (ISO country from the ISIN prefix, or `US` for US-listed tickers without an ISIN), with a unique index on `(market, symbol)` where `symbol` is not null.
- New SECURITY DEFINER `app_upsert_security_by_symbol(p_market, p_symbol, p_exchange, p_name, p_type, p_currency, p_isin, p_sleeve, p_category) RETURNS uuid`:
  - pinned search_path; validates the symbol (`^[A-Z0-9.\-]{1,15}$`), market `US`, currency `USD`
  - an ISIN is optional but validated when present
  - upserts on `(market, symbol)`
  - `REVOKE ALL FROM PUBLIC`; `GRANT EXECUTE TO wealth_app`
- `holdings_lots`: add `currency` (default `INR`), `fx_at_acquisition NUMERIC(12,6)`, `cost_per_unit_inr NUMERIC(16,4)`.
- `realised_trades`: add the same currency/fx columns.
- Downgrade clean.

### 2. IBKR Activity Statement CSV (`wealth/importer/brokers/ibkr.py`, pure)
- Multi-section CSV: column 1 = section name, column 2 = `Header`/`Data`/`SubTotal`/`Total`.
  - Parse the **Open Positions** section (Asset Category `Stocks`/`ETFs` only; skip Forex/Options with warning `unsupported_asset_category`): symbol, currency, quantity, cost price, close price, value.
  - Parse the **Trades** section: symbol, date/time, quantity (negative = sell), price, currency, proceeds, commission. FIFO lots come from C09's `fifo_lots`.
  - Parse **Financial Instrument Information** when present to get ISIN, exchange and description.
- Header detection by section headers, not positions.
- Non-USD positions (e.g. LSE-listed UCITS in USD or GBP): keep the currency. Domicile from the ISIN prefix (`IE` → not US-situs).
- **Don't invent IBKR columns you can't support from the section headers.** Unknown sections are ignored with a warning.

### 3. EU brokers (primary users live in Denmark) + Vested / INDmoney / other exports
- **Nordnet, Saxo Bank, Danske Bank (eBanking/Investering) exports:** CSV/XLSX holdings and transactions. Seed a layout in `rules/brokers.yaml` **only** where you find official or published documentation of the export columns (record `source_note`, `verified: false`). Otherwise rely on the C09 `needs_mapping` flow with `market` per row (`DK`/`EU`/`US`) and currency `DKK`/`EUR`/`USD`.
- Securities from EU exchanges (Nasdaq Copenhagen, Xetra, Euronext, LSE) are keyed by `(market, symbol)` via `app_upsert_security_by_symbol` with `market` in `{DK, EU, GB}` (extend its validation). Domicile comes from the ISIN prefix (`IE`/`LU` UCITS → not US-situs).
- Classification: UCITS equity ETFs → `equity_global`; Danish/EU single stocks → `equity_global` (the user's 'where you live' sleeve); bond ETFs → `debt`; gold ETCs → `gold`.
- Valuation in INR via `fx_rates` (DKK/INR, EUR/INR from the C26a ECB extension; if that isn't merged yet, compute the EUR cross on the fly from the ECB rows C10 stores).

### 3b. Vested / INDmoney / other US exports
No seeded layouts (no verified evidence). They go through C09's `needs_mapping` flow with `market=US`. The mapping UI must offer `currency` and `exchange` as mappable fields, defaulting to USD/US.

### 4. FX and valuation
- Lot acquisition FX = USD/INR from `fx_rates` on the trade date, else the latest earlier date within 7 days, else the lot is marked `fx_missing` (tax estimates then show "add exchange rate").
- Note in code and result file: Indian rules prescribe the SBI TT buying rate. The reference rate is an approximation, flagged in tax notes.
- `/v1/networth` values US holdings at the latest close price (statement close if there's no price feed: US prices are not fetched in v1) × the latest USD/INR. Each holding shows `price_as_of`, `fx_as_of` and `value_usd`.
- Sleeve via classification:
  - US stock or equity ETF → `equity_global`
  - bond ETF names (Bond, Treasury, TIPS, Aggregate) → `debt`
  - gold ETFs (names containing Gold, or symbols GLD, IAU, GLDM, SGOL) → `gold`

### 5. Estate-tax exposure helper (`wealth/policy/us_situs.py`, pure)
- `us_situs_value_usd(holdings) -> Decimal`: sum of holdings whose domicile is `US` (US-listed stocks/ETFs). UCITS (`IE`, `LU`) excluded.
- Feeds C12's `RouteContext.us_situs_holdings_usd`.
- Test: VOO 100 sh @ $500 + CSPX 50 @ $600 → US-situs **$50,000**; CSPX excluded.

### 6. Completeness and checklist
`us_investments` is done when any confirmed holding has market `US` or currency `USD` (C08b rule).

### 7. Tests (synthetic fixtures only; build CSVs in tests)
- IBKR parse: open positions (2 stocks, 1 ETF, 1 UCITS in LSE, 1 forex row ignored), trades with a partial sell → FIFO lots with USD cost and acquisition FX from a fixture `fx_rates`, and the `fx_missing` path.
- Merge statuses as in C09 (enrich / replace_with_lots / quantity_mismatch) for US lots.
- Net worth: US holdings in INR at the latest FX, sleeves per classification.
- Isolation for US lots.
- Domicile and US-situs helper.

## Out of scope
US price feed (later), Schedule FA report generation (later), Form 67 / foreign tax credit (later).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,importer,dev]"`, then `.venv\Scripts\python -m pytest`. All pass; DB tests run.
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. `alembic upgrade head` / downgrade one step work; OpenAPI refreshed; no binary fixtures committed
