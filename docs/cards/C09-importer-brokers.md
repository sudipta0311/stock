# C09: Importer part 2: broker holdings & tradebooks (India), column-mapping fallback, FIFO lots, merge with CAS

**Plan:** `docs/UPGRADE_PLAN.md` §3.3 (broker holdings + tradebook give cost and dates for tax; "CAS + tradebook covers ~80%"; review screen merges duplicates), §4.9–4.10 (lots needed for tax-aware exits). Read `AGENTS.md` first. Reference `legacy/utils/broker_parser.py` and `legacy/src/stock_platform/utils/direct_equity_merge.py`; port their logic, don't import them. Builds on C08b (import jobs/items/confirm, `app_upsert_security`, `holdings_lots`).

## Goal
Users can upload their broker's **holdings** export (quantity + average cost) and/or **tradebook** (every buy and sell). The app turns them into tax lots without double-counting what the CAS already imported. For any broker format it doesn't recognise, the user maps the columns once, and the app remembers.

## Decisions (follow them)
- Formats: CSV and XLSX. XLSX via `openpyxl>=3.1` (read-only mode) in the `importer` extra. No pandas.
- Header-driven, not position-driven. `rules/brokers.yaml` lists known layouts. Each has: `broker`, `kind` (`holdings` or `tradebook`), `fingerprint` (a set of normalised header names that must all appear), and a `columns` map from our canonical fields to header aliases.
  - Canonical holdings fields: `isin`, `symbol`, `name`, `quantity`, `avg_price`, `last_price`, `exchange`.
  - Canonical tradebook fields: `isin`, `symbol`, `trade_date`, `trade_type` (buy/sell), `quantity`, `price`, `exchange`, `order_id`, `trade_id`.
- Seed `rules/brokers.yaml` for **Zerodha (Console holdings + tradebook), Groww (holdings), ICICI Direct (holdings)** from the legacy parser. Add **Angel One** and **Upstox** holdings layouts marked `verified: false`. Every layout records `source_note`. **Don't invent columns for brokers you have no evidence for.** Unknown layouts go to the mapping fallback.
- Header normalisation: lower-case, strip, collapse spaces/underscores/dots, drop units in brackets, e.g. `"Avg. cost (₹)"` → `avg cost`. Rows: skip blank/total/summary rows; numbers parsed with C02 helpers (Indian grouping allowed); dates in `YYYY-MM-DD`, `DD-MM-YYYY`, `DD/MM/YYYY`, `DD-Mon-YYYY`.
- **Fallback mapping:** when no fingerprint matches, the import job gets status `needs_mapping`. It stores only the normalised **header names** (never data rows) plus up to 3 sample rows with values masked (digits → `9`, letters → `x`). The user submits a mapping, saved per user in a new table `import_templates` (`user_id, name, kind, header_fingerprint, columns JSONB`) and reused automatically next time the same headers appear.
- Symbol → security: use the ISIN when present. Otherwise match `symbol` + `exchange` against `securities.nse_symbol` (C10 filled the NSE master list). Unresolved rows become items with status `unknown` and reason `symbol_not_found`.

## Scope

### 1. Parsers (`wealth/importer/brokers/`, pure, no DB)
- `read_table(data: bytes, filename: str) -> (headers, rows)` (CSV with BOM/encoding detection: utf-8-sig, then cp1252; XLSX first sheet or the sheet whose header row matches; header row = first row with ≥ 3 non-empty cells containing a known alias).
- `detect_layout(headers, layouts, templates) -> Layout | None`.
- `parse_holdings(rows, layout) -> list[BrokerHolding]` and `parse_tradebook(rows, layout) -> list[Trade]`.
- `fifo_lots(trades) -> (open_lots, realised: list[RealisedLot], warnings)`:
  - Per security, sorted by date then trade_id.
  - Sells consume the oldest buys.
  - A sell with no buys left gives warning `sell_without_buy` (statement starts mid-history), and the quantity is ignored.
  - Same-day buy+sell (intraday) is netted out and reported as `intraday_netted`.
  - Returns lots with `quantity, cost_per_unit, acquired_on, trade_ids`.

### 2. Import flow (extend C08b's API; same staging tables)
- `POST /v1/imports` accepts CSV/XLSX for broker files (detect by extension + content: `PK` zip header for XLSX, text for CSV). Size cap and semaphore as in C08b. **Excel macros/formulas are never evaluated** (`openpyxl` read-only, `data_only=True`).
- `needs_mapping` flow:
  - `GET /v1/imports/{id}` returns `headers` + masked samples + a suggested mapping (best alias guess).
  - `POST /v1/imports/{id}/mapping` `{kind, columns: {canonical: header}, save_as: "My broker"}` → parse → items.
- **Item statuses for broker data:**
  - Holdings file: a security with an existing `cas` lot of equal total quantity → `enrich` (add the average cost to that lot, keep `source='cas'`, set `cost_per_unit`). Different quantity → `changed`. None → `new`.
  - Tradebook: per security, if FIFO open quantity == current total of the user's lots for that security (any source) → `replace_with_lots` (on confirm, delete that security's `cas`/`broker` lots and insert the FIFO lots with `source='broker'`). If it differs → `quantity_mismatch` with both numbers; the user picks *keep statement* or *use tradebook*.
  - Realised sells are stored in a new table `realised_trades` (user-owned, RLS) for the tax engine's FY-to-date realised gains.
- **Confirm** applies the decisions atomically per job. Never deletes `manual` lots.

### 3. Migration: next free number after the current head (check `src/wealth/db/migrations/versions/`), revising that head
`import_templates` and `realised_trades` (both with `user_id`, RLS policy + grants like C05). `holdings_lots.trade_ids JSONB`. Downgrade clean.

### 4. Synthetic fixtures (no real data)
`tests/fixtures/brokers/` with CSV and generated-XLSX (build XLSX at test time with openpyxl; don't commit binaries) for each seeded layout, plus one "unknown broker" CSV. All symbols, ISINs (check-digit-valid), quantities and dates are synthetic.

### 5. Tests
- Header normalisation table (≥ 12 cases incl. `"Avg. cost (₹)"`, `"Qty."`, `"ISIN Code"`).
- Detection for every seeded layout; the unknown CSV → `needs_mapping`; a saved template auto-applies on the next upload.
- FIFO worked example:
  - Buys: 10 @ ₹100 (2024-01-10), 5 @ ₹120 (2024-06-01). Sell 12 @ ₹150 (2025-02-01).
  - Open lot = **3 @ ₹120 (2024-06-01)**.
  - Realised = 10 @ cost ₹100 (LT) + 2 @ ₹120 (ST).
  - Sell without buys → warning. Intraday buy+sell same day → netted.
- Merge:
  - CAS lot 20 shares + holdings file 20 @ avg ₹250 → `enrich`; after confirm, cost ₹250 on the CAS lot.
  - Tradebook totalling 20 open → `replace_with_lots`; after confirm, FIFO lots replace the CAS lot.
  - Tradebook totalling 18 → `quantity_mismatch`.
- Privacy: masked samples contain no original digits/letters; no file written; isolation (B can't see A's templates, jobs or realised trades).

## Out of scope
US brokers (IBKR/Vested/INDmoney) and USD securities (C09b). The securities master (C10). Tax computations (already in C13, which reads `realised_trades` later).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,importer,dev]"`, then `.venv\Scripts\python -m pytest`. All pass; DB tests run (non-owner login fixture).
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. `alembic upgrade head` / downgrade one step work; OpenAPI export refreshed
4. No binary fixtures committed (`git status` shows no `.xlsx`/`.pdf`)
