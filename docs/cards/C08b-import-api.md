# C08b: Import API: upload (in memory, no retention) → staging → review → confirm → securities + holdings; holdings in net worth

**Plan:** `docs/UPGRADE_PLAN.md` §3.3 (import pipeline steps 1–5, review screen, monthly refresh), §7.3 (`import_jobs`, `import_items`, `holdings_lots`, shared `securities`), §7.4 (statement passwords never stored, files deleted), Appendix D.2 (512 MB, one parse at a time). Read `AGENTS.md` first. Builds on C05 (RLS, `wealth_app`), C06 (API, problem+json), C07 (net worth), C08 (`parse_cas`, `ParsedStatement`).

## Goal
A user uploads a CAS PDF with its password. Within seconds they see what was found (new, changed, duplicate, unknown), fix anything unclear, and confirm. Holdings then appear in their net worth. Nothing about the file or password survives the request.

## Scope

### 0. Follow-up from the C07 review
Manual assets that need a market price (physical/digital gold, gold ETF/MF, SGB) currently store `valuation = null` and **silently drop out** of `/v1/networth`.
- Add `valuation_status` to every asset response: `valued` | `price_needed` | `rate_needed` | `not_applicable` (term/health/loans).
- Add `unvalued_assets: [{id, asset_type, label, reason}]` to `/v1/networth`.
- Test both.

### 1. Migration `0005_imports`
- `import_jobs`: add `status` (`ready`/`confirmed`/`failed`/`discarded`), `error_code`, `statement_date`, `source` (`nsdl`/`cdsl`/`cams_kfin`), `counts` JSONB, `confirmed_at`. Always set `file_deleted_at = now()` on creation: the file is never stored.
- `import_items`: add `match_lot_id`, `match_security_id`, `diff` JSONB, `user_override` JSONB.
- `holdings_lots`: add `account_ref` (masked), `statement_value NUMERIC(16,2)`, `statement_date`, `import_job_id`.
- `securities`: add `sleeve` (C02 `Sleeve` value), `category` (free text, e.g. `"mf:debt:liquid"`).
- SECURITY DEFINER `app_upsert_security(p_market text, p_isin text, p_symbol text, p_name text, p_type text, p_currency text, p_sleeve text, p_category text) RETURNS uuid`:
  - pinned `search_path`
  - validates ISIN format (`^[A-Z]{2}[A-Z0-9]{9}[0-9]$`) and market/type/currency enums
  - upserts on ISIN; never overwrites a non-null `sleeve`/`category` with null
  - `REVOKE ALL FROM PUBLIC`, `GRANT EXECUTE TO wealth_app`
  - This is the only way the API writes the shared table.

### 2. Classification: `wealth/importer/classify.py` (pure, table-tested)
`classify_holding(h: ParsedHolding) -> (sleeve, category)`:
- `sgb` → gold
- `bond` → debt
- MF/ETF by name keywords:
  - Liquid / Overnight / Money Market / Gilt / Debt / Bond / Corporate Bond / Banking & PSU / Short Duration / Bharat Bond → debt
  - Gold / Silver → gold
  - Nasdaq / S&P / US / Global / International / Overseas / FoF (overseas) / Hang Seng / NYSE / FANG → equity_global
  - Arbitrage → debt (with category `mf:arbitrage`, taxed as equity, noted)
  - everything else → equity_india
- `equity` with an `INE` ISIN → equity_india; a non-`IN` ISIN → equity_global
- unknown → `None` (item status `unknown`)
- At least 25 name cases in the test table. C10 refines this later with AMFI categories.

### 3. Endpoints (all in `user_session`; problem+json errors with `code`)
| Method | Path | Behaviour |
|---|---|---|
| POST | `/v1/imports` | multipart `file` + optional `password` + optional `doc_type`. Reject before reading past `MAX_UPLOAD_MB` → **413** `too_large`. First bytes must be `%PDF` → else **415** `not_a_pdf`. Parse with `parse_cas` in a worker thread with a **20 s timeout** (→ 504 `parse_timeout`) behind a process-wide semaphore of 1 (Render free has 0.1 CPU and 512 MB). On `ParseError`, return 422 with its `code` (`wrong_password`, `unsupported_format`, …) and record a `failed` job with only `error_code`. On success, create job + items and return `{job_id, status, source, statement_date, counts: {new, changed, duplicate, unknown}, warnings}`. |
| GET | `/v1/imports` | the user's last 20 jobs (no items) |
| GET | `/v1/imports/{id}` | job + items: `{id, kind, isin, name, quantity, statement_value, account_ref, sleeve, status, diff, match}` |
| PATCH | `/v1/imports/{id}/items/{item_id}` | `{status: "accepted"|"rejected"}` and/or `{override: {kind, sleeve}}` for unknown items |
| POST | `/v1/imports/{id}/confirm` | Apply all `new`/`changed`/`accepted` items (skip `rejected`, `duplicate`, and `unknown` without override). For each: `app_upsert_security` → **replace** the user's lots with `source='cas'` for the same `(security, account_ref)` with one statement lot (quantity, statement_value, statement_date). If the statement has transactions for that security (CAMS/KFin), create one lot per buy with cost and date instead, netting sells FIFO. Never touch `broker`/`manual` lots. Mark the job `confirmed`, write `audit_log` (counts only), upsert today's `networth_snapshots`. Idempotent: confirming twice → 409. |
| DELETE | `/v1/imports/{id}` | discard (status `discarded`, items deleted) |

**Item status rules** (against the user's existing `cas` lots for the same ISIN + account_ref):
- no lot → `new`
- same quantity → `duplicate`
- different quantity → `changed`, with `diff = {quantity: {old, new}}`
- missing ISIN or unclassifiable → `unknown`

### 4. Net worth includes holdings
`/v1/networth` now adds market holdings (from `holdings_lots` joined to `securities.sleeve`) valued at `statement_value` (live prices come in C10). They count toward `by_sleeve`, the liquidity ladder (`instant`), completeness, and `investable_total`.

### 5. Privacy and safety tests
- Capture all log records during a successful upload and a wrong-password upload. Assert neither the password nor any statement line appears.
- Assert no file is written: monkeypatch `builtins.open`, `tempfile.NamedTemporaryFile` and `tempfile.mkstemp` to fail if called with write modes during the upload request.
- Responses never include the password.
- 413, 415, 422 (`wrong_password`, `unsupported_format`) paths.
- Cross-tenant: B gets 404 for A's job and items, and can't confirm A's job.
- Confirm flow end to end, using the C08 synthetic encrypted PDF builder:
  1. Upload → counts → confirm → net worth shows the holdings.
  2. Upload again (same statement) → all items `duplicate`.
  3. Upload a variant with one changed quantity → exactly one `changed`, with the right diff.
- Concurrency: two uploads at once → the second waits (semaphore), both succeed.
- Asset `valuation_status` and net worth `unvalued_assets` (section 0).

## Out of scope
Broker statements and cross-source de-dup (C09), live prices (C10), PWA screens (C21).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,importer,dev]"` (the `api` extra must now include what the import endpoints need)
2. `.venv\Scripts\python -m pytest`. All pass; DB tests run with the non-owner login fixture.
3. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
4. `alembic upgrade head` / `downgrade 0004_money_map` both work; OpenAPI export refreshed
5. Report: peak RSS of a uvicorn process after 3 uploads of the synthetic PDF (measure with `psutil` in a test helper or a one-off script). It must stay under 300 MB.
