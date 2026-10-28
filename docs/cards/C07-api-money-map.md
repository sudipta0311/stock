# C07: Money Map API: members, accounts, assets (with instant valuation), goals, quick-add parser, net worth, liquidity ladder, maturity calendar, completeness

**Plan:** `docs/UPGRADE_PLAN.md` §3.1 (setup flow, autosave, rough-is-fine, quick-add, household), §3.2 (asset catalogue), §3.5 (what the Money Map shows, the "aha" insight), §7.3 (data model). Read `AGENTS.md` first. Builds on C02 (schemas), C03/C04 (`value_asset`), C05 (DB, RLS), C06 (API app, auth, `user_session` dependency), C11 (`WealthSnapshot`).

## Goal
Everything the PWA needs to capture a user's whole financial picture, simply. Each write is validated, valued immediately, and stored. Each read returns money as fixed-point strings plus the valuation notes ("estimate", "add start month for exact value"…).

## Scope

### 0. Follow-ups from the C06 live review (do these first, each with a regression test)
- **Secret leak in settings errors.** With `AUTH_MODE` unset, startup fails with a pydantic `ValidationError` whose message includes `input_value={'DATABASE_URL': 'postgre...'}`. Pydantic truncates long values, but a long Neon URL can still expose part of the password in Render logs. Set `hide_input_in_errors=True` on the `Settings` model config. Test: `str()` of the settings error contains neither the URL host nor `postgresql://`.
- **Migration 0002 resets `LOGIN`.** `ALTER ROLE wealth_api NOLOGIN ...` runs every time 0002 is applied. Roles are cluster-wide, so every test run (and any production re-run) turns off the `LOGIN` that ops enabled, and the API then cannot connect. Change 0002 so it creates the role `NOLOGIN` only when it's missing. When the role exists, it may enforce `NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE` but must never change `LOGIN`. Test: set `wealth_api` to `LOGIN` on the test cluster, run `downgrade base` + `upgrade head`, and assert `rolcanlogin` is still true. Restore the original attribute in teardown.
- **Hidden cause.** `api_engine` raises `ValueError("Unable to verify API database role capabilities") from None`, which hides errors such as `role "wealth_api" is not permitted to log in`. Include a safe hint: the exception class name and the Postgres SQLSTATE / short message with any URL, host or password removed. Test: a login that isn't permitted produces a message containing `not permitted to log in` and not the password.

### 1. Migration `0004_money_map` (revises `0003_profile_financial_context`)
- `profiles.checklist JSONB NOT NULL DEFAULT '[]'` (the categories the user ticked in "What else do you have?")
- `manual_assets.valuation JSONB` (the last `Valuation` as JSON: value_today, maturity_value, maturity_date, accrued_interest, principal_so_far, method, notes)
- RLS and grants are unchanged (new columns inherit them). Downgrade removes the columns.

### 2. Endpoints (all under `user_session`; all money in/out as fixed-point strings; RFC 7807 errors)
| Method | Path | Notes |
|---|---|---|
| GET/POST | `/v1/members` | POST `{display_name, relationship}`; a `self` member is auto-created on first `/v1/me` if none exists |
| PATCH/DELETE | `/v1/members/{id}` | deleting a member with assets → 409 with a count |
| GET/POST | `/v1/accounts` | `masked_number` must already be masked (≤ 4 visible trailing chars) or 422 |
| PATCH/DELETE | `/v1/accounts/{id}` | |
| GET | `/v1/assets` | filters `asset_type`, `member_id`, `status` (default active); each item includes `valuation` |
| POST | `/v1/assets` | body = any C02 asset (accepts Indian shorthand like `"5L"`). Validate → `value_asset(asset, today, assumed_start=today)` (loans skip valuation; market-priced manual gold without prices stores `valuation=null` with note `price_needed`) → store. On `RateUnavailable` return **422** with `code: "rate_needed"` and `field: "rate_pct"` so the PWA can ask for the certificate rate. Returns the stored asset. |
| GET | `/v1/assets/{id}` | includes `valuation` and the last 20 `asset_history` entries |
| PATCH | `/v1/assets/{id}` | partial update: merge into stored details → re-validate the full model → revalue → write `asset_history` (`changed_fields` = keys whose values changed, with old/new as strings) |
| DELETE | `/v1/assets/{id}` | soft close (`status=closed`) by default; `?hard=true` deletes |
| POST | `/v1/assets/quick-add` | `{text}` → `{asset_type, draft, missing: [field], confidence: "high"/"medium"/"low"}`. **Doesn't save.** Deterministic parser only (no LLM). |
| GET/POST/PATCH/DELETE | `/v1/goals[/{id}]` | C02 `Goal` |
| PUT | `/v1/me/checklist` | `{categories: [...]}`, values from the Money Map tiles: `epf ppf fd post_office nps insurance gold bonds savings loans property us_investments` |
| GET | `/v1/networth` | see §3 |

Cross-tenant ids always return **404** (never 403). RLS hides the row; the API just reports not found.

### 3. Net worth (`GET /v1/networth?member_id=`)
Build a `WealthSnapshot` from the active manual assets (using the stored valuation, revalued lazily if `value_as_of < today`) and holdings (none yet; `holdings_lots` is empty until C08/C09). Return:
- `net_worth`, `assets_total`, `liabilities_total`, `investable_total`
- `by_sleeve`: list of `{sleeve, amount, pct}`
- `liquidity_ladder`: `{instant, within_1y, locked}`
- `debt_insight`: `{debt_pct_including_locked, locked_debt_amount, message_key: "debt_share"}`, the §3.5 "aha" figure
- `maturity_calendar`: the next 12 months of dated events, sorted:
  - FD/RD/NSC/KVP/SCSS/MIS/PO TD/bond maturity
  - SGB maturity
  - traditional policy maturity
  - term/traditional premium due (the start-month anniversary, if known)
  - health renewal month
  - a PPF "deposit before the 5th" reminder each April
  Each event is `{date, asset_id, kind, amount|null, label}`.
- `completeness`: `{ticked: [...], done: [...], pct}`. A category is done when at least 1 active asset of a matching type exists. With nothing ticked, pct = 0.
- `as_of`

Upsert `networth_snapshots` for `(user, today)` on each call.

### 4. Quick-add parser (`wealth/importer/quick_add.py`): pure, table-tested
Case-insensitive, tolerant of `₹`, commas, `%`, `till`/`until`/`maturing`, and month formats from C02. Must handle at least:
| Text | Result |
|---|---|
| `FD 5L SBI 7.1% till Mar 2027` | fd, bank SBI, principal 500000.00, rate 7.1, maturity 2027-03 → missing [] |
| `fd 2,00,000 hdfc 7.25 % mar-2028` | fd, HDFC, 200000.00, 7.25, 2028-03 |
| `PPF 6.5 lakh` | ppf, balance 650000.00 → missing [opened_fy_start_year] |
| `EPF 12L` | epf, balance 1200000.00 |
| `savings 80k icici` | savings, ICICI, 80000.00 |
| `NSC 50000 Jan 2025` | po_scheme nsc, amount 50000.00, purchase 2025-01 |
| `gold 20 g 22k` | gold physical, grams 20, purity 22 |
| `home loan 35L 8.5%` | loan home, outstanding 3500000.00, rate 8.5 |
| `credit card 60k 42%` | loan credit_card, 60000.00, 42 |
| `LIC 10L premium 52000` | traditional_policy, insurer LIC, sum_assured 1000000.00, annual_premium 52000.00 → missing [plan, start_year, term_years, policy_kind] |
| `term 1cr hdfc life 2050` | term_insurance, insurer "HDFC Life", cover 10000000.00, ends_year 2050 |
| `SCSS 15L Oct 2026` | po_scheme scss, 1500000.00, purchase 2026-10 |
| `random words` | 422 problem with `code: "not_understood"` |

The bank and insurer canonical names come from a small YAML list `rules/institutions.yaml` (SBI, HDFC Bank, ICICI Bank, Axis Bank, Kotak, IDFC First, Post Office, LIC, HDFC Life, ICICI Prudential Life, SBI Life, Max Life, Tata AIA, Star Health, Niva Bupa, Care Health). Unknown names are kept as typed.

### 5. Tests
- Parser table above (every row).
- Per-endpoint happy paths.
- PATCH writes history with only the changed fields.
- `rate_needed` 422 for an NSC bought in a month with no rate and no `rate_pct`.
- Isolation: user B gets 404 for every A resource (members, accounts, assets, goals) and B's `/v1/networth` excludes A's data.
- Net-worth worked example (put the numbers in the docstring):
  - savings ₹6,00,000
  - EPF ₹15,00,000
  - PPF ₹5,00,000
  - FD ₹4,00,000 at 7% (start 2026-01, maturity 2028-01)
  - term cover ₹1 Cr
  - home loan ₹20,00,000
  Check: assets_total = sum of valuations; liabilities ₹20,00,000; net worth = assets − liabilities; ladder instant ₹6,00,000; the debt-insight percentage; the FD maturity appears in the calendar only when within 12 months of `as_of` (use a frozen `as_of` via a dependency override).
- Completeness: tick `[epf, fd, insurance]`, add EPF + FD → pct 66.67.

## Out of scope
CAS/broker imports (C08/C09), market prices (C10), recommendations.

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,dev]"`, then `.venv\Scripts\python -m pytest`. All pass; DB tests run with a non-owner login (reuse the C05/C06 fixture).
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. `.venv\Scripts\python -m wealth.api.export_openapi` → `schemas/openapi.json` refreshed
4. `alembic upgrade head` and `alembic downgrade 0003_profile_financial_context` both work
