# C05: Database: SQLAlchemy models, Alembic migration, Postgres Row-Level Security, session helper

**Plan:** `docs/UPGRADE_PLAN.md` §7.3 (data model, RLS), §7.4 (security), §1.3 P0-2 (no tenant isolation today). Read `AGENTS.md` first. Builds on C02 schemas.

## Goal
A Postgres schema where **one user can never read or change another user's rows**, enforced by the database itself (RLS), not just by API code.

## Decisions (follow them)
- Postgres only (Neon in production). SQLAlchemy 2.0 typed ORM (`Mapped[...]`), driver `psycopg` 3 (`postgresql+psycopg://`). Alembic for migrations: `alembic.ini` at repo root, `script_location = src/wealth/db/migrations`.
- **All manual assets, insurance and loans live in one table, `manual_assets`.** `asset_type` is the C02 discriminator and `details` is JSONB validated by the C02 Pydantic model before writing. (This simplifies the plan's separate insurance/liabilities tables.)
- Money columns are `NUMERIC(16,2)`, rates are `NUMERIC(7,4)`, IDs are `UUID` (generated in Python with `uuid4`), and timestamps are `timestamptz`.
- Roles: migrations run as the **owner** (the connection user). The API never queries as the owner: every request transaction runs `SET LOCAL ROLE wealth_app` and `SELECT set_config('app.user_id', :uid, true)`.
  - The migration creates `wealth_app` as `NOLOGIN` if missing, then grants it table privileges and grants membership to `current_user`.
  - RLS is **enabled, not forced**, so the owner (scheduled jobs) can work across users. The API is protected because it always switches to `wealth_app`.

## Scope

### 1. Dependencies (extra `api` and `dev`)
`sqlalchemy>=2.0.36`, `psycopg[binary]>=3.2.10`, `alembic>=1.14`.

### 2. Models: `wealth/db/models.py`
User-owned tables. Each has `id UUID PK` and `user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE`, with an index on `user_id`. Exceptions: `users` uses `id` as the user key, and child tables keep `user_id` too (denormalised, so RLS stays a single-column check).

| Table | Key columns |
|---|---|
| `users` | id, email (unique, citext or lower-cased), display_name, auth_subject (unique), created_at, deleted_at |
| `profiles` | user_id (unique), tax_regime, tax_slab_pct, city_tier, timezone, language, risk_answers JSONB, risk_profile, updated_at |
| `household_members` | user_id, display_name, relationship, created_at |
| `goals` | user_id, name, goal_type, target_amount, target_year, currency, priority |
| `accounts` | user_id, member_id → household_members (nullable), institution, account_type, masked_number (CHECK: at most 4 trailing visible characters, e.g. `^X*[0-9A-Za-z]{0,4}$`) |
| `manual_assets` | user_id, member_id, account_id, asset_type, details JSONB, value_cached NUMERIC, value_as_of date, status (`active`/`closed`), created_at, updated_at |
| `asset_history` | user_id, asset_id → manual_assets, changed_fields JSONB, changed_at, changed_by (`user`/`import`/`system`) |
| `asset_cashflows` | user_id, asset_id, on_date, amount, kind (`deposit`/`premium`/`interest`/`maturity`/`withdrawal`), planned bool |
| `securities` **(shared, no RLS)** | id, market (`IN`/`US`), symbol, isin (unique nullable), name, security_type (`equity`/`etf`/`mf`/`bond`/`sgb`), currency |
| `holdings_lots` | user_id, account_id, member_id, security_id → securities, quantity NUMERIC(20,6), cost_per_unit NUMERIC(16,4) nullable, acquired_on date nullable, source (`cas`/`broker`/`manual`) |
| `import_jobs` | user_id, doc_type, status, parse_report JSONB, file_deleted_at, created_at |
| `import_items` | user_id, job_id → import_jobs, item_kind, payload JSONB, status (`new`/`changed`/`duplicate`/`unknown`/`accepted`/`rejected`) |
| `networth_snapshots` | user_id, on_date, totals JSONB, UNIQUE(user_id, on_date) |
| `actions` | user_id, week (text `YYYY-Www`), rank, kind, payload JSONB, status (`proposed`/`done`/`dismissed`), created_at |
| `exit_cards` | user_id, lot_id → holdings_lots, rules JSONB, status JSONB, updated_at |
| `reminders` | user_id, asset_id nullable, due_on, kind, sent_at nullable |
| `push_subscriptions` | user_id, endpoint (unique), keys JSONB, created_at |
| `consents` | user_id, purpose, granted_at, withdrawn_at |
| `audit_log` | user_id, at, actor, action, detail JSONB |

### 3. Migration `0001_initial`
Creates everything above, plus:
- `ALTER TABLE <t> ENABLE ROW LEVEL SECURITY` and policy `<t>_isolation` on every table except `securities`. The policy is `FOR ALL TO wealth_app USING (user_id = nullif(current_setting('app.user_id', true), '')::uuid) WITH CHECK (same)`. For `users`, compare on `id`.
- Grants: `SELECT, INSERT, UPDATE, DELETE` on all user tables to `wealth_app`; `SELECT` on `securities`.
- `downgrade()` drops everything cleanly (policies, tables). It does **not** drop the role.

### 4. Sessions: `wealth/db/session.py`
- `make_engine(url: str | None = None)` reads `DATABASE_URL` and normalises `postgres://` / `postgresql://` to `postgresql+psycopg://`.
- `@contextmanager user_session(engine, user_id: UUID) -> Session`: begins a transaction, `SET LOCAL ROLE wealth_app`, sets `app.user_id`, yields, commits on success, rolls back on error.
- `@contextmanager system_session(engine) -> Session`: owner role, for scheduled jobs and migrations only. Docstring warns that it bypasses RLS.
- `wealth/db/repo.py`: `add_manual_asset(session, user_id, asset: AnyAsset, ...)` validates with C02 models and writes `details` via `model_dump(mode="json")`. Also `get_manual_asset` and `list_manual_assets`.

### 5. Tests
- **Unit (no DB):** every table in metadata except `securities` has a non-null `user_id` (or is `users`); the migration source contains an RLS policy for each of those tables (parse the migration module's table list); URL normalisation.
- **Integration (`@pytest.mark.db`):** auto-skip when `TEST_DATABASE_URL` is unset.
  - Create a fresh throwaway schema or database for the session and run `alembic upgrade head` against it.
  - Users A and B, with rows created through `system_session`.
  - In `user_session(A)`: selecting `manual_assets` returns only A's rows; inserting a row with `user_id=B` raises (WITH CHECK); updating or deleting B's row affects 0 rows; `users` shows only A.
  - A session with no `app.user_id` (role `wealth_app` but no setting) sees 0 rows.
  - `system_session` sees both users.
  - Deleting user A cascades A's rows.
  - `alembic downgrade base` then `upgrade head` works.
- Register the `db` marker in `pyproject.toml`.

## Out of scope
API endpoints (C06/C07), market-data and research tables (C10/C14 add their own migrations).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,dev]"`
2. `.venv\Scripts\python -m pytest` → all pass. **DB tests must actually run** (not skip): the reviewer runs with `TEST_DATABASE_URL` set. If it's unset in your environment, say so in the result file.
3. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
4. With `TEST_DATABASE_URL` set: `.venv\Scripts\python -m alembic upgrade head` then `.venv\Scripts\python -m alembic downgrade base` both succeed (use `-x dburl=...` or an env var as you prefer; document how in the README's dev section)
