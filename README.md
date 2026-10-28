# Wealth Copilot

Wealth Copilot is a personal wealth planning app for organizing financial assets and producing rules-based portfolio guidance. Its architecture and migration plan are documented in [the upgrade plan](docs/UPGRADE_PLAN.md).

## Repository layout

- `src/wealth/` — Python package and domain subpackages
- `tests/` — automated tests
- `rules/` and `schemas/` — versioned rule data and exported JSON Schemas
- `apps/web/` — planned static web app
- `docs/cards/` — implementation cards
- `legacy/` — archived Streamlit app and its supporting files

## Development setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --no-cache-dir -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m ruff format --check src tests
```

## Database

- `DATABASE_URL` is the API connection string. Its login must be a non-superuser, must not bypass
  RLS, must own no tables in the current schema, and must be a member of `wealth_app` (directly or
  through another role).
- `OWNER_DATABASE_URL` is for Alembic migrations and scheduled jobs only. It must use the database
  owner role. The API must never use this connection string.
- `TEST_DATABASE_URL` is the local PostgreSQL URL used for isolated database tests and as the
  migration fallback when `OWNER_DATABASE_URL` is unset.

As a one-time database-owner operation, enable the API role login with a generated password and
store that password in the deployment secret manager:

```sql
ALTER ROLE wealth_api LOGIN PASSWORD '<generated>';
```

Run migrations with `OWNER_DATABASE_URL` set:

```powershell
.venv\Scripts\python -m alembic upgrade head
.venv\Scripts\python -m alembic downgrade base
```

See [`docs/cards/README.md`](docs/cards/README.md) for how implementation work is organized.

`legacy/` holds the old Streamlit app for reference only.

## Run the API locally

Development authentication is disabled by default. Set both development mode variables and a
generated signing secret of at least 32 characters. Set `DATABASE_URL` to a non-owner login that
is a member of `wealth_app`; set `OWNER_DATABASE_URL` to the migration-owner connection:

```powershell
$env:ENV = "dev"
$env:AUTH_MODE = "dev"
$env:DEV_AUTH_SECRET = [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
.venv\Scripts\python -m alembic upgrade head
.venv\Scripts\python -m uvicorn wealth.api.main:app --reload
```

Mint a short-lived token for local requests with the same `DEV_AUTH_SECRET`:

```python
import datetime
import jwt, os
print(jwt.encode({"sub": "local-user", "email": "user@example.test", "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)}, os.environ["DEV_AUTH_SECRET"], algorithm="HS256"))
```

## Run the PWA locally

In PowerShell, install the locked frontend dependencies, refresh the generated API types, then start Vite:

```powershell
cd apps/web
npm ci
npm run gen:api
$env:VITE_API_URL = "http://127.0.0.1:8765"
$env:VITE_AUTH_MODE = "dev"
npm run dev
```

Use the local development token snippet above to sign in. The PWA keeps it in memory and
`sessionStorage` for the current browser session; it is not part of the source or build output.

The normal `npm run build` uses Vite's standard transforms and minifier. In a restricted local
sandbox that blocks esbuild child processes, set `$env:SANDBOX_BUILD = "1"` before building; that
enables the local TypeScript transform and asset minification workaround. Leave it unset for
Render and GitHub Actions builds.
