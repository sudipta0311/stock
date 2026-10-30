# Wealth Copilot

Wealth Copilot is a personal wealth planning app for Indian investors and their households, in India or abroad (Denmark is supported). It brings savings, deposits, funds, shares, insurance, loans, and reminders into one Money Map. Its deterministic policy engine creates next steps, while research agents summarize market evidence.

## Architecture

```text
Phone or browser
  └─ Vite + React static PWA ── HTTPS JSON ── FastAPI API
                                              ├─ schemas, valuation, policy
                                              ├─ CAS and broker importers
                                              └─ market data and household services
                                                    │
                                                    └─ PostgreSQL with row-level security

GitHub Actions ── scheduled market, policy, alert, and research jobs ──┘
       └─ weekly research ── ModelRouter ── Azure AI Foundry
```

The API does not import the research package. Research runs in scheduled jobs, and code owns all money calculations. Household data is stored in PostgreSQL and isolated by row-level security.

## Repository layout

- `src/wealth/core/`, `schemas/`, `valuation/`, `rates/`, and `policy/` contain deterministic domain code.
- `src/wealth/importer/`, `db/`, `api/`, and `marketdata/` contain import, persistence, HTTP, and data-source code.
- `src/wealth/research/` and `alerts/` contain research agents and scheduled communications.
- `apps/web/` is the Vite, React, and TypeScript progressive web app.
- `e2e/` contains Playwright browser journeys. `tests/` mirrors the Python package.
- `rules/`, `schemas/`, `copy/`, and `evals/` contain versioned rules, schemas, interface copy, and research evaluation fixtures.
- `docs/cards/` tracks the implementation cards. `legacy/` is an archived, read-only reference.
- `.github/workflows/` contains deployment, migration, keep-awake, and scheduled job workflows.
- `tools/gmail-statements/` is the Google Apps Script that forwards statement emails to the inbox (the PWA serves a copy).

## Local development

Use Python 3.14, Node.js compatible with the lockfiles, and the local PostgreSQL 18 instance on `localhost:54329`. The development environment provides `TEST_DATABASE_URL`; keep its value in your environment and never paste it into logs or tracked files. Database tests create and remove their own schema and temporary API login.

Create an editable install with the extras needed across the app:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --no-cache-dir -e ".[dev,api,importer,research]"
```

Set `DATABASE_URL` to a non-owner API login that is a member of `wealth_app`. Set `OWNER_DATABASE_URL` to the database-owner connection for local migrations and scheduled jobs. The API must never use the owner connection. Copy setting names and safe placeholders from [`.env.example`](.env.example); supply credentials through environment variables or a secret manager.

### API

For local development, enable dev authentication and create a fresh signing secret in the current PowerShell session. Then migrate and start the API:

```powershell
$env:ENV = "dev"
$env:AUTH_MODE = "dev"
$env:DEV_AUTH_SECRET = [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
.venv\Scripts\python -m alembic upgrade head
.venv\Scripts\python -m uvicorn wealth.api.main:app --reload --port 8765
```

Development tokens are for local use only. Keep the token and signing secret out of source files, command transcripts, and logs.

### PWA

In a second PowerShell window:

```powershell
cd apps/web
npm ci
npm run gen:api
npm run gen:copy
$env:VITE_API_URL = "http://127.0.0.1:8765"
$env:VITE_AUTH_MODE = "dev"
npm run dev
```

Generated API types and copy files come from `schemas/openapi.json` and `copy/`; do not hand-edit their generated outputs. Standard builds use Vite's normal transforms. Set `$env:SANDBOX_BUILD = "1"` only when a restricted local sandbox blocks build subprocesses; CI and deployment use the standard build.

### Jobs

Market refresh, weekly policy, email, reminders, and research jobs run from `.github/workflows/`. Run them with recorded fixtures or fake model responses during development. The keep-awake workflow pings the API's `/healthz` endpoint every ten minutes; the health check does not wake or query PostgreSQL.

## Testing and evaluation

Run the Python suite, including database tests when `TEST_DATABASE_URL` is set, and the deployment configuration check:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check --no-cache src tests scripts
.venv\Scripts\python -m ruff format --check --no-cache src tests scripts
.venv\Scripts\python scripts/check_deploy_config.py
```

For dependency review, audit the installed Python environment and the shipped npm dependencies:

```powershell
.venv\Scripts\python -m pip_audit --skip-editable
cd apps/web; npm audit --omit=dev
cd ../../e2e; npm audit --omit=dev
```

For the PWA, run these from `apps/web` after refreshing generated files:

```powershell
npm run gen:api
npm run gen:copy
npm run typecheck
npm run lint
npm run format:check
npm test
npm run build
npm run budget
```

Playwright uses a local API, database schema, and static PWA. Install Chromium once, then run the journeys:

```powershell
cd e2e
npm ci
npx playwright install chromium
npm test
```

Research evaluations replay committed synthetic recordings without a network or an LLM:

```powershell
.venv\Scripts\python -m wealth.research.evals run --router fake --cases evals/golden --out out/evals/fake-current --budget-usd 0.50
```

Live evaluations use Azure AI Foundry and may incur charges. Run them deliberately with credentials supplied through environment variables, or dispatch the `Live research eval` GitHub Actions workflow. Live model output does not set portfolio amounts, rates, tax figures, weights, or dates.

## Deployment

The target is a static PWA and FastAPI service on Render with Neon PostgreSQL and GitHub Actions for scheduled jobs. Review [`render.yaml`](render.yaml) and the deployment workflows before configuring service URLs or secrets. Before deploying code that requires a new schema, run the **Database migrations** workflow and confirm it reaches the current Alembic head. The API uses the `wealth_api` login; owner credentials stay in migrations and scheduled jobs.

## Security and privacy

- Use synthetic financial fixtures and `@example.test` addresses only. Never add statements, personal identifiers, credentials, or generated signing keys to the repository.
- Tests use fakes and recorded fixtures. They do not call the network or a language model.
- Statements uploaded in Money are parsed in memory and not kept. Statements the Gmail helper sends wait in the household's inbox until they are read, then the PDF is dropped. Saved statement passwords are encrypted with AES-GCM (`STATEMENT_PASSWORD_KEY`).
- Research prompts contain no personal identifiers. The API remains independent of research and provider packages.
- Dev-token authentication can create a user when a valid token makes a request. A late request can recreate a bare user after deletion in dev or e2e. Production app-session tokens use the lookup introduced in migration 0025 and cannot recreate a deleted account; see `test_a_request_after_deleting_the_account_does_not_recreate_the_user`.
- Production migrations run through the **Database migrations** workflow before deploying dependent code. Never use the owner connection for API requests or change shared PostgreSQL roles by hand.
- Keep local environment files, Playwright browser downloads, build output, and evaluation reports untracked.

See [`AGENTS.md`](AGENTS.md) for repository rules and [`docs/UPGRADE_PLAN.md`](docs/UPGRADE_PLAN.md) for the implementation record.
