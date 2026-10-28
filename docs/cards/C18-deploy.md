# C18: Deployment: Render Blueprint (static PWA + free API), GitHub Actions (CI, daily/weekly jobs, migrations), DEPLOY.md, config checker

**Plan:** `docs/UPGRADE_PLAN.md` §7.1 (launch architecture: Render free tier + Neon + GitHub Actions; the API holds **no LLM keys**), §7.4 (secrets in Render env / GitHub secrets; data residency), Appendix D (free-tier limits, the job schedule table, render.yaml sketch, upgrade path). Read `AGENTS.md` first. Builds on every backend card and C19 (PWA build).

## Goal
A repo that deploys with one Render Blueprint plus a handful of secrets, and runs all scheduled work in GitHub Actions. **No secret value is ever committed.**

## Scope

### 0. Owner's existing services (decided; follow this)
- **Database: the owner's new Neon project in Frankfurt (`aws-eu-central-1`, Postgres 18)**, used only by this app. The old app stays on its own US project; never touch it.
  - The app uses a **separate database `wealth`** in that project; every migration runs there.
  - Roles are per Neon project. Create `wealth_app`/`wealth_api` only through the migrations/DEPLOY.md steps.
  - The reviewer runs the migrations; you don't have the URL.
- **Login: Google**, reusing the owner's existing OAuth 2.0 **Web application** client.
  - Google's token endpoint needs the client secret even with PKCE, so a pure-browser code exchange can't work, and the secret must never reach the PWA.
  - Add a small **token-exchange endpoint on the API**: `POST /v1/auth/google/exchange` `{code, code_verifier, redirect_uri}`. It calls Google's token endpoint with `GOOGLE_CLIENT_SECRET` (Render secret) and returns only the `id_token` + expiry.
  - The PWA keeps PKCE + `state`/`nonce` checks via `oidc-client-ts`, or a small custom flow if that library can't use a custom token step.
  - The API keeps verifying Google ID tokens through JWKS:
    - `AUTH_JWKS_URL=https://www.googleapis.com/oauth2/v3/certs`
    - `AUTH_ISSUER=https://accounts.google.com`
    - `AUTH_AUDIENCE=<client id>`
    - require `email_verified=true`
  - Tests use fakes (no network). Rate-limit the endpoint and allow-list `redirect_uri` against `ALLOWED_ORIGINS`.
  - DEPLOY.md lists the exact URIs to add to the Google client: the JavaScript origin `https://<pwa>.onrender.com` and the redirect `https://<pwa>.onrender.com/auth/callback`.
- **Personal mode allow-list:** `ALLOWED_EMAILS` (comma-separated, Render secret). When it's set, any other Google account gets a friendly 403 ("This is a private family app"), and invite acceptance (C27c) also works only for allow-listed emails. Default: unset in dev, **required in production** (the config checker fails without it).
- **Branch:** Render deploys from `redesign/v2` until the owner merges to `main`.
- **Done by the reviewer already, on 2026-10-28:**
  - Neon database `wealth` created in the Frankfurt project (Postgres 18.6).
  - All migrations 0001–0011 applied.
  - The `wealth_api` login has a password; the API URL uses the pooled host. That's safe because `user_session` uses only `SET LOCAL ROLE` and transaction-local `set_config`.
  - Two Neon incompatibilities in 0002 fixed: a non-superuser owner can't name SUPERUSER/REPLICATION in ALTER ROLE, nor REVOKE a role it has no ADMIN on.
  - A smoke test from Denmark (me, asset, plan, foreign household 404, delete account) passed at ~0.4 s per request.
  - DEPLOY.md documents these steps for a fresh install; don't redo them.
- **Render region:** **`frankfurt`** for the static site and the API, next to the database. Personal data stays in the EU (GDPR).
- **CI realism:** the CI Postgres service runs migrations as a **non-superuser** owner role with `CREATEDB CREATEROLE BYPASSRLS` (like Neon's `neondb_owner`), not as `postgres`, so managed-Postgres permission problems fail in CI. Create that role in the workflow's setup step.

### 1. `render.yaml` (Blueprint)
- **Static site `wealth-pwa`:**
  - `rootDir: apps/web`, build `npm ci && npm run build`, publish `dist`
  - SPA rewrite `/* → /index.html`
  - headers: `/sw.js` and `/manifest.webmanifest` `Cache-Control: no-cache`; `/*`: `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy: camera=(), microphone=(), geolocation=()`
  - env (all `sync: false`): `VITE_API_URL`, `VITE_AUTH_MODE=oidc`, `VITE_OIDC_AUTHORITY`, `VITE_OIDC_CLIENT_ID`, `VITE_OIDC_AUDIENCE`
- **Web service `wealth-api`:**
  - `runtime: python`, `plan: free`, `region: frankfurt` (primary users live in Denmark; GDPR: keep personal data in the EU), repo root
  - build `pip install --no-cache-dir ".[api,importer]"`
  - start `uvicorn wealth.api.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips '*'`
  - `healthCheckPath: /healthz`
  - env: `PYTHON_VERSION` (a 3.12.x patch), `ENV=production`, `AUTH_MODE=oidc`, `MAX_UPLOAD_MB=10`, `CDSL_PARSER_BETA=false`; `sync: false` for `DATABASE_URL` (the **wealth_api** login), `AUTH_JWKS_URL`, `AUTH_ISSUER`, `AUTH_AUDIENCE`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `ALLOWED_EMAILS`, `ALLOWED_ORIGINS`, `EMAIL_LINK_SECRET`
  - **No `FOUNDRY_*`, no `OWNER_DATABASE_URL`.**
- **CSP:** the PWA build injects a `<meta http-equiv="Content-Security-Policy">` at build time from `VITE_API_URL` and `VITE_OIDC_AUTHORITY`. Add a small Vite plugin in `apps/web`:
  - `default-src 'self'`; `script-src 'self'`; `style-src 'self'`
  - `connect-src 'self' <api> <oidc>`; `img-src 'self' data:`; `worker-src 'self'`; `manifest-src 'self'`
  - `base-uri 'self'`; `form-action 'self' <oidc>`
  Add a vitest that the built `index.html` contains it.

### 2. GitHub Actions (`.github/workflows/`)
All workflows: `permissions: contents: read`, `timeout-minutes`, `concurrency` groups, Python 3.12, `pip install` with cache, **never echo secrets**. Jobs that need the DB use `OWNER_DATABASE_URL` from secrets.

| File | Trigger | Does |
|---|---|---|
| `ci.yml` (update) | push/PR | Python: ruff + pytest with a **`postgres:18` service container** (so `TEST_DATABASE_URL` DB tests run in CI). PWA: Node 24, `npm ci`, typecheck, lint, test, build, budget. A drift check: `export_openapi` + `gen:api` produce no diff. |
| `db-migrate.yml` | manual (`workflow_dispatch`) | `alembic upgrade head` on `OWNER_DATABASE_URL` |
| `marketdata-daily.yml` | cron `30 13 * * 1-5` (19:00 IST) + manual | `refresh --only master,nse,amfi,fx,gold --prune`; Saturdays also `edgar` when `SEC_USER_AGENT` is set |
| `marketdata-abroad.yml` | cron `0 18 * * 1-5` (after EU close) + manual | `refresh --only eu_etfs,us_stocks` (C29a); the price API key comes from a GitHub secret if the chosen source needs one; skip cleanly when unset |
| `marketdata-monthly.yml` | cron `0 3 2 * *` + manual | `refresh --only amfi_history` (C29b month-end NAVs) |
| `sentinel-daily.yml` | cron `30 0 * * *` (06:00 IST) | `wealth.research.sentinel --days 2` (Foundry secrets) then `wealth.alerts.daily` |
| `research-weekly.yml` | cron `0 0 * * 6` (start) and `0 2-20/2 * * 6` (continue) + manual | `wealth.research.run start/continue` with `--budget-usd ${{ vars.RESEARCH_BUDGET_USD || 5 }}`; the run id is stored/read via the DB (`research_runs` latest for the week), not artifacts |
| `policy-weekly.yml` | cron `30 1 * * 0` | `wealth.policy.run_weekly --week <ISO week>` |
| `alerts-weekly.yml` | cron `0 * * * 0` (hourly on Sundays) | `wealth.alerts.weekly --week <ISO week> --local-hour 8`: sends only to users whose **profile timezone** currently reads 08:xx (Copenhagen users at 08:00 CET/CEST, India users at 08:00 IST). Idempotent via `alerts_log`. Add the `--local-hour` option to the C17b job if missing. |

Secrets referenced (document them; never set them yourself): `OWNER_DATABASE_URL`, `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY`, `FOUNDRY_MODEL_WORKHORSE`, `FOUNDRY_MODEL_REDTEAM`, `SEC_USER_AGENT`, `EMAIL_PROVIDER`, `EMAIL_API_KEY`, `EMAIL_FROM`, `EMAIL_LINK_SECRET`, `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`.

**Missing modules:** if the module a workflow calls doesn't exist yet in this repo (some cards land later), the workflow step must check for the module and skip with a notice instead of failing. Use `python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('wealth.research.run') else 78)"` and handle exit 78 as skip.

### 3. `docs/DEPLOY.md` (≤ 2 pages, exact steps)
1. **Neon:** create a project/database in an **EU region (Frankfurt, `aws-eu-central-1`)**, since users live in Denmark (GDPR data residency). Get the owner connection string. Run `db-migrate`. Then run the one-time SQL to set the `wealth_api` login password, and build the API `DATABASE_URL` from it.
2. **GitHub secrets / vars:** the list above, with where each comes from.
3. **Render:** New → Blueprint → select the repo/branch → fill each `sync: false` value. Set `ALLOWED_ORIGINS` to the static site URL. Note free-tier sleep and the cold-start UX.
4. **Google login:** in the existing OAuth web client add the JavaScript origin `https://<pwa>.onrender.com` and the redirect URI `https://<pwa>.onrender.com/auth/callback`; the PWA env uses `VITE_OIDC_AUTHORITY=https://accounts.google.com` and `VITE_OIDC_CLIENT_ID=<client id>`. Keep `localhost` entries for dev.
5. **Email:** verify a sending domain (SPF/DKIM/DMARC) with the chosen provider; free-tier caps.
6. **VAPID keys:** the generation command (`python -m wealth.alerts.push --generate-vapid`, which prints to stdout; add it if missing).
7. **First-run checklist:**
   - `/healthz?deep=1`
   - log in and complete onboarding
   - upload a CAS
   - `marketdata-daily` manual run
   - `policy-weekly` manual run
   - `alerts-weekly` with `EMAIL_PROVIDER=dry_run` first
8. **Upgrade path** (paid tiers).
9. **GDPR checklist** (EU users):
   - personal data stays in EU-region Render and Neon
   - sign/accept the DPAs of Render, Neon and the email provider
   - Azure AI Foundry (`stockllm`, East US) receives **no personal data** by design (research prompts contain only market data; state the guard test that enforces this)
   - a privacy notice in the PWA covering purpose, lawful basis, retention, processors and rights (access/export, rectification, erasure; the app already has export + delete)
   - email consent is opt-in
   - device storage is functional only (no tracking cookies)
   - breach-contact placeholder
10. **Regulatory note:** personalised investment recommendations to EU residents fall under MiFID II (Finanstilsynet in Denmark). The app ships with an information-only disclaimer; a public launch needs authorisation or a regulated partner.

### 4. `scripts/check_deploy_config.py` + tests
Parses `render.yaml` and every workflow:
- required keys/env names present
- **no value that looks like a secret** (patterns: `postgres://…:…@`, `sk-`, `api-key`, `-----BEGIN`, 32+ char hex/base64 strings in `value:` fields)
- the API service has no `FOUNDRY_`/`OWNER_DATABASE_URL`
- every cron is valid
- every `wealth.*` module referenced either exists or is guarded

Run it in CI.

## Out of scope
Actually creating Render/Neon/GitHub resources or secrets (the user does that following DEPLOY.md).

## Acceptance
1. `.venv\Scripts\python scripts\check_deploy_config.py` → OK; pytest all pass (incl. the checker tests)
2. In `apps/web`: `npm run build` produces `dist/index.html` with the CSP meta; `npm run test` passes
3. `ruff` clean; YAML files parse; `actionlint` if available (optional; report)
4. No secret values anywhere (the checker + a repo-wide grep for the patterns above, reporting only counts)
