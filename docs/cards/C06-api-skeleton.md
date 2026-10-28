# C06: API skeleton: FastAPI app, settings, JWT auth (OIDC/JWKS + dev mode), health, CORS, user bootstrap, profile endpoints, OpenAPI export

**Plan:** `docs/UPGRADE_PLAN.md` §7.1 (free Render web service; no LLM keys on the API), §7.4 (security), §8.7 (auth: any OIDC provider, verified by the API), Appendix D (512 MB, cold starts). Read `AGENTS.md` first. Builds on C05 (`make_engine`, `user_session`, `system_session`, models).

## Goal
A small, secure FastAPI service that verifies the user's login token, maps it to a `users` row, and runs every request inside that user's RLS session. Later cards add endpoints to it.

## Decisions (follow them)
- Package `wealth.api`. App factory `create_app(settings: Settings | None = None) -> FastAPI` in `wealth/api/main.py`; module-level `app = create_app()` for uvicorn.
- Settings via `pydantic-settings` (`wealth/api/settings.py`):
  - `ENV` (`dev`/`test`/`production`, default `dev`)
  - `DATABASE_URL`
  - `AUTH_MODE` (`oidc`/`dev`), `AUTH_JWKS_URL`, `AUTH_ISSUER`, `AUTH_AUDIENCE`, `DEV_AUTH_SECRET`
  - `ALLOWED_ORIGINS` (comma-separated)
  - `MAX_UPLOAD_MB` (default 10)
- **Production guard:** startup fails if `ENV=production` and `AUTH_MODE=dev`, or if `ALLOWED_ORIGINS` contains `*`.
- Auth: PyJWT.
  - `oidc` mode: fetch and cache JWKS (TTL 1 h, refetch once on unknown `kid`), RS256/ES256 only, verify `iss`, `aud`, `exp`, `nbf`, 60 s leeway.
  - `dev` mode: HS256 with `DEV_AUTH_SECRET`.
  - Claims used: `sub` (required), `email` (required), `name` (optional).
- User bootstrap: use C05's `resolve_user(engine, sub, email, name)`, which calls the `app_resolve_user` SECURITY DEFINER function. **The API never uses `system_session` and never needs the owner credential.** `DATABASE_URL` for the API logs in as `wealth_api`. Add a test that `wealth.api` modules don't reference `system_session` (grep the source) and that all integration tests run with a non-owner login role, reusing C05's throwaway `wealth_api_test_*` login fixture. After bootstrap, **every** handler runs inside `user_session(engine, user_id)` via a dependency.
- Errors: RFC 7807 `application/problem+json` (`type`, `title`, `status`, `detail`, `instance`). Never leak stack traces or SQL. 401 for bad/missing token; 403 reserved.
- Security headers on every response: `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, `Cache-Control: no-store` for `/v1/*`.
- **The API must not import `wealth.research`.** Add a test that imports `wealth.api.main` and asserts `"wealth.research"` is not in `sys.modules` and neither is `langgraph`.

## Scope

### 1. Dependencies (`api` extra)
`fastapi>=0.115`, `uvicorn[standard]>=0.32`, `pydantic-settings>=2.6`, `pyjwt[crypto]>=2.10`. Add `httpx>=0.28` to `dev`.

### 2. Endpoints
| Method | Path | Auth | Behaviour |
|---|---|---|---|
| GET | `/healthz` | none | `{"status":"ok","version":...}`; `?deep=1` also runs `SELECT 1` and returns `db: ok/error` (503 on error) |
| GET | `/v1/me` | user | `{id, email, display_name, created_at, profile: {...} or null}` |
| PUT | `/v1/me/profile` | user | Body = C02 `Profile` + `FinancialContext` fields (all optional except `tax_regime`). Upserts `profiles`. Returns the stored profile. |
| PUT | `/v1/me/risk-answers` | user | Body `{answers: {question_id: option_id}}`. Validates against `rules/allocation.yaml` questions (C11), stores `risk_answers`, computes and stores `risk_profile` with C11 `risk_profile()` (use horizon 10 and `emergency_ok` from the request body flag, default true). Returns the `RiskResult` (tolerance, capacity, final, limited_by). |
| GET | `/v1/meta/risk-questions` | user | The six questions and options (ids + text) for the PWA |
| DELETE | `/v1/me` | user | Inside `user_session`: `DELETE FROM users WHERE id = :me`. RLS allows only the caller's row, and FK cascades remove the rest. Returns 204. No audit row is kept (deletion is the point under DPDP). |

### 3. OpenAPI export
`python -m wealth.api.export_openapi` writes `schemas/openapi.json` (sorted keys, 2-space indent). A test fails when the file is stale.

### 4. Tests
- Unit (no DB): settings production guard; JWT verification. Generate an RSA key in the test, serve a JWKS through a monkeypatched fetch, and cover valid / expired / wrong issuer / wrong audience / unknown kid → refetch → still unknown → 401 / HS256 token rejected in oidc mode. Also: problem+json shape, security headers, no `wealth.research`/`langgraph` import, OpenAPI file fresh.
- Integration (`@pytest.mark.db`, uses `TEST_DATABASE_URL`, `alembic upgrade head` once per session) via `TestClient`:
  - first `/v1/me` creates the user; second returns the same id
  - two users never see each other's profile
  - `PUT /v1/me/profile` round-trip with Decimal fields as fixed-point strings
  - risk answers produce the expected band for a fixture answer set
  - `DELETE /v1/me` removes the user's rows (the test may check via an owner connection; the API code may not)
  - `/healthz?deep=1` → db ok

### 5. Local run docs
Add a "Run the API locally" section to `README.md`:
- `$env:AUTH_MODE="dev"`, `$env:DEV_AUTH_SECRET=...`
- `alembic upgrade head`
- `.venv\Scripts\python -m uvicorn wealth.api.main:app --reload`
- a 3-line Python snippet that mints a dev token

## Out of scope
Money Map CRUD (C07), imports (C08/C09), rate limiting, Render config (C18).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,dev]"`
2. `.venv\Scripts\python -m pytest`. All pass; DB tests run (not skipped) with `TEST_DATABASE_URL`.
3. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
4. `.venv\Scripts\python -m wealth.api.export_openapi` leaves `schemas/openapi.json` up to date
5. Peak memory sanity: `python -c "import wealth.api.main"` imports in under 2 seconds. Report the measured time.
