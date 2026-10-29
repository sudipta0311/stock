# C32: Stay signed in: app sessions with rotating refresh tokens

**Found in production by the reviewer:** after Google sign-in, the app signs the person out after one hour, and whenever the tab or the installed app is reopened.
- The bearer token is Google's ID token, which expires after 60 minutes.
- `automaticSilentRenew` can't work with Google: its authorize page can't be framed, and our token endpoint only accepts `authorization_code`.
- The OIDC user store is `sessionStorage`, which is per tab and gone when the app is closed.

**Plan:** `docs/UPGRADE_PLAN.md` security sections; C18 (Google exchange, `ALLOWED_EMAILS`), C27 (households). Read `AGENTS.md` first.

## Goal
Sign in with Google once per device, then stay signed in for up to 30 days of use (90 days at most), the way banking and email apps behave. Signing out, removing a family member from `ALLOWED_EMAILS`, or "Sign out everywhere" takes effect immediately on the next refresh. Code never stores Google refresh tokens.

## Design (decided; follow it)
- **Google is used only to prove identity, once, at sign-in.** `POST /v1/auth/google/exchange` already verifies the Google ID token (`email_verified`, `ALLOWED_EMAILS`, issuer, audience). It now also creates an **app session** and returns:
  - `access_token`: an app JWT signed HS256 with the env secret `APP_SESSION_SECRET` (≥ 32 bytes; add it to `.env.example`, `render.yaml` as `sync: false`, and `scripts/check_deploy_config.py`). It is valid for 30 minutes, has `iss = "wealth-api"` and `aud = "wealth-pwa"`, and carries `sub` (the existing `auth_subject`), `email`, `sid` (the session id) and `typ = "access"`.
  - `refresh_token`: 32 random bytes, base64url, returned once. Only its SHA-256 is stored.
  - `expires_in`, and the session's absolute expiry.
- **`POST /v1/auth/refresh`** with `{refresh_token}`:
  - looks up the hash
  - checks it is not expired or revoked
  - re-checks `ALLOWED_EMAILS` for the session's email
  - **rotates**: issues a new refresh token and marks the old one used
  - extends the sliding expiry (30 days from now, capped at 90 days from session start)
  - returns a new pair
  - **Reuse detection:** presenting an already-rotated token revokes the whole session (token theft), returns 401 and writes an audit row.
- **`POST /v1/auth/logout`** (authenticated): revokes the current session. **`POST /v1/auth/logout-all`**: revokes every session of this login. **`GET /v1/auth/sessions`**: lists this login's sessions (device label from the User-Agent family, created, last used, current yes/no).
- **Auth dependency (`wealth.api.auth`):**
  - Accept app access tokens: verify the signature, `iss`, `aud`, `typ`, expiry, and that the session is not revoked. Revocation is checked against the DB at most once per request, using the same `user_session` connection; keep it cheap.
  - Keep accepting Google ID tokens until the PWA no longer sends them, so a deploy never locks anyone out.
  - Dev mode is unchanged.
- **Table `auth_sessions`.** Next free migration number, revising the current head.
  - Columns:
    - `id`, `user_id` (the login, not the household)
    - `token_hash` (unique)
    - `previous_hash` (for reuse detection)
    - `created_at`, `last_used_at`, `expires_at`, `absolute_expires_at`, `revoked_at`
    - `device_label` (≤ 60 chars)
  - It is a **per-login** record: RLS on `app.user_id`, like `push_subscriptions`, not the household pattern.
  - The refresh endpoint runs before a user context exists, so do the lookup through a narrowly scoped `SECURITY DEFINER` function that takes only a token hash and returns only what the endpoint needs. It gets `REVOKE ALL FROM PUBLIC` and `GRANT EXECUTE` to the API role. No new table grants to `wealth_api`.
  - Deleting a user cascades.
  - Sessions are included in `DELETE /v1/me` cleanup.
- **Rate limits:** the refresh and exchange endpoints share the existing per-IP limiter (fix the C18 follow-up while here: use the right-most `X-Forwarded-For` hop that Render adds, not the first).
- **PWA:**
  - Keep the access token in memory only.
  - Keep the refresh token in `localStorage` under one key, wrapped in try/catch; when storage is unavailable, fall back to memory, so the person stays signed in for this tab only.
  - On app start: if a refresh token exists, call `/v1/auth/refresh` before the first API call and show the splash.
  - Refresh proactively 2 minutes before expiry, and once on any 401 (single-flight, so parallel requests wait for one refresh). Then retry the original request once.
  - Remove `automaticSilentRenew` and the silent callback route. Switch the oidc-client-ts user store to `sessionStorage`, used only for the short sign-in redirect state.
  - Sign out (C31 menu) calls `/v1/auth/logout`, then clears tokens.
  - **Settings → Privacy** gets a "Signed-in devices" list (from `/v1/auth/sessions`) and a "Sign out on all devices" button with a confirm.
- **Copy and Help:**
  - Plain words: "You stay signed in on this phone for 30 days."
  - Update the C31 help article "Keep your account safe": staying signed in, signing out everywhere, removing a lost phone.

## Tests (no network)
- Exchange returns an app pair; the access token works on `/v1/me`; an expired access token gives 401; refresh gives a new pair and the old refresh token is now invalid.
- Reuse of a rotated refresh token revokes the session: both the new and the old token stop working, and an audit row is written.
- A sliding expiry is capped at the absolute expiry; an expired session gives 401.
- Removing an email from `ALLOWED_EMAILS` makes the next refresh fail with 403.
- Logout and logout-all revoke; the sessions list shows only this login's sessions (RLS: another login's sessions are invisible).
- The `SECURITY DEFINER` function can't be used to list or modify other rows (DB test).
- An app token with the wrong `aud`, `iss` or `typ`, or a bad signature, is rejected; a Google ID token is still accepted.
- **PWA:**
  - the app start refreshes and shows the screen without a sign-in prompt
  - a 401 triggers one refresh and a retry
  - parallel 401s trigger a single refresh
  - logout clears storage and calls the API
  - blocked storage falls back to memory
- The deploy checker requires `APP_SESSION_SECRET` with `sync: false`.

## Acceptance
1. Python: full `pytest` and `ruff` (`--no-cache`) pass; alembic upgrade/downgrade/upgrade on the test DB; OpenAPI refreshed.
2. PWA: `gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, full `npm test`, `build`, `budget` (≤ 170 KB initial).
3. `scripts/check_deploy_config.py` passes. The result file states the new env var the reviewer must set on Render.
