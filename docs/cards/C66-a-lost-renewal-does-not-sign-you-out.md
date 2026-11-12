# C66: a lost sign-in renewal does not sign you out

**Source:** the reviewer, 2026-11-12, with C65. The owner uses the app on mobile data. A renewal response lost on a weak connection, or cut off by a reload, currently signs the person out.

**Cause:** `POST /v1/auth/refresh` in `src/wealth/api/auth_sessions.py` rotates the refresh token. When the client never receives the response, it presents the previous token again:
- within `REUSE_GRACE` (30 seconds), the server answers **409**, and the PWA clears the session because it holds no newer token;
- after the grace window, the server treats it as reuse and revokes the session.

Either way the person must sign in again. Auth providers solve this with a short "reuse interval": inside it, the previous token is accepted again and a fresh pair is issued.

## Read
- `AGENTS.md`.
- `src/wealth/api/auth_sessions.py` (`refresh_session`, `REUSE_GRACE`).
- `tests/db/test_auth_sessions.py`, especially `test_reusing_rotated_refresh_token_revokes_and_audits_session`.

## Scope
1. **Re-issue inside the grace window.** In `refresh_session`, change the case where the previous token is presented, the session is not revoked and `now - record.last_used_at <= REUSE_GRACE`. Instead of 409, issue a fresh pair:
   - a new refresh token: `record.token_hash` becomes its hash;
   - keep `record.previous_hash` as it is, so a second lost response inside the same window also recovers;
   - **do not** move `record.last_used_at` or `record.expires_at`. The window is measured from the original rotation, so repeating the old token cannot stretch it;
   - return the normal `AuthSessionPair` (new access token, new refresh token, same `session_expires_at` rules) with `Cache-Control: no-store`.

   The token from the lost response then no longer works: presenting it gives 401 without revoking the session, because it is neither the current nor the previous token.
2. **Everything else stays as it is:**
   - previous token after the window → revoke the session and write the `auth_refresh_reuse_detected` audit row;
   - unknown, expired or revoked token → 401;
   - email no longer allowed → 403;
   - rate limiting is unchanged.
3. **Keep 409 documented** in the route's `responses`, because older PWA builds still handle it. Update the comment above `REUSE_GRACE` to describe the new behaviour.
4. **Same checks as a normal renewal:** the re-issue path also checks the allowed email and the absolute expiry, like a normal renewal.

## Tests (`@pytest.mark.db`, skip without `TEST_DATABASE_URL`; no network)
- **Lost response recovered:**
  - log in, renew (A→B), then present A again inside the window → 200 with a new pair C;
  - C renews normally;
  - B now gets 401, and the session is **not** revoked (C still works).
- **Window not stretched:** two re-issues inside the window both succeed. After you move `last_used_at` back past the window, presenting A revokes the session and writes exactly one `auth_refresh_reuse_detected` audit row (adapt the existing test).
- **Unchanged:** after revocation, the access token and every refresh token give 401.
- **Edge:** a re-issue for a user whose email is no longer allowed → 403, and the session is revoked.

## Out of scope
- PWA changes (C65 covers the client side).
- Migrations: the existing columns are enough.
- Access token lifetime and session lengths.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
