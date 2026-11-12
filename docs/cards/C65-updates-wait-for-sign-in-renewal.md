# C65: a new version waits for sign-in renewal before it reloads

**Source:** the reviewer, 2026-11-12, while running the live look-only check. The check was signed out twice in a row: a page load was cut short while the sign-in was being renewed.

**Cause:** every app load renews the sign-in. `apps/web/src/auth/AuthContext.tsx` calls `refreshAuthSession()`, which rotates the refresh token. If the page reloads while that request is in flight:
- the server has already rotated the token, but the new one is never stored;
- the next load presents the old token, the server answers 409 (or, after 30 seconds, treats it as reuse and revokes the session);
- `apps/web/src/auth/session.ts` then clears the session, and the person must sign in again.

C64 now reloads the app by itself within 15 seconds of opening, which is exactly when that renewal runs. The toast's "Tap to refresh" can hit the same window.

## Read
- `AGENTS.md` (UX rules).
- `apps/web/src/app/PwaLifecycle.tsx` and `apps/web/tests/pwa/PwaLifecycle.test.tsx`.
- `apps/web/src/auth/session.ts` (`refreshAuthSession`, `refreshInFlight`, `waitForAuthBootstrap`) and `apps/web/src/auth/AuthContext.tsx` (start-up renewal).
- `apps/web/tests/auth/` for how session tests mock `fetch`.

## Scope
1. **A way to wait for renewal.** Export a small function from `apps/web/src/auth/session.ts`, for example `waitForSignInRenewal(): Promise<void>`. It resolves once:
   - the start-up sign-in has finished, if it has started; and
   - no sign-in renewal is in flight. If a new one starts while waiting, wait for that one too.

   It never rejects: a failed renewal also counts as finished.
2. **Automatic update waits, then checks again.** In `PwaLifecycle.tsx`, when a new version is waiting:
   - first wait for `waitForSignInRenewal()`;
   - then run the existing safe-moment check, using the current time;
   - if the check still passes, call `updateServiceWorker(true)`; otherwise show the existing toast.

   A slow renewal (for example while the free API wakes up) therefore leads to the toast, not a late surprise reload.
3. **The toast tap waits too.** "Tap to refresh" waits for `waitForSignInRenewal()` before it calls `updateServiceWorker(true)`. Show nothing extra while it waits.
4. **Nothing else changes:** no new copy, no service worker configuration, no server changes. The C64 rules stay the same: 15 seconds, nothing typed, no dialog, no queued offline save.

## Tests (Vitest; no network)
- **`session.ts`:**
  - `waitForSignInRenewal()` resolves at once when nothing is in flight;
  - it stays pending while a renewal's `fetch` is unresolved, and resolves after it settles (success, 409 and network failure);
  - a renewal that starts while it is waiting is also waited for.
- **`PwaLifecycle`** (mock `virtual:pwa-register/react` as the existing test does):
  - a waiting version plus a renewal in flight → `updateServiceWorker` is not called until the renewal settles, then it is called (still within 15 seconds, nothing typed);
  - the renewal settles after the 15-second window → the toast, not an automatic update;
  - tapping "Tap to refresh" while a renewal is in flight → `updateServiceWorker(true)` is called only after it settles;
  - all existing C64 tests still pass.

## Out of scope
- Server token rotation and the 30-second reuse grace (`src/wealth/api/auth_sessions.py`).
- Caching rules and `vite.config.ts`.
- The live look-only check (`e2e/live/`). The reviewer already changed it to wait for the app to settle.

## Acceptance
In `apps/web`, run each of these and make it pass:
- `npm run gen:api`
- `npm run gen:copy`
- `npm run typecheck`
- `npm run lint`
- `npm run format:check`
- `npm run build`
- the full `npm test`
- `npm run budget`
