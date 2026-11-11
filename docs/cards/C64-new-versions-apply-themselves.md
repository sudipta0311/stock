# C64: a new version applies itself at a safe moment

**Source:** the owner, 2026-11-11, after C63 was live and verified on the owner's account: "still not seeing buying price... also when I am trying to expand the stock it is also expanding the MF view". Both are fixed in the deployed build.

**Cause:** the owner's phone was still running the previous version. `apps/web/src/app/PwaLifecycle.tsx` waits for a tap on the "New version is ready" toast. If the toast is missed, an installed app on a phone can keep the old version for days, because it is rarely fully closed. This has happened several times.

## Read
- `AGENTS.md` (UX rules).
- `apps/web/src/app/PwaLifecycle.tsx` and `apps/web/tests/pwa/PwaLifecycle.test.tsx`.
- `apps/web/tests/pwa-register-stub.ts`.
- `apps/web/src/money/offline-queue.ts` (`listQueuedAssetWrites`).

## Scope
1. **Apply a waiting version by itself when that is safe.** When a new version is waiting, call `updateServiceWorker(true)` without a tap, but only if **all** of these are true:
   - **Just opened or returned:** the app became visible (page load, or `visibilitychange` to visible) within the last 15 seconds.
   - **Nothing typed:** since this page loaded, the person has not typed into or changed any field (no `input` or `change` events on the document).
   - **No dialog is open.**
   - **No saves waiting:** the offline save queue is empty (`listQueuedAssetWrites()` returns none).
2. **Otherwise nothing changes.** The current toast stays, and so does the visibility-triggered `registration.update()`.
3. **No new copy.** Nothing is shown when the update applies by itself; the page simply reloads into the new version.
4. **Small and pure.** Put the "safe moment" decision in a small pure function with its inputs passed in, so it is easy to test.

## Tests (Vitest; mock `virtual:pwa-register/react` as the existing test does)
- **Applies by itself:**
  - a waiting version found right after load, with no input → `updateServiceWorker(true)` is called and no toast is shown;
  - after returning from the background (visible within 15 s), with no input → it applies.
- **Toast instead:**
  - the person typed in a field → a toast, not an automatic update;
  - a dialog is open → a toast;
  - a queued offline save → a toast;
  - the update arrives more than 15 seconds after the app became visible → a toast.
- **No network.**

## Out of scope
- Service worker configuration in `vite.config.ts`.
- Caching rules.
- Server changes.

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
