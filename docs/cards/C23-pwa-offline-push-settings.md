# C23: PWA: install prompt, web push subscription, offline/cold-start polish, Settings (profile, tax regime, household, investing prefs, notifications, privacy), data export & account deletion, passkey-style app lock

**Plan:** `docs/UPGRADE_PLAN.md` §8.1 (install after the first value moment; iOS instructions; web push only after install on iOS; offline read-only cache; cold starts), §7.4 (PWA device security: auto-lock, shared-device mode, clear on logout; DPDP export/delete), §8.2 (Settings behind the avatar), §9 (push consent). Read `AGENTS.md` first. Builds on C19–C22 and the API from C06 (`DELETE /v1/me`), C17a (`/v1/me/investing`), C17b (`/v1/me/consents`, `/v1/push/subscriptions`, `/v1/me/checklist`): **feature-detect** anything missing.

## Goal
The app feels native on Android and iPhone. It installs at the right moment, sends useful pushes only with consent, works offline for reading, survives Render cold starts gracefully, and gives users full control over their data.

## Scope

### 0a. Fix 3 stale PWA tests (found by the reviewer's full `npm test` after C17b/C26a; do first)
- `tests/features/HoldingPage.test.tsx` expects the old trigger label "Investment case". The labels now come from `copy/en.yaml` (e.g. thesis_break → "Business results weakening").
- `tests/features/HomePage.test.tsx` expects "Build your emergency fund · fixed deposit". The instrument labels changed (e.g. sweep_fd → "Sweep-in FD").
- `tests/features/InboxPage.test.tsx` "coming soon when inbox paths are missing" now fails because C17b added `/v1/inbox`. Mock the feature-support module in that test instead of relying on the real generated types.
- Make the tests derive expected strings from the generated copy JSON (no hard-coded label text), so copy edits can't break them again. **Run the FULL `npm test`, not targeted files**, and report the count.

### 0. Follow-ups from the C17b review (do first)
- C17b added sell-side weekly action kinds (`SELL_EXIT`, `SELL_TRIM`, `REVIEW_EXIT`, `SWAP`) and a `labels` section in the repo-root `copy/en.yaml` (instrument categories, action kinds, exit actions, trigger keys and states).
- Make `apps/web/src/lib/labels.ts` read every label from the generated `src/copy/en.json` (single source shared with emails) instead of hard-coding strings.
- Home's "This week" card must render sell-side actions correctly: quantity, "estimated tax ₹X", and a "Why" sheet that shows the triggered exit-card rows.
- Add tests: every action kind / instrument category / exit action / trigger key in the API's OpenAPI enums or the copy file has a label; a sell action renders its quantity and tax.

### 0b. Simplicity (AGENTS.md **UX rules**; the C28 simplicity pass builds on this)
- **Settings has 4 top-level groups:** **Family**, **Notifications**, **Privacy**, **About**. Profile, tax details, investing preferences and the risk questionnaire sit inside a collapsed **"Fine-tune your plan"** section, closed by default. Every field there is optional and has a sensible default.
- **Plain labels in Settings:**
  - "satellite %" becomes "Money for your own stock picks (0–25%)"
  - "LRS broker" becomes "I can invest abroad through a US broker"
  - "prefer no paperwork" becomes "Keep it simple: fewer accounts and forms"
  - "slab" becomes "Income tax bracket"
- Install, push and lock prompts: one short sentence + one button. Never show more than one prompt at a time.
- New copy goes in `copy/en.yaml`; sentences ≤ 20 words.

### 1. Install prompt
- Capture `beforeinstallprompt` (Android/Chrome). Show a custom, dismissible card **after the first value moment**: the user has ≥ 3 assets or has completed an import. Never on first load.
- On iOS Safari (no prompt API), show a short "Share → Add to Home Screen" sheet with an illustration, once per 30 days. Detect standalone mode and don't show it when already installed.

### 2. Web push
- Settings → Notifications: weekly email / reminders / push / urgent alerts toggles, wired to `/v1/me/consents` (or local-only with a "sync coming soon" note if missing).
- Turning on push: request permission **only on the user's tap**, subscribe with the VAPID public key from `VITE_VAPID_PUBLIC_KEY`, then `POST /v1/push/subscriptions`.
- iOS: explain that push works only after installing to the Home Screen.
- The service worker shows notifications with a title/body/url payload; a click focuses or opens the deep link (`/` actions, `/inbox/...`).
- Unsubscribe deletes server-side and locally.

### 3. Offline and cold start
- An offline banner when `navigator.onLine` is false. Read-only screens use the cached snapshots (C19 SW) with an "as of <time>" label.
- Writes queue (C20 queue) and show a pending badge.
- The cold-start chip from C19 shows across all data screens; long waits (> 20 s) show "Free server is waking up, this happens after 15 minutes of no use".

### 4. Settings (`/settings`)
- **Profile:** name, tax regime (old/new), slab, city tier, timezone, language (English only for now; the list comes from i18n config).
- **Household:** members list/add/rename/remove (with the 409 handling from the API).
- **Investing preferences** (`/v1/me/investing`): monthly SIP capacity, existing SIPs editor, "I have money to invest this week", LRS broker yes/no, prefer no paperwork, satellite % slider (0–25, explained), birth year.
- **Risk profile:** retake the questionnaire.
- **Privacy:**
  - "Shared device mode" (disables offline caching; clears caches now)
  - "Lock app after 5 minutes" (see §5)
  - "Download my data": calls the endpoints the PWA already uses (`/v1/me`, members, accounts, assets, goals, imports list, actions, inbox) and saves a JSON file client-side. No new backend export needed; label it as a JSON export.
  - **"Delete my account"**: type DELETE to confirm → `DELETE /v1/me` → clear all local data/caches/IndexedDB/SW snapshot cache → sign out.
    - **Shared households (C27c API, feature-detect `/v1/households`):** if the user owns a household that others can open (`GET /v1/households/access` shows active people), the confirm sheet says so first: "This also deletes the family plan you share with Anita. They will lose access." For a partner it says: "Your login is deleted. The family plan stays with <owner>."
    - "Download my data" exports the household currently open, labelled with the owner's name.
- **About:** version, disclaimer, links.

### 5. App lock (lightweight)
- An optional lock screen after N minutes in the background, unlocked with WebAuthn **platform authenticator** (fingerprint/face/PIN) when available (`navigator.credentials.create/get` with a locally stored credential id; no server verification. This is a device convenience lock, not authentication, and the copy says so).
- Fallback: re-authenticate via OIDC.
- Locking hides balances (blurred) until unlocked.

### 6. Calm defaults
Setting "Show daily changes" (off by default) controls whether Home shows week/day change numbers.

### 7. Tests (vitest + MSW; mock browser APIs)
- Install card appears only after the value moment; the iOS sheet logic.
- Push: permission requested only on tap; subscription payload posted; unsubscribe path; SW notification-click routing (pure function).
- Offline banner and pending badge.
- The long cold-start message after 20 s (fake timers).
- Settings forms save with the right API calls.
- Shared-device mode clears caches.
- Data export file contents (keys present, no tokens inside).
- **Delete account clears every storage area** (assert `localStorage`, `sessionStorage`, IndexedDB and cache-storage mocks are emptied).
- App-lock timer and unlock flows (WebAuthn mocked; graceful fallback when unsupported).
- Axe checks on the Settings pages.

## Out of scope
Backend changes (list any truly needed ones in the result file).

## Acceptance (inside `apps/web`, plus the repo-root Python suite)
1. `npm run typecheck`, `lint`, `test`, `build`, `budget` → pass (Settings is code-split)
2. `.venv\Scripts\python -m pytest` still green
3. The result file has a screen-by-screen note
