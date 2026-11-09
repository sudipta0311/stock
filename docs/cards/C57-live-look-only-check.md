# C57: a look-only check of the live app

**Source:** the owner, 2026-11-09: "do a full e2e test using the live app and test each and every feature". The owner chose: **their own account, look only**. The test opens every screen of the live app and checks it, but never saves anything. Features that save data get local tests in a later card (C58).

## Why
The local Playwright suite (`e2e/tests`) runs on a local stack with synthetic data. Nothing checks the deployed app with real data: real holdings, real plan steps, real briefs and real data gaps. This card adds a reusable, opt-in live check.

Constraints that shape the design:
- The live app only accepts Google sign-in from allow-listed emails, so the owner signs in once in a browser profile. The suite reuses that profile.
- The PWA keeps the refresh token in `localStorage` (`apps/web/src/auth/token.ts`). Refresh tokens are single-use with reuse detection (`src/wealth/api/auth_sessions.py`), so the suite uses **one browser context, one worker, run serially**.
- Screens show the owner's real money data. **Screenshots, traces, reports and findings must never be written inside the repository.**

## Read
- `AGENTS.md` (UX rules: jargon list, at most 2–3 notes, short copy).
- `e2e/playwright.config.ts`, `e2e/support/app.ts` (`expectFitsScreen`, `writeReport`), `e2e/tests/usability.spec.ts`, `e2e/tests/accessibility.spec.ts`.
- `apps/web/src/app/App.tsx` (routes), `apps/web/src/auth/token.ts`, `apps/web/src/auth/session.ts`, `apps/web/src/money/offline-queue.ts`, `apps/web/src/money/AssetFormPage.tsx` (when a save is queued offline).
- Every API call the PWA makes: `apps/web/src/**/api.ts`, `apps/web/src/pwa/push.ts`, `apps/web/src/inbox/GmailStatementInbox.tsx`, `apps/web/src/settings/*.tsx`.
- `copy/en.yaml` (error strings, the `<Term>` component's terms).

## Scope

### 1. Files and commands
All new code goes in `e2e/live/` plus one config and two npm scripts. Nothing under `e2e/tests` runs against the network.
- `e2e/playwright.live.config.ts`:
  - `testDir: "./live"`, no `globalSetup`;
  - `workers: 1`, `fullyParallel: false`, `retries: 0`;
  - 120 s per test, 60 s navigation timeout (Render's free tier can be slow to wake);
  - reporter `list` only, no HTML report;
  - `trace`, `video` and automatic failure screenshots off;
  - `outputDir` inside `LIVE_OUT_DIR`.
- `e2e/package.json` scripts:
  - `"live:login": "node live/login.mjs"`;
  - `"live": "playwright test --config playwright.live.config.ts"`.
- Environment:
  - `LIVE_PROFILE_DIR` (required): the browser profile holding the owner's sign-in.
  - `LIVE_OUT_DIR` (required): screenshots, findings and Playwright output.
  - `LIVE_PWA_URL` (default `https://wealth-pwa.onrender.com`) and `LIVE_API_URL` (default `https://wealth-api-84le.onrender.com`).
  - Fail fast with a plain message when `LIVE_PROFILE_DIR` or `LIVE_OUT_DIR` is missing, **or resolves inside the repository**.
- `e2e/live/login.mjs`:
  - Starts the Playwright-installed Chromium as a **plain process**: `child_process.spawn` of `chromium.executablePath()`, never `launch`/`launchPersistentContext`. Google refuses sign-in in automation-controlled browsers.
  - Arguments: `--user-data-dir=<LIVE_PROFILE_DIR> --no-first-run --no-default-browser-check --window-size=430,900 <LIVE_PWA_URL>/login`.
  - Prints: "Sign in, wait for Home, then close the window." Waits for the process to exit.
- `e2e/live/README.md`: a short how-to (sign in once, run, where output goes, how to end the session from Settings → signed-in devices).

### 2. The browser fixture
- A worker-scoped fixture opens `chromium.launchPersistentContext(LIVE_PROFILE_DIR, …)`:
  - `channel: "chromium"`: the full Chromium binary, the same one the sign-in used, not the headless shell;
  - headless by default, `LIVE_HEADED=1` for headed;
  - viewport 360×740, `isMobile`, `hasTouch`, `deviceScaleFactor: 1`;
  - `serviceWorkers: "block"`.
- When the profile is locked (the sign-in window is still open), fail with "Close the sign-in window first."
- When the first screen redirects to `/login`, stop the whole run with "Not signed in: run npm run live:login first."
- Before the tour, warm the API with `GET <LIVE_API_URL>/healthz` (retry up to 90 s).

### 3. The write guard (the most important part)
A pure, exported function decides every request: `decideRequest({ method, url, isNavigation, frameIsMain })` → `allow` | `block`. The fixture applies it with `context.route("**/*", …)`.
- **Allow:**
  - `GET`, `HEAD` and `OPTIONS` anywhere, except a main-frame navigation to an origin other than `LIVE_PWA_URL`;
  - `POST <LIVE_API_URL>/v1/auth/refresh`, which keeps the session alive.
- **Block everything else.** This includes every `POST`, `PUT`, `PATCH` and `DELETE` to the API (logout too), any non-GET to any other origin, and main-frame navigations away from the PWA (Google, Gmail connect, external links).
- **Answering a blocked request:**
  - a blocked API write gets `route.fulfill` with status 403 and `{"detail":"blocked by live look-only check"}`. Never `route.abort`: a network error is what the offline queue reacts to;
  - a blocked navigation gets `route.abort("blockedbyclient")`.
- **Recording:** every blocked request is a `fail` finding, recording the method and the path template with ids replaced by `:id`. Never record bodies.
- Do not allow-list any other `POST`, even one that looks like a preview. If a screen needs one to render, record a `note` and move on.
- **At the end of the run:**
  - read the IndexedDB database `wealth-money-map-offline` in the PWA origin. Any queued write is a `fail` finding;
  - clear it, so a later plain launch of the profile can never replay it.
- **Tests:** `e2e/tests/live-guard.spec.ts` runs in CI without a browser page or network. It tests `decideRequest` (each allowed and blocked case above) and the "inside the repository" path check. It imports from `e2e/live/`.

### 4. The tour: every screen, look only
Never type into a field, submit a form, upload a file, or press a button that saves, confirms, sends, deletes, connects, buys, marks done, dismisses or signs out. **Opening something and closing it again is fine:** expanders, "Why?", "Learn more", details, tabs, menus, and dialogs closed with Cancel or Escape. Client-only view switches (currency view, household member filter) are fine. Restore them to their first value at the end.

Discover lists from the screen, then cross-check against the matching `GET` responses the page made. Do not hard-code the owner's holdings.
1. **Home `/`:** every expander on the page; every plan step's details.
2. **Bottom navigation:** each tab opens its screen.
3. **Plan `/plan`:** every step and its details; the held-share review rows, if present.
4. **Money `/money`:**
   - totals and every group;
   - each currency view and member filter the screen offers;
   - **every** asset, holding and loan opens its detail page (`/money/asset/:id`, `/money/holding/:id`, `/money/loan/:id`);
   - on each detail page, open every collapsed section ("Why this suggestion?", fund check, details).
5. **Add forms:** every type the add menu offers opens `/money/add/:type`. Check that it renders, then leave without typing.
6. **Import `/import`:** the page; every past import listed opens `/import/:jobId`.
7. **Ideas:** `/ideas` and `/ideas/world`; expand every idea card.
8. **Inbox `/inbox`:**
   - every listed weekly brief opens `/inbox/brief/:week`;
   - the statement inbox items list.
9. **Settings `/settings`:**
   - open every panel: investing, residency, household sharing, members, signed-in devices, notifications/email, SIP fine-tune, delete account;
   - open dialogs and close them with Cancel;
   - `/settings/gmail`.
10. **Help `/help`:** expand every topic.
11. **Account menu:** open and close it. Open the sign-out confirmation, then Cancel.
12. **Guided tour:** read `apps/web/src/tour/`.
    - If stepping through the tour saves nothing to the server, step through to the end.
    - Otherwise open the first step only and close it.
13. **Edge routes, each must show a calm screen, not an error:**
    - `/does-not-exist`;
    - `/login` while signed in;
    - `/join` without an invite;
    - `/onboarding` for a finished profile.

### 5. Checks on every screen and state
A shared `checkScreen(page, name)` runs after each screen settles, and again after each expander or dialog opens.
- **`fail` findings:**
  - an uncaught page error or a `console.error`;
  - an API response with status ≥ 500, or a 4xx on a `GET` (the guard's own 403s are already recorded);
  - an app error message visible: derive the list from `copy/en.yaml` strings that report a failure ("could not", "try again", error keys). Exclude field-validation hints;
  - broken text visible:
    - `NaN`, `undefined`, `null` as a word, `Infinity`, `[object Object]`;
    - an unreplaced `{placeholder}`;
    - a raw copy key such as `holding_review.checks.price_history`;
  - horizontal overflow at 360 px (same rule as `expectFitsScreen`);
  - a list item whose detail page fails to load;
  - any blocked request (§3);
  - two values that must be equal by design but differ on screen. Read the code to find these, for example a total shown on Home and on Money. Cross-check counts too: a holding or asset in the `GET` response that the screen never shows.
- **`note` findings:**
  - a jargon word from `AGENTS.md` visible outside the "What's this?" term component (whole word, case-insensitive; "a lot" is not jargon);
  - more than 3 notes or warnings on one screen (find the highlight/warning component in the code);
  - a sentence over 20 words;
  - axe-core `serious` or `critical` violations (rule id and count);
  - an API response slower than 5 s after warm-up, or a screen slower than 8 s to settle.

### 6. Output (all inside `LIVE_OUT_DIR`, never the repo)
- `screens/NNN-<route-slug>.png`: a full-page screenshot of every screen and every opened state. Name files by order and route, not by holding name.
- `findings.json`: `[{severity, screen, check, detail}]`. `detail` may quote the offending on-screen text (at most 120 characters).
- `findings.md`: findings grouped by severity, then screen. Then a coverage table: every screen visited, how many expanders and dialogs were opened, and how many list items were followed.
- **Console output:** only counts, check names and route templates. Never on-screen text, amounts, names or response bodies.
- The run collects every finding first. The final test then fails if any `fail` finding exists, with a summary of the counts.

## Tests
- `e2e/tests/live-guard.spec.ts` as described in §3.
- Do not run `npm run live` or `npm run live:login`: they need the owner's sign-in and the network. The reviewer runs them.

## Out of scope
- Any change to the PWA or the API. If you find a bug while reading code, list it under "Decisions / open issues" and do not fix it.
- Testing features that save data: that is C58, on the local stack.
- CI: the live check never runs in CI.

## Acceptance
In `e2e`:
- `npx playwright test --list` (the default config must not list anything from `live/`).
- `npx playwright test --config playwright.live.config.ts --list`.
- `node --check live/login.mjs`.

Also run `npx prettier --check` on the new files, if the repo's Prettier setup covers `e2e`; otherwise match the existing e2e style.
