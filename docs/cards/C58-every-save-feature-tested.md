# C58: every feature that saves data has a local end-to-end test

**Source:** the owner, 2026-11-09: "do a full e2e test ... test each and every feature ... if any bug found get it fixed". The owner chose a look-only check of the live app (C57). Everything that **saves** data is therefore tested here, on the local stack with synthetic data.

## Why
`e2e/tests` has 22 tests. They cover:
- sign-up and onboarding;
- fixed deposit, Danish savings and quick add;
- CAS and broker imports, and Inbox upload;
- the Denmark "I bought it" flow;
- holding reviews, household partner and viewer access;
- account deletion, isolation, accessibility and styles.

The API has 44 save endpoints. Many have a screen that calls them but no end-to-end test: editing and deleting things, goals, loans and loan events, members, settings, plan step status, recorded buys, highlights, the checklist, invites, the Gmail statement inbox, import corrections and the offline queue.

## Read
- `AGENTS.md`, `e2e/playwright.config.ts`, `e2e/setup/start-stack.mjs`, `e2e/support/app.ts`, and every spec in `e2e/tests` (learn the helpers and seeds before adding new ones).
- The API's save routes: `grep -n "@router.\(post\|put\|patch\|delete\)" src/wealth/api/*.py`.
- The PWA code that calls them: `apps/web/src/**/api.ts` and the screens that use those functions.

## Scope

### 1. A coverage table
Create `e2e/COVERAGE.md`. Give it one row per save endpoint (method + path). Each row lists:
- the screen and the user action that triggers it ("Settings → Members → Add a person");
- the test that covers it: an existing test or a new one;
- or why it has no end-to-end test, for example "no screen calls it" or "needs a real push service".

Keep the table short and factual. Future cards update it.

### 2. New tests for every uncovered action a screen offers
Confirm each item against the code first. Skip anything the PWA does not offer, and say so in the table. Group related steps into one test where that saves sign-ins.

Every new test:
- runs at the narrow 360×740 project;
- does the action **through the screen**, as a person would;
- then **reloads** and checks that the change persisted;
- checks that the change shows where it matters, for example the Money total or a plan step.

Expected candidates:
1. **Money:**
   - edit an asset's value and see the new value after reload;
   - delete an asset;
   - add one asset of each form family the add menu offers that no test adds yet.
2. **Loans:**
   - add a loan;
   - record a prepayment and a rate change;
   - see the prepay preview;
   - delete an event; the loan page updates.
3. **Offline queue:** save an asset edit while offline (`context.setOffline(true)`). The offline note shows. Go back online, and the save replays and persists.
4. **Goals:** create, edit and delete a goal wherever the screens offer it.
5. **Members:**
   - add a family member and edit their details;
   - give an asset to them;
   - the member filter shows it.
6. **Settings:**
   - change the monthly amount and see Home and Plan update;
   - change a consent;
   - save a SIP fine-tune;
   - "sign out everywhere" returns to sign-in.
7. **Plan steps:**
   - mark a step done, skip one, and undo;
   - dismiss a highlight, and it stays dismissed after reload;
   - edit a recorded buy, then delete it.
8. **Checklist:** tick an item; it stays ticked after reload.
9. **Household:**
   - create an invite and revoke it;
   - a second synthetic identity joins with an invite, then leaves;
   - the owner removes a member's access.
10. **Gmail statement inbox** (`/settings/gmail`, `/inbox`):
    - create a forwarding key and delete it;
    - delete saved statement passwords;
    - process an inbox item and delete one (synthetic files only, as the existing Inbox test does).
11. **Imports:**
    - correct a row on the review screen;
    - discard a whole import.
12. **Push notifications:** test only what needs no push service. The setting shows the right state when notifications are denied (`context.grantPermissions` / no permission). No real subscribe.
13. **Email unsubscribe:** if a seed helper can build a valid signed link for the local stack, open it and check that the email setting turns off. Otherwise list it as not covered, with the reason.

### 3. Bugs found
When a new test exposes a real bug, write the test for the correct behaviour and fix the bug in the same card. Keep each fix small and in the code that owns it. List every bug under **"Bugs found and fixed"** in the result file: what broke, the fix, and the test that covers it.

A fix that needs a migration or a design decision goes under "Decisions / open issues" instead. Its test gets `test.fixme` with a one-line reason.

### 4. Rules for the new tests
- Synthetic data only, through the existing helpers (`startReadyAccount`, the `seed*` helpers, synthetic CAS and broker files). Add new seed helpers next to the existing ones.
- No network beyond the local stack (`guardLocalNetwork`).
- Keep each new test under about 45 s locally. The whole suite must stay inside the config's 20-minute global timeout with 2 workers.
- Assert what the person sees (text, roles, values after reload), not implementation details.
- Every new test also checks `expectFitsScreen` on the screens it visits, and fails on `console.error` and page errors.

## Tests
New specs in `e2e/tests/`, plus unit tests for any code you fix (Python in `tests/`, PWA in `apps/web/tests/`).

You cannot run the end-to-end suite in your sandbox. Write the tests carefully against the code and `npx playwright test --list`. The reviewer runs the full suite and sends back the failures.

## Out of scope
- The live app (C57).
- Visual redesigns and copy changes, except where a bug fix needs them.
- New features. Changes to money, plan or tax rules.

## Acceptance
- `.venv\Scripts\python -m pytest`, `ruff check src tests`, `ruff format --check src tests` (if Python changed).

Inside `apps/web` (if the PWA changed):
- `npm run gen:api`, `npm run gen:copy`
- `npm run typecheck`, `lint`, `format:check`
- the full `npm test -- --run`
- `npm run build`, `npm run budget`

In `e2e`: `npx playwright test --list`.
