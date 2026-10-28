# C20: PWA: onboarding + Money Map (checklist tiles, schema-driven mini-forms, quick-add, autosave, asset detail, net worth summary)

**Plan:** `docs/UPGRADE_PLAN.md` §3.1 (setup flow, UX rules: autosave, completeness meter, Indian number input, month-year pickers, "rough is fine", quick-add, household), §3.2 (asset catalogue), §3.5 (what the Money Map shows), §8.3 (onboarding), §8.4 wireframes ("What else do you have?", "Add a Fixed Deposit", "Add an LIC policy"). Read `AGENTS.md` first. Builds on C19 (shell, design system, API client, money helpers) and the API from C06/C07/C08b (`/v1/me`, `/v1/me/profile`, `/v1/me/risk-answers`, `/v1/meta/risk-questions`, `/v1/me/checklist`, `/v1/assets`, `/v1/assets/quick-add`, `/v1/members`, `/v1/goals`, `/v1/networth`).

## Goal
A first-time user can sign in and record **all** their money in under 7 minutes on a 360 px phone: EPF, PPF, FDs, post office, NPS, LIC/insurance, gold, savings, loans, property and US investments. Every field autosaves, rough numbers are fine, and they immediately see net worth, the debt insight and what's still missing.

## Scope

### 0. Follow-up from the C19 review (do first)
C19 disabled Vite's standard pipeline (`esbuild: false`, `oxc: false`, `minify: false`, a custom TypeScript transform plugin, plus a post-build Terser script) because **your sandbox** blocks esbuild child processes. Render and GitHub Actions don't have that restriction, and a non-standard pipeline is fragile across Vite upgrades.
- Make the **standard Vite pipeline the default**: no custom TS plugin, default transforms and minification.
- Keep the workaround behind the env var `SANDBOX_BUILD=1`: `vite.config.ts` picks the plugin and `esbuild/oxc/minify` settings from it, and `npm run build` runs `minify-assets.mjs` only when it's set. `SANDBOX_BUILD=1` is already set in your environment, so your own builds keep working.
- Document this in `apps/web/README` or the root README dev section.
- Acceptance additionally: the reviewer runs `npm run build && npm run budget` **without** `SANDBOX_BUILD` outside the sandbox; it must pass under budget.

### 1. Schema-driven forms (single source of truth)
- At build time, read `schemas/assets/*.json` (C02 export) and generate typed form definitions: `scripts/gen-forms.mjs` → `src/money/forms.generated.ts`. Commit the output; a test fails when it's stale.
- `AssetForm` renders **required** fields first. Optional fields sit under "More details (for exact value)" collapsed by default (plan wireframe). Widgets:
  - `Rupees` → `MoneyInput` (accepts `5L`, `1.2 cr`, `50k`, shows the parsed full value below: "= ₹5,00,000")
  - `RatePct` → numeric with `%`
  - `MonthYear` → `MonthYearInput` (month + year selects)
  - enums → segmented control or select
  - insurer/bank → combobox using `rules/institutions.yaml` names, exported to JSON at build time (free text allowed)
- **Plain-language labels and help** per field live in `src/money/labels.en.ts`. The schema gives types; labels are written for humans, e.g. "Matures (month and year)". A test fails if any required field lacks a label.
- **Autosave:** the first valid save `POST`s; later edits `PATCH` with a 600 ms debounce. Show a saved/saving/error indicator. API 422 `errors[]` map to the right fields. A 422 with `code: "rate_needed"` shows the friendly prompt "Enter the interest rate printed on your certificate". Offline edits queue in IndexedDB and replay when online (simple queue; last write wins per asset).
- After save, show `valuation.value_today` and `maturity_value` / `maturity_date` and the notes, as in the wireframe ("Value today ≈ …", "We'll remind you 30 days before"). `valuation_status: price_needed` shows "We'll value this once today's price is in" instead of ₹0.

### 2. Money tab (`/money`)
- **Header:** net worth (compact INR), assets / liabilities, last updated, a completeness meter ("Money Map 80% complete · add NPS to finish").
- **"Upload what you have" card** linking to `/import` (C21 builds the flow; here only the entry point).
- **"What else do you have?"** tile grid (12 tiles from §3.1): multi-select `TileToggle`s, saved via `PUT /v1/me/checklist`; then "Continue (n selected)" walks the selected types one card at a time ("1 of 3").
- **List view** grouped by sleeve (Equity India, Equity Global, Debt, Gold, Cash, Property, Protection, Liabilities): each item shows label, value, a one-line status (matures Mar 2027 / price needed / estimate), tap → `/money/asset/:id`.
- **Asset detail:** the form in edit mode + valuation + history (from `GET /v1/assets/{id}`) + "Close this asset" (soft delete) with confirm.
- **Quick-add bar** at the top: free text → `POST /v1/assets/quick-add` → shows the draft in the right form, with `missing` fields highlighted. "not understood" → a friendly hint with examples.
- **Household:** a member switcher chip (All / Self / Spouse / …) filtering list and totals (`member_id` query); "Add family member" sheet.
- **Net worth insights** from `/v1/networth`:
  - the debt insight sentence (from `debt_insight`, e.g. "Including EPF, PPF and FDs, 41% of your money is in debt")
  - the liquidity ladder (3 bars)
  - the maturity calendar (next 12 months, compact list)
  - `unvalued_assets` callout

### 3. Onboarding (`/onboarding/*`)
1. Welcome
2. Tax regime (old/new, with a one-line explainer) + city tier + timezone default `Asia/Kolkata`
3. Money basics: monthly expenses, annual income (optional), dependants, employment type
4. Goals: pick from 6 types, then target amount (MoneyInput) + year + currency (INR/USD for education abroad); multiple goals
5. Six risk questions as large tappable cards (`/v1/meta/risk-questions` → `PUT /v1/me/risk-answers`) → show the result ("Your plan: Growth, limited by your 3-year goal…" using `limited_by`)
6. Consent screen: weekly email, reminders, push. **Unticked by default**, plain-language purpose for each (DPDP). Uses `/v1/me/consents` if it exists; otherwise store locally and show "coming soon". Don't fail.
7. Done → Money tab with the tiles open

Every step is skippable and resumable (progress stored in the profile, or local storage as a fallback).

### 4. Accessibility and UX
- Every input has a visible label; error text is linked via `aria-describedby`.
- Numeric keypad (`inputmode="decimal"`) for money and rates. Tap targets ≥ 48 px.
- No layout shift on validation messages.
- Works at 360 px, at 200% text zoom, and in dark mode.

### 5. Tests (vitest + Testing Library; mock the API with MSW)
- Generated forms in sync with the schemas.
- `MoneyInput` shows the parsed value and rejects junk.
- FD form: required fields only → autosave POST → valuation shown. Edit principal → debounced PATCH.
- `rate_needed` → certificate prompt.
- 422 field errors mapped.
- Quick-add for 3 phrases → the correct form pre-filled + `missing` highlighted.
- Checklist tiles → walk-through order.
- Completeness meter text.
- Member filter passes `member_id`.
- Onboarding happy path; consent defaults unticked.
- Offline queue replays on reconnect (fake `navigator.onLine` + events).
- An axe accessibility check on the Money tab and the FD form (`vitest-axe`), 0 violations.

## Out of scope
The import flow screens (C21), Home/Plan/Ideas/Inbox (C22), push/install/offline polish (C23).

## Acceptance (inside `apps/web`, plus the repo-root Python suite)
1. `npm run gen:api && node scripts/gen-forms.mjs` leaves no diff
2. `npm run typecheck`, `lint`, `test`, `build`, `budget` pass. Report the gzipped initial JS (route-level code splitting keeps it under budget).
3. `.venv\Scripts\python -m pytest` still green
4. A short screen-by-screen note in the result file (what each route shows)
