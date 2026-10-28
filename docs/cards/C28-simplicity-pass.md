# C28: Simplicity pass: make the app simple, intuitive and easy; hide the complexity behind the scenes

**Requested by the product owner:** "Make the app simple, intuitive and easy to use. Hide the complexity behind the scenes."
**Plan:** `docs/UPGRADE_PLAN.md` §2.3 (plain language, calm), §8 (UX). Read `AGENTS.md` first, especially the **UX rules** section. Runs **after** C23 and C27b (all screens exist). Touches the backend only to rank and trim what the UI shows.

## Goal
A first-time user (an Indian NRI in Denmark, or their spouse in India) understands every screen in seconds and always knows the one next thing to do. The engine's sophistication (tax rules, residency, placement, exits) stays invisible unless they tap "Why?" or "Learn more".

## Scope

### 1. Backend: rank and trim (`wealth/policy/highlights.py`, pure)
- `highlights(plan, context) -> list[Highlight]`: from all plan notes, residency notes, placement notes, foundation gaps and warnings, choose **at most 3 per screen context** (`home`, `plan`, `ideas`, `money`). Rank by:
  1. needs the user's action now
  2. money at stake (₹)
  3. is about something on that screen
  4. first time seen (user hasn't dismissed it)
- Each highlight: `{key, screen, title (≤ 8 words), body (≤ 25 words), learn_more_key, severity: info|attention}`, with copy from `copy/en.yaml`.
- `GET /v1/plan` adds `highlights_by_screen`. Everything else stays available under `details` for the "Learn more" views.
- `POST /v1/highlights/{key}/dismiss` (per user, per household, stored; RLS). A dismissed highlight doesn't come back unless its underlying values change materially.
- **Tests:** NRI-DK with ~20 notes → Home ≤ 1, Plan ≤ 3, Ideas ≤ 2; ranking order; dismissal persistence; RI user with few notes.

### 2. Plain-language guard (PWA, emails, research text)
- **Single source:** the repo-root `copy/banned_terms.yaml`, generated into `apps/web/src/copy/banned-terms.json` by the existing copy generator (and checked for sync like `en.json`). The Python side (emails, research text in C15c) reads the YAML.
- **Terms:**
  - `sleeve`, `satellite`, `tranche`, `STP`, `route`, `PFIC`, `ABIS`, `PRIIPs`, `LRS`, `TCS`
  - `Schedule FA`, `XIRR`, `IRR`, `RNOR`, `glide path`, `rebalance`, `allocation`, `tax lot`, `drift`
  - `CAS`: allowed only on the import screen, as "CAS statement"
  - `ETF`: allowed with an explainer on first use
  - `UCITS`, `ISIN`
- **Matching:** whole-word and case-insensitive. **Security names and symbols are excluded**: mark them `data-security` and skip them. "TCS" is also Tata Consultancy Services, and fund names contain "ETF".
- **Test:** render every route in its main states (MSW fixtures) and scan visible text. Any banned term outside an explicitly allowed context fails. The allowed context is a `<Term>` component that shows a "What's this?" tooltip.
- **Sentence-length test** over `copy/en.yaml` user-facing strings: ≤ 20 words per sentence; titles ≤ 8 words.
- **Weekly email:**
  - at most 3 actions + 1 note; everything else is a "See details in the app" link
  - a Python test renders the NRI-DK household fixture's email and runs the banned-term scan on its text and HTML

### 3. Screen simplification (apply to existing screens)
- **Home:**
  - net worth (one number; INR + DKK for NRI households)
  - **"This week"** (0–3 actions; each one line + amount + who; buttons **Done** and **How?**; "Why?" inside How?)
  - at most 1 highlight
  - "Coming up": the next 1 dated item only, with a "See all" link
  - nothing else above the fold
- **Plan:**
  - top: one visual, "Your money vs your plan" (simple stacked bar with words like "A bit too much in deposits")
  - one sentence answering "How much should be safe money?" (the debt answer)
  - up to 3 highlights
  - everything else (per-member breakdown, limits, global route, insurance, stress, foundation details) in collapsed "Details" sections, closed by default
- **Money:** a big **"Upload your CAS statement"** card first; quick-add box second; the list grouped with plain headings ("Indian shares", "Deposits & PPF", "Gold", "Abroad"); the 12-tile checklist collapsed after first completion.
- **Ideas:** core picks as 2–3 simple cards ("Indian index fund", "Global fund in your Aktiesparekonto"); research ideas behind "More ideas"; the disclaimer one line at the bottom.
- **Onboarding: the shortest path.** Sign in → "Who are we planning for?" (just me / me + spouse) with residency → "Upload CAS or add a few things" → plan. Risk questions, goals, income and consent move to an optional "Make it more accurate" checklist on Home (with defaults: moderate risk, no goals, emails off). Target: ≤ 5 taps to the first plan when skipping uploads.
- **Settings:** grouped into 4 sections (Family, Notifications, Privacy, About). Advanced investing preferences are under "Fine-tune" (collapsed).
- **Numbers:** round to the nearest ₹100 in action amounts (≥ ₹10,000) and show lakh/crore compact values in summaries; full values only in detail views and forms.
- **Empty states and first-run hints:** one friendly sentence + one button each.

### 4. Measurable checks (tests)
- Each route renders **one primary button** (`data-primary` attribute; test counts exactly 1 on Home, Money, Plan, Import and each onboarding step).
- Above-the-fold content on a 360×740 viewport: ≤ 4 cards on Home and Plan (a vitest + jsdom layout heuristic counting top-level cards before the first collapsed section; or Playwright in C24 if jsdom is insufficient).
- Collapsed-by-default details on Plan.
- Highlights ≤ the limits per screen.
- Banned-terms and sentence-length tests green.
- Axe checks still green.

### 5. Usability script (for the reviewer and C24)
`docs/usability-script.md`: 5 tasks with target times. Add an FD (< 30 s); upload a CAS (< 2 min); find "what should I do this week" (< 10 s); find "how much should be safe money" (< 15 s); add your spouse and set their tax residence (< 60 s). Plus the observations to record.

## Acceptance
1. Python: `pytest` (full) + `ruff` pass; OpenAPI refreshed
2. PWA: `npm run typecheck`, `lint`, **full** `test`, `build`, `budget` pass; banned-terms and sentence-length tests included
3. The result file shows before/after for Home and Plan (the list of elements above the fold) and the onboarding tap count
