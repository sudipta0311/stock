# C31: Who's signed in, sign out, a detailed help centre, and a guided first visit

**Requested by the product owner:** "Show the logged-in user at the top and give log out options. Also add detailed help. Make it very intuitive."
**Plan:** `docs/UPGRADE_PLAN.md` §8 (UX). Read `AGENTS.md` first, especially the **UX rules**. Builds on C28/C28b (shell, header, notes) and every existing screen. PWA plus copy; backend only if `/v1/me` lacks the display name or email.

## Goal
A person always knows who is signed in and which family plan they are looking at, and can sign out in two taps. Help is never more than one tap away: a guide written for this app, in plain words, with steps and examples, opened from the screen it explains. A first-time user gets a short, skippable tour of the four tabs.

## Scope

### 1. Account menu (header, every signed-in screen)
- **Avatar on the right of the header:**
  - a circle with the person's initials (no external image; no Google photo fetch)
  - the first name next to it from 400 px width up
  - an accessible label "Account menu for <name>"
- **Tapping it opens a sheet:**
  - name and email (from `/v1/me`)
  - the household being viewed and the person's role, in plain words ("Your family plan · You can edit")
  - "Switch family plan", only when there is more than one
  - Settings
  - Help and guide
  - **Sign out**
- **Sign out:**
  - a confirm step ("Sign out of Wealth Copilot on this phone?")
  - clears the session token, the React Query cache and session storage, and unregisters push for this device only if the person asks
  - goes to `/login` with "You've signed out." shown once
  - in OIDC mode, no Google account logout. Say so in Help ("This signs you out of this app, not of Google").
- **The login screen:**
  - "Continue with Google" instead of "Continue with secure sign-in" in OIDC mode
  - one line saying who can use it ("Only invited family members can sign in")
- **Expired session:** show "Please sign in again" and keep the screen they were on, so they return to it after sign-in.

### 2. Help centre (`/help`, lazy route)
- **Content lives in `copy/help.en.yaml`.** It is generated to its own JSON that is loaded only by the help route; extend the copy generator and its sync check. Same writing rules: ≤ 20 words per sentence, titles ≤ 8 words, no banned terms (use `<Term>` where a term is unavoidable), calm tone, India and Denmark examples, synthetic numbers only.
- **Layout:**
  - a search box (client-side filter over titles and text)
  - topic groups
  - each article: a one-sentence summary, 3–7 numbered steps, an example, and "Related" links
  - deep links: `/help#<article>`
- **Articles:**
  - **Getting started:** what the app does and doesn't do; the four tabs; the first 5 minutes; rough numbers are fine.
  - **Adding your money:**
    - one article per asset group: deposits, PPF/EPF, post-office schemes, insurance policies, gold, shares and funds, property, abroad
    - what each field means, and where to find the number (passbook, statement, app)
  - **Uploading statements:**
    - Indian fund statements: how to download from CAMS or KFintech, and the password hint
    - broker files (Indian brokers; Nordnet, Saxo and Danske Bank in Denmark)
    - what happens to the file
  - **Your plan:** money mix, safe money, why we suggest what we suggest, how often it changes.
  - **This week:** what an action is, "Done", "How?", why there are at most three.
  - **Loans (C30):**
    - adding a loan
    - recording an extra payment, a new rate or a balance from a statement
    - "pay extra or invest" explained
  - **Investing from Denmark:** the share savings account (Aktiesparekonto) first, EU-listed funds, why not US-listed funds, the tax notes that still need a check.
  - **Family plans:** adding people, inviting a partner, roles (can edit / can view), whose name to invest in.
  - **Notes and emails:** what the notes mean, hiding them, the Sunday email, reminders and how to turn them off.
  - **Privacy and safety:**
    - what is stored and where (EU servers)
    - who can see it
    - how to delete everything
    - signing out
    - "We never ask for passwords or PINs"
  - **Glossary:** every `<Term>` used in the app in one list, with plain explanations.
  - **When something looks wrong:** wrong numbers, a missing price, a stuck upload. How to fix them, or how to tell us.
- **Contextual help:**
  - a small "?" button in the header of Home, Money, Plan, Ideas, Inbox and Settings opens the matching article
  - forms get "Where do I find this?" links to the right article section
  - existing `<Term>` explanations link "Learn more" to the glossary entry

### 3. Guided first visit (lazy, once per person per device)
- After onboarding, a 4-card tour over the real screens (Home "This week", Money "Add what you have", Plan "Your money vs your plan", Ideas): one sentence each, with Next, Skip and a progress dot row.
- Remember "seen" in localStorage wrapped in try/catch. It can be replayed from Help ("Show the tour again").
- Respects `prefers-reduced-motion`. Focus is trapped while open, and Escape closes it.

### 4. Intuitiveness pass (small, everywhere)
- Every screen's empty state says what to do next in one sentence and offers one button (check Inbox, Ideas, Plan details, the loan list, the import history).
- Every number on a summary card that a newcomer may not understand (net worth, safe money, money mix, monthly loan payments) gets a `<Term>`-style "What's this?" with a link to Help.
- Back links name where they go. Destructive actions confirm in plain words.
- Toasts after saving say what happened ("Saved. Your plan will update in a moment.").

## Constraints
- **Initial JS budget:** stay within the existing 170 KB gzip limit. It has only ~0.4 KB left, so first move rarely used code out of the initial bundle (e.g. the Sheet/Toast internals, Inbox, Settings sub-panels, the import flow) into lazy chunks. The help, the tour and the account sheet content must be lazy; only the avatar button may be in the shell. Do not raise the budget.
- No new runtime dependencies.
- Keep `data-primary` (the account menu and "?" buttons are not primary), the one-primary-action tests, the highlight limits, the banned-terms and sentence-length tests (extend them to `help.en.yaml`), and axe checks in light and dark.
- No real personal data in help examples.

## Tests
- The header shows initials and the name from `/v1/me`; the sheet shows the email, household and role. Sign out asks for confirmation, clears the token and cache, and lands on `/login` with the notice.
- The help route renders, search filters articles, deep links scroll to the article, and every screen's "?" button points at an existing article.
- Help copy passes the banned-terms and sentence-length checks; the generator sync check covers `help.en.json`.
- The tour shows once after onboarding, Skip works, and "Show the tour again" replays it; reduced motion is respected.
- An expired session returns to the same screen after sign-in.

## Acceptance
1. PWA: `gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, full `npm test`, `build`, `budget` (≤ 170 KB initial; list the lazy chunks you created) pass.
2. Python: full `pytest` and `ruff` pass if any backend or copy shared with emails changed.
3. The result file lists the help articles, and what was moved out of the initial bundle and how many KB each saved.
