# C36: A simpler everyday app

**Requested by the product owner:** "is there a possibility to make the UI simple and intuitive? should be very easy."
**Builds on:** all PWA cards (C07 Money Map, C21 Home, C22 Ideas, C26–C34), C33 Gmail inbox, C35 buy hints. Read `AGENTS.md` first. Its UX rules are the goal of this card: one primary action per screen, details behind "Why?" or closed sections, at most 2–3 notes per screen, plain words, and "when unsure, show less".

## What the owner hit on their phone (2026-10-31)
- **Finding this week's suggestions.** They had to ask. The research ideas sit in Ideas → a closed "More research" section, below a "Buy from Denmark" card that says "Nothing to buy this week".
- **No buy steps at all.** The plan has no new money. The amount is set in Settings → "Fine-tune your plan" → "Monthly investing amount", which is four taps deep on a 4.6-screen page.
- **The same notes everywhere.** "Review your health cover" and "Check local tax details" appear on Home, Money *and* Ideas.
- **Long screens.** Money is 5.7 phone screens long. Settings is 4.6 and mixes everyday choices with rare ones: 31 "monthly fund amounts" rows, devices, privacy.
- **Waking server.** After a quiet spell the free API takes up to 3.5 minutes to wake. The app shows "Waking up the server…", then gives up with "We couldn't reach the server. Try again" on the sign-in page.

## Scope (PWA only; no API change unless a step says so)

### 1. Home answers "what do I do this week?"
- **Order on Home:**
  1. one "This week" card
  2. the money total (one line, links to Money)
  3. at most 2 notes
  4. "Coming up" (closed if empty)
- **"This week" card:**
  - With actions: show them as today, in plain words, with the buy hint's parts where an action buys in parts.
  - With no money to invest set: the card asks "How much can you invest each month?". It has one amount field (in the member's main currency, DKK for a Denmark member) and **Save**. Saving writes the same field as Settings → "Monthly investing amount", through the existing API, and refreshes the plan. Show "Nothing to buy this week ✓" only when an amount is set and the plan has no buys.
  - A quiet link "I have extra money this week" opens the same field for "new money this week".
- **"Make it more accurate"** (risk questions, add a goal, email updates) becomes one closed section under "This week", not a list in the main flow.

### 2. Ideas shows the ideas
- **Order:**
  1. "Buy from Denmark" (unchanged, first when it has something to buy)
  2. **"This week's research ideas"**, open by default: the top 3 per market from the pack, each with its C35 buy hint line. "See all ideas" opens the rest.
  3. "Funds to compare" (the core picks): closed
  4. "Explore the wider picture": link only
- **When "Buy from Denmark" has nothing to buy,** shrink it to one line ("Nothing to buy from Denmark this week") and put it after the ideas.
- **Remove the notes** from Ideas. They live on Home (see 4).

### 3. Money gets shorter
- **Top of the page:**
  - the total, then the list by kind (funds, shares, deposits, savings, loans), each row tappable (existing detail pages)
  - **one** primary button, **"Add or update"**. It opens the existing sheet with three choices: upload a statement, get statements from Gmail, add one by hand.
- **Closed by default:** "By family member", "Imports", "Check an investment", and "What else do you have?" (the EPF/PPF/FD tiles).
- **No notes on Money** (see 4).
- **Move "Monthly fund amounts"** (the 31 rows from Settings) here, as a closed "Monthly amounts" section. It belongs next to the funds.

### 4. Notes appear once
- The `highlights` notes render only on Home: at most 2, ranked by the backend order. "See all notes (N)" opens the rest.
- Money and Ideas stop rendering them.
- Dismissal ("Hide") works as today.

### 5. Settings becomes four short sections
- Four closed sections, each with a one-line summary of its current state:
  1. **You and your family:** household members, share with partner
  2. **Your plan:** monthly amount, extra money this week, risk answers, the fine-tune fields
  3. **Statements:** Gmail statements, saved passwords
  4. **App and privacy:** display currency, notifications, privacy, signed-in devices
- Nothing is removed, only grouped. Every existing control keeps its label and test id, so e2e tests keep working; update the selectors if a section must be opened first.

### 6. A waking server doesn't look like an error
- **While the API wakes,** keep showing the last cached screen (React Query already has it) with a small "Updating…" chip. Don't replace the screen with the sign-in page.
- **Wait up to 4 minutes** before showing "We couldn't reach the server". Keep the refresh token, as today.
- **If the page was showing the sign-in screen only because the server was unreachable,** retry quietly every 20 seconds and return to the app when the server answers.

## Tests
- **PWA:**
  - Home with no monthly amount shows the question and saves it; with an amount and no buys it shows "Nothing to buy this week ✓"
  - Ideas shows research ideas open by default, top 3 per market, and no notes
  - Money shows one primary button and the closed sections
  - notes render only on Home (max 2 plus "See all")
  - Settings sections start closed and each control is still reachable
  - a slow API keeps the cached screen with "Updating…"
- **e2e (Playwright):** update the journeys that relied on the old layout. The full suite must pass.
- **Copy:** every new string goes through `gen:copy` and the copy rules: at most 20 words a sentence, titles of 8 words or fewer, and no banned terms.

## Acceptance
- In `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the **full** `npm test`, `build`, `budget`.
- Python suite and ruff, if any Python changed.
- The e2e suite (`e2e/`), run as `docs/cards/C24-e2e.md` describes.
- Write `docs/cards/C36.result.md`, with before/after notes for each screen.

## Notes
- No new dependencies. Reuse existing components (`Card`, `Highlights`, the add sheet, form helpers).
- Don't change any money logic or API contract, except to read existing fields.
