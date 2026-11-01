# C43: Polish round 3 (Home, Money, Settings, Inbox) and a layout guard for Ideas

**From the reviewer's phone walkthrough (2026-11-01, 384 px, real data)** after C37–C42. Read `AGENTS.md` first, especially the UX rules.

**Builds on:**
- C36 (Home "This week", notes only on Home, Settings sections)
- C37 (polish rules: no card-in-card, no "A · B" mixes, empty states invite an action)
- C33 (Gmail statement inbox)
- C41 (company names on Ideas)

## Findings and what to do

### 1. Home
- **Notes are a card inside a card.** "Helpful notes" is a white card holding a yellow card. Render the notes (at most 2, unchanged rules) directly under a plain section heading, with no outer card. Keep each note's "Why?" and "Hide".
- **"Coming up" opens to nothing.** When nothing is due it reads "Nothing is due in the next 90 days." behind a closed section. Hide the section entirely when the list is empty.
- **"I have extra money this week"** is a ghost button, so its padding pushes the text right of the card's left edge. Style it as a quiet link aligned with the content. Keep a 44 px tap height.

### 2. Money
- **Group headings say nothing about size.** Add each group's total on the right of its heading ("Funds", "Shares", "Deposits", …), in the household display currency, using the same formatter as the rows (`MoneyValue` or `formatHouseholdAmount`).
  - Sum with decimal-string helpers, not floats.
  - The total covers the whole group, including rows hidden behind "Show all N".

### 3. Settings
- **The marker sits alone above the title.** Each section's ▶ is on its own line above the title and summary. Put the marker on the title's line, as a chevron on the right that rotates when open, and keep the summary under the title. The whole row stays the tap target. Keep the existing test ids.
- **Mixed summaries:** "2 people · Only you" and "Currency: INR · Alerts off" mix unlike facts. Use a comma ("2 people, only you"), or two short lines if a comma reads badly.

### 4. Inbox
- **Repeated title.** The page header says "Inbox", and the page repeats "Inbox" as a heading above the updates list. Rename that heading "Updates" and keep its one-line intro.
- **Raw email subject.** The statement card shows the raw subject ("Your NSDL CAS - September 2026 (NSDL ID - 10910…"), with an internal ID and the term "CAS".
  - Show "Statement for September 2026" when the subject contains a month and year (English month names, any case, full or 3-letter).
  - Otherwise show no subject line.
  - Keep "Received 16 Oct".
  - The review page may still show the full subject.
  - Do the month parsing in the API, as `statement_month: "2026-09" | null`, so the PWA never parses subjects. Add a unit test with a few real-world subject shapes, all synthetic.

### 5. Layout guard for Ideas (e2e)
- Two layout bugs reached the owner's phone this week: Home's amount row and Ideas' long names. No e2e journey opens Ideas with data.
- Add an e2e journey that seeds a research pack for the current week through the test database: 2 India and 2 US ideas, one with a 45-character company name, plus `why` and `entry_hint`. It signs in as a Denmark household, then:
  - opens Ideas
  - checks the long name shows on one line
  - checks the US and world funds card shows its three rows
  - calls `expectFitsScreen`
- Also call `expectFitsScreen` on Settings and Inbox in an existing journey.
- Seed through the test database the way other journeys seed data. Never call a model.

## Tests
- **PWA:**
  - notes render without an outer card
  - "Coming up" is absent when empty
  - the group totals use the formatter and include hidden rows
  - the Settings rows have the marker on the title line, with test ids kept
  - Inbox shows "Updates" and "Statement for September 2026", and no subject line when `statement_month` is null
- **Python:** `statement_month` parsing, and the field on the inbox item response.
- **e2e:** the new Ideas journey and the extra `expectFitsScreen` calls.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), ruff check and format (`--no-cache`), `python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- The e2e suite runs as `docs/cards/C24-e2e.md` describes. The reviewer also runs it.
- Write `docs/cards/C43.result.md`, with before/after notes per screen.

## Notes
- No new dependencies and no migration. No change to money logic, policy or research.
