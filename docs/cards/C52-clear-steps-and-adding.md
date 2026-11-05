# C52: Clear weekly steps and one way to add holdings

**Source:** the C51 review (`docs/reviews/C51-ui-review.md`), items 1–4 plus the goal-save fix from item 8. The reviewer checked its findings against the code. This card lists what was **accepted**, with the reviewer's changes. Where this card and the review differ, **follow this card**.

## Why
Users can't tell which share a sell step is about, and its instructions say to **invest**. Adding holdings has five overlapping entrances, and the owner looked for broker upload in Inbox. Saving a deposit shows three competing buttons. First use promises "Your plan is ready" before anything is recorded. Typing a goal and pressing "Continue" loses it.

## Read
- `AGENTS.md` (UX rules, copy rules, PWA commands).
- `docs/reviews/C51-ui-review.md`: items 1, 2, 3, 4 and 8, the rows for screens 05, 07, 08, 12, 13–14, 16, 23 and 24 in section 2, and "Across the app".
- Source: `apps/web/src/home/HomePage.tsx`, `apps/web/src/lib/labels.ts`, `apps/web/src/money/MoneyMapPage.tsx`, `apps/web/src/money/AssetFormPage.tsx`, `apps/web/src/imports/`, `apps/web/src/inbox/`, `apps/web/src/app/`, `apps/web/src/onboarding/OnboardingPage.tsx`, `apps/web/src/plan/PlanPage.tsx`, `apps/web/src/features/api.tsx`, `apps/web/src/ui/primitives.tsx`, `copy/en.yaml`.

## Scope

### 1. Weekly steps on Home say what to do (review item 1)
Two bugs to fix:
- **Sell steps tell users to invest.** `actionHowTo` (`HomePage.tsx`) gives a sell step the buy text ("Use this person's account to invest {amount} as planned."). The instructions must match the step:
  - **Sell steps** (`SELL_EXIT`, `SELL_TRIM`): "Sell these shares in your investment account, then tap I sold it."
  - **Review step** (`REVIEW_EXIT`): "Read the reasons, then decide whether to sell."
  - **Buy steps:** keep the existing account-specific text.
  - **Any other step with no specific text:** "Check the details before taking this step."
  - Keep showing the existing quantity and tax lines in the sheet.
- **The "Why?" rows can describe a different share.** `exitCardForAction` falls back to the first exit card *of the same kind*. Remove that fallback: match only by `security_id`, or by `symbol` when there's no `security_id`. When nothing matches, show no trigger rows.

Wording and layout:
- **Step titles** match the holding page's words from C49. In `copy/en.yaml` `exit_actions`:
  - `EXIT` → "Sell"
  - `TRIM_TO_TARGET` → "Sell some"
  - `REVIEW_THEN_EXIT` → "Think about selling"
  - Other labels stay.
- **Name the share** on sell, sell-some and review steps, on its own line that wraps (no one-line clamp). Look up the name by exact `security_id` in data Home already loads. If no name is found, show the payload `symbol`; if there's no symbol either, show no name line. Never guess. If no loaded data carries share names, you may add a display-only `name` to sell-step payloads in the backend. Mark it **[backend]** in the result and update the golden file. That is the only backend change allowed in this card.
- **Buttons:**
  - "How?" becomes "Show steps"; on the first step only, it's the primary button.
  - The completion button reads "I bought it" on buy steps, "I sold it" on sell and sell-some steps, and "Done" on all others.
  - Completion buttons are secondary. Their behaviour, mutations, and the existing purchase sheet and stale-step handling stay exactly as they are.
- **"Why?" in the sheet:**
  - Replace the raw `key.replaceAll("_", " ")` rows with an explicit label map in `copy/en.yaml`, for the params that real steps carry.
  - Leave out any param that has no label.
- **Monthly amount:** when steps are listed *and* a monthly amount is already set, put the monthly amount editor below the steps in a section titled "Monthly investment", closed by default. When no amount is set, keep it open: it's the most useful input.
- Keep at most 3 steps, in the server's order. Amounts, dates and payloads are unchanged.

### 2. One way in to add holdings (review item 2)
- **Money:** one primary button, "Add holdings", which opens the existing sheet with the same title. In the sheet:
  - "Upload a file" is the first and primary choice, with the line "Broker files and investment statements."
  - "Add manually" is secondary.
  - "Get statements from Gmail" is a quiet link below them.
- **Empty Money:**
  - Title "Add your first holding", text "Upload a file or add an item manually."
  - Its button opens the same sheet. Today it only scrolls to a closed section.
  - Remove the duplicate chooser button.
- **"Add manually"** opens and focuses the existing chooser (the tiles). Rename its summary "Add manually". Move "Quick add" below the tiles, in a section that starts closed. Rename import history "Previous uploads".
- **Remove the "Money Map {n}% complete" line from the chooser.** Keep the selected-item count and the step-by-step progress.
- **Import page:**
  - Heading "Upload holdings".
  - Choices: "Broker file (Excel or CSV)", "Mutual fund statement (PDF)" and "Combined investment statement (PDF)". Provider names stay in their "How to get it" help.
  - Make each choice a compact row; selection and help keep working.
  - Merge the two privacy notes into one under Upload: "Your file and password are used for this upload only. They are not stored." Keep the existing promise; make no broader claim.
- **Inbox:**
  - Add one quiet "Upload a file" link near the intro, outside any conditional block, so it shows even with no emails. It goes to the Import page.
  - Don't add a second uploader.
  - Rename "Review email settings" to "Email settings".
- **Bottom navigation:** highlight Money on `/import` and `/import/:jobId`. URLs stay the same.
- **Dead link:**
  - "Your plan is ready" links to `/money?checklist=open`, and nothing reads that parameter.
  - Add a query parameter that opens the shared "Add holdings" sheet: `/money?add=1`, or `?add=manual` to go straight to the chooser.
  - Use it for every "add holdings" link from other screens.
- **Names on Money:** holding and asset names may wrap to two lines instead of being cut off. Use "1 unit", not "1 units".

### 3. Saving and viewing a deposit or other item (review item 3)
- **Editing buttons:**
  - Keep autosave.
  - Replace the competing "Save now" / "Finish Money Map setup" / "View asset" buttons with one primary button: "Save and continue" in a multi-item flow, "Save and return" for a single item.
  - Leaving must wait until the latest edit is saved, or queued while offline; it must not wait for the server when offline.
  - Invalid input keeps the form open. A failed save keeps the draft.
  - Keep the truthful status text: "Saving…", "Saved", and the offline "waiting for connection" text. A queued edit must never look synced.
- **Detail page** (`/money/asset/:id`):
  - Show a summary first: the existing value, maturity value and date, bank and owner.
  - One primary "Edit details" button opens the existing form. After saving, return to the summary.
  - Remove the "View asset" link that points to the same page.
- **Whose item it is:** when the household has more than one adult, show the existing "Whose is this?" choice before the deposit fields, keeping its default and stored member id. Put the "new money" eligibility note next to it, under the heading "Adding new money". Lead with "You can still record an existing account."
- **Closing an item:**
  - Collapse "Edit history".
  - Inside it, a quiet "Mark as closed" button opens the existing confirmation. Text: "This removes the item from your current totals. Its history stays saved." Buttons: "Mark as closed" and "Keep it".
- **Deposit fields** run in this order: bank, amount, interest rate, maturity. Show the maturity date in the app's readable date format, not ISO. Don't edit generated form definitions.

### 4. First use points to the next step (review item 4)
- **Welcome:** the button reads "Get started".
- **After the household step:**
  - Title "You're ready to start", text "Add what you own to see your plan."
  - Primary "Add holdings" goes to `/money?add=1`. Secondary "Explore the app" goes to Home.
  - Setup stays skippable, with the same defaults.
- **Empty Plan:** its "Add holdings" action goes to `/money?add=1`.
- **Home with no holdings and no steps:**
  - "Add holdings" is the primary action; the monthly amount stays below it.
  - "Money: ₹0" becomes "No holdings added yet" only when the loaded record counts are zero (holdings, other items and loans). Never decide this from a zero total.
- **Loading screens:** show visible text matching the existing accessible label (for example, "Loading your plan…"). Keep cached content while refreshing. On a real error, show the existing error message and a "Try again" button that retries the query.
- Keep the health-cover note and all existing highlights as they are.

### 5. A typed goal is never lost (review item 8, first part)
- On the goal form, if the user has entered an amount, "Continue" saves the goal first, then moves on. If saving fails, stay on the form, keep the values and show the existing error.
- With nothing typed, show only "Skip for now"; drop the duplicate "Continue".
- After saving, show "Goal saved" and a quiet "Add another goal".
- When the form was opened from Home, remove the setup progress bar and use the title "Add a goal", keeping "Back to Home".
- Leave targets, currencies, default year and priority rules unchanged.
- Risk questions are **not** in this card.

### 6. Tap targets on the screens you touch
Every control you add or change, plus the "What's this?" (`Term`) button and the disclosure summaries on Home, Money, Import and Inbox, is at least 44 × 44 px. Keep visible focus and expanded state, and don't shrink anything that is larger today.

## Rules
- Front end and copy only (`apps/web`, `copy/en.yaml`, `e2e`), except the one optional [backend] name field in §1.
- Never change amounts, weights, rates, tax figures, dates, or the rules behind them.
- No new runtime dependencies. Keep within the bundle budget.
- Copy: sentences of 20 words or fewer, titles of 8 words or fewer, no banned jargon. Put every new string in `copy/en.yaml` and regenerate with `npm run gen:copy`; never hand-edit generated files.
- WCAG 2.2 AA:
  - Opening a sheet or chooser moves focus to its heading.
  - Closing a sheet returns focus to the button that opened it.
  - Every control has a label.

## Tests
- **Vitest** (update or add):
  - `tests/features/HomePage.test.tsx`: instructions for buy, sell, sell some, review and loan steps; no trigger rows from another share; the name line for a matched step, the symbol fallback, and no line when neither exists; button labels per step kind; the first step's "Show steps" is the primary button; completion still calls the same handlers; the monthly section is closed when an amount is set and open when not; the "Why?" rows show labels and drop unknown keys; order and amounts unchanged.
  - `tests/money/`: every "add holdings" entry (Money button, empty state, `?add=1`, `?add=manual`) opens the same sheet or chooser; "Quick add" starts closed; the chooser has no % line; "1 unit"; save-and-leave after the latest edit; invalid input stays; a failed save keeps the draft; offline queues and leaves with the waiting text; the detail summary and "Edit details"; "Mark as closed" confirmation.
  - `tests/imports/`, `tests/features/InboxPage.test.tsx`, `tests/features/GmailStatementInbox.test.tsx`: new choice labels; one privacy note; Inbox shows "Upload a file" with and without emails, and it goes to `/import`.
  - `tests/app/`: Money is active on `/import` and `/import/:jobId`.
  - `tests/onboarding/`: the "ready" screen's buttons and destinations; Continue saves a typed goal; a failed save keeps values; only "Skip for now" when empty; Home-opened goal title and back link.
  - `tests/features/PlanPage.test.tsx`: the empty-plan action goes to `/money?add=1`; visible loading text; "Try again" retries.
  - Update `tests/copy.test.ts` and any role or name queries the new labels break.
- **e2e** (update the specs these changes break, and add):
  - Start on Inbox, tap "Upload a file", choose "Broker file (Excel or CSV)", upload the synthetic broker fixture `broker-import.spec.ts` already uses, and reach the review.
  - `usability.spec.ts`: start the upload timer on Money (not `/import`) before finding "Add holdings".
  - A sell step on Home shows "I sold it", and its sheet has no "invest" text (use the existing `seedHoldingReviews`).

  Codex's sandbox can't start the e2e stack (`spawn EPERM`). Write and type-check the specs, say in the result file that you couldn't run them, and the reviewer will run them.

## Out of scope
- Review items 5, 6, 7, 9 (except §6 above) and 10, the risk-question flow, and the remaining section-2 rows. They go to C53.
- Visual rebranding. Any change to how steps, amounts or verdicts are decided.

## Acceptance
Inside `apps/web`:
- `npm run gen:api`
- `npm run gen:copy`
- `npm run typecheck`
- `npm run lint`
- `npm run format:check`
- the full `npm test -- --run`
- `npm run build`
- `npm run budget`

Python, only if you made the optional [backend] change:
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`

e2e: `cd e2e; npx playwright test --list` must list the specs without errors (it loads and compiles every spec). The reviewer runs the e2e suite.
