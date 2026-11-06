# C53: Understand a share, the plan and ideas at a glance

**Source:** the C51 review (`docs/reviews/C51-ui-review.md`), items 5–10 and the accepted rows of section 2. C52 is done: it added the shared "Add holdings" sheet (`/money?add=1`), the "Show steps" pattern, and 44 px targets on Home, Money, Import and Inbox. Reuse them. Where this card and the review differ, **follow this card**.

## Why
The owner's rule: "the UI needs to be simple and intuitive". On a share's page the suggestion sits below a closed "Details" row. The Plan shows a target without what you have now. Ideas cut company names to "S…". The risk questions are one long page. Small "What's this?" links are hard to tap. Help is one very long page. Every change here should leave a screen **shorter or clearer**, never busier.

## Read
- `AGENTS.md` (UX rules, copy rules, PWA commands).
- `docs/reviews/C51-ui-review.md`: items 5–10, the section 2 rows for screens 03–04, 05, 09, 16, 17–18, 20, 21, 22, 24 and 25, and "Across the app".
- `docs/cards/C52.result.md`, for what changed already.
- Source: `apps/web/src/money/HoldingPage.tsx`, `apps/web/src/money/MoneyMapPage.tsx`, `apps/web/src/money/labels.en.ts`, `apps/web/src/money/AssetFormPage.tsx`, `apps/web/src/plan/PlanPage.tsx`, `apps/web/src/ideas/IdeasPage.tsx`, `apps/web/src/ideas/WorldPage.tsx`, `apps/web/src/onboarding/`, `apps/web/src/help/`, `apps/web/src/settings/SettingsPage.tsx`, `apps/web/src/inbox/InboxPage.tsx`, `apps/web/src/ui/primitives.tsx`, `apps/web/src/household/currency-view.tsx`, `copy/en.yaml`.

## Scope

### 1. A share's page answers "what is it worth, and what should I do?" (review item 5)
- **Order from the top:**
  1. A quiet "Back to Money" link.
  2. The name.
  3. The value card.
  4. "Our view": the verdict, **one** leading reason (the first the API gives), the tax note if there is one, and the check date.
  5. A closed "Why this suggestion?" with the remaining reasons and "How we check".
  6. A closed "Holding details" with the code, exact quantity and the like (today's "Details").
- **Buying price:** "Cost not available" becomes "Buying price not recorded". Inside "Holding details", add "Profit or loss needs your buying price."
- **Value in the household currency:** show the value with the shared `MoneyValue`, the same way Money does: INR plus DKK when the household uses DKK. Reuse the existing rate and its "rate unavailable" fallback. Keep original-currency prices separate from converted totals.
- **Don't change:** verdicts, reasons, their order, unknown or unchecked states, or tax text. Add no buy or sell button.

### 2. Plan shows what you have next to the target (review item 6)
- **Safe money:** put the answer before the chart, as two plain rows, "You have {current}" and "Target {target}", using the existing values for the selected person or family. Keep the existing gap sentence.
- **Its button:** "Add safe money" opens a fixed-deposit form, which reads like an instruction to invest. Replace it with "Add holdings", going to `/money?add=1`, with the line "Add items you already own." There is still only one primary action on the screen.
- **Labels:**
  - "Show amounts" becomes "Compare amounts".
  - In the chart legend, the gold row reads "Gold", not "Other holdings". Change only that row.
- Keep the chart's text alternative.
- **Don't change:** targets, gaps, the new-money rule, or how any amount is worked out.

### 3. Ideas lead with the company and what you already own (review item 7, and section 2 row 21)
- **Company name and symbol:** the name gets its own full-width row and wraps naturally, with no one-line clamp and no `title`-only access. The symbol goes on a quiet line below it.
- **Details:** move confidence, sector and valuation badges into the existing "Why this idea?" disclosure.
- **Order:** the "You already own about … through your funds" sentence comes **before** the entry hint ("Buy in 3 parts…"). Keep the sentence word for word.
- **World funds:** keep the fund the plan uses visible. Group the other fund rows under "Other funds to compare", closed by default, keeping their product names and broker search details.
- **Monthly amount:** when it isn't set, drop "Monthly buying: keep going." and keep the "Set your monthly amount" link.
- **Don't change:** entry parts, months, prices, ownership amounts, thresholds, or the "You already own enough" group.
- Replace the one-line-clamp assertion in `e2e/tests/ideas-layout.spec.ts` with "the full name is visible and nothing overflows".

### 4. World scenarios page (section 2 row 22)
- Add a quiet "Back to Ideas" link.
- The intro becomes "These examples show how your holdings might move."
- Show the INR results with `MoneyValue`, so the household currency matches the rest of the app. Keep the signs, the "up" and "down" words, and the scenario definitions.

### 5. Risk questions one at a time (review item 8, second part)
- Show one question at a time:
  - Above it, "Question {n} of {count}".
  - Below it: "Back" (from the second question on), the primary "Next", and on the last question the primary "Save answers".
  - A quiet "Skip for now" returns where the user came from without saving.
- After "Next" or "Back", move focus to the new question.
- Answers chosen so far survive "Back".
- **Don't change:** wording, answer ids or scoring. Pre-select nothing, and never submit an unanswered question.
- When opened from Home, no setup progress bar is shown, like the goal form in C52.

### 6. Household setup wording (section 2 rows 03–05 and 09)
- **Who option:** the "My family" option describes another adult, so call it "Another adult". Keep its description, and keep the "children come later" line.
- **Spouse mode:** the second person is "Your spouse", not "Adult 2". Other modes keep an editable relationship.
- **Tax residence:** under the per-adult "Tax residence" field, add one quiet line: "Your tax residence helps tailor the plan." Keep the options and their meaning.
- **Top bar:** it already says "Getting started", so drop the repeated "Getting started" label above the setup progress bar.
- **"Make it more accurate" on Home:** replace "Your plan starts balanced. Goals and email updates stay off until you choose them." with "You can add goals and change these choices anytime." Keep the three links.

### 7. Plain words in forms and reviews (review item 9 and "Across the app")
Change labels only; don't change fields, values or validation:
- "ISIN" → "Investment code", with the help line "Copy the code from your statement."
- Allocation percentage fields → "Shares (%)", "Company bonds (%)" and "Government bonds (%)", matching their meaning.
- "Coupon rate (%)" → "Interest rate (%)"; "Issuer" → "Issued by".
- "Equity" on broker-import rows → "Shares".
- "Position size" in holding checks → "Share of your investments".
- Import review: "Buy and sell history will set the lot dates" → "Buy and sell history will supply purchase dates."
- Keep EPF, PPF and NPS, and keep the residence status names; they have legal meaning.

### 8. Tap targets everywhere else (review item 9)
- Every `Term` ("What's this?") button, disclosure summary and small link on the screens this card touches is at least 44 × 44 px. That includes the Ideas summaries now at `min-h-9`, the Plan, Holding, Help and Settings disclosures, and the onboarding links.
- Grow the hit area, not the text.
- Keep visible focus on controls. Don't shrink anything that's larger today, and don't add horizontal scroll at 360 px.

### 9. Help opens on what you asked about (review item 10)
- Keep search and all article text.
- Each article shows as a disclosure row (title and one-line summary), closed by default.
- **Arriving with a hash:** a contextual link like `/help#this-week`, or a glossary term, opens that article, scrolls to it and focuses its heading.
- **Search:** it still matches body text and shows the matching titles. Choosing a result opens it.
- **"Back":** it returns to the page the user came from, or Home when Help was opened directly.
- Keep "Show the tour again".

### 10. Settings, Inbox and Money list tidy-ups (section 2 rows 16, 24, 25)
- **Inbox → "Email settings":** opens Settings with the notifications section open and focused, not the top of Settings.
- **Settings summary:** "2 people, only you" becomes "2 people · Access: only you", from the same counts and sharing state. "Statements" becomes "Email statements".
- **Money:** the disclosures under the list ("Add manually", "Previous uploads", "Check an investment", "More details") get one consistent style. Put them in that order.

## Rules
- Front end and copy only (`apps/web`, `copy/en.yaml`, `e2e`). No backend change.
- Never change amounts, weights, rates, tax figures, dates, verdicts, or the rules behind them.
- No new runtime dependencies. Stay within the bundle budget.
- Copy:
  - Sentences of 20 words or fewer, titles of 8 words or fewer, no banned jargon.
  - Put every new string in `copy/en.yaml` and regenerate with `npm run gen:copy`. Never hand-edit generated files.
- WCAG 2.2 AA:
  - Labels on every control.
  - Focus moves to new content (the next question, an opened Help article).
  - Programmatically focused headings show no ring, as in C52.
- Each screen should end up with the same number of elements or fewer. Don't add banners, badges or explanations that weren't asked for.

## Tests
- **Vitest** (update or add):
  - `tests/features/HoldingPage.test.tsx` and `tests/features/HoldingReview.test.tsx`: the order (value, then Our view, then closed sections); one visible reason, with the rest in the closed section; the tax note stays visible; unchecked states are unchanged; the household-currency value and its fallback; "Buying price not recorded".
  - `tests/features/PlanPage.test.tsx`: the "You have" and "Target" rows for the family and for one person; the Add holdings link goes to `/money?add=1`; "Compare amounts"; the "Gold" legend; numbers are unchanged.
  - `tests/features/IdeasPage.test.tsx`: a long name is fully present with no clamp class; the badges sit inside "Why this idea?"; ownership comes before the entry hint; "Other funds to compare" starts closed; no "keep going" when there's no monthly amount; amounts are unchanged.
  - The world-page test, if one exists (add one if not): back link, intro, currency.
  - `tests/onboarding/`: the question stepper (Next, Back keeping answers, focus, Save answers on the last question, Skip saves nothing, exact answer ids); "Another adult", "Your spouse"; the tax-residence line; no repeated "Getting started".
  - `tests/help/`: articles closed; a hash opens and focuses its article; search finds body text; Back goes to the origin, with Home as the fallback.
  - `tests/settings/` and `tests/features/InboxPage.test.tsx`: the Email settings deep link opens and focuses the section; the new summary text.
  - `tests/money/`: the label changes in manual forms; the order and style of the disclosures.
  - `tests/copy.test.ts` and `tests/plain-language.ts`: also scan opened disclosures and optional fields for banned words.
- **e2e:**
  - Update `holding-review.spec.ts`, `ideas-layout.spec.ts`, `onboarding.spec.ts`, `plan-home.spec.ts`, `usability.spec.ts` and `accessibility.spec.ts` for the new layouts and labels.
  - Add `help-navigation.spec.ts`: Help from the Home help icon opens the "this week" article; Back returns Home.
  - At 360 × 740, use long synthetic names, and assert no horizontal overflow (`expectFitsScreen`).

  Codex's sandbox can't start the e2e stack (`spawn EPERM`). Write the specs, run `npx playwright test --list`, and say so in the result file. The reviewer runs the suite and takes before and after screenshots.

## Out of scope
- Anything already done in C52. Visual rebranding. Backend or rule changes. New features.

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

In `e2e`: `npx playwright test --list` must list every spec without errors.
