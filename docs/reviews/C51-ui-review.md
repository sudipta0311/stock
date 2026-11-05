# C51 UI review

This review prioritises the family's five jobs: complete weekly steps, add holdings, understand a holding, check the plan, and explore ideas. The main improvements are clearer action instructions and one recognisable way to add holdings.

Evidence: the 28 supplied screenshots, current PWA source, `copy/en.yaml`, `docs/usability-script.md`, and only plan sections 2.3 and 8. References below use repository-relative paths and current line numbers. Screenshots are synthetic; their names, balances, and recommendations are not findings. This was a static review, without running the app, a database, network requests, or model calls. Accessibility and task times need the proposed implementation checks; they were not measured here.

Capture qualifications: fixed navigation appearing halfway through full-page captures is not a layout defect. Screenshot 06 is a loading state, not evidence of a permanently empty or broken Plan. Screenshot 22 actually shows the scenario page; the world funds are in 21. Screenshot 28 repeats the long Home content, so conclusions about the first viewport use only its upper 740 pixels. Screenshot 26 supports a finding about page length; its scaled text is checked against source.

## 1. Top 10 changes

### 1. Make weekly steps identify the task and explain it correctly

**Now — Home, screenshots 15 and 28:** “Exit”, “Shares”, “Invest in parts”, “How?” and “Done”. The first prominent button marks completion before explaining the step. `apps/web/src/home/HomePage.tsx:664` displays the action kind and category, rather than a specific holding name. More seriously, `actionHowTo` at `apps/web/src/home/HomePage.tsx:417` handles extra loan payments specially, then uses account/member investment instructions without distinguishing a sale. Its member fallback is “Use this person's account to invest {amount} as planned.” (`copy/en.yaml:1111`). This is a code-confirmed possible sell/invest mismatch, not an observed opened sheet.

**Why:** the user cannot reliably identify which shares to sell, or know whether “Done” records a purchase or simply ticks a step. Instructions must match the existing action.

**Change:**

- Use “Sell”, “Sell some”, and “Review before selling” for the corresponding existing action types. Put the supplied holding name on its own wrapping line, followed by the existing person and amount. Resolve identity from the action or an exact security ID match in already-loaded data. If unavailable, use “Holding details unavailable”; never guess from the first action of the same kind.
- Replace “How?” with “View steps”, primary on the first action only. Other actions stay visible and quieter. Replace buy completion with “Record purchase”; keep “Mark done” for status-only completion. Preserve the existing purchase sheet and mutations.
- In a sale sheet use “Complete this sale in your investment account, then mark this step done.” Show the existing quantity, tax information and reason without deriving new values. A review action says “Review the reasons before deciding whether to sell.” Keep account-specific buy guidance confined to buy actions. Missing guidance says “Check the details before taking this step.”
- Replace raw parameter labels and JSON under “Why?” with an explicit display-label map; omit unsupported technical fields. The current raw rendering is at `apps/web/src/home/HomePage.tsx:1291`.
- When actions exist, move monthly amount editing below them into “Monthly investment”, closed initially. Keep it open when there are no steps and the amount is unset. Preserve the three-action limit and server ordering.

**Files:** `apps/web/src/home/HomePage.tsx`, `apps/web/src/lib/labels.ts`, `copy/en.yaml`.

**Size: M. Risk:** confusing a review with a sale, mismatching a security, or breaking purchase recording, stale-action recovery and view-only permissions. Tests must cover each action kind and prove payload amounts and dates are unchanged.

### 2. Give adding holdings one clear entrance, including from Inbox

**Now — screenshots 08, 12, 16, 23 and 24:** “Add or update”, “Choose what you have”, “What else do you have?”, “Imports”, and “Upload your CAS statement” describe overlapping paths. The card records an actual failed attempt to upload an INDmoney file from Inbox. The current Money sheet says “Upload a statement”, not the card's more explicit broker wording (`apps/web/src/money/MoneyMapPage.tsx:1571`, `copy/en.yaml:962`). The empty-state button only scrolls to a closed disclosure (`apps/web/src/money/MoneyMapPage.tsx:1455`); the sheet's manual-add button correctly opens it at line 1598. Inbox's upload recovery appears on a statement error, but its empty statement component disappears (`apps/web/src/inbox/GmailStatementInbox.tsx:174`).

**Why:** users think about files or deposits they have, not statement formats or the distinction between two inboxes. A scroll without opening the chooser looks unresponsive.

**Change:**

- Use “Add holdings” for Money's single primary button and sheet title. Keep the existing sheet, with “Upload a file” as its primary choice and “Add manually” secondary. Move “Get statements from Gmail” to a quiet link below these choices. Under upload, show “Broker files and investment statements.” This replaces ambiguous copy, not a new dashboard panel.
- Route the empty-state entry to the same sheet; remove its separate chooser button. Use “Add your first holding” and “Upload a file or add an item manually.” as empty-state text. Keep one prominent entry, placed before the empty-state body.
- “Add manually” opens and focuses the existing chooser. Rename its summary “Add manually”, move “Quick add” below the tiles into a closed section, and rename history “Previous uploads”. Keep the existing multi-item workflow.
- On Import use “Upload holdings” as heading. Options: “Broker file (Excel or CSV)”, “Mutual fund statement (PDF)”, and “Combined investment statement (PDF)”. Keep provider names in their existing “How to get it” help. Reduce each choice to a compact row, preserving selection and help controls. Keep “Choose file” and “Upload file” visible without unnecessary introductory cards.
- Add one quiet “Upload a file” link beside Inbox's existing introductory text, above conditional content. It leads to the same import screen even when there are no emails. Preserve the selected format in URL state so returning from review restores that choice; retain current file-format detection. Do not add a second uploader to Inbox.
- Treat Import as part of Money in the active bottom-tab state; title it “Upload holdings”. Retain the existing `/import` URLs.

**Files:** `apps/web/src/money/MoneyMapPage.tsx`, `apps/web/src/imports/ImportPage.tsx`, `apps/web/src/imports/ImportHistorySheet.tsx`, `apps/web/src/inbox/InboxPage.tsx`, `apps/web/src/inbox/GmailStatementInbox.tsx`, `apps/web/src/app/AppShell.tsx`, `apps/web/src/app/BottomNavigation.tsx`, `copy/en.yaml`.

**Size: M. Risk:** source selection, PDF share-target entry, keyboard focus after the sheet closes, member/account attribution, and duplicate uploads. Preserve review-before-confirmation, feature availability and read-only restrictions. Do not promise support for every broker export variant.

### 3. Make saving a deposit and viewing it two understandable states

**Now — screenshots 13, 14 and 19:** “Changes save automatically”, “Save now”, “Saved”, “Finish Money Map setup”, and “View asset” coexist. On the detail page, “View asset” navigates to that same asset URL (`apps/web/src/money/AssetFormPage.tsx:1011`); `AssetDetailPage` reuses the editor at line 1211. “Close this asset” is a prominent red button below “Edit history”. The eligibility box says “New money can't go in with this tax residence.” followed by “You can still record an existing account.”

**Why:** users cannot tell when they can leave, whether clicking again is required, or whether adding an existing deposit is prohibited. Opening an item to see its value instead opens a long editable form.

**Change:**

- Keep autosave. Use “Save and continue” during a multi-item workflow and “Save and return” for a single item. Validate and wait for the latest save before leaving; replace the redundant two-button row. Keep “Saving…”, “Saved”, and “Saved on this device · waiting for connection” as truthful status text. A queued write must not look synced.
- On detail entry show the existing value, maturity value/date, bank and owner first. Use one primary “Edit details” button to reveal the existing form. Remove the self-link “View asset”. Return from editing to the same summary after saving.
- Place the existing “Whose is this?” selector before the four deposit fields when a household has multiple adults; preserve its default and stored member ID. Show the eligibility note adjacent to that person. Lead the existing restriction with “You can still record an existing account.” Use “Adding new money” for its heading. Preserve the backend's restriction and conditions verbatim in meaning; this review does not judge their financial correctness.
- Collapse “Edit history”. Put a quieter “Mark as closed” control there, retaining confirmation. Confirmation: “This removes the item from your current totals. Its history stays saved.” Buttons: “Mark as closed” and “Keep it”. This records a status; it does not close a bank account.
- Render the existing maturity date in the app's readable date format. Do not change the underlying date or increase its claimed precision.

**Files:** `apps/web/src/money/AssetFormPage.tsx`, `copy/en.yaml`.

**Size: M. Risk:** autosave races, leaving with unsaved edits, offline replay, duplicate creation, multi-item progression, and accidental closure. Reuse existing validation, save queue and close endpoint; test last-edit-then-immediate-continue explicitly.

### 4. Make first use promise the next useful step

**Now — screenshots 02, 05, 06, 07 and 08:** “Set up my Money Map” leads to “Your plan is ready” before holdings are recorded. “View your plan” can then reach an empty Plan. The real empty state already exists: “Add what you have to see your plan” (`apps/web/src/plan/PlanPage.tsx:529`). Screenshot 06 only shows skeletons, and `ScreenLoading` supplies its label to assistive technology without visible loading copy (`apps/web/src/features/api.tsx:63`).

**Why:** “ready” overstates what an account with no holdings can show. A blank skeleton does not explain whether the user needs to act or wait.

**Change:**

- Welcome action: “Get started”. After household basics use “You're ready to start”, with “Add what you own to see your plan.” Primary: “Add holdings”, opening the shared Money sheet from item 2. Secondary: “Explore the app”, leading to Home. Preserve skippable setup and the existing defaults.
- Keep the actual empty Plan message. Its “Add holdings” action should open the chooser, not merely navigate to Money and require another search.
- On Home with no recorded holdings and no proposed steps, use “Add holdings” as primary. Keep “Set a monthly amount” as a quieter expandable action; preserve any existing proposed steps if present. Never infer emptiness from a zero net balance: use loaded record counts, including imported holdings and loans.
- Give skeletons the same visible label as their accessible status, such as “Loading your plan…”. Keep cached content while refreshing. On a genuine request failure show the existing error plus “Try again”, retrying the query rather than asking the user to wait indefinitely.
- Keep the existing health-cover note if supplied. Do not assert it is wrong from the screenshot, suppress it based on an invented rule, or call an empty account “on track”.

**Files:** `apps/web/src/onboarding/OnboardingPage.tsx`, `apps/web/src/home/HomePage.tsx`, `apps/web/src/plan/PlanPage.tsx`, `apps/web/src/money/MoneyMapPage.tsx`, `apps/web/src/features/api.tsx`, `copy/en.yaml`.

**Size: M. Risk:** onboarding completion, refresh versus first-load state, missing-feature responses, imported-only accounts, and shared-device drafts. Reuse existing query data and distinguish unknown/loading from known empty.

### 5. Put a holding's recommendation next to its value

**Now — screenshots 17 and 18:** “Value”, “Profit or loss”, “Cost not available”, then a closed “Details” row, then “Our view”, “Add more” or “Sell”, and “How we check”. The optional identity details precede the main answer (`apps/web/src/money/HoldingPage.tsx:251`). Up to three reasons plus a tax note can be visible (`apps/web/src/money/HoldingPage.tsx:106`).

**Why:** the user's question is what this holding is worth and what the suggestion means. Missing buying cost occupies a prominent cell without explaining its effect; technical details interrupt the answer.

**Change:** keep the holding name and value first, followed immediately by “Our view”, the existing verdict, one leading reason, the existing tax note and check date. Move the remaining reasons and checks into a single “Why this suggestion?” disclosure; move code and exact quantity to “Holding details” below it. Do not rewrite verdicts or fabricate supporting evidence. Change “Cost not available” to “Buying price not recorded”, with “Profit or loss needs your buying price.” in details. Add a quiet “Back to Money” link above the name, using the existing route. Do not add a trading button to an informational review.

**Files:** `apps/web/src/money/HoldingPage.tsx`, `copy/en.yaml`.

**Size: S. Risk:** hiding a material tax condition, losing unchecked/unknown states, and misleading reason order. Keep tax qualifications visible and use the supplied reason order. This does not change how the existing review is selected.

### 6. Show the plan's current position beside its target

**Now — screenshot 20:** “Your money vs your plan”, two thin coloured bars, “Your safe money target is ₹13.12 L.” and “Add safe money”. Exact comparisons are behind “Show amounts”. The button opens a fixed-deposit recording form (`apps/web/src/plan/PlanPage.tsx:508`), not a way to buy or transfer money.

**Why:** a target alone leaves the user to find their current amount. “Add safe money” can be understood as an investment instruction when it actually records an existing item.

**Change:** put the safe-money answer before the chart and display “Recorded: {current}” and “Target: {target}”, using the corresponding existing values for the same selected scope. Keep the existing gap sentence. Rename the recording CTA “Update your holdings” and send it to the shared chooser, rather than assuming a fixed deposit. Explain it with “Add items you already own.” Retain one primary action. Rename “Show amounts” to “Compare amounts”; keep the chart's text alternative and expanded comparison. Replace the narrow label “Other holdings” with “Gold” only for the existing gold row, not for arbitrary assets.

**Files:** `apps/web/src/plan/PlanPage.tsx`, `copy/en.yaml`.

**Size: M. Risk:** mixing family and person figures, changing gap calculations accidentally, or implying a recording action executes an investment. Reuse existing amounts, gap logic and currency formatters; do not introduce a new target, score or rule.

### 7. Let an idea's name and existing ownership lead

**Now — screenshot 21:** company names shrink to “S…” alongside a symbol and “Higher confidence”. “Buy in 3 parts over 3 months” precedes “You already own about DKK 2,430 of this through your funds.” `ResearchIdea` uses a one-line clamp beside a non-shrinking metadata row (`apps/web/src/ideas/IdeasPage.tsx:624`). A title attribute does not provide reliable touch access to the name. The world, S&P 500 and Nasdaq fund rows have similar visual weight despite “Your plan uses this one”.

**Why:** the reader must recognise the company before judging its idea or whether they already own it. A confidence badge should not consume the space needed to identify an investment.

**Change:** give the company name the full row, naturally wrapping; put the symbol below it and move confidence/sector/valuation into “Why this idea?”. Put the unchanged existing-ownership sentence before the entry hint. Keep “You already own enough ({count})” collapsed and preserve the supplied classification. Keep the plan's selected world-fund row visible; group alternative fund rows under “Other funds to compare”, closed initially. Preserve their broker lookup details and actual product names. Do not change entry parts, months, recommendations, prices, ownership amounts or thresholds.

**Files:** `apps/web/src/ideas/IdeasPage.tsx`, `copy/en.yaml`.

**Size: M. Risk:** overly long names, hidden risk information, loss of broker search names, and duplicate visibility of already-owned ideas. `e2e/tests/ideas-layout.spec.ts:45` currently asserts a one-line clamp; replace that implementation-specific expectation with readable-name and no-overflow checks.

### 8. Make optional goals and risk questions finish cleanly

**Now — screenshots 09, 10 and 11:** Home's “Add a goal” opens “Getting started” and an onboarding progress bar. The goal form offers “Add Emergency fund”, “Continue” and “Skip for now”. Only submitting the form creates the goal (`apps/web/src/onboarding/OnboardingPage.tsx:603`); Continue merely navigates (`apps/web/src/onboarding/OnboardingPage.tsx:1086`). The code already returns Home-opened tasks to Home (`apps/web/src/onboarding/OnboardingPage.tsx:461`); the issue is misleading framing and an unsaved draft, not a forced walkthrough. The risk page displays all six questions and 24 answers before “Save answers”.

**Why:** a user can enter a goal and press the most prominent button without saving it. Optional refinement feels like restarting account setup. A long questionnaire hides its save/skip actions.

**Change:**

- For Home-entered tasks remove the setup progress and use “Add a goal” or “Your comfort with risk” as the task title, retaining “Back to Home”. Initial onboarding keeps its own progress.
- While a goal draft has an amount, make “Save goal” the single primary action. Keep the entered values on validation or save failure. After success show “Goal saved”, primary “Back to Home” for a Home entry or “Continue” during setup, and a quiet “Add another goal”. When no draft exists, offer “Skip for now” rather than two equivalent navigation buttons. Do not change the target, currency choices, default year, or priority rules.
- Present one existing risk question at a time with “Question {number} of {count}”, “Next”, “Back”, and a quiet “Skip for now”. The last primary is “Save answers”. Retain question/answer wording, answer IDs, saved drafts and scoring; do not preselect answers or submit incomplete answers. Keep controls reachable without traversing all six questions.

**Files:** `apps/web/src/onboarding/OnboardingPage.tsx`, `apps/web/src/onboarding/OnboardingFrame.tsx`, `copy/en.yaml`.

**Size: M. Risk:** duplicate goals, lost edits, changing answer semantics, focus placement between questions, and optional-task return paths. Do not add new financial questions or alter consents.

### 9. Make small disclosure controls usable and remove remaining form jargon

**Now — screenshots 08, 20, 21 and 23:** small “What's this?”, “Show amounts”, “Why this idea?” and “How to get it” controls. `Term` has no minimum target dimensions (`apps/web/src/ui/primitives.tsx:152`); Ideas uses `min-h-9` for several summaries (`apps/web/src/ideas/IdeasPage.tsx:670`), and import choices use `min-h-10` (`apps/web/src/imports/ImportPage.tsx:366`). At the normal 16px root size these are 36px and 40px, below this card's 44px requirement. Code also reveals unwrapped “ISIN” and “Equity allocation (%)” labels in manual forms (`apps/web/src/money/labels.en.ts:31`, line 110), plus “Buy and sell history will set the lot dates” in import review (`copy/en.yaml:1378`). These were not visible in the supplied open screens.

**Why:** help is difficult to tap precisely, and optional fields still expose terms the product explicitly avoids. Collapsing jargon does not make it plain once opened.

**Change:** give interactive disclosure and help targets at least 44 × 44px, preferably the existing 48px button height. Use inline-flex/block layouts with spacing so targets do not overlap or create horizontal scrolling. Preserve labels, visible focus, expanded state and the existing colour palette. Use “Investment code” for the bond identifier, “Shares (%)”, “Company bonds (%)” and “Government bonds (%)” for the corresponding percentage fields, and “Buy and sell history will supply purchase dates.” for the import note. Help text for the code: “Copy the code from your statement.” These changes rename fields only.

**Files:** `apps/web/src/ui/primitives.tsx`, affected summaries in `apps/web/src/home/HomePage.tsx`, `apps/web/src/ideas/IdeasPage.tsx`, `apps/web/src/imports/ImportPage.tsx`, `apps/web/src/onboarding/OnboardingPage.tsx`, `apps/web/src/plan/PlanPage.tsx`, `apps/web/src/help/HelpPage.tsx`, `apps/web/src/money/AssetFormPage.tsx`, `apps/web/src/settings/SettingsPage.tsx`; labels in `apps/web/src/money/labels.en.ts` and `copy/en.yaml`.

**Size: M. Risk:** expanded help increasing card height, focus rings being clipped, and field-label selectors breaking. This is not a claim of measured contrast failure; check WCAG 2.2 AA contrast and focus in both themes during implementation. Do not shrink existing 48px inputs or 64px navigation targets to compensate.

### 10. Make Help show the requested answer first

**Now — screenshot 26:** “Help and guide” renders an extremely long page. `apps/web/src/help/HelpPage.tsx:110` renders every article's steps, example, glossary and related links expanded. Search exists; a hash link merely scrolls to an article (`apps/web/src/help/HelpPage.tsx:62`).

**Why:** a tap on contextual help lands in a manual with many unrelated tasks. Users must keep their place while navigating a long page.

**Change:** retain search and existing article content. Show article titles as disclosure rows with all bodies closed initially. A contextual hash opens its article (or the containing glossary article), then focuses and scrolls to the requested heading/term. Search matches hidden body text as it already does; show matching article titles and summaries, with “Read steps” where expansion needs an explicit label. Open the selected result on activation. Preserve related links and the quiet “Show the tour again” action. Make “Back” return to a known originating app page, with “Back to Home” as the direct-entry fallback.

**Files:** `apps/web/src/help/HelpPage.tsx`, `apps/web/src/help/context.ts` if needed for origin handling, contextual links in `apps/web/src/app/AppShell.tsx` and `apps/web/src/ui/primitives.tsx`, `copy/en.yaml`. Existing help article copy needs no rewrite.

**Size: M. Risk:** broken deep links, search hiding the active target, browser history loops, and focus landing inside a closed disclosure. Test glossary anchors as well as article anchors.

## 2. Screen by screen

These are smaller findings or qualifications in addition to the ranked changes. Numbers refer to the supplied image order.

| Screens | Further finding and concrete treatment |
|---|---|
| 01 — Login | “Development token” is specifically guarded by `mode === "dev"` in `apps/web/src/app/AuthPages.tsx:109`. Do not redesign production sign-in based on this development capture. The production sign-in flow was not demonstrated. |
| 02 — Welcome | Keep the short reassurance about approximate amounts. The title and one action already fit a phone; change only the action wording in item 4. |
| 03–04 — Household | “My family” sounds like it includes children now, but its description says another adult and the intro says children come later. Use “Another adult” for this option. In spouse mode use “Your spouse” instead of “Adult 2”; retain an editable relationship for other modes. Keep tax residence per adult and optional details closed. Source: `apps/web/src/onboarding/OnboardingPage.tsx:735`, line 751. |
| 05 — Ready | “Add a statement or a few items first” links to `/money?checklist=open` (`apps/web/src/onboarding/OnboardingPage.tsx:1257`). The inspected Money page has no query handler to open that disclosure. Route it to the shared add entry from items 2 and 4. |
| 06 — Plan loading | Add visible loading status; do not claim a stuck request from one screenshot. Validate settled empty, populated, error and cached-refresh states separately. |
| 07 — Empty Home | “Money: ₹0” is ambiguous when nothing is recorded. With confirmed empty records use “No holdings added yet” for that summary link. Keep genuine zero or negative recorded totals as numbers. |
| 08 — Empty Money | “0 items” and an oversized empty illustration compete with the useful add action. Keep the count for consistency but reduce empty-card padding enough for its explanation to fit the first viewport. Remove the duplicate chooser action as in item 2. |
| 09 — Optional Home section | “Goals and email updates stay off” groups saving goals with notifications. Replace the paragraph with “You can add goals and change these choices anytime.” Keep the three existing links; do not add more settings here. |
| 10 — Goal | Keep rough-number help, optionality and existing currency choices. Fix save/navigation first; do not add DKK goal support in this review. The currency and year remain explicit. |
| 11 — Risk | The short intro contrasts with a very long page. Use item 8's question progression. Do not simplify the financial meaning of the supplied questions or options. |
| 12 — Manual chooser | “Money Map 0% complete” resembles an assessment of financial completeness. Omit this percentage from the first-use chooser; retain selected-item count and workflow progress. “Add manually” should put tiles first and Quick add behind a disclosure. Source: `apps/web/src/money/MoneyMapPage.tsx:1493`. |
| 13–14 — Deposit | Retain the four essential fields and rate-from-certificate help. Reorder fields in the frontend to bank, amount, rate, maturity, matching the task's natural sequence; do not edit generated form definitions. Replace the ISO date presentation in the saved summary, retaining the saved date. Source: `apps/web/src/money/AssetFormPage.tsx:958`, line 1057. |
| 15 and 28 — Home with data | Three actions fit before the monthly editor begins, but the visible first action lacks a holding identity. Keep three rather than adding a carousel or hiding two. Use item 1's first action emphasis. Ignore the full-page navigation placement. |
| 16 — Money with data | Names such as “Fixed deposit at Syn…” lose identifying text beside amounts. Allow two wrapping lines in holding/asset names; retain amount columns and row-wide links. Use singular “1 unit”, not “1 units”. Source: `apps/web/src/money/MoneyMapPage.tsx:469`, line 1408. |
| 17–18 — Holding | INR-only value presentation differs from Money's household currency presentation (`apps/web/src/money/HoldingPage.tsx:229`). Reuse `MoneyValue` for INR-denominated API totals, preserving the existing rate and unavailable-rate fallback. Keep original-currency prices distinct from converted totals. |
| 19 — Asset details | The all-red closure button makes an infrequent destructive step unusually prominent. Use item 3's summary/edit distinction and quiet entry to confirmation. Do not remove the confirmation or edit history. |
| 20 — Plan | “Other holdings” labels the chart's gold segment, while Money uses “Other” for miscellaneous items. Label the known gold row consistently as “Gold”. Keep “Cash” distinct from the wider safe-money grouping. Source: `apps/web/src/plan/PlanPage.tsx:148`. |
| 21 — Ideas and world funds | “Monthly buying: keep going.” appears beside a “Set your monthly amount” link. When that existing unset-amount condition is true, omit the encouragement to continue; retain the setup link. Do not infer that a purchase was made. Source: `apps/web/src/ideas/IdeasPage.tsx:879`, line 950. |
| 22 — World scenarios | Add “Back to Ideas”. Replace the intro with “These examples show possible changes in your holdings.” Keep “up”, “down” and both current/target results. Use shared household currency presentation for existing INR result amounts, preserving signs and scenario definitions. Source: `apps/web/src/ideas/WorldPage.tsx:21`, line 123. |
| 23 — Import | Two privacy messages repeat the same reassurance. Merge into one beneath Upload: “Your file and password are used for this upload only. They are not stored.” Retain the existing factual promise; do not make broader claims about Gmail storage. Source: `apps/web/src/imports/ImportPage.tsx:594`, line 601. |
| 24 — Inbox | “Review email settings” leads to the top of Settings, where “App and privacy” is closed. Link to the existing notification section, open its parent, and focus its heading. Use “Email settings” as the link text. Source: `apps/web/src/inbox/InboxPage.tsx:120`, `apps/web/src/settings/SettingsPage.tsx:1265`. |
| 25 — Settings | “2 people, only you” mixes family membership and login access. Use “2 people · Access: only you” for this state, preserving the actual counts and sharing state. In the collapsed summary replace “Statements” with “Email statements” to distinguish Gmail settings from file upload. Source: `copy/en.yaml:690`. |
| 26 — Help | Preserve search across body text when collapsing articles. Do not rewrite or remove useful help to shorten the page; change its presentation. |
| 27 — Account menu | Account identity, edit permission, Settings, Help and Sign out form a clear hierarchy. Keep the menu. The captured synthetic email wrapping and overlay position do not establish a defect. |

## 3. Across the app

### Navigation and naming

Keep Home, Money, Plan, Ideas and Inbox. They match the five jobs well enough; changing all tabs would add learning cost. The observed upload failure warrants a shared upload entry and recovery link, not renaming Inbox or adding a sixth tab. `/import` currently matches none of the bottom-tab paths (`apps/web/src/app/BottomNavigation.tsx:4`); highlight Money for import and review pages.

Use “Money” for page navigation and “holdings” for adding owned items. “Money Map” can remain a descriptive concept in help, but buttons such as “Your Money Map”, “Finish Money Map setup” and “View asset” should name the destination or task. Do not introduce “portfolio” as another name. Keep the Family/person picker, but make ownership explicit in the deposit form rather than treating the viewing filter as an ownership label (screenshots 13–14; `apps/web/src/money/AssetFormPage.tsx:894`).

### Jargon and copy

The visible main screenshots largely avoid the AGENTS.md banned list. Technical keys in code are not necessarily displayed words: internal `RNOR`, `STP` and other action codes do not alone establish a violation. “Returning NRI” is the displayed residence option (`copy/en.yaml:390`); “ISIN” on the bond form and “allocation” on NPS/insurance fields are actual copy findings. “lot dates” in import review is another confirmed occurrence. Item 9 gives replacements.

Other finance words remain: “Equity” on broker import rows (`copy/en.yaml:1300`), “Coupon rate (%)” and “Issuer” in bond forms (`apps/web/src/money/labels.en.ts:28`), and “Position size” in holding checks (`copy/en.yaml:928`). Use “Shares”, “Interest rate (%)”, “Issued by”, and “Share of your investments”, respectively, where those existing fields have that meaning. “Tax residence” is important enough to retain, with a quiet explanation: “Your tax residence helps tailor the plan.” Do not relabel residence statuses in a way that changes legal meaning. Product acronyms such as EPF, PPF and NPS can retain their familiar names, with existing help available; avoid putting an explanation under every tile.

“Net worth”, “Money mix” and “Safe money” already use `Term` appropriately. Retain them with usable help targets. Use “Why this suggestion?” for a holding decision, “Why?” for a short highlight explanation, “View steps” for instructions, and “Holding details” for identifiers. These are different tasks; do not flatten them into an uninformative universal “Details”. Unknown API parameters should never become labels by replacing underscores (Home item 1).

All replacement UI sentences in this report are intended to stay within 20 words, and titles within eight. Existing quoted copy is evidence, not approved replacement text. Dynamic names must wrap rather than be shortened to satisfy a word count. Keep actual security names and broker search identifiers intact.

### Repeated actions and first use

The app already makes secondary buttons quieter. The larger problem is competing meanings: auto-save versus Save now/Finish, Add goal versus Continue/Skip, and Add or update versus Choose what you have. Merge these paths as described rather than adding introductory banners. On reading screens such as a holding or world scenarios, a primary trade action is not required merely to fill space.

Maintain independent empty, pending, error, offline and populated states. An unset monthly amount is not proof of no holdings; zero net worth is not proof of no records. Screenshot 06 cannot establish request duration. Preserve the existing offline and unknown-price language rather than displaying zeros for unavailable values. Empty first use should lead to recording holdings; genuinely quiet weeks should remain a successful outcome.

### Numbers, dates and notes

INR summaries in screenshots 16 and 20 use lakh abbreviations; DKK uses international grouping. Keep those formatters and the primary/secondary currency distinction. Full precision in deposit inputs and detail values is appropriate. The raw maturity date is a presentation inconsistency, not permission to change maturity assumptions. Holding and scenario screens should use the same selected display currency for totals, using existing conversions. Do not convert a reference price silently or use a new exchange rate.

The Home health note is one note, not a warning overload. `Highlights` already shows two ranked items and collapses the remainder (`apps/web/src/features/Highlights.tsx:98`); keep that pattern and backend order. Holding review can expose three reasons plus tax, so item 5 limits visible reasons while retaining the material tax qualification. Import duplicates privacy notes; merge them. On a deposit, keep the eligibility restriction near its owner and make optional valuation detail secondary. Do not suppress real restrictions or invent new alerts to satisfy a visual count.

### Readability and accessibility at 360px

Full-width inputs, large tile controls, labelled navigation and the existing Radix sheet are useful foundations (`apps/web/src/ui/primitives.tsx:40`, `apps/web/src/ui/sheet.tsx:20`). The main known gaps are small disclosures and names squeezed by metadata. Enlarge their actual hit areas, not just their icons. Test at 360 × 740 and with text zoom; avoid horizontal scroll and keep focus visible above fixed navigation. Retain the app's safe-area bottom padding (`apps/web/src/app/AppShell.tsx:63`).

For keyboard users, sheet dismissal must restore focus; opening a chooser must focus its heading; next-question actions must focus the new question; hash links must open before focusing hidden content. Selection cannot depend on colour alone. Check normal text at 4.5:1 and large text/non-text controls at the applicable 3:1 thresholds, in both themes. These are implementation checks, not measured pass/fail claims from compressed images. Preserve the colour scheme, typeface and logo.

### What existing checks miss

`e2e/tests/usability.spec.ts:39` goes directly to `/import` before timing upload; it does not test whether the user can find upload from Money or Inbox. Its Home check verifies the “This week” heading, not recognition of a specific action. `e2e/tests/ideas-layout.spec.ts:45` expects truncation rather than testing whether the name is readable. `apps/web/tests/plain-language.ts:10` excludes closed details from the default visible-text scan. Future checks should cover opened optional fields and disclosures as well as landing states. Automated timings are regression signals; use the usability script with a person to assess comprehension.

## 4. Keep as is

- **Five labelled bottom tabs and the compact account menu:** screenshots 07, 25 and 27. Keep Settings behind the avatar, the Family/person picker and the clear view-only/edit distinction.
- **Calm visual language:** neutral cards, one accent colour, generous inputs and no decorative dashboard charts. This review requires no rebranding or runtime dependency.
- **Short, skippable household setup:** screenshots 02–05. Keep optional financial questions out of the basic setup and preserve the existing per-adult tax-residence choices.
- **Simple deposit essentials and autosave:** screenshots 13–14. Rough amounts, rate-from-certificate help, optional details and calculated value feedback are useful; clarify their save/navigation presentation.
- **Money grouped by type:** screenshot 16. Keep totals, household currency information and row-wide links. Improve name space without removing context.
- **Plain holding verdicts and dated evidence:** screenshots 17–18. Keep “Keep”, “Add more”, “Sell”, unknown-data states, tax qualifications and check dates; change their layout, not decisions.
- **Concise plan explanation and expandable comparisons:** screenshot 20. Preserve existing target/current calculations and text alternatives to charts.
- **Ownership context in Ideas:** screenshot 21. Keep the through-funds amount, expandable source funds and collapsed already-enough group. They directly support avoiding duplicated purchases.
- **Scenario direction words:** screenshot 22. “up” and “down” communicate better than colour or a bare signed amount. Keep current-versus-target comparisons.
- **Review before confirming an upload and clear privacy copy:** screenshot 23 and `apps/web/src/imports/ImportPage.tsx:339`. Retain the stages and confirmation; combine duplicated reassurance.
- **Two ranked Home highlights with Why/Hide:** screenshots 07 and 15; `apps/web/src/features/Highlights.tsx:98`. Do not restore a long list of policy notes.

## 5. Proposed implementation

These are two candidate implementation cards for reviewer selection, not a claim that changes are already accepted. Do not create or execute them in C51. All proposed work stays in `apps/web`, `copy/en.yaml` and `e2e`. No backend change is required for the bounded changes above; unresolved financial correctness or unavailable identity data must not be solved by inventing values or rules. No runtime dependencies or edits to earlier migrations are proposed.

### Proposed C52 — Clear weekly steps and adding holdings

**Include:** ranked items 1–4; the upload privacy merge, Import/Money navigation state, manual chooser ordering, deposit field order, explicit owner context and “Previous uploads”. Include the shared target sizing from item 9 needed on these touched screens. This delivers the largest gains to weekly action completion and adding money before broader polishing.

**Production files:**

- `apps/web/src/home/HomePage.tsx`, `apps/web/src/lib/labels.ts`.
- `apps/web/src/money/MoneyMapPage.tsx`, `apps/web/src/money/AssetFormPage.tsx`.
- `apps/web/src/imports/ImportPage.tsx`, `apps/web/src/imports/ImportHistorySheet.tsx`.
- `apps/web/src/inbox/InboxPage.tsx`, `apps/web/src/inbox/GmailStatementInbox.tsx`.
- `apps/web/src/app/AppShell.tsx`, `apps/web/src/app/BottomNavigation.tsx`, `apps/web/src/onboarding/OnboardingPage.tsx`, `apps/web/src/plan/PlanPage.tsx`, `apps/web/src/features/api.tsx`, `apps/web/src/ui/primitives.tsx`.
- `copy/en.yaml`; regenerate its outputs with the existing generator, never by hand.

**Vitest additions/updates:**

- `apps/web/tests/features/HomePage.test.tsx`: correct instructions for buy, sale, partial sale, review, payment and missing identity; same action ordering and values; correct completion handler; monthly editor collapsed only when appropriate; preserve existing purchase/stale-action tests.
- `apps/web/tests/money/money-map.test.tsx`, `apps/web/tests/imports/ImportPage.test.tsx`, `apps/web/tests/features/InboxPage.test.tsx`, `apps/web/tests/features/GmailStatementInbox.test.tsx`: every add entry opens the same chooser; manual disclosure opens/focuses; empty Inbox upload works; selected broker format survives navigation; PDF shared-file entry and review still work.
- `apps/web/tests/money/assets.test.tsx`, `apps/web/tests/money/offline-queue.test.tsx`: latest-edit save-and-leave, invalid form stays open, failed save retains draft, queued versus synced state, member identity preserved, multi-item next step, detail summary/edit and confirmed closure.
- `apps/web/tests/onboarding/OnboardingPage.test.tsx`, `apps/web/tests/features/PlanPage.test.tsx`, `apps/web/tests/app/BottomNavigation.test.tsx`: first-use labels and chooser destination, settled empty versus imported-only accounts, genuine zero totals, visible loading/error retry, Money active on import/review. Update `apps/web/tests/copy.test.ts` and affected role/name queries.

**E2E additions/updates:** use synthetic fixtures and controlled responses. Update `e2e/tests/plan-home.spec.ts`, `e2e/tests/bought-it.spec.ts`, `e2e/tests/fd.spec.ts`, `e2e/tests/broker-import.spec.ts`, `e2e/tests/cas-import.spec.ts`, `e2e/tests/onboarding.spec.ts`, `e2e/tests/household.spec.ts`, and shared helpers in `e2e/support/app.ts`. Add the observed route: start on Inbox, find Upload a file, select a synthetic INDmoney export, review, confirm, and locate its holding. Also cover Money → upload for HDFC Sec, Zerodha and a Danish broker using supported synthetic fixtures; preserve mapping/review for unfamiliar variants. Exercise manual deposit entry from an empty account and a spouse-owned deposit. No owner's file may be used.

Extend `e2e/tests/usability.spec.ts` to start upload timing before finding its entry, and assert an identifiable action and reachable View steps/Mark done controls at 360 × 740. Retain the script's targets: deposit under 30 seconds, statement review under two minutes, weekly action under ten seconds, plan target under 15 seconds, and spouse setup under one minute. Automated completion alone must not be reported as a human comprehension test. Update `e2e/tests/accessibility.spec.ts` for open sheets, chooser focus and save/error states.

### Proposed C53 — Understand holdings, plans and ideas

**Include:** ranked items 5–10; remaining smaller findings in section 2, including task-focused optional setup, long Money names, singular units, notification deep link, Settings summaries, world-page return/currency presentation and non-duplicated notes. Item 4's first-use flow is a dependency; do not redesign it again.

**Production files:**

- `apps/web/src/money/HoldingPage.tsx`, `apps/web/src/money/MoneyMapPage.tsx`, `apps/web/src/money/labels.en.ts`, and field-label use in `apps/web/src/money/AssetFormPage.tsx`.
- `apps/web/src/plan/PlanPage.tsx`, `apps/web/src/ideas/IdeasPage.tsx`, `apps/web/src/ideas/WorldPage.tsx`.
- `apps/web/src/onboarding/OnboardingPage.tsx`, `apps/web/src/onboarding/OnboardingFrame.tsx`.
- `apps/web/src/help/HelpPage.tsx`, `apps/web/src/help/context.ts`, `apps/web/src/app/AppShell.tsx`, `apps/web/src/ui/primitives.tsx`.
- `apps/web/src/settings/SettingsPage.tsx`, `apps/web/src/inbox/InboxPage.tsx`; remaining disclosure sizing in `apps/web/src/home/HomePage.tsx` and `apps/web/src/imports/ImportPage.tsx`.
- `copy/en.yaml`. Reuse `apps/web/src/household/currency-view.tsx` and existing formatting helpers without changing conversion logic. No help-article content change is needed.

**Vitest additions/updates:**

- `apps/web/tests/features/HoldingPage.test.tsx`, `apps/web/tests/features/HoldingReview.test.tsx`, `apps/web/tests/features/PlanPage.test.tsx`: verdict near value, leading reason plus visible tax qualification, closed detail groups, unchecked states, currency selection/fallback, recorded/target scope, and correct recording CTA. Assert unchanged numeric strings supplied to the formatters.
- `apps/web/tests/features/IdeasPage.test.tsx`: name and ownership precede the entry hint, confidence moves into details, already-enough items remain excluded, alternatives start closed, and monthly encouragement is absent when unset. Keep actual entry timing and ownership values unchanged.
- `apps/web/tests/onboarding/OnboardingPage.test.tsx`: save a filled goal before leaving, avoid duplicate submissions, preserve failed drafts, add another, correct Home return, question navigation/focus and exact existing answer IDs; skipping must not submit incomplete answers.
- `apps/web/tests/help/HelpPage.test.tsx`, `apps/web/tests/help/context.test.ts`, `apps/web/tests/settings/SettingsPage.test.tsx`, `apps/web/tests/features/InboxPage.test.tsx`: closed help bodies, search within bodies, article/glossary hash opening, Back fallback, and notification-section opening/focus.
- `apps/web/tests/copy.test.ts`, `apps/web/tests/plain-language.ts`, relevant manual-form tests: render optional fields and opened disclosures to catch banned labels; verify replacement sentence/title lengths. Add a focused `apps/web/tests/ui/primitives.test.tsx` if needed for Term semantics and expanded state. Preserve existing accessibility assertions.

**E2E additions/updates:** update `e2e/tests/holding-review.spec.ts`, `e2e/tests/ideas-layout.spec.ts`, `e2e/tests/plan-home.spec.ts`, `e2e/tests/onboarding.spec.ts`, `e2e/tests/usability.spec.ts`, and `e2e/tests/accessibility.spec.ts`; add `e2e/tests/help-navigation.spec.ts` for contextual Help/Settings return paths. Test at 360 × 740 with long synthetic names and both currencies. Verify rendered names are readable, not merely present in the DOM, and that ownership appears before buying guidance. Check the safe-money target without opening Details. Check both themes, enlarged text, keyboard-only operation, visible focus, 44px hit areas and no horizontal overflow. Keep page checks based on the actual viewport, not full-page screenshot bar placement.

**Validation for both later cards:** run `npm run gen:api` and `npm run gen:copy` before frontend work. Finish with `npm run typecheck`, `npm run lint`, `npm run format:check`, the full `npm test`, `npm run build`, and `npm run budget` inside `apps/web`. Run the updated Playwright suite under its approved synthetic test setup. Use existing MSW/fakes and controlled browser responses; never call an external service or an LLM. Retain the bundle budget and avoid new runtime dependencies. These are proposed implementation checks, not C51 acceptance commands and were not run in this review.
