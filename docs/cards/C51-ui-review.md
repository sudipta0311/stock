# C51: Make the app easier to use (review only)

**Type:** review. **No code changes.** You write a report; a later card (C52) implements what the reviewer accepts.

## Why
The owner's standing request: "keep improving the app... it should be user friendly and intuitive". The users are the owner and their spouse: an Indian family living in Denmark, not finance experts, using the PWA on a phone about 360 px wide. Their main jobs, in order:

1. See what to do this week, do it and tick it off (Home).
2. Get their money into the app: add a deposit or savings, upload a broker file (HDFC Sec, Zerodha, INDmoney for US shares, Danish brokers) or a mutual-fund statement (Money → Add, Import).
3. Understand one holding: what it's worth and what we suggest (keep, add more, sell), and why (Money → holding page).
4. See the overall plan and whether they're on track (Plan).
5. Look at new share and fund ideas without buying something they already own through a fund (Ideas).

**Real confusion we have seen:** the owner tried to upload their INDmoney broker file from the **Inbox** tab and couldn't find how. Broker files are uploaded from Money → Add → "upload a broker file"; Inbox only takes emailed PDF statements. Treat this as evidence that the way into "add my holdings" isn't obvious.

## Read
- `AGENTS.md`, especially **UX rules**: one primary action per screen, the banned-jargon list, at most 2–3 notes per screen, smart defaults, plain numbers, short copy, "when unsure, show less".
- `docs/UPGRADE_PLAN.md` §2.3 (design principles) and §8 (PWA UX redesign, including §8.4 wireframes).
- `docs/usability-script.md`.
- The PWA source in `apps/web/src/` (routes, pages, `features/`, `money/`, `account/`, shared components) and the copy source `copy/en.yaml`. The files in `apps/web/src/copy/` and `apps/web/src/api/schema.d.ts` are generated.

## Screenshots (attached to your prompt)
28 PNG screenshots of the current build at 360 × 740, taken by the reviewer with synthetic data (a "Me and my spouse" household, one NRI and one resident, with a fixed deposit, Danish savings, two shares with weekly reviews, an ideas pack and a Home action). The file names say where each was taken:

01 login · 02 welcome · 03–05 onboarding (who, tax residence, plan ready) · 06 Plan (new account) · 07 Home (new account) · 08 Money (empty) · 09 Home with "Make it more accurate" open · 10 goal form · 11 risk questions · 12 Money "What else do you have?" chooser · 13 fixed-deposit form · 14 fixed deposit saved · 15 Home with data · 16 Money with data · 17 a share with "Add more" · 18 a share with "Sell" · 19 an asset's detail page · 20 Plan with data · 21 Ideas · 22 world funds · 23 Import · 24 Inbox · 25 Settings · 26 Help · 27 account menu · 28 Home, first screen only (no scrolling).

Most are full-page captures. In those, the bottom navigation bar can appear in the middle of the page: that's a capture artifact (the bar is fixed to the screen bottom), not a bug. Synthetic names such as "Synthetic share to add" and the amounts are test data; judge the layout and words around them, not the data itself.

## Scope
Write **`docs/reviews/C51-ui-review.md`** (create the folder). Base every finding on a screenshot or on code you read, and name it (screenshot number or `file:line`). Structure:

1. **Top 10 changes**, ranked by how much they would help the five jobs above. Each one gives:
   - the screen(s), and what the user sees now (quote the exact words);
   - why it slows down or confuses someone who isn't a finance expert;
   - the concrete change: new wording (final text, following the copy rules), what moves, merges, hides or goes away, and which component or file changes;
   - size (S/M/L) and risk (what could break: tests, accessibility, other screens).
2. **Screen by screen:** a short list of further findings for each screen, smaller items included. Skip screens with nothing worth changing.
3. **Across the app:** navigation and naming (do tab names and page titles match what people are looking for?), jargon still on screen (check against the banned list and look for other finance words), repeated or competing actions, empty states and first use, number formatting, notes and warnings (more than 2–3 on a screen?), "Why?"/"Details" patterns, and tap targets and readability at 360 px.
4. **Keep as is:** what already works well and should not change.
5. **Proposed implementation:** split the accepted changes into at most two implementation cards, ordered so the first gives the most value. For each card: the changes it includes, the files it touches, and how to test it (the vitest and e2e checks to add or update).

Rules for proposals:
- **Front end and copy only** (`apps/web`, `copy/en.yaml`, `e2e`). Propose a backend change only when a screen can't be fixed without it, and mark it **[backend]**. Never propose changing amounts, weights, rates, tax figures, dates, or the rules that decide them.
- No new runtime dependencies. The PWA bundle has a size budget (`npm run budget`).
- Keep WCAG 2.2 AA (labels, contrast, focus, tap targets of at least 44 px).
- English copy only, following AGENTS.md: sentences of 20 words or fewer, titles of 8 words or fewer, calm tone, no banned jargon.
- Prefer removing and merging over adding. Every extra element needs a reason.

## Out of scope
- Any code, copy, test or config change. The only files you create are the report and the result file.
- Visual rebranding (colours, fonts, logo).
- New features that aren't about ease of use.

## Acceptance
- `docs/reviews/C51-ui-review.md` exists and contains the five sections above. The Top 10 has exactly 10 ranked items, each with a screenshot number or a `file:line`.
- `git status --porcelain` shows only `docs/reviews/C51-ui-review.md` and `docs/cards/C51.result.md` as new or changed (plus any files that were already untracked before you started).
- No real personal data, secrets or PAN-like strings in the report.
