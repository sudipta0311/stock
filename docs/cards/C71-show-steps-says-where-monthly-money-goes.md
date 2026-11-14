# C71: "Show steps" says where monthly money goes

**Source:** the reviewer, 2026-11-14, from the owner's live Home screen (look-only check). "This week" shows two cards titled "Change your monthly investments" (₹53,500 for you, ₹86,800 for the spouse). The primary button "Show steps" opens a sheet that only says "Check the details before taking this step." Its "Why?" says "This points your monthly SIPs toward areas below target."

Nothing on any screen says **where** the money should go:
- the fund and account (a world fund in the share savings account at Nordnet);
- PPF;
- the gold fund;
- the amount for each.

The plan has all of this in `params.changes` of each `REBALANCE_SIPS` action:
- each change has `sleeve`, `from`, `to`, `instrument_category`;
- some also have `route`, `broker` and `account_type`.

The PWA never reads it. `actionHowTo` in `apps/web/src/home/HomePage.tsx` returns `generic_steps` for every action that isn't a buy.

## Read
- `AGENTS.md` (UX rules: plain words, ≤ 20 words a sentence, no jargon, one primary action).
- `apps/web/src/home/HomePage.tsx`: `actionHowTo`, the step sheet, `actionCategory`.
- `apps/web/src/lib/labels.ts` (`instrumentLabel`) and `apps/web/src/features/data.ts`.
- `src/wealth/policy/placement.py`: `_place_sip_redirection` and `_merge_member_sip_actions`, to learn every shape a change can have. Also `_sip_global_assignments`, for the global fields.
- The `home`, `household_ui`, `denmark` and reason copy in `apps/web/src/copy/en.json`.

## Scope
1. **Steps for a monthly change.** For a `REBALANCE_SIPS` action, the step sheet lists one line per change, in the order the API sends them:
   - **Increase** (`to` > `from`), as "<Where>: ₹<to> a month (was ₹<from>)". Leave out "(was …)" when `from` is 0.
   - **Stop** (`to` is 0 and `from` > 0), as "Stop the monthly ₹<from> into <where>."
   - **No change** (`to` equals `from`): leave the line out.
   - **Where:** a plain place name from `instrument_category`, plus the account and broker when present. For example:
     - "World fund in your share savings account at Nordnet" (`ucits_global_etf` + `ASK` + `nordnet`);
     - "PPF";
     - "Gold fund";
     - "Indian index fund".
   - **Amounts:** use the household currency setting and the existing formatters (INR, or DKK with ₹ secondary, as other action amounts do). Round like other action amounts.
   - **Unknown category:** fall back to the sleeve's plain name ("Safe money", "Abroad", "Gold", "Indian shares"). Never show a raw code.
   - **Ending:** finish with one sentence such as "Change these at your bank or broker, then tap Done."
2. **Plain "Why?" text.** Replace the `sip_capacity_redirected` copy with plain words and no "SIPs": "This moves your monthly investing toward parts of your plan that are below target."
   - Keep the clubbing note for the spouse.
3. **Show the destination on the card.** On the Home card itself, under the title, show the main destination as the category line instead of the generic "Monthly investment". The main destination is the largest increase, plus "+N more" when there are more. Keep the card height about the same.
4. **No API change** is expected. If a field the PWA needs is missing from the generated types, extend the response model the smallest way possible and regenerate.

## Tests
Vitest, with the same mocks as the existing Home tests:
- **The owner's two shapes, with synthetic amounts:**
  - ASK world fund only: the sheet shows the world fund line with the amount;
  - PPF + gold for the spouse: two lines plus the clubbing note.
- **A stop line**, and a change with an unknown `instrument_category` that falls back to the sleeve name.
- **DKK primary currency:** amounts show DKK with the ₹ secondary, as elsewhere on Home.
- **Copy guards:** the banned-jargon and sentence-length checks still pass, and "SIPs" no longer appears in English copy.
- **E2E:** update any local e2e Home journey that clicks "Show steps" on a monthly change, if one exists.

## Out of scope
- How the plan splits money (C69 owns the PPF limit).
- Other action kinds.
- The Plan screen.

## Acceptance
- `.venv\Scripts\python -m pytest` (if any Python changed), `ruff check` and `ruff format --check`.
- In `apps/web`: `npm run gen:api`, `npm run gen:copy`, `npm run typecheck`, `npm run lint`, `npm run format:check`, the full `npm test`, `npm run build` and `npm run budget`.
