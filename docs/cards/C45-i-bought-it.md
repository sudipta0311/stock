# C45: "I bought it": record a buy in two taps, and stop re-suggesting money already spent

**Why:** today "Done" on a weekly step only flips its status. The plan never learns what was bought:
1. **Holdings stay unchanged** until the next statement, which can be weeks away. Denmark buys may never get one, because a Nordnet/Saxo export is imported rarely.
2. **This week's money is counted again.** The plan ignores done actions. When inputs change mid-week (new prices every day), `_compute_if_stale` rebuilds the plan and `save_weekly_plan` re-adds the same buy as a new proposed step next to the done one. It also keeps re-suggesting until a statement arrives.

Read `AGENTS.md` first, especially the money-math rules, the household RLS pattern and the UX rules. Read `docs/UPGRADE_PLAN.md` §3.3 (imports, de-dup) and §4.8 (buying rules).

## Behaviour

### A. Recording a buy (Home)
- **Done on a buy step opens a sheet "What did you buy?".** Buy steps are `BUY_LUMPSUM`, `BUY_STP`, `BUY_ABROAD`, `BUY_STOCK_TRANCHE`.
- **One row per thing to buy**, with its amount filled in from the step:
  - `BUY_ABROAD`: one row per `abroad_buy_list` line. Show the fund or share name and the account in plain words (reuse existing copy for the share savings account and the ordinary account). The amount is in DKK.
  - `BUY_STOCK_TRANCHE`: the share name, units filled in from the step's quantity, and the amount.
  - `BUY_LUMPSUM` / `BUY_STP`: the step's plain fund-type name (as Home shows it now) and the amount the step tells the user to put in now.
  - When there is more than one row, each has a "Bought" checkbox, ticked by default. Unticked rows are not recorded.
- **Optional fields** sit behind "More details", closed by default:
  - the date (default today; not in the future; not more than 14 days before the step's week)
  - units for funds
- **Primary "Save", quiet "Cancel".** After saving:
  - the step shows "Done ✓" and a short line "Added to your money."
  - Home, Plan and Money refresh
- **Other step kinds keep today's Done**, with no sheet.
- **Viewers** see no Done, as today. Partners can record buys.
- **Inputs:** amount and units inputs use the existing Indian/DKK number inputs. Every number input in a flex row gets `w-0`/`w-full` (the Android width trap from 2026-11-01).

### B. What a recorded buy does
A recorded buy is stored with:
- member, link to the step, `bought_on`, amount and currency, optional units
- the security when known: `BUY_ABROAD` lines carry ISIN/ticker, stocks carry `security_id`
- otherwise the target sleeve and instrument category, for India fund-type buys

It has a status: `waiting` → `matched` (a statement now holds it) or `removed`.

1. **It counts as owned at once** in Money, the net worth snapshot and the policy inputs. A household cannot double count it.
   - A buy with a known security feeds the same holdings path the policy and Money already use. Stock and sector caps, satellite totals, the US-situs check, the fund check and tax lots must all see it.
   - Units are the entered units, or the amount ÷ the security's price on `bought_on` (FX-converted with the existing helpers) when a price within 5 days exists.
   - With units it is valued like any holding; otherwise at cost.
   - Prefer the existing `holdings_lots.source = 'manual'` value (allowed by the check constraint, unused today) if it fits cleanly. Otherwise a separate table plus a loader merge. Say which in the result file.
   - A buy with no known security counts at cost (converted to INR) in its sleeve.
2. **This week's money is spent once.** Before the policy runs, reduce each member's `new_money_this_week` for the plan's week by:
   - the INR value of that member's recorded buys (waiting or matched) linked to this week's steps, plus
   - the amount of done steps with no recorded buy, for kinds the policy paid from new money. Read `placement.py` to find which ones.

   So after a buy is recorded, a mid-week rebuild proposes only the money that is left (usually nothing), and never the same buy again.
3. **Share savings account room:** a waiting buy into the share savings account this calendar year reduces the remaining room.
   - It does not when that member's used-room figure comes from imported broker cash flows dated on or after `bought_on`. Those flows already include it.
4. **Plan inputs digest** includes recorded buys, so recording, editing or removing one rebuilds the plan.

### C. When a statement arrives (de-dup)
Do this on import confirm, inside the same transaction, for the import's member only:
- **A waiting buy with a security becomes `matched`** when the confirmed import holds that security for that member and the statement date is on or after `bought_on`. A broker file with no statement date counts as dated on its confirm date.
- **A waiting buy without a security becomes `matched`** when a confirmed CAS (NSDL, CDSL, CAMS/KFintech) for that member is dated at least N days after `bought_on`.
  - N is `recorded_buys.statement_settlement_days: 3` in `rules/buying.yaml`, with a comment: fund units are allotted a day or two after payment.
  - CAMS/KFintech covers funds only, so it never matches a share buy.
- **Matched buys stop counting at once**, so nothing is counted twice.
- **The import confirmation response** gains `counts.recorded_buys_matched`. The import review shows "Includes N buys you added" when N > 0.
- **A buy no statement ever matches** stays as the household's record (the normal case for Denmark buys when no Nordnet/Saxo file is imported).

### D. Money
- **Waiting buys appear in their usual group** (Funds, Shares, …) like other rows, with a quiet line "Added by you".
  - A buy without a security shows its plain fund-type name, e.g. "India index fund".
  - Group totals (C43) include them.
- **Their details** say "Shown from what you entered. Your next statement will replace it." Then **Edit** (amount, units, date) and a quieter **Remove**, which asks for confirmation.
- **Removing** marks the buy `removed`. If the linked step then has no recorded buys left, the step goes back to proposed.

## API (typed, in the OpenAPI export)
- `POST /v1/actions/{action_id}/bought` with `{bought_on?, lines: [{line, amount, currency, units?}]}`.
  - `line` is the index into `abroad_buy_list`; it is `0` for other kinds.
  - The endpoint marks the step done (same audit as `PATCH /v1/actions/{id}`), creates the buys, and returns both.
  - Returns 422 for: a non-buy kind, a step that is not proposed, an unknown line, amount ≤ 0, a currency that doesn't match the line, units ≤ 0, or a date outside the window.
  - Returns 403 for viewers.
- `GET /v1/recorded-buys?status=waiting&member_id=`, `PATCH /v1/recorded-buys/{id}`, `DELETE /v1/recorded-buys/{id}` (sets `removed`).
- Each write adds an `audit_log` row.
- `PATCH /v1/actions/{id}` keeps working for older clients and non-buy steps.
- **Money:** amounts are decimal strings in and out, and `Decimal` everywhere in Python. No floats in the PWA either: use the existing decimal-string helpers.

## Database
- **A new migration `0027_...`** revising `0026`. Never edit earlier migrations.
- **A new table follows the household RLS pattern** from migration 0011 (see AGENTS.md):
  - `user_id` is the household owner
  - add the `<table>_household_access` policy and the `trg_<table>_household_audit` trigger
  - the composite `(user_id, member_id)` FK to `household_members` is `DEFERRABLE INITIALLY IMMEDIATE`
  - the link to `actions` uses `ON DELETE SET NULL (action_id)`
- **Update `app_merge_into_household`** if the table has a per-user unique key.
- **The data export** in Settings includes recorded buys. Account and member deletion cascade.
- **RLS tests:** a viewer can read but not write, another household sees nothing, and a partner can write.

## Tests
- **Policy and loader:**
  - a recorded fund buy is counted once in holdings and the snapshot
  - a rebuild after recording proposes no repeat buy from this week's money
  - a stock buy counts toward the stock and sector caps
  - share savings account room shrinks, and does not shrink twice once cash flows cover the buy
  - a done step without a recorded buy still spends its money
  - all money in `Decimal`
- **Import:**
  - CAS with the same fund dated after the buy → matched and counted once
  - a CAS dated before the buy, or within 3 days for a category buy → still waiting
  - a broker import with the ISIN → matched
  - CAMS/KFintech never matches a share buy
  - a different member's import matches nothing
- **API:** each 422/403 case, line indexes for `BUY_ABROAD`, and the remove-reopens-step rule.
- **PWA:**
  - the sheet is filled in per kind
  - unticked rows are skipped
  - "More details" is closed by default
  - Money shows "Added by you", details, Edit and Remove
  - group totals include waiting buys
  - copy passes the banned-jargon and sentence-length tests (no "tranche", "lot", "allocation", …)
- **e2e journey** (Denmark household, seeded through the test database like the other journeys; no model calls):
  1. Home → Done on the Denmark buy → sheet filled in → Save
  2. the step shows done, and Money shows the fund with "Added by you"
  3. reload: no new proposed buy for the same money
  4. `expectFitsScreen` on the open sheet and on Money

## Acceptance
- `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache`, `.venv\Scripts\python -m scripts.check_deploy_config`
- Migration upgrade and downgrade on a fresh test database. Keep any existing migration guard tests passing.
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- The e2e suite runs as `docs/cards/C24-e2e.md` describes. The reviewer also runs it.
- Write `docs/cards/C45.result.md`. List every place that now reads recorded buys, and how a double count is prevented in each.

## Notes
- No new dependencies. No LLM. No change to research.
- Out of scope (later cards): recording sells, extra loan payments from a step, the emergency-fund step updating savings, and reminders for later parts of a multi-part buy.
