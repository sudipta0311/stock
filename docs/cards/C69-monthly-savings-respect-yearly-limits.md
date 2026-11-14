# C69: monthly savings respect yearly limits

**Source:** the reviewer, 2026-11-14, from a look-only read of the live plan. The plan moves part of the household's monthly savings into safe money. All of it goes to one person's PPF: **₹66,783 a month**, about ₹8 lakh a year. PPF allows at most **₹1.5 lakh a year** per person (`rules/debt_instruments.yaml`: `ppf.eligibility.annual_limit`). The owner would be told to do something the scheme does not allow.

**Cause:** `_place_sip_redirection` in `src/wealth/policy/placement.py`, debt/cash branch:
- it takes the first eligible pick from `_ranked_debt_candidates` and places the **whole** monthly increase there;
- it never checks the yearly room that `_debt_room(member, instrument)` exposes (`limits_room["ppf_annual_inr"]` and the others).

The one-off path, `_place_debt`, already caps each pick at its room and moves the rest on to the next candidate. The monthly path does not.

## Read
- `AGENTS.md`.
- `src/wealth/policy/placement.py`: `_place_sip_redirection`, `_place_debt`, `_ranked_debt_candidates`, `_debt_room`, `_merge_member_sip_actions`.
- `src/wealth/policy/household.py`, where `limits_room` is built (`ppf_annual_inr`, `ssy_annual_inr`, `scss_inr`, `pomis_inr`).
- The SIP-redirection tests in `tests/policy/test_household.py`, `tests/policy/test_personalise.py` and `tests/policy/test_actions.py`.

## Scope
1. **Cap each monthly amount by yearly room.** In the debt/cash branch of `_place_sip_redirection`, walk the ranked candidates the way `_place_debt` does:
   - for an instrument with a yearly limit (PPF, SSY), the monthly amount is at most **remaining yearly room ÷ 12**, rounded **down** to paise, so twelve payments never exceed the limit;
   - for an instrument with a total cap (SCSS, POMIS), the monthly amount is at most **remaining cap ÷ 12**, rounded down;
   - an instrument with no limit takes whatever is left;
   - a member and instrument pair is used at most once.
2. **The rest goes to the next candidate,** for the same member or another eligible member, in the existing ranking order. The existing clubbing note still applies when one person's money funds another person's savings.
3. **If nothing can take the rest,** place what fits and add the existing `sip_increase_unplaced:debt` note. The total of the placed changes never exceeds the increase.
4. **Show it plainly.** The Plan screen already shows one row per destination in a member's SIP change, so a split appears as two rows (for example "PPF ₹12,500 a month" and "<next option> ₹54,283 a month"). Change the PWA only if a split cannot be shown today, and keep the copy short.

## Tests (no network)
- **Spouse, Indian resident, no PPF yet, debt increase ₹66,783 a month:**
  - PPF gets ₹12,500.00 a month;
  - the next ranked eligible option gets ₹54,283.00;
  - the total is unchanged.
- **Existing PPF contribution of ₹1,00,000 a year:** PPF gets ₹4,166.66 a month (rounded down).
- **No PPF room left:** PPF gets nothing, and the next option gets the full amount.
- **Only capped options and too little room:** the placed total is less than the increase, and `sip_increase_unplaced:debt` is noted.
- **One-off buys:** existing `_place_debt` tests still pass unchanged.
- **Golden plan:** if `tests/golden/weekly_plan_user1.json` changes, explain why in the result file.

## Out of scope
- Allocation targets, debt ranking or yields.
- Lump-sum placement.
- PPF eligibility rules.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
- If the PWA changes, run in `apps/web`: `npm run gen:api`, `npm run gen:copy`, `npm run typecheck`, `npm run lint`, `npm run format:check`, the full `npm test`, `npm run build` and `npm run budget`.
