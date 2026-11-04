# C50: Shares from Denmark go in the share savings account when it has room

**Why:** from Denmark the owner buys mainly ETFs, plus some global shares, in a share savings account (aktiesparekonto, "ASK") (2026-11-04). The plan already puts ETF buys in the share savings account first (`ask_first`, `account_lines`). But approved US share buys (`approved_us_stock_buys` in `personalise.py`) are always written as `"account": "ordinary"`, even when the share savings account still has room. That account is taxed at 17 % (`rules/residency.yaml` `denmark.ask`). An ordinary account pays 27 % / 42 % share-income tax. So the plan sends shares to the worse account.

Also, the yearly deposit cap is read from a 2026-only key (`deposit_cap_dkk_2026` in `household.py`; `tax_rate_pct_2026` in `placement.py`). The 2027 cap isn't law yet: a proposal is expected in Parliament in February 2027. The app must keep working on 1 January without a code change.

Read `AGENTS.md` first. Builds on C29a (invest from Denmark), C40, C45 (recorded buys count against the share savings account room) and C47 (through-fund room for US picks).

## Decisions (follow them)
- **ETFs keep first claim on the share savings account.** Share buys use only the room left after this week's ETF lines for that member.
- **A share buy is never split across two accounts.** If the room left covers the whole buy, it goes in the share savings account. Otherwise it goes in the ordinary account. Don't shrink the buy to fit.
- **Which shares may go in the account is rule data, not code.** SKAT allows shares "admitted to trading on a regulated market or a multilateral trading facility" (Den Juridiske Vejledning C.B.5.2). The reviewer found no official line naming NYSE/Nasdaq, and brokers decide which markets they offer in the account. So the list is a rule, marked unverified.
- **No new numbers are invented.** The cap formula stays as it is (`ask_deposit_room_dkk`).

## Behaviour

### A. Rules
- `rules/eu_etfs.yaml` `selection.stock_picks`:
  - `ask_eligible_markets: [US]`
  - `ask_verified: false`
  - `ask_source_url: "https://skat.dk/borger/aktier-og-andre-vaerdipapirer/aktiesparekonto"`
  - a comment citing Den Juridiske Vejledning C.B.5.2 (regulated market or MTF), and saying that brokers choose which markets they offer in the account
- `rules/residency.yaml` `denmark.ask`: replace the year-suffixed keys with per-year maps, keeping today's values:
  - `deposit_cap_dkk: {2026: "174200"}`
  - `tax_rate_pct: {2026: "17"}`
  - keep `source_url`, `verified`, `last_checked`, `note_key`
- **One helper** (e.g. `wealth.rates` or `wealth.policy.denmark`): `ask_rule_for_year(rules, key, year) -> (value, exact: bool)`.
  - It returns the entry for the plan's `as_of` year.
  - When that year is missing, it returns the latest earlier year with `exact=False`.
  - Use it in `household.py` and `placement.py`; remove every `_2026` lookup. A guard test fails if any source file reads a key ending in `_20NN` from `denmark.ask`.
  - The plan inputs digest already includes the rules version; make sure a new year entry changes it.

### B. Plan: where share buys go (`personalise.py`)
- After this action's ETF lines are built, the member's room left = `ask_room` minus the ETF lines' share-savings-account amounts (unrounded).
- For each approved share buy, in the existing order:
  - **Share savings account:** the share's market is in `ask_eligible_markets`, the action didn't ask for `ORDINARY`, and the room left ≥ the whole unrounded buy. Write `"account": "ask"` with `reason_codes: ["approved_us_stock_idea", "ask_room_remaining"]`, carry `_amount_dkk_unrounded`, and reduce the room left.
  - **Ordinary account:** otherwise, as today, with `reason_codes: ["approved_us_stock_idea"]`. Add `ordinary_after_ask` only when the member may use the account and the room ran out.
- The existing per-member running room (`running_ask_room`) must count share lines too. Next week's room must count a recorded share buy in the account (C45 already counts waiting `recorded_buys` with `account_type == "ask"`; add a test).
- The action's `account_type` param stays the first accepted line's account.
- The 3-lines-per-member cap, sizing, the C47 room and the minimum buy are unchanged.
- **Share lines in the account:** `details` gains `ask_unverified: true` while `ask_verified` is false. The PWA shows one quiet line behind the step's "Why?": "If your broker doesn't offer this share in your share savings account, buy it in your ordinary account." Nothing on the step itself.

### B2. US share ideas carry their ISIN (found in the C47 review)
- **The gap:**
  - `approved_us_stock_buys` skips any idea without an ISIN.
  - Ideas take `security.isin` (`_research_pack`).
  - The S&P 500 refresh writes US securities through `store.upsert_us_equity_security` with `p_isin = NULL`.
  - So S&P 500 ideas normally have no ISIN, and the plan can't suggest a US share from Denmark.
  - It also means an imported US share (INDmoney and Nordnet files carry ISINs) becomes a separate security from the S&P 500 row.
- **Fix:**
  - The S&P 500 universe refresh passes `isin_from_cusip(row.cusip)` (C47 added the helper and kept CUSIP on `UniverseSecurity`) to `app_upsert_security_by_symbol`.
  - **No clashes.** Before passing an ISIN, check whether another security already has it:
    - if it is a different row from the `(US, symbol)` row, pass `NULL` for that symbol and count it as `isin_conflicts`
    - never re-key, merge or delete securities here
    - one conflict must never fail the whole S&P 500 batch
  - Run JSON counts: `isin_filled`, `isin_conflicts`, `isin_unavailable` (no CUSIP, or a CUSIP starting with a letter).
- **Don't relax** the ISIN check in `approved_us_stock_buys`. The buy line shows the ISIN so the owner finds the right share at the broker.
- **Tests:** CUSIP → ISIN on the stored security; an existing imported security with that ISIN → conflict counted, batch succeeds, both rows unchanged; a CINS code → no ISIN; and a db test where a US idea from the refreshed universe now produces a Denmark share line.

### C. Health (`wealth/alerts/health.py`)
- Extend the existing rules-freshness check (or add `ask_rules`): **warn** when the current year has no exact `deposit_cap_dkk` entry. The message names the key, never a number.

### D. Worked examples (tests)
1. Room 20,000 DKK. ETF line takes 12,000 in the account. A 5,000 DKK US share buy → share savings account; room left 3,000.
2. Room 14,000 DKK. ETF line takes 12,000. A 5,000 DKK share buy → ordinary, with `ordinary_after_ask`. It is never split into 2,000 + 3,000.
3. The action asks for `ORDINARY` → every share line is ordinary.
4. A member who may not use the account (not Denmark-resident) → ordinary, without `ordinary_after_ask`.
5. Two share buys of 3,000 with 4,000 room left after the ETF → the first goes in the account, the second goes ordinary.
6. `as_of` 2027-01-04 with only a 2026 entry → the 2026 cap is used, `exact=False`, and health warns. Add a 2027 entry in a test rules copy → that value is used and the digest changes.

## Tests
- The six worked examples, as pure tests where possible.
- **db:** a recorded share buy in the share savings account lowers next week's room; household isolation is unchanged.
- Update `tests/golden/weekly_plan_user1.json` only if this change explains the difference. Say which lines moved, and why, in the result file.
- **PWA:** the "Why?" line shows only for share lines with `ask_unverified`, and is closed by default.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache`, `.venv\Scripts\python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- The e2e suite runs as `docs/cards/C24-e2e.md` describes (the reviewer also runs it).
- Write `docs/cards/C50.result.md` with every place that reads the share savings account rules.

## Notes
- No migration. No new dependencies. No LLM.
- After deploy, the reviewer dispatches `marketdata-weekly` (or `refresh --only universes`) and checks the `isin_*` counts.
- Don't change the cap formula, the 17 % figure or any ETF placement.
- Out of scope: European share ideas, the proposed 2027 changes (add them as rule data once they are law), and dividend-tax detail for the account.
