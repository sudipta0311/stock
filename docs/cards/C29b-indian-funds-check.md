# C29b: Indian mutual-fund check from the CAMS/KFintech statement, plus an early warning when the fund mix needs adjusting

**Requested by the product owner:** "For Indian options it will analyse the MF portfolio (from the CAMS statement upload) and accordingly try to predict if any rebalance is needed."
**Plan:** `docs/UPGRADE_PLAN.md` §3.3 (CAS import), §4 (allocation, bands, tax-aware actions), §4.10 (India tax). Read `AGENTS.md` first, including the **UX rules** ("rebalance" and "drift" never appear on screen). Builds on:
- `wealth/importer/cas/cams_kfin.py` (holdings + transactions)
- `wealth/marketdata/sources/amfi.py` (NAVAll + sections)
- `wealth/policy/rebalance.py` (bands, then the cash → SIP → sell order, tax lots)
- `wealth/policy/tax.py`
- C27a members/placement (a Denmark-resident member must not get new Indian-fund money)

## Goal
After the user uploads a CAMS or KFintech statement, the app says in one line whether their Indian funds are fine ("Your Indian funds look good ✓") or need a change ("1 change suggested"). It also warns **before** the mix moves outside the plan ("At your current SIPs, small-company funds will pass your limit in about 5 months"). The fix it suggests costs the least tax.

## Scope

### 1. Better fund data (AMFI)
- Extend `classify_amfi_section` to keep the **SEBI sub-category** from the section title. Examples: Large Cap, Large & Mid Cap, Mid Cap, Small Cap, Flexi Cap, Multi Cap, ELSS, Index Funds/ETFs, Sectoral/Thematic, Aggressive/Balanced/Conservative Hybrid, Arbitrage, Liquid, Gilt, other debt, FoF Overseas, Gold.
- Keep it on `securities` in a new column `sub_category` (new migration, next free number).
- Derive `plan_type` (direct/regular) and `option` (growth/IDCW) from the scheme name, with tests over real-looking synthetic names.
- **Month-end NAV history:**
  - store month-end NAVs (5 years) for Direct-Growth equity and hybrid schemes plus every scheme a user holds
  - use AMFI's official NAV-history download (investigate the URL; network allowed during the card; tests use recorded fixtures)
  - table `nav_monthly(security_id, month_end, nav, source)`; budget it to ≤ 30 MB on Neon (state the row count)
  - wire it into `python -m wealth.marketdata.refresh --only amfi_history` and a monthly GitHub Actions run (add to the C18 notes)
- **TER:** use AMFI's published TER data if a stable official file exists; else `null` with `ter_unavailable` (never guessed).

### 2. Fund check (`wealth/policy/fund_check.py`, pure; thresholds in `rules/fund_check.yaml`)
Per member and for the family, from the imported MF holdings and transactions:
- **Mix:**
  - the Indian-equity sub-mix (large / mid / small / flexi-multi / sector-thematic / international) and equity vs hybrid vs debt
  - the targets come from new `rules/fund_check.yaml` by risk profile; the Indian sleeve totals still come from the existing plan
- **Checks** (each yields a `reason_code` and money at stake in ₹):
  - `regular_plan`: a Direct plan of the same scheme exists. Estimate the yearly cost from the TER difference, or skip the estimate if TER is unknown. The switch cost (exit load, capital-gains tax via `tax.py` lots) and the payback in years go in details. Suggest it only when payback ≤ the rule value (default 2 years).
  - `too_many_funds`: > N funds in one sub-category (default 2) or > M funds in total (default 7). Suggest consolidating into the best one held, or an index fund.
  - `closet_index`: ≥ 2 large-cap active funds whose 3-year return trails the Nifty 50 index fund in the same data. Suggest an index fund for new money.
  - `lagging_fund`: 3-year and 5-year returns in the bottom quartile of the sub-category (month-end data) on **two consecutive** monthly checks. The outcome is "review", never an automatic sell.
  - `idcw_option`: suggest Growth for new money (tax), with no switch suggested if the switch cost is above the rule value.
  - `sector_heavy`: sector/thematic > the rule limit (default 10% of Indian equity).
  - `debt_fund_vs_fd`: debt funds bought after 1 Apr 2023 are taxed at the slab rate. Compare them with the debt picker's best option and add a note only.
- **Residency (C27a):** for a Denmark-resident member:
  - no new money into Indian funds (C26a `dk_fund_tax_review`)
  - existing funds: hold by default; the yearly Danish tax note stays `verified: false` in details
  - new SIP money is redirected using C27a placement: the RI spouse's name with the clubbing note, or the C29a route when it exists
- **SIP detection:** recurring purchases in the transactions (same scheme, ~monthly, similar amounts) give the current SIP per scheme and member. The user can correct it in Settings → Fine-tune.

### 3. Early warning ("predict")
- Project the mix month by month for 12 months:
  - start: current values
  - contributions: detected SIPs
  - growth: **deterministic** expected monthly returns per sub-category from `rules/fund_check.yaml` (conservative; cite the source or mark `verified: false`)
- Report the first month a band (from `rebalance.py` `BandRules`) would be breached, per member and family: `{bucket, months_until, direction}`. Report nothing when nothing is breached within 12 months.
- **The fix order** (reuse `rebalance_plan`):
  1. redirect SIPs (no tax)
  2. new money
  3. sells, using the cheapest lots first: avoid < 12 months (STCG + exit load) and use each member's ₹1.25 lakh LTCG exemption for the FY (per person; note C27a's per-member harvest)
- Every amount comes from code. Round to ₹100 in actions.
- **Tests with fake clocks:**
  - a breach now
  - a breach in 5 months
  - no breach
  - SIP redirect only (no sells)
  - a sell with tax and the exemption used

### 4. API, actions, email
- `GET /v1/fund-check?member=<id|family>`:
  - `{status: fine|attention, summary_key, checks[] (≤ 3 shown, ranked by money at stake), early_warning, details}`
  - `details` holds the per-fund table (sub-category, plan type, option, 3-year/5-year return vs sub-category median in plain percentages, TER if known)
  - RLS as for holdings
- **Weekly actions** gain `SWITCH_TO_DIRECT`, `REDIRECT_SIP`, `CONSOLIDATE_FUNDS`, `REVIEW_FUND`. They share the ≤ 3 actions per week limit with everything else, ranked by money at stake, with the member who acts.
- **Copy keys** in `copy/en.yaml` (plain, ≤ 20 words):
  - "Your Indian funds look good ✓"
  - "Switch to the direct plan to save about ₹4,200 a year"
  - "Your small-company funds will pass your limit in about 5 months. Send new SIP money to your large-company fund instead."
- **Email:** at most one "Indian funds" line.

### 5. PWA (small, follows the UX rules)
- **Money:**
  - an "Your Indian funds" card: one status line + one button (**See why** or **Upload CAMS statement** when none)
  - the detail sheet lists funds in plain words ("Large companies", "Small companies"; "Direct plan ✓" / "Regular plan: costs more") with the per-fund numbers collapsed
- **Upload helper:** 3 short steps to get the CAMS statement:
  - where to request it
  - choose "Detailed" with transactions
  - set a password
  - plus a note that summary statements give fewer insights (no tax estimates)
- **Home:** the actions appear in "This week". At most 1 highlight comes from the early warning.

### 6. Tests (no network)
- AMFI sub-category, plan type and option parsing.
- Each check, positive and negative, with money at stake.
- The payback gate on `regular_plan`.
- `lagging_fund` needs two consecutive checks.
- SIP detection: a monthly series, a missed month, a one-off lump sum ignored.
- The projection and fix order (above).
- A Denmark-resident member: no new Indian-fund money, SIPs redirected.
- The API contract and RLS (another user's household sees nothing).
- PWA: the status card states, the plain labels, the banned-terms scan (if C28's test exists), axe.
- A summary-only CAS (no transactions) → checks that need lots are skipped with `tax_estimate_unavailable`.

## Acceptance
1. `pytest` (full) + `ruff` pass; migration upgrade/downgrade tested; OpenAPI refreshed; PWA `typecheck`, `lint`, full `test`, `build`, `budget` pass
2. The result file shows a worked example on the synthetic household (an RI spouse with 9 funds, 2 of them regular; a Denmark-resident self with 2 Indian funds): checks, early warning, and the actions with their amounts and tax
3. `nav_monthly` row count and size estimate; TER source found or `ter_unavailable` explained
