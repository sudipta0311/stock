# C13: Policy Engine part 3: India tax engine, tax-aware lot selection, LTCG harvesting, rebalancing bands, insurance policy review

**Plan:** `docs/UPGRADE_PLAN.md` §4.10 (tax engine, harvesting), §1.3 P0-7 (the exemption is per FY across all equity, not per stock; cess), §4.11 (insurance review: never "surrender now"). Read `AGENTS.md` first. Builds on C02 (money/dates), C04 (`policy_irr`), C11 (`AllocationPlan`). Pure Python, deterministic, no DB, no LLM.

## Goal
Compute the tax on any proposed sale correctly, choose the cheapest lots to sell, spot LTCG-harvest opportunities, tell when an asset class has drifted far enough to act, and compare continue / paid-up / surrender for LIC-style policies with real numbers.

## Scope

### 1. Rules: `rules/tax/india.yaml` (every entry has `effective_from`, `source_url`, `verified: false`)
Asset tax classes, with the holding period for long-term ("more than N months"):

| Class | LT after (months) | STCG | LTCG |
|---|---|---|---|
| `listed_equity` and `equity_mf` | 12 | 20% | 12.5% above a **₹1,25,000 per-FY** exemption shared across both classes |
| `foreign_equity` (direct US stocks, overseas-listed/UCITS ETFs) | 24 | slab | 12.5% |
| `intl_fof` (Indian FoF investing abroad) | 24 | slab | 12.5% |
| `intl_etf_listed_in` (Indian-listed international ETF) | 12 | slab | 12.5% |
| `debt_mf_post_2023` (bought on/after 2023-04-01) | — | slab always | slab always |
| `gold_etf` | 12 | slab | 12.5% |
| `sgb` | 12 | slab | 12.5%; redemption **at maturity** is exempt |

- Cess 4% on all tax. Optional `surcharge_pct` input (default 0); for 20%/12.5% equity gains the surcharge is capped at 15%.
- Equity rates apply from `2024-07-23`. Sales before that date are out of scope and raise `ValueError`.
- Mark `intl_fof`, `intl_etf_listed_in`, `gold_etf` and `sgb` with `needs_review: true` in the YAML (recent rule changes).

### 2. Tax engine: `wealth/policy/tax.py`
- `Lot`: `lot_id, tax_class, quantity, cost_per_unit, acquired_on, fmv_2018_01_31 | None`.
- `is_long_term(lot, sale_date) -> bool` means "held for **more than** N months": LT only when `sale_date > add_months(acquired_on, N)`. Bought 2025-01-10 → a sale on 2026-01-10 is ST; on 2026-01-11 it's LT.
- `days_to_long_term(lot, today) -> int` (0 if already LT).
- Grandfathering for `listed_equity`/`equity_mf` acquired before 2018-02-01: cost = `max(actual_cost, min(fmv_2018_01_31, sale_price))`. Ignore it when FMV is missing, and add a note.
- `estimate_sale_tax(sales: list[(Lot, qty, sale_price)], sale_date, ctx: TaxContext) -> TaxEstimate`:
  - `TaxContext` = `slab_pct`, `surcharge_pct`, `fy_realised_equity_ltcg` (already booked this FY), `fy_realised_equity_stcg`.
  - Per lot: gain, `ST`/`LT`, rate, tax before cess.
  - Equity LTCG: subtract the **remaining** FY exemption (`1,25,000 − fy_realised_equity_ltcg`, floor 0) from the aggregate LTCG of this sale set.
  - Losses: ST losses offset ST then LT gains; LT losses offset only LT gains (Indian set-off rules, within the same sale set).
  - Totals: `tax_before_cess`, `cess`, `total_tax` (paise), `exemption_used`, `net_proceeds`.
- `select_lots(lots, quantity, sale_price, sale_date, ctx) -> list[(Lot, qty)]`. Order: losses first (largest loss per unit first), then LT gains (smallest gain per unit first), then ST gains (smallest gain per unit first). Partial lots allowed.
- `ltcg_harvest(lots, price_by_lot, today, ctx) -> HarvestSuggestion | None`. Only for 1 Feb–31 Mar. Remaining exemption > 0 and LT lots with gains → choose lots and quantities whose gains add up to ≤ the remaining exemption (greedy, smallest gain per unit first; whole units only). Return gains booked, tax saved vs a 12.5% rate, and the lots.

### 3. Rebalancing bands: `wealth/policy/rebalance.py`
`band_breaches(plan, rules) -> list[Breach]` using the 5/25 rule: a sleeve breaches when `|actual% − target%| ≥ 5` **or** `|actual% − target%| ≥ 0.25 × target%` (relative test only when target% > 0). Each breach has `sleeve, actual_pct, target_pct, direction ("over"/"under"), excess_amount`.

`rebalance_plan(breaches, week_input, lots_by_sleeve, sale_date, tax_ctx) -> RebalancePlan` returns the order of operations:
1. New cash. Reuse C12b's allocator with the breach sleeves first.
2. SIP redirect.
3. Sells, only when still in breach after 1–2, and only for over-weight sleeves, using `select_lots` with a tax estimate attached.

**Never sell the core because of a short-term move.** Sells need a breach and are capped at the amount that brings the sleeve back to the band edge, not to target.

### 4. Insurance review: `wealth/policy/insurance_review.py`
`review_policy(policy, as_of, compare_rates: {"ppf": Decimal, "fd": Decimal}) -> PolicyReview`:
- `irr_pct` (from C04's `policy_irr`; None without an illustration).
- `continue_value` = maturity_illustration.
- `paid_up_value` = `sum_assured × premiums_paid / premiums_payable + bonus_accrued`, where premiums_paid = years elapsed (capped at PPT) and premiums_payable = PPT (or term).
- `surrender_value` = the user-entered amount (None if missing → reason `ask_insurer_for_surrender_value`).
- `comparison` against PPF/FD.
- `reason_codes`, e.g. `low_return_vs_ppf`, `cover_low_vs_premium` (sum_assured < 10 × annual_premium).
- `recommendation_kind` is **always** `"compare_options"`. There is no field or value that says "surrender".

### 5. Required worked tests (exact; put derivations in docstrings)
| Case | Expected |
|---|---|
| Listed equity 100 sh bought 2025-01-10 @ ₹1,000, sold 2026-01-10 @ ₹1,500 (slab 30%) | ST: tax **₹10,400.00** (20% + cess) |
| same sold 2026-01-11 | LT, gain ₹50,000 inside exemption → tax **₹0.00**, exemption_used ₹50,000 |
| Two LT equity lots, gains ₹1,00,000 + ₹75,000, nothing realised earlier this FY | taxable ₹50,000 → **₹6,500.00** |
| same, but ₹1,00,000 LTCG already realised this FY | remaining exemption ₹25,000 → taxable ₹1,50,000 → **₹19,500.00** |
| Grandfathered: bought 2017-06-01 @ ₹500, FMV ₹800, sold @ ₹1,200 | cost ₹800, gain **₹400/share** |
| Foreign equity gain ₹1,00,000, slab 30%: sold 23 months 30 days after purchase / 24 months + 1 day | **₹31,200.00** (ST, slab) / **₹13,000.00** (LT) |
| Debt MF bought 2023-05-01, held 3 years, gain ₹10,000, slab 20% | **₹2,080.00** (slab always) |
| `days_to_long_term`, equity bought 2026-01-15, today 2026-11-01 | **76** |
| `select_lots`: lots A (loss), B (LT gain), C (ST gain); sell qty = A + B | picks A then B; C untouched |
| ST loss ₹20,000 + LT gain ₹50,000 in one sale set, exemption fully used already | LT taxable ₹30,000 → **₹3,900.00** |
| `ltcg_harvest` on 2027-02-10, realised ₹25,000, one LT lot with ₹200 gain/unit, 2,000 units | **500 units** (₹1,00,000 gains); tax saved **₹12,500** (before cess) |
| `ltcg_harvest` on 2026-11-10 | `None` (outside window) |
| Bands: gold target 10 / actual 7.4 → breach (relative); equity 55/61 → breach (absolute); debt 35/31.6 → no breach | as stated |
| Paid-up value: SA ₹10,00,000, 10 of 21 premiums paid, bonus ₹3,00,000 | **₹7,76,190.48** |
| Review with no surrender value | reason `ask_insurer_for_surrender_value`; `recommendation_kind == "compare_options"` |

## Out of scope
Exit triggers and stress tests (C13b). Fetching FMV/prices (C10). Tax on dividends/interest income (later).

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass (every number above exactly)
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
