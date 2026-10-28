# C12: Policy Engine part 2: India/global equity split (with bounded geopolitical tilts), global route selector, debt-instrument picker

**Plan:** `docs/UPGRADE_PLAN.md` §4.4 (global share), §4.5 (routes A–D), §4.6 (debt sleeve), Appendix B (`routes.yaml`), §5.5 (tilts bounded, applied via new money). Read `AGENTS.md` first. Builds on C11 (`AllocationPlan`, `WealthSnapshot`, `FinancialContext`). Pure Python, deterministic, no DB, no LLM.

## Goal
Split the equity target into **India** and **global**, apply research tilts only within hard bounds, pick the simplest cost- and tax-efficient **route** for buying global assets, and pick the best **after-tax** debt instrument for each need.

## Scope

### 1. Global share: `wealth/policy/global_split.py` + `rules/global.yaml`
```yaml
version: 1
base_global_share_of_equity_pct: {conservative: "12.5", moderate: "17.5", growth: "25", aggressive: "30"}
foreign_goal_bonus:          # by share of investable wealth earmarked for USD / education_abroad goals
  - {min_share_pct: "20", bonus_pp: "10"}
  - {min_share_pct: "5",  bonus_pp: "5"}
tilt_bounds: {global_pp: "5", gold_pp: "3"}
```
- `GeoTilts` (Pydantic): `global_pp: Decimal`, `gold_pp: Decimal`, `evidence_ids: list[str]`, `as_of: date`. The validator **rejects** values outside the bounds (from the YAML) and rejects any non-zero tilt with fewer than 2 `evidence_ids`.
- C11's `build_targets` takes `fx_usd_inr` and stores it as `plan.fx_usd_inr`. For the foreign-goal share, compute from the `goals` argument: a goal is *foreign* if `currency == "USD"` or `goal_type == "education_abroad"`. Its rupee amount is `target_amount × plan.fx_usd_inr` for USD goals (rounded to paise), else `target_amount`. Share = sum of foreign rupee amounts / `plan.investable_total`. If a USD goal is present and `plan.fx_usd_inr` is None, raise `ValueError`.
- `split_equity(plan, goals, profile_band, tilts | None, rules) -> AllocationPlan` returns a new plan where sleeve `equity` is replaced by `equity_india` + `equity_global`:
  - `global_share = base + bonus + tilts.global_pp`, clamped to 0–60.
  - Gold tilt moves percentage points between `gold` and `debt` (it never touches equity).
  - Recompute gaps against current holdings. Targets still sum to 100 ± 0.01.
- **Tilts never create sell actions.** Expose `new_money_only=True` on the plan when any tilt ≠ 0, so the buying rules (C12b) know to use new money only.

### 2. Route selector: `wealth/policy/routes.py` + `rules/routes.yaml` (exactly the plan's Appendix B, plus `fx_markup_note`)
Inputs (`RouteContext`):
- `monthly_global_new_money: Rupees`
- `annual_global_new_money: Rupees` (derived as monthly × 12 if missing)
- `has_lrs_broker: bool`
- `prefers_no_paperwork: bool`
- `us_situs_holdings_usd: Decimal`
- `lrs_used_this_fy: Rupees`
- `amc_international_suspended: bool`
- `etf_inav_premium_pct: Decimal | None`

`select_route(ctx, rules) -> RouteDecision` returns `{route: "A"|"B"|"C"|"D"|"WAIT", vehicle: str, reasons: list[str], warnings: list[str], alternatives: list[route]}`, where `vehicle` is a category like `"in_international_index_fund"`, `"in_listed_international_etf"`, `"ucits_etf"` or `"us_etf"`. Logic:
1. If `prefers_no_paperwork` or `annual_global_new_money < route_c_min_annual_inr` → prefer A, unless `amc_international_suspended`. Then B, unless `etf_inav_premium_pct > etf_max_inav_premium_pct` or the premium is unknown. Then C if `has_lrs_broker`, else WAIT with reason `no_open_route`.
2. Otherwise (large and paperwork OK) → C if `has_lrs_broker`, else A (reason `open_lrs_broker_to_save_costs`).
3. On route C, the vehicle is `ucits_etf` when `us_situs_holdings_usd ≥ switch_to_ucits_above_usd`, otherwise `us_etf` with warning `us_estate_tax_threshold_near` when ≥ 80% of the threshold.
4. Warnings: `tcs_applies` when `lrs_used_this_fy + annual_global_new_money > lrs_tcs_threshold_inr` (params: excess amount, TCS = 20% of the excess, rounded to rupees); `schedule_fa_reporting` whenever the route is C or D.

### 3. Debt picker: `wealth/policy/debt_picker.py` + `rules/debt_instruments.yaml`
Catalogue entries: `liquid_fund`, `sweep_fd`, `bank_fd`, `po_td_1y/2y/3y/5y`, `arbitrage_fund`, `target_maturity_fund`, `gsec_rbi_retail_direct`, `nsc`, `ppf`, `vpf`, `scss`, `pomis`, `rbi_frsb`. Each has:
- `min_horizon_years` / `max_horizon_years`
- `liquidity` (C02 enum)
- `tax`: one of `slab` (interest/gains at slab), `equity` (STCG 20% if held ≤ 1 year, else LTCG 12.5%, ignoring the exemption in this picker), `exempt` (PPF), or `slab_deferred` (debt MF, still slab after 1 Apr 2023)
- `eligibility`: `min_age`, `max_amount`, `annual_limit` (PPF 1,50,000; SCSS 30,00,000; POMIS 9,00,000 single)
- `yield_key` (looked up in the `yields: dict[str, Decimal]` input; the picker never hard-codes yields)

`pick_debt(need: DebtNeed, ctx: FinancialContext, profile_tax_slab_pct, age, yields, existing: snapshot, rules) -> list[DebtPick]`:
- Filter by horizon, liquidity need and eligibility. For PPF, compute remaining room = annual_limit − this FY's PPF contributions.
- After-tax yield:
  - `slab`: y × (1 − slab)
  - `equity`: y × (1 − 0.20) if horizon ≤ 1 year, else y × (1 − 0.125)
  - `exempt`: y
  - `slab_deferred`: y × (1 − slab)
- Rank by after-tax yield, then liquidity, then simplicity order (the order in the YAML). Return the top 3 with `after_tax_yield_pct` (2 dp) and `reason_codes`.
- **FD split rule:** if a `bank_fd` pick amount exceeds ₹5,00,000, add `split_across_banks` with the number of banks needed (DICGC cover is ₹5 L per bank).
- Tax regime note: when `tax_regime == "old"` and the pick is `ppf`, add reason `80c_deduction_available`.

### 4. Tests (worked numbers; put them in test docstrings)
- **Global split** (the C11 example: equity target ₹23,10,000, investable ₹50,00,000, moderate):
  - no foreign goals, no tilt → global **₹4,04,250**, India **₹19,05,750**
  - with an `education_abroad` goal of ₹20,00,000 (40% of investable → +10 pp) → global share 27.5% → **₹6,35,250**
  - adding tilt `global_pp=+2` (2 evidence ids) → 29.5% → **₹6,81,450**
  - tilt `+6` → ValidationError
  - tilt `+2` with 1 evidence id → ValidationError
  - gold tilt `+3` → gold +3 pp, debt −3 pp, equity unchanged
- **Routes:**
  - ₹10,000/month, no broker → A
  - same with `amc_international_suspended` and premium 0.4% → B
  - premium 2.5% and no broker → WAIT (`no_open_route`)
  - ₹6,00,000/year with broker → C (`us_etf`)
  - `us_situs_holdings_usd=65000` → C (`ucits_etf`)
  - 50,000 → C with `us_estate_tax_threshold_near`
  - `lrs_used_this_fy=₹6,00,000` + ₹6,00,000 new → `tcs_applies` with excess ₹2,00,000 and TCS **₹40,000**
- **Debt picker** (yields fixture: bank_fd 7.0, arbitrage_fund 6.5, liquid_fund 6.0, ppf 7.1, scss 8.2):
  - 30% slab, 1-year horizon → arbitrage first (5.20% after tax) ahead of bank FD (4.90%)
  - 10% slab, same → bank FD first (6.30%) ahead of arbitrage (5.20%)
  - long horizon with PPF room → PPF (7.10% exempt) ranks first
  - PPF room 0 → PPF excluded
  - age 59 → SCSS excluded; age 60 → SCSS included
  - FD pick of ₹12,00,000 → `split_across_banks` with 3 banks

## Out of scope
Turning targets into weekly actions (C12b), tax engine and exits (C13), live yields (market data / research later).

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass (every worked number exactly)
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
