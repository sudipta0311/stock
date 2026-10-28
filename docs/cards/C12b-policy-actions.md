# C12b: Policy Engine part 2b: core/satellite targets and buying rules → this week's action list (≤ 3)

**Plan:** `docs/UPGRADE_PLAN.md` §4.7 (core/satellite), §4.8 (buying rules), §2.3 (≤ 3 actions, "do nothing" is valid), §4.12 (behavioural guardrails), Appendix C (action contract). Read `AGENTS.md` first. Builds on C11 (`AllocationPlan`, `FoundationReport`) and C12 (`split_equity`, `RouteDecision`, `pick_debt`). Pure Python, deterministic, no DB, no LLM.

## Goal
Turn targets plus this week's new money into at most 3 concrete, explainable actions. Never sell (sells come from exits/rebalancing in C13). Return `NONE` when nothing is worth doing.

## Scope

### 1. Rules: `rules/buying.yaml`
```yaml
version: 1
round_to_inr: "500"
min_action_inr: "1000"
max_actions: 3
satellite:
  default_pct_of_india_equity: "20"     # user setting 0–25
  max_stock_pct_of_total_equity: "5"
  max_sector_pct_of_satellite: "25"
  tranches: 3
  min_quant_decile: 10                   # top decile only
  max_valuation_percentile: "90"         # exclusive
global_satellite: {requires_route: "C", min_global_equity_inr: "500000", max_pct_of_global_equity: "20"}
lumpsum:
  stp_threshold_multiple_of_monthly_sip: "3"
  stp_tranches_normal: 3
  stp_tranches_expensive: 6
  expensive_valuation_percentile: "80"   # index valuation percentile at or above → more tranches
```

### 2. Inputs (`wealth/policy/actions.py`)
`WeekInput`:
- `plan: AllocationPlan` (after `split_equity`), `foundation: FoundationReport`
- `new_money: Rupees`, `monthly_sip_capacity: Rupees`
- `existing_sips: list[Sip]`, where `Sip = {sleeve, instrument_category, monthly_amount}`
- `route: RouteDecision`, `debt_picks: list[DebtPick]`
- `satellite_pct: Decimal`
- `satellite_holdings: list[SatelliteHolding]` (`security_id, sector, value`)
- `ideas: list[SatelliteIdea]`: `security_id, symbol, market ("IN"/"US"), sector, price: Decimal, quant_decile: int, dossier_status ("APPROVED"/"PENDING"/"VETOED"/"UNRESOLVED"), valuation_percentile: Decimal, confidence ("HIGH"/"MEDIUM"/"LOW")`
- `index_valuation_percentile: dict[str, Decimal]` (keys `"IN"`, `"US"`)
- `week: str` (ISO week, e.g. `2026-W45`)

### 3. Output (`ActionList`, matching plan Appendix C)
`{week, status: "ACTIONS"|"NONE", actions: [Action], carry_forward: Rupees, notes: [str]}`. Each `Action` has:
- `rank`
- `kind`: `PAY_DOWN_DEBT`, `BUILD_EMERGENCY_FUND`, `BUY_LUMPSUM`, `BUY_STP`, `BUY_STOCK_TRANCHE`, `REBALANCE_SIPS`
- `sleeve`, `instrument_category`, `route`, `security_id`, `quantity`, `amount`
- `tranche` (e.g. `{"n": 1, "of": 3}`)
- `reason_code`, `params: dict[str, str]`

Numbers are Decimal, serialised as fixed-point strings. **No prose.**

### 4. Algorithm (in this order)
1. **Foundation first.**
   - High-interest-debt gap → `PAY_DOWN_DEBT` with `min(new_money, gap)`.
   - Then emergency-fund gap → `BUILD_EMERGENCY_FUND` with `min(remaining, gap)`, into the top `instant`-liquidity debt pick.
2. **Lump sum.** Spread the remaining new money over sleeves with a **positive** gap, in proportion to the gap.
   - Floor each share to `round_to_inr`, then give the remainder to the largest-gap sleeve.
   - Drop shares below `min_action_inr`; their money goes to the largest-gap sleeve.
   - Instruments per sleeve:
     - `equity_india`: satellite first, then core (step 3)
     - `equity_global`: `route.vehicle`, with `route.route` recorded
     - `debt`: `debt_picks[0]`
     - `gold`: `gold_etf`
     - `cash`: `sweep_fd`
   - If the sleeve amount > `stp_threshold_multiple × monthly_sip_capacity`, emit `BUY_STP` with tranches = `stp_tranches_expensive` when that market's valuation percentile ≥ 80, else `stp_tranches_normal`. Set `params.tranche_amount` = amount / tranches, floored to ₹500. Otherwise emit `BUY_LUMPSUM`.
3. **India satellite.**
   - satellite_target = equity_india_target × satellite_pct; satellite_gap = target − sum(satellite_holdings).
   - If satellite_gap > 0, pick the best eligible idea, ranked by quant_decile, then confidence, then lower valuation_percentile. Eligible means all of:
     - market `IN`, `dossier_status == "APPROVED"`
     - `quant_decile ≥ 10`, `valuation_percentile < 90`
     - the new position ≤ 5% of the **total equity target**
     - sector ≤ 25% of satellite_target after buying
   - tranche_amount = (5% × total equity target) / 3. Buy amount = `min(india_alloc, tranche_amount, satellite_gap)`; quantity = floor(buy amount / price); skip if 0.
   - Emit `BUY_STOCK_TRANCHE` (`tranche {n:1, of:3}`). The rest of the India allocation goes to core (`in_core_index`), floored to ₹500; the leftover adds to `carry_forward`.
4. **SIP redirect.**
   - pool = SIPs going into sleeves with a negative gap + (monthly_sip_capacity − sum(existing SIPs)).
   - If pool ≥ `min_action_inr`, emit ONE `REBALANCE_SIPS` action. Its `params.changes` list sets overweight SIPs to 0 and spreads the pool over underweight sleeves in proportion to the gap (floor ₹500, remainder to the largest gap), shown as `{sleeve, from, to}` per changed sleeve.
5. **Cap at 3 actions.**
   - Priority: foundation actions → `REBALANCE_SIPS` (keeps a slot when present) → lump-sum/STP/stock actions by amount, descending.
   - Dropped lump-sum amounts go to `carry_forward`.
   - If no actions remain → `status = "NONE"`, reason `on_plan`.
6. **Never** emit a sell or a negative amount. When `plan.new_money_only` is true, add the note `tilt_applied_via_new_money`.

### 5. Required worked tests (exact numbers; put the derivation in docstrings)
Shared plan (from the C12 example, moderate, no tilt):
- Targets: equity_india ₹19,05,750; equity_global ₹4,04,250; debt ₹20,20,000; gold ₹4,20,000; cash ₹2,50,000.
- Current: ₹18,00,000; 0; ₹24,00,000; ₹3,00,000; ₹5,00,000.
- Gaps: +1,05,750; +4,04,250; −3,80,000; +1,20,000; −2,50,000.
- Foundation OK; route A (`in_international_index_fund`); no satellite ideas.

| Test | Input | Expected |
|---|---|---|
| lump split | new_money ₹1,00,000, capacity ₹50,000, SIPs: India ₹30,000 + debt ₹20,000 | shares India **16,500**, global **64,500**, gold **19,000** (remainder 500 → global) |
| SIP redirect | same | `REBALANCE_SIPS` changes: debt 20,000→0, India 30,000→33,000, global 0→13,500, gold 0→3,500 |
| cap 3 | same | actions = [REBALANCE_SIPS, BUY_LUMPSUM global 64,500, BUY_LUMPSUM gold 19,000]; **carry_forward 16,500** |
| STP | new_money ₹3,00,000 all into global (make global the only positive gap), capacity ₹50,000, US valuation percentile 50 → then 85 | `BUY_STP` tranches 3 × **₹1,00,000**; with 85 → 6 × **₹50,000** |
| satellite | India allocation ₹50,000; satellite_pct 20, satellite holdings ₹2,00,000; idea price ₹1,420, decile 10, APPROVED, valuation 42 | 5% cap ₹1,15,500 → tranche ₹38,500 → **27 shares (₹38,340)**; core **₹11,500**; carry **₹160** |
| idea gates | valuation 92 / PENDING / decile 9 / sector already at 25% | no stock action; all to core |
| foundation | emergency gap ₹2,00,000, new_money ₹1,50,000 | single `BUILD_EMERGENCY_FUND` ₹1,50,000; nothing else |
| none | all gaps ≤ 0, no SIP pool | `status NONE`, reason `on_plan` |
| never sell | any input | every action amount > 0; no kind contains SELL |

## Out of scope
Exits, trims, rebalancing sells, tax (C13). Explanations/prose (Communicator, C17).

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass (every number above exactly)
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
