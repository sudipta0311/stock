# C13b: Policy Engine part 3b: Exit Engine (exit cards, triggers, tax-aware timing) + personal stress tests

**Plan:** `docs/UPGRADE_PLAN.md` §4.9 (Exit Engine table, "never stop-loss the core", tax-aware timing, lot selection), Appendix A (`rules/exits.yaml`), §8.4 ("World & your money" stress test, exit card wireframe), §5.5 (geopolitical changes need ≥ 2 independent cited sources). Read `AGENTS.md` first. Builds on C13 (`tax.py`: `is_long_term`, `days_to_long_term`, `select_lots`, `estimate_sale_tax`) and C12 (`AllocationPlan` with `equity_india`/`equity_global`). Pure Python, deterministic, no DB, no LLM.

## Goal
For every **satellite** holding, an Exit Card that says plainly which triggers are OK, on watch, or fired, and what to do (keep / trim / exit / review / swap / wait for long-term), with the exact quantity and the tax cost. Plus a stress test showing what defined scenarios would do to the user's money in rupees, compared with their target plan.

## Scope

### 1. Rules: `rules/exits.yaml`
Exactly the plan's Appendix A shape (thesis_break, governance_break, geopolitical_break, valuation_stretch, drift, risk_stop, tax_wait, core) with those default numbers. Add a `watch_fraction: "0.8"` meaning "status `watch` once a metric reaches 80% of its trigger threshold" for numeric triggers.

### 2. Inputs (`wealth/policy/exits.py`)
`HoldingFacts` (one per satellite security; built by later cards from market data and research):
- `security_id`, `symbol`, `sector`, `is_core: bool`
- `lots: list[Lot]` (C13), `price: Decimal`
- `weight_pct_of_equity`, `target_weight_pct`
- `revenue_yoy_pct_last_quarters: list[Decimal]` (most recent first), `roce_pct_last_quarters: list[Decimal]`, `guidance_cut_pct: Decimal | None`
- `governance_events: list[str]` (event codes), `pledge_pct: Decimal | None`, `pledge_pct_prev_quarter: Decimal | None`
- `pe_own_band_percentile: Decimal | None`, `pe_vs_sector_multiple: Decimal | None`
- `below_200dma: bool | None`, `thesis_status: Literal["on_track","watch","broken","unknown"]`
- `geo_revenue_hit_pct: Decimal | None` and `geo_evidence_ids: list[str]`
- `better_candidate: {security_id, z_advantage: Decimal, weeks_ahead: int, expected_edge_pct: Decimal} | None`
- `goal_within_years: int | None`

### 3. Exit card (`build_exit_card(facts, today, tax_ctx, rules) -> ExitCard`)
- One `TriggerStatus` per family: `{key, status: "ok"|"watch"|"triggered"|"unknown", value: str|None, threshold: str}`.
  - "unknown" when data is missing. A missing metric must never trigger an exit.
- `action`: `KEEP`, `TRIM_TO_TARGET`, `EXIT`, `REVIEW_THEN_EXIT`, `SWAP`, `WAIT_FOR_LONG_TERM`, `GLIDE_TO_DEBT`.
- `urgency`: `LOW`/`MEDIUM`/`HIGH`/`CRITICAL`.
- Also: `quantity`, `lots_to_sell` (from `select_lots`), `tax_estimate` (from `estimate_sale_tax`), `reason_codes`.
- **Priority** when several fire: governance (CRITICAL, EXIT) > thesis (HIGH, EXIT) > geopolitical (`REVIEW_THEN_EXIT`; only fires with ≥ 2 evidence ids) > risk stop (`REVIEW_THEN_EXIT`) > valuation (`TRIM_TO_TARGET`) > drift (`TRIM_TO_TARGET`) > swap > goal approach (`GLIDE_TO_DEBT`).
- **Trim quantity:** sell down to `target_weight_pct`. quantity = floor(current_value × (weight − target)/weight ÷ price).
- **Tax-aware wait:** for non-urgent actions (valuation, drift, swap) when any lot chosen by `select_lots` has `0 < days_to_long_term ≤ 30` and waiting saves tax (STCG − LTCG estimate > 0): action = `WAIT_FOR_LONG_TERM` with `params.wait_until` (date) and `params.tax_saved`. Urgent triggers (governance, thesis) never wait.
- **Swap:** only when `z_advantage > 1.0`, `weeks_ahead ≥ 4` and `expected_edge_pct` × position value > the tax cost of selling.
- **Core** (`is_core`): every trigger reports `ok` except drift. No stop-loss, no thesis or geo exits, ever. Card action is `KEEP` unless drift → `TRIM_TO_TARGET` (the rebalancing path).

### 4. Stress tests (`wealth/policy/stress.py`) + `rules/stress.yaml`
Scenarios (fractions; label them `assumption: true` in the YAML; the Scorekeeper replaces them with historical analogues later):
| key | equity_india | equity_global_usd | fx (USD/INR change) | gold | debt |
|---|---|---|---|---|---|
| `inr_minus_10` | 0 | 0 | +0.10 | 0 | 0 |
| `india_equity_minus_25` | −0.25 | 0 | 0 | 0 | 0 |
| `us_equity_minus_30` | 0 | −0.30 | 0 | 0 | 0 |
| `oil_shock` | −0.15 | 0 | +0.05 | +0.08 | −0.01 |
| `global_risk_off` | −0.25 | −0.25 | +0.08 | +0.15 | 0 |

- Global equity in rupees moves by `(1 + equity_global_usd) × (1 + fx) − 1`. Cash is unaffected.
- `stress_test(amounts_by_sleeve, scenario) -> {by_sleeve, total}`.
- `stress_compare(current, target, scenario) -> {current_total, target_total, difference}`. A positive difference means the target plan loses less.

### 5. Required tests (exact)
**Stress** (current: India ₹90,00,000, global ₹5,00,000, debt ₹70,00,000, gold ₹11,00,000, cash ₹10,00,000; target: India ₹78,12,000, global ₹22,32,000, debt ₹66,96,000, gold ₹18,60,000, cash 0; both total ₹1,86,00,000):
| Scenario | current | target | difference |
|---|---|---|---|
| inr_minus_10 | **+50,000.00** | **+2,23,200.00** | **+1,73,200.00** |
| india_equity_minus_25 | **−22,50,000.00** | **−19,53,000.00** | **+2,97,000.00** |
| us_equity_minus_30 | **−1,50,000.00** | **−6,69,600.00** | **−5,19,600.00** |
| oil_shock | **−13,07,000.00** | **−9,78,360.00** | **+3,28,640.00** |
| global_risk_off | **−21,80,000.00** | **−20,98,080.00** | **+81,920.00** |

**Exit cards:**
- Governance: `AUDITOR_RESIGNATION` → EXIT, CRITICAL, all quantity, no wait even if a lot turns LT in 10 days.
- Thesis: revenue YoY `[-2, -1, 5]` → thesis triggered → EXIT HIGH. `[-2, 3]` → thesis `watch`, action KEEP.
- Valuation: PE percentile 93 and sector multiple 1.6 → TRIM_TO_TARGET. Weight 7.0% with target 5.0%, value ₹1,40,000, price ₹1,000 → **quantity 40**.
- Same as above, but the chosen lot turns LT in 12 days and waiting saves tax → `WAIT_FOR_LONG_TERM` with `wait_until` = that date.
- Geopolitical: hit 25% with 1 evidence id → status `watch` (not triggered); with 2 ids → `REVIEW_THEN_EXIT`.
- Risk stop: drawdown 30%, below 200-DMA, thesis `watch` → REVIEW_THEN_EXIT; same with thesis `on_track` → KEEP.
- Missing data everywhere → all `unknown`, action KEEP.
- Core ETF with a 40% drawdown and thesis "broken" → KEEP. The same ETF drifted to 1.6× target → TRIM_TO_TARGET.
- Swap: z 1.3, 5 weeks, edge 6% on ₹2,00,000 (₹12,000) vs tax ₹8,000 → SWAP; with tax ₹15,000 → KEEP with reason `swap_not_worth_tax`.

## Out of scope
Collecting facts from market data/research (C10/C15), DB persistence of exit cards (later), prose.

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass (every number above exactly)
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
