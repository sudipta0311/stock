# C11: Policy Engine part 1: foundation check, risk profile, goal buckets, whole-wealth allocation, "how much in debt"

**Plan:** `docs/UPGRADE_PLAN.md` §4.1, §4.2, §4.3, §3.5 (the "aha" insight), §2.3 principles 1, 5, 6. Read `AGENTS.md` first. Builds on C02 (schemas, money) and C03/C04 (`Valuation`). Pure Python, no DB, no LLM.

## Goal
Given everything a user owns plus their profile and goals, produce: (1) a foundation checklist, (2) a risk profile = min(capacity, tolerance), (3) a target allocation in **rupees and percent** across `equity`, `debt`, `gold`, `cash` built from goal buckets, and (4) the direct answer to "how much should be in debt, and where should new money go?".

## Scope

### 1. Rules file `rules/allocation.yaml` (versioned; every number lives here, not in code)
```yaml
version: 1
profiles:            # long-horizon targets (midpoints of plan §4.3 ranges), percent of investable wealth
  conservative: {equity: "35",   debt: "55",   gold: "10",  cash: "0"}
  moderate:     {equity: "55",   debt: "35",   gold: "10",  cash: "0"}
  growth:       {equity: "75",   debt: "17.5", gold: "7.5", cash: "0"}
  aggressive:   {equity: "87.5", debt: "7.5",  gold: "5",   cash: "0"}
horizon:
  short_years: 3        # < 3 years → 0% equity: debt 80, cash 20
  medium_years: 7       # 3–7 years → profile capped at moderate
  short_mix: {equity: "0", debt: "80", gold: "0", cash: "20"}
glide: {full_years: 2, start_years: 6}   # medium-bucket goals: equity factor = clamp((h - 2) / (6 - 2), 0, 1); removed equity goes to debt
foundation:
  high_interest_debt_rate_pct: "12"
  emergency_months: {salaried: 6, self_employed: 9, business: 12, retired: 12, other: 6}
  term_cover_multiple_min: 10
  term_cover_multiple_target: 12
  health_cover_min: {tier1: "1000000", tier2: "700000", tier3: "500000"}   # heuristic; configurable
risk:
  bands: {conservative: 0, moderate: 30, growth: 55, aggressive: 75}      # score lower bounds (0-100)
  questions: []   # six questions, see below
```

### 2. Profile additions (extend `wealth/schemas/profile.py`, re-export schemas)
`FinancialContext`: `monthly_expenses: Rupees`, `annual_income: Rupees | None`, `dependants: int = 0`, `employment_type: Literal["salaried","self_employed","business","retired","other"] = "salaried"`, `income_stability: Literal["high","medium","low"] = "medium"`.

### 3. Risk profile: `wealth/policy/risk.py`
- Six tolerance questions in `rules/allocation.yaml`. Each has an `id`, `text` (plain English, Indian context), and 4 options with `score` 0/33/67/100. Include: reaction to a 30% fall ("₹10 L becomes ₹7 L in 3 months"), investing experience, the portion of savings you're comfortable in equity, preference between guaranteed and growth outcomes, loss-tolerance threshold, and how often you check prices.
- `tolerance_score(answers: dict[str, str]) -> Decimal` = mean of the option scores. Unknown question or option → `ValueError`.
- `capacity_score(ctx: FinancialContext, horizon_years: int, emergency_ok: bool) -> Decimal` (0–100), a documented additive formula:
  - horizon: < 3 → 0, 3–7 → 40, > 7 → 70
  - income stability: high +20, medium +10, low 0
  - dependants: 0 → +10, 1–2 → +5, 3+ → 0
  - emergency fund ok → +0, missing → −20
  - clamp 0–100
- `risk_profile(...) -> RiskResult` with `tolerance`, `capacity`, `final` = lower band, and `limited_by` (`"capacity"`/`"tolerance"`/`None`) so the UI can say *why*.

### 4. Foundation check: `wealth/policy/foundation.py`
`foundation_check(snapshot: WealthSnapshot, ctx: FinancialContext, rules) -> FoundationReport` with four items, each `{key, status: "ok"|"gap"|"unknown", required: Decimal|None, have: Decimal|None, gap: Decimal|None}`:
- `high_interest_debt`: loans with rate ≥ threshold → gap = their outstanding.
- `emergency_fund`: required = months × monthly_expenses; have = **liquid only** (`LiquidityClass.instant` cash + debt).
- `term_cover`: if dependants > 0 and annual_income is known, required = multiple_target × income + outstanding loans − liquid assets (floor 0). have = sum of term covers. `unknown` when income is missing; `ok` with required 0 when there are no dependants.
- `health_cover`: required by city tier (unknown tier → tier1); have = sum of health covers + top-ups.

### 5. Snapshot input: `wealth/policy/snapshot.py`
`WealthSnapshot` is built from a list of `(asset, Valuation)` plus market holdings `(sleeve, value, liquidity)`. It exposes:
- totals by sleeve
- totals by liquidity
- `investable_total`, which excludes `real_estate`, `protection`, physical gold with `purpose = jewellery`, and liabilities
- `locked_by_sleeve` (EPF, PPF, NSC, SSY, insurance savings, etc.)
Use `default_sleeves` and `default_liquidity` from C02.

### 6. Allocation: `wealth/policy/allocation.py`
`build_targets(snapshot, ctx, goals, profile_band, as_of, rules) -> AllocationPlan`. Horizon `h` is always `goal.target_year − as_of.year` (an integer). Reserve money in this order:
1. **Emergency fund** = required amount → 100% `cash`/`debt` (50/50).
2. **Goals with horizon < 3 years**, by priority then year: reserve `min(target_amount, remaining)` at `short_mix`.
3. **Goals 3–7 years:** reserve `min(target, remaining)` at the profile mix capped at `moderate`. Then apply the glide: equity share × `clamp((h − full_years) / (start_years − full_years), 0, 1)`; the removed equity share moves to debt. So h=3 → 25% of the equity share, h=4 → 50%, h=5 → 75%, h≥6 → 100%.
4. **Remainder** (long-term / wealth) at the profile mix.

`AllocationPlan` contains:
- target rupees per sleeve and target percent (sums to 100 ± 0.01)
- current rupees and percent per sleeve
- gap per sleeve (target − current)
- `buckets` (the reserves above, for explanation)

USD goals use the same logic (the global split is C12's job).

### 7. Debt answer: `wealth/policy/debt_answer.py`
`debt_answer(plan, snapshot) -> DebtAnswer`:
- `target_debt`, `current_debt`, `locked_debt` (EPF/PPF/etc.), `gap`
- `status` (`"enough"` if current ≥ target × 0.95, `"add"` otherwise)
- `new_money_order`: sleeves sorted by most under target
- `message_key` plus `params`, for example `{"current_pct": "41.0", "target_pct": "33.0"}`

No prose generation here; the PWA and Communicator turn keys into text.

### 8. Tests (all deterministic)
- Profile mixes in the YAML each sum to 100.
- Tolerance and capacity scoring, with the `limited_by` explanation.
- Foundation: the emergency fund ignores locked assets; term cover `unknown` without income; the high-interest debt threshold boundary (exactly 12% → gap).
- Allocation worked example (write it out in a test docstring), as_of 2026-11-01, profile moderate:
  - Investable ₹50,00,000.
  - Monthly expenses ₹50,000 (salaried → ₹3,00,000 emergency).
  - Goal A: car, ₹5,00,000 in 2028 (short).
  - Goal B: child education, ₹20,00,000 in 2032 (medium).
  - Expected targets: emergency ₹3,00,000 → cash 1.5 L + debt 1.5 L; goal A ₹5 L at short_mix → debt 4 L, cash 1 L; goal B ₹20 L at moderate → equity 11 L, debt 7 L, gold 2 L; remainder ₹22 L at moderate → equity 12.1 L, debt 7.7 L, gold 2.2 L.
  - **Totals: equity ₹23,10,000; debt ₹20,20,000; gold ₹4,20,000; cash ₹2,50,000** (sum ₹50,00,000).
- Debt answer for a snapshot whose current debt (EPF + PPF + FD) is 41% vs a target of 33% → `status == "enough"`, and `new_money_order` starts with the most-underweight sleeve.
- Glide: a ₹10,00,000 medium goal with h = 4 (profile moderate) reserves equity ₹2,75,000 (27.5%), debt ₹6,25,000 (62.5%), gold ₹1,00,000 (10%). With h = 6 it reserves equity ₹5,50,000, debt ₹3,50,000, gold ₹1,00,000.

## Out of scope
India vs global split, routes, debt instrument choice, buying actions (C12). Tax and exits (C13).

## Acceptance
1. `.venv\Scripts\python -m wealth.schemas.export`, then `.venv\Scripts\python -m pytest` → all pass (including the worked example exactly)
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
