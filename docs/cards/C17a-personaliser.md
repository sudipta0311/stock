# C17a: Personaliser (weekly per-user plan → actions), investing preferences, Actions & Plan API

**Plan:** `docs/UPGRADE_PLAN.md` §4 (the whole Policy Engine chain), §5.4 T4 (Personaliser = pure code per user), §8.4 (Home "This week", Plan screen), Appendix C (action contract, `no_llm_numbers`). Read `AGENTS.md` first. Builds on C07 (Money Map API), C10 (prices/FX), C11 (foundation, risk, allocation, debt answer), C12 (split, routes, debt picker), C12b (actions), C13/C13b (tax, exits, rebalancing, stress), C15c (Research Pack; may be empty in tests).

## Goal
For every user, every week, run the deterministic Policy Engine on their whole Money Map, store this week's ≤ 3 actions, and expose the plan and actions to the PWA. No LLM anywhere in this card.

## Scope

### 1. Investing preferences (new migration; RLS like other user tables)
`profiles.investing JSONB` validated by a new Pydantic model `InvestingPrefs`:
- `monthly_sip_capacity: Rupees`, `existing_sips: [{sleeve, instrument_category, monthly_amount}]`
- `new_money_this_week: Rupees` (default 0; the PWA sets it when the user says "I have ₹X to invest")
- `has_lrs_broker: bool`, `prefers_no_paperwork: bool`
- `satellite_pct` (0–25, default 20), `birth_year: int | None` (for age-gated instruments like SCSS)
- `lrs_used_this_fy: Rupees`

API: `GET/PUT /v1/me/investing`.

### 2. Personaliser: `wealth/policy/personalise.py` (pure orchestration over existing functions)
`personalise(user_data: UserData, market: MarketContext, research: ResearchPack | None, week: str, as_of: date) -> WeeklyPlan`, where:
- `UserData` = profile, financial context, goals, manual assets + valuations, holdings + lots, investing prefs, realised trades this FY.
- `MarketContext` = latest prices/NAVs/FX/gold, debt yields (from `rules/schemes` rates + a `rules/debt_yields.yaml` table of current FD/liquid/arbitrage yields with `as_of` and `verified: false`), index valuation percentiles (may be missing → treat as 50), geo tilts (from the research pack; default none).
- Steps:
  1. snapshot → foundation → risk band (stored profile) → `build_targets` (FX from `fx_rates`)
  2. → `split_equity` (tilts) → `select_route` → `pick_debt` per need
  3. → C12b `WeekInput` → `ActionList`
  4. Plus `band_breaches`, exit cards for satellite holdings (facts available so far; missing facts → `unknown`, never a trigger), and stress tests (current vs target) for the 5 scenarios.
- Output `WeeklyPlan`: `{week, as_of, allocation (target/current/gap by sleeve, India/global), debt_answer, foundation, route, actions, exit_cards, stress, notes, inputs_digest}`. `inputs_digest` = sha256 of the inputs, for audit/reproducibility.

### 3. Weekly job: `python -m wealth.policy.run_weekly --week 2026-W45`
- Iterates users with an owner `system_session` to list ids, then loads each user's data inside `user_session` (RLS) and computes.
- Upserts `actions` rows (status `proposed`) and a new `weekly_plans(user_id, week, plan JSONB, created_at)` table (RLS, new migration).
- Idempotent per (user, week): rerunning replaces `proposed` actions but keeps `done`/`dismissed` ones.
- One user's failure is logged (user id only) and the job continues; exit code 2 if any user failed.

### 4. API
| Method | Path | Behaviour |
|---|---|---|
| GET | `/v1/plan` | the latest `weekly_plans` row; if none, or the user's data changed since (compare `inputs_digest`), compute **on demand** with the same `personalise` (pure, fast, no LLM) and return it with `fresh: true` |
| GET | `/v1/actions?week=` | this week's actions with status |
| PATCH | `/v1/actions/{id}` | `{status: "done"|"dismissed"}`; writes `audit_log` |
| GET | `/v1/stress` | the stress section of the latest plan |
| GET | `/v1/research/ideas` | the current week's Research Pack ideas (from `research_packs`, read-only) for the Ideas tab, without dossier internals |

### 5. Tests
- `personalise` golden test: one fully specified synthetic user (EPF, PPF, FD, savings, 2 goals incl. one USD goal, CAS holdings in India equity + a US ETF, ₹1,00,000 new money, SIPs) → exact snapshot of the `WeeklyPlan` JSON committed as `tests/golden/weekly_plan_user1.json`. Regenerated only with an explicit `--update-golden` flag.
- Determinism (same inputs → same digest and plan).
- A missing price → the holding is excluded from valuation with a note, and no crash.
- No research pack → no satellite stock actions.
- Job idempotency and done/dismissed preservation.
- RLS: the job reads each user only through `user_session`.
- API on-demand compute when no stored plan exists.
- PATCH isolation (B can't mark A's action).
- **Import guard:** importing the personaliser must not import `wealth.research`, `openai` or `langgraph` (it reads `research_packs` rows from the DB as plain data).

## Out of scope
Email/push/reminders (C17b). LLM copywriting (not needed: templates in C17b).

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass; DB tests run
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. New migration up/down; OpenAPI refreshed
4. Run `python -m wealth.policy.run_weekly --week <current>` against `TEST_DATABASE_URL` after seeding the golden user; report counts
