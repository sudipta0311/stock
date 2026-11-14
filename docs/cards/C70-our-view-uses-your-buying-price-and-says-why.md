# C70: "Our view" uses your buying price and says why

**Source:** the owner, 2026-11-14: "for all stock the suggestion was keep". The reviewer's look-only read of the live plan found that "Keep" is the correct result of the rules for all 13 shares, but two things make it weaker or harder to trust.

1. **The buying price is ignored.**
   - Money shows a buying price and profit or loss for 11 of the owner's Indian shares (C63). One of them is down more than 60%.
   - The plan's exit engine sees no cost for any of them: `_normalised_lots` in `src/wealth/policy/personalise.py` drops every lot without `acquired_on` and notes `holding_lot_tax_facts_missing`, and broker files give an average price without dates.
   - So the "Price and cost" check (`risk_stop`) never knows how far a share is below its cost.
   - The date matters for tax, not for the price-versus-cost check.
2. **"Keep" never says why there is no "Add more".**
   - Indian shares are about 84% of the owner's investments, against a plan target of about 14%. So `add_more_room_inr` is 0 for every Indian share and "Add more" can never appear.
   - Correct, but the screen shows only "Keep", so 13 identical "Keep"s look broken.

## Read
- `AGENTS.md` (UX rules: plain words, at most three reasons, short sentences).
- `src/wealth/policy/personalise.py`: `_exit_cards`, `_normalised_lots`, `_add_more_room`.
- `src/wealth/policy/exits.py`: `HoldingFacts`, `_position_drawdown`, `_risk_trigger`.
- `src/wealth/policy/holding_review.py`: `holding_review_decision`, `_reason_rows`, `_drawdown_pct`.
- `src/wealth/api/money_map.py` (`_holding_cost_response`): how Money works out cost, so both screens agree.
- `apps/web/src/money/HoldingPage.tsx` and the `holding_review` copy in `apps/web/src/copy/en.json`.
- `tests/policy/test_holding_review.py`, `tests/policy/test_exits.py`, and the holding-review tests in `tests/policy/test_personalise.py`.

## Scope
1. **Use a known buying price without a date.**
   - Add an optional `average_cost: Decimal | None` (INR per unit) to `HoldingFacts`.
   - In `_exit_cards`, set it when **every** lot of the holding with a positive quantity has a cost: an INR `cost_per_unit`, or a `cost_per_unit_inr` with no FX gap. Use the same rule as Money's cost, so the two screens agree.
   - `_position_drawdown` and `_drawdown_pct` use the lots when there are any; otherwise they use `average_cost`.
   - Tax estimates keep using dated lots only: a lot without a date still gives no tax figure, and the existing note stays.
2. **Say why "Keep" is not "Add more" when the plan already has enough Indian shares.**
   - Add the reason code `india_shares_enough` when all of these hold:
     - the share is Indian;
     - the verdict is `keep`;
     - `add_more_room_inr` is below the minimum buy;
     - the plan's current Indian-shares amount is at or above its target.
   - Put it last, inside the existing three-row limit.
   - Copy: "Your plan already has enough in Indian shares, so we don't suggest adding more."
   - Do not add a reason when room is short for any other reason (stock or sector cap).
3. **Fix the US market sentence.** The `ideas.index_market_*` copy says "compared with the last 3 years". The percentile uses up to five years of stored history, so say "compared with recent years".
4. Nothing else changes: no rules, thresholds or other verdicts.

## Tests (no network)
- **Price and cost without dates:** a holding with undated INR lots that have costs, 30% below cost, below its 200-day average, with revenue down last quarter, gets `risk_stop` triggered and the verdict `review`. The same holding with no costs keeps today's result.
- **Mixed lots:** one lot without a cost means no `average_cost`.
- **Foreign currency:** a lot with `fx_missing` means no `average_cost`.
- **Tax unchanged:** tax notes and estimates for undated lots stay as they are today.
- **`india_shares_enough`:**
  - shown when India is over target and room is 0;
  - not shown when India is under target, for US shares, or when the verdict is not `keep`;
  - the three-reason limit still holds.
- **PWA:** the holding page shows the new reason under "Why this suggestion?" or as the first line, matching how reasons are shown today. Update the copy and its tests.

## Out of scope
- Rules in `rules/exits.yaml`.
- The exit engine's thresholds.
- New data sources.
- Any database migration.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
- In `apps/web`: `npm run gen:api`, `npm run gen:copy`, `npm run typecheck`, `npm run lint`, `npm run format:check`, the full `npm test`, `npm run build` and `npm run budget`.
