# C73: practical sell advice for weak shares

**Source:** the owner, 2026-11-14. Today's rules show "Keep" on every share they hold. Two cases look wrong to them:
- a share that ranks weak in our weekly research and costs about 5× similar companies;
- a share down more than 60% from cost and still falling, which stays "Keep" because nothing says its sales are falling.

The owner agreed to both changes below: "be a bit practical; if there is no point in holding it, I can sell". This is an owner-approved rules change: update tests that pin the old behaviour, and list each one in the result.

## Read
- `AGENTS.md`: the UX rules (plain words, at most three reasons, ≤ 20 words per sentence) and the money rules.
- `rules/exits.yaml`.
- `src/wealth/policy/exits.py`: `HoldingFacts`, `_risk_trigger`, `_position_drawdown`, `build_exit_card`, `_sale_size`, the tax-wait block.
- `src/wealth/policy/holding_review.py`: `holding_review_decision`, `_reason_rows`.
- `src/wealth/policy/personalise.py`: `_exit_cards`, `_exit_action`, `_weekly_action_key`, `_merge_weekly_actions`, `_holding_review_tax`.
- `src/wealth/alerts/brief.py`: how sell actions appear in the weekly email.
- `apps/web/src/money/HoldingPage.tsx` and `apps/web/src/home/HomePage.tsx` (`sellDetails`, the sell sheet).
- `copy/en.yaml`: `reasons`, `holding_review`, the action labels and `trigger_keys`.
- `docs/UPGRADE_PLAN.md`: §4.9 (the exit table) and Appendix A.
- Tests: `tests/policy/test_exits.py`, `tests/policy/test_holding_review.py`, `tests/policy/test_personalise.py`, `tests/alerts/` (golden briefs), and the HoldingPage, HomePage and copy vitest files.

## Scope

### 1. New satellite rule `no_reason_to_hold` (verdict "Sell")
Add to `rules/exits.yaml` under `satellite:`:
```yaml
  # Research ranks the company weak, and it also looks expensive or keeps falling at a loss.
  no_reason_to_hold:
    max_quant_decile: 3
    own_band_percentile: "90"
    sector_multiple: "1.5"
    loss_from_cost_pct: "20"
    below_dma: 200
    action: EXIT
    urgency: MEDIUM
```
- Add `quant_decile: int | None = None` to `HoldingFacts`, validated as an int from 1 to 10 or None (a bool is invalid). `_exit_cards` passes `check.quant_decile`.
- New trigger `no_reason_to_hold`, for satellite holdings only (core: `ok`, "not applicable to core"):
  - `unknown` when `quant_decile` is None;
  - `ok` when `quant_decile > max_quant_decile`;
  - `triggered` when the decile is weak **and** at least one of these holds:
    - **expensive:** own PE band percentile > 90, **or** PE vs sector > 1.5×. Either one is enough here, unlike `valuation_stretch`;
    - **falling at a loss:** `below_200dma is True` **and** drawdown from cost ≤ −20%. Use `_position_drawdown`, so `average_cost` works.
  - `ok` when the decile is weak but neither holds. This trigger never uses `watch`.
- Priority in `build_exit_card`: governance > thesis > geopolitical > **no_reason_to_hold** > risk_stop > valuation > drift. When it fires: action `EXIT`, urgency `MEDIUM`, reason code `no_reason_to_hold`.
- **Tax wait applies** (the sale is not urgent): add `no_reason_to_hold` to `tax_wait.applies_to`. The existing wait (≤ 30 days to long-term with a positive saving) may turn this `EXIT` into `WAIT_FOR_LONG_TERM`, as it does for trims. Governance and thesis exits still never wait.

### 2. Risk stop no longer needs a business-results warning
- Remove `requires_thesis` from `risk_stop` in the rules and the code. New rule: drawdown ≤ −25% from cost **and** below the 200-day average → `triggered` (`REVIEW_THEN_EXIT`, verdict "Think about selling"). `watch` at ≤ −20% and below the average.
- It's `known` when the drawdown and `below_200dma` are both known; the thesis status no longer matters. A broken thesis still exits first through `thesis_break`.
- Update the threshold and value strings: no thesis text.

### 3. Verdicts, reasons and checks (`holding_review.py`)
- **Verdict:** `EXIT` → `sell`, as today. A `WAIT_FOR_LONG_TERM` card whose reason codes include `no_reason_to_hold` → `sell`, not `sell_some`. The tax line already shows the wait.
- **Reasons when `no_reason_to_hold` caused the verdict** (sell or the wait). Use exactly these rows, in this order, at most 3:
  1. `company_strength_weak` (existing copy);
  2. if expensive: `valuation_stretch` (own band) or `valuation_vs_sector` (sector only), with today's params;
  3. if falling at a loss: `risk_stop` with `drawdown_pct` and `below_dma200`;
  4. `india_shares_over_target` (below).
- **New reason `india_shares_over_target`:** append it last, within the three-row limit, when all of these hold:
  - market `IN`;
  - the verdict is `sell`, `review` or `sell_some`;
  - current Indian shares ≥ target (the same inputs `india_shares_enough` uses).
  
  `india_shares_enough` stays as it is, for `keep` only.
- **Checks:** `company_strength` is `triggered` when `no_reason_to_hold` fired; otherwise as today.

### 4. Sells reach Home even without buy dates
Many broker files have no buy dates, so `HoldingFacts.lots` is often empty and every whole-holding sale today has quantity 0. As a result, no action appears.
- Add `quantity_held: Decimal | None = None` to `HoldingFacts` (validated ≥ 0). `_exit_cards` sets it to the holding's units.
- `_sale_size`: for `EXIT` and `REVIEW_THEN_EXIT`, use the lots' total when there are lots, else `quantity_held` (or 0). Leave `TRIM_TO_TARGET` and `SWAP` unchanged: they still need lots.
- `_exit_action`:
  - accept an `EXIT` with urgency `MEDIUM` only when its reason is `no_reason_to_hold`. `EXIT` with `CRITICAL`/`HIGH` behaves as today;
  - accept `EXIT`/`REVIEW_THEN_EXIT` with quantity > 0 and **no** lots to sell. Then `lots_to_sell` is `[]` and `amount` is `round_paise(quantity × price)` in INR, not `gross_proceeds`;
  - the tax estimate stays empty; the holding review already says `tax_not_estimated`.
- **Weekly order:**
  - a `SELL_EXIT` below `HIGH` urgency sorts with the other exits (group 3), before `SELL_TRIM` and `REVIEW_EXIT`;
  - among sells of equal urgency, `EXIT` comes before `REVIEW_THEN_EXIT`, then the larger amount;
  - still at most **one** sell action a week.
- **Home:** the sell sheet without a tax figure shows the quantity and no empty tax label. If it has no tax line at all, add "We can't estimate tax without your buy dates." (new copy key). "Sold it" works as today.
- **Weekly email:** the sell title (symbol and shares) works without lots. The reason text comes from the top-level `reasons` copy.

### 5. Copy (`copy/en.yaml`, then `npm run gen:copy`)
- `reasons.no_reason_to_hold`: "It ranks low in our research and looks expensive or keeps falling."
- `reasons.risk_stop` (top level, used by Home and the email): "The price is far below your cost and still falling."
- `holding_review.reasons.risk_stop`: "The price is {drawdown_pct}% below your cost and still falling."
- `holding_review.reasons.india_shares_over_target`: "Your plan already has more than enough in Indian shares."
- `trigger_keys.no_reason_to_hold`: "Weak and expensive, or falling"
- Any other place that lists trigger keys or reason codes, such as copy coverage tests, gets the new codes. No jargon from the AGENTS.md list on screen.

### 6. Plan document
Update `docs/UPGRADE_PLAN.md` §4.9 (a new table row; the risk stop row loses the thesis condition) and Appendix A to match the YAML.

## Tests (synthetic data, no network)
- **exits:**
  - weak (decile 2) + 1.6× sector → `EXIT`/`MEDIUM`/`no_reason_to_hold`;
  - weak (decile 3) + 22% below cost + below the average → `EXIT`;
  - weak + 1.2× sector + 10% above cost → `KEEP`, trigger `ok`;
  - decile 6 + 2.0× sector → not triggered;
  - decile None → `unknown`;
  - a governance event still wins;
  - a core holding gets `ok`;
  - a lot 20 days from long-term with a gain → `WAIT_FOR_LONG_TERM`.
- **risk stop:**
  - 30% below cost, below the average, thesis `on_track` → `REVIEW_THEN_EXIT`;
  - the same above the average → `ok`;
  - 21% below and below the average → `watch`;
  - thesis `unknown` with drawdown and average known → evaluated.
- **holding review:**
  - the reason rows and order for the expensive and the falling case;
  - `india_shares_over_target` shown for Indian sells and reviews when India is over target; not shown for US shares, under target, or keep;
  - the three-row limit;
  - `company_strength` is `triggered`;
  - a wait card with `no_reason_to_hold` → `sell`.
- **personalise:**
  - a holding with undated lots that has a cost and triggers `no_reason_to_hold` gives one `SELL_EXIT` action. Its quantity is the units held and its amount is units × INR price, rounded to paise. Use at least one awkward amount, e.g. 1234.567891 units or price;
  - `lots_to_sell` is empty and the holding review's tax note is `tax_not_estimated`;
  - two sells (EXIT and REVIEW) → only one action, the EXIT;
  - the sell sorts after foundation and monthly-change actions and before buys;
  - existing HIGH/CRITICAL exits are unchanged.
- **Weekly email:** a sell without lots renders. If a golden file changes, check that only this card's changes differ, then regenerate with `--update-golden` and say so in the result.
- **PWA:**
  - HoldingPage shows "Sell" with the new reason rows;
  - the Home sell action without a tax figure renders cleanly;
  - copy tests.

## Out of scope
- Other thresholds and rules.
- The research screener and deciles.
- Database migrations.
- New data sources.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
- In `apps/web`: `npm run gen:api`, `npm run gen:copy`, `npm run typecheck`, `npm run lint`, `npm run format:check`, the full `npm test`, `npm run build` and `npm run budget`.
