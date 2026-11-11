# C63: buying prices show, and "Show all" opens one list

**Source:** the owner, 2026-11-11, after importing an ICICI Direct portfolio file (C61): "I am not seeing the buying price in stock while the uploaded broker statement has all the details... also when I am expanding the stock list (see more) it is expanding both stock and MF making it difficult to navigate".

## What is wrong
1. **The buying price is never shown.**
   - `apps/web/src/money/HoldingPage.tsx` (around line 265) reads `cost_basis ?? cost_value` and `profit_loss ?? pnl` from the `/v1/networth` holding.
   - `HoldingValuationResponse` in `src/wealth/api/money_map.py` has none of these fields. So every share and fund shows "Buying price not recorded", even when its lots have `cost_per_unit` (broker imports and C61's enrich step store it).
2. **One switch for every list.** `MoneyMapPage.tsx` keeps one `showAllHoldings` state for all four `HoldingsGroup`s (funds, shares, bonds and gold bonds, other). "Show all" on one opens every list.

## Read
- `AGENTS.md` (UX rules: plain numbers, lakh/crore for summaries, short copy).
- `src/wealth/api/money_map.py`: the lot query (around line 778), how each lot becomes a holding row, `_merge_lots`, `HoldingValuationResponse`, and `/v1/networth`.
- `src/wealth/db/models.py`: `HoldingLot` (`cost_per_unit`, `cost_per_unit_inr`, `currency`).
- `apps/web/src/money/HoldingPage.tsx` and `apps/web/src/money/MoneyMapPage.tsx` (`HoldingsGroup`).
- The tests next to each.

## Scope
1. **The API returns what each holding cost.** Add typed fields to `HoldingValuationResponse`:
   - `cost_inr`: total buying cost in rupees;
   - `avg_cost_inr`: per unit, `cost_inr / quantity`;
   - `profit_loss_inr`: `value - cost_inr`.

   **Rules:**
   - **A lot's cost in rupees:**
     - an INR lot: `cost_per_unit × quantity`;
     - another currency: `cost_per_unit_inr × quantity`, or unknown when that is missing.
   - **A holding's cost is known only when every lot in it has a cost.** Otherwise all three fields are `null`. Merge across lots the same way `_merge_lots` merges value.
   - **Profit or loss:** `profit_loss_inr` is `null` when the value is unknown.
   - **Money math:** use `Decimal` throughout and round only for the response, as the other rupee fields do.
   - **Recorded buys** (`added_by_you`) are unchanged.
   - Regenerate `schemas/openapi.json` and the PWA API types.
2. **The holding page shows them.**
   - **Typed fields:** `HoldingPage.tsx` uses the new typed fields instead of the guessed names.
   - **Profit or loss:**
     - When `profit_loss_inr` is known, show it rounded like the value, with a sign, plus a percentage of cost (for example "+₹2.9 lakh · +220%").
     - When the cost is unknown, keep "Buying price not recorded".
   - **In "Holding details":**
     - "Bought for" (`cost_inr`);
     - "Average buying price" (`avg_cost_inr` a share or a unit, in full rupees and paise).
   - **No new warnings or notes.**
3. **Each list opens on its own.**
   - **Separate switches:** each `HoldingsGroup` has its own "Show all" state.
   - **"Show fewer" keeps your place.** When it collapses a list whose heading is above the screen, scroll that heading into view.
   - **Unchanged:** labels and the 8-row default.
4. **E2E.**
   - **Buying price:** in `e2e/tests/icici-portfolio-import.spec.ts`, after confirming, open one imported share and check that its profit or loss shows instead of "Buying price not recorded".
   - **Lists:** add a check (in an existing Money spec) that "Show all" on shares leaves funds collapsed. The seeded account needs more than 8 of each; seed it through the existing e2e database path if needed.
   - The reviewer runs e2e.

## Tests
- **API (`@pytest.mark.db`).**
  - Every lot has a cost: the three fields are correct across two lots.
  - One lot without a cost: all three are `null`.
  - A USD lot with `cost_per_unit_inr`: its cost uses that value. Without it: `null`.
  - No price: `profit_loss_inr` is `null` while `cost_inr` is set.
- **PWA.**
  - A holding with cost shows profit or loss and the two details rows.
  - Without cost, it shows "Buying price not recorded".
  - On Money, "Show all" on shares does not open funds.
  - "Show fewer" calls `scrollIntoView` when the heading is above the screen.
- **No network or LLM anywhere.** All data is synthetic.

## Out of scope
- Profit or loss on the Money list rows.
- Asking for missing buying prices.
- Tax.
- Returns over time.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
- In `apps/web`, run each of these and make it pass:
  - `npm run gen:api`
  - `npm run gen:copy`
  - `npm run typecheck`
  - `npm run lint`
  - `npm run format:check`
  - `npm run build`
  - the full `npm test`
  - `npm run budget`
