# C56: "Our view" uses the data we have

**Source:** the owner, 2026-11-08: "for most of my stocks the suggestion is keep and no data are available...pls check if it is correct". The reviewer read the Saturday stock-check run (2026-W45) and the code.

## Why
The weekly stock checks ran: 13 held shares checked, 5 of them outside the screener lists. Coverage on all 931 checked shares: `pe_percentile` **0**, `pe_vs_sector` 8 (held shares only), `dma200` 450, `revenue_yoy_pct` 859.

On a held share's page, "Why this suggestion?" lists 7 checks. Most say "Not enough information", and the verdict is "Keep" almost by default:

| Row | Why it is "Not enough information" |
|---|---|
| Goal timing | `HoldingFacts.goal_within_years` is never set by any caller. It can never have data. |
| Company strength | Needs `pe_percentile`. That needs 36 monthly P/E points (`valuation.py`), but prices are kept for 400 days and Indian results go back about 6 quarters. It can never have data. |
| Price history | `_valuation_trigger` needs both `pe_percentile` and `pe_vs_sector`, so the same gap blocks it. |
| Company events | `known` only when an event or a pledge figure exists. The daily watcher (`sentinel.py`) checks every held IN/US share, but a quiet company still reads "unknown". |
| Price and cost | `_risk_trigger` needs all three inputs (drawdown, below 200-day average, thesis), even when one known input already rules the trigger out. US shares have 1 price a week (S&P 500 file) plus 1 a day (Alpha Vantage), so no US share has 200 sessions. |
| Company results | `stock_checks.py` loads revenue only for screened shares, and fundamentals are fetched only for NIFTY 500 / S&P 500 members. Held shares outside those lists never get results. |

"Add more" also needs `pe_percentile`, so it can never appear. The plan's held add-more buys need it too (`personalise.py`, the IN and US paths).

## Read
- `AGENTS.md` (UX rules, money rules, commands).
- `src/wealth/policy/exits.py`, `src/wealth/policy/holding_review.py`, `src/wealth/policy/personalise.py` (`_exit_cards`, the two held add-more paths), `src/wealth/policy/data.py` (`_stock_check_inputs`), `rules/exits.yaml`.
- `src/wealth/research/stock_checks.py`, `src/wealth/research/sentinel.py` (`run_sentinel`), `src/wealth/research/graph.py` (the neutral `valuation_percentile` of 50).
- `src/wealth/marketdata/refresh.py` (`_fetch_nse_fundamentals`, `_fetch_edgar`, `_fetch_abroad_us_stocks`), `src/wealth/marketdata/sources/alpha_vantage.py`, `src/wealth/alerts/health.py` (`_SOURCE_WORKFLOWS`).
- `apps/web/src/money/HoldingPage.tsx` (`OurView`), `copy/en.yaml` (`holding_review`).

## Scope

### 1. A check is unknown only when missing data could change it
Rules with "and" can be decided once one known part fails. Apply this in `exits.py`:
- **Price and cost** (`_risk_trigger`): the result is "ok" when any known input rules out even "watch":
  - the drawdown is known and above the watch level (−25% × 0.8 = −20%);
  - `below_200dma` is known and False;
  - the thesis is known and is not the required status.

  "triggered" and "watch" need every input known, as today. Otherwise "unknown".
- **Price level** (`_valuation_trigger`):
  - both inputs known: today's rule;
  - own-history percentile unknown, sector multiple known: "watch" when the multiple is above `sector_multiple` (1.5), else "ok". This never returns "triggered", so it never creates a sale on its own;
  - sector multiple unknown, own percentile known: "ok" when the percentile is below the watch level (90 × 0.8), else "unknown";
  - both unknown: "unknown".
- A sector-only "watch" gets its own reason code, `valuation_vs_sector`: "The price looks high compared with similar companies." The own-history reason keeps its text.

### 2. Company events read "ok" when the watcher looked
- At the end of `run_sentinel`, store one `MarketdataRun` row. Its counts hold only statuses and numbers, never symbols or text: `{"sentinel": {"IN": <nse status>, "US": <edgar status>, "targets": n}}`. Status: succeeded, partial (one market unavailable) or failed. No migration: the table exists.
- Data health must name the right workflow for this source. Add `sentinel` → "Daily safety checks and reminders" to `_SOURCE_WORKFLOWS`. No other health check may change because of these rows.
- `HoldingFacts` gains `events_watched: bool = False`. In `_governance_trigger`, an empty event list with `events_watched` is "ok".
- The plan sets `events_watched` for a held share when the latest sentinel run is at most `events_watch_max_age_days` old and its status for the share's market is "available". Add that rule to `rules/exits.yaml` with the value 3.

### 3. Company strength and "Add more" use what exists
Add these rules to `rules/exits.yaml`, under `accumulate`:
- `max_pe_vs_sector: "1.2"`;
- `weak_quant_decile_max: 3`.

**Company strength** uses the research score alone:
- unknown when the share has no `quant_decile`;
- "ok" when the decile is at least `min_quant_decile`;
- "watch" when it is at most `weak_quant_decile_max`;
- "ok" otherwise.

**Price is reasonable:**
- `pe_percentile` known: true when below `max_pe_percentile`;
- otherwise `pe_vs_sector` known: true when at most `max_pe_vs_sector`;
- otherwise false.

Use it in place of the `pe_percentile` condition in `can_add_more`. Everything else in `can_add_more` stays.

**Held add-more buys in the plan** (the IN and US paths in `personalise.py`):
- use the same test, so a share shown as "Add more" can be bought;
- when `pe_percentile` is missing, pass the neutral valuation percentile the research ideas use (50);
- define it once as a named constant in `wealth.policy`, and make `graph.py` use the same constant.

### 4. Drop the row we never measure
- Remove "Goal timing" from the holding review's checks. The plan handles goals.
- Keep the `goal_approach` trigger in the exit engine.
- Remove the copy key if nothing else uses it.

### 5. Held shares get results and longer US prices
- **Results:** `stock_checks.py` computes revenue changes for held shares too, not only screened ones. Set `sources.fundamentals` whenever revenue was read.
- **Fundamentals:** fetch them for held equities as well. NSE fundamentals targets = current NIFTY 500 members plus held IN equities with a symbol. EDGAR targets = current S&P 500 members plus held US equities. No duplicates.
- **US prices:** Alpha Vantage's compact daily call already returns up to 100 sessions. For held US shares (`_fetch_abroad_us_stocks`), store every valid close on or before `on`, not only the newest. Keep the request count and budget unchanged. Curated funds stay as today.
  - Add a parser that returns the series. Keep `parse_daily_series` for existing callers, or build it on the new parser.
- **Run summary:** the stock-check summary JSON gains `held_coverage`, counts over held rows only: `price`, `dma200`, `revenue_yoy_pct`, `pe_vs_sector`, `pe_percentile`, `quant_decile`.

### 6. The screen says how much was checked
In `OurView`:
- **Summary line:** when the verdict is shown and there is no reason line, show one quiet line under it:
  - "No warning signs in {ran} of {total} checks." (`ran` = rows with ok, watch or triggered);
  - "No warning signs in any of the {total} checks." when all ran.
- **How we check:** list only rows with a known status. Then add one muted line for the rest: "Waiting for data: Price level, Company strength." Leave the line out when nothing is unknown.
- **Copy:** rename the `price_history` row label to "Price level". It now covers the sector comparison too.
- All new text goes in `copy/en.yaml`. Regenerate the catalogue.
- The "not checked" message for Danish and other non-IN/US shares stays as it is.

## Tests
- **Python:**
  - `exits.py`: each part of the new logic. Risk: a known drawdown of −5% with unknown 200-day average and unknown thesis gives "ok", and the same goes for `below_200dma` False and for an on-track thesis. Valuation: sector-only 1.6 gives "watch", 1.1 gives "ok", and the trigger never fires without the own percentile. Governance: watched and quiet gives "ok"; not watched gives "unknown".
  - `holding_review.py`:
    - add more with `pe_percentile` None and `pe_vs_sector` 1.1;
    - keep with `pe_vs_sector` 1.3;
    - company strength at deciles 9, 5 and 2, and with no decile;
    - no `goal_timing` row.
  - `personalise.py`: an IN and a US held add-more share without `pe_percentile` enter the buy ideas with the neutral percentile.
  - DB (`@pytest.mark.db`):
    - a held share outside every universe, with synthetic revenue facts, gets `revenue_yoy_pct`;
    - `held_coverage` counts held rows only;
    - a sentinel run stores its row;
    - the plan reads `events_watched` from it, ignoring a run older than 3 days and a market whose status is not "available".
  - `refresh.py`: the fundamentals targets include held equities once. The Alpha Vantage held-stock path stores every close from a recorded compact response.
  - The golden weekly plan: regenerate it the way its test documents. Explain each changed field in the result file.
- **Vitest:**
  - the summary line in both forms;
  - known rows only, plus the "Waiting for data" line;
  - no line when every check is known;
  - "Price level" as the label.
- **e2e:** `holding-review.spec.ts` asserts "Not enough information" for company results. Update it to the new screen: the row is absent and named in "Waiting for data". Search `e2e/tests` for other old review text. Codex's sandbox can't start the e2e stack, so run `npx playwright test --list` and say so in the result file.

## Out of scope
- A real 3-year own P/E range. It needs longer monthly price history and older Indian results, so it is a later card.
- Danish and EU share checks. New data sources. Changes to sell rules other than those above.
- Any change to how much money a buy uses, beyond letting held "Add more" shares in.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`

Inside `apps/web`:
- `npm run gen:api`
- `npm run gen:copy`
- `npm run typecheck`
- `npm run lint`
- `npm run format:check`
- the full `npm test -- --run`
- `npm run build`
- `npm run budget`

In `e2e`: `npx playwright test --list`.
