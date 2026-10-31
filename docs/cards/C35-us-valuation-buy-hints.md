# C35: US market valuation and a buy hint on each research idea

**Requested by the product owner:** "yes" to two proposals: (1) a US market valuation, so larger US buys are spread out more when the US market is expensive, and (2) a code-decided buy hint on each research idea ("Buy in 3 parts" / "Wait for a better price").
**Builds on:** C15a/C15b (fundamentals, screener), C16a/C16b (research pack, `MacroGeoToolbox`, index percentiles), the Denmark picks (`rules/eu_etfs.yaml`, `wealth.policy.personalise`), and the reviewer's 2026-10-31 data fixes (SSGA implied closes with source `SSGA_SPY_IMPLIED`, GICS sectors, the `nse_history` source, year-to-date EDGAR quarters). Read `AGENTS.md` first.

## How it works today (verify before changing)
- **India:** the `index_pe` refresh source stores NSE index P/E rows in `index_valuations`. `store.latest_index_valuation_percentile` ranks the latest P/E against the stored history (5-year retention). `MacroGeoToolbox.policy_index_percentiles` fills only `IN`. The pack's `index_valuation_percentiles` reach the policy through `wealth.policy.data._research_index_percentiles`, then `WeekInput.index_valuation_percentile`, and lump sums are staged by `rules/buying.yaml` `lumpsum` (6 parts when the percentile is at least `expensive_valuation_percentile`).
- **US:** no value. `personalise` substitutes 50 and adds the note `index_valuation_percentile_missing:US`.

## Scope

### 1. S&P 500 valuation from data the app already loads
- **New refresh source `us_index_pe`** in `wealth.marketdata.refresh`. Add it to `_SOURCES`, and to `.github/workflows/marketdata-weekly.yml` directly after `edgar`. Update `tests/test_deploy_config.py`, which asserts that list.
  - **Current point.** S&P 500 trailing P/E = Σ market cap / Σ trailing-twelve-month net income, over current `sp500` universe members.
  - **Market cap** = latest `SSGA_SPY_IMPLIED` close (at most 20 days old) × latest known `shares_outstanding`.
  - **Net income** follows the quarter rules in `wealth.research.factors`. Reuse those helpers; don't re-implement trailing-twelve-month logic.
  - Include loss-makers in the earnings sum, as an index P/E does.
  - **Coverage:** members with both inputs must hold at least 90% of the summed SPY weight. Otherwise store nothing and report an error code in the refresh result.
  - **Storage:** `IndexValuation(index_name="S&P 500", as_of=<SPY holdings date>, pe=…, source="COMPUTED_SPY_EDGAR", source_url=<SSGA holdings URL>)`. P/B and dividend yield stay empty.
- **History backfill (monthly, within the 5-year index-valuation retention),** so the percentile means something now:
  - Rebuild monthly points from point-in-time data the app can load:
    - SPY's daily NAV history: `SPY_NAV_HISTORY_URL`, the SSGA NAV workbook. `parse_spy_total_net_assets` keeps only total net assets today; extend parsing to NAV without breaking it.
    - EDGAR net income of the current constituents, using only facts with `known_at` at or before each month end.
  - **Calibration:** calibrate one scale factor, k = today's computed Σ market cap / today's SPY NAV. Then historical P/E(t) = k × NAV(t) / Σ trailing-twelve-month net income(t).
  - Label these rows `source="COMPUTED_SPY_EDGAR_BACKFILL"`. Document the survivorship bias (current constituents only) in the docstring and the result file.
  - Backfill only what the NAV history covers. Never invent data. Re-running must not duplicate rows (one row per index and date).
- **Wiring:**
  - `MacroGeoToolbox.policy_index_percentiles` fills `US` from "S&P 500", exactly as it fills `IN` (history label, count, as_of).
  - **Policy:** make sure lump-sum staging for global exposures (`global_all_world`, `global_sp500`, `global_nasdaq100`, `global_developed`) uses the US percentile. Check `wealth.policy.actions` and `personalise`, and wire it if the abroad path ignores it. With a US value present, the "missing" note must not appear.

### 2. A buy hint on each research idea (code decides, no model)
- **Rules** go in `rules/buying.yaml` under `entry_hint`, each threshold with a one-line comment giving its reason:
  - `wait_above_200dma_pct: "15"`: the price is more than 15% above its 200-session average, so wait.
  - `wait_market_percentile: "90"`: the idea's market index percentile is at least 90, so wait.
  - `more_parts_market_percentile: "80"`: at least 80 means buy in `lumpsum.stp_tranches_expensive` parts.
  - Otherwise buy in `satellite.tranches` parts.
  - **US ideas** have no 200-session price history yet (weekly closes only). Skip that check, and record a reason code saying so.
- **Where to compute it:** when the pack is built (`pack_node` in `wealth.research.graph`), from stored inputs only. Use the price factors the dossier already has and the pack's `index_valuation_percentiles`.
- **Format:** add `entry_hint` next to each idea in the pack, as `{"code": "buy_in_parts" | "wait", "parts": 3 | 6 | null, "reasons": [codes]}`. Don't change `SatelliteIdea`'s existing fields; the policy depends on them.
- **API:** expose it on `ResearchIdeaSummary` in `GET /v1/research/ideas`. Ideas from older packs have no hint and must still load.
- **PWA** (Ideas → Research ideas card): one short line per idea.
  - "Buy in 3 parts over 3 months"
  - "Buy in 6 parts. Markets look expensive."
  - "Wait for a better price"
  - A closed "Why?" shows plain reasons from copy. No jargon: no "200dma", "percentile" or "tranche". Copy goes in `en.json` and must pass `gen:copy` and the copy rules (at most 20 words per sentence).
- Never show an amount or a weight. This is timing only.

## Tests (no network)
- **Python:**
  - the index P/E sums, with one loss-maker
  - coverage below 90% stores nothing
  - backfill calibration of k, and re-running adds no rows
  - the US percentile reaches the pack and the policy: a lump sum with US percentile 85 is split into 6 parts, and the missing-US note is gone
  - each entry-hint rule, and the US no-history path
  - a pack without hints still loads in the API
- **PWA:** the three hint lines, and "Why?" closed by default.
- **Deploy config:** the weekly workflow lists `us_index_pe` after `edgar`.

## Acceptance
- `.venv\Scripts\python -m pytest`: full run, with DB tests via `TEST_DATABASE_URL`
- `.venv\Scripts\python -m ruff check --no-cache src tests scripts` and `ruff format --no-cache --check src tests scripts`
- `.venv\Scripts\python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- Write `docs/cards/C35.result.md` (template in `AGENTS.md`)

## Notes
- Money and ratio math in `Decimal` only.
- If `index_valuations` needs a unique key or column, add a **new** migration `0027_…`, revising `0026_audit_log_hardening`. Never edit an older migration.
- Don't change research model prompts. Agents research; code decides.
