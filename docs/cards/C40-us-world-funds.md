# C40: US and world funds on Ideas

**Requested by the product owner (2026-11-01):** "in ideas only indian stocks are shown... the USA stock ETFs are not present". Earlier (C35): "will the app suggest when to buy USA index ETF like NASDAQ or S&P 500?"
**Builds on:**
- C35 (S&P 500 valuation in `index_valuations`, `rules/buying.yaml`)
- the Denmark picks (`rules/eu_etfs.yaml`, `wealth.policy.personalise`, the "Buy from Denmark" card)
- C36/C37 (Ideas layout)

Read `AGENTS.md` first, especially the UX rules.

## Why
- S&P 500, Nasdaq 100 and world index funds appear only inside "Buy from Denmark", and only when the plan has new money to place. The owner hasn't set a monthly amount, and their biggest gap is safe money, so they never see a US fund or any timing advice for one.
- C35 stores a US market valuation (S&P 500 P/E and its percentile against about 3 years of history). The plan uses it only to split larger buys into 3 or 6 parts. Nothing shows it to the person.

## Scope

### 1. API: `index_funds` on the research ideas response
Add `index_funds` to the response of `GET /v1/research/ideas` (or the endpoint the Ideas page already reads for the Denmark card, whichever fits better). It is code-built from rules and stored data only, with no LLM. It is filled for households with a Denmark-resident member, and `[]` otherwise. Each entry:
- `exposure`: `global_all_world` | `global_sp500` | `global_nasdaq100`, in that order. Exposures are from `rules/eu_etfs.yaml` `selection.global_exposures`; take only these three.
- `role`: `plan_main` for `selection.global_core_exposure` (the fund the plan buys for money abroad), `option` for the others.
- `instrument`: the instrument `personalise` would pick for that exposure, with the same order and checks: share savings account first when `ask_first`, then the existing broker and ABIS rules. Return `id`, `name`, `currency`, `brokers` and `ask_eligible`. Reuse the existing selection code; don't write a second one.
- `market`: `US` for S&P 500 and Nasdaq 100. `null` for the world fund (no world valuation is stored).

Add `us_market` once on the response:
- `percentile` and `history_label`: from `wealth.marketdata.store.latest_index_valuation_percentile(session, "S&P 500", today)`. When `history_label` isn't the long-history label, set `percentile` to `null`.
- `label`: `cheap` (percentile ≤ `cheap_pe_percentile_max`), `expensive` (≥ `expensive_pe_percentile_min`) or `fair`, with both thresholds read from `rules/research.yaml`.
- `extra_money_parts`: `lumpsum.stp_tranches_expensive` when the percentile is at least `lumpsum.expensive_valuation_percentile`, otherwise `lumpsum.stp_tranches_normal`, from `rules/buying.yaml`. This is the plan's own rule for larger buys.
- `null` when nothing is stored.

**Placement:** put the pure mapping (percentile to label and parts) in `wealth.policy` (for example `wealth.policy.market_timing`) so the API can use it. The API must never import `wealth.research`. Decimal throughout.

### 2. PWA: a "US and world funds" card on Ideas
- **Where:** for a Denmark-resident household, a card directly after "This week's research ideas", open by default (the owner asked to see it). Title "US and world funds".
- **One line about the US market** (from `us_market`):
  - "US shares look cheap compared with the last 3 years." Use "fairly priced" and "expensive" for the other labels.
  - Then, from `extra_money_parts`: "Extra money: buy in 3 parts." or "Extra money: buy in 6 parts."
  - Then "Monthly buying: keep going." Regular monthly buying is never paused by the plan.
  - With `us_market` null: "No US market reading yet."
- **Three plain rows:**
  - "World fund": the instrument name, with a quiet "Your plan uses this one" badge on the `plan_main` row
  - "S&P 500 fund"
  - "Nasdaq 100 fund"
  - Second line on each row: "Share savings account" when `ask_eligible`, and the broker names as the Denmark card shows them
- **"Why these funds?"** (closed): two short sentences.
  - The world fund spreads money over thousands of companies in many countries, and about two-thirds of it is in the US.
  - The S&P 500 and Nasdaq 100 funds put everything in large US companies, and the Nasdaq 100 mostly in technology.
  - No advice to switch.
- **When no monthly amount is set:** add the quiet link "Set your monthly amount" (to Home), as C37 did under the Denmark line. Don't repeat it if the Denmark line already shows it: show it once, on this card.
- **Tidy-up while here:** the page repeats its title ("Ideas" in the header and again as a page heading). Drop the in-page heading and keep the one-line intro.
- **Copy:** all strings go through `copy/en.yaml` and `gen:copy`. No banned terms (UCITS, ISIN, allocation, etc.). Sentences of 20 words or fewer.

## Tests
- **Python:**
  - the percentile mapping at the thresholds (33/34, 66/67, 79/80)
  - the short-history case gives `percentile: null`
  - `us_market` is null with no stored valuation
  - `index_funds` order and roles from the rules
  - the picked instrument matches `personalise`'s choice for the same household
  - `[]` for a household with no Denmark resident
  - a DB test that the API role can read the stored S&P 500 valuation (mark `@pytest.mark.db`)
- **PWA:**
  - the card renders open with three rows and the plan badge on the world fund
  - each `us_market` label and both part counts
  - the null reading line
  - no card for a household with no Denmark resident
  - "Set your monthly amount" shows once
  - no in-page "Ideas" heading
- Keep the existing Ideas tests passing.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), ruff check and format (`--no-cache`), `python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`. Ideas is a route chunk; keep the initial bundle under 170 KB.
- Write `docs/cards/C40.result.md` with before/after notes.

## Notes
- No new dependencies and no migration.
- Don't change which fund the plan buys, the policy, or the research pack. This card only shows existing choices and the existing rule.
