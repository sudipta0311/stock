# C49: Keep, add more or sell: a weekly check on every share you own

**Why:** the owner wants the app to say when to sell a share they own, and when to buy more of it. The exit engine from C13b (`wealth/policy/exits.py`, rules in `rules/exits.yaml`) exists and runs in every plan, but `_exit_cards` feeds it almost nothing. It gets only lots, price and weight. Revenue trend, valuation, governance events, the 200-day average and thesis status are never filled in, so almost every check is "unknown" and only the size check can fire.

Also:
- The Holding page shows a raw "Exit Card": hard-coded English, trigger names, "Threshold" text. That breaks the UX rules.
- Nothing ever says "add more".

Read `AGENTS.md` first (agents research, code decides money; UX rules). Read `docs/UPGRADE_PLAN.md` §4.9–4.10 (exits, tax-aware selling) and §4.8 (buying). Builds on C13b (exit engine), C15a/C15b (fundamentals, screener, sentinel), C47 (through-fund amounts and the shared stock-room function) and C48 (more shares imported).

## Decisions (follow them)
- **Everything here is deterministic code.** No LLM sets a verdict, an amount or a date. Thesis status comes from code-owned evidence only (see B). Dossier text may be *linked* but never decides.
- **Weekly stock checks are computed by the research job and stored.** The API and the plan only read them; the API must not import `wealth.research`.
- **Selling stays conservative:**
  - the size check uses the **direct** holding's share of equity (as today), not through-fund amounts
  - Home shows **at most one sell step a week**: the most urgent, following `_exit_action`'s rules
  - every other verdict appears only on Money and the Holding page
- **"Add more" uses the same limits as new ideas,** counted with funds (C47's shared room function), so Money, Ideas and the plan always agree.

## Behaviour

### A. Weekly stock checks (`wealth/research/stock_checks.py`, new table `stock_checks`, migration `0029`)
- **Table:** public reference data, no user data; `SELECT` for `wealth_app`, as other reference tables.
  - `security_id` and `week` (together the PK), plus `as_of`
  - `quant_decile` (nullable) and `factor_coverage`
  - `pe_percentile` (own-history P/E percentile, nullable) and `pe_vs_sector` (sector median earnings yield ÷ own earnings yield, nullable)
  - `revenue_yoy_pct` (JSONB list of Decimal strings, newest first, up to 4 quarters)
  - `price`, `dma200` (nullable), `below_dma200` (nullable bool)
  - `sources` (JSONB: which inputs and their dates)
  - Keep the last 12 weeks; delete older rows in the same job.
- **Which shares:** every security the screener ranks that week (`ScreeningResult.ranked`: NIFTY 500 and S&P 500), **plus** every directly held equity:
  - a lot with quantity > 0 in `holdings_lots`, or a waiting `recorded_buys` row
  - read with a system session, as `sentinel._targets` does
  - store security ids only; never user ids
- **How (all `Decimal`, point-in-time `known_at <= as_of`):**
  - `quant_decile`: from the screener's ranking, for ranked shares only.
  - `pe_percentile`, `pe_vs_sector`: reuse `calculate_valuation_stats` and the inputs the dossier already builds for it. Don't write a second P/E history.
  - `revenue_yoy_pct`: quarterly `revenue` from `fundamentals_pit`, each quarter against the same quarter a year earlier.
  - `dma200`: needs ≥ 200 stored sessions in `prices_eod`, else null.
- **Run:** `python -m wealth.research.stock_checks --week <ISO week>`. Add it as its own step in `research-weekly.yml` right after `run start`, with `if: always()`-style isolation so a failure only warns.
  - Run JSON counts: `checked`, `held_checked`, `held_not_covered`, and per-input coverage.
- **A held share outside both lists** gets a row with price and DMA only, where prices exist.

### B. Verdicts (`wealth/policy/holding_review.py`, pure)
- **Facts.** For each directly held share (equity, any market), fill `HoldingFacts` from:
  - the newest `stock_checks` row ≤ 14 days old (`rules/exits.yaml` `checks_max_age_days: 14`)
  - `sentinel_events` for that security in the last 90 days (governance events)
  - lots and the current price (drawdown from cost), as `_exit_cards` does today
  - **thesis status, code-owned:** run the revenue part of the thesis rule first; `triggered` → `broken`, `watch` → `watch`, `ok` → `on_track`, no revenue data → `unknown`. Then build the card. The risk check (`requires_thesis: WATCH`) can now work.
- **Verdict per holding**, from the exit card plus one new rule:
  - `sell`: the card says `EXIT`
  - `review` ("Think about selling"): `REVIEW_THEN_EXIT`
  - `sell_some`: `TRIM_TO_TARGET` with a triggered check
  - `add_more`: all of these hold (new `accumulate:` section in `rules/exits.yaml`, commented):
    - no check is `triggered` or `watch`
    - `quant_decile >= 8`
    - `pe_percentile < 80`
    - the check is fresh
    - the C47 room function gives room ≥ the minimum buy for that market
  - `keep`: everything else
  - `not_checked`: no fresh stock check at all
- **Reasons:** up to 3 reason codes, most important first, with the numbers the copy needs as params (e.g. `revenue_down_quarters: 2`, `share_of_equity_pct`, `long_term_on`).
- **Tax:**
  - Indian-tax holdings reuse the card's tax estimate and `tax_wait`: "Waiting until 12 Mar 2027 lowers the tax."
  - Holdings in a share savings account (`Account.tax_wrapper == "ask"`): note code `ask_same_tax`, "In your share savings account, the tax is the same whether you sell or keep." The account is taxed each year on its gain (`rules/residency.yaml` `denmark.ask`), so selling adds no tax. Never show a waiting date for these.
  - Other holdings whose tax the engine doesn't model (ordinary Danish accounts): show no amount, only the note code `tax_not_estimated`.
- **The plan JSON** gains `holding_reviews`: `security_id`, `member_id`, `verdict`, `reasons`, `tax`, `checked_on`, `add_more_room_inr`. Keep `exit_cards` for now (the Holding page stops reading it; see D). Add the stock checks' week to the plan inputs digest.

### C. Plan: buying more of what you own
- **India stock step:** held Indian shares with verdict `add_more` join this week's candidates next to approved new ideas.
  - **Finishing a started share comes first.** Holdings whose direct value is below `max_stock` rank before new ideas, ordered by `quant_decile`, then lowest `pe_percentile`.
  - The step's tranche number is `floor(direct value ÷ tranche amount) + 1`, capped at `tranches`.
  - Sizing, sector cap and room are exactly as for new ideas (C47 room, sector cap on direct holdings).
- **Denmark US picks:** a member's held US shares with `add_more` are candidates in `approved_us_stock_buys` before new ideas, sized by C47's room.
- **Sell steps:** unchanged rules (`_exit_action`), but at most one on Home a week (the most urgent; ties → the larger amount). The rest stay visible on Money and the Holding page.
- **Worked examples (tests):**
  1. A held India share, decile 9, P/E percentile 60, no triggers, direct ₹1,00,000, `max_stock` ₹3,00,000, through funds ₹50,000 → `add_more`. The India stock step picks it before a new decile-10 idea and calls it part 2 of 3 (tranche ₹1,00,000).
  2. Revenue down 2 quarters in a row → `sell`, with reason `revenue_down_quarters: 2`.
  3. P/E percentile 95 and 1.6× sector → `sell_some`. Long-term status is 20 days away, so the tax note says to wait (existing `tax_wait`).
  4. Down 30 % from cost, below the 200-day average, one revenue decline (thesis watch) → `review`.
  5. No stock check → `not_checked`, never a sell step.

### D. PWA
- **Money list:** directly held shares show a small label when the verdict isn't `keep`/`not_checked`:
  - "Add more" (calm green)
  - "Sell some" / "Think about selling" (amber)
  - "Sell" (strong, not alarming)
  - No label for keep.
- **Holding page:** replace the "Exit Card" with a card **"Our view"**:
  - the verdict as the title: "Keep", "Add more", "Sell some", "Think about selling" or "Sell"
  - up to 3 reasons in plain words
  - one tax line when known
  - "Checked on 7 Nov 2026"
  - "How we check" (closed by default) lists each check in plain words with ✓ / ! / –. **No thresholds, no codes, no trigger names.**
  - `not_checked`: "We don't follow this company's results yet, so we only check its price and size."
- **Home:**
  - the one sell step, as today
  - when other holdings have `sell`/`review` verdicts and no sell step is shown, one highlight: "1 share to look at" (plural form), linking to Money
  - stay within the 2–3 notes rule (use the backend `highlights` ranking)
- **Copy** in `en.json`; the banned-terms check passes. Words to avoid: thesis, drift, trim, tranche, DMA, percentile, decile, ROCE, valuation band. Sentences ≤ 20 words.
- Mobile 384 px: no overflow; axe scan.

### E. Health (`wealth/alerts/health.py`)
New check `stock_checks`:
- **warn** when the newest week is older than 9 days
- **warn** when fewer than 90 % of held IN/US shares that are in NIFTY 500 / S&P 500 have a fresh row

## Tests
- **Stock checks:**
  - revenue YoY from synthetic quarters (including a missing quarter)
  - point-in-time `known_at`
  - DMA needs 200 sessions
  - held shares outside the lists get price-only rows
  - no user ids stored
  - old weeks pruned
- **Facts and verdicts:** the five worked examples; thesis derived in code; governance event → `sell` (CRITICAL); a stale check → `not_checked`; a US share in the share savings account → tax note `ask_same_tax` and no waiting date; a European share → `not_checked`.
- **Plan:**
  - the add-more candidate ranks before a new idea
  - tranche number
  - C47 room respected (funds can turn `add_more` into `keep`)
  - Denmark US add-more
  - only one sell step on Home a week
  - the digest changes when a new stock-check week lands
- **API (db):** `holding_reviews` in the plan; household isolation; a viewer sees verdicts but no Done/sell actions (as today).
- **PWA:** the Money labels, the "Our view" card (every verdict), "How we check" closed by default, the Home highlight, no overflow, axe.
- **e2e:** seed a held share with a fresh stock check that gives `add_more`, and one that gives `sell`. Check the Money labels, the Holding page, and the single sell step on Home.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache`, `.venv\Scripts\python -m scripts.check_deploy_config`
- Migration upgrade and downgrade on a fresh test database; guard tests pass.
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- The e2e suite runs as `docs/cards/C24-e2e.md` describes (the reviewer also runs it).
- Write `docs/cards/C49.result.md` with a table of every verdict, the facts that drive it and the copy shown.

## Notes
- No new dependencies. No LLM.
- Don't change existing exit thresholds; only add `accumulate:` and `checks_max_age_days`.
- **Markets:** the owner buys Indian shares from India, and from Denmark mainly ETFs plus some global shares in a share savings account (2026-11-04). Full checks cover `IN` and `US` shares in any account, including the share savings account. Shares from other markets (e.g. a European share) get `not_checked`, with no sell step and no "add more", because the app has no company results for them. ETFs and funds get no verdict; C47 already counts what they hold.
- **No buying price:** since C48, a holding can be imported without a buying price (`cost_per_unit` is null). Its drawdown from cost is unknown, so checks built on it stay not triggered, and the tax shows only `tax_not_estimated`. The other checks still run. Test it.
- **Migration number:** C48 added no migration, so this card's migration is `0029`, revising `0028`.
- **US 200-day average:** the daily overseas job fetches `outputsize=compact` (about 100 sessions), so `dma200` for US shares fills in only as stored prices build up. Until then the risk check stays unknown. That's expected; don't add a new price source.
- Out of scope: Danish share tax estimates, DK/EU fundamentals, recording sells (a later card), weekly email copy for verdicts.
