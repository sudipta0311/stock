# C41: Ideas show company names and one "Why"

**From the reviewer's phone check (2026-11-01)** after the US ideas came back (C39). Builds on C35 (buy hints), C37 ("Why this idea?") and C40 (the US and world funds card). Read `AGENTS.md` first, especially the UX rules.

## Findings
- **Only the ticker shows.** "SNDK", "ALL" and "GEV" mean nothing to most readers. Production stores names, but they're mixed:
  - India names are clean ("Lupin Ltd.").
  - Many US names come from the SPY holdings file in capitals ("ALLSTATE CORP", "AMERIPRISE FINANCIAL INC", "FAIR ISAAC CORP").
  - Others are fine ("GE Vernova Inc.").
- **Two "why" sections per idea.** The buy hint has its own closed "Why?" ("There are too few price observations for this check. The price and market checks support the usual pace."), and a second closed "Why this idea?" follows two lines later. Two disclosures that sound alike are confusing.
- **The US timing reason reads like a fault.** "There are too few price observations for this check" shows for every US idea. US prices start on 2026-10-29 (SSGA implied closes), so the 200-day trend check can't run until about August 2027 (200 trading sessions later).

## Scope

### 1. Proper US company names
- The public S&P 500 constituents table, already read for GICS sectors (`universes.parse_sp500_sectors`), has a `Security` column with proper names ("Allstate", "Ameriprise Financial", "Fair Isaac").
  - Parse it next to the sector. Extend `Sp500Sector` with `name`, or a sibling mapping: whichever reads better.
  - The universes refresh writes it as the security's `name` when present.
  - The SPY holdings name stays the fallback when the table has no row.
- Don't touch India names.

### 2. API: `name` on each idea
- `ResearchIdeaSummary.name: str | None`, read from `securities` by `security_id` in one query for all ideas, not one per idea. Older packs work unchanged.

### 3. PWA: the idea row
- **Title:** the company name, truncated to one line, with the ticker as quiet text on the right of the same line. Without a name, show the ticker as today.
- **One closed "Why this idea?"** holds everything:
  1. up to two reasons
  2. "Main risk: …"
  3. a short "About the timing" line built from the hint's reasons
- Remove the hint's separate "Why?". The visible hint line ("Buy in 3 parts over 3 months") stays where it is.
- **Copy for the US trend reason** (`entry_hint_us_price_history_unavailable`): "We have under a year of US prices, so we skip the trend check." Keep it ≤ 20 words. Check the other reason strings read as plain facts, not faults.
- Keep C37's order: hint line, then sector and valuation words, then "Why this idea?".

### 4. US and world funds: findable at the broker (C40 card)
- The row names come from `plain_fund_name`, which turns "UCITS ETF" into "fund" ("iShares Core S&P 500 fund USD (Acc)"). That name can't be searched at Nordnet or Saxo, and "(Acc)" means nothing to most readers.
- **Visible name:** also drop the trailing share-class markers "(Acc)" and "USD". The result is "iShares Core S&P 500 fund".
- **Each fund row gets a closed "Find it at your broker"** with:
  - the exact listed name from `rules/eu_etfs.yaml`
  - the search code, labelled "Search code"
  - one line: "It reinvests dividends for you." when the class accumulates
- This follows C37's pattern of codes only inside a closed section. Add `isin` and `listed_name` to `ResearchIndexFundInstrument`.
- The Denmark buy card has the same naming problem. If it shares the helper, apply the same change there, closed by default too.

## Tests
- **Python:**
  - the constituents parser reads names
  - the refresh stores a table name over a SPY capitals name, and keeps the SPY name when the table lacks the ticker
  - `name` is on `ResearchIdeaSummary` and filled for every idea in one query (a DB test)
- **PWA:**
  - the idea shows the name with the ticker
  - without a name it falls back to the ticker
  - there is one disclosure per idea, with the reasons, risk and timing inside it and closed by default
  - no separate hint "Why?"

## Acceptance
- `.venv\Scripts\python -m pytest` (full), ruff check and format (`--no-cache`), `python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- Write `docs/cards/C41.result.md`.

## Notes
- No migration and no new dependencies. Don't change the hint logic, the pack, the policy or the research.
