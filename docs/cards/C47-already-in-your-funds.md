# C47: "Already in your funds": count what you own through funds before suggesting a share

**Why:** the plan suggests single shares (India stock steps, and US stock picks for Denmark members) without looking inside the household's funds. A Nifty 50 index fund already holds a lot of HDFC Bank, and an S&P 500 or MSCI World fund already holds a lot of Apple. The owner's rule:

> If a stock is already part of an ETF or fund we own, suggest buying it only if it makes sense to buy more.

"Makes sense" = the per-stock limits the plan already has, now counted on **direct holding + what the household owns of that company through its funds**. Also fix a gap found in review: `approved_us_stock_buys` ignores even the direct holding of the same share.

Read `AGENTS.md` first, especially the money-math, data and UX rules. Read `docs/UPGRADE_PLAN.md` §4.8 (buying rules) and §3.2 (sleeves). Builds on C10/C16 (market data sources, `universes`), C29a (Denmark picks, `rules/eu_etfs.yaml`), C29b (AMFI `sub_category`), C44 (health check).

## Decisions (follow them)
- **Approach: index baskets, not fund factsheets.**
  - Each fund the household holds maps to a **basket**: a list of companies with weights. Usually that's the index the fund tracks.
  - Active Indian funds use their category's index as a stand-in (a proxy).
  - Real per-fund portfolios (AMC monthly disclosures) are out of scope. Proxies are conservative enough here: a wrong guess only makes a share suggestion smaller, never bigger, except for funds that hold *less* of a company than their index.
- **Sources.** The reviewer checked each one on 2026-11-04 with a plain GET (no cookies, no special headers).
  - Use only these. **Don't add sources that need cookies, sessions or bot-check workarounds** (iShares was rejected for that).
  - State Street (SSGA) EMEA daily holdings, xlsx: `https://www.ssga.com/library-content/products/fund-data/etfs/emea/holdings-daily-emea-en-{ticker}-gy.xlsx`
    - tickers: `sppw` (MSCI World), `spyy` (MSCI ACWI), `spyi` (MSCI ACWI IMI), `spym` (MSCI Emerging Markets)
    - layout, sheet `holdings`:
      1. four label rows: `Fund Name:`, `ISIN:`, `Ticker Symbol:`, `Holdings As Of:` (e.g. `03-Nov-2026`)
      2. a blank row
      3. the header row: `ISIN, SEDOL, Security Name, Currency, Number of Shares, Percent of Fund, Trade Country Name, Local Price, Sector Classification, Industry Classification`
    - the last rows are cash and FX forwards, with ISIN `Unassigned` and zero or negative weights
    - skip every row without a check-digit-valid ISIN or with weight ≤ 0
    - the same ISIN can appear twice; sum its rows
  - S&P 500: the SSGA SPY holdings that `universes.py` already downloads and parses. It gives ticker and weight, no ISIN. Reuse the parser; don't download twice in one run if you can share it cleanly.
  - NASDAQ 100: Invesco JSON `https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/QQQ/holdings/fund?idType=ticker&productType=ETF`
    - `effectiveDate` is the as-of date
    - each item in `holdings[]` has `ticker`, `issuerName`, `percentageOfTotalNetAssets`, `cusip` and `securityTypeCode` (`COM` for shares)
    - keep shares only
  - NIFTY indices: NSE archive member lists `https://archives.nseindia.com/content/indices/ind_{list}list.csv`, columns `Company Name, Industry, Symbol, Series, ISIN Code`
    - lists: `nifty50`, `niftynext50`, `nifty100`, `niftymidcap150`, `niftysmallcap250`, `nifty500`, `niftylargemidcap250`, `niftybank`, `niftyit`, `niftypharma`, `niftyfmcg`, `niftyauto`. `nifty500` is already read by `universes`; reuse the fetch helper.
    - NSE publishes no full weights openly, so **weights are estimated**: each member gets its MSCI EM (`spym`) weight; members missing from `spym` get their `spyi` weight × k.
      - k = Σ spym weight / Σ spyi weight over Indian ISINs (`IN…`) found in both files.
      - Then renormalise to 100 % over the members found. Members found in neither count as 0.
      - Store `weights_estimated = true`, `members_total`, `members_found` and `weight_source = "msci_em+imi"`.
    - The reviewer's check on 2026-11-04 found every NIFTY 50 member, 94 of NIFTY 100, 94 of Midcap 150 and 335 of NIFTY 500. The missing ones are small companies.
  - **Derived baskets** (no extra download):
    - `msci_india` = `spym` rows with an `IN` ISIN, renormalised
    - `msci_em_ex_india` = the other `spym` rows, renormalised
- **Storage (new migration 0028, revising 0027):** public reference data, written by the market-data job, read by the API.
  - **`fund_baskets`**: `id` (text PK, e.g. `nifty50`, `sp500`, `msci_world`), `label`, `as_of`, `source`, `source_url`, `weights_estimated`, `weight_source`, `members_total`, `members_found`, `fetched_at`
  - **`fund_basket_weights`**: `basket_id` FK, `market` (`IN`/`US`/null), `isin` (nullable), `symbol` (nullable), `name`, `weight_pct NUMERIC(9,6)`
    - a check: `isin` or `symbol` is present
    - an index on `isin`, and one on `(market, symbol)`
  - Keep **only the latest snapshot per basket**: replace a basket's rows in one transaction. A failed download leaves the old snapshot.
  - Grants: follow the existing reference tables (e.g. migration 0017): `SELECT` for `wealth_app`, no RLS, no user data.
- **Rules file `rules/fund_lookthrough.yaml`** (versioned, commented). It holds:
  - `baskets`: id, label and how to build each (source, ticker or list, derived-from)
  - `max_age_days: 21`. An older snapshot is not used.
  - `by_index_name`: maps every index name used in `rules/etf_categories.yaml` (`index_name`) and `rules/eu_etfs.yaml` (`index`) to a basket
  - `by_us_symbol`: US-listed ETFs:
    - `VOO`, `IVV`, `SPY`, `SPLG` → `sp500`
    - `QQQ`, `QQQM` → `nasdaq100`
    - `URTH` → `msci_world`
    - `ACWI` → `msci_acwi`
    - `VT` → `msci_acwi_imi` (`proxy: true`)
    - `INDA` → `msci_india`
    - `EEM`, `IEMG`, `VWO` → `msci_em` (proxy where the index differs)
  - `by_sub_category`: for Indian funds, keyed by `securities.sub_category`:
    - Large Cap → `nifty100`
    - Large & Mid Cap → `niftylargemidcap250`
    - Mid Cap → `niftymidcap150`
    - Small Cap → `niftysmallcap250`
    - Flexi Cap / Multi Cap / ELSS / Other Equity → `nifty500`
    - Aggressive Hybrid → `nifty500` with `equity_share: "0.75"`
    - Balanced Hybrid → `nifty500` with `equity_share: "0.5"`
    - Conservative Hybrid → `nifty500` with `equity_share: "0.2"`
    - Sectoral/Thematic → by `name_keywords` (bank → `niftybank`; IT/technology → `niftyit`; pharma/healthcare → `niftypharma`; FMCG/consumption → `niftyfmcg`; auto → `niftyauto`), else `nifty500`
    - Index Funds/ETFs and International / FoF Overseas → by `name_keywords` ("nifty 50", "nifty next 50", "s&p 500", "nasdaq 100", "msci world", …)
    - Every proxy is marked `proxy: true`.
  - `by_name_keywords`: a fallback by name for funds and ETFs with no sub-category, such as Danish broker imports. Use the same keyword style as `etf_categories.yaml`, including its `exclude_any` handling (e.g. "nifty 50" must not match "nifty next 50").
  - `not_looked_through`: debt, gold, arbitrage, liquid. No basket.
  - `same_company`: share classes that are one company (`GOOGL`+`GOOG`, `FOXA`+`FOX`, `NWSA`+`NWS`); their weights add up.
  - A guard test: every `index_name` in `etf_categories.yaml` and every `index` in `eu_etfs.yaml` maps to a basket, or is listed under `not_looked_through` with a reason.

## Scope

### 1. Market data source `fund_baskets` (`wealth/marketdata/sources/fund_baskets.py` + `refresh.py`)
- Pure parsers for the SSGA xlsx, the Invesco JSON and the NSE CSV, plus the weight estimation and derived baskets.
  - Use `openpyxl` read-only with `data_only=True`, as the importer does. No pandas.
- `python -m wealth.marketdata.refresh --only fund_baskets` stores every basket it can build.
  - Counts per basket go in the run JSON: `rows`, `members_total`, `members_found`, `as_of`.
  - One failed download marks the run `partial` and keeps going. A NIFTY basket whose MSCI source failed is skipped, not stored with zero weights.
  - All money/weight math is `Decimal`.
- Add `fund_baskets` to the source loop in `.github/workflows/marketdata-weekly.yml`, **before** the research run (Friday evening).
- `prune` must not touch the new tables.

### 2. Pure look-through (`wealth/policy/lookthrough.py`)
- `basket_for_holding(holding, rules) -> BasketChoice | None`: basket id, `equity_share` (default 1) and `proxy` flag. Order:
  1. `by_us_symbol` for US ETFs
  2. the curated ETF's index (`eu_etfs.yaml`, matched by ISIN)
  3. `etf_categories.yaml` matching for Indian index funds/ETFs
  4. `by_sub_category` (with its name keywords)
  5. `by_name_keywords`
  6. otherwise None
- `through_fund_amounts(fund_holdings, weights) -> dict[CompanyKey, Decimal]`, where `CompanyKey` is an ISIN, or `(market, symbol)` when there's no ISIN.
  - Amount per fund = fund value in INR × `equity_share` × weight / 100.
  - Summed over funds. Weights add up for `same_company` classes. No rounding inside (round at presentation).
  - Also return per-fund parts for the explanation: fund name and amount, largest first.
- **A security matches a basket row** by ISIN first, else by `(market, symbol)`.
- Only directly held shares, ETFs and funds feed this, valued exactly as `_holding_value` values them today; recorded buys (C45) too. Holdings without a value are skipped and noted (`lookthrough_fund_unvalued`).

### 3. Loader (`wealth/policy/data.py`)
- Load `fund_baskets` (fresh ones only, ≤ `max_age_days`) and **only the weight rows needed**: those whose ISIN or symbol matches a research-pack idea or a directly held share.
  - Don't load whole baskets per request.
- Missing or old baskets: the look-through is empty for those funds, with a plan note `fund_lookthrough_missing:<basket>`. The plan otherwise behaves as today.
- Add each used basket's `as_of` to the plan inputs digest, so the plan rebuilds once when new weights land.

### 4. Plan: share sizing
- **India stock step (`actions.py`):**
  - `WeekInput` gains `through_fund_inr: dict[str, Rupees]` (by `security_id`, default empty).
  - `stock_room = max_stock − direct value − through_fund_inr[security_id]`. The sector cap and the satellite gap stay on direct holdings only.
  - When fund holdings leave no room, the idea is skipped as it is today, and the plan gets a note `stock_owned_through_funds:<security_id>`.
  - **Worked example (make it a test):** the household holds ₹20,00,000 in a Nifty 50 index fund, where HDFC Bank's weight is 12.5 % (₹2,50,000). It also holds ₹4,00,000 in an aggressive hybrid fund (NIFTY 500 proxy, `equity_share` 0.75), where HDFC Bank's weight is 7 % (₹21,000). It holds ₹50,000 of HDFC Bank directly.
    - Equity target ₹60,00,000 → `max_stock` ₹3,00,000 → room −₹21,000 → **no HDFC Bank step**. The next eligible idea or the core index step gets the money, exactly as when an idea is ineligible today.
    - Equity target ₹80,00,000 → `max_stock` ₹4,00,000 → room **₹79,000** → the step is the floor of `min(tranche, gap, ₹79,000) ÷ price` shares.
- **Denmark US picks (`denmark_picks.approved_us_stock_buys`):**
  - New keyword argument `held_dkk_by_security: Mapping[str, Decimal]`: the member's direct value of that share plus the member's through-fund amount, in DKK.
  - `amount = min(remaining, per_stock_cap − held)`. Below the minimum buy → skipped, as now.
  - The caller passes the same holdings scope it already uses for `investable_wealth_dkk` (that member's holdings).
- **One shared room function** feeds both the plan and the Ideas page (§5), so the two can never disagree.

### 5. Ideas API + PWA
- `ResearchIdeaSummary` gains `owned: IdeaOwnership | None`:
  - `direct_inr` and `through_funds_inr` (money strings)
  - `status`: `none` | `some` | `enough`
  - `funds`: up to 3 entries `{name, amount_inr}`, largest first
  - For the household; household view rules as today.
  - **`enough` means the shared room function gives no member room of at least the minimum buy** for that market:
    - India: the household India stock step
    - US: each opted-in Denmark member's US-pick room
  - **`some`** = owned > 0 but not enough. **`none`** otherwise.
- **PWA Ideas (`apps/web/src/ideas/IdeasPage.tsx`):**
  - `some`: one quiet line under the idea.
    - Funds only: "You already own about ₹2.7 L of this through your funds."
    - Direct + funds: "You already own about ₹3.2 L of this, ₹2.7 L through your funds."
    - Use `formatInr(…, "compact")`.
    - The idea's existing details section lists the funds: "Nifty 50 index fund: about ₹2.5 L". Mark proxy amounts with "estimate".
  - `enough`: the idea moves to a section at the end, **closed by default**: "Already in your funds (N)". It holds one line, "You own enough of these through your funds, so we won't suggest more.", and the ideas below it.
  - `none` or `owned` missing: no change.
  - No jargon: never "look-through", "exposure", "basket", "weight", "constituent", "proxy". Copy in `en.json`; ≤ 20 words a sentence; run the banned-terms check.
  - Mobile: lines wrap; no sideways overflow on a 384 px screen (`expectFitsScreen`).

### 6. Health check (`wealth/alerts/health.py`)
New check `fund_lookthrough`:
- **fail** when no basket is stored
- **warn** when a basket mapped by the rules is missing or older than `max_age_days`
- **warn** when NIFTY 50 or NIFTY 100 has `members_found / members_total` < 90 %

Plain labels, as in C44.

## Tests
- **Parsers** (synthetic fixtures built at test time; no network; no real fund files committed):
  - SSGA layout: label rows, header, cash/`Unassigned` rows skipped, duplicate ISIN summed
  - Invesco JSON with a non-`COM` row skipped
  - NSE CSV
- **NIFTY estimation:** a hand-worked example with 4 members (2 in `spym`, 1 only in `spyi`, 1 in neither): check k, the renormalised weights to 6 dp, `members_found`, and that a failed MSCI source skips the basket.
- **Derived baskets:** India and EM ex India.
- **Rules:** the guard test above; each `by_sub_category` entry; Nifty 50 vs Next 50 keywords; a Danish-imported "iShares Core S&P 500 UCITS" with no sub-category maps by name; debt/gold are not looked through; `GOOGL`+`GOOG` add up.
- **Look-through math:** the worked example, per member and household; hybrid `equity_share`; an unvalued fund is noted and skipped.
- **India step:** both halves of the worked example. Fund holdings never change the sector cap or the satellite gap.
- **US picks:**
  - direct holding alone reduces room (the old gap)
  - direct + funds ≥ cap → skipped
  - partial room sized to the room
- **Loader:** old basket → not used + note; only needed rows are queried (assert the query filters by ISIN/symbol); digest changes when `as_of` changes.
- **API (db):** `owned` statuses `none`/`some`/`enough` for an India and a US idea; household isolation (another household's funds never count).
- **PWA:** the two "some" lines, the closed "Already in your funds" section, no overflow, axe scan.
- **e2e:** extend the Ideas journey seed with one fund holding and a small basket, so one idea shows the "through your funds" line and one sits in the closed section.
- **Health:** fail / warn / ok cases for `fund_lookthrough`.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache`, `.venv\Scripts\python -m scripts.check_deploy_config`
- Migration upgrade and downgrade on a fresh test database; existing migration guard tests pass.
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- The e2e suite runs as `docs/cards/C24-e2e.md` describes (the reviewer also runs it).
- Write `docs/cards/C47.result.md`. List every basket, its source and whether its weights are estimated or a proxy, plus every place that now reads through-fund amounts.

## Notes
- No new dependencies (`openpyxl` is already in the `importer` extra; check the weekly workflow installs it). No LLM. The API must not import `wealth.research`.
- Don't change which funds the plan picks, the satellite percentage, or any cap value.
- Out of scope (later cards): real AMC monthly portfolios for active funds; per-stock sell or "add more" advice (C49); new broker formats (C48).
