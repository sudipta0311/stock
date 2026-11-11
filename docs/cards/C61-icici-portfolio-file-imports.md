# C61: an ICICI Direct portfolio file imports

**Source:** the owner, 2026-11-11: "This portfolio upload is failing". The file is ICICI Direct's portfolio download (`DEFAULT_Portfolio_<date>.csv`). The reviewer reproduced it locally with the owner's file. The file is not in the repo and must not be added; use the synthetic fixture below.

## What happens today
- The file reads fine: `read_table` gives 23 headers and every row, and `auto_map` maps it as holdings.
- But the rows carry no ISIN and no NSE symbol. "Stock Name" is ICICI's own code (for example `ABCDEFEQ`), not an NSE symbol. `_security_match` in `src/wealth/api/broker_imports.py` matches only by ISIN or symbol, and `can_create_security` refuses Indian rows without an ISIN.
- So every row is `unknown` (`symbol_not_found`). The review shows "Could not match this share" for all rows, and confirming says "Nothing changed".
- The broker picker defaults by country of residence: Denmark gives `nordnet` (DK, DKK). `find_layout` drops built-in layouts unless the picker names the same broker. So an Indian file uploaded with the default picker is auto-mapped as **Danish shares in DKK**.
- The reviewer checked the owner's company names against NSE's public lists (the same ones the daily job stores):
  - **Plain name rule:** most match exactly one listed security once case, `Ltd`/`Limited` and `&`/`and` are normalised.
  - **Apostrophe rule:** one more matches once apostrophes are dropped (a listed name with `'s` against a broker name without it).
  - **No safe rule:** one ETF has no safe name match, because NSE lists it under a compressed name with no spaces.

## Read
- `AGENTS.md` (UX rules: one primary action, plain copy, no jargon).
- `src/wealth/api/imports.py`:
  - the broker branch of the upload handler (about lines 518–760);
  - `patch_import_item`;
  - `ImportItemPatch`.
- `src/wealth/api/broker_imports.py`:
  - `find_layout`, `_security_match`, `stage_broker_items`, `_security_for_confirmation`, `can_create_security`;
  - the tradebook part of staging.
- `src/wealth/importer/brokers/parser.py`:
  - `detect_layout`, `load_layouts`, `auto_map`;
  - `read_table`, `parse_holdings`.
- `rules/brokers.yaml`.
- `apps/web/src/imports/ImportPage.tsx` (the unknown-row section and `acceptUnknown`) and `apps/web/src/imports/brokers.ts`.
- `e2e/tests/save-imports.spec.ts` and `e2e/COVERAGE.md`.
- The tests next to each of these.

## Scope
1. **Recognise ICICI Direct's portfolio download.** Add a verified `icici_direct` holdings layout to `rules/brokers.yaml`, next to the existing ICICI one.
   - **Fingerprint:** `[stock name, company name, cmp, average cost value, qty]`.
   - **Columns:**

     | Field | Column |
     |---|---|
     | `name` | Company Name |
     | `quantity` | Qty |
     | `avg_price` | Average Cost Value |
     | `cost_value` | Invested Value |
     | `last_price` | CMP |
     | `market_value` | Current Value |

   - **Qty is the total holding.** Never use Long Term Qty or Short Term Qty: they do not always add up to Qty.
   - **Stock Name is never `symbol`.**
   - **Quirks to keep working:**
     - rows can carry one more trailing value than there are headers;
     - `-` means empty;
     - two headers repeat (`% Chg.`, `% Change`).
   - Add a short `source_note`.
2. **A verified layout wins over the picker.**
   - **When the headers match a verified built-in layout,** use that layout's broker, market and currency, even when the upload picked another broker.
   - **When the picker named a different broker,** record `detected_broker` in the parse report. The review shows one short line, for example "We read this as an ICICI Direct file."
   - **Unchanged:**
     - saved templates;
     - unverified layouts;
     - auto-mapped files without a verified layout (they keep the picker's defaults).
3. **Match Indian rows by company name.**
   - **When it runs:** when a broker row with market `IN` finds no security by ISIN or symbol.
   - **What it searches:** listed Indian securities only: `market = 'IN'`, `symbol IS NOT NULL`, `security_type` equity or etf.
   - **Load the candidates once per staging call,** not once per row.
   - **Normaliser:** a pure function in `wealth.importer` that:
     - casefolds;
     - turns `&` into `and`;
     - deletes apostrophes (`'` and `’`);
     - turns other non-alphanumerics into spaces;
     - drops the words `limited` and `ltd`;
     - collapses spaces.
   - **Unique hits only.**
     - First compare normalised names.
     - If there is no unique hit, compare again with every space removed (so `NIFTY50` equals `NIFTY 50`).
     - Two or more hits leave the row `unknown`.
   - **What a match sets:**
     - `match_security_id`;
     - the payload's `symbol` and `isin` from the security, so the review shows the NSE symbol;
     - `matched_by: "name"`.
   - **Status is unchanged.** The existing `new` / `enrich` / `changed` / `duplicate` / `several_accounts` logic then runs as it does today. Move that per-row status logic into one helper so step 4 reuses it.
   - **Tradebook rows:** apply the same fallback where staging matches them.
4. **Let the owner pick the share for a row that still does not match.**
   - **API: `GET /v1/securities/search?q=<text>&market=IN`.**
     - Signed-in users only.
     - `q` must be at least 2 characters after trimming; otherwise 422.
     - Search listed securities in that market, equities and ETFs: symbol starts with `q`, or name contains `q`, case-insensitive.
     - Return at most 8, exact symbol match first, then by name.
     - Each result gives `id`, `symbol`, `name`, `security_type`.
     - The API must not import `wealth.research`.
   - **API: `PATCH /v1/imports/{job}/items/{item}` accepts `security_id`.**
     - Allowed only for broker jobs, on items with status `unknown` and reason `symbol_not_found`.
     - The security must exist and have the row's market.
     - Anything else gives 422. A job that is not ready keeps today's 409. The household write rules stay as they are.
     - On success, run the step 3 status helper for that item, and set `match_security_id`, `symbol`, `isin` and `matched_by: "owner"`.
     - Return the updated item.
   - **Screen.**
     - Each unmatched row gets a quiet "Find this share" button. It opens an inline search: label "Share name or NSE symbol", with up to 8 results as buttons ("SYMBOL · Name").
     - Choosing one moves the row to its new group.
     - "Leave it out" stays.
     - The page's primary action does not change.
     - Debounce the search and cancel stale requests.
     - All new copy goes through the copy source and `npm run gen:copy`.
   - Update the generated API types with `npm run gen:api`.
5. **E2E test.** Add a test in `e2e/tests/` following `save-imports.spec.ts`, and its row in `e2e/COVERAGE.md`. It covers:
   - uploading the synthetic ICICI file with the picker left on a Danish broker;
   - most rows matching by name;
   - finding one row with "Find this share";
   - confirming.

   Seed the Indian securities it needs through the e2e setup's existing database path. The reviewer runs e2e; you cannot.

## Fixture
`tests/fixtures/brokers/icici_direct_portfolio.csv`: ICICI's exact header line (copy it from below), with 5 or 6 **made-up** companies and numbers. Include:
- one row with an extra trailing `-` value;
- `-` in the NAV columns;
- a name ending `LTD.`;
- a name with `&`;
- a name with no apostrophe whose listed name has one;
- an ETF whose name matches nothing (for the picker);
- one row where Long Term Qty plus Short Term Qty is less than Qty.

```
Sr. No.,Stock Name,Company Name,CMP,Change,% Change,NAV,NAV Chg,% Chg.,Portfolio Holdings,Invested Value,Average Cost Value,Unrealized Profit/Loss,% Chg,Current Value,Days Profit/Loss,% Chg.,Realized Profit/Loss,% Change,Qty,Long Term Qty,Short Term Qty,XIRR
```

## Tests
- **Parser.**
  - The fixture is detected as the new `icici_direct` layout.
  - Quantity comes from Qty and cost from Average Cost Value.
  - `symbol` is None; the trailing value is tolerated.
- **Normaliser.**
  - `Example Paints Ltd.` = `EXAMPLE PAINTS LIMITED`.
  - `Sample & Sons Ltd` = `Sample and Sons Limited`.
  - `Nandi Wall's (India) Limited` = `NANDI WALLS (INDIA) LIMITED`.
  - The space-free pass matches `Alpha Nifty50 ETF` to `Alpha Nifty 50 ETF`.
  - Two candidates with the same normalised name give no match.
- **DB (`@pytest.mark.db`).**
  - Upload the fixture with broker `nordnet`: the job reads as `icici_direct` with `IN`/`INR`, and `detected_broker` is recorded.
  - Seeded securities match by name and give `new`. A seeded CAS lot with the same quantity gives `enrich`, and after confirm the lot has its buying price.
  - Search returns the seeded ETF.
  - PATCH with its `security_id` moves the row out of `unknown`, and confirm stores it.
  - PATCH with a security from another market → 422; PATCH on a non-broker job → 422; a search with `q` shorter than 2 characters → 422.
- **PWA.** The unmatched row, then "Find this share", then a pick moves the row. Use a mocked network and check accessible names.
- **No network or LLM anywhere.** All data is synthetic.

## Out of scope
- A table mapping ICICI codes to NSE symbols, and downloading ICICI's security master.
- Other brokers' layouts.
- Fuzzy or partial name matching beyond the two passes above.
- Migrations: the search runs over about 3,000 listed rows, so no new index is needed.

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
