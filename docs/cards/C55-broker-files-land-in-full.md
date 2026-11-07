# C55: Broker files land in full

**Source:** the reviewer's upload walkthrough on 2026-11-06. It used synthetic files in each broker's real layout, at 360 × 740, on the local stack. The owner asked for the upload flow to be checked and improved.

## Why
The owner's family holds shares through Zerodha and HDFC Securities in India, INDmoney in the US, and Saxo and Nordnet in Denmark. The walkthrough found:

| File | What happened |
|---|---|
| Saxo positions (`Symbol` like `NOVOb:xcse`, `IWDA:xams`, with a valid ISIN) | Every row was "Could not match this share". Nothing could be imported, and "Confirm import" stayed disabled. |
| Saxo `Open Price` (Saxo's average buying price) | Not read: "This file has no buying prices". |
| Zerodha holdings (`Previous Closing Price`) | Not read, so every row was "Value not provided" and Money showed "Not available" for each holding. |
| Zerodha `NIFTYBEES` (ISIN `INF…`, Sector `ETF`) | Filed as a share ("50 shares" under Shares), not as a fund. |
| Review screen | Four count chips, three of them "0", and three empty sections ("No changed holdings", …). Quantities as "10.00 units" for shares. "Account hidden" on every row without an account. |

The cause of the Saxo failure: `can_create_security` (`src/wealth/api/broker_imports.py:1015`) accepts a non-Indian row only if its symbol matches `[A-Z0-9.\-]{1,15}`. That check runs even when the row has a valid ISIN.

## Read
- `AGENTS.md` (UX rules, money rules, PWA and Python commands).
- `src/wealth/importer/brokers/parser.py`, `src/wealth/api/broker_imports.py`, `rules/brokers.yaml`, `src/wealth/importer/classify.py`.
- `apps/web/src/imports/ImportPage.tsx`, `apps/web/src/imports/domain.ts`, `apps/web/src/money/quantity.ts`.
- Tests: `tests/importer/test_brokers.py`, `tests/db/test_broker_import_api.py` (DB tests, `@pytest.mark.db`), `e2e/tests/broker-import.spec.ts`.

## Scope

### 1. A row with a valid ISIN always lands
- When a row has an ISIN with a valid check digit, it can be matched or created by that ISIN, whatever its symbol looks like. Keep the existing Indian rule: an `IN…` ISIN needs market `IN`.
- Clean broker symbols before matching and storing:
  - an exchange suffix after `:` moves out of the symbol (`NOVOb:xcse` → symbol `NOVOB`, exchange hint `XCSE`; `IWDA:xams` → `IWDA`, `XAMS`);
  - spaces inside a share-class symbol are removed (`NOVO B` → `NOVOB`).

  Use the exchange hint only where the code already has an exchange field. Don't invent a new mapping table.
- Rows without a valid ISIN keep today's symbol rule.
- If a security already exists with that ISIN, match it. Never create a second security for the same ISIN.

### 2. Read the columns brokers really use
In `rules/brokers.yaml` (holdings), add these aliases:
- `last_price`: `previous closing price`, `previous close`, `prev close` (Zerodha).
- `avg_price`: `open price`, `avg open price`, `average open price` (Saxo).

The existing header tests and the unknown-header test must still pass. These aliases must not create an equal-alias tie with an existing column.

### 3. Never show a value in the wrong currency
- `normalize_header` drops bracketed units, so `Market Value (DKK)` on a EUR row is read as EUR today. Keep the unit when it names a currency: an ISO code (`INR`, `USD`, `EUR`, `DKK`, `GBP`, `SEK`, `NOK`) or a symbol (`₹`, `$`, `€`).
- When a price or value column's unit differs from the row's currency, leave that value out for the row; prices fill it in later. Note in the parse report how many values were left out.
- Same-currency columns behave as today. Never convert currencies in the importer.

### 4. Indian fund units are funds
- An Indian ISIN starting with `INF` is a mutual fund or ETF unit: file it as `etf` in `_security_type`.
- Also read a `Sector` or `Type` column whose value is `ETF` (Zerodha puts "ETF" in Sector). Add `sector` to the holdings aliases only if that doesn't break existing layouts; otherwise read it only for this check.
- Such rows show as units in a funds group, not as shares. Holding reviews and look-through follow the existing ETF rules.

### 5. A shorter, plainer review screen
- **Counts:** show only non-zero counts. For a file that is all new, show one line, e.g. "3 new holdings", instead of four chips. Hide empty sections ("No changed holdings", "Already up to date" with 0, "No items need a decision").
- **Quantities:** use the shared `quantityLabel` from `apps/web/src/money/quantity.ts`: "10 shares", "1.25 shares", "50 units" for funds and ETFs. Never "10.00 units" for a share.
- **Account:** when a row has no account reference, show nothing. Keep `•••• 1234` when there is one, and remove the "Account hidden" text.
- **Values:** format each row's value with the app's existing money formatter for its currency. Don't add a new formatter.
- **Nothing to add:** when a file has nothing that can be added (only unknown or duplicate rows), don't show a disabled "Confirm import". Show one short line saying why, with "Try another file" as the primary action and "Discard" quieter.
- Unknown rows stay out by default. The user doesn't have to tap "Leave out" on each one before confirming the rest.

## Tests
- **Python** (`tests/importer/test_brokers.py` and `tests/db/test_broker_import_api.py`). Use synthetic ISINs with valid check digits (the existing `_synthetic_isin` helper) and invented names:
  - a Saxo-style file (`Symbol` with `:xcse`/`:xams` suffixes, `Open Price`) stages every row as new, with buying prices;
  - re-importing the same file matches the same securities;
  - a Zerodha-style file with preamble rows and `Previous Closing Price` gets a value per row;
  - an `INF…` row is an ETF;
  - a `Market Value (DKK)` column on a EUR row leaves the value out, and the parse report counts it;
  - a row without an ISIN and with a bad symbol is still unknown.
- **Vitest:** the review screen hides zero counts and empty sections, shows "10 shares" and "50 units", has no "Account hidden", and shows "Try another file" when nothing can be added.
- **e2e:** in `broker-import.spec.ts`, add a Saxo-style upload (synthetic ISINs, suffixed symbols) that confirms and appears in Money with its value. Assert no horizontal overflow.

  Codex's sandbox can't start the e2e stack (`spawn EPERM`). Write the spec, run `npx playwright test --list`, and say so in the result file. The reviewer runs the suite.

## Out of scope
- CAS (NSDL/CDSL/CAMS) parsing. Trade history files. Changes to exit or add-more rules.
- New brokers. Currency conversion in the importer.

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
