# C48: Broker files that just work: HDFC Securities, Zerodha, INDmoney, Nordnet, Saxo and others

**Why:** the owner wants to upload broker statements from India (HDFC Securities, Zerodha), US shares bought from India (INDmoney) and Danish brokers (Nordnet, Saxo, others). The engine from C09/C09b exists, but review found these gaps:

1. **Indian brokers can't be picked.** The broker picker lists only Danish brokers, IBKR, Vested, INDmoney and Other. An HDFC Securities file goes in as "Other", so `broker_defaults` gives it **US / USD**.
2. **UTF-16 files are refused.** `_looks_like_text_csv` rejects any file containing a zero byte, so every UTF-16 export fails with "Choose a PDF, CSV, or XLSX". Several sources describe Nordnet's export as UTF-16 with tabs (unverified, so test both UTF-16 and UTF-8).
3. **`.xls` is refused.** That includes the many Indian broker "Excel" downloads that are really an HTML table saved as `.xls`.
4. **The column guesser knows only English words.** Danish headers (Antal, Kurs, Værdipapir, Handelsdag, Transaktionstype, Beløb, Kurtage, GAK, …) and common Indian headers ("Scrip Name", "Avg Cost Price", "Invested Value") aren't recognised. Every such file lands in the column picker, where the user maps up to 14 fields by hand.
5. **Market comes from the broker, not the share.** A US share bought through Nordnet gets market `DK` (only an `IN…` ISIN overrides the broker default).
6. **Only buy/sell rows are read**, in English only. Danish trade words (KØBT/SOLGT) are dropped silently.

No public source documents the exact export columns of HDFC Securities, INDmoney, Nordnet or Saxo. The reviewer searched on 2026-11-04 and found only forum posts. So, as C09 decided, **don't seed layouts with invented columns**. Make recognition broad, data-driven and safe instead, and let the first upload teach the app (saved templates).

Read `AGENTS.md` first (UX rules, no personal data in fixtures), and `docs/cards/C09-importer-brokers.md`, `C09b-importer-us.md` and `C21-pwa-import.md` for the existing flow.

## Behaviour

### A. Choosing the broker (PWA import screen)
- **The broker picker is grouped,** with plain labels:
  - **India:** Zerodha, HDFC Securities, Groww, ICICI Direct, Angel One, Upstox, Other Indian broker
  - **US shares from India:** INDmoney, Vested
  - **Denmark:** Nordnet, Saxo, Danske Bank, Other Danish broker
  - **Other:** Interactive Brokers, Other
- **The default choice** comes from the selected member's country of residence: India → Zerodha; Denmark → Nordnet; else Other. Remember the last choice per member in `localStorage`, guarded by try/catch.
- **The API accepts the new values.** Each has defaults in one table (`rules/brokers.yaml`, a new `brokers:` section, not code):
  - Indian brokers → `IN`/`INR`
  - INDmoney, Vested → `US`/`USD`
  - Danish brokers → `DK`/`DKK`
  - IBKR, Other → `US`/`USD` (today's default)
- **The help text** under the file picker is one sentence: "In your broker's portfolio or reports page, choose Export or Download, and pick Excel or CSV."

### B. Files the app can read (`wealth/importer/brokers/parser.py` + the upload check)
- **Accepted:**
  - `.csv`, `.tsv`, `.txt` (text tables) and `.xlsx`, as today
  - UTF-8 (with or without BOM), UTF-16 LE/BE (with BOM, or detected by the zero-byte pattern), cp1252
  - delimiters: comma, semicolon, tab
- **`.xls`:**
  - **An HTML table saved as `.xls`** (starts with `<` or contains `<table`): read the first table with a header row, using the stdlib `html.parser`. Never run scripts or follow links. Cap rows and cells as the CSV path does.
  - **Real Excel 97–2003** (OLE2 header `D0 CF 11 E0`): read with `xlrd>=2.0.1`, added to the `importer` extra (cached values only; nothing is evaluated).
- **A PDF uploaded as a broker statement** gets a friendly 415: "This looks like a PDF. Please download the Excel or CSV version from your broker." CAS PDFs are unaffected.
- **The upload check (`_looks_like_text_csv`) and `read_table` share one detection function,** so they can't disagree again.
- **Numbers:**
  - Indian grouping (`1,23,456.78`)
  - Danish/European style (`1.234,56`, `1 234,56`, NBSP grouping)
  - minus signs: `-`, `−` (U+2212) and accounting brackets `(1.234,56)`
  - currency codes or symbols next to the number
  - Rule: a value with a single comma and 1–2 digits after it is a decimal comma when the row's currency isn't INR (keep the current rule). The INR path stays strict.
- **Dates:** add `DD.MM.YYYY`, `DD-MM-YY`, `YYYY/MM/DD`, `DD Mon YYYY` and `Mon DD, YYYY` (US style) to the existing formats. Excel serial dates in xlsx/xls cells too.

### C. Recognising columns
- **Move the alias table** (`_ALIASES` in `parser.py`) into `rules/brokers.yaml` as `aliases:`. Keep **kind-specific aliases where words differ**: "kurs" is the price in a transactions file and the current price in a holdings file.
- **Add English, Danish, Norwegian and Swedish words and common Indian broker words.** At least:
  - name: værdipapir, værdipapirer, papir, navn, instrument name, scrip name, stock name, company, security, description, fund name
  - symbol: kortnavn, ticker symbol, nse symbol, scrip code, trading symbol
  - isin: isin, isin code, isin no
  - quantity: antal, beholdning, quantity available, total qty, free qty, net qty, no of shares, number of shares, shares owned, units held
  - avg_price: gak, gns. købskurs, gns kurs, anskaffelseskurs, købskurs, avg cost price, average cost price, average buy price, cost basis per share
  - **cost_value (new canonical field):** indkøbsværdi, anskaffelsesværdi, invested value, investment value, total cost, cost basis, buy value
  - last_price: seneste kurs, senest, current market price, closing price, ltp, cmp (holdings: kurs)
  - market_value: markedsværdi, værdi, current value, present value
  - currency: valuta
  - trade_date: handelsdag, handelsdato, order date, execution date (prefer the trade date over bogføringsdag/valørdag; booking dates are a fallback)
  - trade_type: transaktionstype, buy/sell, side, b/s
  - price: kurs, pris, execution price (tradebook)
  - amount: beløb, total amount
  - commission: kurtage, brokerage, samlede afgifter
  - trade_id: verifikationsnummer, trade no
- **`cost_value`:** when a holdings row has a cost total but no average price, average price = cost_value ÷ quantity (Decimal; no rounding until stored).
- **Trade words:**
  - buy = buy, b, bought, purchase, køb, købt, kjøpt, köp, köpt
  - sell = sell, s, sold, sale, salg, solgt, solgte, sålt
  - Matched case-insensitively after normalisation.
  - **Other rows** (dividends, deposits, withdrawals, fees, interest, transfers…) are **skipped and counted** in the parse report (`skipped_rows: {"not_a_trade": n}`), never errors.
- **Market and currency per row:**
  1. a market column
  2. the ISIN prefix (`IN` → IN; `US` → US; `DK` → DK; `GB` → GB; any other valid ISIN → EU)
  3. the broker default

  Currency: the currency column, else the market's currency (IN → INR, US → USD, DK → DKK), else the broker default.
  So a US share in a Nordnet file is a US share in USD.

### D. Auto-mapping (fewer questions)
When no layout or saved template matches, use the suggested mapping. **If it covers a kind's required fields unambiguously, parse at once** instead of opening the column picker:
- **Holdings:** one of isin/symbol/name, plus quantity.
- **Tradebook:** trade_date, trade_type, quantity, price, plus isin or symbol.
- **Choosing the kind:** if both fit, tradebook wins only when trade_type values in the rows are mostly buy/sell words; otherwise holdings.
- **Ambiguous** = two different headers both match one required field (e.g. "Kurs" and "Seneste kurs" for last_price): prefer the more specific alias. A real tie → the column picker.
- **The job continues as `ready`,** with `parse_report.auto_mapped = true` and `parse_report.columns_used` (field → header name; header names only, never row data).
- **The review screen** shows one quiet line, "We matched your file's columns automatically.", and a secondary "Fix columns" link. The link opens the existing column picker, prefilled, and re-parses on save.
- **On confirm, the mapping is saved as the household's template** (name = the broker label), so the same file is recognised next time without the line.
- **The column picker** (when still needed) shows the **required fields first** and puts the rest behind "More columns" (closed). Market/currency questions appear only when no row could infer them.

### E. Holdings without cost
A holdings file with no average price or cost:
- imports quantities as today
- on the review screen, shows one quiet line: "This file has no buying prices, so tax estimates and some checks will be missing. A trade history file fills them in."
- doesn't block confirm

## Tests (synthetic fixtures only: invented names, check-digit-valid synthetic ISINs, build files at test time)
- **Detection matrix:** UTF-8 comma CSV, UTF-8-BOM semicolon CSV, UTF-16 LE tab file with BOM, UTF-16 LE without BOM, cp1252, HTML-table `.xls`, real `.xls` (build with a small helper; if `xlrd` can't be fed a generated file without `xlwt`, commit one tiny synthetic `.xls` fixture made of invented values and say how it was made), xlsx, PDF-as-broker (friendly 415).
  - The upload check and `read_table` agree on every case.
- **Numbers:** a table of ≥ 15 cases (Indian, Danish, NBSP, U+2212, brackets, currency symbols), including ones that must fail.
- **Dates:** a table for every format, Excel serials included.
- **Aliases:** a Danish holdings file (Værdipapir, ISIN, Antal, GAK, Kurs, Markedsværdi, Valuta) and a Danish transactions file (Handelsdag, Transaktionstype KØBT/SOLGT/UDBYTTE/INDBETALING, Værdipapirer, ISIN, Antal, Kurs, Beløb, Kurtage, Valuta) both auto-map and parse.
  - Dividends and deposits are counted as `not_a_trade`.
  - A US ISIN row is market US / USD; an IE ISIN row is EU.
- **Indian-style holdings file** with "Scrip Name, ISIN, Qty, Avg Cost Price, LTP": auto-maps; market IN/INR via the broker default *and* via the ISIN.
- **US-style file** with "Symbol, Quantity, Invested Value": auto-maps; average price = invested ÷ quantity; market US.
- **Ambiguous headers** go to the column picker; a real tie is not auto-mapped.
- **Saved template:** confirming an auto-mapped job saves a template; the next upload of the same headers is `ready` without `auto_mapped`.
- **Broker defaults:** each new broker value gets its market/currency. Old values keep today's behaviour. An unknown value → 422.
- **Privacy:**
  - the parse report holds header names only, never row values
  - masked samples stay masked
  - no file is written to disk
  - household isolation for templates as today
- **PWA:**
  - the grouped picker and its default by residence
  - the auto-mapped line and "Fix columns"
  - required-first column picker with "More columns" closed
  - the no-cost line
  - the PDF message
  - no overflow at 384 px; axe scan
- **e2e:** through the real import screen, upload three synthetic files and, for each, see the auto-mapped review, confirm, and see the holding in Money:
  - an Indian-broker-style holdings file saved as an HTML-table `.xls`, picked as "HDFC Securities"
  - a US-shares holdings CSV, picked as "INDmoney"
  - a Danish holdings file (UTF-16, tab-separated, Danish words and numbers) with one UCITS ETF and one US share, picked as "Nordnet". Answer the existing account question with the share savings account. The holding shows in Money under that account, and the US share has market `US`.

  These match how the owner invests (2026-11-04): Indian shares and funds from India, US shares through INDmoney, and mainly ETFs plus some global shares in a Danish share savings account (ASK).

## Acceptance
- `.venv\Scripts\python -m pip install --no-cache-dir -e ".[dev,api,importer]"` after adding `xlrd`; then `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache`, `.venv\Scripts\python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`
- The e2e suite runs as `docs/cards/C24-e2e.md` describes (the reviewer also runs it).
- If `xlrd` must be importable by the API service, check the Render build installs the `importer` extra. Say what you found.
- Write `docs/cards/C48.result.md` with:
  - the full alias list, by field and language
  - which brokers have defaults
  - every place that decides a row's market or currency

## Notes
- No migration expected (templates and parse reports exist). If you find you need one, it is `0029` revising C47's `0028`.
- No LLM, and no guessing of columns from row values beyond the kind check in D.
- Out of scope: PDF broker statements; email (Gmail) import of broker files; DK/EU share prices (shares without a price feed keep the statement price, as today); sell or "add more" advice (C49).
