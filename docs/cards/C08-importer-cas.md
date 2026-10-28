# C08: Importer part 1: CAS parsers (NSDL port, CAMS/KFintech, CDSL) → normalised holdings, with synthetic fixtures only

**Plan:** `docs/UPGRADE_PLAN.md` §3.3 (statement imports: what each document gives, import pipeline, no file retention, password never stored), §7.4 (security). Read `AGENTS.md` first. Reference the old parser `legacy/src/stock_platform/services/pdf_parser.py`, but don't import from `legacy/`.

## Goal
Turn a password-protected CAS PDF into a clean list of normalised holdings (and, where the statement has them, transactions/lots), entirely in memory, with no LLM. Tests never use a real statement.

## Decisions (follow them)
- Two layers, so most logic is testable without PDFs:
  1. `pdf_text(data: bytes, password: str | None) -> list[str]` (decrypt + extract text lines in order).
  2. Pure parsers `parse_nsdl(lines)`, `parse_cdsl(lines)`, `parse_cams_kfin(lines)` → `ParsedStatement`.
- `detect_cas_type(lines) -> Literal["nsdl", "cdsl", "cams_kfin", "unknown"]` uses header markers.
- Libraries: `pypdf>=6.1` (decrypt/extract). For CAMS/KFintech, **evaluate the `casparser` package** (MIT, widely used for Indian MF CAS): add it if it installs cleanly on Python 3.12–3.14 without PyMuPDF, otherwise write the parser yourself. Note the outcome in the result file. Put these in the `importer` extra. The API extra will include `importer` later (C08b); keep the importer's import time and memory small (no pandas).
- **CDSL:** if `casparser` doesn't support CDSL CAS and you have no reliable format spec, implement `detect_cas_type` for it and return a `ParseError(code="unsupported_format", detail="CDSL CAS support is coming soon. Upload your NSDL CAS or broker statement instead.")`. Don't guess a format.
- Password errors raise `ParseError(code="wrong_password")`; non-PDF or corrupt input → `ParseError(code="not_a_pdf")`; over the size cap (default 10 MB) → `ParseError(code="too_large")`; unknown layout → `ParseError(code="unsupported_format")`. **Never** include the password, file bytes or statement text in exceptions or logs.

## Scope

### 1. Normalised output (`wealth/importer/models.py`, Pydantic)
- `ParsedStatement`: `source` (`nsdl`/`cdsl`/`cams_kfin`), `statement_date: date`, `holdings: list[ParsedHolding]`, `transactions: list[ParsedTransaction]`, `warnings: list[str]`.
- `ParsedHolding`:
  - `kind` (`equity`/`etf`/`mf`/`bond`/`sgb`/`other`), `isin` (validated: 12 chars, ISIN check digit; invalid → kept with warning `bad_isin`), `name`, `quantity: Decimal`
  - `market_value: Rupees | None`, `price: Decimal | None`, `cost_value: Rupees | None`
  - `account_ref`: **masked**, e.g. last 4 of DP ID/client ID or folio, `XXXX1234`
  - `amc` / `registrar` (optional)
- `ParsedTransaction`: `isin`, `date`, `kind` (`buy`/`sell`/`switch_in`/`switch_out`/`dividend_reinvest`/`other`), `units`, `amount`, `nav`.
- Add `isin_check_digit_ok(isin) -> bool` (Luhn over letters converted to numbers) in `wealth/core/ids.py`, with tests on real-format sample ISINs such as `INE002A01018` and `US0378331005` (public identifiers, not personal data).

### 2. Parsers (`wealth/importer/cas/`)
- `nsdl.py`: port the legacy NSDL logic (demat equities, ETFs, bonds/SGBs held in demat, and MF folios section) into the pure `parse_nsdl(lines)`. Keep what worked; drop everything platform-specific. Statement date from the header.
- `cams_kfin.py`: holdings per folio + transactions (cost basis via buys/sells).
- `cdsl.py`: per the decision above.
- `__init__.py`: `parse_cas(data: bytes, password: str | None, *, max_mb=10) -> ParsedStatement` = size check → `pdf_text` → `detect_cas_type` → parser.

### 3. Synthetic fixtures (no real data, ever)
- `tests/fixtures/cas/`: text-line fixtures for NSDL and CAMS/KFin that mimic the real layouts closely enough to exercise every parser branch:
  - names like `SYNTHETIC INVESTOR`
  - fake but check-digit-valid ISINs (generate them with your own check-digit function)
  - DP/client IDs like `IN300000`/`12345678`
- `tests/fixtures/cas/make_pdf.py` (test helper) builds a small encrypted PDF from those lines at test time, using `fpdf2` (dev extra) + `pypdf` encryption, so the full `parse_cas` path is tested. Generated PDFs are created in a temp dir and **never committed**.

### 4. Tests
- `detect_cas_type` for each fixture plus a random text → unknown.
- NSDL fixture: exact holdings count, quantities, values, kinds, masked account refs, statement date.
- CAMS/KFin fixture: holdings + transactions; cost value from transactions.
- Full `parse_cas` on generated encrypted PDFs: right password OK; wrong password → `wrong_password`; bytes `b"hello"` → `not_a_pdf`; 11 MB → `too_large`.
- No exception message contains the password (assert on `str(exc)`).
- An ISIN with a bad check digit → warning `bad_isin`, holding kept.
- Timing: parsing the generated NSDL PDF takes < 3 s (report the time).

## Out of scope
Upload API, staging tables, review/confirm, writing holdings to the DB (C08b). Broker statements (C09).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,importer,dev]"`
2. `.venv\Scripts\python -m pytest` → all pass
3. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
4. `git status` shows no `.pdf` files anywhere outside `legacy/` (and none inside either)
