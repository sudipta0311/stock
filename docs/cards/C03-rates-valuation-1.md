# C03: Scheme-rate reference data + valuation engines part 1 (FD, RD, PO TD, NSC, KVP)

**Plan:** `docs/UPGRADE_PLAN.md` §3.1 (pre-filled rates), §3.4 (valuation engines), §5.2 agent #9 (rates need human approval). Read `AGENTS.md` first. Builds on C02 (`wealth.core`, `wealth.schemas`).

## Goal
Deterministic, Decimal-exact valuation for deposit-style assets, using interest rates that come from a versioned reference file and are never guessed.

## Scope

### 1. Rate reference data: `rules/schemes/small_savings.yaml` + `wealth/rates/`
YAML shape:
```yaml
version: 1
note: "Rates are notified quarterly by the Ministry of Finance. verified:false entries must be human-checked before production use."
schemes:
  nsc:      {compounding: yearly,    tenure_months: 60,  periods: []}
  kvp:      {compounding: yearly,                          periods: []}   # each period also has maturity_months
  td_1y:    {compounding: quarterly, tenure_months: 12,  periods: []}
  td_2y:    {compounding: quarterly, tenure_months: 24,  periods: []}
  td_3y:    {compounding: quarterly, tenure_months: 36,  periods: []}
  td_5y:    {compounding: quarterly, tenure_months: 60,  periods: []}
  rd_5y:    {compounding: quarterly, tenure_months: 60,  periods: []}
  ppf:      {compounding: yearly,                          periods: []}
  scss:     {compounding: simple,    tenure_months: 60,  periods: []}
  mis:      {compounding: simple,    tenure_months: 60,  periods: []}
  ssy:      {compounding: yearly,                          periods: []}
epf:
  periods: []          # one entry per financial year: {fy: "2023-24", rate_pct: "...", source_url: ..., verified: false}
```
Each scheme period entry is `{from: YYYY-MM-DD, to: YYYY-MM-DD | null, rate_pct: "x.x", maturity_months: int (kvp only), source_url: str, verified: false}`.

**Data rule (strict):** only add a period if you fetched an official source during this run that states it: finmin.nic.in, dea.gov.in, indiapost.gov.in, nsiindia.gov.in, epfindia.gov.in, or rbi.org.in. Put that exact URL in `source_url`. **Never fill rates from memory.** Every entry is `verified: false`. If you can't fetch a source for a scheme, leave its `periods` empty and say so in the result file. An empty file is acceptable; invented numbers are not.

`wealth/rates/reference.py`:
- `load_rates(path: Path | None = None) -> RateBook` (defaults to `rules/schemes/small_savings.yaml`; validated with Pydantic).
- `RateBook.rate_for(scheme: str, on: date) -> SchemeRate` returns `rate_pct`, `maturity_months`, `verified` and `source_url` for the period containing `on`. Raises `RateUnavailable(scheme, on)` when there's no covering period.
- `RateBook.epf_rate(fy: str) -> SchemeRate`.

Tests use `tests/fixtures/rates_fixture.yaml` with clearly fake numbers (comment in the file: `# FIXTURE ONLY: not real rates`), never the real file.

### 2. Valuation result type: `wealth/valuation/base.py`
```python
@dataclass(frozen=True)
class Valuation:
    value_today: Decimal          # rupees, 2 dp
    maturity_value: Decimal | None
    maturity_date: date | None
    accrued_interest: Decimal     # value_today - principal contributed so far (>= 0 for these types)
    principal_so_far: Decimal
    method: ValuationMethod       # formula
    notes: tuple[str, ...]        # human-readable assumptions, e.g. "start month missing: assumed 2026-10"
```
Date convention: a `MonthYear` means the **1st of that month**. Day count is actual/365. Round only the final outputs to paise (ROUND_HALF_UP). Keep full Decimal precision inside (set a local context with `prec=40`).

### 3. Engines (`wealth/valuation/deposits.py`), all pure functions
Each takes `(asset, as_of: date, *, rates: RateBook | None = None, assumed_start: date | None = None) -> Valuation`.

- **FD** (`value_fd`):
  - Compounding per `asset.compounding` (default quarterly): complete periods compound; leftover days earn simple interest on the compounded amount.
  - `payout != cumulative`: value_today = principal + interest accrued since the last payout date; maturity_value = principal.
  - If `start` is missing, use `assumed_start` and add a note. If both are missing, value_today = principal, maturity_value = None, plus a note "add start month for exact value".
  - If `as_of` ≥ maturity: value_today = maturity_value.
- **RD** (`value_rd`) for bank and post-office RDs: installment *i* (1-based, paid on the start day of month *i*) grows as `R × (1 + r/4)^(4 × months_remaining_i / 12)`, the standard Indian RD formula with a fractional quarterly exponent. value_today sums the installments paid up to `as_of`, compounded to `as_of`.
- **PO Time Deposit** (`value_po_td`): quarterly compounding, interest **paid out annually**. annual_interest = `P × ((1 + r/4)^4 − 1)`. value_today = principal + interest accrued since the last anniversary (pro-rata on the compounded basis). maturity_value = principal. Expose `annual_payout` in `notes` as text and as a helper `po_td_annual_interest(principal, rate_pct)`.
- **NSC** (`value_nsc`): annual compounding, paid at maturity (60 months). Full years compound; the partial year accrues simple interest. Rate = `asset.rate_pct` if set, else `rates.rate_for("nsc", purchase_date)`, which is **locked at purchase**.
- **KVP** (`value_kvp`): maturity_months from the rate table for the purchase date. value_today = `P × 2^(months_elapsed / maturity_months)`, capped at 2P from maturity onward.
- `value_asset(asset, as_of, *, rates, assumed_start=None)`: a dispatcher for the types above. `po_scheme` routes by `scheme` (`nsc`, `kvp`, `td` → `td_{n}y`, `rd` → `rd_5y`). Other asset types raise `NotImplementedError` (C04 adds them).
- Whenever a rate comes from the table with `verified: false`, add the note `"rate from reference table (unverified)"`.
- When the rate is unavailable and the asset has no `rate_pct`, raise `RateUnavailable`. The API will ask the user to type the rate from their certificate.

### 4. Required anchor tests (exact to the paisa unless stated)
| Case | Expected |
|---|---|
| FD ₹1,00,000 at 7% quarterly, exactly 1 year (start 2025-01-01, as_of 2026-01-01) | maturity/value **₹1,07,185.90** |
| FD ₹1,00,000 at 7% quarterly, 5 years | **₹1,41,477.82** |
| NSC ₹1,000 at 7.7%, at maturity (60 months) | **₹1,449.03** (matches India Post's published ₹1,449) |
| NSC ₹1,00,000 at 7.7%, at maturity | **₹1,44,903.38** |
| PO TD 5-year ₹1,00,000 at 7.5%: `po_td_annual_interest` | **₹7,713.59** (India Post publishes ₹7,714) |
| RD ₹1,000/month at 6.7% for 60 months, at maturity | **₹71,365.83** |
| KVP ₹50,000 with maturity_months = 115, at month 115 | **₹1,00,000.00**; at month 200 still ₹1,00,000.00 |

Also:
- Cross-check RD by an independent month-by-month simulation in the test (the two must agree to the paisa).
- FD non-cumulative payout.
- FD missing start, with and without `assumed_start`.
- `as_of` before start → value = principal, accrued 0.
- `as_of` after maturity.
- Rate lookup boundaries (period `from`/`to` inclusive).
- `RateUnavailable` raised.
- The unverified note is present.

## Out of scope
PPF, EPF, SSY, SCSS, MIS, SGB, bonds, gold, NPS, insurance, loans (that's C04). DB/API.

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass (including all anchors above)
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. `.venv\Scripts\python -c "from wealth.rates.reference import load_rates; b=load_rates(); print({k: len(v.periods) for k, v in b.schemes.items()})"` prints the period count per scheme (zeros are acceptable)
4. The result file lists every `source_url` fetched and which schemes are still empty.
