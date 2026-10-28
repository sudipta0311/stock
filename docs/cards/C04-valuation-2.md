# C04: Valuation engines part 2 (PPF, EPF/VPF, SSY, SCSS, MIS, SGB, gold, bonds, NPS, insurance, ULIP, property, savings, loans)

**Plan:** `docs/UPGRADE_PLAN.md` §3.2, §3.4, §4.11 (insurance return). Read `AGENTS.md` first. Builds on C02 (schemas) and C03 (`Valuation`, `RateBook`, `value_asset`). Reuse C03's conventions exactly: a `MonthYear` means the 1st of the month, actual/365, full Decimal precision inside, round only the final outputs.

## Goal
`value_asset` covers every asset type in the Money Map, so net worth can be computed for any user.

## Scope

### 1. Small schema additions (in `wealth/schemas/assets.py`, then re-run the export)
- `ppf.balance_as_of: MonthYear | None`
- `po_scheme.balance_as_of: MonthYear | None`, used by `ssy` where `amount` means the current balance
- `gold.issue_price_per_gram: Rupees | None` (SGB coupon)
- `bond.coupon_frequency: Literal[1, 2] | None` (default 2 for gsec/rbi_frsb, 1 otherwise)

### 2. Prices protocol: `wealth/valuation/prices.py`
```python
class PriceProvider(Protocol):
    def gold_24k_per_gram(self, on: date) -> Decimal: ...
    def nav(self, key: str, on: date) -> Decimal: ...        # key = ISIN or symbol
class PriceUnavailable(Exception): ...
class StaticPrices:  # simple dict-backed implementation for tests and manual overrides
```
`value_asset` gains `prices: PriceProvider | None = None`. Market-priced types raise `PriceUnavailable` when no price is available.

### 3. Engines (`wealth/valuation/`, pure functions, same signature style as C03)
- **savings / ulip / property:** value = entered amount (`balance`, `fund_value`, `estimated_value`), method `user`.
- **PPF:** interest is credited each 31 March.
  - value_today = balance + simple interest accrued on the balance from `balance_as_of` (or `assumed_start`) to `as_of` at the reference rate, labelled "accrued, credited 31 Mar".
  - maturity_date = **1 April of (opened_fy_start_year + 16)** (15 complete FYs after the opening FY).
  - maturity_value = projection: each remaining FY, add `yearly_contribution` on 1 April, then compound one year at today's rate. Note "projection at current rate".
- **EPF/VPF:** value_today = balance + monthly contributions (employee + employer + vpf) made after `balance_as_of` + interest on the running monthly balance at the FY rate from `rates.epf_rate(fy)`, credited at FY end (simple within the year, compounding yearly). If the EPF rate is unavailable, use 0% accrual plus a note. No maturity.
- **SSY** (`po_scheme`, scheme `ssy`): like PPF using `amount` as the balance. Deposits are allowed for 15 years from opening; maturity is 21 years from opening (`purchase`).
- **SCSS:** simple interest paid quarterly on 1 Apr / 1 Jul / 1 Oct / 1 Jan. value_today = principal + interest accrued since the last payout date. maturity = purchase + 60 months; maturity_value = principal.
- **MIS:** simple interest paid monthly. value_today = principal + accrued for the current month. maturity = purchase + 60 months.
- **Gold:**
  - `physical`: grams × (karat/24) × gold_24k_per_gram.
  - `digital`: grams × gold_24k_per_gram, plus the note "digital gold is not regulated by SEBI".
  - `sgb`: units × gold_24k_per_gram. maturity = issue + 96 months. If `issue_price_per_gram` is set, add a note with the half-yearly coupon amount = units × issue price × 1.25%.
  - `etf` / `mf`: units × nav(key = `series` or `label`).
- **Bonds:**
  - value_today = face_value + coupon accrued since the last coupon date (coupon dates count back from maturity in steps of 12/frequency months), with a note "approximate; listed bonds should use market price".
  - `tbill`: value = face_value with a note "discount instrument; shown at face value".
- **NPS:** value = tier1 + tier2 (method `user`).
- **Term / health insurance:** value_today 0, method `user`, with a note giving the cover amount (protection, not an asset).
- **Traditional policy:**
  - value_today = `surrender_value` if given, else premiums paid so far (years elapsed since start_year, capped at premium_paying_term or term, × annual_premium) with the note "estimate: premiums paid; add surrender value for accuracy".
  - maturity_date = 1 Apr of start_year + term_years.
  - maturity_value = `maturity_illustration` if given.
- **Loans:** don't go through `value_asset`. Add `wealth/valuation/loans.py`:
  - `emi(principal, rate_pct, months) -> Decimal`
  - `amortise(outstanding, rate_pct, emi) -> LoanSchedule` (months_to_payoff, total_interest, payoff MonthYear). Raise `ValueError` when the EMI doesn't cover the monthly interest.

### 3b. Follow-up from the C03 review
When `rates` is `None`, `value_asset` and every engine must use the default rate book, `load_rates()` (cache it with `functools.lru_cache`), instead of raising `RateUnavailable`. Today `value_asset(nsc_bought_nov_2026, ...)` without `rates` raises "No reference rate for 'nsc' on 2026-11-01" even though the default table covers that date. Add a regression test.

### 4. Return helper: `wealth/valuation/returns.py`
`xirr(cashflows: Sequence[tuple[date, Decimal]]) -> Decimal` returns an annual rate as a fraction. Use bisection on NPV with Decimal (no float). Raise `ValueError` if the cash flows don't change sign. Also add `policy_irr(asset: TraditionalPolicy) -> Decimal | None`: premiums at the start of each premium-paying year, `maturity_illustration` received at the end of the term (None if no illustration).

### 5. Required anchor tests
| Case | Expected |
|---|---|
| `policy_irr`: premium ₹52,000 at the start of years 0…20 (21 premiums, start_year 2005, term 21, PPT 21), maturity ₹15,00,000 at the end of year 21 | **2.7978%** (±0.0005 pp) |
| `emi(₹10,00,000, 9%, 240)` | **₹8,997.26** |
| PPF projection: balance ₹5,00,000 on 1 Apr, ₹1,50,000 added each 1 Apr for 5 years (including the first), 7.1% | **₹16,30,260.43** after 5 years (test the projection helper directly) |
| Physical gold 10 g, 22 karat, 24k price ₹7,500/g | **₹68,750.00** |
| SCSS ₹10,00,000 at 8.2%, as_of = 1 Jul | accrued **₹0.00** (paid out that day); on 16 Aug: 46 days of simple interest |
| `amortise` where EMI < monthly interest | raises `ValueError` |

Also add at least one test per asset type through `value_asset`, plus `PriceUnavailable` for gold/ETF without prices. Use `tests/fixtures/rates_fixture.yaml` (extend it with EPF/PPF/SCSS/MIS/SSY fixture rates).

## Out of scope
Fetching real prices (C10), DB/API.

## Acceptance
1. `.venv\Scripts\python -m wealth.schemas.export` (schemas regenerated), then `.venv\Scripts\python -m pytest` → all pass, including every anchor
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. `value_asset` handles every `AssetType` except `loan` (a test enumerates `AssetType` to prove it)
