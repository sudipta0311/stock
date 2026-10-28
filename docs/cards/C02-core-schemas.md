# C02: Core helpers + asset/profile schemas + JSON Schema export

**Plan:** `docs/UPGRADE_PLAN.md` §3.1 (Indian number input, month-year pickers), §3.2 (asset catalogue), §7.3 (`assets_manual.details` validated by per-type JSON Schema). Read `AGENTS.md` first.

## Goal
Build the shared building blocks every later card uses: Decimal money helpers with Indian parsing and formatting, month-year dates and Indian financial years, and Pydantic models for every asset type a user can enter, exported as JSON Schema for the API and the PWA.

## Scope

### 1. `wealth/core/money.py`
- `parse_inr(text: str) -> Decimal`. Accepts, case-insensitive, with optional `₹`, `Rs`, `Rs.`, `INR` prefix and surrounding spaces:
  - plain and comma-grouped numbers in Indian or international grouping: `500000`, `5,00,000`, `500,000`, `1,23,45,678.50`
  - suffixes: `k`/`K`/`thousand` (×1,000), `l`/`L`/`lac`/`lakh`/`lakhs` (×1,00,000), `cr`/`crore`/`crores` (×1,00,00,000), with or without a space: `5L`, `5 lakh`, `1.2 cr`, `2.5Cr`, `50k`
  - Returns an exact `Decimal` (no float anywhere). Rejects with `ValueError` (clear message): empty, negative, garbage (`abc`, `5L5`, `1..2`, `5 lakh crore`), more than 2 decimal places after scaling to rupees.
- `format_inr(amount: Decimal, style: Literal["full", "compact"] = "full", decimals: int = 2) -> str`
  - `full`: Indian grouping, e.g. `Decimal("12345678.5")` → `₹1,23,45,678.50`; `decimals=0` → `₹1,23,45,679` (ROUND_HALF_UP).
  - `compact`: `< 1,00,000` → full with 0 decimals (`₹99,999`); `< 1 crore` → lakhs with up to 2 decimals, trailing zeros trimmed (`₹5.4 L`, `₹12.35 L`); otherwise crores (`₹1.24 Cr`, `₹12 Cr`).
  - Negative amounts render with a leading minus: `-₹5,000.00`.
- `format_usd(amount: Decimal, decimals: int = 2) -> str` → `$1,234.50` (international grouping).
- `round_paise(amount: Decimal) -> Decimal` → 2 dp, ROUND_HALF_UP.
- `pct(value: str | int | Decimal) -> Decimal` that validates 0 ≤ value ≤ 100.

### 2. `wealth/core/dates.py`
- `MonthYear`: frozen dataclass or Pydantic model `(year: int, month: int)`, ordered and hashable, with `first_day()`, `last_day()`, `add_months(n)`, and `months_until(other) -> int`.
- `parse_month_year(text) -> MonthYear` accepts `Mar 2027`, `March 2027`, `03/2027`, `3/2027`, `2027-03`, `Mar-27` (two-digit year means 20xx). Rejects garbage with `ValueError`.
- `financial_year(d: date) -> str` returns e.g. `"2026-27"` (Indian FY runs 1 Apr to 31 Mar). Also `fy_start(d)` and `fy_end(d)`.
- `add_months(d: date, n: int) -> date` clamps to month end (31 Jan + 1 month → 28/29 Feb).

### 3. `wealth/schemas/common.py`
Enums (StrEnum): `AssetType`, `Sleeve` (`equity_india`, `equity_global`, `debt`, `gold`, `cash`, `real_estate`, `protection`), `LiquidityClass` (`instant`, `within_1y`, `locked`), `ValuationMethod` (`formula`, `user`, `statement`, `market`), `Compounding` (`simple`, `monthly`, `quarterly`, `half_yearly`, `yearly`), `PayoutType` (`cumulative`, `monthly`, `quarterly`, `half_yearly`, `yearly`). Plus Pydantic-friendly annotated types: `Rupees` (Decimal ≥ 0, max 2 dp; accepts strings via `parse_inr`), `RatePct` (Decimal 0–30, max 4 dp), and `MonthYear` serialised as `"YYYY-MM"`.

### 4. `wealth/schemas/assets.py`
One model per asset type in plan §3.2. Each has `asset_type: Literal[...]` (the discriminator), `member_id: str | None = None`, `label: str | None = None` (max 80 chars), `notes: str | None = None` (max 500), plus:

| `asset_type` | Required | Optional |
|---|---|---|
| `savings` | `bank`, `balance` | — |
| `fd` | `bank`, `principal`, `rate_pct`, `maturity: MonthYear` | `start: MonthYear`, `compounding` (default quarterly), `payout` (default cumulative), `joint: bool` |
| `rd` | `institution`, `monthly_amount`, `rate_pct`, `maturity` | `start` |
| `ppf` | `balance`, `opened_fy_start_year: int` | `yearly_contribution`, `held_at` |
| `epf` | `balance` | `monthly_employee`, `monthly_employer`, `vpf_monthly`, `balance_as_of: MonthYear` |
| `nps` | `tier1_balance` | `tier2_balance`, `equity_pct`, `corporate_debt_pct`, `govt_debt_pct`, `alternate_pct` (must sum to 100 when given) |
| `po_scheme` | `scheme` (`nsc`, `kvp`, `scss`, `mis`, `td`, `rd`, `ssy`), `amount`, `purchase: MonthYear` | `td_years` (1, 2, 3 or 5; required when scheme = td), `rate_pct` (override) |
| `bond` | `kind` (`gsec`, `tbill`, `rbi_frsb`, `corporate`, `tax_free`, `bond_54ec`), `face_value`, `maturity` | `coupon_pct`, `isin`, `issuer` |
| `gold` | `kind` (`sgb`, `etf`, `mf`, `physical`, `digital`) | `units` (sgb/etf/mf), `grams` + `purity` (physical: 24/22/18 karat), `series`, `issue: MonthYear`, `purpose` (`jewellery` or `investment`). Validate that the right fields exist for each kind. |
| `term_insurance` | `insurer`, `cover`, `ends_year` | `premium`, `premium_frequency` |
| `health_insurance` | `insurer`, `cover`, `renewal_month: int (1-12)` | `members_covered`, `top_up_cover` |
| `traditional_policy` | `insurer`, `plan`, `policy_kind` (`endowment`, `money_back`, `whole_life`), `sum_assured`, `annual_premium`, `start_year`, `term_years` | `premium_paying_term`, `bonus_accrued`, `surrender_value`, `maturity_illustration` |
| `ulip` | `insurer`, `fund_value`, `annual_premium` | `equity_pct`, `start_year` |
| `property` | `property_kind`, `estimated_value` | `linked_loan_id`, `monthly_rent` |
| `loan` | `loan_kind` (`home`, `car`, `personal`, `education`, `credit_card`, `gold`, `other`), `outstanding`, `rate_pct` | `emi`, `end: MonthYear` |

- `AnyAsset = Annotated[Union[...], Field(discriminator="asset_type")]` plus `parse_asset(data: dict) -> AnyAsset`.
- `default_sleeves(asset) -> dict[Sleeve, Decimal]`: fractions summing to 1. Use the plan §3.2 "Counts toward" column. NPS splits by its percentages (equity → equity_india; the rest → debt), defaulting to 50/50 when they're missing. Traditional policies → `debt`. Term/health → `protection`. Property → `real_estate`. Loans return `{}` (liabilities are handled separately).
- `default_liquidity(asset) -> LiquidityClass`, following the plan §3.2 Liquidity column.

### 5. `wealth/schemas/profile.py`
`TaxRegime` (`old`, `new`), `GoalType` (`emergency`, `education_india`, `education_abroad`, `home`, `retirement`, `wealth`, `other`), `Goal` (`name`, `goal_type`, `target_amount: Rupees`, `target_year: int`, `currency: Literal["INR", "USD"]`, `priority: int 1-5`), `HouseholdMember` (`id`, `display_name` max 40, `relationship`), `Profile` (`tax_regime`, `tax_slab_pct: RatePct | None`, `city_tier: Literal[1, 2, 3] | None`, `timezone` default `Asia/Kolkata`, `language` default `en`).

### 6. JSON Schema export
`python -m wealth.schemas.export` writes `schemas/assets/<asset_type>.json` (one per type), `schemas/assets/any_asset.json` (the union) and `schemas/profile.json`. The output is deterministic (sorted keys, 2-space indent, trailing newline). Commit-ready files must be generated and present.

## Out of scope
Valuation formulas, DB, API.

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[dev]"`
2. `.venv\Scripts\python -m pytest`. Must include table-driven tests covering at least: 25 `parse_inr` cases (valid + invalid), 12 `format_inr` cases (full + compact + negative), 10 `parse_month_year` cases, FY edge dates (31 Mar / 1 Apr), `add_months` month-end clamping, one valid and one invalid example per asset type, discriminated-union parsing from a plain dict for every type, `default_sleeves` sums to 1 for every non-loan type, and a test that the files in `schemas/` match a fresh export (fails if stale).
3. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
4. `.venv\Scripts\python -m wealth.schemas.export` runs and leaves `git status` showing only the expected files in `schemas/`
