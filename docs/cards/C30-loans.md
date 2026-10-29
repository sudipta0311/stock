# C30: Loans: track every loan, record payments, and decide "pay extra or invest"

**Requested by the product owner:** "Add options to manage and track the loans."
**Plan:** `docs/UPGRADE_PLAN.md` §3 (the Loans row in the asset table: amortised; prepay-vs-invest nudge if the rate is above the expected return), the foundation check (high-interest debt first), §8 (UX). Read `AGENTS.md` first, including the **UX rules** and the **household RLS pattern**. Builds on the existing `LoanAsset` (`schemas/assets.py`), `valuation/loans.py` (EMI and amortisation), `policy/foundation.py` (the high-interest debt rule), C27 households and members, and the C28/C28b UI.

## Goal
Today a loan is a static number. After this card:
- the balance goes down by itself every month
- the person can record what really happened: extra payments, rate changes, a new EMI, a balance from a statement
- they always see what is left, when it ends and how much interest is still to pay
- the app says, in one plain sentence, whether spare money should go to paying the loan off faster or to investing

Code decides every amount; nothing here uses an LLM.

## Scope

### 1. Data
- **Extend `LoanAsset`** (backward compatible; existing rows must still load):
  - `lender` (optional)
  - `currency` (`INR` default, `DKK`, `EUR`). Amounts are in that currency; net worth converts with the FX table like the C09b holdings.
  - `outstanding_as_of` (MonthYear; default = the month the row was last saved)
  - `emi_day` (1–28, optional)
  - `rate_type` (`fixed` | `floating`, optional)
  - `original_amount` and `start` (optional; unlock "% repaid")
  - `tax_relief` (`none` | `in_home_self_occupied` | `in_home_let_out` | `in_education` | `dk_interest_deduction`, default `none`)
- **New table `loan_events`.** Use the next free migration number and revise the current head. It follows the household RLS pattern from AGENTS.md: policy, audit trigger, and composite `(user_id, member_id)` FK `DEFERRABLE INITIALLY IMMEDIATE`; the FK to the loan row goes through the tenant key.
  - Columns:
    - `id`, `user_id`, `member_id`, `loan_id`
    - `kind`: `prepayment` | `rate_change` | `emi_change` | `balance_update` | `missed_emi` | `closed`
    - `on_date`, `amount`, `rate_pct`, `emi`
    - `note` (≤ 140 chars, plain text)
  - Deleting a loan deletes its events. Add the table to `app_merge_into_household` handling only if it needs a duplicate rule; say which in the result file.

### 2. Engine (pure, Decimal; `wealth/valuation/loans.py` + `wealth/policy/loans.py`)
- **`loan_position(loan, events, as_of)`:**
  - Anchor on the latest `balance_update` (else `outstanding` at `outstanding_as_of`).
  - Roll forward month by month: monthly interest at the current rate (`rate/1200`), then the EMI. EMIs are assumed paid unless a `missed_emi` event exists for that month.
  - Prepayments reduce principal on their date's month. Rate and EMI changes apply from their month.
  - Returns:
    - outstanding today
    - principal and interest paid since the anchor
    - months left and the payoff month
    - interest still to pay
    - next EMI date
    - % repaid when `original_amount` is known
    - a yearly balance series (for a small chart)
  - **Guard:** if the EMI doesn't cover the monthly interest, return `never_pays_off` and show a plain warning instead of a payoff date.
- **`prepay_options(loan, events, amount, as_of)`:** two outcomes for the same extra payment:
  - (a) keep the EMI, finish earlier: months saved and interest saved
  - (b) keep the end date, lower the EMI: the new EMI and interest saved
- **`pay_extra_or_invest(loans, plan_context)` (code decides):**
  - **Effective loan rate** = rate minus tax relief, from rule data in `rules/tax.yaml` (India home-loan interest and education-loan relief, by regime and caps) and `rules/residency.yaml` (Danish interest deduction). Every rule has a source URL and stays `verified: false`; the detail view says "please check".
  - **Credit cards and loans above the foundation's high-interest threshold:** always "pay this first".
  - **Never** suggest prepaying below the emergency-fund target.
  - Otherwise compare the effective rate with the plan's expected after-tax return for safe money (from existing rule data). If the loan costs more, suggest paying extra; if not, suggest investing.
  - Several loans: the highest effective rate goes first (avalanche).
  - Output: `PAY_EXTRA` | `INVEST` | `BUILD_SAFETY_FIRST`, with reason codes and a suggested amount from this month's spare money.
- **Debt load:** total monthly EMIs ÷ monthly income (when the income is known). The thresholds live in rule data: under 30% fine, 30–50% attention, over 50% high. This produces a highlight, never a block.

### 3. Plan, notes, reminders
- **Net worth and the plan** use the rolled-forward outstanding (all currencies converted).
- **New action kind `PAY_EXTRA_ON_LOAN`.**
  - It enters "This week" only when the engine says `PAY_EXTRA` and there is spare money, within the existing 3-action limit and ranking.
  - The email renders it with the same copy keys.
- **Highlights (C28):**
  - the high-debt-load note
  - "This card costs 36% a year. Clear it first."
  - the never-pays-off warning
  - Limits as before (Home ≤ 1).
- **Calendar and reminders:**
  - the next EMI date (when `emi_day` is set) and the payoff month go into the existing calendar
  - an optional "EMI due in 3 days" reminder through the existing reminder settings (default off)
  - for floating rates, a yearly "check your loan rate" reminder (default off)

### 4. API (household roles: viewers read-only)
- `GET /v1/loans`: the position summary per loan plus household totals (remaining, monthly EMIs, debt load, the one-sentence pay-extra-or-invest answer).
- `GET /v1/loans/{id}`:
  - the position
  - the next 12 scheduled payments
  - the yearly balance series
  - events
  - the pay-extra-or-invest reasoning for this loan
- `POST /v1/loans/{id}/events` and `DELETE /v1/loans/{id}/events/{event_id}`.
- `POST /v1/loans/{id}/prepay-preview` with `{amount}`: the two options from §2. Nothing is saved.
- OpenAPI refreshed, and the PWA client regenerated.

### 5. PWA (UX rules; the new screen is a lazy-loaded route, because the initial JS budget has under 1.1 KB left)
- **Money:**
  - A "Loans" group with one list row per loan: icon tile, name/lender, a muted sub-line "Ends Mar 2031" (or the warning), the remaining amount (rounded), and a thin progress bar when % repaid is known.
  - The group header shows the total left and the total monthly payments.
- **Loan detail** (`/money/loan/:id`):
  - A hero: remaining amount (rounded), "Paid off in Mar 2031", and the % repaid.
  - A small inline-SVG balance line (no chart library).
  - One primary action: **Record a payment**. A sheet offers "Extra payment", "New interest rate", "New monthly payment", "Balance from a statement" and "Missed a payment".
  - A secondary **"What if I pay extra?"** sheet: an amount field, then two cards ("Finish 14 months sooner, save ₹1.2 L interest" / "Pay ₹2,300 less each month, save ₹48,000"). The amounts come from the API.
  - A history list of events, with delete for owner/partner.
  - Details (collapsed): rate, EMI, rate type, tax relief with "please check", and the next 12 payments.
- **Plan:** a collapsed "Loans" details section with the total left, monthly payments, debt load and the one-sentence answer.
- **Add loan form:** keep the required fields short (type, outstanding, rate, monthly payment). Everything new from §1 goes under optional details.
- **Wording:**
  - "monthly payment (EMI)" on first use with `<Term>`
  - "repayment schedule", never "amortisation"
  - no banned terms, ≤ 20 words per sentence

### 6. Tests (no network)
- **Engine:**
  - roll-forward matches a hand-computed schedule
  - a prepayment shortens the payoff
  - a rate change applies from its month
  - a balance update re-anchors
  - a missed EMI adds interest
  - never-pays-off detection
  - both prepay options against hand-computed figures
- **Decision:**
  - a credit card always comes first
  - nothing below the emergency fund
  - a home loan with India tax relief vs safe-money return, both ways
  - avalanche order
  - a DKK loan converted for net worth
- **DB:**
  - household RLS: a partner writes, a viewer can't, a foreign household gets 404
  - the audit trigger
  - the deferrable FK guard test still passes
  - the merge of a household with loan events
- **API contract** and the `PAY_EXTRA_ON_LOAN` action in email and PWA.
- **PWA:** the Loans group, the detail screen, the record-payment flow, the prepay preview, viewer read-only. The banned-terms and one-primary tests pass.

## Acceptance
1. Python: full `pytest` and `ruff` (`--no-cache`) pass; alembic upgrade/downgrade/upgrade on the test DB; OpenAPI and JSON Schemas refreshed.
2. PWA: `gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, full `npm test`, `build`, `budget` pass. The initial JS stays within 170 KB; the loan detail is its own chunk.
3. The result file lists every tax-relief rule with its source and `verified: false`, and gives one worked example: a ₹30 L home loan at 8.5% with ₹2 L extra, both options.
