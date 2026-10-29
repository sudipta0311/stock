# C34: Savings in kroner or euro, plus fixes from the mock-data walkthrough

**Requested by the product owner:** "yes pls complete all" (after the reviewer's mock-data walkthrough on the phone).
**Builds on:** C07 Money Map, C26a/C27 residency and households, C30 loans (the loan currency pattern), C23 push, C32 device handling, C33 Gmail statements. Read `AGENTS.md` first.

A family in Denmark keeps most of its cash in Danish bank accounts. Today a savings account can only hold rupees, so a DKK balance can't be recorded honestly. The other items are small bugs found on the phone.

## Scope

### 1. Savings accounts in INR, DKK or EUR
- **Schema:** `SavingsAsset` gets `currency: Literal["INR", "DKK", "EUR"] = "INR"`, and `balance` becomes `CurrencyAmount` (the loan pattern in `LoanAsset`).
  - Existing rows have no `currency` key and stay INR. No migration is needed: details are JSON.
  - Regenerate the JSON Schemas, OpenAPI and `apps/web/src/money/forms.generated.ts` with the existing scripts.
- **Valuation:** the native balance is converted to INR at the latest `FxRate` `"<CUR>/INR"` row not older than 10 days, exactly like loans in `refresh_networth_snapshot`.
  - With no usable rate, the account goes into `unvalued_assets` with reason `fx_needed`, the same as a loan. It never counts as rupees.
  - Find **every** place a savings valuation is computed or stored (asset create/update, net-worth refresh, household snapshot, plan, foundation/emergency-fund check, fund check, stress tests, weekly brief, export). A DKK balance must reach all of them as its INR value. Put the conversion in one helper; don't copy it.
  - Decimal only. Round only where the existing code already rounds.
- **API responses** for a savings asset keep the native `balance` and `currency` in `details`, so the PWA can show "DKK 85,000".
- **PWA form** (`/money/add/savings` and edit):
  - a currency select right after Bank, before the balance (the loan form already does this; reuse it, don't copy it)
  - **smart default:** DKK when the selected member lives in Denmark (`residency.country == "DK"`), EUR for another euro country if the data has one, otherwise INR
  - the balance input shows the right prefix (₹, kr, €) and uses the currency-aware amount parsing in `form-utils.ts` (`normalizeAmount`)
- **Money Map list and Home:** a DKK account shows its native amount ("DKK 85,000", using `formatDkk`) with the household-currency amount as the quieter second line, as loans do. Totals use the converted value.
- **Quick add:** "Danske Bank savings 85000 kr" (or "DKK 85000", "85.000 kr.") drafts a savings account in DKK. Rupee phrases keep working. Unit tests for both.

### 2. Small fixes from the walkthrough
1. **Money Map hint names a category the person doesn't have.** `MoneyMapPage.tsx` builds "add EPF to finish" from the first checklist tile not done, even for a Denmark household that never ticked EPF. Pick the next category from `networth.completeness.ticked` minus `done`, in checklist order. If nothing is ticked, show only the percentage line.
2. **Indian funds card says "One change suggested" with no funds.** Reproduce it with a household whose members have no Indian fund holdings (for example one Denmark-resident member and one India-resident member, savings only).
   - With no fund holdings, `wealth.policy.fund_check` must return `status="fine"`, `summary_key="fund_check.upload"` and no checks or actions.
   - Find which check fires today (likely a residency or debt note) and attach it only when there are funds.
   - Add a policy test and a PWA test.
3. **Wrong-password message looks like help text.** In `GmailStatementInbox.tsx` (and the manual import password step, if it has the same issue), a wrong password must use the app's error style (`text-danger` or the existing error component) with `role="alert"`, not grey muted text.
4. **Required selects have no required marker.** The gold Purity select (required for physical gold through `field-rules.ts`) shows no star, while required text inputs do. Every required select, radio group and month field must show the same marker as a required input. Add a test with physical gold.
5. **Quick add puts "500000.00" in the amount box.** When a draft prefills an INR amount field, show it the way a person types it: "5,00,000" (Indian grouping, no ".00"; keep paise when non-zero, e.g. "1,25,000.50"). DKK/EUR use "85,000". Saving must send the same value as before. Use the helpers in `form-utils.ts` and `wealth`-matching formatters in the PWA; don't add a new number library.
6. **Push notifications follow the wrong person on a shared phone.** `claimDeviceFor` (`apps/web/src/auth/device-account.ts`) keeps the push subscription when a *different* account signs in. The previous person's reminders would then show on the new person's phone.
   - On an account switch, unsubscribe the browser's push subscription (`registration.pushManager.getSubscription()` → `unsubscribe()`) and remove the stored push subscription id.
   - The server already drops a subscription when a send returns 404/410; leave that as it is.
   - The new person can turn notifications on again in Settings.
   - Normal sign-out (same person) keeps today's behaviour.
   - Extend `tests/auth/DeviceAccount.test.ts`: a switch unsubscribes and clears the id; the same account keeps it; a missing service worker or Push API doesn't break sign-in.
7. **"Check today's rate" is vague.** The `missing_yield` highlight says "A current rate is not available for this choice." without naming the choice.
   - `wealth.policy.highlights` passes `amount` and `rate_pct` today. Add an optional instrument code from the unranked option (e.g. `po_td`), stored in the highlight context. The PWA maps it to a plain name from copy; the backend never sends free text.
   - Body: "We don't have today's rate for {name}. Check it with the bank or post office first." The title stays "Check today's rate".
   - The dismissal fingerprint must stay stable for existing dismissed highlights. Add a test.
   - Without a name, keep a generic body that still says what to do: "We don't have today's rate for this option. Check it with the bank or post office first."
   - Keep within the copy rules (≤ 20 words per sentence, no banned terms).
8. **Onboarding counter jumps.** The step counter shows "2 of 6" and then "6 of 6" when optional steps are skipped. Count only the steps a person can actually see in their path, or drop the number and keep the progress bar. Pick one and keep the label and `aria` text consistent. Test the skip path.

## Tests (no network)
- **Python:**
  - a DKK savings account (DKK 10,000 at DKK/INR 12.50) counts as INR 1,25,000 in net worth, the cash sleeve, the emergency-fund check, the household snapshot and the plan
  - with no rate (or one older than 10 days) it is listed as `fx_needed` and adds nothing
  - an INR account without `currency` behaves exactly as before
  - quick-add parsing for kroner phrases
  - the fund-check "no funds" case
- **PWA:**
  - the savings form defaults to DKK for a Denmark member and posts `{currency: "DKK", balance: "85000"}` from "85,000"
  - the list row shows "DKK 85,000"
  - one test for each small fix above

## Acceptance
1. Python: full `pytest`; `ruff check` and `ruff format --check` with `--no-cache` on `src tests scripts`; JSON Schemas and OpenAPI exported and committed; `scripts/check_deploy_config.py` passes.
2. PWA: `gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, full `npm test`, `build`, `budget` (≤ 170 KB initial).
3. No migration unless you find you need one; if you do, it is `0021_…` revising `0020_statement_inbox`.
