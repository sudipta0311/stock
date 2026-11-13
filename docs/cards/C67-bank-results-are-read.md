# C67: bank results are read

**Source:** the reviewer, 2026-11-12, after a look-only read of the owner's live holding reviews. For the three banks the owner holds, "Company results", "Price level" and "Company strength" all say "Not enough information". Only 3 of 6 checks run, so "Keep" means much less for them than for other shares.

**Likely cause:** Indian banks file quarterly results in NSE's **banking** XBRL format. The parser in `src/wealth/marketdata/sources/nse_fundamentals.py` (`_PERIOD_METRICS`) only knows the element names used by ordinary companies (`RevenueFromOperations`, `ProfitLossForPeriod`, `DilutedEarningsPerShare...`). As a result, a bank's filing gives no revenue, no profit and no EPS:
- no revenue means no "company results" check;
- no EPS means no P/E, and so no "price level" check.

## Read
- `AGENTS.md`.
- `src/wealth/marketdata/sources/nse_fundamentals.py`.
- `tests/marketdata/test_fundamental_sources.py` (the NSE tests) and `tests/fixtures/marketdata/nse_xbrl_excerpt.xml`.
- `src/wealth/research/stock_checks.py` (`revenue_yoy_percentages`): to see which metric names the checks use.

## Scope
1. **Confirm the cause.** Find the element names that NSE banking-format result filings use for:
   - total income (or interest earned plus other income);
   - net profit for the period;
   - diluted EPS.

   You may read one public NSE banking XBRL result file over the network, only to learn the element names. If NSE refuses automated access, use NSE's published banking taxonomy. Do not commit any downloaded file. Record what you used, and how sure you are, in the result file.
2. **Map them.** Extend the tag lists so a bank filing yields:
   - `revenue`: the bank's **total income** for the period. If there is no single total element, use interest earned plus other income, and only when both are present in the same unit.
   - `net_income` and `eps_diluted`.

   Ordinary companies' tags keep their current priority. Do not invent an `operating_income` for banks (leave it missing).
3. **Nothing else changes:** no change to checks, rules, thresholds, the API or the PWA. Once the next weekly data run stores bank facts, the existing checks will start to work for banks.

## Tests (no network; synthetic fixture)
- Add a small synthetic banking-format XBRL fixture next to the existing one: a made-up bank, quarter and amounts, using the confirmed element names. A parser test reads `revenue`, `net_income` and `eps_diluted` from it.
- Also test the "interest earned + other income" fallback, and that a unit mismatch gives no revenue.
- The existing NSE parser tests still pass unchanged.

## Out of scope
- Insurance and NBFC formats, unless they turn out to use exactly the same elements.
- Re-running the weekly data job, and any database migration.
- Company-strength ranking rules.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
