# C38: Skip NSE filings whose XBRL file is gone

**Found by the reviewer:** every weekly `nse_fundamentals` run ends `partial` with `HTTPStatusError` for ADANIENT, LT and UPL, so those three companies never get India fundamentals.
**Builds on:** C27 (NSE integrated filings). Read `AGENTS.md` first.

## What happens
- NSE's filings list for each of the three symbols includes one or two filings whose XBRL file returns **404** on `nsearchives.nseindia.com`. A live probe on 2026-11-01:

  | Symbol | Recent filings | XBRL 404 | Notes |
  |---|---|---|---|
  | LT | 10 | 1 | the 404 is a second FY2025 Q4 filing |
  | UPL | 8 | 1 | the same pattern |
  | ADANIENT | 9 | 2 | each 404 has a readable filing for the same period end |
  | TCS (control) | 7 | 0 | |

- `fetch_symbol_fundamentals` calls `client.fetch_xbrl` in a loop. The first 404 raises, so the readable filings for that symbol are thrown away too.
- Successful symbols only fetch filings newer than their last stored `filed_at`. These three never store anything, so each run retries every filing and fails on the same dead file.

## Scope
1. In `wealth.marketdata.sources.nse_fundamentals.fetch_symbol_fundamentals`:
   - when `fetch_xbrl` raises `httpx.HTTPStatusError` with status **404 or 410**, skip that filing and continue with the rest
   - any other status, network error or parse error still raises, as today
2. Make the skip visible without failing the symbol:
   - return the skipped count alongside the observations (a small result type, or a separate list, whichever fits the code best)
   - `refresh._fetch_nse_fundamentals` adds a count `nse_fundamentals.xbrl_missing` to the run counts
   - a symbol whose filings were all skipped records the error `XbrlMissing`, so it still shows up in `errors`
   - a symbol with at least one readable filing is a success
3. Leave the rest of the refresh logic as is: the `after` filter, the request interval and the archive-host check.

## Tests (no network; use `httpx.MockTransport` like the existing NSE tests)
- One filing returns 404 and the others are readable: observations come from the readable ones, and the skipped count is 1.
- Every filing returns 404: the symbol records `XbrlMissing`.
- A 500 or a timeout still raises, so the symbol records the error.
- A 410 behaves like a 404.
- In the refresh: the `xbrl_missing` count appears in the run counts.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache` on `src tests`
- Write `docs/cards/C38.result.md`.

## Notes
- No new dependencies. No schema change.
- Don't call NSE in tests.
