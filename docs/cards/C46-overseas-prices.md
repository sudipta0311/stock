# C46: Overseas prices that are fetched, counted and kept

**Found by the C44 health check on its first production run (2026-11-01):** "No overseas share prices are stored." The overseas job (`marketdata-abroad`) is green every day, but:

1. **The daily clean-up deletes them.** `store.prune` (run by `marketdata-daily --prune`) keeps prices only for held securities, index proxies, current index members and Indian core-pick candidates (`core_candidate_ids`). The 11 curated UCITS funds in `rules/eu_etfs.yaml` are none of these. When the daily job runs after the overseas job (cron is often hours late), their prices are deleted.
2. **Most requests fail silently.** The 2026-10-30 run stored 2 of 11 fund prices and reported `succeeded` with no errors. Every exception is swallowed in `_fetch_abroad_etfs` / `_fetch_abroad_us_stocks`. Three failure causes:
   - **No pacing.** Back-to-back calls hit Alpha Vantage's free-tier limits, and the API answers with an `Information`/`Note` body.
   - **Listings the code can't price.** `_upsert_curated_security` raises for a fund with no USD London listing (e.g. GBP-only London).
   - **Missing exchange suffix.** It sends a bare symbol for Xetra/Euronext listings, but Alpha Vantage needs an exchange suffix.

Read `AGENTS.md` first. No PWA change and no migration.

## What to do

### 1. Keep the prices we need
`store.prune` must also keep prices (and their history) for:
- every curated fund in `rules/eu_etfs.yaml`, matched by ISIN to `securities`
- every security referenced by a recorded buy (C45) that is still `waiting`

Keep the existing keep-sets. Add a test showing that a curated fund's and a recorded buy's prices survive `prune` while an unrelated security's are removed.

### 2. Fetch every curated fund, politely
- **Pace the requests.** Wait between Alpha Vantage calls, as a named constant with a comment citing the free-tier limit. On a rate-limit answer (`Information`/`Note`), wait and retry that symbol once.
  - The per-day cap stays `_MAX_ABROAD_QUOTES_PER_DAY`.
  - The job's `timeout-minutes` must still fit (11 funds plus the US held stocks).
- **Choose a listing Alpha Vantage can price,** and that the app can convert with the FX pairs it stores (INR rates for DKK, USD, EUR):
  - prefer London USD (`.LON`)
  - else Xetra EUR (`.DEX`)
  - else skip the fund and record why
  - Put the suffix mapping in one place, with a comment linking Alpha Vantage's symbol-format documentation.
  - Never price in GBP or GBX, which the app can't convert.
- **Don't swallow errors.** Count per reason: `rate_limited`, `no_supported_listing`, `unknown_symbol`, `no_series`, `http_error`.
  - Put the counts in the run's `counts` (e.g. `eu_etfs: {prices_eod, requested, skipped: {...}}`).
  - Add an `errors` entry when anything was skipped, so the run reports `partial`, not `succeeded`.
  - Error text must never contain the API key or the request URL.
- **The workflow's step** shows a `::warning::` when the run is partial. Follow how `marketdata-weekly.yml` reads the JSON status.

### 3. Monthly fund history: no long downloads inside an open transaction
On 2026-11-02, `marketdata-monthly` (`--only amfi_history`) failed after 12½ minutes with `PendingRollbackError: Can't reconnect until invalid transaction is rolled back`.
- **Cause:** `run_refresh` keeps one session and transaction open while `_fetch_missing_amfi_history` downloads every missing window. The database dropped the idle connection, and the final `session.flush()` of the run row failed. The run row was never saved, so the health check could not see the failure.
- **Fix:**
  - Download first, outside any open transaction. Or store each window in its own short transaction as it arrives.
  - Save the run row in a fresh transaction at the end.
  - If a store step fails, roll back and still record the run as `partial`/`failed` with the error class name.
  - Create the engine with `pool_pre_ping=True` for this command.
  - Apply the same rule to any other source in `run_refresh` that downloads for minutes. Note in the result file which ones you checked.
- **Test:** simulate a connection that is invalidated during a fetch, and check that the run row is still stored with status `failed` or `partial`. Use a fake connection error; no real database drop.

### 4. Check the health check
`abroad_prices` in `wealth.alerts.health` should pass once prices are kept. Add a DB test:
1. seed a curated fund price
2. run `prune`
3. the health loader still sees it

## Tests
- Recorded Alpha Vantage responses only (fixtures or `respx`/monkeypatch); no network.
- **Listing choice:** London USD, Xetra fallback, skip with a reason.
- **Pacing:** use a fake sleep; don't sleep for real in tests.
- **Retry:** one retry after a rate-limit body, then counted as `rate_limited`.
- **Results:** a partial run reports `partial` with counts; the API key appears nowhere in the output.
- **Prune:** keeps curated and recorded-buy prices.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache`, `.venv\Scripts\python -m scripts.check_deploy_config`
- Write `docs/cards/C46.result.md`. List which curated funds now get a price, which are skipped and why. Work this out from the listing data in `rules/eu_etfs.yaml`, without calling the network.

## Notes
- No new dependencies. Don't change which funds the plan picks.
