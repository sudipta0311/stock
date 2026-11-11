# C62: the fund price check skips market holidays

**Source:** "Daily data health" failed on 2026-11-11 (commit 6b6dd2e) and emailed the owner: "Indian fund prices stored fewer than half the usual rows."

**Facts:** `latest_date` 2026-11-10, `latest_rows` 121, `previous_median_rows` 260.

**Cause:** 2026-11-10 was an NSE holiday (Diwali Balipratipada). NSE has no share-price file for that day (the fallback returns 2026-11-09 closes), and India share prices correctly end on 2026-11-09. On a market holiday only some funds (liquid, overnight, some debt) publish a NAV, so the few NAVs dated that day are normal, not a failed load. Weekends behave the same way.

## Read
- `AGENTS.md`.
- `src/wealth/alerts/health.py`: `_check_price`, `_date_row_counts`, how `india_fund_prices` is loaded, and the `india_fund_prices` check.
- `tests/alerts/test_health.py`.

## Scope
1. **Count fund prices only on NSE trading days.**
   - **Which days count:** for `india_fund_prices` only, the low-volume test ("fewer than half the usual rows") uses NAV dates on which India share prices are stored.
   - **What the check compares:** the latest such date against the median of the 10 such dates before it.
   - **The share-price rows are the trading calendar.** Do not add a holiday list or a table.
   - **Enough history:** load enough NAV dates to find 11 trading-day dates (for example the last 40 dates), and keep the query read-only.
2. **Staleness is unchanged.** "N trading days old" still uses the newest NAV date of any kind.
3. **Explain which date was used.** When the latest NAV date is not a trading day, the check's `facts` say which date the row count used. Add one field, for example `volume_date`.
4. **No other change.** Every other price check works exactly as before.

## Tests
- **Holiday:** the latest NAV date has few rows and no share prices, and the previous trading days are normal → ok.
- **Real shortfall:** a trading day with fewer than half the usual NAV rows → still a problem with the same message.
- **Weekend:** NAV rows dated on weekends with few rows do not lower the median or trigger the check.
- **Staleness:** a stale case is still reported as before.
- All synthetic, with no network.

## Out of scope
- Workflow files.
- Share price, overseas and FX checks.
- Market data loading.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
