# C59: the daily NSE share list survives ISIN changes

**Source:** the owner, 2026-11-09: "I am getting regular email from github about job failures". The reviewer checked the last 300 workflow runs. Only "Daily market data" fails now: on 2026-11-05 and 2026-11-06, both scheduled runs ended `status: partial` with `errors: {"master": "IntegrityError"}`. Every other source in those runs succeeded. The job exits with code 2 on `partial`, so GitHub emails a failure.

## Why
`securities` has two unique keys: `uq_securities_isin` (isin) and `uq_securities_market_symbol` (market, symbol, where symbol is not null). `store.upsert_nse_master` → `_upsert_security(..., by_isin=True)` inserts with `ON CONFLICT (isin)`.

When NSE issues a new ISIN for an existing symbol (common after a split or a face-value change), the insert does not conflict on ISIN. It conflicts on (market, symbol), which is not the conflict target, so Postgres raises `IntegrityError`.
- One such record aborts the whole `master` source, so new listings and changes have not been stored since 2026-11-05.
- The summary records only the exception class, so nobody can see which row failed.

## Read
- `AGENTS.md`.
- `src/wealth/marketdata/store.py` (`upsert_nse_master`, `_upsert_security`, `_with_security_conflict`) and `src/wealth/marketdata/refresh.py` (how a source's error becomes `errors[...]` and `status`).
- `src/wealth/db/models.py` (`Security`).
- Existing tests under `tests/marketdata/` and `tests/db/` for the NSE master.

## Scope
1. **ISIN change for the same listing.** An NSE record's (IN, symbol) may already exist with a different ISIN, while the old ISIN is no longer in today's NSE list. Treat that as the same listing with a new ISIN:
   - update the existing row's `isin` (and the other refreshed fields) in place;
   - keep the security id, so holdings, prices and history stay attached.
2. **Symbol swaps or renames inside one batch.** The old ISIN may still be in today's list under another symbol. Order or stage the updates so the batch never trips a unique key mid-way, for example by clearing the moving rows' symbols first, within the same transaction.
3. **One bad row must not stop the whole list.**
   - Wrap each record that still conflicts in a savepoint. Skip it, keep going, and report it.
   - `errors["master"]` should read like `"1 of 2,345 rows skipped: ABC (uq_securities_market_symbol)"`. NSE symbols and constraint names are public market data, so they are fine in logs.
   - A run with skipped rows stays `partial`, so real surprises still alert.
4. **Clearer errors for every source.** Where a source fails with a database error, include the constraint name from the driver's diagnostics next to the class name. Never include connection details or row values other than public symbols.
5. List under "Decisions / open issues": an old statement that still carries the old ISIN will create a separate security row on import. There is no ISIN alias table, and adding one is a later card.

## Tests
DB tests with synthetic securities (invented symbols and ISIN-shaped codes that are not real):
- same symbol, new ISIN: the row is updated in place, its id is kept, and an attached price row still points to it;
- two symbols swap within one batch: both end correct, with no error;
- a conflict that cannot be resolved: that row is skipped, the others are saved, and the error text names the symbol and the constraint;
- a unit test for the error text of a database failure (class + constraint).

## Out of scope
- An ISIN alias table, or a new migration.
- Other sources' logic.
- Workflow files.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
