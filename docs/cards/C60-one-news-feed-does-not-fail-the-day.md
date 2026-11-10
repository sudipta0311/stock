# C60: one news feed hiccup does not fail the daily data job

**Source:** the owner, 2026-11-09: "I am getting regular email from github about job failures". After C59 fixed the NSE list, the next "Daily market data" run (2026-11-09 23:02 UTC, commit 642b849) stored the master (3,005 rows) and all prices. It still ended `status: partial` with `errors: {"feeds:pib": "ParseError"}` and exit code 2, so GitHub emailed a failure.

## Why
`src/wealth/marketdata/refresh.py` exits 2 whenever `result["errors"]` is not empty (around line 1813). The news feeds (`federal_register`, `pib`, `rbi`, `ustr`, in `sources/macro_geo.py`) are best-effort background for research. One feed returning a page that is not RSS for a day should be recorded, not emailed. A feed that stays broken should still be noticed.

## Read
- `AGENTS.md`.
- `src/wealth/marketdata/refresh.py` (how `errors`, `status` and the exit code are built; the `MarketdataRun` record).
- `src/wealth/marketdata/sources/macro_geo.py`.
- `src/wealth/alerts/health.py` (how "feeds" and other sources are judged, and what the daily data-health job reports).
- Their tests.

## Scope
1. **Feed failures are warnings, unless every feed fails.**
   - A `feeds:<name>` failure goes into a new `warnings` map in the run result and the `MarketdataRun` counts, not into `errors`.
   - It does not change `status` or the exit code.
   - If every feed fails in a run, keep today's behaviour: an error, `partial`, and exit 2.
2. **A feed that keeps failing is noticed.** When the same feed has failed in 3 consecutive daily runs, the data-health check reports it in its usual way, as a stale or failing source with the feed's name.
   - Use the stored run records; do not add a table.
   - Follow how health already reports other sources. If health already exits non-zero for stale sources, this alert does the same.
3. **A clearer reason when a feed is not RSS or Atom.** When a feed returns HTML or another non-XML body, record `"NotAFeed"` instead of a bare `"ParseError"`. Keep the real parse errors as they are.

## Tests
- Refresh: one feed fails while others succeed → `status` is unchanged, `warnings` holds the feed, and the exit code is 0. All feeds fail → exit code 2.
- Health: three consecutive runs with the same feed warning → reported; two → not reported.
- Parser: an HTML body → `NotAFeed`.
- All with synthetic payloads and no network.

## Out of scope
- Workflow files.
- Other sources' error policy.
- Changing feed URLs.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
