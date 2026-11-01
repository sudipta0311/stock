# C44: A daily data health check that turns red when a job stored nothing

**Why:** twice a scheduled job finished green but stored nothing: the daily India prices stored 0 rows for days, and the first research run shortlisted 0 ideas. GitHub cron also drops or delays runs by hours. A green tick is not proof. Add one daily check that reads what is actually in the database. It fails loudly when data is missing, old or empty, so GitHub emails the owner and the reviewer can see what broke.

Read `AGENTS.md` first. This card changes no money logic, no policy, no research and no PWA screen.

## What to build

### 1. `python -m wealth.alerts.health`
- New module `src/wealth/alerts/health.py`. Pure evaluation functions take plain facts plus `now`. A thin loader reads those facts from the database.
- Database URL: `OWNER_DATABASE_URL`, falling back to `TEST_DATABASE_URL`, the same way `wealth.marketdata.refresh` does. Exit code 2 with a short message if neither is set.
- **Read-only.** Run every query inside a transaction set `READ ONLY`, and never write.
- Options: `--now <ISO datetime, UTC>` for tests and reruns (default: the current UTC time).
- Output: one JSON line on stdout: `{"ok": bool, "checked_at": ..., "checks": [{"key", "level", "message", "facts"}]}`.
  - `level` is `ok`, `warning` or `problem`.
  - `message` is one plain sentence, ≤ 20 words, saying what is wrong and which workflow to rerun, e.g. "India share prices are 3 working days old. Rerun Daily market data."
  - `facts` holds dates, counts, sources and ISO weeks only.
- When `GITHUB_ACTIONS=true`:
  - print `::error title=<key>::<message>` for each problem and `::warning title=<key>::<message>` for each warning
  - append a short Markdown table (check, result, message) to `$GITHUB_STEP_SUMMARY` when that variable is set
- Exit 1 if any check is a `problem`, else 0. Warnings never fail the job.
- **Never print** the database URL, environment values, emails, names, user ids, amounts or symbols tied to a user. Counts of households or users are fine.

### 2. The checks
Keep each threshold as a named constant with a one-line comment saying why. Reuse an existing NSE trading-session or holiday helper if the repo has one. Otherwise count weekdays and accept the occasional false alarm on back-to-back holidays.

| Key | Problem when | Notes |
|---|---|---|
| `india_prices` | The newest NSE close in `prices_eod` is more than 2 trading days older than the last trading day before `now`. Also a problem when the newest date has fewer than half the median row count of the 10 previous stored dates. | Daily job runs 13:30 UTC Mon–Fri. |
| `india_fund_prices` | Same rule for `nav_daily` (AMFI). | |
| `abroad_prices` | Same rule for the US and EU prices that `marketdata-abroad` stores. | Weekdays; 18:00 UTC job. |
| `fx` | DKK/INR, USD/INR or EUR/INR has no rate in the last 4 days. | Name the pair in the message. |
| `weekly_data` | From Sunday on, any of `universes`, `nse_history`, `edgar`, `us_index_pe`, `nse_fundamentals` has no `succeeded`/`partial` run in `marketdata_runs` within the last 8 days, or its latest run stored zero rows. | Read each source's real `counts` keys. If zero rows is normal for a source, exclude it with a comment saying why. |
| `failed_runs` | Any `marketdata_runs` row from the last 2 days has status `failed`. A `partial` run is a **warning**. | List the sources. |
| `research_pack` | From Sunday 00:00 UTC to the next Friday, there is no `research_packs` row for the ISO week of the most recent Saturday. A **warning** (not a problem) when that pack has no India idea or no US idea; name the market. | Research starts Saturday 00:00 UTC and continues every 2 h until 20:00 UTC. |
| `weekly_plans` | On Monday's run, a household that finished setup has no `weekly_plans` row for the week that ended on Sunday. | Use the same "finished setup" rule as `wealth.policy.run_weekly`. The message gives a count only. |
| `weekly_email` | On Monday's run, a user who opted in to the weekly email has no weekly-brief row in `alerts_log` for that week. Skip this check, with an `ok` explaining why, when the email provider is not configured for sending. | Use the existing consent rules from `wealth.alerts`. Read how `alerts-weekly` records sends. |

Timing is deliberate. A check is evaluated only after its job's latest normal finish time plus several hours, because cron runs up to 5 h late. Before that it reports `ok` with `facts.not_due_until`.

### 3. Workflow `.github/workflows/data-health.yml`
- `schedule: "0 9 * * *"` (09:00 UTC, after every job's late window), plus `workflow_dispatch` with an optional `now` input.
- Same setup as the other jobs (checkout, Python 3.12, `pip install -e ".[api]"` or the smallest extra that works), `permissions: contents: read`, a `concurrency` group, `timeout-minutes: 10`.
- Env: `OWNER_DATABASE_URL` from secrets only.
- The job fails when the command exits 1, so GitHub's failed-workflow email reaches the owner.
- Update `scripts/check_deploy_config.py` and its tests if they list workflows or required secrets.

### 4. Docs
In `docs/DEPLOY.md`, add a short section "When the daily data check fails". Give one line per check key with the workflow to rerun, in plain words.

## Tests (`tests/alerts/test_health.py`, plus DB tests marked `db`)
- **Pure evaluation:**
  - each check's ok, warning and problem cases
  - the "not due yet" windows: Saturday before research ends, Sunday before and after 00:00, Monday for plans and email
  - a weekend
  - a single holiday
  - a source whose latest run stored zero rows
  - a pack with only India ideas → warning naming the US
- **DB test:** seed synthetic rows in a fresh test database, run the loader with a fixed `--now`, and check the JSON. Also check that the transaction is read-only (an attempted write inside it fails).
- **Output:**
  - the GitHub annotation and step-summary format
  - exit codes 0, 1 and 2
  - stdout and stderr never contain a fake database URL or password passed in the environment
- No network, no model calls.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), `ruff check` and `ruff format --check` with `--no-cache`, `.venv\Scripts\python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.alerts.health --now 2026-11-02T09:00:00+00:00` against the local test database, with `OWNER_DATABASE_URL` unset. Paste its JSON (no URLs) into the result file.
- Write `docs/cards/C44.result.md`.

## Notes
- No new dependencies, no migration, no change to existing jobs' behaviour.
- The reviewer will dispatch the workflow against production after review.
