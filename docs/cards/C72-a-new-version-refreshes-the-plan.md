# C72: a new version refreshes the plan

**Source:** the reviewer, 2026-11-14. After C70 and its review fix went live, the owner's plan still showed the old share reasons.

`_compute_if_stale` in `src/wealth/api/personalisation.py` recomputes the plan on every read. It then returns the **stored** plan whenever:
- the stored plan is for this week;
- `inputs_digest` matches;
- `actions` match.

The digest covers the user's data and market inputs, not the code. So any policy change that leaves the actions unchanged stays invisible until the owner's data changes, sometimes for days. Holding verdicts, reasons, checks, highlights and notes are all affected. The live plan kept a C70-era copy, and the `inputs_digest` stayed `541b7087…` across the deploy.

## Read
- `AGENTS.md`.
- `src/wealth/api/personalisation.py` (`_compute_if_stale`) and `src/wealth/policy/store.py` (`save_weekly_plan`, which keeps action ids and statuses for unchanged actions).
- `src/wealth/policy/personalise.py` (`WeeklyPlan`, `_digest_inputs`).
- Existing tests that read or rebuild plans: `tests/db/test_personalisation.py`, `tests/db/test_weekly_job.py`, `tests/db/test_shared_households.py`, `tests/db/test_money_map_api.py`.

## Scope
1. **Return the stored plan only if nothing changed.** It must be the same as the computed plan in every field, compared as `model_dump(mode="json")`. Otherwise save the computed plan and return it with `fresh=True`, as for any other change.
   - Viewers still get the computed plan without a write, as today.
   - Action ids and statuses must survive: `save_weekly_plan` already matches unchanged actions, so a re-save with identical actions keeps every id and status (Done, recorded buys).
2. **No churn.** Check that a plan computed twice from the same inputs is byte-for-byte identical. Look for timestamps, `now()`, random ids or unordered sets inside `WeeklyPlan`. If something differs between two computations, make it deterministic, or leave that one field out of the comparison with a comment saying why. Otherwise every read would write.
3. Leave `inputs_digest` as it is: no code version in it. It keeps its current meaning.

## Tests (`@pytest.mark.db` where they need the database)
- **Reason changes:** the stored plan for this week has the same digest and actions as the computed one, but a holding review's `reasons` differ (simulate an older stored copy). The read returns the computed reasons with `fresh=True`, and stores them.
- **Nothing changed:** two reads with unchanged inputs both return `fresh=False` on the second read, and the second read writes nothing. Assert the stored row's `created_at` is unchanged, or count statements.
- **Ids and statuses:** after a re-save caused by a reason change, every action id and status (one marked done) is unchanged.
- **Viewers:** a viewer read never writes.

## Out of scope
- Policy rules.
- The PWA.
- Weekly jobs, except where their tests need the new behaviour.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
