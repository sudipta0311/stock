# C39: The weekly pack keeps every approved idea of the week

**Reported by the product owner (2026-11-01):** "in ideas only indian stocks are shown". The reviewer traced it to how the weekly pack is written.
**Builds on:** C16a/C16b (research graph, pack), the 2026-10-31 fixes (`alternate_markets`, `keep_unresearched_markets`), C35 (`entry_hint`) and C37 (`why`, `valuation_label`). Read `AGENTS.md` first.

## What happens
- `research_packs` holds one row per week. `pack_node` in `wealth.research.graph` builds `ideas` only from **this run's** approved dossiers, then overwrites the row.
- `keep_unresearched_markets` keeps a market's earlier ideas only when this run researched **no** company in that market.
- Production week 2026-W44 shows the result:
  - The `dossiers` table holds 5 APPROVED India and 5 APPROVED US dossiers from Saturday's runs.
  - The pack shows only the 5 India ideas, because a later India-only run overwrote the row before the market fix landed.
- A manual run started today screened again. 3 shortlisted companies are still `SCREENED`; the rest of the week's top 20 are already decided. Continuing it would replace each researched market's ideas with at most these 3 companies' approvals. The reviewer has left that run paused at the screen step until this card lands.

## Scope

### 1. Build the pack from the week's decided dossiers, not one run's
- Load candidate data for every dossier of the week whose stage is `APPROVED`, from any run. Generalize `wealth.research.run.load_shortlist_candidates` (today it accepts only stage `SCREENED`) so it can load the week's APPROVED dossiers with their screener snapshot: price, sector, decile, composite and the stored `dist_from_200dma`. Don't write a second loader.
- In `pack_node`, the ideas are this run's approved dossiers **plus** the week's other APPROVED dossiers. Build all of them through the same code path as today: `SatelliteIdea`, `pack_idea_why`, and `entry_hint` with this run's index percentiles.
- **One idea per dossier id** (`<week>/<market>/<symbol>`). The `dossiers` row holds the latest verdict, so a company re-researched and vetoed in a later run drops out.
- `idea_outcomes` follows the same rule: this run's outcomes, plus the week's other decided dossiers (APPROVED, VETOED, UNRESOLVED) that this run didn't touch.
- **Order:** by market (IN, then US), then by screener `composite`, highest first. The Ideas page shows the top 3 per market, so order matters.
- `markets.<M>.idea_count` counts the merged ideas.
- Remove `keep_unresearched_markets` if it becomes redundant. Otherwise keep it only for the fields the table can't rebuild: `markets` status, `summary`, `cio_agenda`, `macro`. Keep its existing tests passing or replace them with equivalent ones.

### 2. A no-model `repack` command
- `python -m wealth.research.run repack --week <week>` rebuilds `ideas`, `idea_outcomes` and `markets` from the week's decided dossiers, with the same builder as section 1.
- It keeps the existing pack's `summary`, `cio_agenda`, `macro` and `run_id`.
- It makes **no** model calls, spends no budget and doesn't touch `research_runs`.
- It refreshes `index_valuation_percentiles` and the hints from stored data, as `pack_node` does.
- It prints a JSON line with the idea count per market. If the week has no pack row and no approved dossiers, it exits non-zero with a clear message.
- Add `repack` to the `step` choices in `.github/workflows/research-weekly.yml`: a step that runs only on `workflow_dispatch` with `inputs.step == 'repack'`. It doesn't run the watchers, and start and continue don't run on repack. Update any deploy-config test that asserts the choices.

## Tests (DB tests marked `@pytest.mark.db`; no network or LLM: a fake router that fails if called)
- A run that approves one new IN company keeps the week's earlier IN and US approvals, with no duplicates.
- A company APPROVED earlier and VETOED in this run is not in the ideas, and its outcome shows VETOED.
- The ideas are ordered by market, then by composite.
- `repack`:
  - rebuilds the ideas from APPROVED dossiers and keeps `summary` and `macro`
  - makes no model calls
  - exits non-zero for an empty week
- Every merged idea carries `why` and `entry_hint`.
- The workflow `step` choices include `repack`, and the repack step's `if` excludes start and continue.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), ruff check and format (`--no-cache`), `python -m scripts.check_deploy_config`
- Write `docs/cards/C39.result.md`.

## After acceptance (reviewer)
- Push, then dispatch `repack` for 2026-W44 so the 5 US ideas come back.
- Then dispatch `continue` for the paused run so its 3 companies join the week's ideas.

## Notes
- No migration, no new dependencies, and no change to the policy, `SatelliteIdea` fields, screener or agents.
