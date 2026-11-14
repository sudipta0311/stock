# C68: research survives share symbols with "&"

**Source:** the reviewer, 2026-11-14. The scheduled weekly research run for 2026-W46 failed in `python -m wealth.research.run start` after about six minutes with `{"error": "ValidationError", "status": "failed"}`. The stock checks in the same workflow then ran fine. The later "continue" run found no open run, so this week has no new research. Weeks W44 and W45 ran cleanly.

**Likely cause (not yet confirmed):** a dossier ID is `f"{week}/{market}/{symbol}"`. Both the Pydantic `Dossier.dossier_id` pattern (`src/wealth/research/dossier.py`) and the database CHECK `ck_dossiers_id` (migration 0016) allow only `[A-Za-z0-9._-]` in the symbol part. Several NSE symbols contain `&`: `M&M`, `M&MFIN`, `J&KBANK`, `ARE&M`, `GMRP&UI`. When one of them reaches the screener's shortlist, `_save_pick` builds `Dossier(dossier_id="2026-W46/IN/M&M", ...)` and it raises `ValidationError`, which ends the whole run. C67 now reads bank results, so banks such as J&K Bank can now rank high enough to be shortlisted.

The run prints only the exception class, so nobody can tell where it failed.

## Read
- `AGENTS.md`.
- `src/wealth/research/dossier.py` (`Dossier`), `src/wealth/research/screener.py` (`run_screener`, `_save_pick`), `src/wealth/research/run.py` (`load_shortlist_candidates`, the `start` and `continue` error handling), `src/wealth/research/graph.py` (where the symbol is read back from a dossier ID).
- `src/wealth/db/migrations/versions/0016_research.py` (the `ck_dossiers_id` CHECK). Read it; **do not edit it**.

## Scope
1. **Safe dossier IDs without a migration.** Add one helper in `wealth.research.dossier`, for example `dossier_id_for(week, market, symbol)`, plus its inverse `symbol_from_dossier_id(dossier_id)`.
   - Characters in `[A-Za-z0-9.-]` stay as they are.
   - Every other character, **including `_`**, becomes `_` followed by two uppercase hex digits for each UTF-8 byte. For example, `M&M` becomes `M_26M`.
   - The result always satisfies both the Pydantic pattern and the database CHECK, and the mapping is reversible.
   - Existing IDs for ordinary symbols do not change.
2. **Use the helper everywhere a dossier ID is built from a symbol:** the screener's `_save_pick`, the lookup in `run.py`, and any other place you find. Where code reads a symbol back from an ID (graph.py's fallback), use the inverse.
   - Never show an encoded ID to the user.
   - Packs, ideas and API responses carry the real symbol (`M&M`).
3. **Find other places where a share symbol with `&` would fail,** for example a pattern on a symbol field in a research pack, an idea or an API schema, or an exported JSON Schema under `schemas/`. Allow `&` there, or use the helper. List what you checked in the result file.
4. **One bad pick does not stop the week.** In `run_screener`, if saving one pick raises:
   - skip that pick and continue;
   - count it in the output (`"skipped": {"IN": 1}`) along with the exception class.

   Do not swallow errors anywhere else.
5. **Failures say where they happened.** When `run start` or `run continue` prints `{"status": "failed", ...}`, add:
   - `"where"`: the module and function of the innermost stack frame inside `wealth`, e.g. `"wealth.research.screener._save_pick"`;
   - for a Pydantic `ValidationError` only, `"fields"`: a list of `"<loc joined by .>:<error type>"`.

   Never print exception messages or input values; they can contain data. Apply the same to `stock_checks` failures if it has a similar handler.

## Tests (no network, no LLM)
- **Helper:** round trip for `M&M`, `J&KBANK`, `ARE&M`, `BRK.B`, `BF-B`, `RELIANCE`, and a symbol containing `_`. Every encoded ID matches the `ck_dossiers_id` regex copied from migration 0016. Ordinary symbols are unchanged.
- **DB test (`@pytest.mark.db`):**
  - a synthetic `M&M` security on the shortlist saves a dossier through the real CHECK constraint;
  - `load_shortlist_candidates` finds it;
  - the pack or idea built from it shows the symbol `M&M`.
- **Resilience:** a pick whose save raises is skipped and counted, and the other picks are still saved.
- **Failure output:** a forced `ValidationError` inside `run start` prints `where` and `fields`, and prints no input value.

## Out of scope
- Any database migration or edit to an existing migration.
- Re-running production research (the reviewer does that after the push).
- Screener ranking rules.

## Acceptance
- `.venv\Scripts\python -m pytest`
- `.venv\Scripts\python -m ruff check src tests`
- `.venv\Scripts\python -m ruff format --check src tests`
