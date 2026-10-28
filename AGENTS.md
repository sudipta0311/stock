# AGENTS.md: rules for coding agents in this repo

You are implementing the redesign described in `docs/UPGRADE_PLAN.md`. Work arrives as **cards** in `docs/cards/`. Do exactly one card per run.

## How to work a card
1. Read this file, then the card, then only the plan sections the card names.
2. Implement everything in the card's **Scope**. Do not start other cards. Do not "improve" unrelated code.
3. Run every command in the card's **Acceptance** section and fix failures until they pass.
4. Write `docs/cards/<CARD-ID>.result.md` using the template at the bottom of this file.
5. If something in the card is impossible or ambiguous, make the smallest reasonable choice, finish, and list it under "Decisions / open issues" in the result file. Do not stop halfway.

## Hard rules
- **Never** run `git commit`, `git push`, `git mv`, `git add`, `git reset`, `git checkout`, `git stash` or change git config. Read-only git (`git status`, `git diff`, `git log`) is fine. Move files with normal filesystem moves.
- Only touch files inside this repository. Never read, modify or delete `azure-cli-2.90.0-x64/` or `.git/`.
- **Never open, read, print, copy or reference the contents of `secrets.txt`** (repo root, git-ignored; it belongs to the owner and the reviewer).
- No secrets in code, tests, fixtures, logs or result files. Config comes from environment variables (see `.env.example`). Never print environment variable values.
- No real personal data anywhere (names, PAN, account numbers, statements). Test fixtures are synthetic. Never reproduce a string that looks like a PAN (`[A-Z]{5}[0-9]{4}[A-Z]`) in any output.
- Tests never call the network or an LLM. Use fakes, recorded fixtures or `respx`/`monkeypatch`.
- `legacy/` is the old app, kept read-only for reference and porting. Do not edit it unless the card says so.

## Architecture (decisions already made: follow them)
- One Python package `wealth` in `src/wealth/` (src layout), `requires-python >= 3.12`. The local dev interpreter is Python 3.14.
- Subpackages: `core` (money/dates/ids helpers), `schemas` (Pydantic v2 models), `valuation` (deterministic asset formulas), `rates` (scheme-rate reference data loader), `policy` (Policy Engine), `importer` (CAS/broker parsers, staging, de-dup), `db` (SQLAlchemy 2 models, sessions, RLS, Alembic), `api` (FastAPI), `marketdata`, `research` (LangGraph agents + ModelRouter), `alerts` (weekly brief, email, push, reminders).
- Versioned rule data lives in `rules/*.yaml` (allocation, exits, routes, tax, schemes). Exported JSON Schemas live in `schemas/`.
- The PWA lives in `apps/web/` (Next.js static export). It's added by later cards.
- Optional-dependency extras keep the Render API slim: `api`, `importer`, `research`, `dev`. The API must never import `wealth.research`.
- **Money math:** use `decimal.Decimal` for all amounts, rates and money results. Never use float for money. Round to paise (2 dp) with `ROUND_HALF_UP` only at presentation or explicitly-specified boundaries. Rates are percentages as Decimal (e.g., `Decimal("7.10")`).
- **Agents research; code decides money.** No LLM output may set an amount, weight, rate, tax figure or date. LLM code is only in `wealth.research` and `wealth.alerts` (copy writing), always behind the `ModelRouter` interface.
- LLM provider: Azure AI Foundry, OpenAI-compatible v1 endpoint. Config: `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY`, `FOUNDRY_MODEL_WORKHORSE` (gpt-6-luna), `FOUNDRY_MODEL_REDTEAM` (deepseek-v4-flash). The weekly run uses `service_tier="flex"`.
- Hosting target: Render free tier (static PWA + one free FastAPI web service) + Neon Postgres + GitHub Actions for all scheduled jobs (see plan §7.1 and Appendix D).

## UX rules (product owner: "simple, intuitive, easy; hide the complexity")
Apply these to every PWA screen and every user-facing string (emails included):
- **One primary action per screen.** Secondary actions are visually quieter. Details sit behind "Why?" / "Learn more" / collapsed sections, closed by default.
- **No jargon on screen:** sleeve, satellite, tranche, STP, route, PFIC, ABIS, PRIIPs, LRS, TCS, XIRR, RNOR, rebalance, allocation, lot, drift, ISIN, UCITS. Use plain words, or the `<Term>` "What's this?" component where a term is unavoidable.
- **At most 2–3 notes or warnings per screen**, ranked by importance (backend `highlights`). Never dump lists of rules.
- **Smart defaults.** Optional questions are skippable, and the app works with defaults.
- **Plain numbers:** rounded, lakh/crore for INR summaries, DKK with international grouping; full precision only in forms and details.
- **Short copy:** ≤ 20 words per sentence, titles ≤ 8 words, calm tone. "Nothing to do this week ✓" is a good outcome.
- When unsure, show less.

## Tooling and commands (Windows PowerShell)
- Virtualenv: `.venv` at repo root. Create with `python -m venv .venv` if missing.
- Install: `.venv\Scripts\python -m pip install --no-cache-dir -e ".[dev]"` (add other extras when a card needs them).
- Tests: `.venv\Scripts\python -m pytest`
- Lint: `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests`
- PWA (inside `apps/web`):
  - before you start: `npm run gen:api`, `npm run gen:copy`
  - before you finish: `npm run typecheck`, `lint`, `format:check`, the **full** `npm test`, `build`, `budget`
  - generated files are excluded from Prettier (`.prettierignore`); never hand-edit them
- When you add a dependency, add it to `pyproject.toml` in the right extra with a lower bound, then reinstall.
- Database tests: a local PostgreSQL 18 runs on `localhost:54329`. Its URL is already in the environment variable `TEST_DATABASE_URL` (the database `wealth_test` may be dropped and recreated freely). Never print or log the URL; it contains a password. Mark DB tests `@pytest.mark.db` and skip them when the variable is unset.
- **Never change cluster-wide database state by hand**: no `ALTER ROLE`, `CREATE/DROP ROLE`, `CREATE/DROP DATABASE` and no changes to the roles `wealth_app`/`wealth_api` outside code that runs inside a test fixture and restores what it changed. Roles are shared by every database on the cluster (the developer's local app uses the same cluster). If a test fails because of cluster state, fix the test or report it. Don't change the state.
- Don't add dependencies you don't use.
- **New user-owned tables use the household RLS pattern** from migration 0011:
  - `user_id` = the household owner; a policy `<table>_household_access` reads with `app_household_role(user_id) IS NOT NULL` and writes with `IN ('owner','partner')`, both on `app.household_owner`
  - add the `trg_<table>_household_audit` trigger
  - only per-login records (like `consents`, `push_subscriptions`) use `app.user_id` isolation
  - never re-create the old C05 per-user policy
  - `app_merge_into_household` loops over every table with `user_id`, so add a duplicate rule there if the table has a per-user unique key
- **Never edit a migration created by an earlier card.** Databases (including the developer's) have already applied it, so edits never reach them. Schema changes always go in a **new** migration with the next free number that revises the current head. Editing a migration you created in the *same* card is fine.

## Code style
- Type hints everywhere. Small, focused modules. Pure functions for calculations.
- Match the surrounding code. Comments only where the *why* is not obvious (e.g., cite the scheme rule a formula implements).
- Tests live in `tests/` mirroring the package path (`tests/valuation/test_fd.py` etc.). Test names describe behaviour.
- Indian formatting helpers (lakh/crore) are in `wealth.core.money`. Use them; don't re-implement.

## Result file template (`docs/cards/<CARD-ID>.result.md`)
```markdown
# <CARD-ID> result
## Summary
<3-6 bullets: what was built>
## Files
<created / moved / modified / deleted, grouped>
## Acceptance
<each acceptance command and its final outcome, e.g. "pytest: 42 passed">
## Decisions / open issues
<anything you had to decide or could not do>
```
