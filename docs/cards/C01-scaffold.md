# C01: Repo restructure, sensitive-data scrub, new package scaffold, CI

**Plan:** `docs/UPGRADE_PLAN.md` §1.3 (P0-1), §7.2, §13. Read `AGENTS.md` first.

## Goal
Move the old Streamlit app out of the way into `legacy/`, remove personal data from the working tree, and create a clean, installable, tested skeleton for the new `wealth` package.

## Scope

### 1. Move the old app into `legacy/` (plain filesystem moves, no git commands)
Move these paths, keeping their internal structure:

| From | To |
|---|---|
| `streamlit_app.py` | `legacy/streamlit_app.py` |
| `src/stock_platform/` | `legacy/src/stock_platform/` |
| `agents/` | `legacy/agents/` |
| `utils/` | `legacy/utils/` |
| `backtest/` | `legacy/backtest/` |
| `tests/` | `legacy/tests/` |
| `rules/` | `legacy/rules/` |
| `.streamlit/` | `legacy/.streamlit/` |
| `.devcontainer/` | `legacy/.devcontainer/` |
| `.github/workflows/backtest.yml` | `legacy/github-workflows/backtest.yml` |
| `README.md`, `SUMMARY.md`, `SESSION_LOG.md`, `PLATFORM_DETAILED_NOTES.md` | `legacy/docs/` |
| `stock_platform_v5.puml`, `stock_platform_v6.puml` | `legacy/docs/` |
| `pyproject.toml`, `requirements.txt`, `runtime.txt` | `legacy/` |

After the move, `src/` must contain only the new `wealth` package, and the repo root must not contain any of the moved files.

### 2. Scrub personal data (P0-1)
- Delete the real statement PDF (it ends up at `legacy/tests/NSDLe-CAS_*.PDF` after the move). Do not open or read it.
- In every text file under `legacy/` (and anywhere else in the repo except `.git/` and `azure-cli-2.90.0-x64/`):
  - Replace any string matching the PAN pattern `[A-Z]{5}[0-9]{4}[A-Z]` with `<REDACTED>`. Where code used it as a password, read it from the environment variable `CAS_TEST_PASSWORD` instead, and skip the test (`pytest.skip`) when that variable is unset.
  - Replace the statement filename `sample_cas.pdf` with `sample_cas.pdf`. Replace the bare number `<REDACTED>` with `<REDACTED>` wherever it identifies that statement.
- **Never print, log or write the matched values** in command output or the result file. Use commands that report only counts, e.g. a Python script that prints `files_changed=N`.

### 3. `.gitignore`
Rewrite it to cover: `__pycache__/`, `*.py[cod]`, `.venv/`, `venv/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, `*.egg-info/`, `.env`, `.env.*` (but keep `!.env.example`), `/data/`, `*.db`, `*.sqlite3`, `*.db-journal`, `*.log`, `node_modules/`, `apps/web/.next/`, `apps/web/out/`, `.streamlit/secrets.toml`, `legacy/.streamlit/secrets.toml`, `azure-cli-*/`, `.claude/`. Remove the stray `=0.40.0` line.

### 4. New `pyproject.toml` (root)
- setuptools build, src layout, project name `wealth-copilot`, package `wealth`, version `0.1.0`, `requires-python = ">=3.12"`.
- Base dependencies: `pydantic>=2.11`, `pyyaml>=6.0`.
- Extras (empty lists are fine for now, later cards fill them): `api = []`, `importer = []`, `research = []`, `dev = ["pytest>=8.3", "ruff>=0.8"]`.
- `[tool.ruff]`: `line-length = 100`, `target-version = "py312"`, `src = ["src", "tests"]`; lint `select = ["E", "F", "I", "B", "UP", "SIM"]`; exclude `legacy`, `apps`, `azure-cli-2.90.0-x64`.
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "-q"`.

### 5. Package skeleton
Create `src/wealth/__init__.py` with `__version__ = "0.1.0"`. Then create these subpackages, each with an `__init__.py` holding a one-line docstring that says what it will contain (from AGENTS.md "Architecture"): `core`, `schemas`, `valuation`, `rates`, `policy`, `importer`, `db`, `api`, `marketdata`, `research`, `alerts`.
Create empty folders with a `.gitkeep`: `rules/`, `schemas/`, `apps/web/`.

### 6. Tests
`tests/test_smoke.py`:
- Imports every subpackage listed above.
- Asserts `wealth.__version__ == "0.1.0"`.
- Asserts no file under `src/` or `tests/` contains a PAN-pattern string (scan the text files and assert the count is 0, without printing matches).

### 7. Env template
`.env.example` with variable **names** and placeholder values only: `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY`, `FOUNDRY_MODEL_WORKHORSE=gpt-6-luna`, `FOUNDRY_MODEL_REDTEAM=deepseek-v4-flash`, `DATABASE_URL`, `TEST_DATABASE_URL`, `AUTH_JWKS_URL`, `AUTH_ISSUER`, `AUTH_AUDIENCE`, `ALLOWED_ORIGINS`, `EMAIL_API_KEY`, `EMAIL_FROM`, `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `CAS_TEST_PASSWORD`.

### 8. CI
`.github/workflows/ci.yml`: on `push` and `pull_request`; ubuntu-latest; Python 3.12; `pip install -e ".[dev]"`; `ruff check src tests`; `ruff format --check src tests`; `pytest`.

### 9. README
New root `README.md` (≤ 60 lines):
- One paragraph on what the app is, with a link to `docs/UPGRADE_PLAN.md`.
- Repo layout.
- Dev setup on Windows: create venv, install, run tests, run lint.
- Pointer to `docs/cards/README.md`.
- Note that `legacy/` holds the old Streamlit app for reference only.

## Out of scope
Any business logic. Changes to `docs/UPGRADE_PLAN.md`, `docs/cards/README.md` or `AGENTS.md`.

## Acceptance (run all; all must pass)
1. `python -m venv .venv` (if missing), then `.venv\Scripts\python -m pip install --no-cache-dir -e ".[dev]"`
2. `.venv\Scripts\python -m pytest` → all pass
3. `.venv\Scripts\python -m ruff check src tests` → clean
4. `.venv\Scripts\python -m ruff format --check src tests` → clean (run `ruff format src tests` first if needed)
5. A Python one-off that counts PAN-pattern matches in all text files outside `.git/`, `.venv/` and `azure-cli-2.90.0-x64/` and prints only the count → `0`
6. `Test-Path legacy\tests\*.PDF` → `False`; `Test-Path streamlit_app.py` → `False`
7. `git status --short` runs (show a summary count of moved/added/deleted entries in the result file, not the full list)
