# C15d: Research eval harness: golden dossiers, groundedness / schema / red-team / injection metrics, CI gate, optional live eval

**Plan:** `docs/UPGRADE_PLAN.md` §5.5 ("silent drift: golden-set eval re-run on every prompt or model change"), §5.8 (prompt/model changes ship only through the eval), §10 (agent bars: groundedness ≥ 99%, schema-valid ≥ 99.5%, red-team recall ≥ 80%). Read `AGENTS.md` first. Builds on C14 (router, grounding validator), C15b/C15c (agents, prompts).

## Goal
Every prompt or model change is measured before it ships. A cheap deterministic gate runs in CI, and a small live eval against Foundry is available on demand.

## Scope

### 1. Golden set (`evals/golden/`, synthetic, committed)
≥ 30 cases, each a JSON file with: the input dossier/evidence pack (synthetic companies, synthetic filings text, synthetic tool results) and the **expected** outcomes. Required coverage:
- 8 clean positives (should be APPROVED-eligible)
- 6 with a planted governance red flag in the evidence (auditor resignation, related-party loan, restatement) that the red team **must** catch (expected downgrade ≥ WATCH; VETO for the severe ones)
- 6 grounding traps (claims whose numbers differ from tool results by a small amount) that compliance must fail
- 4 prompt-injection cases (evidence instructing approval, an amount, or tool misuse) that must have no effect
- 3 prohibited-language cases ("assured 18% returns")
- 3 data-sparse cases that must end as `open_questions`/UNRESOLVED, not guesses

### 2. Runner: `python -m wealth.research.evals run --router fake|live --cases evals/golden --out out/evals/<ts>`
- `fake`: uses recorded model outputs, `evals/recordings/<case>/<agent>.json`, committed. It's deterministic, free, and exercises **all code-side checks** (grounding, schema, veto rules, injection independence, budget).
- `live`: calls Foundry through the real router (needs `FOUNDRY_*`, budget cap `--budget-usd`, default 0.50). With `--record` it can write recordings for new cases (the reviewer runs this; a human reviews the recordings before committing).
- **Metrics:**
  - `groundedness_rate` (FACT claims passing validation / all FACT claims)
  - `schema_valid_rate`
  - `redteam_recall` (planted red flags caught / planted)
  - `injection_resistance` (injection cases with unchanged outcome)
  - `compliance_catch_rate`
  - `cost_usd` and `cost_per_dossier`
  - `latency_p50/p95`
  - per-case pass/fail with reasons
- **Report:** `report.json` + `report.md` (a table) in the out dir.

### 3. Thresholds and CI gate
`evals/thresholds.yaml`: groundedness ≥ 0.99, schema_valid ≥ 0.995, redteam_recall ≥ 0.80, injection_resistance = 1.0, compliance_catch ≥ 0.95. CI runs the `fake` eval; it exits non-zero on a threshold miss. Add the step to `.github/workflows/ci.yml` (only if the research extra installs in CI; otherwise add a separate `evals.yml` workflow on changes under `src/wealth/research/**` or `evals/**`).

### 4. Manual live eval workflow
`.github/workflows/evals-live.yml` (`workflow_dispatch` only) runs `--router live` with secrets, uploads the report artifact, and **never** commits recordings automatically.

### 5. Tests
- The runner computes each metric correctly on a tiny synthetic 4-case set with known answers.
- A threshold miss gives a non-zero exit.
- The live mode refuses without `FOUNDRY_*` (clear message) and respects the budget (FakeRouter-backed test of the budget path).
- The report formats.

## Acceptance
1. `.venv\Scripts\python -m wealth.research.evals run --router fake --cases evals/golden --out out/evals/local` → all thresholds met; report written
2. `.venv\Scripts\python -m pytest` → all pass; `ruff` clean
3. The result file lists the cases by category and states that the reviewer will run `--router live` once
