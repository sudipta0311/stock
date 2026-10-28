# C25: Wrap-up: follow-ups, security & dependency audit, docs and plan sync, final full verification

**Plan:** the whole of `docs/UPGRADE_PLAN.md`, and the "Follow-ups noted during review" list at the bottom of `docs/cards/README.md`. Read `AGENTS.md` first.

## Goal
Close the loose ends found during reviews, make sure nothing unsafe ships, and leave documentation that matches what was actually built.

## Scope

### 1. Follow-ups from reviews (each with a regression test)
- **EPF accrual (C04):** switch from actual/365 to EPFO's method. Interest = (sum of monthly running balances) × annual rate / 1200, credited at FY end. Update the tests with a hand-computed example: balance ₹10,00,000 on 1 Apr, ₹13,670 contributed monthly, 8.25%. Each month's contribution earns interest from the following month, so April's contribution earns 11 months and March's earns 0. Expected FY interest = 10,00,000 × 8.25% (₹82,500.00) + 13,670 × 66 × 8.25 / 1200 (₹6,202.76) = **₹88,702.76**; put this derivation in the docstring.
- **Plan sync additions:**
  - the Denmark buying guide (C29a: ASK first, ABIS/UCITS only, US stocks allowed, no US-listed ETFs)
  - the Indian funds check with early warning (C29b)
  - the UX rules / simplicity pass (C28)
  - the household and shared access (C27a–c)
- Any other open item in the README follow-ups list that's still applicable. If an item is already fixed, mark it done with the commit-free evidence (test name).

### 2. Security & privacy audit (report counts, never values)
- Repo-wide secret scan over tracked + untracked files (not `.git`, `.venv`, `node_modules`, `.npm-cache`, `.playwright-browsers`, `azure-cli-2.90.0-x64`) for:
  - private keys
  - `postgres(ql)?://user:pass@`
  - `api-key`/`sk-`/`Bearer` literals
  - 32+ char base64/hex in config files
  - PAN-pattern strings
  - email addresses other than `@example.test` / `@example.invalid` / `@example.com` in code and fixtures

  Result: 0 or justified.
- `pip-audit` (add to dev) over the installed environment and `npm audit --omit=dev` in `apps/web` and `e2e`. Fix high/critical issues by upgrading pinned versions where a non-breaking fix exists; list the rest with a rationale.
- License check: list the direct dependency licenses (Python + npm). Flag anything non-permissive (GPL/AGPL) for the reviewer.
- **Confirm:**
  - the API never imports research/openai/langgraph (test exists)
  - no LLM keys in `render.yaml`
  - no file writes during upload (test exists)
  - `wealth_api` capability check (test exists)
  - CSP present in the PWA build

### 3. Legacy
Delete `legacy/` **only if** nothing outside it references it (grep). It remains in git history for reference. If anything still references it, keep it and list the references.

### 4. Docs (written for a new developer joining the project)
- **`README.md`:** what the product is (3 sentences), architecture diagram (text), repo layout, local dev (Postgres, API, PWA, jobs), testing (Python, PWA, e2e, evals), deployment pointer, security notes.
- **`docs/UPGRADE_PLAN.md`:** a short "What changed during implementation" section at the top, recording the decisions made in the cards. Edit only what's needed; keep the rest:
  - Vite instead of Next.js
  - Foundry deployments `gpt-6-luna` (all tiers via reasoning effort, Flex for weekly) + `deepseek-v4-flash` (red team) with real prices
  - ECB for FX
  - portal AMFI host
  - the manual_assets single table
  - the `wealth_api` login + SECURITY DEFINER bootstrap
  - CDSL behind a beta flag
  - fundamentals source outcome for India
  - templates instead of LLM copy in emails
  - the `SANDBOX_BUILD` flag
  - tilt decay
- **`docs/cards/README.md`:** final statuses; follow-ups list updated.
- **`AGENTS.md`:** unchanged except to fix anything now inaccurate.

### 5. Final verification
- Full Python suite with DB tests.
- Ruff.
- The PWA suite + standard build (no `SANDBOX_BUILD`) + budget.
- Evals (fake).
- e2e if the stack can run in your environment; otherwise state that the reviewer runs it.

## Acceptance
All the commands in §5 pass and are reported with counts; the audit sections are reported; the docs are updated. Don't commit.
