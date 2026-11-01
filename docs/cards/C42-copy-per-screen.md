# C42: Each screen loads its own words

**Why now:**
- After C40 the initial JavaScript is 169.92 KB gzipped against the 170 KB budget, so the next screen change will fail `npm run budget`.
- The mobile load-speed check (`e2e/tests/performance.spec.ts`, 2.5 s target, not blocking in CI) went from 2.14 s to 2.34 s in CI over C35–C36, and runs 2.80–2.86 s locally.

**Builds on:** the copy pipeline (`copy/en.yaml` → `npm run gen:copy` → `apps/web/src/copy/en.json`) and the route splitting already in `apps/web/src/app/App.tsx`. Read `AGENTS.md` first.

## What happens
- `src/copy/en.json` is 71 KB raw, about 20 KB gzipped. About 27 modules import it whole (`import englishCopy from "../copy/en.json"`), including `src/i18n/config.ts`, `HouseholdContext`, `HouseholdControls`, `Highlights` and `features/data.ts`, which are on the initial path.
- So every screen's words ship in the initial chunk. The reviewer's check on a fresh build: "Why this idea?" (an Ideas-only string) is in `dist/assets/index-*.js`, the initial chunk.
- Vite stringifies JSON files over 10 KB (`json.stringify: 'auto'`), which also blocks tree-shaking of named JSON imports.

## Scope
- Make each route's strings ship with that route's chunk. Keep only the strings the shell and shared initial components need in the initial chunk: navigation, auth and sign-in, the waking-server and offline states, household picker, highlights, labels and errors.
- **Pick the simplest approach that works with the existing generator,** and say which you chose and why in the result file. For example:
  - `gen:copy` also emits one small module per top-level section (`copy/sections/ideas.ts`, …), and screens import only their sections
  - or named JSON imports plus `json: { stringify: false }`, if the build then really tree-shakes them; check that in `dist`
- `en.json` stays the generated source of truth. `en.yaml` stays the one file people edit.
- Generated files stay excluded from Prettier, and nobody hand-edits them.
- `help.en.json` (21 KB) must not be in the initial chunk either. Check it and move it if it is.
- **Keep the copy checks working:** the banned-terms check, word limits and `gen:copy` validation must still cover every string.
- **No visible change.** Every screen shows the same words. Existing tests pass unchanged, except for import paths.

## Guard against regressions
- Extend `scripts/budget.mjs`, or add a small check run by `npm run budget`, that fails when a string known to belong to one route only is in an initial chunk. Use one string each from Ideas, Plan, Money and Settings.
- Print the initial size, as today.

## Acceptance
- In `apps/web`: `npm run gen:copy`, `gen:api`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget`.
- **The initial JavaScript drops by at least 10 KB gzipped** from 169.92 KB. Report before and after in the result file.
- Python is unchanged, so no pytest is needed unless a Python file changes.
- Write `docs/cards/C42.result.md`, with the approach, the sizes, and the four guard strings with their chunks.
- The reviewer runs the e2e suite, including `@performance`.

## Notes
- No new dependencies. Don't change any wording, layout, API or Python.
