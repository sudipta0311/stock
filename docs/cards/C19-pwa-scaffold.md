# C19: PWA scaffold: Vite + React + TypeScript, design system, app shell with bottom navigation, service worker, typed API client, auth (OIDC + dev token), Indian formatting

**Plan:** `docs/UPGRADE_PLAN.md` §8.1 (why a PWA; iOS limits; Indian device reality; performance budget; cold-start UX), §8.2 (five tabs), §8.6 (accessibility, lakh/crore), §8.7 (frontend tech), §7.4 (PWA device security), Appendix D (Render static site, API sleeps on the free tier). Read `AGENTS.md` first.

## Decision (supersedes plan §8.7's Next.js choice)
**Vite + React 19 + TypeScript (strict)** SPA, built to static files in `apps/web/dist/` and served by a Render Static Site. Reason: a client-only PWA gets nothing from Next's server features; Vite gives a smaller bundle and no static-export dynamic-route workarounds.

## Tooling (pin exact versions in `package.json`; commit `package-lock.json`)
- Use the latest **stable** releases available on npm at run time of: `vite`, `react`, `react-dom`, `typescript`, `react-router` (v7, data/declarative routes), `@tanstack/react-query`, `tailwindcss` (v4 + `@tailwindcss/vite`), `@radix-ui/react-*` (dialog, tabs, toast, progress, dropdown-menu), `lucide-react`, `react-hook-form`, `zod`, `vite-plugin-pwa`, `oidc-client-ts`, `openapi-typescript`.
- Dev: `vitest`, `@testing-library/react`, `jsdom`, `eslint` (+ `typescript-eslint`, `eslint-plugin-jsx-a11y`, `eslint-plugin-react-hooks`), `prettier`.
- Node: use the `node`/`npm` on PATH (v24). npm's cache is already redirected into the repo. Run npm commands **inside `apps/web/`**.
- Scripts:
  - `dev`, `build`, `preview`, `test` (vitest run), `lint`, `typecheck` (`tsc --noEmit`), `format:check`
  - `gen:api` (openapi-typescript `../../schemas/openapi.json` → `src/api/schema.d.ts`)
  - `budget` (fails if the gzipped initial JS > 170 KB; implement it as a small Node script reading `dist/` + Vite's manifest)

## Scope

### 1. Design system (`src/ui/`)
- Tailwind v4 theme tokens as CSS variables for light and dark (`prefers-color-scheme` + a manual toggle stored in `localStorage` wrapped in try/catch).
- Colours: neutral surfaces, one brand accent. Status colours always paired with an icon + text (never colour alone).
- Components with accessible defaults (focus rings, ≥ 48 px tap targets):
  - `Button` (primary/secondary/ghost/danger, loading state), `Card`, `Stat`, `Badge`, `Progress`
  - `Skeleton`, `EmptyState`, `Sheet` (Radix dialog as a bottom sheet on mobile), `Toast`
  - `Field` (label, help, error from problem+json `errors[]`), `MoneyInput` (see §5), `MonthYearInput`, `Select`, `Checkbox`, `TileToggle` (Money Map tiles)
- Typography scale and spacing for a 360 px-wide screen first.

### 2. App shell and routes (`src/app/`)
- Bottom nav with 5 tabs: **Home, Money, Plan, Ideas, Inbox** (icons + labels; `aria-current`). Top bar: title + avatar → Settings.
- Routes (stub pages with `EmptyState` placeholders, filled by C20–C23):
  - `/`, `/money`, `/money/add/:type`, `/money/asset/:id`, `/import`, `/import/:jobId`
  - `/plan`, `/ideas`, `/ideas/world`, `/inbox`, `/inbox/brief/:week`
  - `/settings`, `/onboarding/*`, `/login`, `/auth/callback`
- Error boundary with a friendly message plus "Try again"; 404 page.

### 3. Typed API client (`src/api/`)
- `schema.d.ts` generated from `schemas/openapi.json` (commit the generated file; `gen:api` regenerates it; a test fails if it's stale versus the OpenAPI file's hash).
- `apiFetch<T>(path, init)`:
  - base URL from `import.meta.env.VITE_API_URL`
  - adds `Authorization: Bearer <token>`
  - parses RFC 7807 problems into a typed `ApiError {status, code, title, errors[]}`
  - **cold-start aware:** on a network error or 502/503/504 it retries with backoff for up to 60 s while emitting a `waking` event that the shell shows as a small "Waking up the server… (free hosting)" chip
- TanStack Query client with sensible stale times; mutations invalidate the related queries.

### 4. Auth (`src/auth/`)
- Provider-agnostic OIDC with `oidc-client-ts` (Authorization Code + PKCE). Config from `VITE_OIDC_AUTHORITY`, `VITE_OIDC_CLIENT_ID`, `VITE_OIDC_SCOPE` (default `openid email profile`), `VITE_OIDC_AUDIENCE` (optional). Silent renew. Tokens in memory + session storage only (never localStorage).
- **Dev mode** (`VITE_AUTH_MODE=dev`, refused when `import.meta.env.PROD` is true): `/login` shows a "paste a dev token" field. The token is minted locally with the API's documented snippet; the PWA never holds the dev secret.
- Route guard: unauthenticated → `/login`; first login with no profile → `/onboarding`.
- Logout clears the query cache, IndexedDB caches (§6) and storage.

### 5. Indian formatting in TypeScript (`src/lib/money.ts`), mirroring C02 exactly
- `parseInr(text)` → canonical string `"500000.00"` (same accepted forms and rejections as Python `parse_inr`)
- `formatInr(value, "full" | "compact")` and `formatUsd`
- Use string/decimal-safe arithmetic only (no floating-point money). Use a tiny decimal helper or `big.js`; if you add `big.js`, pin it.
- **Port the Python test tables** (the same input → output cases from `tests/core/test_money.py`) into vitest, so both implementations stay in lockstep.

### 6. PWA (`vite-plugin-pwa`, `injectManifest` or `generateSW`, your choice)
- Manifest:
  - name "Wealth Copilot", short_name "Wealth", `display: standalone`
  - theme/background colours, `start_url: "/"`, `scope: "/"`
  - icons 192/512 + maskable, generated at build time from an SVG in `public/`; don't commit large binaries, small PNGs are fine
- Precache the app shell. Runtime caching:
  - `NetworkFirst` with a 3 s timeout for GET `/v1/networth`, `/v1/plan`, `/v1/inbox` (cache name `api-snapshots`, max 20 entries, 7 days)
  - **never** cache auth endpoints, imports, or any non-GET
- On logout, clear `api-snapshots`. A setting "Shared device: don't keep data offline" disables the runtime cache.
- An update toast when a new service worker is waiting ("New version, tap to refresh").

### 7. Quality gates
- `npm run typecheck`, `lint`, `test`, `build`, `budget` all pass.
- Tests (vitest): the money tables; `apiFetch` problem parsing and cold-start retry (fake timers); route guard; bottom-nav renders 5 tabs with accessible names; theme toggle persists (with storage throwing → still works).
- `README.md` → a "Run the PWA locally" section: `cd apps/web`, `npm ci`, `npm run gen:api`, set `VITE_API_URL=http://127.0.0.1:8765`, `VITE_AUTH_MODE=dev`, `npm run dev`.

## Out of scope
Real screens (C20–C23), Playwright e2e (C24), Render config (C18).

## Acceptance (run inside `apps/web/`)
1. `npm ci` (or `npm install` the first time to create the lockfile), then `npm run gen:api`
2. `npm run typecheck`, `npm run lint`, `npm run test`, `npm run build`, `npm run budget` → all pass. Report the gzipped initial JS size.
3. Python suite still green: `.venv\Scripts\python -m pytest` (repo root)
4. No secrets in `apps/web` (the dev token is pasted at runtime, never committed); `node_modules/`, `dist/` and `.npm-cache/` are gitignored
