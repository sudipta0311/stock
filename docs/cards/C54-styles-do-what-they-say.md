# C54: Styles do what they say

**Source:** the C53 reviewer note in `docs/cards/C53.result.md` ("Reviewer fix (one token)"), approved by the owner on 2026-11-06. The reviewer then measured the built CSS in Chromium. The problem is wider than headings, and this card fixes all of it.

## Why
`apps/web/src/styles.css` has app-wide rules outside any CSS layer. In CSS, unlayered rules beat every layered rule. Tailwind's utilities sit in a layer, so these rules silently override the classes written in components. Measured on the current build:

| Element | Classes say | It renders |
|---|---|---|
| `<h3 class="text-sm font-semibold">` inside a `Card` | 13 px | 17 px (card title) |
| `<h2 class="text-base">` inside a `Card` | 15 px | 17 px |
| `<Button>` (primitive: `text-sm font-semibold`) | 13 px, semibold | 15 px, **regular** |
| `<Card className="p-2">` | 8 px padding | 20 px |
| `<summary class="min-h-12">` | 48 px | 44 px |

The `button, input, select, textarea { font: inherit }` rule is the cause for buttons. So no `<button>` in the app is ever semibold, while primary `<Link>`s styled as buttons are 13 px semibold. The main action looks different from screen to screen.

C53 had to use `text-sm!` to get one small heading. The goal: **a class means what it says**, primary actions look the same everywhere, and screens look the same as today unless this card names a change.

## Read
- `AGENTS.md` (UX rules, PWA commands).
- `apps/web/src/styles.css`, `apps/web/src/ui/primitives.tsx`.
- Every `.tsx` under `apps/web/src/` that renders a `<button>`, a `<Link>` styled as a button, a form field, or an `h2`/`h3` inside a `Card`.

## Scope

### 1. Put the app-wide rules in layers
- Move these rules into `@layer base`:
  - `*` box-sizing;
  - `html, body, #root`; `body`;
  - `button, input, select, textarea { font: inherit }`;
  - the field `min-height`;
  - `input[inputmode="decimal"]`, `input[type="number"]`;
  - tap highlight; `a[data-primary="true"]`;
  - `summary { min-height }`; `details[open] > summary`.
- Move the card rules into `@layer components`: `:where(.card-surface)` (with its `max-width: 359px` media rule), the card heading rule and `.home-hero`.
- The card heading rule becomes `.card-surface :is(h2, h3)` with no `:not(.text-xl)`: a size class on a heading now wins by itself.
- **Leave these unlayered,** because they must always win:
  - `:focus-visible` and `summary:focus-visible`: several fields use `outline-none`, and the focus ring must stay visible;
  - `body[data-scroll-locked]`;
  - the `prefers-reduced-motion` block.
- Add a short comment above them saying why they stay unlayered.
- Keep every `:root` token and the `@theme` block as they are.

### 2. One look for buttons
- In `primitives.tsx`, export one helper (e.g. `buttonClassName(variant)`) that returns the classes `Button` uses today. `Button` uses it.
- Change its text from `text-sm font-semibold` to `text-base font-semibold`. Buttons keep today's 15 px size and become semibold, as written.
- Every `<Link>` or `<a>` styled as a button uses the same helper: rounded, filled or bordered, `px-4`, `min-h-12`. Examples: `HomePage.tsx` around lines 1128–1144, `data-primary` links in `ImportPage.tsx`, `InboxPage.tsx`, `LoanDetailPage.tsx`, `MoneyMapPage.tsx`, `OnboardingPage.tsx`, and `FundCheckCard.tsx:142`. Primary links then match primary buttons: 15 px semibold.
- Raw `<button>` elements (13 in total) keep their own classes. Check each one: wherever the classes now take effect and change its look, keep the declared classes only if the button is a quiet text action. Never let a tap target drop below 44 px.

### 3. Headings inside cards
- Once section 1 is done, every `h2`/`h3` inside a `Card` gets the size its class asks for. Check each one:
  - **The heading that names the card** (the card's title) stays at today's 17 px. Use `text-card-title`, or no size class, in place of `text-base` or `text-sm` on such a heading.
  - **A heading for a part of a card** (e.g. "How we check" inside "Why this suggestion?") becomes the small semibold label its class asks for (`text-sm font-semibold`).
- `HoldingPage.tsx`: remove the `text-sm!` and its comment. Plain `text-sm` now works.
- List every heading whose rendered size changes in the result file: file, heading text, old and new size.

### 4. Form fields stay readable
- Fields stay at today's 15 px: an `input`, `select` or `textarea` with `text-sm` or `text-xs` changes to `text-base`. Typed text never drops below 15 px.
- Fields stay at least 48 px tall (today's `3rem`). Never set a field's height below that.

### 5. Name the destination on back links
The C51 review asked links to name their destination. Three back links still say "Your Money Map":
- `AssetFormPage.tsx` (2);
- `LoanDetailPage.tsx` (1).

Change them to "Back to Money", reusing the existing `holding_review.back_to_money` copy, or move that string to a shared key if that reads better. Keep the arrow icon.

### 6. Nothing else changes
- Where a class was ignored and section 1 now applies it, keep today's look unless sections 2–4 ask for a change. If a newly applied class looks wrong, remove it rather than adding a new one. Don't restyle unrelated screens, colours, spacing or the type scale.
- No new runtime dependencies. Don't hand-edit generated files.

## Tests
- **Vitest:** `tests/styles-cascade.test.ts` reads `src/styles.css` as text. It asserts that every rule outside `@layer` is one of these:
  - a token block: `:root`, `:root[data-theme="dark"]`, the dark-mode `@media` wrapping `:root:not([data-theme])`, or `@theme`;
  - `@import`;
  - one of the allowed unlayered rules from section 1.

  Use a small brace-depth scan; add no new dependency. A new unlayered rule must fail the test, with a message that explains the layer rule.
- **Vitest:** `Button` and the shared helper produce the same classes; a primary link (e.g. Home "Add holdings") uses the helper.
- **Update** any snapshot or class-name assertion that this card's intended changes break. Don't weaken tests otherwise.
- **e2e:** add `e2e/tests/styles.spec.ts` at 360 × 740, reading `getComputedStyle` on seeded synthetic screens:
  - a `Button` and a primary `Link` have the same font size (15 px) and weight (600);
  - on a holding page, "How we check" (opened) is 13 px and the "Our view" card title is 17 px;
  - a form field is at least 48 px tall with 15 px text;
  - a disclosure summary is at least 44 px tall.

  Codex's sandbox can't start the e2e stack (`spawn EPERM`). Write the spec, run `npx playwright test --list`, and say so in the result file. The reviewer runs the suite and takes before and after screenshots.

## Out of scope
- The colour scheme, type scale tokens, spacing tokens, icons and copy (apart from section 5).
- Backend or rule changes. New features.

## Acceptance
Inside `apps/web`:
- `npm run gen:api`
- `npm run gen:copy`
- `npm run typecheck`
- `npm run lint`
- `npm run format:check`
- the full `npm test -- --run`
- `npm run build`
- `npm run budget`

In `e2e`: `npx playwright test --list` must list every spec without errors.

In the result file: list the headings, buttons and links whose look changed, and the tests you updated.
