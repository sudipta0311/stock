# C28b: Visual polish: a professional, calm, consistent look on phones first

**Requested by the product owner:** "Improve the look and feel of the UI. Make it professional and polished."
**Plan:** `docs/UPGRADE_PLAN.md` §8 (UX). Read `AGENTS.md` first, especially the **UX rules**. Builds on C28 (simplified screens). PWA only, plus copy. No backend changes unless a listed bug needs one.

## Goal
The app should feel like a well-made banking app on a phone (360–430 px wide): one clear type scale, steady spacing, quiet surfaces, numbers that line up, and nothing that looks unfinished. Same content and flows as C28; this card changes how it looks, not what it does.

## Scope

### 1. Design tokens (`apps/web/src/styles.css`)
- **Type scale** (use everywhere, no ad-hoc sizes):
  - page title 22/28 semibold
  - card title 17/24 semibold
  - body 15/22
  - small 13/18
  - hero number 34/40 bold, tight tracking
- **Numbers:** `font-variant-numeric: tabular-nums` for every amount.
- **Font:** the system stack (Roboto on Android, SF on iOS). Remove "Inter" from the stack unless you self-host a Latin-subset variable woff2 of 50 KB or less with `font-display: swap`.
- **Radius:** cards 16, controls 12, pills full.
- **Elevation:** one soft card shadow token plus the 1 px line. Same on every card.
- **Spacing:** card padding 20 (16 below 360 px), 16 between cards, 8/12 inside.
- Keep the existing colour tokens and dark theme. Every new style uses tokens and passes WCAG AA contrast in both themes.

### 2. App shell
- **Header:** 56 px high; the page title on the left and the profile button on the right.
  - Replace the "WEALTH COPILOT" eyebrow with a small logo mark beside the title, or drop it on inner pages.
  - Remove the "Owner" chip from the header. Show the role only in Settings, and only when the household has more than one login.
- **Bottom nav:** 64 px plus `env(safe-area-inset-bottom)`. The active tab gets the brand colour and a soft pill behind the icon; labels are 12 px medium. Page content must never hide behind the nav.
- **Icons:** add `<link rel="icon">` (it currently 404s on `/favicon.ico`) and `theme-color` metas for light and dark.

### 3. Components (`apps/web/src/ui/`)
- **Buttons:**
  - primary: filled brand
  - secondary: outline
  - tertiary: text
  - all 48 px high (44 px minimum for compact ones)
  - a pressed state (`active:scale-[.98]`) that respects `prefers-reduced-motion`
- **Highlights / notes** (one component, used on every screen):
  - a compact callout with a severity icon
  - `attention` uses the warning-soft tint; `info` uses muted
  - a title and body, then a footer row with "Why?" on the left and "Hide" on the right, both quiet text buttons with the same size and baseline, tap height at least 44 px
- **Empty states:** an icon in a soft circle, one sentence, one button.
- **Loading:** skeletons shaped like the cards they replace. No blank screens.
- **List rows** (Money, Ideas):
  - a 36 px icon tile
  - title plus one muted sub-line (e.g. "SBI · matures Mar 2027")
  - the amount on the right, tabular and rounded
  - a chevron when the row opens something

### 4. Screens
- **Home:**
  - The net-worth card becomes the hero: a brand-tinted surface, the big rounded number, DKK below it (rounded) and a tiny "as of" line.
  - "This week" rows use the list-row style.
  - The empty state is a success icon with "Nothing to do this week ✓".
- **Plan:**
  - "Your money vs your plan" is a 12 px rounded stacked bar with 2 px gaps between segments.
  - With no money added yet, show an empty state ("Add what you have to see your plan", button to Money) instead of a grey bar.
- **Money:** grouped sections with small muted headings and list rows; the upload card stays first.
- **Forms:**
  - inputs 48 px high
  - amount fields get a ₹/DKK prefix and `inputmode="decimal"`
  - the formatted lakh/crore value stays under the amount
- **Onboarding:** keep the step cards; make the progress bar and step label quieter and align the back link with the content.

### 5. Bugs found on a real phone (OnePlus, Chrome, 384 px wide)
1. **Plan with no assets:** it says "Your safe money is close to your plan." That's wrong copy for an empty account; use the empty state from §4.
2. **Onboarding household step:** "You" shows twice, once as the legend and again as a stray label above "Tax residence".
3. **Rounding in summaries:** DKK isn't rounded on Home ("DKK 0.00", "DKK 35,242.29"). Summaries round per the UX rules: DKK to whole kroner with international grouping, INR in lakh/crore. Full precision only in forms and details.
4. **Repeated helper text:** "Rough numbers are fine. You can edit this anytime." repeats under every field of the FD form, even under the bank name. Show it once under the form title; field hints only where they add information.
5. **Done step back link:** on "Your plan is ready" it is labelled "Welcome". Use the previous step's name, or no back link on the final step.
6. **Favicon 404:** see §2.

## Constraints
- No new runtime dependencies. The initial JS budget stays at 170 KB gzip or less (now 168.65 KB). Prefer CSS and Tailwind classes to new components.
- Keep `data-primary`, the one-primary-action tests, collapsed-by-default details, the highlight limits, the banned-terms and sentence-length tests, and the axe checks. Update test expectations only where markup changed on purpose.
- Copy changes go through `copy/en.yaml` and `npm run gen:copy`.

## Acceptance
1. PWA: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, full `npm test`, `build`, `budget` all pass.
2. Python: full `pytest` and `ruff` pass (the copy changes are shared with the emails).
3. The result file lists, per screen, what changed visually, and confirms each bug in §5 is fixed.
