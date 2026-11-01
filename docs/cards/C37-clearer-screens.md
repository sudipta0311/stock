# C37: Clearer screens (UX round 2)

**Requested by the product owner:** "keep improving the app... it should be user friendly and intuitive."
**Builds on:** C36 (simpler everyday app), C35 (buy hints), C22 Ideas, C07 Money Map, C21 Plan. Read `AGENTS.md` first, especially the UX rules.
**Reviewed against:** the official Anthropic `frontend-design` skill (github.com/anthropics/skills). Its rules: plain words named after what the user does; an action keeps the same name through a flow; empty states invite an action; structure carries information and isn't decoration; avoid "A · B · C" meta strings and identical nested cards. Also the AGENTS.md UX rules, and the reviewer's walkthrough on the owner's phone (384 px, real data) after C36.

## Findings from the phone, and what to do

### 1. Money: wrong count and internal codes (bugs)
- The "Your money" heading shows a badge "0 items" while the owner has 43 holdings. Show the real count of listed items (holdings plus other assets), or drop the badge. Add a test with holdings present.
- Holding rows show internal codes on the second line ("AMFI-122639-G · 38,647.…", "SBINIFTY50 · 250 units"). Codes and ISINs are on the banned list for screens. Show a plain second line instead:
  - mutual funds and ETFs: the units, rounded ("38,647 units")
  - shares: the share count ("250 shares")
  - bonds and SGBs: the units
  - Keep the code only on the holding's detail page, inside a closed "Details" section.

### 2. Ideas: say why, in plain words
- **Valuation words.** "Below the middle" means nothing to a reader. The API computes a `valuation_label` per idea (`cheap` | `fair` | `expensive`) from `valuation_percentile` and the thresholds in `rules/research.yaml` (`cheap_pe_percentile_max`, `expensive_pe_percentile_min`). The PWA shows "Looks cheap", "Fairly priced" or "Looks expensive". The PWA never hard-codes thresholds.
- **Confidence.** "How sure we are: Medium" repeats on every row. Show confidence only when it isn't Medium, as a quiet badge ("Higher confidence" or "Lower confidence"), and explain the scale once in the card's "What's this?".
- **"Why this idea?"** Add a closed section under each idea with up to two reasons and the main risk, in plain words.
  - Source: the dossier's compliance-checked `drivers` and `risks` (and the red team's falsifier as the risk if `risks` is empty).
  - Copy them into the pack idea at pack time (`why: {"reasons": [...], "risk": "..."}`) and expose them on `ResearchIdeaSummary`.
  - Truncate each to 160 characters on a word boundary.
  - Older packs have no `why`, and the section then doesn't render.
  - These are already-compliant research texts. Don't add new model calls.
- **Order.** Keep the C35 buy-hint line first under the name (when present), then the valuation words, then "Why this idea?".

### 3. Plan: show now vs plan and the gap
- "Your money vs your plan" draws only one bar (today's mix). Draw two thin bars, "Now" and "Plan", with the same colours and legend.
- Under them, one plain sentence for the biggest gap, from code (the plan's targets minus today's values), rounded with the lakh/crore helpers. For example: "Your safe money is ₹12 L below plan. New money goes there first."
- Keep amounts behind "Show amounts", as today.
- The primary button stays "Update your money". If there's a gap, its copy follows the gap ("Add safe money"), linking to the right Money add flow.

### 4. Family picker takes a whole row on every screen
- Move the household or member picker into the header row as a compact control, to the left of the help icon (the same component in a smaller variant). On a 360 px screen the title must not wrap, so check it.
- Show the picker only on screens where it changes what's shown (Home, Money, Plan). Inbox, Ideas and Settings don't need it.

### 5. Home amount in the currency the person thinks in
- When the signed-in member lives in Denmark, default the Home amount field to DKK, with a small "kr / ₹" switch beside the field. Otherwise default to ₹.
- Saving still converts to INR with `dkkToInr` (C36 round 1). With no FX rate, disable the DKK option and say why in one short line.

### 6. Small polish
- **Less card nesting.** Holdings lists and the ideas list use a plain list inside one card, not card-in-card. Section headings carry the structure.
- **Meta strings.** Replace "A · B" strings that mix unlike facts with two short lines, or one line where the second fact moves to the right column. Examples: "Information Technology · Below the middle", "DKK 1,304,457 · Updated 1 Nov 2026". Keep the dot only for true lists of the same kind.
- **Empty states invite one action** ("Nothing to buy from Denmark this week" → add "Set your monthly amount" when no amount is set).

## Tests
- **Python:**
  - `valuation_label` mapping at the thresholds (32, 33, 66, 67)
  - `why` packed from a dossier: two reasons and a risk, truncation, and the falsifier fallback
  - an older pack without `why` or a label still serves
- **PWA:**
  - the Money count and plain second lines (no AMFI/ISIN codes on the list)
  - Ideas valuation words, the confidence badge only when not Medium, and "Why this idea?" closed by default
  - Plan's two bars and the gap sentence
  - the picker in the header on Home/Money/Plan and not on Inbox/Ideas/Settings
  - the DKK default and switch for a Denmark member
- **e2e:** update selectors for the moved picker. The reviewer runs the suite locally.

## Acceptance
- `.venv\Scripts\python -m pytest` (full), ruff check and format (`--no-cache`), `python -m scripts.check_deploy_config`
- `.venv\Scripts\python -m wealth.api.export_openapi`, then in `apps/web`: `npm run gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, the full `npm test`, `build`, `budget` (stay under 170 KB; it's about 169 KB now, so keep new code light or lazy)
- Write `docs/cards/C37.result.md`, with before/after notes per screen.

## Notes
- No new dependencies. Decimal for any money math in Python; string or decimal helpers in the PWA.
- Don't change policy or research decisions. This card only changes presentation, plus the two pack fields.
