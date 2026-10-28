# C22: PWA: Home ("This week"), Plan (targets, debt answer, global, foundation, insurance), Ideas (core picks, research ideas, World & your money + stress test), holding Exit Cards, Inbox

**Plan:** `docs/UPGRADE_PLAN.md` §8.2 (five tabs), §8.4 wireframes (Home, Plan, Ideas "World & your money", Holding with its Exit Card), §8.5 (plain-language labels replace jargon), §2.3 principles (calm by default, "do nothing" is valid, ≤ 3 actions), §4.11 (insurance: compare options, never "surrender now"). Read `AGENTS.md` first. Builds on C19/C20 and the API from C17a (`/v1/plan`, `/v1/actions`, `/v1/stress`, `/v1/research/ideas`), C07 (`/v1/networth`), C17b (`/v1/inbox`, `/v1/brief/{week}`): **feature-detect** any endpoint missing from the OpenAPI types and show a friendly "coming soon" state instead of failing.

## Goal
Opening the app answers "Do I need to do anything this week?" in one glance, then lets the user see why: their plan, their global diversification, what the world means for their money, and when each holding would be sold.

## Scope

### 1. Language map (`src/lib/labels.ts`, single source; tests enforce it)
| Code | Shown as |
|---|---|
| `BUY_LUMPSUM` / `BUY_STP` / `BUY_STOCK_TRANCHE` / `REBALANCE_SIPS` / `BUILD_EMERGENCY_FUND` / `PAY_DOWN_DEBT` | Invest now / Invest in parts / Buy shares (part n of 3) / Change your SIPs / Build your emergency fund / Pay off costly debt |
| Exit actions `KEEP` / `TRIM_TO_TARGET` / `EXIT` / `REVIEW_THEN_EXIT` / `SWAP` / `WAIT_FOR_LONG_TERM` / `GLIDE_TO_DEBT` | Keep / Take some profit / Exit / Review, likely exit / Swap for better / Wait for lower tax / Move to safer money |
| Trigger states `ok` / `watch` / `triggered` / `unknown` | On track / Watch closely / Act now / Not enough data |
| Confidence `HIGH` / `MEDIUM` / `LOW` | How sure we are: High / Medium / Low |
| Sleeves | Indian shares / Global shares / Debt & deposits / Gold / Cash |

Reason codes are rendered with the same template keys as C17b's `copy/en.yaml`. **Export that YAML to JSON at build time and reuse it.** Never invent wording for a code; unknown codes show a neutral fallback, and a test logs them.

### 2. Home (`/`)
- **Greeting** + net worth (compact) + week change if available + "Plan health" score (from the plan if present).
- **"This week" card:** 0–3 actions from `/v1/actions`. Each shows the title, amount (INR formatted), one-line "why" (template), and buttons:
  - **Mark done** → `PATCH`
  - **How to** → a sheet with route-specific steps, e.g. Route A: "Open your MF app → search 'S&P 500 index fund' → start SIP ₹X"
  - **Why** → a sheet with params, explained
  - **No actions** → a positive state: "You're on plan. Nothing to do this week ✓"
- **Coming up:** the next 30 days from the maturity calendar.
- **Foundation alert** (if any gap) as a calm banner linking to Plan.

### 3. Plan (`/plan`)
- **Allocation:** target vs actual per sleeve (horizontal bars with text values + gap arrows), India vs global inside equity, a donut (lazy-loaded chart; text equivalent below). Use a small charting approach (inline SVG or a lazy-loaded lib). Keep the route chunk small.
- **"How much in debt?":** the debt answer sentence + current/target + locked amount + where new debt money should go (top debt pick with after-tax yield and tenure).
- **Global diversification:** global target %, the route chosen and why, warnings (TCS, Schedule FA, estate tax near the threshold) in plain words.
- **Foundation checklist:** 4 items with status and gap amounts (term cover gap, health cover, emergency fund, costly debt).
- **Your LIC/insurance policies:** per traditional policy, estimated return vs PPF/FD, and continue / paid-up / surrender values when known. The copy **always** says "compare options" and suggests confirming the surrender value with the insurer. **Never** "surrender now".
- **Goals list** with progress if available.

### 4. Ideas (`/ideas`)
- **Core picks for you:** the instrument categories behind this week's actions and route (e.g. "S&P 500 index fund (in INR)", "Nifty 50 index fund"). Categories only, no product selling.
- **Research ideas** (`/v1/research/ideas`): symbol, sector, confidence, valuation percentile in words, "part 1 of 3" if an action references it. Show the disclaimer ("Research, not personal advice") on this tab.
- **`/ideas/world` "World & your money":** the stress scenarios from `/v1/stress` as a list ("If Indian shares fall 25%: your money −₹X; with your target plan −₹Y"), plus geopolitical tilt notes if present ("What we changed and why", with source links if the plan includes evidence URLs).

### 5. Holdings and exit cards
From Money → a holding row → `/money/holding/:id`: quantity, value, P&L if a cost is known, long-term date, and the **Exit Card**: trigger rows with status icon + text + value/threshold, the action in plain words, the tax estimate if selling. Data comes from the plan's `exit_cards`.

### 6. Inbox (`/inbox`)
Briefs (subject, week, status), reminders and alerts, newest first. A brief opens `/inbox/brief/:week`, rendered as in-app HTML from `/v1/brief/{week}` (sanitised: render the structured brief JSON with React components, never `dangerouslySetInnerHTML` from the API).

### 7. Calm UI rules
No red/green tickers on Home. Daily P&L is hidden by default (setting in C23). Amounts in lakh/crore. Skeletons while loading; the cached snapshot shows instantly offline (C19 SW).

### 8. Tests (vitest + MSW)
- **Label map:** every action/exit/trigger/confidence/sleeve code has a label; unknown → fallback.
- **Home:** 0, 1 and 3 actions; mark-done updates; the "How to" sheet text for routes A/B/C; the no-action positive state.
- **Plan:** bars render text equivalents; the debt answer sentence; route warnings mapped; the insurance card never contains the word "surrender now" (assert absence), and does say "compare".
- **Ideas:** research list + disclaimer; the stress list numbers formatted.
- **Exit card:** rows and the plain-language action.
- **Inbox:** list + brief view; no `dangerouslySetInnerHTML` anywhere in `src/` (grep test).
- **Feature-detection:** a missing endpoint gives a "coming soon" state.
- Axe checks on Home, Plan, Ideas and Inbox.

## Out of scope
Push, install prompt, offline queue polish, settings (C23). Backend changes. If one is truly needed, list it in the result file instead of changing Python.

## Acceptance (inside `apps/web`, plus the repo-root Python suite)
1. `npm run typecheck`, `lint`, `test`, `build`, `budget` → pass (route-level code splitting)
2. `.venv\Scripts\python -m pytest` still green
3. The result file includes a screen-by-screen note + the gzipped size per route chunk
