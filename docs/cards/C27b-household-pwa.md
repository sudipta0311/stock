# C27b: PWA: household members with individual tax residence, family view of plan and actions (replaces the C26b screens)

**Requested by the product owner:** add a spouse, plan the family as a whole, change tax residence per member. Primary household: self NRI in Denmark, spouse resident in India. **This card supersedes C26b.** Implement C26b's residency UI, Denmark-first and dual-currency requirements here, at member level.
**Plan:** `docs/UPGRADE_PLAN.md` §3.1 (household), §8.2–8.4. Read `AGENTS.md` first. Builds on C19–C23 and the C26a/C27a API (`/v1/members` with residency + tax; plan `members`, `placement_notes`, actions with `member_id`/`member_name`).

## Scope

### 0. Simplicity (AGENTS.md **UX rules**; the C28 simplicity pass builds on this)
- **Onboarding asks only the minimum:**
  - per adult: name, relationship, and the three-choice tax residence
  - everything else (tax regime, bracket, birth year, income, return year) is optional, behind "Add more details (optional)", and defaults sensibly
- **Plain words on screen:**
  - "sleeves" becomes "Indian shares / Safe money / Abroad"
  - "LRS used" becomes "Sent abroad this year"
  - "ASK room" becomes "Room left in your Aktiesparekonto"
  - "placement" becomes "whose name to invest in"
  - "clubbing" becomes "Income from money you give your spouse is taxed as yours"
- **Family is the default view.** The member switcher is a single quiet chip, not tabs. Per-member breakdowns and limits are in collapsed "Details" sections on Plan.
- **Residency confirm sheet:** one sentence of impact + **Confirm** (the one primary action) + Cancel.

### 1. Onboarding
- **Step "Who are we planning for?"**:
  - "Just me" / "Me and my spouse" / "My family" (+ add children/parents later)
  - for each adult: name, relationship, **tax residence** (three choices: **Resident in India** / **NRI living in Denmark** / **Returning NRI**, plus a planned return year for NRIs; no country picker or US-person question for now), tax regime, slab, birth year, annual income (optional)
- Consent and the rest stay as in C20.

### 2. Settings → Household
List members. Each member page edits: name, relationship, **tax residence** (with the C26b confirm sheet: "Plan will be recalculated: Anita is resident in India, so PPF, NSC and Indian mutual funds can be suggested in her name"), tax details, and per-member investing prefs (monthly SIP capacity, new money this week). Removing a member with assets → the API's 409 with guidance.

### 3. Family views
- **Header member switcher** chip: **Family** (default) / each member. Money, Plan and Home filter by member; Family shows totals with a per-member breakdown row.
- **Home "This week":** each action shows **who** acts (avatar initial + name, "You" for self) and member-specific "How to" steps (e.g. Anita: Indian MF app; You: Nordnet ASK).
- **Plan:** the household target mix, plus a **"Who holds what"** card with per-member net worth and sleeves. **Limits** per member: PPF room, LRS used, ASK room. Placement notes in plain words, including the **clubbing note** where present.
- **Money add forms:** "Whose is this?" member selector (default = the currently filtered member or self). Eligibility notes use **that member's** residency (C26b rules: e.g. no new PPF for an NRI member, but recording existing ones is allowed).
- **Dual currency:** amounts show INR and, for NRI members/households with a DK member, DKK with the FX rate and date (international grouping for DKK). A Settings toggle sets the primary display currency.
- **Disclaimer:** the personal-mode copy key ("Personal tool for family and friends…").

### 3b. Shared household access (C27c API)
- **Settings → Household → "Share with your partner":** enter their email, pick a role (Partner = full access; Viewer = read-only), create the invite. Show the share link with Copy/Share buttons (Web Share API), explained in one sentence. List pending invites with revoke, and people with access with role change/revoke.
- **`/join?token=...` route:** if not signed in → login, then return. Show "Join <owner name>'s family plan as Partner". If the user already has their own data, offer the checkbox **"Bring my existing investments into the family plan"** (default on, explained). Accept → switch to that household.
- **Household switcher:** in the header when the user can open more than one household (own vs shared), plus the role badge. Viewers see edit controls disabled with "View only".
- The API client sends `X-Household` from the selected household (persisted in session storage). On a 404 for a household, fall back to own and show a toast ("Access to this family plan was removed").
- Actions show "done by <name>" when a partner completed them.

### 4. Tests (vitest + MSW)
- The onboarding household step (two adults, different residences).
- The member residency change → confirm → plan invalidated.
- The member switcher filters Money/Plan/Home.
- Actions render member names and member-specific How-to text.
- "Who holds what" and the limits room.
- The clubbing note rendered from its code (no raw codes).
- The asset add form uses the selected member's eligibility notes.
- DKK + INR rendering.
- The personal disclaimer.
- Axe on all new screens.

## Acceptance (inside `apps/web`, plus the Python suite)
1. Generated API/copy in sync; `typecheck`, `lint`, `test`, `build`, `budget` pass (new screens code-split)
2. `.venv\Scripts\python -m pytest` still green
