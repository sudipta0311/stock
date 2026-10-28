# C27a: Household planning with per-member tax residence: member profiles, family asset location, per-member limits/tax, clubbing note, actions per member

**Requested by the product owner:** "There should be an option to add the spouse and plan the whole family's finances as a whole. I am an NRI in Denmark but my wife is in India; tax residence must be changeable per family member." Non-commercial use for family and friends.
**Plan:** `docs/UPGRADE_PLAN.md` §3.1 (household, members), §4 (Policy Engine), §3.5 (residency rulebook). Read `AGENTS.md` first. Builds on C05/C07 (`household_members`, assets with `member_id`), C11–C13b, C17a/b (personaliser, brief), **C26a** (residency model + `rules/residency.yaml` + residency-aware policy functions). Reuse C26a functions; don't duplicate them.

## Goal
One login manages a household with several members. Each member has their own tax residence and tax details. The plan is computed for the **household as a whole** (one net worth, shared goals, one target mix), while each recommendation says **which member** should act, chosen for that member's eligibility, limits and tax.

## Scope

### 1. Member profiles (new migration; RLS unchanged: rows stay under the owner's `user_id`)
- `household_members` gains:
  - `is_self bool` (exactly one per user; auto-created; the C26a profile residency migrates onto it)
  - `residency` JSONB (the C26a `Residency` model)
  - `tax` JSONB: `tax_regime`, `tax_slab_pct`, `birth_year`, `annual_income`, `employment_type`
  - `relationship` (`self`/`spouse`/`child`/`parent`/`other`)
- **API:**
  - `GET/POST/PATCH /v1/members` accept and return these fields
  - `PUT /v1/me/residency` (C26a) becomes a shim that updates the self member
  - `GET /v1/me` includes members with residency
- **Ownership:** every asset, account and holding lot belongs to one member (`member_id`, default = self). Joint assets: `joint: true` + `member_id` = primary holder; tax follows the primary holder (a note says so).
- Goals get optional `member_id` (null = household goal).

### 2. Household snapshot and per-member contexts (`wealth/policy/household.py`)
- `HouseholdSnapshot`: the C11 `WealthSnapshot` over all members **plus** a per-member breakdown (net worth, sleeves, liquidity, locked).
- `MemberContext` per member: residency, tax context (C13 `TaxContext`: slab, surcharge, the member's own FY realised gains), eligibility (C26a rulebook), and **limits with room left**:
  - PPF annual ₹1,50,000 minus this FY's contributions
  - LRS US$250,000 per FY (RI only), with TCS above the threshold per member
  - ASK cap (DK residents; from the C26a YAML)
  - SCSS (age ≥ 60, cap per individual)
  - Sukanya/other per-member caps as in the rulebook
- **Foundation (household):**
  - emergency fund for household expenses
  - term cover per **earning** member (that member's income × multiple + their loans)
  - health cover per member or a family floater
  - costly debt for any member

### 3. Family asset location (`wealth/policy/placement.py`, pure, deterministic)
Input: household gaps per sleeve (from the C11/C12 plan built on the household snapshot) + new money per member (`investing` prefs become per member: monthly SIP capacity and new money per member; default all on self).
- For each sleeve allocation, build candidate `(member, instrument_or_route)` pairs:
  - **eligibility** from the member's residency (e.g. an NRI can't open new PPF/NSC/SCSS; a US person avoids Indian MFs/ETFs; a DK resident avoids non-ABIS funds such as Indian MFs/ETFs for new money)
  - **limits/room** (PPF, LRS, ASK, SCSS)
  - **after-tax ranking** for the member: the slab-based debt picker per member, NRE vs NRO for NRIs, ASK flat rate for DK (use YAML values, `verified: false`)
  - **simplicity** order
- **Pick** the best member/instrument per chunk, spilling over to the next candidate when a limit fills. Prefer the member whose new money is available. Moving money between members is allowed only with a clubbing note (below).
- **Clubbing note (India):** when an action places investment in a spouse's name while the funds may come from the other spouse, attach `clubbing_check`. Source it from the Income-tax Act (spouse clubbing rule) in `rules/residency.yaml` or a new `rules/tax/india.yaml` entry with `source_url`, `verified: false`. Plain copy: "In India, income from money you give your spouse can be taxed as your income. Invest from their own income or savings, or keep records of a loan."
- **Per-member LTCG harvesting:** each Indian-tax-resident member has their own ₹1.25 L exemption. Run C13 `ltcg_harvest` per member on that member's lots. NRIs also get the exemption under Indian law per the rulebook; mark `needs_review` if unconfirmed.
- **Exit cards** use the lot owner's tax context.

### 4. Personaliser and outputs
- `personalise` builds the household plan and passes per-member contexts into the C12b action builder via `placement`. Every action has `member_id`, `member_name` and `member_relationship`. Sell actions carry the owner's tax estimate.
- Plan JSON adds `members: [{id, name, relationship, residency, net_worth, sleeves, limits_room, notes}]` and `placement_notes`.
- The weekly brief (C17b) shows the member on each action ("Anita: start ₹10,000/month SIP in an Indian index fund"; "You: buy a global UCITS ETF in your Aktiesparekonto") and a per-member net-worth line. Copy keys go in `copy/en.yaml`.
- **Disclaimer** copy key for this deployment: "Personal tool for family and friends. Information and education only, not investment advice." Used by the brief and the PWA when `settings.deployment_mode = "personal"` (env `DEPLOYMENT_MODE=personal`, the default). Keep the jurisdiction texts from C26a for a future public mode.

### 5. Tests (golden household; derivations in docstrings)
Household "Self NRI-DK + Spouse RI-IN":
- **Self:** salaried in Denmark, DKK income; Indian direct stocks + an old Indian MF; a UCITS ETF at Nordnet; an NRE FD.
- **Spouse:** resident India, new regime 20% slab; PPF with ₹50,000 contributed this FY; EPF; bank FD.
- **Goals:** child education INR (household); retirement DKK (self).
- **Expected:**
  - India equity new money → the spouse's Indian index fund (eligible, no Danish fund tax), with `clubbing_check` if the spouse's new money is 0 and funds come from self
  - PPF top-up → spouse, capped at ₹1,00,000 remaining room
  - global equity → self, UCITS via ASK up to the YAML cap, then a normal Nordnet account
  - debt → whichever member has the better after-tax result (state the computed yields)
  - no new PPF/NSC/SCSS for self
  - LTCG harvest computed separately per member
  - the self MF holding gets the `dk_fund_tax_review` note
- Changing the spouse's residency to **NRI-DK** (both now live in Denmark) → no new PPF/NSC/SCSS for anyone; India equity placement moves to **direct Indian stocks** (no Indian MFs/ETFs because of Danish fund taxation); the ASK cap applies **per person** (each spouse has their own ASK room); dk_* notes for both. Supported residences: RI, RNOR, NRI-DK only (scope decision 2026-10-28).
- RI single-member user → plan unchanged versus before (backward-compatible golden).
- API: create a spouse with residency, PATCH residency, plan recomputes (digest changes); isolation (another user can't see or affect members).

## Out of scope
PWA screens (C27b). Shared logins for a spouse (C27c).

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass; DB tests run; goldens updated only with `--update-golden` and diffs explained
2. `ruff` clean; migration up/down; OpenAPI + schemas refreshed; `copy/en.yaml` complete (the label test passes)
