# C26b: PWA: residency in onboarding and Settings (RI / NRI / RNOR), residency-aware copy and warnings across Money, Plan, Ideas

**Requested by the product owner:** let users change their profile between Resident Indian and NRI in the app; recommendations update accordingly.
**Plan:** `docs/UPGRADE_PLAN.md` §8.3 (onboarding asks "Where do you live and pay tax?" first), §2.3 (plain language), §3.5 (residency rules). Read `AGENTS.md` first. Builds on C20/C22/C23 and the C26a API (`/v1/me/residency`, plan `residency` + `residency_notes`, new labels in `copy/en.yaml`).

## Scope

### 1. Onboarding
A new first step, **"Where do you live and pay tax?"**, with three big choices:
- Resident in India
- NRI (living abroad)
- Returning NRI

For NRI and Returning NRI, ask:
- country (a list with flags/names; "Other")
- "Are you a US citizen, green-card holder or US tax resident?" (yes/no with a one-line why)
- "Do you plan to move back to India?" (No / Yes in year … / Not sure)

Save via `PUT /v1/me/residency`.

### 2. Settings → Profile → Residency
The same controls. On change, show a short confirm sheet: "Your plan will be recalculated for NRIs living in the UAE. Small-savings schemes like PPF and NSC won't be suggested for new money." Then invalidate the plan, actions and networth queries, and show the updated plan.

### 3. Residency-aware UI
- **Money:** add forms for instruments that aren't allowed as new money for the user's residency show a calm note ("As an NRI you can keep your existing PPF until maturity but can't open a new one") and still allow recording existing holdings.
- **Money → Add deposit:** offers NRE FD / FCNR(B) / NRO FD for NRIs (these map to the `fd` asset type with an `account_kind` detail if C26a added one; otherwise show them as labels on the bank FD form with a note). Don't change backend schemas here; if a field is missing, list it in the result file.
- **Plan:** a "For NRIs" (or "Returning to India") card listing `residency_notes` with plain labels. The global section title changes to "Shares where you live / global" for NRIs. The route shows the LOCAL vehicle text.
- **Ideas:** US persons see "Indian mutual funds and ETFs aren't suggested for US taxpayers (PFIC rules)" and the India-exposure alternatives.
- **Home:** the residency badge in the header (e.g. "NRI · UAE"), tappable to Settings.

### 3b. Denmark first, dual currency
- Country picker: **Denmark** first, then the other EU/EEA countries, then the rest.
- For NRIs, show net worth and key totals in **both INR and the residence currency** (DKK for Denmark), with the FX rate and its date, e.g. "₹1.86 Cr · DKK 1,310,000" (at about ₹14.2 per DKK). Use international grouping for DKK, never lakh/crore grouping. A Settings toggle picks the primary display currency (INR or local). Money stays INR-based in the API; conversion uses the FX rates the plan returns. If they're missing, show INR only with a note.
- The disclaimer shown in Ideas and the footer follows the jurisdiction copy keys from C26a (EU/MiFID wording for EU residents).
- Plan's "For NRIs" card shows the `dk_*` notes in plain Danish-context English, e.g. "Indian mutual funds are taxed every year in Denmark, even before you sell. Direct Indian shares are taxed only when sold.", with a link to the source URL from the rule.

### 4. Tests (vitest + MSW)
- The onboarding residency step and validation (country required for NRI).
- The Settings change → confirm → API call → queries invalidated.
- Add-form notes for PPF/NSC as NRI; the NRE/FCNR/NRO options appear only for NRIs.
- The Plan "For NRIs" card renders labels for every note code (none raw).
- The US-person PFIC message on Ideas.
- Dual-currency rendering for NRI-DK (INR + DKK with rate date); the EU disclaimer for DK.
- Axe checks on the new screens.

## Acceptance (inside `apps/web`, plus the Python suite)
1. `npm run gen:api` (and copy generation) leave no diff after commit-ready changes; `typecheck`, `lint`, `test`, `build`, `budget` pass
2. `.venv\Scripts\python -m pytest` still green
