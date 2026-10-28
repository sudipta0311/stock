# C26a: Residency (RI / NRI / RNOR): profile, residency rulebook, and residency-aware recommendations (allocation, routes, debt picks, PFIC rules, tax/TDS notes, multi-currency goals)

**Requested by the product owner:** users must be able to switch their profile between Resident Indian and NRI, and recommendations must update accordingly.
**Plan:** `docs/UPGRADE_PLAN.md` §2.2 (NRI personas, previously Phase 6, now in v1), §3.4 (India vs global split by residency/currency), §3.5 (residency rulebook matrix), §3.7 (debt sleeve by residency), §4.1 (emergency fund 9–12 months for visa holders), §4.5 (routes; LRS/TCS apply to residents only). Read `AGENTS.md` first. Builds on C11–C13b policy, C17a personaliser, C17b brief.

## Goal
Residency is a first-class input. Changing it in the profile changes the eligible instruments, the India/global split, the global route, the debt picks, the tax notes and the warnings, and `/v1/plan` reflects it immediately. On-demand recompute already happens when the inputs digest changes.

## Primary users: Indian NRIs living in Denmark (EU)
The product owner confirmed the main users live in **Denmark**. Design and test for `status=NRI, country=DK` first, while keeping RI and the other countries working. **Never fill Danish rules from memory.** Fetch official sources (skat.dk / Skattestyrelsen, Finanstilsynet, EU PRIIPs regulation text) during the run, record `source_url`, and keep `verified: false` (plus `needs_review: true` where unconfirmed). The YAML must make it easy for a human to correct values.

Denmark/EU rules to encode in `rules/residency.yaml` (data + note keys; **don't compute Danish tax** in v1, only flag and explain):
- **EU retail product access (PRIIPs/KID):** EU retail investors generally can't buy US-domiciled ETFs at EU brokers. For `country` in the EU/EEA, the LOCAL route vehicle is `ucits_global_etf` (Ireland/Luxembourg domiciled, EUR/DKK listing), never `us_broad_market_etf`. Individual US stocks remain possible.
- **Danish taxation of investments** (note codes with plain explanations in `copy/en.yaml`):
  - `dk_share_income`: realised gains on directly held shares are share income (progressive share-income rates; fetch the current rates and threshold)
  - `dk_abis_list`: investment companies/ETFs on SKAT's ABIS list are taxed annually on unrealised gains as share income
  - `dk_non_abis_capital_income`: funds **not** on the ABIS list, which normally includes **Indian mutual funds and Indian ETFs**, are taxed annually on unrealised gains as capital income, usually less favourable. Verify from skat.dk and record the source.
  - `dk_aktiesparekonto`: the share savings account (ASK) with a flat rate and a deposit cap; fetch the current cap and rate
  - `dk_pension_pal`: pension savings taxed under PAL
  - `dk_worldwide_income`: Danish residents are taxed on worldwide income; the India–Denmark tax treaty gives credit for Indian tax; NRE/FCNR interest that's exempt in India is still taxable in Denmark
- **Policy consequences for `country=DK`:**
  - India exposure for new money defaults to **direct Indian equities** (`in_equity_direct_core`) instead of Indian MFs/ETFs (flag `dk_non_abis_capital_income`). Existing Indian MF holdings get the info note `dk_fund_tax_review` (no auto-sell).
  - The global sleeve uses UCITS ETFs. Add the note "check the fund is on SKAT's ABIS list" (`dk_abis_check`).
  - Suggest the ASK as the first wrapper for new EU/global equity money up to its cap (`dk_use_ask_first`, informational). The cap comes from the YAML, `verified: false`.
  - Debt/cash for DK: local DKK/EUR savings and deposits (a `local_bank_deposit` category, yield from `rules/debt_yields.yaml`, `verified: false`), plus NRE/FCNR (if DKK isn't an accepted FCNR currency per official RBI/bank sources, offer EUR) with the Danish-tax note.
- **Goal currencies:** add `DKK` (and keep `EUR`) via the ECB EUR cross (DKK/INR = (INR per EUR) / (DKK per EUR)).
- **Disclaimers by jurisdiction** (copy keys used by the brief and the PWA): for EU/EEA residents, "Information and education only. Not investment advice under MiFID II; we are not authorised by Finanstilsynet." For RI, the existing SEBI wording.
- **Matrix test additions:** `NRI-DK` must:
  - use the UCITS vehicle
  - never propose Indian MF/ETF buys
  - show `dk_*` notes
  - convert a DKK goal
  - use the EU disclaimer

## Scope

### 1. Profile (new migration; RLS unchanged)
Add `Residency` to the profile (Pydantic + DB JSONB or columns):
- `status`: `RI` | `NRI` | `RNOR` (default `RI` for existing users)
- `country`: ISO code, required for NRI/RNOR; from a supported list `DK, SE, NO, FI, DE, NL, IE, FR, US, AE, SA, QA, KW, OM, BH, GB, SG, CA, AU, NZ, OTHER` (DK first: primary market)
- `us_person: bool` (US citizen / green card / US tax resident; must be true when country = US and status = NRI unless the user overrides)
- `planned_return_year: int | null`
- `settle_plan`: `india` | `abroad` | `undecided`

API: `GET/PUT /v1/me/residency`, also included in `GET /v1/me` and the plan inputs digest. Validation: country is required for NRI/RNOR; `planned_return_year` ≥ the current year.

### 2. Residency rulebook: `rules/residency.yaml` (data, not code)
Every entry has `source_url`, `verified: false`, `last_checked`. **Fetch official sources during this run** (RBI/FEMA master directions, India Post/NSI FAQs, PFRDA, Income Tax Dept, IRS for PFIC/FBAR). Never fill from memory. Where no official confirmation is found, keep the entry but set `needs_review: true` and say so in the result file.
- **Eligibility** per instrument for **new** investments and for **existing** holdings (`new: allowed|not_allowed|conditional`, `existing: continue|continue_no_extension|must_close|conditional`, `note`). Instruments:
  - `ppf`, `nsc`, `kvp`, `scss`, `pomis`, `po_td`, `po_rd`, `ssy`, `epf`, `vpf`, `nps`
  - `bank_fd` (resident), `nre_fd`, `fcnr_b`, `nro_fd`
  - `in_equity_direct` (via PIS), `in_mf`, `in_etf`, `sgb`, `rbi_retail_direct_gsec`, `arbitrage_fund`, `liquid_fund`
- **Tax/TDS notes** per residency: NRI TDS on capital gains (listed equity STCG/LTCG rates, debt at slab-equivalent TDS), NRE/FCNR interest exemption in India, NRO interest TDS. Each has a short user-facing note key in `copy/en.yaml`.
- **US-person rules:** Indian MFs/ETFs → `pfic_avoid`. Indian accounts → `fbar_fatca_reporting` note. NRE/FCNR interest taxable in the US → note.
- **Resident-only rules** to switch off for NRIs: LRS/TCS, Schedule FA.
- **RNOR:** a note about the window where foreign income isn't taxed in India (length per official source).

### 3. Policy changes (pure, tested)
- **Geography split (C12 `split_equity`):** a `rules/global.yaml` section by residency:
  - RI: current table
  - NRI with `settle_plan=abroad` (or undecided with no return year): India share of equity 15–30% (midpoint by profile)
  - NRI country in the Gulf (AE, SA, QA, KW, OM, BH): 50–70% India
  - NRI with `planned_return_year` within 7 years: India share glides linearly from the abroad level to the RI level by the return year
  - RNOR: RI table
  
  For NRIs, "equity_global" means **the user's residence-country/global market**, and the sleeve label changes in copy ("Shares where you live / global").
- **Routes (C12 `select_route`):** for NRI/RNOR, LRS routes don't apply. Return route `LOCAL` with vehicle by country: `us_broad_market_etf` for US persons; `ucits_global_etf` for non-US NRIs, with an estate-tax note for US-domiciled assets. No TCS or Schedule FA warnings. Keep RI behaviour unchanged.
- **Debt picker (C12):**
  - Filter by residency eligibility (new money only).
  - Add catalogue entries `nre_fd`, `fcnr_b`, `nro_fd` with tax treatment: NRE interest exempt in India; FCNR exempt in India with no INR risk (preferred for foreign-currency goals); NRO taxed at slab/TDS. For US persons, add a `us_tax_applies` note on NRE/FCNR.
  - Yields for these come from `rules/debt_yields.yaml` (add rows, `verified: false`).
- **PFIC (US persons):**
  - Never propose Indian MF/ETF purchases (core India picks become `in_equity_direct_core` = a basket note of large-cap direct stocks, or `us_listed_india_etf`; the vehicle is chosen by rules).
  - Holdings in Indian MFs/ETFs get an info note `pfic_holding_review` (no auto-sell).
- **Foundation (C11):** emergency months for NRI with `country != IN` and employment `salaried` = 9 (the visa-dependency default, configurable).
- **Goals:** support currencies `INR, DKK, EUR, USD, SEK, NOK, GBP, AED, SGD, CAD, AUD`. Convert with `fx_rates`; extend the C10 ECB adapter to store `X/INR` for these via the EUR cross (same method as USD/INR; add fixtures). A missing FX → a clear error note, never a silent 1:1.
- **Tax engine (C13):** for NRIs, `estimate_sale_tax` adds `tds_estimate` (equal to the tax for listed equity) and notes `tds_deducted_at_source`, `residence_country_tax_may_apply`. No US tax computation, only the note.
- **Personaliser (C17a):** pass residency everywhere. The weekly plan gets `residency: {...}` and `residency_notes: [codes]`. The brief (C17b) shows up to 2 residency notes in a small "For NRIs" section.

### 4. Tests
- A matrix test over (RI, NRI-US us_person, NRI-AE, NRI-GB, RNOR) for one fixed synthetic user. Assert the eligible debt instruments, the route, the India share of equity, and the absence of `in_mf`/`in_etf` buys for US persons. Put the expected values in the docstring.
- PPF/NSC/SCSS never proposed as new money for NRIs; existing PPF shows `continue_no_extension` (if that's what the source supports).
- NRE vs NRO after-tax ranking for an NRI (NRE wins at equal yield).
- A glide with `planned_return_year` 3 years out.
- AED goal conversion via the EUR cross fixture.
- Switching residency via `PUT /v1/me/residency` changes `/v1/plan` (digest changes; on-demand recompute).
- Golden plan unchanged for the RI user (backward compatible).

## Out of scope
PWA screens (C26b). Computing residence-country taxes.

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass; DB tests run; no network in tests
2. `ruff` clean; migration up/down; OpenAPI + schema exports refreshed; `copy/en.yaml` has every new code
3. The result file lists each residency rule with its `source_url` and which are `needs_review`
