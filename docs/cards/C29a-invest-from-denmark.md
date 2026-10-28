# C29a: Investing from Denmark: which US stocks and EU-listed ETFs to buy, in which account, at which Danish broker

**Requested by the product owner:** "I will invest in US stocks and ETFs from Denmark. For Denmark the app should guide me to the US ETFs/stocks or EU ETFs which I can buy from Denmark."
**Plan:** `docs/UPGRADE_PLAN.md` §4.5 (global routes), §4.7 (ETF selection is deterministic), §3.4 (residency). Read `AGENTS.md` first, including the **UX rules**. Builds on:
- C26a: `rules/residency.yaml` `denmark:` section (ABIS, Aktiesparekonto (ASK), share income, PRIIPs, PAL)
- C27a: members with their own residency, placement engine
- C09b: Danish broker imports; `securities.symbol/exchange/domicile`; `app_upsert_security_by_symbol`
- C12: routes, global split; the research pack ideas (US market) when present

**Feature-detect** C15/C16 research outputs (they may not exist yet): without them, the guide still works with the curated ETF list and no stock ideas.

## Goal
For a household member who is tax-resident in Denmark, the plan's "Abroad" money turns into a short, concrete buy list. Each line says what to buy, how much (in DKK), in which account, and at which broker:
- an EU-listed (UCITS) ETF on the Danish ABIS list
- or a US stock
- **never** a US-listed ETF: EU retail investors can't buy those (no PRIIPs KID)

Code decides every amount. Danish tax facts stay `verified: false` with a skat.dk source until the product owner approves them.

## Scope

### 1. Curated universe (`rules/eu_etfs.yaml`, versioned rule data + JSON Schema in `schemas/`)
- **Per ETF:**
  - name, ISIN, tickers per exchange (Xetra / Euronext / LSE / Nasdaq Copenhagen) with the trading currency
  - domicile (expect IE/LU), index, exposure
  - TER %, accumulating/distributing, fund size band
  - `abis_listed` (+ the ABIS list year)
  - `ask_eligible`, `available_at: [nordnet, saxo, danske_bank]`
  - `us_equivalent` (the US ETF this replaces, e.g. an S&P 500 or all-world fund)
  - `source_url`, `last_checked`, `verified: false`
- **Exposures to cover:**
  - `global_all_world`, `global_sp500`, `global_nasdaq100`, `global_developed`, `global_em_ex_india` (if one exists on the ABIS list)
  - `india_equity`: UCITS India ETFs, see §3
  - `gold` (only if an ABIS-listed physical-gold ETC/ETF is eligible; else omit it with a note)
- 2–4 candidates per exposure, from the **issuers' official pages and SKAT's published ABIS list**. Fetch them live during this card (network is allowed); record the source per row. If the ABIS list is published as a spreadsheet, add `python -m wealth.rates.abis import <file>`. It sets `abis_listed` from the official file and keeps provenance. Test it on a small synthetic excerpt.
- **ISINs and tickers of public funds are not personal data. Never guess them.** A row you can't confirm from a source is left out.

### 2. Denmark account order (`wealth/policy/denmark.py`, pure)
For a DK-resident member with `new_money_dkk` for the "Abroad" bucket:
1. **ASK first**, up to the remaining room (deposit cap from `denmark.ask`, minus that member's ASK deposits from imports or the member's "already in ASK" pref). Only `ask_eligible` instruments.
2. **Ordinary account (frie midler) next.**
   - ABIS-listed ETFs: share income, taxed yearly on the gain (lager).
   - Individual shares: taxed when sold, which defers tax. Note this in Learn more.
   - Non-ABIS funds are **never** suggested (capital income).
3. **Pension (PAL)** only when the member says they have a Danish pension account they can invest in (an optional pref, default no). One Learn-more note, no amounts.

**Output:** an ordered list of `(account, instrument, amount_dkk, reason_codes)`. Amounts are Decimal and rounded to whole DKK at presentation (≥ DKK 1,000 → nearest 100).

### 3. Instrument choice (`wealth/policy/denmark_picks.py`, pure, rules in `rules/eu_etfs.yaml` `selection:`)
- **Global core:** choose by exposure from the plan's global split, then this filter and ranking:
  - filter: `abis_listed` + available at the member's broker (`brokers` pref; default `nordnet`) + `ask_eligible` when the account is ASK
  - rank: lower TER, then accumulating, then a bigger fund, then a listing in DKK/EUR before USD/GBP (fewer currency fees)
- **US stocks (optional, inside the "own stock picks" share):**
  - only when the member opted in, and only from research-pack ideas with `market=US` and a passing dossier (feature-detected)
  - code caps: total stock picks ≤ the satellite % pref (default 10%, max 25%); each stock ≤ 5% of investable wealth
  - per-buy minimum DKK 2,000 (rule data) to keep broker fees sensible
  - Learn-more notes:
    - US dividend withholding 15% with a W-8BEN, credited in Denmark
    - the US-situs total vs `estate_tax_us_domiciled_assets`; mention the Denmark–US estate-tax treaty as a `verified: false` note
- **India exposure for a DK-resident member:** compare two vehicles deterministically:
  - (a) direct Indian shares through an NRI account (the existing `in_equity_direct_core`)
  - (b) an ABIS-listed UCITS India ETF, ASK-first
  - Default to **(b)** unless the member's pref "I have an Indian NRI share account" is on. Explain it in one plain sentence ("Simplest from Denmark: one Indian fund in your Aktiesparekonto").
  - Keep C26a's note that Indian mutual funds are taxed less favourably in Denmark.
- **Honest comparison with the US twin** (details/Learn more only, code-computed): for each picked UCITS ETF with a `us_equivalent`, show the estimated yearly cost difference:
  - (a) the TER difference
  - (b) dividend withholding lost inside an Irish fund: 15% × the index dividend yield, from rule data with a source; a Danish investor in a US-domiciled fund could credit that 15% in Denmark
  - set against the advantages:
    - the ETF can be bought at all
    - it's allowed in an Aktiesparekonto at 17% tax, against capital-income tax for non-ABIS funds
    - no US estate-tax exposure
    - no W-8BEN needed
  - Use ABIS facts and rates from the rules (`verified: false` until approved). One plain summary line, e.g. "Costs about 0.2% a year more than the US version, but can save far more in tax."
- **US-listed ETFs the member already holds** (e.g. bought before moving): keep them, but never add to them. One note: "You can't add to this US fund from Denmark. We suggest its EU twin: <name>." This uses `us_equivalent`.

### 4. Prices for the universe
- Investigate free EOD sources whose terms allow personal, non-commercial use, for (i) the curated UCITS ETFs and (ii) US stocks. Candidates: issuer NAV files, an exchange's free EOD data, or a free-tier API key from an env var (add it to `.env.example`; GitHub secret in C18).
- Implement **one** working source as `wealth/marketdata/sources/<name>.py`, wired into `python -m wealth.marketdata.refresh --only eu_etfs,us_stocks` (held symbols + the curated list only).
- Fallback order: broker-statement prices (C09b), then `rules/manual_prices.yaml`. Every price carries its source and date; the UI shows "as of".
- Tests use recorded fixtures.

### 5. Integration
- **Plan:** for DK members, `GET /v1/plan` adds `abroad_buy_list` (max 3 lines per member):
  - `{member_id, account: ask|ordinary|pension, broker, instrument {name, ticker, exchange, isin, currency}, amount_dkk, amount_inr, reason_codes, how_to_key}`
  - `details` holds the alternatives considered and why they were not picked
- **Weekly actions:** new kind `BUY_ABROAD` (the "Abroad" share of "This week" for DK members), ≤ 3 actions overall as before. The email uses the same copy keys.
- **How-to copy** per broker (`copy/en.yaml`, plain, ≤ 5 short steps). Example: "Open Nordnet → Aktiesparekonto → search the fund code → buy DKK 5,000". The fund code is shown with `<Term>` ("fund code (ISIN)").
- **PWA** (small, follows the UX rules):
  - Ideas shows a "Buy from Denmark" card: up to 3 lines, each with **How?**; alternatives and tax notes are behind "Why?"
  - Settings → Fine-tune gets three member prefs: broker (Nordnet / Saxo / Danske Bank / Other), "I have an Indian NRI share account" (default off), "I have a Danish pension I can invest" (default off)
- **Highlights (C28):** at most 1 Denmark note per screen. Default: "We use your Aktiesparekonto first: it has the lowest tax."

### 5b. Follow-ups from the C27a review (do first)
- **Gold for a DK member:** placement currently gives the DK member a `gold_etf` SIP, but an Indian gold ETF is an Indian fund (non-ABIS) for Danish tax.
  - Treat `gold_etf` like `in_etf` under the DK override (new: not allowed, with the `dk_fund_tax_review` note).
  - Place gold with an RI member through the placement rules (clubbing note), or with an EU-listed gold product only if §1 found an eligible one. Otherwise carry it forward with a note.
- **Wrong condition key:** an RI member's `eligibility_conditions.ucits_global_etf` shows `eu_priips_kid`. For RI it should be an LRS note (e.g. `ucits_via_lrs_broker`, a new copy key).
- **Account type:** `params.account_type` values are `ASK` / `ORDINARY` (from C27a). Add the broker from the new pref as `params.broker`; don't overload the account type.

### 6. Tests (no network)
- ASK room left → ASK first, then ordinary; ASK full → ordinary only.
- Non-ABIS and US-listed ETFs are never picked; a held US ETF gets the "EU twin" note.
- The ranking order (TER, accumulating, size, currency) and the broker filter.
- US stock ideas: opt-in only, caps per stock and in total, per-buy minimum, absent research pack → no stock lines.
- India vehicle default (b) and the switch to (a) with the pref.
- An RI spouse's plan is unchanged (Indian routes).
- Rounding and DKK + INR amounts; `verified: false` notes surface as "please check" in details.
- API contract plus the `BUY_ABROAD` action rendering in email and PWA; the banned-terms scan passes (if the C28 test exists).

## Acceptance
1. `pytest` (full) + `ruff` pass; OpenAPI and JSON Schemas refreshed; PWA `typecheck`, `lint`, full `test`, `build`, `budget` pass
2. The result file lists every ETF row with its source URL, every rule still `verified: false`, and the price source chosen (with its terms link)
3. A worked example in the result file: an NRI-DK member with DKK 10,000 new money and DKK 50,000 ASK room, plus the same member with a full ASK
