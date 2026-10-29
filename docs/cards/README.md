# Implementation cards

The redesign in [../UPGRADE_PLAN.md](../UPGRADE_PLAN.md) is built one card at a time by Codex (`gpt-6-luna`, max effort). Each card is reviewed and tested before the next one starts. Rules for the coding agent are in [../../AGENTS.md](../../AGENTS.md).

Status: ⬜ not started · 🟦 in progress · ✅ accepted · 🔁 rework · ⏭ superseded

| Card | Title | Plan § | Status |
|---|---|---|---|
| C01 | Repo restructure, legacy move, sensitive-data scrub, new package scaffold, CI | 1.3, 7.2 | ✅ |
| C02 | Core helpers (Decimal money, Indian number parse/format, month-year dates) + asset/profile schemas + JSON Schema export | 3.2, 7.3 | ✅ |
| C03 | Scheme rates reference (YAML + loader) + valuation: FD, RD, PO TD, NSC, KVP | 3.4 | ✅ |
| C04 | Valuation: PPF, EPF/VPF, SSY, SCSS, POMIS, SGB, bonds, gold, NPS split, insurance (IRR), loans | 3.4, 4.11 | ✅ |
| C05 | Database: SQLAlchemy models, Alembic migration, Postgres RLS, session helper, isolation tests | 7.3 | ✅ |
| C06 | API skeleton: FastAPI app, settings, JWT auth (JWKS + dev mode), health, CORS, user bootstrap | 7.1, 8.7 | ✅ |
| C07 | API: Money Map CRUD (household, accounts, assets, insurance, liabilities, goals), audit history, net-worth summary | 3.1, 3.5 | ✅ |
| C08 | Importer 1: CAS parsers (NSDL port, CAMS/KFin, CDSL detect) → normalised holdings; synthetic encrypted-PDF fixtures | 3.3 | ✅ |
| C08b | Importer 1b: upload API (in-memory, no retention), staging + review + confirm → securities/holdings, `app_upsert_security` | 3.3, 7.4 | ✅ |
| C09 | Importer 2: Indian broker holdings/tradebooks (Zerodha, Groww, ICICI; Angel/Upstox unverified), column-mapping fallback + saved templates, FIFO lots, merge with CAS | 3.3 | ✅ |
| C09b | Importer 2b: US brokers (IBKR, Vested, INDmoney), USD securities keyed by symbol, FX-aware valuation | 3.3, 4.5 | ✅ |
| C10 | Market data: securities master, AMFI NAV, EOD prices, FX, gold, point-in-time store, provenance | 6 | ✅ |
| C11 | Policy 1: foundation check, risk profile, goal buckets, whole-wealth allocation, debt answer | 4.1–4.3 | ✅ |
| C12 | Policy 2: India/global equity split with bounded geopolitical tilts, global route selector (A–D), after-tax debt-instrument picker | 4.4–4.6 | ✅ |
| C12b | Policy 2b: core/satellite targets, buying rules → weekly action list (≤ 3), STP tranching | 4.7–4.8 | ✅ |
| C13 | Policy 3: India tax engine, tax-aware lot selection, LTCG harvest, rebalancing bands, insurance policy review | 4.10–4.11 | ✅ |
| C13b | Policy 3b: Exit Engine (exit cards, triggers, tax-aware timing) + personal stress tests | 4.9, 8.4 | ✅ |
| C14 | Research 1: ModelRouter (Foundry, flex, budgets, JSON outputs), Dossier schema/store, LangGraph skeleton + checkpointer | 5.3, 5.6, 5.7 | ⬜ |
| C15a | Research 2a: point-in-time fundamentals (SEC EDGAR; NSE XBRL if a free stable source exists), price factors, universes (survivorship-labelled) | 6, 4.7 | ⬜ |
| C15b | Research 2b: factor Screener (no LLM, sector-neutral), filings Sentinel (8-K item rules first, nano LLM for free text), research tools, sector data | 4.7, 5.2 | ⬜ |
| C15c | Research 2c: Analyst, Valuation (math in tools), Red Team (DeepSeek), Compliance gate, CIO sign-off → weekly Research Pack | 5.2, 5.3, 5.5 | ⬜ |
| C15d | Research 2d: golden-set eval harness, groundedness/schema/red-team metrics, CI gate | 5.5, 10 | ⬜ |
| C16a | Research 3a: Macro & Geopolitics agent: NSE index valuations, US Treasury yields, official RSS (PIB/RBI/USTR/Fed Register), bounded 2-source tilts with decay | 4.4, 5.2, 5.5 | ⬜ |
| C16b | Research 3b: ETF/index-fund selector, Rates/Rulebook watchers + human approval queue, Scorekeeper (paper portfolio, track record, calibration), CIO agenda | 4.7, 5.2, 5.8 | ⬜ |
| C17a | Alerts 1: investing prefs, Personaliser (weekly per-user plan → actions, pure code), weekly job, /v1/plan, /v1/actions, /v1/research/ideas | 4, 5.4 | ✅ |
| C17b | Alerts 2: weekly brief (templated, no LLM numbers), HTML+text email via HTTP provider, web push, daily reminders, urgent alerts, consent/unsubscribe, Inbox API | 9 | ✅ |
| C18 | Deploy: GitHub Actions workflows, `render.yaml`, env docs | App. D | ✅ (owner creates the Render services) |
| C19 | PWA 1: Vite + React + TS scaffold, design system, shell + bottom nav, service worker, typed API client, OIDC + dev token, Indian formatting | 8.1, 8.7 | ✅ |
| C20 | PWA 2: onboarding + Money Map checklist + all mini-forms + Indian number input + autosave | 3.1, 8.3, 8.4 | ✅ |
| C21 | PWA 3: import flow (upload, share target, progress, review & merge) | 3.3, 8.4 | ✅ |
| C22 | PWA 4: Home, Plan, Ideas (World & your money, stress tests), holding Exit Cards, Inbox | 8.4 | ✅ |
| C23 | PWA 5: offline cache, cold-start UX, install prompt, web push subscription, settings/export/delete | 8.1, 7.4 | ✅ |
| C26a | Residency backend: RI/NRI/RNOR profile, residency rulebook (eligibility, NRE/FCNR/NRO, PFIC, TDS), residency-aware split/routes/debt picks, multi-currency goals | 2.2, 3.4, 3.5 | ✅ |
| C26b | Residency in the PWA: onboarding + Settings switch RI/NRI/RNOR, residency-aware notes in Money/Plan/Ideas, NRI deposit options | 8.3 | ⏭ superseded by C27b |
| C27a | Household planning: members with individual tax residence + tax details, household snapshot, family asset location (eligibility/limits/after-tax per member), clubbing note, per-member LTCG harvest, actions per member | 3.1, 4 | ✅ |
| C27b | Household PWA: onboarding 'who are we planning for', per-member residency in Settings, Family/member switcher, who-acts on actions, who-holds-what, DKK+INR (supersedes C26b) | 3.1, 8 | ✅ |
| C27c | Shared household access (core): spouse logs in with own account, both see and edit the same household (partner/viewer), invites, bring-my-data merge, RLS by household grant | 7.3, 7.4 | ✅ |
| C29a | Invest from Denmark: curated UCITS/ABIS ETF list, Aktiesparekonto-first account order, US stocks (no US-listed ETFs), India via UCITS ETF or NRI shares, broker how-tos, EU/US prices | 4.5, 4.7, 3.4 | ✅ |
| C29b | Indian funds check from CAMS/KFintech CAS: sub-categories, direct vs regular, too many funds, lagging funds, SIP detection, 12-month early warning, least-tax fix | 3.3, 4, 4.10 | ✅ |
| C28 | Simplicity pass: backend highlights (≤ 3 per screen), banned-jargon + sentence-length tests, one primary action per screen, collapsed details, 5-tap onboarding, rounded numbers | 2.3, 8 | ⬜ |
| C24 | End-to-end tests (Playwright), accessibility and performance checks | 10 | ⬜ |
| C25 | Docs, cleanup, final plan sync | — | ⬜ |

## Follow-ups noted during review
- (Deploy prep, reviewer) Migration 0002 was edited for managed Postgres. It no longer names SUPERUSER/REPLICATION in ALTER ROLE (it verifies them instead) and only REVOKEs an owner membership that exists. The end state is identical, so existing databases are unaffected. Neon `wealth` (owner's Frankfurt project) is migrated to 0011. → C18 adds a non-superuser CI owner so this class of bug is caught.
- (C27c review) `DELETE /v1/households/leave` by the owner returns 404 instead of 409 (owner leave returns 404). The request is refused either way; the PWA (C27b) must not show Leave to owners. → C27b
- (C18 review) The Google-exchange rate limiter keys on `request.client.host`, and uvicorn runs with `--forwarded-allow-ips '*'`, so a caller can vary `X-Forwarded-For` to dodge it. That is low risk (a code still needs a real Google sign-in), but key on the right-most hop Render adds, or trust only Render's proxy range. Also run `render blueprints validate` when the Render CLI is available. → C25
- (C27c review) `wealth_app` has UPDATE/DELETE on `audit_log`. Make the household audit append-only: revoke UPDATE/DELETE from `wealth_app` and check that account deletion cascades still work (FK cascades run as the table owner). → C25
- (C27c review) The audit trigger skips rows when the session setting `app.household_deleting_user` equals the tenant. Any session can set it, so it's an audit bypass if the API is ever compromised. Replace it with a check that needs no session setting (cascades run after the users row is gone, so the existing NOT EXISTS users check may be enough), with a test. → C25
- C04: EPF accrual uses actual/365 days; EPFO credits interest on monthly running balances at rate/1200 (~0.02% difference). Align in C25.
- C03: post-office TD rates per tenure are empty (India Post page shows only a range); historical small-savings periods not yet loaded. Rates watcher (C16) + human approval.
- C05: residual risk accepted: SQL that can call set_config(app.user_id) with another user's UUID would read that tenant. Mitigated by parameterised SQLAlchemy only; owner-role escalation is closed (API logs in as wealth_api).
- C07: manual assets needing a market price (gold physical/ETF/SGB) store valuation=null and silently drop out of /v1/networth. Add per-asset `valuation_status` and networth `unvalued_assets` (do in C08b); wire PriceProvider from market data in C10.
- GitHub history rewritten and force-pushed 2026-10-28 (user-approved). Old commits 40375c0 (main), 6bf7538 (streamlit-multi-agent), 4065438 (codex/streamlit-cloud-ready) are still served by SHA until GitHub Support purges cached views.
- C20: API has no GET for the Money Map checklist (PWA keeps it locally) and no consent endpoints yet (C17b). Add GET checklist to C17b.
- C26a: FX refresh fails as a whole if one ECB currency request times out (seen once); fetch all currencies in one ECB multi-series request (D.INR+DKK+USD+...EUR) and isolate per-currency failures. Do in C25.

**Run order for the remaining cards:** C27a → C27c → C23 → C27b → **C18 (moved up: first deploy so the owner can try it)** → C29b → C09 → C09b → C29a → C28 → C14 → C15a–d → C16a/b → C24 → C25. Each later card ships through the same deploy. C29b (Indian funds) needs only the existing CAMS parser. C29a (Denmark buying) needs C09b's Danish broker imports. C28 (simplicity) runs after every screen exists. C24 (e2e) checks the simplified flows.
