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
| C14 | Research 1: ModelRouter (Foundry, flex, budgets, JSON outputs), Dossier schema/store, LangGraph skeleton + checkpointer | 5.3, 5.6, 5.7 | ✅ |
| C15a | Research 2a: point-in-time fundamentals (SEC EDGAR; NSE XBRL if a free stable source exists), price factors, universes (survivorship-labelled) | 6, 4.7 | ✅ |
| C15b | Research 2b: factor Screener (no LLM, sector-neutral), filings Sentinel (8-K item rules first, nano LLM for free text), research tools, sector data | 4.7, 5.2 | ✅ |
| C15c | Research 2c: Analyst, Valuation (math in tools), Red Team (DeepSeek), Compliance gate, CIO sign-off → weekly Research Pack | 5.2, 5.3, 5.5 | ✅ |
| C15d | Research 2d: golden-set eval harness, groundedness/schema/red-team metrics, CI gate | 5.5, 10 | ✅ |
| C16a | Research 3a: Macro & Geopolitics agent: NSE index valuations, US Treasury yields, official RSS (PIB/RBI/USTR/Fed Register), bounded 2-source tilts with decay | 4.4, 5.2, 5.5 | ✅ |
| C16b | Research 3b: ETF/index-fund selector, Rates/Rulebook watchers + human approval queue, Scorekeeper (paper portfolio, track record, calibration), CIO agenda | 4.7, 5.2, 5.8 | ✅ |
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
| C30 | Loans: balances roll down monthly, record extra payments/rate changes/statement balances, payoff date and interest left, 'pay extra or invest' decided by code, EMI reminders, loan detail screen | 3, 8 | ✅ |
| C31 | Account menu (who's signed in, sign out), detailed help centre with contextual '?' links, a guided first visit, intuitiveness pass | 8 | ✅ |
| C32 | Stay signed in: app sessions (30-day sliding, 90-day max) with rotating refresh tokens, reuse detection, sign out everywhere, signed-in devices list | sec | ✅ |
| C33 | Statements from Gmail: own-account Apps Script posts CAMS/KFintech/NSDL/CDSL PDFs with a revocable upload key; password prompt or encrypted remember; same review flow; manual upload stays the fallback | import | ✅ |
| C29b | Indian funds check from CAMS/KFintech CAS: sub-categories, direct vs regular, too many funds, lagging funds, SIP detection, 12-month early warning, least-tax fix | 3.3, 4, 4.10 | ✅ |
| C28 | Simplicity pass: backend highlights (≤ 3 per screen), banned-jargon + sentence-length tests, one primary action per screen, collapsed details, 5-tap onboarding, rounded numbers | 2.3, 8 | ✅ |
| C28b | Visual polish: type scale, spacing, app shell, highlight callouts, empty states, list rows, rounded summaries; fixes from the real-phone test | 8 | ✅ |
| C24 | End-to-end tests (Playwright), accessibility and performance checks | 10 | ✅ |
| C25 | Docs, cleanup, final plan sync | — | ✅ |
| C34 | Savings in DKK/EUR; fixes from the mock-data walkthrough (hint, fund check, errors, required marks, quick-add amounts, push on account switch, rate note, onboarding counter) | 3, 8 | ✅ |

## Follow-ups and review outcomes

### Closed with regression evidence

| Review item | Outcome and evidence |
|---|---|
| Managed-Postgres migration role checks | Closed. `tests/db/test_rls.py::test_migrations_preserve_an_enabled_api_role_login` protects the API login during migration. |
| Household owners must not see Leave | Closed. `apps/web/tests/settings/SettingsPage.test.tsx::shows four simple groups, keeps plan details closed, and passes the accessibility scan` checks the owner view. |
| Google sign-in rate-limit proxy key | Closed. `tests/api/test_google_auth.py::test_google_exchange_uses_render_rightmost_forwarded_hop_for_rate_limit` checks Render's right-most forwarded hop. |
| Household audit bypass and append-only log | Closed (migration 0026). `tests/db/test_shared_households.py::test_household_audit_cannot_be_suppressed_with_a_session_setting` and `test_audit_log_is_append_only_and_account_deletion_cascades` cover the trigger and cascade behavior. The old setting never took effect (set inside a function with its own SET clause, so reverted on return), which made a partner's account deletion fail when they had hidden a note in another family plan; `test_a_partner_can_delete_their_account_with_notes_hidden_in_the_family_plan`. |
| EPF monthly interest | Closed. `tests/valuation/test_asset_engines.py::test_epf_uses_epfo_monthly_running_balance_interest` checks the hand-computed FY example and that a month counts only once it has ended. |
| Long Flex response timeout | Closed. `tests/research/test_router.py::test_flex_requests_use_the_configured_long_read_timeout` covers the weekly read timeout. |
| US bulk daily-price source | Re-evaluated and closed for v1. The screener uses fundamentals-only coverage; `tests/research/test_screener.py::test_us_fundamentals_only_requires_quality_and_value_groups` protects it. |
| Weekly research workflow and CLI | Closed for code. `tests/research/test_run.py::test_cli_matches_scheduled_workflow_week_selectors` checks the workflow's supported start and continue selectors. `RESEARCH_ENABLED` turned on 2026-10-30 after the live smoke run. |
| Manual assets without a price | Closed. `tests/db/test_money_map_api.py::test_member_account_asset_goal_and_quick_add_endpoints` checks `valuation_status` and the net-worth unvalued-assets list. |
| Checklist and consent endpoints | Closed. `tests/db/test_alerts_delivery.py::test_consent_defaults_off_and_checklist_is_available_on_me` checks both endpoints and household isolation. |
| ECB failure isolation | Closed. `tests/marketdata/test_sources.py::test_ecb_multi_request_failure_isolated_by_currency` checks that one currency failure does not discard the others. |
| Dev-token account recreation after deletion | Documented in the README. `tests/db/test_auth_sessions.py::test_a_request_after_deleting_the_account_does_not_recreate_the_user` protects production app-session tokens; dev tokens remain a local/e2e-only caveat. |

### Still open or owner-run

- **Post Office rates (C03):** tenure-specific Post Office TD rates and older small-savings periods still need authoritative reference data.
- **RLS `app.user_id` risk (C05):** SQL able to call `set_config` with another tenant UUID could cross tenants. Parameterised SQLAlchemy use and the restricted `wealth_api` role mitigate this; `tests/db/test_rls.py::test_api_role_bootstrap_and_reset_role_remain_tenant_scoped` is regression evidence. Keep this risk visible.
- **GitHub history cache:** ask GitHub Support to purge the old commit views after the approved history rewrite.
- **Render CLI validation:** `render blueprints validate` could not run because the Render CLI is unavailable in this environment. The repository deployment checker remains the local check.
- **Home load speed:** authenticated Home LCP is 2.14 s in CI (run 41122116458) and 2.66–2.73 s on the reviewer's laptop, under 4x CPU and slow 4G with the standard build, against the 2.5 s target. Earlier figures (3.5–3.9 s) measured the unminified `SANDBOX_BUILD` bundle, which the e2e stack used by default until the post-release fix. `e2e/tests/performance.spec.ts` is tagged `@performance` and runs non-blocking in CI. Next step: profile the sign-in refresh before the first data request.
- **npm low findings:** `npm audit --omit=dev` reports four low-severity transitive findings through `vite-plugin-pwa`/Workbox. No high or critical findings were reported. npm's forced fix downgrades the PWA plugin across a breaking boundary, so retain the current stack and revisit when a compatible upstream fix is available.
- **Weekly research activation:** the live Foundry smoke run and live eval passed on 2026-10-30. `SEC_USER_AGENT` and `RESEARCH_ENABLED` were set the same day, after migrations 0024–0026 went live. Review found that no job loaded the S&P 500 and Nifty 500 lists or their company results: the daily job's Saturday SEC step could never run (weekday schedule, missing module). `marketdata-weekly.yml` now loads them on Friday evening, before the Saturday research. `tests/test_deploy_config.py::test_weekly_company_data_loads_stock_lists_before_results_and_before_research` guards the order. GitHub dropped every scheduled run in the first Saturday's early hours, so a manual research run now takes a `step` input (start or continue).
- **First live research run (2026-W44) shortlisted nothing.** Three data gaps, all fixed with tests: (1) the daily NSE price load asked for the run date's bhavcopy, but GitHub starts the 19:00 IST job after midnight IST and manual runs came before NSE posts the file, so it never stored a price; it now re-reads the last 7 days, and NSE writes are bulk (one lookup per row took minutes per day from GitHub). (2) India momentum and volatility need 200 sessions; the new `nse_history` source loads missing sessions in the 400-day window (about a year on its first run) every Friday, committing each day: the first production run held one transaction across ~285 downloads, lost its Neon connection and rolled back. (3) US value needs a price and sector-neutral scores need sectors: SSGA now lists every SPY sector as "-", and no US price feed existed. The universes refresh now stores each holding's implied close (weight x SPY total net assets / shares held, from SSGA's NAV history) and GICS sectors from the public S&P 500 constituents table. (4) The SEC parser kept only 3-month quarters, but 10-Q cash flows are year-to-date and 10-Ks give full-year EPS, so 23 of 503 companies had an earnings yield; year-to-date differencing and a derived fourth-quarter EPS raise that to 464 (cash-flow yield 10 to 330). A local run on live data then shortlisted 25 US companies (the cap) and about 47 of a 50-company India sample were eligible. Prices for a year of NIFTY 500 sessions take about 68 MB (322 bytes a row). India Post's rate page also timed out and stopped the run; each official rate page is now optional, and a failed watcher only warns.

`legacy/` stays, read-only. No code or test imports it; only `.gitignore`, the Ruff exclude in `pyproject.toml`, `AGENTS.md`, `README.md` and card docs name the directory. Deleting it (it stays in git history) is the owner's call.

All implementation cards are accepted; C26b was superseded by C27b. The remaining items above are data work, accepted risk, environment limitations, or owner-run launch steps.
