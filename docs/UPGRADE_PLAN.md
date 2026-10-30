# Upgrade Plan: From "Stock LangGraph Platform" to an Indian Investor's Global Wealth Copilot

> **Status:** Implemented through cards C01–C34; see `docs/cards/README.md` for accepted work and remaining owner actions.
> **Reviewed:** `github.com/sudipta0311/stock` @ `40375c0` (main, 2026-10-25)  
> **Version:** v2.1 (2026-10-27): adds Render free-tier hosting  
> **Scope:** Full product + architecture redesign. The current README rules are **not** treated as constraints; every rule below is derived fresh from first principles.

## What changed during implementation

The implementation cards deliver the v1 scope, with these decisions recorded here:

- The PWA uses Vite, React, TypeScript, and a static export instead of Next.js.
- Azure AI Foundry routes nano, mini, and strong work to `gpt-6-luna` with low, medium, and high reasoning effort. Weekly calls may use Flex; Red Team uses `deepseek-v4-flash`. Configured prices per million tokens are $0.10 input, $0.50 output, and $0.01 cached input for `gpt-6-luna`; $0.15 input and $0.31 output for `deepseek-v4-flash`. The price table lives in `rules/models.yaml`.
- FX uses ECB's multi-series reference data, with per-currency failure handling and a fallback source. India mutual-fund NAV loads from the AMFI portal host first.
- User-entered and statement-derived non-market holdings share the `manual_assets` table with typed asset details.
- The API connects through the restricted `wealth_api` login. A SECURITY DEFINER bootstrap function establishes request context without giving the API owner credentials.
- CDSL parsing is behind the `CDSL_PARSER_BETA` flag. India company fundamentals come from NSE Integrated Filing XBRL; US coverage is fundamentals-only until a suitable bulk daily-price source is available.
- Weekly email copy uses reviewed templates; no LLM writes user email copy or determines money figures.
- `SANDBOX_BUILD=1` is only a local workaround for restricted build sandboxes. CI and deployment use the standard Vite build.
- Research tilts require independent evidence, stay within policy limits, and decay when current evidence does not reconfirm them.
- The Denmark guide prioritises an Aktiesparekonto, lists ABIS/UCITS funds, permits US-listed shares, and excludes US-listed ETFs. The India fund check flags fund overlap, direct-versus-regular plans, lagging funds, and a 12-month early warning.
- The PWA follows the simplicity rules: plain language, one main action, short highlights, collapsed details, and household-aware views for shared access.
- EPF interest follows the monthly running-balance method and is credited at financial-year end. Household access supports owner, partner, and viewer roles with database-enforced isolation.

The remaining sections describe the original proposal and its historical context; implementation choices above and accepted cards take precedence where they differ.

### What changed in v2

| Area | v1 | v2 |
|---|---|---|
| Primary user | Resident Indian + NRIs equally | **Resident Indian investor** (NRI module moved to a later phase) |
| What the user can record | Stocks, MFs, ETFs | **Everything:** FD, RD, PPF, EPF/VPF, NPS, Post Office schemes, bonds, SGB/gold, LIC & other insurance, savings, loans, US holdings, all **stored in the DB** |
| Input UX | CAS upload + manual tables | **Money Map** setup right after first login: CAS/broker upload + a "what else do you have?" checklist with 2–4-field mini-forms, autosaved |
| Global angle | India vs US split by residency | **Geopolitics-aware global diversification** for an Indian investor, while India investments keep growing |
| Client | Native app (Expo) | **Progressive Web App** (installable, mobile-first, works on low-end Android) |
| Hosting | Azure (Container Apps, Static Web Apps, Key Vault) | **Render free tier** (static PWA + free API) + **GitHub Actions** for scheduled jobs + **Neon free Postgres**. Azure AI Foundry is kept for LLMs only. The Azure design becomes the scale-up path (§7, Appendix D). |

---

## 0. TL;DR

| | Today | Redesign |
|---|---|---|
| **What it answers** | "Which NIFTY stock fills my sector gap?" | "Here's *all* my money (FDs, EPF, PPF, LIC, stocks, funds). How should it be split between India, US, debt and gold given today's world, what do I buy this week, and when do I exit?" |
| **Inputs** | CAS PDF, broker CSV, JSON | CAS (NSDL/CDSL/CAMS-KFin), broker statements (Zerodha, Groww, Angel One, Upstox, ICICI, HDFC, IBKR, Vested, INDmoney), plus quick manual entry for every Indian asset type |
| **Markets** | India large/mid caps | India + US stocks and ETFs; Indian debt, small-savings schemes, gold |
| **Core logic** | Sector-gap scoring + LLM rationale | Deterministic **Policy Engine** (whole-wealth allocation, debt %, global %, sizing, exits, tax) + an autonomous **Research Desk** of agents that research, verify, veto and explain |
| **Geopolitics** | Hardcoded tariff table | **Macro & Geopolitics agent** gives bounded, evidence-cited tilts + personal **stress tests** ("If the rupee falls 10%, your net worth changes by …") |
| **LLMs** | Claude + GPT, per user, per click | Cheap **Azure AI Foundry** models (nano/mini), run **once a week for the market**, Batch API, ~US$1–5/week |
| **UI** | Streamlit, desktop-first | **PWA**: install to home screen, Home = "This week: 0–3 actions" |
| **Alerts** | None | **Weekly email** (Sunday 08:00 IST) + web push + maturity/premium reminders |
| **Hosting** | Streamlit Cloud | **Render free tier:** PWA as a static site (never sleeps) + FastAPI free web service; **GitHub Actions** runs research, reminders and emails; **Neon** free Postgres. ≈ US$0/month hosting + ~US$1–5/week Foundry. |

**Three things to do this week, before any redesign work** (details in §1.3):
1. The **public** repo contains a real NSDL CAS statement **and its password (a PAN)**. Make the repo private and purge both from git history.
2. In Neon mode **every signed-in Google user shares one portfolio database**, and there is no email allowlist. Restrict the live app to your own account.
3. The live quality model is **100% debt-to-equity**, and the backtest scorer counts "no data yet" as a miss. Don't trust current recommendations.

---

## 1. Review of the Current Repository

### 1.1 What exists

- ~30k lines of Python: Streamlit UI (`streamlit_app.py`, 3,193 lines), 4 LangGraph flows (signal → portfolio → buy → monitor), 35 graph nodes, 8 LLM prompts, dual Anthropic/OpenAI providers.
- Data: yfinance, Screener.in scraping, NSE archive CSVs, Tickertape, AMC pages, mfdata.in. SQLite locally, Neon Postgres in prod.
- Inputs: NSDL CAS PDF, broker CSVs (Zerodha/Groww/ICICI), JSON/CSV, manual tables for MF/ETF/stocks only.
- Weekly GitHub Actions backtest committing a markdown summary to `backtest/results/` (22 consecutive weekly commits).
- ~150 unit tests (per `SESSION_LOG.md`; I did not run them).

### 1.2 What is genuinely good and worth carrying forward

| Asset | Why it's valuable | Where |
|---|---|---|
| NSDL CAS parser | Hard India-specific problem, already solved | `src/stock_platform/services/pdf_parser.py` |
| Broker CSV parser | Same | `utils/broker_parser.py` |
| Holdings de-duplication across sources | Needed when CAS and broker files overlap | `src/stock_platform/utils/direct_equity_merge.py` |
| MF/ETF look-through and overlap | Real differentiator for Indians holding 10+ overlapping funds | `agents/portfolio_agents.py`, `services/mf_lookup.py`, `services/amc_adapters.py` |
| Tax calculator | Rates are current (STCG 20%, LTCG 12.5% over ₹1.25L) | `utils/tax_calculator.py` |
| Bull-vs-bear analyst debate | Good pattern. Keep it as the Red-Team agent | `services/llm.py` |
| Data provenance + source-health gate | Right instinct: never score on defaulted data | `utils/source_health.py`, `screener_fetcher.py` |
| Tranche-based entry plans | Right instinct for retail behaviour | `utils/entry_calculator.py` |
| Backtest harness concept | Needs a rebuild, but the discipline is right | `backtest/` |

### 1.3 Critical findings: fix now (P0)

| # | Finding | Evidence | Impact | Action |
|---|---|---|---|---|
| P0-1 | **Personal financial data in a public repo.** A real NSDL CAS statement is committed, and its password (the account holder's PAN) is written in plain text. | `tests/sample_cas.pdf` (added `9b5305c`, 2026-04-10); password in `PLATFORM_DETAILED_NOTES.md:137-139`, also referenced in `tests/run_pdf_e2e.py` and `tests/test_engine_monitoring.py` | Anyone can open the full statement: name, address, PAN, every holding. A PAN can't be rotated. | Make the repo private now. Purge the PDF and password from **all** history with `git filter-repo`, then force-push all 3 branches. Replace with a synthetic fixture. Treat the data as exposed (watch for PAN misuse). |
| P0-2 | **No tenant isolation in production.** Per-user `db_path` is ignored when `NEON_DATABASE_URL` is set; no table has a `user_id`; `direct_equity.symbol` is `UNIQUE`. | `streamlit_app.py:600-618`, `data/db.py:162-195`, `data/schema.py:45-53`, `services/engine.py:139-141` | Every Google user who signs in reads and overwrites the **same** portfolio. No sign-in allowlist. | Allowlist your email now, or take the URL down. Real fix: the multi-tenant schema with Postgres RLS in §7.3. |
| P0-3 | **Degenerate live quality model.** Weights are `debt_equity: 1.0`, all others `0.0`, from a 4-month train / 2-month validate window. | `rules/quality_weights.yaml` | The "quality score" is just a leverage filter. That explains why the rank-1 pick is the worst. | Revert to multi-factor defaults; calibrate only with ≥ 36 months of point-in-time data. |
| P0-4 | **Backtest scorer bug.** `hit = bool(alpha is not None and alpha > 2.0)` turns "no forward data yet" into a miss. | `backtest/scorer.py` (`score_run`) | The newest ~26 weeks always count as 6m misses, and `hit_rate_12m` can only be 0.0. The reported 25.8% is really about **50%**. CI has "failed" for 5 months on a measurement artifact. | `hit = None` when `alpha is None`. |
| P0-5 | **Backtest biases.** Fundamentals are stamped with the *quarter-end* date (results publish 45–60 days later): look-ahead. The universe is a hardcoded list of *today's* NIFTY 200: survivorship. Screener enrichment backfills *current* values into history. | `backtest/snapshot.py:40,181-202`, `backtest/replay.py:200-228`, `backtest/screener_history_fetcher.py` | Every historical metric is optimistic. | Rebuild on point-in-time data (§6.3). |
| P0-6 | **Stale or mislabeled docs and signals.** README says LTCG 10% / STCG 15% (the code's 20% / 12.5% is correct). "Geopolitical" and "Policy" signals (35% and 25% weight) are really sector price momentum + a hardcoded tariff table. | `README.md` §5, `providers/live.py:594-640`, `utils/signal_sources.py` | The model claims to do things it doesn't. | The redesign replaces them with a real Macro & Geopolitics agent (§5). |
| P0-7 | **Tax bug.** The ₹1.25L LTCG exemption is applied per stock (it's once per FY across all listed equity). Surcharge + 4% cess are missing. | `utils/tax_calculator.py` | Overstates "tax saved by waiting". | Fix in the new Tax Engine (§4.10). |

### 1.4 Why redesign rather than refactor

1. **The app only sees a fraction of an Indian investor's money.** For a typical salaried Indian, EPF + PPF + FDs + LIC + post-office schemes are often **30–60% of net worth**. Without them, every allocation answer ("buy more equity?", "how much in debt?") is wrong. Most users are already far more debt-heavy than they realise.
2. **Asset allocation drives ~80–90% of long-run portfolio variance, and the app has none.** It never answers equity vs debt vs gold, or India vs global.
3. **"Sector gap" is not a reason to buy a stock.** Diversification belongs at the allocation level; stock selection is about expected return vs risk.
4. **No global diversification**, even though an Indian investor's whole life is concentrated in INR: salary, house, EPF, FDs, equity. That is a single-country, single-currency bet.
5. **Per-click, per-user LLM pipelines** are slow and costly, and they don't scale. Market research is the same for every user: do it once a week.
6. **Streamlit can't deliver an intuitive mobile experience.** Keep it only as an internal console.
7. **Tangled modules:** a root `utils/` with 9-line shims next to `src/stock_platform/utils/`, mixed imports, Turso remnants, CI committing logs to git.

**Verdict:** Keep the parsers, de-duplication and look-through. Rebuild everything else around a new core.

---

## 2. Product Vision & Principles

### 2.1 One-line vision

> A calm wealth copilot for Indian investors that sees **all** your money (FDs to LIC to stocks), keeps your **India investments growing**, helps you **diversify globally** as the world changes, and tells you each week what to buy, how much to keep in debt, and when to exit.

### 2.2 Personas (v1 = resident Indians)

| Persona | Situation | Their #1 question |
|---|---|---|
| **Riya, 29, Bengaluru, IT** | EPF, 6 overlapping MF SIPs, a few stocks via Zerodha, no global exposure | "Am I diversified? How do I start investing in the US?" |
| **Vikram, 42, Pune, manager, 2 kids** | EPF, PPF, 3 LIC endowment policies, FDs, NSC, MFs via Groww, home loan; child's education abroad in 2034 | "Are my LIC policies worth it? How much in debt? With tariffs and the rupee falling, should I move some money abroad?" |
| **Meera, 56, Kolkata, pre-retiree** | Large FDs, post-office MIS, PPF, some blue chips, retiring in 4 years | "How do I make my money last, safely, and still beat inflation?" |
| NRI (Phase 6) | US/Gulf/UK resident | Residency-specific rulebook (PFIC, NRE/FCNR, TDS); designed for, not built in v1 |

### 2.3 Design principles

1. **Whole money, not just stocks.** Allocation is computed on everything the user owns, including EPF, PPF, FDs and insurance savings.
2. **Rough is fine.** Every asset can be added with 2–4 fields; details are optional and only unlock extra features (maturity alerts, exact accrual).
3. **Allocation first, selection second.** Stock picks fill a capped satellite sleeve.
4. **Grow India, add the world.** Global diversification is funded mainly by *new* money and SIP redirection, not by selling Indian winners.
5. **Agents research; code decides money.** LLM agents research, verify, veto and explain. Amounts, weights, tax and exit triggers come from deterministic, tested code. *(This is what makes cheap models safe.)*
6. **"Do nothing" is a valid, common answer.** Most weeks the right action is none.
7. **Every exit is decided at entry.** Each buy creates an Exit Card.
8. **Research once, personalise cheaply.** Weekly research is shared by all users; per-user work is arithmetic.
9. **Personal data never goes to a research LLM.** Statements are parsed by deterministic code; user-facing text is generated from pseudonymised numbers.
10. **Honest track record.** Forward paper-trading results vs benchmark, misses included.
11. **Plain language, Indian formats.** ₹12.4 L, ₹1.2 Cr; "Buy in 3 parts", not "STRONG ENTER".

---

## 3. The Money Map: Capturing Everything a User Owns (new)

The Money Map is the first experience after login and the foundation for every recommendation. **Target: a typical user completes it in under 7 minutes.**

### 3.1 Setup flow after first login

```
Login (Google / email OTP)
  │
  ├─ Step 1  "Upload what you have" (optional, biggest win)
  │     • NSDL / CDSL CAS        → stocks, ETFs, bonds, SGBs, MFs
  │     • CAMS / KFintech CAS    → MFs with full transaction history (cost basis)
  │     • Broker statement       → holdings + tradebook (cost & dates for tax)
  │     • US broker CSV          → IBKR / Vested / INDmoney holdings
  │
  ├─ Step 2  "What else do you have?"  (tap all that apply)
  │     [EPF] [PPF] [Fixed Deposits] [Post Office] [NPS] [LIC / Insurance]
  │     [Gold] [Bonds] [Savings] [Loans] [Property] [US investments]
  │
  ├─ Step 3  Mini-forms, one card per item (2–4 required fields, autosaved)
  │
  ├─ Step 4  Review & confirm: merge duplicates, fix unknown items
  │
  └─ Step 5  "Your Money Map": net worth, split by type, completeness meter
             → then risk check + goals → first plan
```

UX rules:
- **Autosave every field to the DB.** Leave at any point and resume where you stopped. Nothing is lost.
- **Completeness meter** ("Money Map 80% complete · add EPF to finish") instead of forcing everything up front.
- **Indian number input:** accepts `5L`, `5 lakh`, `1.2 cr`, `50k`, `₹5,00,000`.
- **Month-year pickers**, because most people don't know exact deposit dates. "I don't know" is allowed for most fields.
- **Pre-filled rates:** PPF, EPF, NSC, SCSS, KVP, SSY, POMIS and post-office TD/RD rates are auto-filled from the official rate table for the purchase period.
- **Quick-add line** (optional): type "FD 5L SBI 7.1% till Mar 2027". A deterministic parser fills the card. A small Foundry model is only a fallback, with user consent and no identity data sent.
- **Household:** add family members (self, spouse, parents, kids) and tag each asset to a member. Indian investors often manage family money. No PANs are stored for members.
- **Edit anywhere, anytime.** Every asset has an edit history (who changed what, when).

### 3.2 Asset catalogue (what each mini-form asks)

| Asset type | Required (minimum) | Optional (unlocks…) | How value is computed | Counts toward | Liquidity | Reminders |
|---|---|---|---|---|---|---|
| **Savings / current account** | Bank, balance | — | As entered | Cash | Instant | Monthly "still correct?" check |
| **Fixed Deposit** | Bank, amount, rate, maturity (month/yr) | Start date, compounding (default quarterly), payout type, joint holder | Accrued value today + maturity value | Debt | Locked (premature penalty) | Maturity −30 / −7 days, with reinvest suggestion |
| **Recurring Deposit** | Bank/PO, monthly amount, rate, maturity | Start month | Accrued value | Debt | Locked | Maturity |
| **PPF** | Current balance, year opened | Yearly contribution, where held (bank/PO) | Balance + accrual at notified rate | Debt | Locked (partial withdrawal from year 7) | Deposit before the 5th to earn that month's interest; FY-end minimum; maturity/extension |
| **EPF / VPF** | Current balance | Monthly contribution (yours + employer), VPF | Balance + accrual at EPFO rate | Debt | Locked (conditional withdrawals) | "Update from your UAN passbook" every 6 months |
| **NPS** | Tier I balance | Tier II, scheme mix (E/C/G/A) | As entered (CRA statement import later) | Split by scheme mix: equity / debt | Locked till 60 | Annual contribution |
| **Post Office** (NSC, KVP, SCSS, MIS, TD, RD, SSY) | Scheme, amount, purchase month | Certificate count, nominee (not stored) | Scheme formula at the rate locked on purchase date | Debt | Locked (varies by scheme) | Maturity; SCSS/MIS payout dates |
| **Bonds** (G-sec, T-bill, RBI FRSB, corporate, tax-free, 54EC) | Type, amount, coupon, maturity | ISIN (or from CAS) | Face + accrued, or market price if listed | Debt | Varies | Coupon, maturity |
| **Gold**: SGB / Gold ETF / Gold MF | From CAS, or units | Series / issue date (SGB) | Units × gold price, or NAV | Gold | Exchange / SGB maturity | SGB interest & maturity |
| **Physical gold** | Grams, purity | Purpose (jewellery / investment) | Grams × benchmark gold rate | Gold (jewellery excluded from investable by default) | Low | — |
| **Term insurance** | Insurer, cover, ends (year) | Premium, riders | Protection only (not an asset) | Protection check | — | Premium due |
| **Health insurance** | Insurer, cover, renewal month | Members covered, top-up / super top-up | Protection only | Protection check | — | Renewal −30 days |
| **LIC / endowment / money-back / whole-life** | Insurer, plan name/number, sum assured, annual premium, start year, term | Premium paying term, bonuses so far, surrender value (from insurer), maturity estimate | Surrender value if given, else "premiums paid (estimate)", clearly labelled | **Insurance-linked savings** (debt-like) | Low (surrender loss) | Premium due; survival benefits; maturity |
| **ULIP** | Insurer, current fund value, premium | Fund split | Fund value | Split by fund (equity/debt) | Locked 5 years | Premium |
| **Indian stocks / ETFs / MFs** | From CAS or broker file, or manual search (name + qty) | Avg buy price + buy month (for tax) | Market price / NAV. Hybrid MFs split by look-through. | Equity India / Debt / Gold | T+1 | Exit Card triggers |
| **US / global holdings** | From broker CSV, or manual search (ticker + qty) | Buy price + date (for 24-month LTCG) | Price × USD/INR | Equity Global | T+1 + remittance | Exit Card; Schedule FA reminder |
| **Property** (optional) | Type, estimated value | Linked loan, rent | As entered | Real estate (outside the investable plan by default) | Low | Annual re-estimate |
| **Loans** | Type, outstanding, rate, EMI | End date | Amortised | Liability | — | Prepay-vs-invest nudge if rate > expected return |

### 3.3 Statement imports

| Document | What it gives | What it doesn't | Parser |
|---|---|---|---|
| **NSDL / CDSL CAS** (monthly, PDF, password-protected) | All demat holdings (stocks, ETFs, bonds, SGBs) + MF folios linked to PAN | **Cost basis and buy dates** for demat shares | Port the existing `pdf_parser.py`; add CDSL format |
| **CAMS / KFintech CAS** (detailed) | MF transactions → exact cost & dates per lot | Demat holdings | New |
| **Broker holdings + tradebook** (Zerodha, Groww, Angel One, Upstox, ICICI Direct, HDFC Sec, Kotak) | Lots with cost and dates → tax-aware exits | Other brokers' holdings | Port `broker_parser.py`; one adapter per broker |
| **US brokers** (IBKR activity/flex CSV, Vested, INDmoney) | US holdings + lots | — | New adapters |
| EPF passbook PDF, NPS CRA statement (Phase 4) | EPF/NPS balances | — | New |
| Account Aggregator (Phase 6) | Live FDs, RDs, MFs, equities, NPS, insurance | — | Needs a regulated FIU (§11) |

**Recommended combination shown to users:** "CAS + your broker's tradebook covers ~80% of most investors. Add EPF, PPF, FDs and LIC by hand in ~3 minutes."

**Import pipeline (no LLM, no file retention):**
1. Upload via the file picker. On Android the PWA is also a **share target**: share the CAS PDF straight from Gmail into the app.
2. The password is typed by the user each time, used in memory only, **never stored**.
3. A sandboxed parse worker extracts holdings to a staging table (`import_items`). The file is deleted immediately (malware-scanned first).
4. **Review screen:** new / changed / duplicate (CAS vs broker) / unknown. The user confirms with one tap; duplicates are auto-merged using the existing `direct_equity_merge` logic.
5. The confirmed import becomes the new snapshot. History is kept, so net worth over time works.

**Monthly refresh nudge:** "Your new CAS usually arrives around the 10th. Upload it (30 seconds)." Plus one-tap "still correct?" checks for manual balances (EPF, savings). Values roll forward automatically between updates (accrual), so the Money Map stays plausible.

### 3.4 Valuation engines (deterministic, unit-tested)

- **FD / RD / NSC / KVP / PO TD / RD:** compound-interest formulas per scheme rules, with rates locked at purchase.
- **PPF / EPF / SSY:** balance + contributions + accrual at the notified rate for each period (rates change quarterly or annually), stored in a `rates_reference` table with effective dates.
- **SCSS / MIS:** principal + payout schedule.
- **SGB:** units × gold price; 2.5% coupon schedule; maturity.
- **Insurance-linked savings:** surrender value if the user enters it; otherwise premiums paid, clearly labelled "estimate". Projected maturity only if the user enters the insurer's illustration.
- **Market assets:** EOD price × FX; MF NAV from AMFI; hybrid-fund look-through splits value across equity / debt / gold.

### 3.5 What the Money Map shows (and why it matters)

- **Net worth** (household or per member) and its trend.
- **Split by:** Equity India · Equity Global · Debt (FD, PPF, EPF, PO, bonds, debt MF, insurance-linked) · Gold · Cash · Property · Liabilities.
- **Liquidity ladder:** available today / within 1 year / locked beyond 1 year. Only liquid money counts toward the emergency fund.
- **Maturity calendar:** FDs, NSC, PPF, LIC maturities, premiums and renewals.
- **"Aha" insight on day one:** "Including EPF, PPF and FDs, you already have **41% in debt**. Your new investments can lean toward equity."

---

## 4. The Advice Engine (Deterministic Policy Engine)

Plain Python + versioned YAML rules + unit tests. Agents supply research inputs; they can't override its arithmetic.

### 4.1 Step 1: Foundation check

| Check | Rule | If failing |
|---|---|---|
| High-interest debt | Credit card / personal loan above ~12% | "Pay this off first. It's a guaranteed 12–40% return." |
| Emergency fund | 6 months of expenses (salaried) / 9–12 (self-employed), **liquid assets only** (savings, sweep FD, liquid fund) | Route new money here first |
| Term cover | ~10–15× annual income + outstanding loans − liquid assets, if anyone depends on you | Show the gap (education only, no product selling) |
| Health cover | Family floater + super top-up appropriate to city and age | Show the gap |

### 4.2 Step 2: Risk profile = min(capacity, tolerance)

- **Capacity** (objective): horizon, income stability, dependants, wealth vs goals.
- **Tolerance** (behavioural): 6 scenario questions, e.g., "Your ₹10 L falls to ₹7 L in 3 months. What do you do?"
- **Final profile = the lower of the two**, explained to the user.

### 4.3 Step 3: Strategic allocation on the *whole* Money Map

Money is bucketed by goal horizon: **< 3 years → 0% equity**; 3–7 years → capped at Moderate; > 7 years → full profile.

| Profile | Equity (India + Global) | Debt | Gold | Cash (beyond emergency) |
|---|---|---|---|---|
| Conservative | 30–40% | 50–60% | 10% | 0–5% |
| Moderate | 50–60% | 30–40% | 10% | 0–5% |
| Growth | 70–80% | 15–20% | 5–10% | 0% |
| Aggressive | 85–90% | 5–10% | 5% | 0% |

**Glide path:** within 3 years of a goal, move 1/3 of that goal's equity to debt each year.

**The "how much in debt" answer** = goal-weighted blend of the table + emergency fund + all money needed within 3 years. Existing EPF, PPF, FDs, post-office and insurance-linked savings **count toward it**. The app then shows the remaining gap, if any, and where new debt money should go (§4.6).

> Locked assets (EPF, PPF, LIC) can't be rebalanced, so the engine rebalances **around** them by steering new money and liquid holdings.

### 4.4 Step 4: Global diversification for an Indian investor (geopolitics-aware)

**Why an Indian investor needs global exposure:**
- **Single-country concentration:** salary, property, EPF, FDs and equity all depend on one economy and one policy regime.
- **Currency:** over long periods the rupee has weakened against the dollar (roughly 3–4% a year on average over recent decades, though not every year). Dollar assets hedge future dollar-linked spending: foreign education, travel, imported goods, a possible move abroad.
- **Geopolitical and trade shocks** (tariffs, sanctions, oil spikes, since India imports most of its crude, regional conflict) tend to hit Indian equities and the INR together. Global assets and gold often move differently.
- **Sector access:** global platforms in semiconductors, cloud/AI, global healthcare and consumer brands that the Indian market doesn't offer at scale.

**Target global share of the equity sleeve (resident Indian):**

| Profile | Base global share of equity |
|---|---|
| Conservative | 10–15% |
| Moderate | 15–20% |
| Growth | 20–30% |
| Aggressive | 25–35% |

**Adjustments:**
- **+ up to 10 pp** for foreign-currency goals (child's education abroad, planned emigration).
- **± 5 pp geopolitical tilt** from the Macro & Geopolitics agent (§5.2), evidence-cited and bounded. Gold gets a separate ± 3 pp tilt.
- **Avoid replacing one concentration with another:** default global core = **US-heavy all-world or S&P 500 index**, with an optional US-only tilt. The US alone is still a single country.

**How "India keeps growing":** India SIPs continue untouched. The global share is built from **new money**: e.g., "Of your ₹50,000 monthly SIP, ₹10,000 goes to the global index from next month". It glides to target over 12–24 months, which also avoids timing the rupee. **Indian holdings are never sold just to buy global.**

### 4.5 Step 5: How to buy global: the route selector

The engine picks the **simplest route that is cost- and tax-efficient** for the user's amount:

| Route | How | Pros | Cons / limits | Default for |
|---|---|---|---|---|
| **A. Indian international index fund / FoF** (S&P 500, Nasdaq-100, global) | Your usual MF app, in INR, SIP-able | Zero paperwork, no LRS | Higher expense ratio; AMCs sometimes **pause new money** when industry overseas-investment limits bind | Global SIP below ~₹25k/month, or users who want zero paperwork |
| **B. Indian-listed international ETF** | Your demat | Simple, intraday | Can trade at a **large premium to iNAV** when limits bind. The engine warns if premium > 1%. | Small lump sums, when the premium is low |
| **C. LRS → international broker** (US ETFs/stocks, or **Ireland-domiciled UCITS ETFs**) | Remit up to US$250k/FY under LRS | Lowest cost, widest choice, direct US stocks | Forex + remittance cost; **TCS 20% on LRS above ₹10 L/FY** (adjustable against tax, but a cash-flow drag); Schedule FA reporting; Form 67 for foreign tax credit. **US estate tax** applies to US-domiciled assets above US$60k, so the engine switches to UCITS above that threshold. | Annual global investing above ~₹5 L, or users wanting US stocks |
| **D. GIFT City (NSE IFSC)** | US stocks via depository receipts, under LRS | Indian platform | Newer, thinner liquidity | Optional |

**US satellite stocks** (direct US shares) are only offered on route C, once the global core exists. They count toward the US$60k estate-tax threshold.

### 4.6 Step 6: The debt sleeve: where new debt money goes (resident Indian)

| Need | Instrument options (engine picks by **after-tax** yield for the user's slab) |
|---|---|
| Emergency / < 1 year | Savings + sweep FD, liquid / overnight fund |
| 1–3 years | FD ladder across banks (stay within the DICGC ₹5 L-per-bank insurance limit), PO Time Deposit, arbitrage fund for 30%-slab users (equity taxation) |
| 3–7 years | Target-maturity G-sec/SDL funds, RBI Retail Direct G-secs, NSC, PO TD 5-year |
| Long-term, tax-free | PPF (up to ₹1.5 L/yr), VPF (watch the taxable-interest threshold on contributions) |
| Retirees | SCSS, POMIS, RBI Floating Rate Savings Bonds, FD ladder for monthly income |

The **tax regime** (old vs new, asked in onboarding) changes advice: 80C-driven products (PPF/ELSS/LIC premiums) only get a tax benefit under the old regime.

### 4.7 Step 7: Equity sleeve = Core (ETFs/index funds) + Satellite (stocks)

```
Equity sleeve
├── India core (≈ 75–85% of India equity)
│     Nifty 50 index fund/ETF + Nifty Next 50 or Midcap 150 index
├── India satellite (≈ 15–25% of India equity)
│     8–15 Indian stocks, ≤ 5% of total equity each, ≤ 25% of satellite per sector
├── Global core (≈ 80–100% of global equity)
│     S&P 500 / all-world index via route A, B or C (§4.5)
└── Global satellite (optional, route C only, ≤ 20% of global equity)
      US stocks from the factor model
```

- **ETF/fund selection** (deterministic): expense ratio, tracking difference, AUM, liquidity, iNAV premium/discount, domicile/tax fit. The ETF Selector agent explains and flags issues.
- **Stock selection** (replaces sector-gap scoring):
  - Universe: India = NIFTY 500 point-in-time; US = S&P 500 + Nasdaq-100.
  - Sector-neutral multi-factor model: Quality, Value (vs the stock's own 10-year band), Momentum (12-1 month), Low volatility, Governance (pledges, auditor changes, related-party flags).
  - **Geopolitical exposure flag:** share of revenue from the US, export tariff exposure, imported-input dependence, from annual-report segment data.
  - Equal factor weights to start; calibrate only with ≥ 36 months of point-in-time data and minimum weights.
  - **Agents can veto or downgrade a quant pick, never originate one.**

### 4.8 Step 8: Buying rules ("what to buy this week")

1. New money first goes to whatever is furthest **below target**: debt gap, global gap, India gap (no selling, no tax).
2. Core funds and ETFs via SIP/STP. Lump sums go in via 3–6 weekly tranches, with more tranches when valuations are in the top quintile of history.
3. Satellite stocks need: top-decile quant rank + dossier `APPROVED` + caps OK + valuation not in the top decile of the stock's own band. Bought in 2–3 tranches.
4. Output is concrete, including the route: "Invest ₹10,000 in an S&P 500 index fund (Route A) via your MF app", "Buy 6 shares of X at or below ₹1,420 (part 1 of 3)".
5. **Max 3 new actions per week.**

### 4.9 Step 9: Exit Engine ("when to exit")

Every satellite buy creates an **Exit Card**, stored and shown with the holding:

| Trigger | Default rule | Action |
|---|---|---|
| **Thesis break** | 2 consecutive quarters of YoY revenue decline; ROCE < ~10% for 4 quarters; guidance cut > 15% | Exit |
| **Governance break** | Auditor resignation; promoter pledge +10 pp or > 25%; regulator action; restatement | Exit (urgent alert) |
| **Geopolitical break** | A new, cited, material event (tariff, sanction, ban) that cuts the stock's addressable revenue by > 20% per the Analyst's evidence | Review → Exit |
| **Valuation stretch** | PE > 90th percentile of the stock's own 10-year band **and** > 1.5× sector | Trim to target |
| **Position drift** | > 5% of equity, or > 1.5× target | Trim to target |
| **Opportunity cost** | Same-sector candidate > 1.0 z better for 4+ weeks **and** edge > tax + cost | Swap |
| **Risk stop** (satellite only) | −25% from cost **and** below the 200-DMA **and** thesis "watch closely" | Review → Exit |
| **Goal approach** | Goal within 3 years | Glide to debt |

Rules that apply on top of the table:
- **Never stop-loss the core.** Index funds are rebalanced, not panic-sold.
- **Tax-aware timing:** non-urgent exits wait up to 30 days if the lot is about to turn long-term and the saving exceeds the cost of waiting.
- **Lot selection:** losses first, then LTCG, then STCG.

### 4.10 Step 10: Rebalancing & tax engine

- **Bands (5/25 rule):** act only when an asset class drifts ±5 pp absolute or ±25% relative. Order of operations: new cash → redirect SIPs → sell only if still in breach.
- **Indian tax engine:**
  - Indian listed equity: STCG 20% / LTCG 12.5% above a **single ₹1.25 L per FY** exemption, plus surcharge + 4% cess; grandfathering for pre-31-Jan-2018 lots.
  - Debt MFs bought after 1 Apr 2023: slab rate.
  - Foreign equity (route C): LTCG after **24 months** at 12.5%, STCG at slab; US dividends with 25% withholding, credited via Form 67.
  - International FoFs/ETFs (routes A/B): holding period and rate per current rules. **Flag for human verification in the Rulebook**, because this changed recently.
  - Insurance: maturity proceeds of non-ULIP policies issued on/after 1 Apr 2023 with aggregate premium > ₹5 L/yr, and ULIPs issued on/after 1 Feb 2021 above ₹2.5 L/yr, are taxable.
- **Annual LTCG harvesting** (Feb–Mar): book gains up to the remaining ₹1.25 L exemption and rebuy, to step up cost basis.
- **Tax-loss harvesting:** India has no codified wash-sale rule, but avoid same-day buybacks that look sham; prefer switching to a similar index fund.

### 4.11 Step 11: Insurance review (LIC & others)

For each traditional policy (endowment / money-back / whole-life), the engine computes:
- **Estimated return (IRR)** from premiums and the insurer's maturity illustration, if the user entered it. Compare with PPF and the FD rate: "This policy is expected to return ~5.1% a year; PPF currently pays more, tax-free."
- **Options:** continue / make paid-up / surrender, with the numbers for each, using the surrender value the user gets from the insurer.
- **Protection check:** is the life cover adequate? Traditional policies usually give low cover per rupee of premium, so the engine shows the term-cover gap separately.

**Guardrail:** the app never says "surrender now". It shows the comparison and the trade-offs (lost cover, surrender charges, tax) and suggests confirming the surrender value with the insurer and, where needed, a registered adviser.

### 4.12 Behavioural guardrails

≤ 3 actions/week; ≤ 1 urgent alert/day; cooling-off for 48 hours after a > 5% index move (except governance exits); show the "cost of acting" (tax + brokerage + spread) on every sell; celebrate SIP streaks and months on plan.

---

## 5. The Autonomous Research Desk (Agents + Handover on Azure AI Foundry)

### 5.1 Autonomy model: who decides what

| Decision | Owner |
|---|---|
| What to research, which sources, which follow-ups | **Agents (fully autonomous)** |
| Qualitative pass/veto/downgrade of securities; geopolitical tilt within bounds | **Agents**, through the Compliance agent gate |
| Facts and numbers (prices, ratios, yields, scheme rates) | **Tools/code only.** Agents must cite a tool output. |
| Allocation, amounts, routes, caps, exits, tax | **Policy Engine (deterministic)** |
| Changes to tax rules and scheme rates | **Human approval** (Rulebook queue) |
| Sending the weekly email | Autonomous, behind the autonomy dial + kill switch |

### 5.2 Agent roster

| # | Agent | Tier | Tools | Hands off to |
|---|---|---|---|---|
| 1 | **CIO / Orchestrator.** Weekly agenda, budgets, conflict resolution, sign-off | Mini (Strong for sign-off) | Agenda store, budget meter, dossier store | All |
| 2 | **Macro & Geopolitics.** India + US macro (rates, inflation, yields, INR, crude, FPI/DII flows, valuations) and geopolitical risk (tariffs, sanctions, conflicts, supply chains, US–India relations). Outputs **bounded tilts** (global ±5 pp, gold ±3 pp, equity ±5 pp), a "World & your money" brief, and stress-scenario selection | Mini | FRED, RBI DBIE, NSE/BSE flows, official government/trade notices, capped web grounding, valuation calculator | CIO |
| 3 | **Screener** | **No LLM** | Factor engine, point-in-time store | Fundamental Analyst |
| 4 | **News & Filings Sentinel.** Triages new filings and headlines for held, watched and shortlisted names | **Nano** | NSE/BSE announcements, SEC EDGAR RSS, issuer IR | Analyst / Compliance (urgent path) |
| 5 | **Fundamental Analyst.** Thesis, drivers, risks, **geographic revenue and geopolitical exposure**, with cited evidence | Mini | Filings reader (XBRL + PDF), ratio calculator, peer table | Valuation |
| 6 | **Valuation.** Own-history bands, reverse DCF | Mini (math in tools) | Valuation calculator | Red Team |
| 7 | **Red Team.** Falsify the thesis | Mini, **different model family** from #5 | Same evidence pack | Compliance |
| 8 | **ETF & Fund Selector.** Indian index funds/ETFs, international FoFs (incl. subscription pauses), iNAV premiums, UCITS/US ETFs | Nano (scoring in code) | ETF master, AMFI NAV, issuer factsheets, exchange iNAV | Compliance |
| 9 | **Rates & Schemes Watcher.** Small-savings rates (quarterly), EPF rate, top bank/SFB FD rates, RBI repo, T-bill yields | Nano | Official notifications, bank rate pages | **Human review queue** → `rates_reference` |
| 10 | **Rulebook Watcher.** Income-tax, LRS/TCS, SEBI, IRDAI changes | Mini | CBDT, RBI, SEBI, IRDAI feeds | **Human review queue** |
| 11 | **Risk & Compliance.** The gate: claims cited, numbers match tools, suitability language, concentration, disclaimers. **Can veto anything.** | Mini (Strong on disagreement) | Groundedness checker, schema validator | CIO |
| 12 | **Personaliser** (per user) | **No LLM** | Policy Engine on the Money Map | Communicator |
| 13 | **Communicator.** Plain-language email/push copy (English first, Hindi later) | Nano | Templates, glossary | Delivery |
| 14 | **Scorekeeper** (offline). Tracks every call vs benchmark; computes historical stress-scenario shocks from data | No LLM + Mini for error analysis | Paper portfolio, price store | CIO |

### 5.3 Handover protocol: the Dossier

Agents never pass free text to each other. Every handover is a typed, versioned **Dossier**:

```json
{
  "dossier_id": "2026-W44/IN/HDFCBANK",
  "schema_version": "1.0",
  "security": {"id": "IN:HDFCBANK", "market": "IN", "type": "EQUITY"},
  "stage": "RED_TEAM_DONE",
  "handoff": {"from": "valuation_agent", "to": "compliance_agent",
              "reason": "valuation complete", "budget_remaining_tokens": 18000},
  "claims": [
    {"id": "c1", "text": "ROCE improved from 14.2% to 16.8% YoY", "kind": "FACT",
     "evidence": [{"tool_call_id": "t-481", "source_url": "https://...", "retrieved_at": "2026-10-31T03:12Z"}]}
  ],
  "geo_exposure": {"us_revenue_pct": 4.1, "tariff_sensitive": false, "evidence": ["t-502"]},
  "verdicts": {"fundamental": "POSITIVE", "valuation": "FAIR",
               "red_team": {"strongest_bear_case": "...", "falsifier": "NIM < 3.4% for 2 quarters"},
               "compliance": "PENDING"},
  "open_questions": ["Q2 credit-cost guidance not found in filing"],
  "confidence": 0.62,
  "trace_id": "foundry-trace-..."
}
```

Handover rules:
- Every FACT must point to a `tool_call_id`, otherwise Compliance rejects it.
- At most 2 hand-backs per dossier, then it's `UNRESOLVED` and skipped this week.
- Per-dossier token and tool budgets set by the CIO.
- Checkpoint after every handover, so a crashed run resumes where it stopped.

### 5.4 Weekly timeline (IST)

```
Sat 05:30  T0  Data refresh (prices India+US, filings, yields, FX, scheme rates)   [code]
Sat 06:00  T1  Screener + Macro & Geopolitics + ETF + Rates agents (parallel)
Sat 07:00  T2  Sentinel → shortlist → Analyst → Valuation → Red Team (Batch API)
Sat 13:00  T3  Compliance gate → CIO sign-off → "Weekly Research Pack" frozen
Sat 14:00  T4  Personaliser per user on their Money Map (pure code)
Sat 15:00  T5  Communicator drafts → Compliance spot-check on a sample
Sun 08:00  T6  Weekly email + web push
Daily 07:00/18:00  Sentinel on held securities → urgent path for governance breaks
Daily 06:00        Reminders: maturities, premiums, PPF deposit window, renewals
```

T0 starts at 05:30 IST because the US market closes at 02:30 IST on Saturdays in winter (01:30 in summer), and EOD data needs time to settle. On the free-tier setup, **every step above runs as a scheduled GitHub Actions workflow**, not on Render. The run is split into resumable jobs that checkpoint to Postgres between Batch API stages (Appendix D.2).

### 5.5 Guardrails against cheap-model failure modes

| Failure mode | Guardrail |
|---|---|
| Made-up numbers | Numbers only from tools; Compliance string-matches every number against tool outputs |
| Geopolitical overreaction | Tilts are bounded (±5 pp), applied only via new money, need ≥ 2 independent cited sources, and decay unless re-confirmed each week |
| Overconfidence | Scorekeeper calibrates stated confidence against outcomes |
| Echo chamber | Red Team runs on a different model family |
| Prompt injection from fetched pages | Fetched content is quarantined as evidence and never treated as instructions; tool allowlist per agent; no agent can send email or edit rules |
| Runaway cost | Hard weekly budget, circuit breaker, cost dashboard |
| Silent drift | Golden-set eval re-run on every prompt or model change |

**Autonomy dial** (the system is fully autonomous end to end; the dial only controls who sees the output):

| Level | Behaviour | When |
|---|---|---|
| L1 Shadow | Output goes only to you + the Scorekeeper | First 8–12 weeks |
| L2 Gated auto-send | Sent automatically when Compliance passes | After the §10 bars are met |
| L3 Adaptive | CIO re-plans its own agenda from error analysis | After 6 months at L2 |

### 5.6 Model routing & cost on Azure AI Foundry

| Tier | Use | Candidate Foundry models (choose by eval) |
|---|---|---|
| T0 – no LLM | Screening, Money Map valuation, allocation, routes, tax, personalisation, statement parsing | — |
| T1 – nano | Triage, extraction, rate-table diffing, email copy | GPT-5-nano / GPT-4.1-nano / Phi-4-mini class |
| T2 – mini | Agent reasoning, analysis, red team, compliance, geopolitics | GPT-5-mini / o4-mini / DeepSeek-V3.x / Llama-4 class (two families) |
| T3 – strong | CIO sign-off and tie-breaks only (≤ 5% of tokens) | A frontier model on Foundry |

**Cost levers:**
1. Research once a week for the whole market.
2. **Global Batch** deployments (~50% cheaper).
3. Prompt caching.
4. Official free feeds over metered web search.
5. Per-dossier budgets.

**Order-of-magnitude weekly estimate.** Prices are illustrative; verify current Foundry pricing.

| Step | Volume | Tokens in / out | Tier |
|---|---|---|---|
| Sentinel triage | ~400 securities × ~6 items | ~2.0M / 0.2M | Nano |
| Analyst + Valuation | ~60 finalists (30 IN + 30 US) | ~2.4M / 0.25M | Mini |
| Red Team | 60 | ~0.9M / 0.1M | Mini |
| Macro & Geo, ETF, Rates, Compliance, CIO | — | ~0.9M / 0.09M | Mini |
| **Total** | | **~6.2M / ~0.64M** | |

At roughly US$0.05/0.40 per M tokens (nano) and US$0.25/2.00 per M (mini), that's **≈ US$2/week, ≈ US$1/week with Batch.** Communicator copy is ≈ US$0.0005 per user per week. **Data licensing and grounded web search will cost more than the models.**

### 5.7 Orchestration framework: recommendation

**LangGraph as the orchestrator, calling Foundry-deployed models through one thin `ModelRouter`** (OpenAI-compatible client pointed at Foundry):
- You already know it, and it mixes code nodes with agent nodes.
- First-class handoffs (`Command(goto=…)`), a Postgres checkpointer for multi-hour runs, and human interrupts for the Rulebook/Rates queues.
- Swapping models is a config change.
- Traces go out via OpenTelemetry: structured logs in the GitHub Actions run plus a free tracing tier (e.g., LangSmith Developer) to start, and Application Insights once you move to Azure.

**Alternative:** spike **Microsoft Agent Framework's handoff orchestration** for one week in Phase 3 if you want everything Foundry-native, and choose by the golden-set eval. Don't run both in production.

### 5.8 Self-improvement loop (bounded)

The Scorekeeper's weekly error analysis becomes next week's research checklist. Prompt and model changes ship only through the golden-set eval in CI, never live.

---

## 6. Data Layer

### 6.1 Principles

Official or licensed sources first; scraping is not a foundation for a product. Store everything point-in-time (`period_end`, `filed_at`, `known_at`). Provenance on every number.

### 6.2 Sources (evaluate cost and licensing in Phase 1)

| Data | India | US / Global |
|---|---|---|
| EOD prices | NSE/BSE bhavcopy or a licensed vendor / broker API (check redistribution terms) | Tiingo / Polygon / Alpaca data |
| Fundamentals | NSE/BSE XBRL filings + a licensed vendor for history | SEC EDGAR XBRL `companyfacts` (official, free) |
| Filings / events | NSE/BSE announcement feeds | EDGAR RSS |
| MF / ETF | **AMFI daily NAV** (official), AMC monthly portfolios, exchange iNAV | Issuer holdings files, UCITS issuer data |
| Scheme rates | Ministry of Finance small-savings notifications (quarterly), EPFO rate, RBI | Treasury yields |
| Gold | Benchmark gold rate (e.g., IBJA), SGB data | — |
| Macro / geopolitics | RBI DBIE, MOSPI, Ministry of Commerce / DGFT notices | FRED, USTR / Federal Register notices |
| FX | RBI / FBIL reference rate | FRED |

### 6.3 Backtest rebuild requirements

Point-in-time constituents (no survivorship), `known_at`-based fundamentals (no look-ahead), costs + STT + taxes, a benchmark per sleeve (Nifty 50 TRI, S&P 500 TR in INR), and metrics including information ratio, max drawdown, turnover and after-tax return. Results go to the DB and a dashboard, **not git commits**.

---

## 7. System Architecture (PWA on Render free tier → Azure later)

### 7.1 Launch architecture: Render free tier

```
   ┌──────────────── Users (phone / desktop browser, installed PWA) ────────────────┐
   └───────┬──────────────────────────────┬─────────────────────────▲──────────────┘
           │ static assets (CDN)          │ HTTPS (JWT)             │ email / web push
 ┌─────────▼──────────────┐   ┌───────────▼─────────────────┐  ┌────┴──────────────┐
 │ Render STATIC SITE     │   │ Render WEB SERVICE (free)   │  │ Email API (free   │
 │ Next.js PWA export     │   │ FastAPI · Singapore         │  │ tier, HTTP) + Web │
 │ free · never sleeps    │   │ auth, Money Map CRUD,       │  │ Push (VAPID)      │
 │ custom domain + TLS    │   │ CAS/broker import (in-mem), │  └────▲──────────────┘
 └────────────────────────┘   │ Policy Engine, valuation    │       │
                              │ sleeps after 15 min idle    │       │
                              │ NO LLM keys here            │       │
                              └───────────┬─────────────────┘       │
                                          │ SQL (RLS)               │
                              ┌───────────▼─────────────────┐       │
                              │ Neon Postgres (free)        │       │
                              │ users, Money Map, market    │       │
                              │ data, dossiers, checkpoints │       │
                              └───────────▲─────────────────┘       │
                                          │                         │
 ┌────────────────────────────────────────┴─────────────────────────┴───────────┐
 │ GitHub Actions (private repo, scheduled)                                      │
 │ weekly-research-submit → weekly-research-continue (×n) → weekly-send          │
 │ daily-reminders · market-data refresh · CI + golden-set eval                  │
 │ LangGraph research desk ──► Azure AI Foundry models (Batch)                   │
 └───────────────────────────────────────────────────────────────────────────────┘
```

Design rules forced by the free tier (details and limits in Appendix D):
- **The public API never calls an LLM and holds no Foundry keys.** All agent work happens in GitHub Actions. That keeps the API inside 512 MB and keeps keys off the internet-facing service.
- **Nothing scheduled runs on Render.** Free Render services can't run cron jobs or background workers, and they sleep. GitHub Actions does all scheduling and writes straight to Neon.
- **Postgres is the only state.** The free web service's filesystem is wiped on every restart, which already matches the "never keep uploaded files" rule.

### 7.1b Scale-up architecture: Azure (when the free tier is outgrown)

```
        ┌──────────────── Clients ─────────────────────────────────────────┐
        │  PWA (Next.js) — installable on Android / iOS / desktop           │
        │  Service worker: app shell + last snapshot cache, web push, share │
        └──────────┬────────────────────────────▲───────────────▲──────────┘
                   │ HTTPS (JWT)                 │ Email          │ Web Push (VAPID)
        ┌──────────▼──────────┐       ┌──────────┴─────┐   ┌──────┴───────┐
        │ Azure Static Web    │       │ ACS Email      │   │ Push service │
        │ Apps (PWA assets)   │       │ (SPF/DKIM/     │   │ (in API)     │
        └──────────┬──────────┘       │  DMARC)        │   └──────▲───────┘
        ┌──────────▼──────────┐       └───────▲────────┘          │
        │ API: FastAPI        │───────────────┴───────────────────┘
        │ (Container Apps)    │
        └─┬───────┬─────────┬─┘
          │       │         │
  ┌───────▼──┐ ┌──▼───────────────┐ ┌──────────────────────────────┐
  │ Policy   │ │ Postgres + RLS   │ │ Import workers (sandboxed)   │
  │ Engine + │ │ (Azure PG        │ │ CAS / broker parsers         │
  │ Valuation│ │  Flexible, India │ │ Blob (temp, auto-delete) +   │
  │ engines  │ │  region, or Neon)│ │ Defender malware scan        │
  └───────▲──┘ └──▲────────▲──────┘ └──────────────────────────────┘
          │       │        │
  ┌───────┴───────┴──┐  ┌──┴───────────────────────────────┐
  │ Weekly Research  │  │ Market Data Service              │
  │ Run (Container   │  │ (ingest, point-in-time store,    │
  │ Apps Job)        │  │  rates_reference, provenance)    │
  │ LangGraph →      │  └──────────────────────────────────┘
  │ Foundry (Batch)  │
  └──────────────────┘
  Key Vault · Entra External ID · App Insights · Daily reminders job
```

### 7.2 Repository layout (new monorepo)

```
/apps/web             PWA: Next.js + TypeScript + Tailwind + shadcn/ui (Radix) + Serwist (service worker)
/apps/admin           Internal console (slimmed Streamlit: run viewer, dossier review, Rulebook/Rates queues)
/services/api         FastAPI
/services/importer    CAS + broker parsers, staging, de-duplication (ported from today's repo)
/services/research    LangGraph research desk + agents + ModelRouter
/services/marketdata  Ingestion + point-in-time store + rates_reference
/packages/policy      Policy Engine: allocation, global routes, debt sleeve, sizing, exits, tax, insurance review
/packages/valuation   FD/RD/PPF/EPF/NSC/KVP/SCSS/MIS/SSY/SGB/insurance formulas (100% unit-tested)
/packages/schemas     JSON schemas for assets, dossiers, actions, exit cards (shared Python + TS types)
/rules                tax/*.yaml, schemes/*.yaml, allocation.yaml, exits.yaml, routes.yaml (versioned, human-approved)
/evals                golden dossiers, groundedness tests, parser fixtures (synthetic only), email snapshots
/infra                render.yaml (Render Blueprint) · .github/workflows (weekly-research-*, weekly-send, daily-reminders, ci) · Bicep later for Azure
```

### 7.3 Data model (stored in Postgres, multi-tenant)

Every user-owned table has `user_id UUID NOT NULL` + **Row-Level Security**. The API sets `app.user_id` per request, so a user physically can't read another user's rows. **This fixes P0-2.**

| Table | Purpose | Key columns |
|---|---|---|
| `users` | Login identity | id, email, auth_provider, created_at |
| `households`, `household_members` | Family money | member name, relationship (no PAN) |
| `profiles` | Advice inputs | tax_regime (old/new), slab, city_tier, risk answers, risk_profile, timezone, language |
| `goals` | Goal buckets | name, target_amount, target_year, currency (INR/USD), priority |
| `accounts` | Where money sits | institution, account_type (demat, bank, PO, EPF, NPS, insurer, US broker), **masked number (last 4 only)**, member_id |
| `assets_manual` | Every non-market asset | asset_type (enum §3.2), member_id, account_id, `details` **JSONB validated against a per-type JSON schema**, principal, current_value, value_as_of, valuation_method (formula / user / statement), maturity_date, liquidity_class, sleeve_split JSONB, status |
| `insurance_policies` | Protection + insurance-linked savings | policy_type (term/health/endowment/money-back/whole-life/ULIP), insurer, plan, sum_assured, premium, frequency, start, term, PPT, surrender_value, maturity_illustration |
| `liabilities` | Loans | type, outstanding, rate, EMI, end_date |
| `holdings_market` + `holdings_lots` | Stocks / ETFs / MFs (India + US), lot-level | security_id, account_id, qty, cost, acquired_at, source (CAS / broker / manual) |
| `asset_cashflows` | Deposits, premiums, interest, maturities | asset_id, date, amount, kind, planned/actual (drives IRR + reminders) |
| `import_jobs`, `import_items` | Upload staging & review | doc_type, status, parse_report, file_deleted_at; item status new/changed/duplicate/unknown |
| `asset_history` | Audit of edits | asset_id, changed_fields, changed_by, at |
| `networth_snapshots` | Trend charts | date, totals by sleeve |
| `target_allocations`, `actions`, `exit_cards` | Advice outputs | |
| `reminders`, `alerts_log`, `push_subscriptions` | Notifications | |
| `consents`, `audit_log` | DPDP compliance | purpose, granted_at, withdrawn_at |
| *Shared market tables* | Read-only for users | `securities`, `prices_eod`, `fundamentals_pit`, `rates_reference` (scheme, rate, effective_from/to), `fx_rates`, `dossiers`, `research_runs`, `paper_portfolio` |

Example `assets_manual.details` for an FD (validated by `schemas/assets/fd.json`):

```json
{"bank": "SBI", "principal": 500000, "rate_pct": 7.10, "start": "2025-03",
 "maturity": "2027-03", "compounding": "QUARTERLY", "payout": "CUMULATIVE", "joint": false}
```

### 7.4 Security & privacy

- PII and statements never in git and never sent to research LLMs.
- Statement passwords never stored; uploaded files deleted after parsing; only extracted holdings persist.
- Account numbers masked (last 4); no PAN storage. If PAN is ever needed (Account Aggregator), store it encrypted, opt-in.
- Encryption at rest (Neon). Secrets live in Render environment groups and GitHub Actions secrets (Key Vault after moving to Azure). Audit log of every recommendation shown or sent.
- **Data residency:** Render has no India region, so run the API in **Singapore** and Neon in the closest region it offers. India's DPDP Act generally allows cross-border transfer except to countries the government restricts; verify the current position. Move personal data to an India region (e.g., Azure Central India) when you leave the free tier, if users expect it. LLM calls carry no PII, so Global model deployments are acceptable.
- **DPDP Act:** explicit consent per purpose, one-tap data export, delete account (hard delete within the stated SLA), breach process.
- **PWA device security:**
  - Passkey (WebAuthn) for quick biometric unlock; auto-lock after 15 minutes idle.
  - The offline cache holds only the last summary snapshot (no account numbers) and is cleared on logout.
  - "Shared device" mode disables caching.

---

## 8. PWA UX Redesign

### 8.1 Why a PWA, and what it means for design

- **One codebase**, no app-store approvals, instant updates, shareable links, and it works on desktop for users who prefer a big screen for data entry.
- **Installable:** show a custom install prompt **after the first value moment** (the Money Map summary). On iOS, show a short "Share → Add to Home Screen" guide, because Safari has no install prompt.
- **iOS limits:** web push works only after the PWA is installed (iOS 16.4+), and background sync isn't available. So **email is the guaranteed alert channel**; push is a bonus.
- **Indian device reality:**
  - Design for 360 px wide, mid/low-end Android on 4G.
  - Performance budget: ≤ 170 KB gzipped initial JS; LCP < 2.5 s; charts lazy-loaded.
- **Offline:** app shell + last weekly brief + Money Map summary are cached read-only. Edits made offline queue and sync when back online.
- **Share target (Android):** "Share" a CAS PDF from Gmail/Files directly into the app.
- **Render free-tier cold starts:** the API sleeps after 15 minutes idle and can take up to about a minute to wake. The PWA must never show a blank screen while that happens:
  - Open instantly from the cached snapshot, with a small "Refreshing…" chip, and wake the API in the background.
  - Queue form saves locally and sync once the API responds.
  - Show CAS imports as "Uploading → Reading your statement → Ready to review", which covers the wake-up time.

### 8.2 Information architecture (bottom navigation, 5 tabs)

| Tab | Purpose |
|---|---|
| **Home** | "This week" card (0–3 actions), net worth, plan health, upcoming maturities/premiums |
| **Money** | Money Map: everything you own by type and member, + Add, + Import, completeness meter |
| **Plan** | Target vs actual (equity India / global, debt, gold, cash), "how much in debt", global diversification progress, goals, protection check |
| **Ideas** | India core & global core picks with the route to use, satellite stock ideas with dossiers, watchlist, "World & your money" |
| **Inbox** | Weekly briefs, urgent alerts, reminders, track record |

Settings (profile, tax regime, household, notifications, number format, export/delete data) sit behind the avatar.

### 8.3 Onboarding (value before effort)

1. Sign in (Google or email OTP; phone OTP later, since Indian SMS needs TRAI DLT registration).
2. **Money Map setup** (§3.1): upload, checklist, mini-forms. Skippable at every step.
3. Six-question risk check + tax regime (old/new) + goals (pick from: emergency, child education in India/abroad, home, retirement, wealth).
4. **First plan:** allocation donut, "how much in debt", global target and route, protection gaps.
5. Install prompt + "Your first weekly brief arrives Sunday 8 AM".

### 8.4 Key screens (wireframes, 360 px)

**Money: "What else do you have?"**
```
┌──────────────────────────────────┐
│ ← Your Money Map       80% ▓▓▓▓░ │
│ Tap everything you have:          │
│ ┌──────┐ ┌──────┐ ┌──────┐       │
│ │ EPF ✓│ │ PPF ✓│ │ FDs ✓│       │
│ └──────┘ └──────┘ └──────┘       │
│ ┌──────┐ ┌──────┐ ┌──────┐       │
│ │ Post │ │ NPS  │ │ LIC /│       │
│ │Office│ │      │ │Insur.│       │
│ └──────┘ └──────┘ └──────┘       │
│ ┌──────┐ ┌──────┐ ┌──────┐       │
│ │ Gold │ │Bonds │ │Loans │       │
│ └──────┘ └──────┘ └──────┘       │
│ ┌──────┐ ┌──────┐ ┌──────┐       │
│ │Saving│ │  US  │ │Prop- │       │
│ │  s   │ │invest│ │ erty │       │
│ └──────┘ └──────┘ └──────┘       │
│ Rough numbers are fine. You can   │
│ edit anytime.                     │
│        [ Continue (3 selected) ]  │
└──────────────────────────────────┘
```

**Add a Fixed Deposit (one card, autosaves)**
```
┌──────────────────────────────────┐
│ ← Fixed Deposit          1 of 3  │
│ Bank          [ SBI          ▾ ] │
│ Amount        [ 5L           ]   │
│               = ₹5,00,000        │
│ Interest rate [ 7.1 ] %          │
│ Matures       [ Mar ▾][ 2027 ▾ ] │
│ ▸ More details (for exact value) │
│   start month · compounding ·    │
│   payout · joint holder          │
│ ─────────────────────────────    │
│ Value today ≈ ₹5,46,200          │
│ At maturity ≈ ₹5,75,900          │
│ We'll remind you 30 days before. │
│ [ + Add another FD ]  [ Save ✓ ] │
└──────────────────────────────────┘
```

**Add an LIC policy**
```
┌──────────────────────────────────┐
│ ← LIC / Insurance policy         │
│ Type   (•) Endowment ( ) Money-back
│        ( ) Term ( ) Health ( ) ULIP
│ Insurer      [ LIC          ▾ ]  │
│ Plan         [ search plan name ]│
│ Sum assured  [ 10L ]             │
│ Premium      [ 52k ] / [ year ▾ ]│
│ Started      [ 2016 ]  Term [ 21 ]│
│ ▸ Optional: bonus so far,        │
│   surrender value, maturity      │
│   illustration (better estimate) │
│ [ Save ✓ ]                       │
└──────────────────────────────────┘
```

**Import review (after CAS + broker upload)**
```
┌──────────────────────────────────┐
│ ← Review import    NSDL CAS · Oct│
│ ✓ 14 stocks  ✓ 9 funds  ✓ 2 SGB  │
│ ⚠ 3 duplicates with Zerodha file │
│   → merged automatically   [View]│
│ ? 1 unknown: "XYZ 8.5% NCD 2028" │
│   [ It's a bond ▾ ]              │
│ ⓘ Buy prices missing for 5 stocks│
│   [Upload Zerodha tradebook] or  │
│   [Enter approx price & month]   │
│ Your file and password were not  │
│ stored.                          │
│            [ Confirm import ]    │
└──────────────────────────────────┘
```

**Home**
```
┌──────────────────────────────────┐
│ Good morning, Vikram     ⓘ  (VS) │
│ Net worth  ₹1.86 Cr  ▲0.6% wk    │
│ Plan health 74 · Global 9% → 20% │
├──────────────────────────────────┤
│ THIS WEEK · 2 actions            │
│ ┌──────────────────────────────┐ │
│ │ ➊ Start ₹10,000/mo SIP in an │ │
│ │   S&P 500 index fund         │ │
│ │   Global is 11% below target.│ │
│ │   [How to]  [Mark done] [Why]│ │
│ └──────────────────────────────┘ │
│ ┌──────────────────────────────┐ │
│ │ ➋ FD ₹3L (HDFC) matures 14   │ │
│ │   Nov → move to target-      │ │
│ │   maturity G-sec fund        │ │
│ │   [Why]  [Remind me later]   │ │
│ └──────────────────────────────┘ │
│ COMING UP                        │
│ • LIC premium ₹52,000 — 28 Nov   │
│ • PPF: deposit before 5 Dec      │
│ • XYZ results — Thu              │
├──────────────────────────────────┤
│ Home  Money  Plan  Ideas  Inbox  │
└──────────────────────────────────┘
```

**Plan: "How much in debt?" + global**
```
┌──────────────────────────────────┐
│ Your plan · Moderate             │
│      ◔ target vs actual          │
│ Equity India 38% ▸ 45%   ▼7      │
│ Equity Global 4% ▸ 12%   ▼8      │
│ Debt         49% ▸ 33%   ▲16     │
│   EPF 18 · PPF 9 · FD 12 ·       │
│   LIC 7 · PO 3                   │
│ Gold          6% ▸ 10%   ▼4      │
│ Cash          3%                 │
│ ─────────────────────────────    │
│ You already hold enough debt.    │
│ Send new money to equity & gold  │
│ until you're back on plan.       │
│ Protection: Term cover gap ₹1.2Cr›│
│ Your LIC policies: ~5.1% return ›│
└──────────────────────────────────┘
```

**Ideas: "World & your money" (geopolitics + stress test)**
```
┌──────────────────────────────────┐
│ World & your money · Week 44     │
│ Oil          ● Elevated          │
│ Rupee        ◐ Under pressure    │
│ US–India trade ◐ Tariff risk     │
│ Global risk  ● Calm              │
│ What we changed: global target   │
│ +2 pp (sources ›)                │
│ ─────────────────────────────    │
│ IF THIS HAPPENS…  (your money)   │
│ Rupee −10%        +₹1.1 L  ▲     │
│ India stocks −25% −₹16.8 L ▼     │
│ Oil shock         −₹9.4 L  ▼     │
│ US stocks −30%    −₹2.2 L  ▼     │
│ Reaching your global target cuts │
│ the India-crash hit by ~₹2.9 L.  │
└──────────────────────────────────┘
```

**Holding with its Exit Card**
```
┌──────────────────────────────────┐
│ XYZ Ltd · 18 sh · +22% · 14 mo   │
│ EXIT CARD                        │
│ ● Thesis     On track            │
│ ● Governance OK                  │
│ ● Geopolitics US rev 4% · low    │
│ ◐ Valuation  81st pct (trim >90) │
│ ● Weight     4.1% (trim > 5%)    │
│ ● Tax        Long-term since Sep │
│ Reviewed Sat 01 Nov · dossier ›  │
└──────────────────────────────────┘
```

### 8.5 Language: replace jargon labels

| Old | New |
|---|---|
| STRONG ENTER / ACCUMULATE / SMALL INITIAL / WAIT | Buy now (in parts) / Add gradually / Start small / Wait for a better price |
| HOLD / TRIM / SELL / REPLACE / BUY MORE | Keep / Take some profit / Exit / Swap for better / Add more |
| GREEN / YELLOW / RED | "How sure we are: High / Medium / Low" + the reason |
| INTACT / WEAKENED / BREACHED | On track / Watch closely / Broken |

### 8.6 Visual system & accessibility

- Status uses icon + text + colour (never colour alone); WCAG 2.2 AA; light and dark themes.
- Large tap targets (≥ 48 px); numeric keypad for amounts.
- Lakh/crore formatting by default, with an international toggle.
- English first; i18n from day one, so Hindi and Bengali come next.
- Every chart has a text equivalent ("Debt 49 percent, target 33").

### 8.7 Frontend tech

Next.js (App Router) + TypeScript, Tailwind + shadcn/ui (accessible Radix primitives), **Serwist** service worker, TanStack Query, React Hook Form + Zod (the same JSON schemas as the backend), Recharts (lazy-loaded), Web Push (VAPID), WebAuthn passkeys. Hosting: **Render Static Site** (Next.js static export: free, CDN, never sleeps) + **Render free web service** (FastAPI). Auth: a free-tier identity provider with Google + email OTP (Entra External ID, Clerk or Supabase Auth; pick the one whose free tier fits). The API verifies its JWTs, so the static site needs no server. No order execution in v1; "How to" sheets explain the steps in the user's own broker/MF app.

---

## 9. Alerts: Weekly Email + Web Push + Reminders

### 9.1 Rules

- **Weekly brief every Sunday 08:00 IST**, even when there's nothing to do.
- **Max 3 actions.** "No action needed this week" is a positive, designed state.
- **Reminders** (email + push), only for things with a date:
  - FD/RD/NSC/LIC maturity (−30 / −7 days, with a reinvest suggestion)
  - Insurance premium and renewal
  - PPF deposit before the 5th; FY-end minimum
  - SSY deposits
  - Advance-tax dates for users with capital gains
  - LTCG-harvest window (Feb–Mar)
- **Urgent:** governance or thesis breaks on holdings only (≤ 1/day).
- **Monthly "Money check-in"** email: "Upload your new CAS" + one-tap "still correct?" for manual balances.

### 9.2 Weekly email template

```
Subject:  2 actions this week · Plan health 74 (▲3)
          — or —  No action needed · You're on plan ✓
Preheader: Global is 11% below target; FD ₹3L matures 14 Nov.

Your Weekly Brief · Week 44 · 2 Nov 2026

1. YOUR NUMBER
   Net worth ₹1.86 Cr · ▲0.6% wk · ▲10.4% 1y

2. THIS WEEK'S ACTIONS (max 3)
   ➊ Start ₹10,000/mo SIP in an S&P 500 index fund — global 11% below target   [Open]
   ➋ FD ₹3L (HDFC) matures 14 Nov → target-maturity G-sec fund               [Open]

3. YOUR PLAN AT A GLANCE   (bars: India · Global · Debt · Gold vs target)

4. COMING UP
   LIC premium ₹52,000 — 28 Nov · PPF deposit before 5 Dec · XYZ results Thu

5. EXIT WATCH
   ABC — valuation 88th percentile (trim at 90th)

6. WORLD & YOUR MONEY (30 seconds)
   2 bullets on geopolitics/macro + "what we changed and why" (with sources)

7. HOW OUR CALLS ARE DOING
   Satellite ideas since launch vs Nifty 50 TRI (wins and misses)

Footer: research-only disclaimer + registration status · why you got this ·
        manage alerts · one-click unsubscribe · postal address
```

### 9.3 Delivery

- **Email through an HTTP email API**, sent from the GitHub Actions `weekly-send` and `daily-reminders` jobs:
  - Start on a free tier (e.g., Brevo or Resend; check current daily/monthly caps against your user count).
  - Move to Azure Communication Services Email or a paid tier when you outgrow it.
  - Use the provider's HTTP API, not SMTP.
  - Custom domain with SPF/DKIM/DMARC and a one-click List-Unsubscribe header.
- **MJML** templates, filled from the Policy Engine's action list. The LLM writes only the short "why" slots.
- **Web Push** via VAPID, sent from the same GitHub Actions jobs (the free API may be asleep), only to subscribed, installed PWAs.
- Snapshot tests for Gmail, Outlook and Apple Mail (light and dark).
- DPDP consent and an audit log of each email's exact content.

---

## 10. Evaluation, Usability & Trust

| Layer | Metric | Bar |
|---|---|---|
| **Usability** | Median Money Map setup time / add-an-FD time / CAS import time | < 7 min / < 30 s / < 2 min |
| **Usability** | Money Map completion (≥ 3 asset types beyond CAS) | ≥ 60% of activated users |
| **Usability** | 5-user moderated tests each phase (tasks: add EPF, add LIC, upload CAS, find "how much in debt", act on a weekly action) | ≥ 4/5 succeed unaided |
| **Parsers** | Field accuracy on a synthetic + consented-real test corpus per document type | ≥ 99% |
| **Valuation** | FD/PPF/NSC/SCSS outputs vs official calculators | Exact to the rupee |
| **Agents** | Groundedness / schema-valid dossiers / red-team recall on golden set | ≥ 99% / ≥ 99.5% / ≥ 80% |
| **Picks** | Forward paper portfolio vs Nifty 50 TRI / S&P 500 TR (INR), after costs & tax | Non-negative excess return over 12+ weeks before autonomy L2 |
| **Engagement** | Weekly email open / action completion | ≥ 50% / ≥ 30% (beta) |
| **Cost** | Research run | ≤ budget (e.g., US$10/week) |
| **Cost** | Hosting (Render + Neon + GitHub Actions + email) | US$0/month until a free limit is hit (Appendix D.4) |
| **Performance** | PWA LCP on a mid-range Android over 4G | < 2.5 s |

---

## 11. Regulatory & Compliance

> Not legal advice. Engage a SEBI-practice lawyer before public launch.

- **SEBI:**
  - Personalised buy/sell recommendations to the public are investment advice under the **SEBI (Investment Advisers) Regulations, 2013**; security-level research falls under the **SEBI (Research Analysts) Regulations, 2014**. Both require registration.
  - Recent amendments require disclosure of **AI-tool use**, and the registered entity remains responsible for AI outputs. The dossier audit trail and Compliance agent support this.
- **Insurance (IRDAI):** recommending specific insurance products is regulated intermediation. The app **analyses** the user's existing policies and shows protection gaps; it doesn't sell or recommend specific insurance products.
- **Practical paths:**
  - (a) personal/family tool first (recommended for the next 6 months);
  - (b) register as an RA or IA;
  - (c) partner with a registered entity.
  Registration also unlocks **Account Aggregator** access (FIUs must be regulated entities).
- **Data:** DPDP Act (consent, purpose limitation, deletion, breach notice). For SMS OTP, TRAI DLT registration.
- **Guardrails either way:** mandatory risk profiling and suitability, no guaranteed-return language, past-performance disclaimers, a record of every recommendation with its evidence, a complaint channel.

---

## 12. Roadmap

| Phase | Weeks | Deliverables | Exit criteria |
|---|---|---|---|
| **0 – Containment** | 0–1 | P0-1…P0-7 (private repo + history purge, allowlist, revert weights, fix scorer, fix README tax). **Don't deploy the current Streamlit app on Render publicly until P0-1 and P0-2 are done** (Appendix D.5). | Repo private and clean; only you can sign in |
| **1 – Foundations** | 1–5 | Monorepo; `render.yaml` Blueprint + GitHub Actions skeleton (Appendix D); Postgres + RLS schema on Neon (§7.3); asset JSON schemas; valuation engines; `rates_reference`; ported parsers (NSDL CAS, Zerodha, Groww) + new (CDSL, CAMS/KFin, Angel One, Upstox, ICICI); market-data service | Valuation engines exact vs official calculators; parsers ≥ 99% on the test corpus |
| **2 – Money Map PWA (first usable release)** | 3–9 | PWA live on Render (static site + free API, custom domain); PWA shell (install, offline, share target, passkeys); login; Money Map setup flow; all mini-forms; import + review; net worth, split, liquidity ladder, maturity calendar; reminders by email | 10 internal/family users complete the Money Map; setup < 7 min median |
| **3 – Advice + Research Desk (shadow)** | 6–13 | Policy Engine v1 (foundation, risk, whole-wealth allocation, debt answer, global target + route selector, insurance review, exits); Research Desk agents 1–11 on Foundry in shadow mode; Scorekeeper; golden-set eval in CI | 4 clean weekly runs; groundedness ≥ 99%; plans hand-checked for 20 personas |
| **4 – Weekly brief + actions** | 11–16 | Weekly email + web push; Home actions; Exit Cards; "World & your money" + stress tests; tradebook-based tax lots; EPF passbook / NPS statement import | Correct weekly briefs for 4 weeks for 10 users |
| **5 – Closed beta** | 16–22 | 50 users; autonomy L2 if the §10 bars are met; Hindi | Email open ≥ 50%; zero P0 incidents |
| **6 – Scale** | 22+ | Account Aggregator; NRI module (residency rulebook: PFIC, NRE/FCNR, TDS); more brokers; regulatory registration or partnership; monetisation | — |

> Phase 2 ships **useful value before any recommendations exist**: a complete, private, always-current picture of all your money, with maturity reminders. It's also the safest first release from a regulatory point of view.

---

## 13. Migration Map (current repo → new design)

| Current | Decision | Notes |
|---|---|---|
| `services/pdf_parser.py` | **Keep → `services/importer`** | Add CDSL + CAMS/KFin; synthetic fixtures only |
| `utils/broker_parser.py` | **Keep → `services/importer`** | One adapter per broker; add tradebooks + US brokers |
| `utils/direct_equity_merge.py` | **Keep → importer de-duplication** | Generalise to all asset types |
| `agents/portfolio_agents.py`, `services/mf_lookup.py`, `services/amc_adapters.py` | **Keep logic → look-through module** | Hybrid-fund split for the Money Map; AMC monthly disclosures + AMFI |
| `utils/tax_calculator.py` | **Rewrite → Policy Engine tax** | Per-FY exemption, cess/surcharge, lots, foreign equity, insurance maturity rules |
| `utils/source_health.py`, provenance in `screener_fetcher.py` | **Keep concept → market-data service** | |
| `utils/entry_calculator.py` | **Rewrite → buying rules** | Keep tranches; replace fixed stops with Exit Cards |
| `services/llm.py` bull/bear prompts | **Keep concept → Analyst / Red Team** | Re-prompt for cheap models with JSON schemas and citations |
| `agents/buy_agents.py`, `agents/monitor_agents.py` | **Replace** | Split into Policy Engine (code) + Research Desk (agents) |
| `agents/signal_agents.py`, `utils/signal_sources.py`, `utils/sector_config.py` | **Drop** | Replaced by the Macro & Geopolitics agent |
| `agents/quant_model.py`, `rules/quality_weights.yaml` | **Replace** | Multi-factor, sector-neutral, point-in-time |
| `backtest/*` | **Rebuild** | Point-in-time, survivorship-free, results to DB |
| `.github/workflows/backtest.yml` | **Replace** | CI = tests + golden-set eval; weekly run = Container Apps Job |
| `streamlit_app.py` | **Shrink → `/apps/admin`** | Internal console only |
| Root `utils/`, `agents/` duplicates, Turso config, stray `=0.40.0` file | **Delete** | |
| `README.md`, `SUMMARY.md`, `SESSION_LOG.md`, `PLATFORM_DETAILED_NOTES.md`, `*.puml` | **Archive** | Rewrite docs; scrub the password |

---

## 14. Decisions Needed From You

1. **Launch path:** personal/family tool first (recommended), or public with SEBI RA/IA registration or a partner?
2. **Scope confirmation:** resident Indians only in v1, NRI module in Phase 6?
3. **Household support in v1** (family members and per-member assets): yes (recommended) or later?
4. **Default global route** for small investors: Indian international index funds (Route A, simplest), or push users to LRS + UCITS (Route C, cheapest long-term)?
5. **Data budget:** monthly ceiling for licensed India fundamentals/prices.
6. **Foundry models:** which are deployed in your subscription and region (to finalise tiers and the Red Team's second model family).
7. **Satellite default:** 20% of equity with opt-out to 0% (recommended), or index-only by default?
8. **Brokers to support first:** recommended Zerodha, Groww, Angel One, Upstox + IBKR for US.
9. **Domain:** a custom domain for the PWA and for sending email (needed for SPF/DKIM/DMARC and deliverability; `*.onrender.com` can't send authenticated email).
10. **Free email provider and auth provider:** confirm the picks in §8.7 / §9.3 once you know your expected user count.

---

## Appendix A: `rules/exits.yaml` (shape)

```yaml
version: 1
satellite:
  thesis_break:      {revenue_yoy_negative_quarters: 2, roce_floor_pct: 10, roce_floor_quarters: 4, guidance_cut_pct: 15, action: EXIT, urgency: HIGH}
  governance_break:  {events: [AUDITOR_RESIGNATION, RESTATEMENT, REGULATOR_ENFORCEMENT], pledge_jump_pp: 10, pledge_abs_pct: 25, action: EXIT, urgency: CRITICAL}
  geopolitical_break: {addressable_revenue_hit_pct: 20, min_independent_sources: 2, action: REVIEW_THEN_EXIT}
  valuation_stretch: {own_band_percentile: 90, sector_multiple: 1.5, action: TRIM_TO_TARGET}
  drift:             {max_weight_of_equity_pct: 5, max_vs_target_ratio: 1.5, action: TRIM_TO_TARGET}
  risk_stop:         {drawdown_from_cost_pct: 25, below_dma: 200, requires_thesis: WATCH, action: REVIEW_THEN_EXIT}
  tax_wait:          {max_days_to_long_term: 30, applies_to: [valuation_stretch, drift, swap]}
core:
  stop_loss: false
  rebalance_bands: {absolute_pp: 5, relative_pct: 25}
```

## Appendix B: `rules/routes.yaml` (global route selector, shape)

```yaml
version: 1
defaults:
  route_a_max_monthly_inr: 25000      # Indian international index fund / FoF
  route_c_min_annual_inr: 500000      # LRS → international broker
  switch_to_ucits_above_usd: 60000    # US-situs estate-tax threshold
  etf_max_inav_premium_pct: 1.0       # warn / avoid route B above this
  lrs_tcs_threshold_inr: 1000000      # TCS 20% above this per FY (verify each Budget)
checks:
  - pause_if_amc_subscription_suspended: true
  - require_schedule_fa_reminder_for_route_c: true
```

## Appendix C: Communicator output contract

```json
{
  "user_ref": "u_7f3a",
  "week": "2026-W44",
  "status": "ACTIONS",
  "subject": "2 actions this week · Plan health 74 (▲3)",
  "actions": [
    {"rank": 1, "kind": "START_SIP", "instrument": "S&P 500 index fund (Route A)",
     "amount": {"value": 10000, "ccy": "INR", "per": "month"},
     "why": "Your global share is 11% below target.", "deeplink": "https://app.example/actions/a_91"}
  ],
  "no_llm_numbers": true
}
```

`no_llm_numbers: true` means every number in the email comes from the Policy Engine. The Communicator only writes `why` strings (≤ 120 characters).

## Appendix D: Deploying on Render's Free Tier

> Limits below reflect Render's published free tier as I understand it. **Check render.com/pricing and the Render docs before building**, because free-tier terms change.

### D.1 What runs where

| Component | Where | Plan | Notes |
|---|---|---|---|
| PWA (Next.js static export) | Render **Static Site** | Free | Global CDN, custom domain + TLS, never sleeps |
| API (FastAPI: auth check, Money Map CRUD, imports, Policy Engine, valuation) | Render **Web Service**, Singapore | Free | Sleeps after 15 min idle |
| Database | **Neon** Postgres | Free | Not Render's free Postgres, which expires 30 days after creation. Neon is already used by this repo. |
| Weekly research, market-data refresh, reminders, weekly send | **GitHub Actions** scheduled workflows | Free minutes | Free Render instances can't run cron jobs or background workers |
| LLMs | **Azure AI Foundry** | Pay-as-you-go (~US$1–5/week) | Called only from GitHub Actions |
| Email | HTTP email API | Free tier | See §9.3 |
| Error tracking (optional) | Sentry | Free tier | |

### D.2 Free-tier limits and how the design handles them

| Limit | Effect | Design response |
|---|---|---|
| Free web service **sleeps after 15 min** without traffic; waking can take **up to ~1 min** | First request after idle is slow | PWA opens from cache and wakes the API in the background (§8.1). Nothing scheduled depends on the API being awake. |
| **512 MB RAM, 0.1 CPU** | Heavy libraries or big PDFs can crash it or run slowly | API image excludes LangGraph, pandas and yfinance. CAS parsed with pypdf in memory, 10 MB upload cap, one parse at a time, async status polling. If parses stay slow, upgrade only this service (D.4). |
| **750 free instance-hours/month** per workspace | Enough for one always-on service, or a few that sleep | Exactly one free web service. Don't use keep-alive pings; design for cold starts instead. |
| **Ephemeral filesystem**, no persistent disk on free | Files vanish on restart/deploy | Uploads are never stored anyway; Postgres is the only state |
| **Free Render Postgres expires after 30 days** | Data loss | Use Neon free. Watch its storage cap: keep about 5 years of daily prices for ~1,100 securities (weekly bars before that), and keep 12 weeks of dossiers in the DB. |
| **No cron jobs / background workers** on free | No in-platform scheduling | GitHub Actions (below) |
| **No India region** (Singapore is closest) | Extra latency for Indian users; data-residency question | Acceptable for v1 (§7.4) |

**GitHub Actions** (private repo on GitHub Free: 2,000 minutes/month, ≤ 6 hours per job, scheduled runs can start late under load). The weekly run is split into resumable jobs that checkpoint to Postgres:

| Workflow | Schedule (UTC → IST) | Does | Est. minutes |
|---|---|---|---|
| `weekly-research-submit` | Sat 00:00 → 05:30 | Data refresh, Screener, submit Foundry Batch jobs for stage 1 | 20–40 |
| `weekly-research-continue` | Sat every 2 h, 02:00–20:00 | If the batch is done: collect results, run the next handover stage, submit the next batch, checkpoint. Exits early otherwise. | 5–15 each |
| `weekly-send` | Sun 02:30 → 08:00 | Personaliser (code) → Communicator → email + web push | 5–15 |
| `daily-reminders` | Daily 00:30 → 06:00 | Maturities, premiums, PPF window, Sentinel urgent path | 2–5 |
| `ci` | On push / PR | Tests, golden-set eval, build checks | 5–10 |

Total ≈ 400–700 minutes/month, under the 2,000 free minutes. If a Batch stage isn't finished by Sunday 02:30 UTC, `weekly-send` uses last week's approved research and says so in the email.

### D.3 `render.yaml` Blueprint (sketch, not final code)

```yaml
services:
  - type: web
    name: wealth-api
    runtime: python
    plan: free
    region: singapore
    rootDir: services/api
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /healthz
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.7
      - key: DATABASE_URL          # Neon connection string (set in dashboard)
        sync: false
      - key: AUTH_JWKS_URL
        sync: false
      - key: ALLOWED_ORIGINS
        value: https://app.yourdomain.in
      # No Foundry / LLM keys on this service by design

  - type: web
    name: wealth-pwa
    runtime: static
    rootDir: apps/web
    buildCommand: npm ci && npm run build      # next build with output: 'export'
    staticPublishPath: out
    headers:
      - path: /sw.js
        name: Cache-Control
        value: no-cache                        # service-worker updates must not be cached
      - path: /*
        name: X-Frame-Options
        value: DENY
    envVars:
      - key: NEXT_PUBLIC_API_URL
        value: https://api.yourdomain.in
```

GitHub Actions secrets: `DATABASE_URL`, `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY`, `EMAIL_API_KEY`, `VAPID_PRIVATE_KEY`.

### D.4 Upgrade path when free isn't enough

| Trigger | Upgrade | Rough cost (verify) |
|---|---|---|
| Cold starts hurt UX, or CAS parsing is too slow on 0.1 CPU | Render paid web service for the API only (no sleep, more CPU/RAM) | ≈ US$7/month entry tier |
| Neon storage or compute limits | Neon paid tier, or a paid managed Postgres | Usage-based, low tens of US$/month |
| Email volume above the free cap | Paid email tier or Azure Communication Services Email | Low |
| GitHub Actions minutes run out | Self-hosted runner, or move the research run to Azure Container Apps Jobs | Low |
| India data residency, or Azure-native ops | Move to the Azure architecture in §7.1b | — |

### D.5 Using the Render dashboard today (current Streamlit app)

The new design has no code yet, so there's nothing from this plan to deploy. If you want the **existing** Streamlit app on Render for yourself in the meantime:

1. **First** do P0-1 (make the repo private and purge the CAS PDF and password) and P0-2 (email allowlist). Otherwise you'd recreate the same leak on a new URL. Render can deploy private repos through its GitHub app.
2. New → Web Service → connect the repo, branch `main` → Runtime **Python 3** → Region **Singapore** → Instance type **Free**.
3. Build: `pip install -r requirements.txt`. Start: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`.
4. Environment: set `PYTHON_VERSION` (e.g., 3.12.x) rather than relying on Streamlit Cloud's `runtime.txt`, plus `NEON_DATABASE_URL` and the LLM keys. Google sign-in reads the nested `[auth]` section from `.streamlit/secrets.toml`, so supply it as a Render Secret File (check the mount path in Render's docs). Set the OAuth redirect to `https://<service>.onrender.com/oauth2callback`.

Expect two things:
- Streamlit sessions drop when the free service sleeps.
- LangGraph + pandas + yfinance + PDF parsing can exceed 512 MB on a full buy or monitoring run. A crash there is a free-tier limit, not an app bug.
