# Stock LangGraph Platform

A production-grade Indian equity recommendation and portfolio monitoring platform built on **LangGraph multi-agent workflows**, dual LLM providers (Anthropic Claude + OpenAI GPT), and a live NSE/Screener.in/yfinance data pipeline. The platform ingests portfolios (PDF/JSON/CSV/manual), analyses holdings, generates personalized buy recommendations, and monitors positions with tax-aware exit guidance.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Four LangGraph Workflows](#2-four-langgraph-workflows)
   - [FLOW 0 — Signal Intelligence](#flow-0--signal-intelligence-pipeline)
   - [FLOW 1 — Portfolio Ingestion](#flow-1--portfolio-ingestion--normalization)
   - [FLOW 2 — Buy Recommendations](#flow-2--portfolio-personalized-buy-recommendations)
   - [FLOW 3 — Monitoring & Decisioning](#flow-3--monitoring--decisioning)
3. [Agents Reference](#3-agents-reference)
4. [LLM Service & All Prompts](#4-llm-service--all-prompts)
5. [Quantitative Models](#5-quantitative-models)
6. [Data Sources & Live Provider](#6-data-sources--live-provider)
7. [Database Schema](#7-database-schema)
8. [Configuration Reference](#8-configuration-reference)
9. [State Definitions](#9-state-definitions)
10. [Data Models](#10-data-models)
11. [Utility Modules](#11-utility-modules)
12. [Streamlit UI Portal](#12-streamlit-ui-portal)
13. [Engine & Orchestration](#13-engine--orchestration)
14. [Source Health Monitoring](#14-source-health-monitoring)
15. [Backtesting](#15-backtesting)
16. [Walk-Forward Weight Calibration](#16-walk-forward-weight-calibration)
17. [Complete File Tree](#17-complete-file-tree)
18. [Run Locally](#18-run-locally)
19. [Streamlit Cloud Deployment](#19-streamlit-cloud-deployment)
20. [Supported Inputs](#20-supported-inputs)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT PORTAL (streamlit_app.py)                    │
│  [Sample Portfolio]  [Upload Portfolio]  [Buy Ideas]  [Portfolio]  [Monitoring] │
└────────────────────────────┬────────────────────────────────────────────────────┘
                             │  PlatformEngine.run_*()
                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PLATFORM ENGINE (services/engine.py)                       │
│  Lazy graph builders · Provider selector · Compare-Both orchestration           │
└──────┬──────────────┬──────────────┬──────────────┬──────────────────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
  ┌─────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ FLOW 0  │   │  FLOW 1  │  │  FLOW 2  │  │  FLOW 3  │
  │ Signal  │   │Portfolio │  │   Buy    │  │ Monitor  │
  │ Graph   │   │  Graph   │  │  Graph   │  │  Graph   │
  └────┬────┘   └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
  ┌─────────────────────────────────────────────────────┐
  │          AGENTS LAYER (agents/)                     │
  │  SignalAgents · PortfolioAgents · BuyAgents         │
  │  MonitoringAgents · QuantModel                      │
  └──────────────────┬──────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐       ┌─────────────────────┐
│  LLM SERVICE  │       │  LIVE DATA PROVIDER │
│  (llm.py)     │       │  (providers/live.py)│
│               │       │                     │
│ ┌───────────┐ │       │ NSE Index CSVs      │
│ │ Anthropic │ │       │ Screener.in         │
│ │  Claude   │ │       │ yfinance            │
│ │ Haiku 4.5 │ │       │ AMC pages           │
│ │ Sonnet 4.6│ │       │ mfdata.in           │
│ └───────────┘ │       │ Tickertape          │
│ ┌───────────┐ │       └──────────┬──────────┘
│ │  OpenAI   │ │                  │
│ │   GPT     │ │                  ▼
│ │ gpt-mini  │ │       ┌─────────────────────┐
│ │ gpt-5.4   │ │       │     DATABASE        │
│ └───────────┘ │       │  SQLite / Neon PG   │
└───────────────┘       │  (data/repository)  │
                        └─────────────────────┘
```

### Key Architectural Decisions

| Decision | Rationale |
|---|---|
| **LangGraph** for orchestration | Explicit node/edge graph models the multi-step analysis pipeline; supports checkpointing and state introspection |
| **Dual LLM providers** | Anthropic = risk analyst (bear-biased); OpenAI = catalyst analyst (bull-biased); synthesis resolves divergence |
| **Anthropic prompt caching** | System prompts cached at 1h TTL; saves ~90% input tokens on 50+ monitoring calls per session |
| **Screener.in for fundamentals** | Indian ROCE/EPS/promoter data more accurate than yfinance for NSE stocks |
| **4-gate stock validation** | Blocks unresolvable/delisted symbols before any LLM call to avoid wasting tokens |
| **Tax-aware monitoring** | LTCG/STCG holding period tracked so exit guidance can say "WAIT X days then EXIT" |
| **Overlap deduplication** | Symbol-level lookthrough across MF + ETF + direct equity prevents double-counting |

---

## 2. Four LangGraph Workflows

All graph definitions are in `src/stock_platform/graphs/`. Each graph is a compiled `StateGraph` with typed nodes and linear edges. Graphs are built lazily by `PlatformEngine` and cached by provider key.

---

### FLOW 0 — Signal Intelligence Pipeline

**File:** `graphs/signal_graph.py`  
**State:** `SignalState`  
**Agent Class:** `SignalAgents` (`agents/signal_agents.py`)

```
START
  │
  ├─► collect_geopolitical_signals  ─► [geo sector signals + tariff penalties]
  │
  ├─► collect_policy_signals        ─► [RBI/govt policy stance per sector]
  │
  ├─► collect_flow_sentiment        ─► [FII/DII/retail flow momentum]
  │
  ├─► detect_contrarian_signals     ─► [value trap / euphoria divergence]
  │
  └─► aggregate_signals             ─► [weighted multi-source sector conviction]
        │
       END
```

**Signal Aggregation Weights:**

| Signal Family | Weight |
|---|---|
| Geopolitical | 35% |
| Contrarian | 25% |
| Policy | 25% |
| Flow sentiment | 15% |

**Outputs:** `unified_signals` list — one record per sector with conviction `BUY / NEUTRAL / AVOID` and a 0.0–1.0 score. Persisted to `signals` table.

**Tariff Integration:** `SECTOR_GEO_OVERRIDES` (`utils/sector_config.py`) and `get_tariff_penalty()` (`utils/signal_sources.py`) apply static penalties to geo signals for tariff-exposed sectors (e.g. Metals, Auto).

---

### FLOW 1 — Portfolio Ingestion & Normalization

**File:** `graphs/portfolio_graph.py`  
**State:** `PortfolioState`  
**Agent Class:** `PortfolioAgents` (`agents/portfolio_agents.py`)

```
START
  │
  ├─► capture_user_portfolio    ─► [parse payload: MF / ETF / direct equity / macro thesis]
  │
  ├─► parse_mutual_funds        ─► [decompose MF holdings via AMC pages + mfdata.in]
  │
  ├─► decompose_etfs            ─► [decompose ETF holdings via yfinance]
  │
  ├─► normalize_exposure        ─► [aggregate symbol-level weights across all instruments]
  │
  ├─► compute_overlap           ─► [% overlap with each MF/ETF holding]
  │
  └─► identify_gaps             ─► [sector under/overweight detection]
        │
       END
```

**Gap Conviction Thresholds:**

| Sector Total Exposure | Conviction |
|---|---|
| ≥ 6% | `AVOID` — sector is full |
| 4–6% | `NEUTRAL` max |
| 2–5% gap | `BUY` |
| > 5% gap | `STRONG_BUY` |

**Inputs accepted:**
- Manual table entry (Streamlit data editor)
- `.json` with keys: `mutual_funds`, `etfs`, `direct_equities`, `macro_thesis`, `investable_surplus`, `direct_equity_corpus`
- `.csv` with columns: `holding_type`, `instrument_name`, `symbol`, `market_value`, `quantity`
- NSDL CAS PDF (password-protected, parsed by `services/pdf_parser.py`)
- Broker CSV (Zerodha Console, Groww, ICICI Direct)

---

### FLOW 2 — Portfolio-Personalized Buy Recommendations

**File:** `graphs/buy_graph.py`  
**State:** `BuyState`  
**Agent Class:** `BuyAgents` (`agents/buy_agents.py`)

```
START
  │
  ├─► load_portfolio_gate        ─► [validate portfolio age ≤ 35 days + non-empty check]
  │
  ├─► discover_universe          ─► [fetch NIFTY50/100/200 members from NSE CSV]
  │
  ├─► recommend_industries       ─► [rank sectors: gap × 0.35 + signal × 0.25 + combo × 0.40]
  │
  ├─► generate_candidates        ─► [filter universe by preferred sectors + overlap < 3%]
  │
  ├─► score_quality              ─► [quant model: ROCE, D/E, EPS, promoter, revenue growth]
  │
  ├─► filter_risk                ─► [universal hard exclusions, STRONG_AVOID list, negative EPS]
  │
  ├─► shortlist                  ─► [top N × 8 buffer by quality score]
  │
  ├─► validate_qualitative       ─► [LLM: news sentiment + signal alignment gate]
  │
  ├─► differentiate_portfolio    ─► [rank by uniqueness vs existing holdings]
  │
  ├─► assess_timing              ─► [entry signal: STRONG ENTER / ACCUMULATE / SMALL INITIAL / WAIT]
  │
  ├─► size_positions             ─► [initial tranche % + target % by conviction + risk profile]
  │
  ├─► assess_tax_costs           ─► [LTCG/STCG net return calculation]
  │
  ├─► check_confidence           ─► [confidence band: HIGH / MEDIUM / LOW / AVOID]
  │
  └─► finalize_recommendation    ─► [LLM rationale + entry snapshot + analyst verdict]
        │
       END
```

**Industry Scoring Formula:**

```python
industry_score = (gap_score * 0.35) + (market_signal * 0.25) + ((gap_score + 0.2) * 0.40)
```

**Entry Signal Ladder:**

| Condition | Signal |
|---|---|
| Quant score ≥ 0.75 + technical_score > 0.6 | `STRONG ENTER` |
| Quant score ≥ 0.65 + sector BUY | `ACCUMULATE` |
| Quant score ≥ 0.50 + any signal | `SMALL INITIAL` |
| Below thresholds or no result date (Conservative) | `WAIT` |

**Position Sizing by Entry Signal (Balanced profile baseline):**

| Signal | Initial % | Target % | Hard Cap |
|---|---|---|---|
| STRONG ENTER | 10% | 28% | 25% |
| ACCUMULATE | 8% | 22% | 25% |
| SMALL INITIAL | 6% | 18% | 25% |
| WAIT | 0% | 10% | 25% |

**Risk Profile Multipliers:**

| Profile | Initial Multiplier | Target Multiplier | Cap |
|---|---|---|---|
| Conservative | 0.65× | 0.75× | 20% |
| Balanced | 1.0× | 1.0× | 25% |
| Aggressive | 1.3× | 1.2× | 30% |

**Minimum R/R gate:** `MINIMUM_RR_RATIO = 1.2×` — stocks below are filtered in `filter_by_risk_reward()`.

**Promoter group concentration:** Maximum 1 stock per group (Adani, Tata, Reliance, Bajaj family, etc.) enforced by `apply_group_concentration_check()`.

**Compare Both mode:** `PlatformEngine.run_buy_analysis_comparison()` runs FLOW 2 for both providers in sequence, merges skipped stocks, and returns results keyed by `"anthropic"` and `"openai"`. A synthesis step calls `llm.synthesise_comparison()` per stock to produce a 4-line consensus verdict.

---

### FLOW 3 — Monitoring & Decisioning

**File:** `graphs/monitor_graph.py`  
**State:** `MonitoringState`  
**Agent Class:** `MonitoringAgents` (`agents/monitor_agents.py`)

```
START
  │
  ├─► refresh_signals        ─► [re-run FLOW 0 before monitoring to get fresh signals]
  │
  ├─► load_context           ─► [load holdings + portfolio + direct equity buy prices]
  │
  ├─► monitor_industries     ─► [sector conviction tracking + FII/DII 30/90-day flow]
  │
  ├─► monitor_stocks         ─► [per-stock: current price, P&L, overlap %, technicals]
  │
  ├─► rescore_quant          ─► [update quality vs stale baseline from Screener.in]
  │
  ├─► review_thesis          ─► [LLM: INTACT / WEAKENED / BREACHED assessment (JSON)]
  │
  ├─► drawdown_risk          ─► [volatility-based urgency: LOW / MEDIUM / HIGH / CRITICAL]
  │
  ├─► decide_actions         ─► [BUY MORE / HOLD / TRIM / SELL / REPLACE logic]
  │
  ├─► behavioural_guard      ─► [prevent over-trading; enforce concentration caps]
  │
  └─► replace_feedback       ─► [alternative stock suggestions for REPLACE actions]
        │
       END
```

**Monitoring Scope:**

| Source | Monitored? |
|---|---|
| Direct equities (statement PDF) | Yes |
| Direct equities (broker CSV upload) | Yes |
| Manual watchlist (`monitoring_watchlist` table) | Yes |
| MF look-through holdings | No — managed by fund manager |
| ETF look-through holdings | No — managed by fund manager |

**Thesis Review States:**

| Status | Meaning |
|---|---|
| `INTACT` | Thesis is holding; hold or buy more |
| `WEAKENED` | Watch closely; reduce tranche if macro worsens |
| `BREACHED` | Exit signal; news, governance, or fundamental deterioration |

**Action Decision Logic:**

| Condition | Action |
|---|---|
| Thesis INTACT + entry signal strong | `BUY MORE` |
| Thesis INTACT + hold conditions | `HOLD` |
| Thesis WEAKENED + nearing cap | `TRIM` |
| Thesis BREACHED or STRONG_AVOID override | `SELL` |
| Low-quality replacement available | `REPLACE` |

**Overlap override:** `BUY MORE` suppressed if symbol overlap ≥ 2%.  
**Tax-aware exit:** `tax_calculator.should_exit()` adds "WAIT X days then EXIT" guidance when LTCG window is close.

---

## 3. Agents Reference

### `SignalAgents` — `agents/signal_agents.py`

```python
class SignalAgents:
    def collect_geopolitical_signals(state: SignalState) -> SignalState
    def collect_policy_signals(state: SignalState) -> SignalState
    def collect_flow_sentiment(state: SignalState) -> SignalState
    def detect_contrarian_signals(state: SignalState) -> SignalState
    def aggregate_signals(state: SignalState) -> SignalState
```

Dependencies: `LiveMarketDataProvider`, `PlatformRepository`

---

### `PortfolioAgents` — `agents/portfolio_agents.py`

```python
class PortfolioAgents:
    def capture_user_portfolio(state: PortfolioState) -> PortfolioState
    def parse_mutual_funds(state: PortfolioState) -> PortfolioState
    def decompose_etfs(state: PortfolioState) -> PortfolioState
    def normalize_exposure(state: PortfolioState) -> PortfolioState
    def compute_overlap(state: PortfolioState) -> PortfolioState
    def identify_gaps(state: PortfolioState) -> PortfolioState
```

Dependencies: `MutualFundHoldingsClient`, `LiveMarketDataProvider`, `PlatformRepository`

---

### `BuyAgents` — `agents/buy_agents.py`

```python
class BuyAgents:
    # Graph nodes (14)
    def load_portfolio_gate(state: BuyState) -> BuyState
    def discover_universe(state: BuyState) -> BuyState
    def recommend_industries(state: BuyState) -> BuyState
    def generate_candidates(state: BuyState) -> BuyState
    def score_quality(state: BuyState) -> BuyState
    def filter_risk(state: BuyState) -> BuyState
    def shortlist(state: BuyState) -> BuyState
    def validate_qualitative(state: BuyState) -> BuyState
    def differentiate_portfolio(state: BuyState) -> BuyState
    def assess_timing(state: BuyState) -> BuyState
    def size_positions(state: BuyState) -> BuyState
    def assess_tax_costs(state: BuyState) -> BuyState
    def check_confidence(state: BuyState) -> BuyState
    def finalize_recommendation(state: BuyState) -> BuyState

    # Key internal helpers
    def apply_group_concentration_check(candidates)
    def get_top_n_with_replacement(shortlist, scored, n)   # 4-gate backfill
    def compute_position_size(entry_signal, quality_score, corpus)
    def filter_by_risk_reward(candidates)
    def ensure_minimum_candidates(state)
```

Constants: `AGGRESSIVE_SECTOR_TARGETS`, `PROMOTER_GROUPS`, `SHORTLIST_BUFFER_MULTIPLIER = 8`

Dependencies: `LiveMarketDataProvider`, `PlatformLLM`, `PlatformRepository`, `AppConfig`

---

### `MonitoringAgents` — `agents/monitor_agents.py`

```python
class MonitoringAgents:
    def refresh_signals(state: MonitoringState) -> MonitoringState
    def load_context(state: MonitoringState) -> MonitoringState
    def monitor_industries(state: MonitoringState) -> MonitoringState
    def monitor_stocks(state: MonitoringState) -> MonitoringState
    def rescore_quant(state: MonitoringState) -> MonitoringState
    def review_thesis(state: MonitoringState) -> MonitoringState
    def drawdown_risk(state: MonitoringState) -> MonitoringState
    def decide_actions(state: MonitoringState) -> MonitoringState
    def behavioural_guard(state: MonitoringState) -> MonitoringState
    def replace_feedback(state: MonitoringState) -> MonitoringState

    # Internal helpers
    def get_monitoring_metrics(symbol)          # ROCE/ROE/D-E for banks vs non-banks
    def compute_monitoring_score(metrics)       # 0.0–1.0 quality score
    def apply_overlap_override(action, symbol)  # suppress BUY MORE if overlap ≥ 2%
```

Dependencies: `LiveMarketDataProvider`, `PlatformLLM`, `PlatformRepository`, `TaxCalculator`, `AppConfig`

---

### `QuantModel` — `agents/quant_model.py`

```python
def compute_quality_score(symbol, fin_data, _unused=None) -> tuple[float, str]
# Returns: (score, data_quality)
#   score        — float in [0.0, 1.0]
#   data_quality — "CLEAN" | "PARTIAL" | "DEGRADED"

def apply_freshness_cap(entry_signal, fin_data, risk_profile) -> str
# Caps entry signal based on data staleness + risk profile
```

**Quality Score Formula (5 rules, weighted):**

| Metric | Default Weight | Rules |
|---|---|---|
| ROCE % | 25% | > 18 → 1.0 · > 10 → 0.6 · > 0 → 0.0 · < 0 → −0.5 |
| EPS | 25% | > 0 → 1.0 · ≤ 0 → 0.0 |
| Revenue Growth % | 20% | > 15 → 1.0 · > 8 → 0.7 · > 0 → 0.3 · ≤ 0 → 0.0 |
| Promoter Holding % | 15% | > 50 → 1.0 · > 35 → 0.7 · ≤ 35 → 0.3 |
| D/E Ratio | 15% | < 0.5 → 1.0 · < 1.0 → 0.5 · < 2.0 → 0.1 · ≥ 2.0 → 0.0 |

Weights are loaded at import time from `rules/quality_weights.yaml` (written by the calibration step); falls back to the defaults above if the file is absent.

**Data quality labels (from `_data_provenance` in fin_data):**

| Label | Condition |
|---|---|
| `CLEAN` | All 5 scored fields fetched live (0 defaults) |
| `PARTIAL` | 1–2 fields defaulted |
| `DEGRADED` | 3+ fields defaulted / no provenance dict |

**Hard rules:**
- Unknown data → `0.5` (never defaults to `1.0`)
- Negative EPS → final score hard-capped at `0.35`
- De-merged / sparse data → capped at `0.45`
- `DEGRADED` stocks excluded from BUY pool in `filter_risk` (appear on WATCHLIST only)

**Freshness cap (`apply_freshness_cap`):**
- Conservative/Balanced + no result date → converts to `WAIT` (hard stop, no recommendation)
- Aggressive + no result date → allows `SMALL INITIAL` (user accepts uncertainty)

---

## 4. LLM Service & All Prompts

**File:** `services/llm.py`  
**Class:** `PlatformLLM`

### Provider Tiers

| Tier | Anthropic Model | OpenAI Model | Use Cases |
|---|---|---|---|
| **Fast** | `claude-haiku-4-5-20251001` | `gpt-5.4-mini` | Buy rationale, monitoring action, news summary |
| **Reasoning** | `claude-sonnet-4-6` | `gpt-5.4` | Industry narrative, qualitative gate, thesis review, synthesis |

Anthropic system prompts are cached using `cache_control={"type": "ephemeral", "ttl": "1h"}` — high-value on 50+ monitoring calls per session.  
OpenAI uses server-side automatic caching.

---

### Prompt 1 — `buy_rationale()` — Dual Analyst Views

**Tier:** Fast · **Max tokens:** 700 · **Temperature:** 0.35  
**Called:** Once per recommended stock (3–4× per buy session)

#### Anthropic — Risk Analyst (Bear-Biased) System Prompt

```
You are a BEAR-BIASED risk analyst reviewing a buy recommendation.
Your job is to find reasons why the quality score is NOT enough to justify buying.
Think like someone who was burned by overconfident buys before.

Answer exactly 4 numbered questions:
1. What is the most likely thesis failure mode in 12 months?
2. What happens to fair value if earnings disappoint by 15%?
3. What governance or regulatory risk is being underpriced?
4. What is the PE reversion downside if sector re-rates?

RULES:
- Label each data point as FACT / DERIVED / INFERENCE
- Quantify margin of safety in rupees, not just %
- No PE-only thesis — must include earnings quality check
- OUTPUT FORMAT:
  RISK VERDICT: [AVOID / WAIT / ACCUMULATE / BUY]
  Bear case: [one sentence]
  We are wrong if: [specific falsifiable condition]
  Supporting metric: [one data point]
```

#### Anthropic — Aggressive Investor Variant System Prompt

```
You are a balanced investor reviewing a buy recommendation.
DEFAULT RULE: If PAT is growing, PE is reasonable, and the stock has corrected 15%+,
the default stance is ACCUMULATE unless you find a specific disqualifying risk.

Answer exactly 4 numbered questions:
1. What is the specific catalyst or growth driver that the market is underpricing?
2. What is the realistic downside if the thesis is wrong (not worst-case)?
3. What governance or regulatory risk should be tracked?
4. Does the PE have fundamental support (earnings growth, sector re-rating)?

OUTPUT FORMAT:
  RISK VERDICT: [AVOID / WAIT / ACCUMULATE / BUY]
  Bear case: [one sentence]
  We are wrong if: [specific falsifiable condition]
  Supporting metric: [one data point]
```

#### OpenAI — Catalyst Analyst (Bull-Biased) System Prompt

```
You are a BULL-BIASED catalyst analyst reviewing a buy recommendation.
Your job is to find the specific trigger that will re-rate this stock upward.
Think like an investor who has missed too many multi-baggers by being too cautious.

Answer exactly 4 numbered questions:
1. What is the single most specific near-term catalyst (name it precisely)?
2. Why is the market underpricing this right now?
3. What does the stock look like in N months if the catalyst fires?
4. What is the most compelling entry price and why?

RULES:
- Name SPECIFIC catalyst (not "continued growth" or "sector tailwind")
- Identify the market's exact misunderstanding
- State the exit signal — what event proves you wrong
- OUTPUT FORMAT:
  CATALYST VERDICT: [AVOID / WATCHLIST / ACCUMULATE / BUY NOW]
  Bull case: [one sentence]
  Market misread: [what the market is getting wrong]
  Exit signal: [specific event that invalidates thesis]
```

**User prompt includes:**
- Factual snapshot block (price, ROCE, D/E, EPS, revenue growth, PE vs 5yr avg, promoter %, sector gap, valuation reliability)
- Recent news (3-bullet summary from `fetch_stock_news_context()`)
- Macro flow block (FII/DII sector momentum from `format_macro_flow_for_prompt()`)
- Investment horizon and LTCG/STCG implication
- Risk profile instruction from `RISK_PROMPT_HINTS` (`utils/risk_profiles.py`)
- Divergence flag if revenue growth ≠ PAT growth

---

### Prompt 2 — `synthesise_comparison()` — Consensus Verdict

**Tier:** Reasoning (always Anthropic Sonnet 4.6) · **Max tokens:** 400  
**Called:** Once per stock in Compare Both mode  
**Rate-limit handling:** 3 retries, 65s × attempt exponential backoff  
**Budget controls:** Factual snapshot capped at 4 000 chars; news block capped at 3 000 chars

```
System: You are a senior investment committee chair resolving a debate between two analysts.
Output exactly 4 lines, no headers, no bullets:
  Line 1: VERDICT [AVOID/WATCHLIST/ACCUMULATE/BUY] · Confidence [HIGH/MEDIUM/LOW]
  Line 2: Key agreement: [what both analysts agree on]
  Line 3: Resolution: [which analyst's view wins and WHY — one specific reason]
  Line 4: Flip condition: [single event that would reverse this verdict]

Rules:
- If RISK VERDICT and CATALYST VERDICT diverge by more than one step, default to the
  conservative view unless the bull case has a named specific catalyst within 90 days
- Never invent data; reference only what appears in the input
- The flip condition must be falsifiable
```

---

### Prompt 3 — `monitoring_rationale()` — Action Assessment

**Tier:** Fast · **Max tokens:** 200 · **Temperature:** 0.2  
**Called:** ~50× per monitoring session (one per held stock)  
**Output:** JSON only (prevents misparse of action words in free text)

```
System: You are a portfolio monitoring assistant.
Return ONLY valid JSON with this exact structure — no markdown, no explanation:
{"action": "...", "severity": "...", "rationale": "..."}

Valid actions: BUY MORE, HOLD, TRIM, SELL, REPLACE
Valid severity: LOW, MEDIUM, HIGH, CRITICAL

Rules:
- action must be one of the 5 valid values exactly as written
- severity reflects urgency of action required
- rationale is a single sentence (max 25 words)
- Return ONLY the JSON object, nothing else
```

---

### Prompt 4 — `industry_reasoning()` — Sector Prioritization Narrative

**Tier:** Reasoning · **Max tokens:** 350 · **Temperature:** 0.3  
**Called:** Once per buy session

```
System: You are a macro strategist briefing a fund manager on sector allocation priorities.
Write 3–4 sentences explaining the current sector prioritization for Indian equities.
Focus on: (1) the macro thesis driving sector selection, (2) the top 2–3 sectors and why
they are preferred right now, (3) any sectors to avoid and the key risk.
Be direct and specific. No bullet points. No headers. No hedging language.
```

**User prompt includes:** Top 6 sectors by gap × signal score, macro thesis string, portfolio summary.

---

### Prompt 5 — `fetch_stock_news_context()` — News Summary

**Tier:** Fast · **Max tokens:** 300  
**Called:** Once per stock in rationale phase  
**Tool used:** Anthropic `web_search_20250305`  
**Cached:** Per-instance to avoid duplicate searches in one session  
**Fallback:** `"No material news in last 30 days."`

```
System: You are a financial news summarizer.
Search for recent news about [COMPANY_NAME] ([SYMBOL]) Indian stock in the last 30 days.
Return exactly 3 bullet points of the most important recent developments.
Each bullet: one sentence, start with the date if available.
If no material news, return: "No material news in last 30 days."
Focus only on: earnings, management changes, regulatory actions, major contracts, M&A.
```

---

### Prompt 6 — `fetch_critical_news()` — News Risk Gate

**Tier:** Reasoning · **Max tokens:** 1 000  
**Called:** For verdicts except `AVOID` (post-rationale safety gate)  
**90-day window**  
**Cached:** Per-instance

```
System: You are a risk compliance officer reviewing a stock for material risks.
Search for news about [COMPANY_NAME] ([SYMBOL]) in the last 90 days.
Return ONLY valid JSON:
{
  "material_risks_found": true/false,
  "flags": ["FLAG_TYPE", ...],
  "revised_verdict_suggestion": "AVOID|WATCHLIST|no_change",
  "summary": "one sentence"
}
Valid flag types: CEO_CHANGE, REGULATORY, CREDIT, OPERATIONAL, LEGAL, EARNINGS, SAFETY
If no material risks, return material_risks_found: false, flags: [], revised_verdict_suggestion: "no_change"
```

---

### Prompt 7 — `qualitative_analysis()` — LLM News Validation Gate

**Tier:** Reasoning · **Max tokens:** 200 · **Temperature:** 0.1  
**Called:** In `validate_qualitative` node of FLOW 2  
**Output:** JSON (deterministic at temp 0.1)

```
System: You are a stock screening assistant performing a qualitative validation check.
Evaluate whether this stock should pass or fail qualitative screening.
Return ONLY valid JSON:
{"approved": true/false, "confidence": "HIGH|MEDIUM|LOW", "reasoning": "one sentence"}

Rules:
- approved: false if news contains fraud, regulatory ban, credit downgrade, or CEO arrest
- approved: false if sector signal is AVOID with no offsetting catalyst
- confidence: HIGH if strong evidence either way, LOW if ambiguous
- reasoning: cite the specific piece of evidence driving the decision
- Return ONLY the JSON, no markdown
```

**User prompt includes:** Candidate fundamentals (symbol, ROCE, EPS, sector), latest news summary, sector signal conviction, quality score.

---

### Prompt 8 — `thesis_review()` — Monitoring Status Gate

**Tier:** Reasoning · **Max tokens:** 200 · **Temperature:** 0.1  
**Called:** In `review_thesis` node of FLOW 3  
**Output:** JSON (auditable, deterministic)

```
System: You are a portfolio risk manager performing thesis review for existing holdings.
Assess whether the investment thesis is still intact.
Return ONLY valid JSON:
{"status": "INTACT|WEAKENED|BREACHED", "reasoning": "one sentence"}

Rules:
- INTACT: fundamentals holding, sector conviction positive, no adverse news
- WEAKENED: 1–2 of fundamentals, sector signal, or news have turned negative
- BREACHED: fundamental deterioration + adverse news, or STRONG_AVOID trigger
- STRONG_AVOID flag in hard exclusion list always → BREACHED regardless of LLM view
- reasoning: cite the primary factor driving the assessment
- Return ONLY the JSON, no markdown
```

**User prompt includes:** Holding quant score (current vs entry), sector signals, latest news, sector correction flag, days held, unrealised P&L %.

---

### Prompt Summary Table

| Method | Provider | Tier | Max Tokens | Temp | Cache | Purpose |
|---|---|---|---|---|---|---|
| `buy_rationale()` | Anthropic / OpenAI | Fast | 700 | 0.35 | Yes (1h) | Dual analyst rationale |
| `synthesise_comparison()` | Anthropic Sonnet | Reasoning | 400 | 0.4 | Yes (1h) | Consensus verdict |
| `monitoring_rationale()` | Either | Fast | 200 | 0.2 | Yes (1h) | Action + severity JSON |
| `industry_reasoning()` | Either | Reasoning | 350 | 0.3 | Yes (1h) | Sector narrative |
| `fetch_stock_news_context()` | Anthropic | Fast | 300 | 0.5 | Per-instance | 3-bullet news summary |
| `fetch_critical_news()` | Anthropic | Reasoning | 1 000 | 0.1 | Per-instance | News risk gate JSON |
| `qualitative_analysis()` | Either | Reasoning | 200 | 0.1 | Yes (1h) | Validation gate JSON |
| `thesis_review()` | Either | Reasoning | 200 | 0.1 | Yes (1h) | INTACT/WEAKENED/BREACHED |

---

## 5. Quantitative Models

### Entry Level Calculation — `utils/entry_calculator.py`

```python
def calculate_entry_levels(symbol, current_price, analyst_target) -> dict:
    # Returns: entry_price, stop_loss, target_price, risk_reward_ratio
    # Entry = current_price × 0.97  (3% below current for limit orders)
    # Stop loss = entry × (1 - stop_pct)  where stop_pct varies by risk profile
    # R/R = (target - entry) / (entry - stop)
    # Minimum R/R gate: 1.2×
```

### PE History & Valuation Reliability — `utils/pe_history_fetcher.py`

```python
def get_pe_historical_context(symbol) -> dict:
    # Returns: current_pe, 5yr_avg_pe, signal (CHEAP/FAIR/EXPENSIVE)
    # CHEAP: current_pe < 5yr_avg × 0.85
    # EXPENSIVE: current_pe > 5yr_avg × 1.20
    # Reliability label from valuation_reliability.py (HIGH/MEDIUM/LOW)
```

### Technical Signal — `utils/technical_signals.py`

```python
def compute_technical_signal(symbol) -> dict:
    # Returns: technical_score (0.0–1.0), momentum label
    # Inputs: 52W high/low position, distance from DMA
    # 52W momentum: (current - 52W_low) / (52W_high - 52W_low)
    # DMA strength: current / DMA200 ratio
```

### Tax Calculator — `utils/tax_calculator.py`

```python
def calculate_pnl(symbol, buy_price, current_price, buy_date, quantity) -> dict:
    # Returns: gross_pnl, days_held, tax_type (STCG/LTCG), net_pnl_est

def should_exit(days_held, unrealised_pnl_pct, action) -> str:
    # Returns: "EXIT NOW" | "WAIT X days then EXIT" | "HOLD"
    # LTCG threshold: 365 days (10% tax vs STCG 15%)
    # If days_held in [340, 364]: "WAIT N days then EXIT" to save STCG tax
```

---

## 6. Data Sources & Live Provider

**File:** `providers/live.py`  
**Class:** `LiveMarketDataProvider`

### Source Priority Chain

| Data Type | Primary Source | Fallback | Cache TTL |
|---|---|---|---|
| Index members (NIFTY50/100/200) | NSE CSV archive | `STATIC_INDEX_MEMBERS` | 7 days |
| Fundamentals (ROCE, EPS, D/E, promoter, revenue growth) | Screener.in (`utils/screener_fetcher.py`) | consolidated/standalone toggle | 24 hours |
| Current price, 52W high/low, analyst target | yfinance | None | 4 hours |
| Financial statements (fallback) | yfinance income statement | None | 24 hours |
| MF holdings decomposition | AMC official pages (`services/amc_adapters.py`) | mfdata.in API | 7 days |
| ETF holdings | yfinance | None | 7 days |
| Earnings result dates | Neon DB cache → Tickertape | yfinance | 30 days |
| FII/DII sector flows | `utils/fii_dii_fetcher.py` | cached last value | 24 hours |
| PE history | `utils/pe_history_fetcher.py` via `pe_history_cache` table | None | 30 days |

### Key Methods

```python
class LiveMarketDataProvider:
    def get_index_members(index_name: str) -> list[str]
    def get_stock_snapshot(symbol: str) -> dict          # price, market_cap, sector, name
    def get_financials(symbol: str) -> dict              # roce, d/e, eps, rev_growth, promoter
    def get_fund_holdings(fund_name, month) -> list[dict]
    def get_etf_holdings(etf_name, month) -> list[dict]
    def normalize_symbol(symbol: str) -> str             # de-merged symbol resolution
    def get_geopolitical_signals() -> list[dict]
    def get_policy_signals() -> list[dict]
    def get_flow_signals() -> list[dict]
```

### Caching Architecture

Three-tier caching:
1. **In-memory** (`_snapshot_cache`, `_financial_cache`, `_index_cache`) — request-scoped, avoids duplicate API calls within one graph run
2. **SQLite/Neon `cache_entries` table** — TTL-based persistent cache across sessions
3. **`pe_history_cache` table** — dedicated store for PE time-series data (30-day TTL)

---

## 7. Database Schema

**Files:** `data/schema.py`, `data/db.py`, `data/repository.py`

### Dual-Dialect DDL

The schema uses conditional SQL that works on both SQLite (local) and PostgreSQL/Neon (cloud). Placeholder binding: `?` for SQLite, `%s` for PostgreSQL (handled automatically by `NeonWrapper`).

### Tables

| Table | Purpose | Key Columns |
|---|---|---|
| `app_state` | Key/value config store | `key`, `value`, `updated_at` |
| `raw_holdings` | Pre-normalization user input | `holding_type`, `instrument_name`, `symbol`, `market_value`, `quantity` |
| `normalized_exposure` | Symbol-level aggregated weights | `symbol`, `weight_pct`, `source` (MF/ETF/DIRECT), `sector` |
| `direct_equity` | Buy prices for monitoring | `symbol`, `buy_price`, `quantity`, `buy_date`, `tax_type` |
| `overlap_scores` | % overlap per symbol with MF/ETF | `symbol`, `overlap_pct`, `source_fund` |
| `identified_gaps` | Sector underweight opportunities | `sector`, `gap_pct`, `conviction` |
| `signals` | Sector conviction records | `family`, `sector`, `conviction`, `score`, `source`, `as_of_date`, `payload` |
| `recommendations` | Buy recommendation records | `symbol`, `action`, `score`, `confidence_band`, `rationale`, `payload`, `created_at` |
| `monitoring_actions` | Per-stock monitoring verdicts | `symbol`, `action`, `severity`, `rationale`, `urgency`, `payload`, `created_at` |
| `monitoring_watchlist` | User-added watchlist stocks | `symbol`, `company_name`, `sector`, `note` |
| `cache_entries` | Generic TTL cache | `cache_key`, `value`, `expires_at` |
| `skipped_stocks` | Validation-failed candidates | `symbol`, `reason`, `gate`, `created_at` |
| `pe_history_cache` | PE time-series data | `symbol`, `pe_data`, `fetched_at` |

### Connection Abstraction

```python
# data/db.py
class SQLiteWrapper:           # Local file-based, default mode
class NeonWrapper:             # PostgreSQL via Neon (psycopg2 bridged to sqlite3 interface)

def database_connection(db_path, neon_url=None) -> ContextManager[Connection]
```

`NeonWrapper` automatically translates `?` → `%s` so all repository queries are write-once for both backends.

### Repository API

```python
class PlatformRepository:
    # Portfolio
    def save_raw_holdings(holdings: list[dict])
    def save_normalized_exposure(exposure: list[dict])
    def save_overlap_scores(overlaps: list[dict])
    def save_identified_gaps(gaps: list[dict])
    def load_portfolio_context() -> dict          # aggregate: normalized + overlap + gaps

    # Signals
    def replace_signals(signals: list[SignalRecord])
    def load_signals(family=None) -> list[dict]

    # Recommendations
    def save_recommendation(rec: RecommendationRecord)
    def load_recommendations(limit=20) -> list[dict]

    # Monitoring
    def save_monitoring_action(action: MonitoringAction)
    def load_monitoring_actions(limit=50) -> list[dict]
    def save_direct_equity(holding: dict)
    def load_direct_equity() -> list[dict]

    # Cache
    def get_cache(key: str) -> str | None
    def set_cache(key: str, value: str, ttl_seconds: int)
    def get_skipped_stocks() -> list[dict]
```

---

## 8. Configuration Reference

**File:** `src/stock_platform/config.py`  
**Class:** `AppConfig` (dataclass with `__post_init__` env loading)

### Environment Variables

```env
# ── Anthropic Claude ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=your_key_here
LLM_FAST_MODEL=claude-haiku-4-5-20251001        # default
LLM_REASONING_MODEL=claude-sonnet-4-6           # default

# ── OpenAI GPT ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY=your_key_here
OPENAI_FAST_MODEL=gpt-5.4-mini                  # default
OPENAI_REASONING_MODEL=gpt-5.4                  # default
OPENAI_TIMEOUT_SECONDS=30                       # default

# ── Database ────────────────────────────────────────────────────────────────────
NEON_DATABASE_URL=postgresql://...              # overrides SQLite when set
DB_PATH=/data/platform.db                       # SQLite path (default)

# ── Portfolio Rules ─────────────────────────────────────────────────────────────
MAX_PORTFOLIO_AGE_DAYS=35                       # refresh required after this
MAX_DIRECT_STOCKS=4                             # max direct equity positions
MAX_SINGLE_STOCK_PCT=30                         # single stock concentration cap
MAX_SECTOR_PCT=30                               # single sector concentration cap
TOTAL_DIRECT_EQUITY_PCT_CAP=25                  # total direct equity allocation cap

# ── MF API ──────────────────────────────────────────────────────────────────────
MF_API_BASE_URL=https://mfdata.in/api/v1        # default
MF_HOLDINGS_TIMEOUT_SECONDS=20                  # default
```

Streamlit Cloud: values read from `st.secrets` using identical key names as `.env`.

---

## 9. State Definitions

**File:** `src/stock_platform/state.py`  
All states are `TypedDict` with `total=False` (all keys optional — LangGraph nodes only update relevant keys).

```python
class SignalState(TypedDict, total=False):
    trigger: str
    macro_thesis: str
    geo_signals: list[dict]
    policy_signals: list[dict]
    flow_signals: list[dict]
    contrarian_signals: list[dict]
    unified_signals: list[dict]
    run_summary: dict

class PortfolioState(TypedDict, total=False):
    payload: dict
    mutual_fund_exposure: list[dict]
    etf_exposure: list[dict]
    normalized_exposure: list[dict]
    overlap_scores: list[dict]
    identified_gaps: list[dict]
    run_summary: dict

class BuyState(TypedDict, total=False):
    request: dict                     # {index_name, horizon_months, risk_profile, top_n}
    portfolio_context: dict
    universe: list[dict]
    preferred_industries: list[dict]
    industry_narrative: str
    candidates: list[dict]
    scored_candidates: list[dict]
    risk_filtered_candidates: list[dict]
    shortlist: list[dict]
    overlap_filtered: list[str]
    differentiated_shortlist: list[dict]
    timing_assessments: list[dict]
    allocations: list[dict]
    tax_assessment: dict
    confidence: dict
    recommendations: list[dict]
    run_summary: dict

class MonitoringState(TypedDict, total=False):
    request: dict
    portfolio_context: dict
    industry_reviews: list[dict]
    stock_reviews: list[dict]
    quant_scores: list[dict]
    thesis_reviews: list[dict]
    drawdown_alerts: list[dict]
    actions: list[dict]
    behavioural_flags: list[dict]
    replacement_prompt: dict
    run_summary: dict
```

---

## 10. Data Models

**File:** `src/stock_platform/models.py`

```python
@dataclass
class SignalRecord:
    family: str          # "geo" | "policy" | "flow" | "contrarian" | "unified"
    sector: str
    conviction: str      # "BUY" | "NEUTRAL" | "AVOID"
    score: float         # 0.0–1.0
    source: str
    horizon: str
    detail: str
    as_of_date: str
    signal_key: str | None
    payload: dict

@dataclass
class PortfolioInput:
    mutual_funds: list[dict]
    etfs: list[dict]
    direct_equities: list[dict]
    macro_thesis: str
    investable_surplus: float
    direct_equity_corpus: float

@dataclass
class BuyRequest:
    index_name: str      # "NIFTY50" | "NIFTY100" | "NIFTY200"
    horizon_months: int
    risk_profile: str    # "Conservative" | "Balanced" | "Aggressive"
    top_n: int

@dataclass
class RecommendationRecord:
    symbol: str
    company_name: str
    sector: str
    action: str          # "STRONG BUY" | "ACCUMULATE" | "WATCHLIST" | "AVOID"
    score: float
    confidence_band: str # "HIGH" | "MEDIUM" | "LOW"
    rationale: str
    payload: dict        # entry_levels, analyst_target, entry_signal, tax_notes, etc.

@dataclass
class MonitoringAction:
    symbol: str
    action: str          # "BUY MORE" | "HOLD" | "TRIM" | "SELL" | "REPLACE"
    severity: str        # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    rationale: str
    urgency: str
    payload: dict        # pnl, days_held, tax_type, thesis_status, etc.
```

---

## 11. Utility Modules

**Location:** `src/stock_platform/utils/`

| Module | Purpose |
|---|---|
| `stock_validator.py` | 4-gate validation: NOT_FOUND / NO_DATA / PRICE_MISSING / OK |
| `screener_fetcher.py` | Screener.in fundamentals (ROCE, D/E, EPS, revenue growth, promoter) |
| `entry_calculator.py` | Entry price, stop-loss, target, R/R ratio |
| `pe_history_fetcher.py` | 5yr PE avg, CHEAP/FAIR/EXPENSIVE signal |
| `technical_signals.py` | 52W momentum, DMA strength, technical_score |
| `fii_dii_fetcher.py` | FII/DII sector flow (30/90 day) + prompt formatter |
| `risk_profiles.py` | Per-profile R/R minimums, staleness caps, `RISK_PROMPT_HINTS` |
| `symbol_resolver.py` | De-merged symbol mapping (e.g. TATAMOTORS → TMCV) |
| `valuation_reliability.py` | PE reliability label (HIGH/MEDIUM/LOW + flags) |
| `evidence_scoring.py` | Confidence score from data freshness + analyst consensus |
| `sector_config.py` | `SECTOR_GEO_OVERRIDES`, `governance_risk_blocks()`, `get_sector()` |
| `result_date_fetcher.py` | Earnings date (Neon + Tickertape + yfinance chain) |
| `index_config.py` | NSE index URLs + `STATIC_INDEX_MEMBERS` fallback |
| `stock_context.py` | Builds factual snapshot dict for LLM prompt injection |
| `recommendation_resolver.py` | Preliminary verdict logic before LLM rationale |
| `direct_equity_merge.py` | Deduplication of direct equity rows across ingestion runs |
| `signal_sources.py` | Tariff signal definitions + `get_tariff_penalty()` |
| `rules.py` | `clamp()`, `conviction_from_score()`, `parse_iso_datetime()` |
| `bse_codes.py` | BSE code ↔ NSE symbol mapping |
| `cache_init.py` | Cache table initialization helpers |
| `tax_calculator.py` | LTCG/STCG computation, `should_exit()` guidance |

---

## 12. Streamlit UI Portal

**File:** `streamlit_app.py`

### Tab Structure

| Tab | Functionality |
|---|---|
| **Sample Portfolio** | Load `seed_demo_data()` for instant walkthrough without uploading a statement |
| **Upload Portfolio** | PDF/JSON/CSV upload, manual entry table, password-protected NSDL CAS parsing |
| **Buy Ideas** | Run FLOW 2; model selector (Anthropic / OpenAI / Compare Both); Top N, risk profile, horizon; shows per-stock analyst card with entry levels |
| **Portfolio** | View normalized exposure, overlap scores, sector gaps; broker CSV upload for buy prices |
| **Monitoring** | Run FLOW 3; LLM selector; displays P&L, days held, tax type, urgency, tax-aware exit guidance per stock |

### UI Features

- **Live LLM model selector** — shows which model generated each output
- **Compare Both** — side-by-side Anthropic vs OpenAI recommendation columns with a third Analyst Synthesis panel
- **Skipped stocks expander** — `⚠️ Skipped stocks` shows validation failures from `skipped_stocks` table
- **Tax-aware exit badges** — `WAIT X days then EXIT` displayed inline with monitoring actions
- **Monitoring watchlist** — persistent across sessions via `monitoring_watchlist` SQLite table
- **Broker statement upload** — Zerodha Console / Groww / ICICI Direct CSV for buy-price enrichment

---

## 13. Engine & Orchestration

**File:** `services/engine.py`  
**Class:** `PlatformEngine`

```python
class PlatformEngine:
    def __init__(self, config: AppConfig | None = None):
        # 1. Load .env + Streamlit secrets
        # 2. Create PlatformRepository (SQLite or Neon based on config)
        # 3. Instantiate LiveMarketDataProvider
        # 4. Build PlatformLLM (default provider: Anthropic)
        # 5. Initialize NSDLCASParser

    # Lazy graph builders — built once, cached by provider key
    def _build_signal_graph() -> CompiledGraph
    def _build_portfolio_graph() -> CompiledGraph
    def _build_buy_graph(llm_provider: str) -> CompiledGraph
    def _build_monitor_graph(llm_provider: str) -> CompiledGraph

    # Public workflow entry points
    def run_signal_refresh(trigger, macro_thesis) -> dict          # FLOW 0
    def ingest_portfolio(payload: dict) -> dict                    # FLOW 1
    def parse_portfolio_pdf(pdf_path, password) -> dict            # NSDL CAS parse → FLOW 1
    def run_buy_analysis(request, llm_provider) -> dict            # FLOW 2 (single provider)
    def run_buy_analysis_comparison(request) -> dict               # FLOW 2 (both providers)
    def run_monitoring(request, llm_provider) -> dict              # FLOW 3
    def seed_demo_data() -> None                                   # test portfolio only
```

**Compare Both sequencing:** `run_buy_analysis_comparison()` calls FLOW 2 for `"anthropic"` then `"openai"`, merges skipped stocks (deduped by symbol), then iterates per stock calling `llm.synthesise_comparison()` with 20-second spacing to avoid rate-limit collisions. Returns `{"anthropic": [...], "openai": [...], "skipped": [...]}`.

---

## 14. Source Health Monitoring

**File:** `src/stock_platform/utils/source_health.py`  
**Class:** `SourceHealthChecker`

A pre-flight gate that probes all three live data sources before FLOW 2 (buy recommendations) and FLOW 3 (monitoring) run. The check is gated on `config.neon_enabled` — it runs only in production (Neon DB configured); local/SQLite environments skip it automatically.

### How it works

```
get_source_health(repo)
  │
  ├─► probe_screener()   — 10 known-good symbols, fetch via screener_fetcher
  ├─► probe_yfinance()   — 10 known-good symbols, yf.Ticker.fast_info.last_price
  └─► probe_nse_csv()    — HEAD request to NSE NIFTY50 CSV URL
       │
       └─► overall_status = HEALTHY / DEGRADED / FAILED
```

Results cached for 30 minutes in `cache_entries` (key `source_health_v1`).

### Status thresholds

| Status | Condition | Effect |
|---|---|---|
| `HEALTHY` | ≥ 80% probes pass across all sources | Run proceeds normally |
| `DEGRADED` | ≥ 50% probes pass | Run proceeds; all recommendations flagged `data_quality_warning: true` |
| `FAILED` | < 50% probes pass | `ValueError` raised — run aborted before any LLM calls |

### Public API

```python
from stock_platform.utils.source_health import get_source_health, assert_source_health

report = get_source_health(repo)
# report["overall_status"] → "HEALTHY" | "DEGRADED" | "FAILED"
# report["screener"]       → {"status": ..., "pass_rate": 0.9, "sampled": 10, "passed": 9}
# report["yfinance"]       → {"status": ..., "pass_rate": ...}
# report["nse_csv"]        → {"status": ..., "pass_rate": ...}

assert_source_health(repo)   # raises ValueError on FAILED; returns report otherwise
```

### Payload impact

When DEGRADED, `finalize_recommendation` attaches:
```json
{"data_quality_warning": true}
```
to every recommendation payload. The Streamlit UI can surface this as a caution badge.

---

## 15. Backtesting

**Package:** `backtest/`  
**Entry point:** `python -m backtest.run_backtest`  
**Full documentation:** [backtest/README.md](backtest/README.md)

The backtest harness replays FLOW 2 (buy recommendations) against 2 years of historical NSE price and fundamental snapshots to measure how well the quant pipeline performs against the NIFTY benchmark.

### Architecture

```
backtest/
├── __init__.py
├── snapshot.py      # Fetch & store 2y weekly prices + quarterly fundamentals
├── replay.py        # HistoricalDataProvider + per-Monday replay loop
├── scorer.py        # Forward-return computation, hit rates, alpha vs NIFTY
├── run_backtest.py  # CLI entry point (argparse, JSON logging, exit codes)
├── calibrate.py     # Walk-forward weight calibration (grid search)
└── README.md        # Full documentation
```

### Four new database tables

| Table | Purpose |
|---|---|
| `historical_fundamentals` | Quarterly ROCE, EPS, D/E, revenue growth snapshots |
| `historical_prices` | Daily/weekly close prices (includes `NIFTY` benchmark row) |
| `backtest_runs` | Summary per replay run (hit rates, alpha, weights hash) |
| `backtest_recommendations` | Per-symbol per-date recommendations with forward returns |

Migration: `repo.initialize()` creates all tables with `CREATE TABLE IF NOT EXISTS` — just restart the engine on an existing DB.

### Hit rate definition

```
hit = (stock_forward_return - nifty_forward_return) > 2.0%
```

Measured at 3m (13w), 6m (26w), and 12m (52w) windows from recommendation date.

### CLI usage

```bash
# Full pipeline: snapshot → replay → score
python -m backtest.run_backtest \
  --mode full \
  --start-date 2024-05-09 \
  --end-date   2026-05-09

# Individual phases
python -m backtest.run_backtest --mode snapshot
python -m backtest.run_backtest --mode replay  --start-date 2024-05-09 --end-date 2026-05-09
python -m backtest.run_backtest --mode score   --run-id <run_id>

# Fail CI if 6-month hit rate below floor
python -m backtest.run_backtest --mode full --hit-rate-floor 0.45
# Exit code 1 if hit_rate_6m < 0.45; exit code 2 on any unhandled exception
```

### HistoricalDataProvider

`replay.py` defines `HistoricalDataProvider`, a drop-in replacement for `LiveMarketDataProvider` that reads from the snapshot tables.

```python
provider = HistoricalDataProvider(repo, replay_date=start_date)
for monday in mondays:
    provider.today = monday          # MUST update before each iteration
    provider._price_cache.clear()   # MUST clear to avoid stale prices
    # ... invoke graph ...
```

### LLM nodes in backtest mode

`request["skip_llm_nodes"] = True` short-circuits:
- `validate_qualitative` — all candidates approved (quality_score as confidence)
- `finalize_recommendation` — lightweight record written without LLM rationale

Token cost: zero. Replay is fully deterministic.

### GitHub Actions

`.github/workflows/backtest.yml` runs weekly on Sunday 02:00 UTC (or manually via `workflow_dispatch`). Requires `NEON_DATABASE_URL` secret. Commits a summary markdown to `backtest/results/` after each run.

---

## 16. Walk-Forward Weight Calibration

**File:** `backtest/calibrate.py`  
**Entry point:** `python -m backtest.calibrate --run-id <run_id>`  
**Output:** `rules/quality_weights.yaml`

After a full backtest run, calibration finds the 5 quality-score weights (ROCE, EPS, revenue growth, promoter, D/E) that maximise the 6-month hit rate on a training window, then validates against a held-out window to catch overfitting.

### Algorithm

```
1. Load backtest_recommendations for a completed run
2. Split: first 18 months = train, remaining = validate
3. Grid-search over 5 weights in 0.05 increments, sum == 1.0
   (~10,626 valid combinations — C(24,4), no scipy needed)
4. For each combination: recompute quality scores, re-rank, compute hit rate vs NIFTY
5. Report top-10 by train hit rate; flag any where val < train − 5pp (overfit)
6. Best = highest train hit rate that is NOT overfit
7. Write best config to rules/quality_weights.yaml
```

### Output file (`rules/quality_weights.yaml`)

```yaml
version: "v20260510"
calibrated_at: "2026-05-10T00:00:00Z"
weights_hash: "a1b2c3d4"
train_period:    ["2024-05-09", "2025-11-09"]
validate_period: ["2025-11-09", "2026-05-09"]
train_hit_rate_6m:    0.54
validate_hit_rate_6m: 0.51
weights:
  roce:           0.30
  eps:            0.25
  revenue_growth: 0.20
  promoter:       0.15
  debt_equity:    0.10
```

### How weights are loaded at runtime

`quant_model.py` reads the YAML at import time:

```python
_WEIGHTS = _load_weights()   # reads rules/quality_weights.yaml; falls back to defaults
```

Weights take effect immediately on the next engine restart — no code change required. A weights hash is stored in `backtest_runs` for reproducibility.

### Running calibration

```bash
# After a completed full backtest run:
python -m backtest.calibrate --run-id <run_id_from_replay>

# Optional: change training window (default 18 months)
python -m backtest.calibrate --run-id <run_id> --train-months 12
```

The calibration result (best weights + top-10 table) is printed as JSON and the YAML is written automatically.

---

## 17. Complete File Tree

```
stock-langgraph-platform/
├── streamlit_app.py                     # Streamlit portal entry point
├── pyproject.toml                       # Package config (stock-langgraph-platform)
├── requirements.txt                     # Pinned dependencies
├── .env                                 # API keys (not committed)
├── .streamlit/
│   ├── secrets.toml.example             # Secrets template for Streamlit Cloud
│   └── config.toml                      # Theme / server config
└── src/stock_platform/
    ├── __init__.py
    ├── config.py                        # AppConfig dataclass
    ├── models.py                        # SignalRecord, PortfolioInput, BuyRequest, etc.
    ├── state.py                         # TypedDict state definitions
    │
    ├── agents/
    │   ├── __init__.py
    │   ├── signal_agents.py             # SignalAgents — 5 nodes
    │   ├── portfolio_agents.py          # PortfolioAgents — 6 nodes
    │   ├── buy_agents.py                # BuyAgents — 14 nodes + helpers
    │   ├── monitor_agents.py            # MonitoringAgents — 10 nodes + helpers
    │   └── quant_model.py               # compute_quality_score, apply_freshness_cap
    │
    ├── graphs/
    │   ├── __init__.py
    │   ├── signal_graph.py              # build_signal_graph() → CompiledGraph
    │   ├── portfolio_graph.py           # build_portfolio_graph() → CompiledGraph
    │   ├── buy_graph.py                 # build_buy_graph() → CompiledGraph
    │   └── monitor_graph.py             # build_monitor_graph() → CompiledGraph
    │
    ├── data/
    │   ├── __init__.py
    │   ├── schema.py                    # DDL — SQLite + PostgreSQL dialects
    │   ├── db.py                        # SQLiteWrapper, NeonWrapper, connect_database()
    │   └── repository.py               # PlatformRepository — all CRUD + caching
    │
    ├── providers/
    │   ├── __init__.py
    │   ├── live.py                      # LiveMarketDataProvider
    │   └── demo.py                      # Legacy demo provider (deprecated)
    │
    ├── services/
    │   ├── __init__.py
    │   ├── llm.py                       # PlatformLLM — 8 public methods, dual provider
    │   ├── engine.py                    # PlatformEngine — orchestration + graph lifecycle
    │   ├── mf_lookup.py                 # MutualFundHoldingsClient
    │   ├── amc_adapters.py              # AMC page scrapers (HDFC, SBI, ICICI, Axis, etc.)
    │   └── pdf_parser.py               # NSDLCASParser — encrypted CAS PDF extraction
    │
    └── utils/
        ├── __init__.py
        ├── bse_codes.py                 # BSE ↔ NSE symbol mapping
        ├── cache_init.py                # Cache table setup helpers
        ├── direct_equity_merge.py       # Direct equity deduplication
        ├── entry_calculator.py          # Entry price, stop-loss, target, R/R
        ├── evidence_scoring.py          # Confidence from data quality
        ├── fii_dii_fetcher.py           # FII/DII sector flows + prompt formatter
        ├── index_config.py              # NSE URLs + static fallback members
        ├── pe_history_fetcher.py        # PE history + CHEAP/FAIR/EXPENSIVE signal
        ├── recommendation_resolver.py   # Pre-LLM verdict logic
        ├── result_date_fetcher.py       # Earnings result dates (chain: Neon → Tickertape → yfinance)
        ├── risk_profiles.py             # Risk config + RISK_PROMPT_HINTS
        ├── rules.py                     # clamp, conviction_from_score, parse_iso_datetime
        ├── screener_fetcher.py          # Screener.in fundamentals scraper (+ _data_provenance)
        ├── sector_config.py             # SECTOR_GEO_OVERRIDES + governance_risk_blocks
        ├── signal_sources.py            # Tariff signal definitions
        ├── source_health.py             # Pre-flight source health gate (NEW)
        ├── stock_context.py             # Factual snapshot builder for LLM prompts
        ├── stock_validator.py           # 4-gate validation (NOT_FOUND / NO_DATA / PRICE_MISSING)
        ├── symbol_resolver.py           # De-merged symbol resolution
        ├── tax_calculator.py            # LTCG/STCG P&L + should_exit() guidance
        ├── technical_signals.py         # 52W momentum + DMA strength
        └── valuation_reliability.py     # PE reliability label
│
├── backtest/                            # NEW — backtest harness package
│   ├── __init__.py
│   ├── snapshot.py                      # 2y price + quarterly fundamental snapshots
│   ├── replay.py                        # HistoricalDataProvider + replay loop
│   ├── scorer.py                        # Forward-return computation + hit rates
│   ├── run_backtest.py                  # CLI entry point (--mode snapshot|replay|score|full)
│   ├── calibrate.py                     # Walk-forward weight calibration (grid search)
│   └── README.md                        # Backtest-specific documentation
│
├── rules/                               # NEW — calibrated model config
│   └── quality_weights.yaml             # YAML weight config (written by calibrate.py)
│
├── .github/workflows/
│   └── backtest.yml                     # NEW — weekly backtest cron + workflow_dispatch
│
└── tests/
    ├── test_source_health.py            # NEW — 15 unit tests for source_health.py
    └── test_backtest_scorer.py          # NEW — forward-return, hit-rate, calibration tests
```

---

## 18. Run Locally

### Prerequisites

- Python 3.11+
- At least one of: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

### Configure `.env`

```env
# ── Anthropic Claude (recommended) ──────────────────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# Optional model overrides (defaults shown):
# LLM_FAST_MODEL=claude-haiku-4-5-20251001
# LLM_REASONING_MODEL=claude-sonnet-4-6

# ── OpenAI GPT (alternative / comparison) ───────────────────────────────────
OPENAI_API_KEY=your_openai_api_key_here
# Optional model overrides (defaults shown):
# OPENAI_FAST_MODEL=gpt-5.4-mini
# OPENAI_REASONING_MODEL=gpt-5.4
```

Both keys can be set simultaneously to enable the **Compare Both** feature.

### Launch

```bash
streamlit run streamlit_app.py
```

If no API keys are set, the platform runs with deterministic text fallbacks for rationale generation while the market-data pipeline remains live.

---

## 19. Streamlit Cloud Deployment

This repo deploys on Streamlit Community Cloud with `streamlit_app.py` as the entrypoint.

**Deployment settings:**
- Repository: `sudipta0311/stock`
- Branch: `codex/streamlit-cloud-ready`
- Main file path: `streamlit_app.py`
- Python version: `3.11`

Add runtime secrets in `Settings → Secrets` using [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) as template.

**Google Sign-In (OAuth):**

```toml
[auth]
redirect_uri = "https://<your-app-subdomain>.streamlit.app/oauth2callback"
cookie_secret = "replace-with-a-long-random-string"

[auth.google]
client_id = "xxxxxxxxxxxx.apps.googleusercontent.com"
client_secret = "GOCSPX-xxxxxxxxxxxxxxxxxxxxxxxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Add these exact redirect URIs to the Google OAuth client:
- `https://<your-app-subdomain>.streamlit.app/oauth2callback`
- `http://localhost:8501/oauth2callback` (local testing)

**Current deployment:**
- Active branch: `codex/streamlit-cloud-ready`
- App URL: `https://sudipta0311-stock-streamlit-app-codexstreamlit-cloud-rea-ynukc3.streamlit.app/`

**Redeploy checklist:**
1. Confirm app is pointing to branch `codex/streamlit-cloud-ready`
2. Use `Reboot app` or `Redeploy latest commit` after pushing
3. Re-check secrets after any delete/recreate cycle
4. Verify `server_metadata_url` present in `[auth.google]` if Google login fails
5. If `redirect_uri_mismatch`, add the exact callback URI shown in browser to Google OAuth client
6. Re-upload portfolio statement if monitoring shows empty after a new deployment
7. Direct equity holdings entered manually persist across PDF re-ingestions

---

## 20. Supported Inputs

| Format | Description |
|---|---|
| **Manual entry** | Editable tables in Streamlit UI |
| **JSON** | Keys: `macro_thesis`, `investable_surplus`, `direct_equity_corpus`, `mutual_funds`, `etfs`, `direct_equities` |
| **CSV** | Columns: `holding_type`, `instrument_name`, `symbol`, `market_value`, `quantity` |
| **PDF** | Encrypted NSDL CAS statements; parser extracts holdings directly into FLOW 1 |
| **Broker CSV** | Zerodha Console, Groww, ICICI Direct, or any CSV with `symbol + quantity + average_buy_price` |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `langgraph` | ≥ 0.2.67 | Multi-agent workflow orchestration |
| `streamlit` | ≥ 1.42.0 | Web portal UI |
| `anthropic` | ≥ 0.40.0 | Claude API (Haiku 4.5 + Sonnet 4.6) |
| `openai` | ≥ 1.75.0 | GPT API (gpt-5.4-mini + gpt-5.4) |
| `yfinance` | ≥ 0.2.0 | Current price, 52W high/low, analyst targets |
| `pandas` | ≥ 2.2.3 | Data manipulation |
| `pypdf` | ≥ 6.10.0 | NSDL CAS PDF parsing |
| `beautifulsoup4` | ≥ 4.12.3 | Screener.in + AMC page scraping |
| `requests` | ≥ 2.32.0 | HTTP calls to NSE, mfdata.in, Tickertape |
| `python-dotenv` | ≥ 1.0.1 | `.env` loading |
| `cryptography` | ≥ 46.0.0 | PDF decryption |
| `Authlib` | ≥ 1.3.2 | Google OAuth for Streamlit Cloud |
| `psycopg2` | (optional) | Neon PostgreSQL connection |
