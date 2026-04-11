# Stock Platform Detailed Notes

## Purpose

This document captures the current implementation state of the LangGraph-based stock recommendation and monitoring platform built from [stock_platform_v5.puml](c:/Project/App/stock_platform_v5.puml).

It includes:

- architecture overview
- key modules
- LLM configuration
- encrypted PDF ingestion support
- AMC-specific look-through adapter status
- end-to-end test results using the provided NSDL CAS statement
- current limitations
- recommended next steps

## High-Level Architecture

The platform is organized around four major flows from the PlantUML sequence diagram:

1. `FLOW 0: Signal Intelligence`
2. `FLOW 1: Portfolio Ingestion`
3. `FLOW 2: New Buy Recommendation`
4. `FLOW 3: Portfolio Monitoring`

### Backend stack

- `LangGraph` for orchestration
- `SQLite` for persistence
- `Streamlit` for the user portal
- `OpenAI API` for narrative reasoning/rationale generation
- live provider inputs via NSE constituent files, `yfinance`, and AMC/fallback holdings adapters

### Main entry points

- App UI: [streamlit_app.py](c:/Project/App/streamlit_app.py)
- Engine facade: [engine.py](c:/Project/App/src/stock_platform/services/engine.py)
- Signal graph: [signal_graph.py](c:/Project/App/src/stock_platform/graphs/signal_graph.py)
- Portfolio graph: [portfolio_graph.py](c:/Project/App/src/stock_platform/graphs/portfolio_graph.py)
- Buy graph: [buy_graph.py](c:/Project/App/src/stock_platform/graphs/buy_graph.py)
- Monitoring graph: [monitor_graph.py](c:/Project/App/src/stock_platform/graphs/monitor_graph.py)

## Core Module Layout

### Agents

- [signal_agents.py](c:/Project/App/src/stock_platform/agents/signal_agents.py)
  Handles geopolitical, policy, flow, contrarian, and unified signal generation.

- [portfolio_agents.py](c:/Project/App/src/stock_platform/agents/portfolio_agents.py)
  Handles MF parsing, ETF decomposition, normalization, overlap scoring, and gap identification.

- [buy_agents.py](c:/Project/App/src/stock_platform/agents/buy_agents.py)
  Handles portfolio-aware candidate generation, scoring, risk filtering, timing, sizing, tax, and recommendation output.

- [monitor_agents.py](c:/Project/App/src/stock_platform/agents/monitor_agents.py)
  Handles monitoring refresh, thesis review, drawdown checks, actions, and behavioural guardrails.

### Services

- [llm.py](c:/Project/App/src/stock_platform/services/llm.py)
  Standard OpenAI client wrapper used for rationale generation.

- [pdf_parser.py](c:/Project/App/src/stock_platform/services/pdf_parser.py)
  Parses encrypted NSDL CAS statements into structured portfolio payloads.

- [mf_lookup.py](c:/Project/App/src/stock_platform/services/mf_lookup.py)
  Live fund holdings resolver. Prefers official AMC adapters first, then tries generic fallback lookup.

- [live.py](c:/Project/App/src/stock_platform/providers/live.py)
  Runtime market-data provider. Uses NSE archive constituent files for the buy universe and `yfinance` for snapshots, price context, financials, sector breadth, and monitoring signals.

- [amc_adapters.py](c:/Project/App/src/stock_platform/services/amc_adapters.py)
  Official AMC adapter framework. Currently includes official Mirae Asset parsing.

### Persistence

- [schema.py](c:/Project/App/src/stock_platform/data/schema.py)
- [repository.py](c:/Project/App/src/stock_platform/data/repository.py)

## Environment and LLM Setup (v6.0)

The app now uses a tiered Anthropic Claude strategy via [.env](c:/Project/App/.env).

Required key:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

Optional model overrides:

```env
LLM_FAST_MODEL=claude-haiku-4-5-20251001      # default
LLM_REASONING_MODEL=claude-sonnet-4-6          # default
```

Config handling is in [config.py](c:/Project/App/src/stock_platform/config.py).
The service class is [PlatformLLM](c:/Project/App/src/stock_platform/services/llm.py).

### LLM tier routing

| Method | Model | Volume | Prompt cached |
|---|---|---|---|
| `buy_rationale` | Haiku 4.5 | 3-4 calls/session | Yes — ~90% input saving |
| `monitoring_rationale` | Haiku 4.5 | ~50 calls/session | Yes |
| `industry_reasoning` | Sonnet 4.6 | 1 call/session | No |
| `qualitative_analysis` | Sonnet 4.6 | 4-8 calls/session | No |
| `thesis_review` | Sonnet 4.6 | ~50 calls/session | No |

All methods fall back to deterministic text when `ANTHROPIC_API_KEY` is absent or a call fails.

## PDF Statement Ingestion

### Supported source

The platform now supports encrypted NSDL CAS PDF ingestion using a password.

Tested statement:

- [NSDLe-CAS_109102284_FEB_2026.PDF](c:/Project/App/tests/NSDLe-CAS_109102284_FEB_2026.PDF)

Password used for testing:

- `AYFPS8467G`

### Extracted sections

The parser currently extracts:

- total consolidated portfolio value
- direct equity holdings from demat
- ETF holdings from the demat statement section
- mutual fund folio holdings and current values
- statement month

### Parser behavior

The parser converts the PDF into the same payload shape expected by the ingestion graph:

- `mutual_funds`
- `etfs`
- `direct_equities`
- `investable_surplus`
- `direct_equity_corpus`
- `statement_month`

## AMC-Specific Look-Through

### Goal

Replace proxy scheme-level exposure with actual stock-level holdings using official AMC sources.

### Current approach

The holdings resolution order is:

1. official AMC adapter
2. generic fallback resolver
3. proxy scheme-level holding

### Official AMC adapter support today

#### Mirae Asset

Implemented using official scheme pages that expose embedded `jsonHoldings` arrays in page source.

Supported examples:

- `Mirae Asset Large Cap Fund - Direct Plan`
- `Mirae Asset Equity Savings Fund - Direct Plan`

Source implementation:

- [amc_adapters.py](c:/Project/App/src/stock_platform/services/amc_adapters.py)

### Remaining AMCs from the tested statement

These are present in the statement and still need stronger official adapter coverage:

- Axis
- Edelweiss
- ICICI Prudential
- Kotak
- Motilal Oswal
- PGIM India
- Parag Parikh / PPFAS
- SBI
- UTI
- quant

### Why they are not fully done yet

The official data exposure pattern varies by AMC:

- some expose machine-readable JSON in HTML
- some expose monthly PDF factsheets
- some expose image-heavy factsheets
- some use protected or unstable download endpoints
- some pages are reachable but not cleanly parseable without AMC-specific logic

Because of that, exact AMC-specific support needs to be built one AMC at a time.

## End-to-End Test Summary

### Test runner

Reusable E2E script:

- [run_pdf_e2e.py](c:/Project/App/tests/run_pdf_e2e.py)

### Flow executed

1. Parse encrypted NSDL CAS PDF
2. Build portfolio payload
3. Run signal refresh
4. Run portfolio ingestion
5. Run buy recommendation flow
6. Run monitoring flow
7. Inspect recommendation and action outputs

### Latest successful result (live-runtime branch — 2026-04-11)

```json
{
  "parsed": {
    "mutual_funds": 45,
    "etfs": 1,
    "direct_equities": 12,
    "statement_month": "2026-02",
    "direct_equity_corpus": 1437879.0,
    "investable_surplus": 527780.71
  },
  "ingestion": {
    "normalized_exposure": 54,
    "overlap_scores": 54,
    "identified_gaps": 13,
    "holdings_sources": {
      "unresolved": 44,
      "official:miraeasset": 21
    }
  },
  "buy": {
    "recommendation_count": 3
  },
  "monitoring": {
    "action_count": 54,
    "action_distribution": { "BUY MORE": 2, "HOLD": 50, "SELL": 2 }
  },
  "signals": {
    "unified": 13, "geo": 6, "policy": 5, "flow": 1, "contrarian": 4
  }
}
```

### Recommendation results from the run

- The buy universe now comes from official NSE archive constituent CSVs rather than a baked-in symbol list.
- The quality scorer consumes live provider financials, and stocks with total live-data failure return `0.0` instead of inheriting demo-perfect scores.
- Monitoring still stays restricted to direct holdings and watchlist names, but the quant/news context now comes from live provider methods rather than demo records.

### Monitoring results

Monitoring now runs only against the direct-holdings/watchlist universe produced by the current portfolio context.
Behavioural guardrails still trigger after repeated monitoring runs in a single day.
LLM rationales still use deterministic fallback when the selected provider API key is absent.

## What Is Accurate Today

### Strongly working

- encrypted PDF ingestion
- direct equity extraction
- ETF extraction from statement
- mutual fund folio extraction and valuation
- LangGraph orchestration
- Streamlit workflow
- provider-differentiated Anthropic/OpenAI rationale generation
- live NSE + `yfinance` recommendation and monitoring inputs

### Partially accurate

- overlap analysis for schemes resolved through official AMC adapters
- sector-level approximation for unresolved schemes

### Not yet statement-accurate across all funds

- full stock look-through for all AMCs in the statement
- exact AMFI/AMC monthly portfolio alignment for every scheme

## Known Limitations

### Python 3.14 warning

There is still a runtime warning from `langchain_core` around Python `3.14` compatibility:

- this did not block the E2E run
- it should still be considered for production hardening

### Generic fallback is still used for many funds

`unresolved` holdings sources mean the system had to fall back rather than use official AMC look-through.

### Some generated symbols are placeholders

For unresolved holdings, proxy or synthesized symbols may appear, such as:

- `ET_576558F2`
- hashed or normalized company-name symbols

These are safe for orchestration but not ideal for investment-grade overlap analytics.

## Recommended Next Steps

### Priority 1

Build official adapters for the AMCs that dominate the statement by value and count:

1. ICICI Prudential
2. PPFAS / Parag Parikh
3. Axis
4. SBI
5. UTI

### Priority 2

For each adapter:

- identify official monthly portfolio source
- parse top holdings and, if available, full holdings
- normalize issuer names to platform symbols
- mark adapter source explicitly in ingestion output

### Priority 3

Improve overlap fidelity:

- distinguish India equity, global equity, debt, liquid, and hybrid sleeves
- exclude non-equity sleeves from stock-overlap scoring
- treat FoFs separately where underlying holdings are unavailable

### Priority 4

Persist AMC holdings cache locally:

- cache by scheme + month
- avoid repeated live scraping
- make repeated E2E runs faster and more stable

## How To Re-Run the Statement Test

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the E2E script:

```bash
python tests/run_pdf_e2e.py
```

Start the UI:

```bash
streamlit run streamlit_app.py
```

Then upload the PDF in the `Portfolio Ingestion` tab and provide the password.

## Suggested Documentation Follow-Up

The next good companion document would be an AMC integration tracker with:

- AMC name
- official source URL
- source type: html/json/pdf/image
- parser status
- scheme coverage
- known parsing caveats

That would make the remaining path to full statement-accurate overlap much easier to manage.
