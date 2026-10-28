# C14: Research Desk foundation: ModelRouter (Azure AI Foundry), budgets, call/tool logging, Dossier schema + store, LangGraph skeleton with handoffs and a Postgres checkpointer, resumable run CLI

**Plan:** `docs/UPGRADE_PLAN.md` §5.1 (who decides what), §5.3 (the Dossier and handover rules), §5.5 (guardrails), §5.6 (model tiers, Flex, budgets), §5.7 (LangGraph + ModelRouter), Appendix D.2 (resumable stages across GitHub Actions jobs). Read `AGENTS.md` first. **Nothing in this card may be imported by `wealth.api`.** Everything lives in `wealth.research` and the `research` extra.

## Goal
The plumbing every agent card builds on:
- a cheap, reliable, budgeted way to call the Foundry models with schema-validated JSON
- an auditable record of every model and tool call
- the typed Dossier handed between agents
- a LangGraph pipeline that can stop between stages and resume in a later GitHub Actions job

## Facts (already verified by the reviewer against the live Foundry resource)
- Endpoint: OpenAI-compatible v1, `FOUNDRY_ENDPOINT` (e.g. `https://<resource>.openai.azure.com/openai/v1/`). Auth header: `api-key: <FOUNDRY_API_KEY>`.
- Deployments: `gpt-6-luna` (accepts `reasoning_effort` and `service_tier: "flex"`) and `deepseek-v4-flash` (different family; don't send `reasoning_effort`).
- Chat Completions works for both: `POST {endpoint}chat/completions` with `model = <deployment name>`, `max_completion_tokens`.
- East US retail prices per 1M tokens: luna input $0.10 / output $0.50 (Flex $0.05 / $0.25; cached input $0.01); deepseek-v4-flash ≈ $0.15 / $0.31.

## Scope

### 1. Dependencies (`research` extra)
`openai>=1.60` (use `OpenAI(base_url=FOUNDRY_ENDPOINT, api_key=FOUNDRY_API_KEY, default_headers={"api-key": FOUNDRY_API_KEY})`), `langgraph>=1.0`, `langgraph-checkpoint-postgres>=2.0`, `tenacity>=9`. Verify they install on Python 3.14. If one doesn't, choose the nearest working version and say so.

### 2. `rules/models.yaml`
```yaml
version: 1
tiers:
  nano:    {deployment: gpt-6-luna,        reasoning_effort: low}
  mini:    {deployment: gpt-6-luna,        reasoning_effort: medium}
  strong:  {deployment: gpt-6-luna,        reasoning_effort: high}
  redteam: {deployment: deepseek-v4-flash}
prices_usd_per_1m:   # for budgeting only; source: Azure Retail Prices API, East US, 2026-10-27
  gpt-6-luna:        {input: "0.10", output: "0.50", cached_input: "0.01", flex_factor: "0.5"}
  deepseek-v4-flash: {input: "0.15", output: "0.31"}
flex: {enabled_for_modes: [weekly], max_flex_retries: 3, fallback_to_default: true}
timeouts_s: {connect: 10, read: 120}
```

### 3. `wealth/research/router.py`: `ModelRouter`
- `complete(tier, messages, *, schema: type[BaseModel] | None, max_output_tokens, run_ctx) -> RouterResult`, where `RouterResult` = `{parsed, raw_text, usage, cost_usd, model, service_tier, latency_ms, call_id}`.
- **Structured output:** when `schema` is given, send `response_format={"type":"json_schema","json_schema":{"name":..., "schema": <pydantic schema>, "strict": true}}`. If the deployment rejects `json_schema` (DeepSeek may), fall back to `{"type":"json_object"}` with the schema in the system prompt. Always validate with Pydantic. On validation failure, do **one** repair attempt (send the error back), then raise `SchemaError`.
- **Flex:** in `weekly` mode send `service_tier="flex"` for luna tiers. On 429/"capacity" errors, retry up to `max_flex_retries` with exponential backoff and jitter, then (if `fallback_to_default`) retry once on the default tier and record `service_tier` accordingly. Other retries: 429/5xx/timeouts, 3 attempts. Never retry 400/401/403.
- **Budget:** `RunBudget(max_usd, max_calls)` in `run_ctx`. Estimate cost from `usage` and the YAML prices (flex factor applied when used). Raise `BudgetExceeded` *before* a call when the estimated spend (prompt tokens × input price + `max_output_tokens` × output price) would exceed the remaining budget.
- **Logging (audit):** every attempt writes a row to `llm_calls` via `system_session` (owner; research jobs only): run_id, dossier_id, agent, tier, deployment, service_tier, prompt_sha256, request_json (truncated to 20 KB), response_json (truncated to 20 KB), usage, cost_usd, latency_ms, status, error_code. **Never log the API key or headers.** Research prompts contain market data only, never user data (assert: a guard rejects any message containing an email address or a PAN-pattern string).
- `FakeRouter` for tests: deterministic, returns schema-valid objects from registered fixtures; counts calls and cost.

### 4. Tools and evidence: `wealth/research/tools.py`
- `ToolRegistry.call(name, args, run_ctx) -> ToolResult{tool_call_id, result, source_url, retrieved_at}`. Every call is written to `tool_calls` (run_id, dossier_id, tool, args_json, result_json truncated to 50 KB, result_sha256, source_url, retrieved_at, status).
- Register two real, read-only tools now:
  - `price_history(security_id, days)`: from C10 tables
  - `latest_nav(isin)`: from C10 tables
- Later cards add more.
- Results fetched from the web are stored as **evidence**. They are passed to models only inside a clearly delimited `<evidence id=...>` block, and the system prompt states that evidence is data, not instructions (§5.5 prompt-injection rule).

### 5. Dossier: `wealth/research/dossier.py` (exactly plan §5.3)
- Pydantic models: `Dossier`, `Claim` (`kind`: FACT/INFERENCE/OPINION; FACT requires ≥ 1 evidence with a `tool_call_id`), `Evidence`, `Handoff`, `Verdicts`, `GeoExposure`.
- Stages: `QUEUED → SCREENED → ANALYSED → VALUED → RED_TEAM_DONE → COMPLIANCE_DONE → APPROVED|VETOED|UNRESOLVED`, with a transition function that rejects illegal jumps.
- Handover rules as code:
  - max 2 hand-backs per dossier, then `UNRESOLVED`
  - budget per dossier
  - `validate_grounding(dossier, tool_calls)`: every FACT's tool_call_id exists **and** every number in the claim text appears in that tool call's result JSON (normalise Indian/international grouping and %). Returns the failures.
- Store: `dossiers` table (id `YYYY-Www/MARKET/SYMBOL`, week, security_id, stage, json, created_at, updated_at) + `research_runs` (run_id, week, mode, stage, status, budget_usd, spent_usd, started_at, finished_at). Owner-written; `wealth_app` gets `SELECT` on `dossiers` only, for the PWA's dossier view later.

### 6. Migration `00NN_research`: use the next free number after the current head (check `src/wealth/db/migrations/versions/`) and revise that head
Tables `research_runs`, `dossiers`, `llm_calls`, `tool_calls`, plus the LangGraph checkpointer tables (use the library's `setup()` from the migration or document why it must run at job start).

### 7. Graph skeleton: `wealth/research/graph.py`
- LangGraph `StateGraph` with nodes `plan → screen → analyse → value → red_team → compliance → signoff`. Each node is a stub that calls the router with a tiny schema through the `FakeRouter` in tests.
- Handoffs via `Command(goto=...)`. Compliance can send a dossier back to `analyse` (counts toward the 2-hand-back limit).
- State carries `run_id`, the list of dossier ids, budget and stage.
- Checkpointing with the Postgres checkpointer on `OWNER_DATABASE_URL`, `thread_id = run_id`.

### 8. CLI: `python -m wealth.research.run`
- `start --week 2026-W45 --mode weekly --budget-usd 5` → creates a run, executes until the first stage boundary marked `interrupt` (simulate a Batch/Flex wait point after `screen`), then exits 0 with status `waiting`.
- `continue --run-id ...` → resumes from the checkpoint to the next boundary or the end.
- `status --run-id ...` prints JSON.
- Exit codes: 0 ok/waiting, 2 budget exceeded, 3 failed.

### 9. Tests (no network)
- Router with an `httpx.MockTransport`/respx:
  - json_schema success
  - json_schema rejected → json_object fallback
  - invalid JSON → one repair → success
  - repair fails → `SchemaError`
  - Flex 429 ×3 → default-tier fallback recorded
  - 401 not retried
  - budget pre-check raises
  - cost maths (luna flex: 1,000 in / 500 out → $0.000175)
  - the guard rejects an email in a prompt
  - no API key in any logged row
- Dossier:
  - illegal transitions
  - 2-hand-back limit → UNRESOLVED
  - grounding: a FACT citing "ROCE improved to 16.8%" passes when the tool result contains `16.8`; fails when it contains `16.5`; fails with an unknown tool_call_id
- Graph: start → waiting after `screen`; continue → completes; checkpoint survives a new process (run `continue` in a fresh engine); budget exceeded → exit 2.
- Import guard: importing `wealth.api.main` does not import `wealth.research`, `langgraph` or `openai`.

### 10. Live smoke script (not a test; the reviewer runs it)
`scripts/smoke_foundry.py` reads `FOUNDRY_*` from the environment. It sends one structured-output request per tier (nano with Flex, redteam) using a 2-field schema, and prints model, tier, tokens, cost and latency. **No keys printed.**

## Out of scope
The real agents (C15/C16), GitHub Actions workflows (C18).

## Acceptance
1. `.venv\Scripts\python -m pip install --no-cache-dir -e ".[api,importer,research,dev]"`, then `.venv\Scripts\python -m pytest`. All pass with no network access.
2. `.venv\Scripts\python -m ruff check src tests scripts` and `.venv\Scripts\python -m ruff format --check src tests scripts` → clean
3. `alembic upgrade head` / downgrade one step both work
4. Don't run the live smoke script yourself (there are no keys in your environment). Say in the result file that it's ready for the reviewer.
