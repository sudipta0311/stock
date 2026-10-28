# C17b: Weekly brief email (templated), web push, daily reminders, urgent alerts, consent & unsubscribe, Inbox API

**Plan:** `docs/UPGRADE_PLAN.md` §9 (rules, template, delivery), §3.1/§3.5 (maturity calendar → reminders), §5.4 (Sunday 08:00 IST send; daily 06:00 reminders; urgent path), §7.4 (DPDP consent, audit log of every email), Appendix C (`no_llm_numbers`), Appendix D.2 (jobs run in GitHub Actions; HTTP email API, not SMTP). Read `AGENTS.md` first. Builds on C07 (net worth, maturity calendar), C15b (`sentinel_events` urgent), C17a (`weekly_plans`, `actions`).

## Goal
Sunday 08:00 IST: every consenting user gets a calm, correct weekly brief (≤ 3 actions, or a designed "no action needed" email). Dated reminders and rare urgent alerts arrive by email and push. Every number comes from the Policy Engine. This card uses **no LLM**: "why" lines are templates.

## Scope

### 0. Follow-ups from the C22 review (do these too, with tests)
- **Single copy source.** C22 created `copy/en.yaml` at the **repo root**, and the PWA build generates its JSON from it. Use that same file for emails (don't create `wealth/alerts/copy/en.yaml`): extend it with any missing codes and email-only keys, and load it from Python. The template-coverage test below checks this file.
- **Plan fields the PWA needs** (extend C17a's `WeeklyPlan`, store and serve them in `/v1/plan`, and refresh the OpenAPI):
  - `plan_health` (the score defined in §2 below)
  - `debt_picks` (top 3 from `pick_debt` for the user's main debt need: instrument, after_tax_yield_pct, suggested tenure, reason codes)
  - `insurance_reviews` (C13 `review_policy` for each traditional policy: irr_pct, continue/paid_up/surrender values, comparison, reason codes, `recommendation_kind`)
  Update the golden weekly-plan test with an explicit `--update-golden` run, and explain the diff in the result file.

### 1. Copy templates (no LLM)
The repo-root `copy/en.yaml` (see §0) maps every `reason_code` / `message_key` used by C11–C13b to a short English template with named params, e.g.:
- `global_under_target: "Your global share is {gap_pct}% below target."`
- `fd_maturity: "FD ₹{amount} ({bank}) matures {date}."`

A test enumerates every reason_code emitted anywhere in `wealth/policy` (grep the source and the rules YAMLs) and fails if a template is missing. Formatting uses the C02 helpers (lakh/crore).

### 2. Brief builder: `wealth/alerts/brief.py`
`build_brief(plan: WeeklyPlan, networth, calendar, track_record | None, user) -> Brief` (Appendix C contract: `week, status, subject, preheader, actions[{rank, kind, title, amount, why, deeplink}], sections{...}`).
- **Subjects:** `"{n} actions this week · Plan health {score}"` or `"No action needed this week · You're on plan ✓"`.
- **Plan health score** (0–100, code): 100 − weighted absolute allocation gaps − foundation gaps. Document the formula; test it.
- **Sections** per plan §9.2:
  1. number
  2. actions
  3. plan at a glance (target vs actual bars as table cells)
  4. coming up (next 30 days from the maturity calendar)
  5. exit watch (exit cards in `watch`/`triggered`)
  6. world & your money (stress headline: target-vs-current difference for the India −25% scenario; tilt notes)
  7. track record: omitted until the Scorekeeper exists
- `no_llm_numbers: true` is asserted: every number in the rendered output must come from brief data (test by extracting all numbers from the HTML/text and checking each against the brief's data values).

### 3. Rendering
`render_html(brief)` and `render_text(brief)` with Jinja2 (`jinja2>=3.1`, autoescape on):
- table-based, inline-CSS, light/dark-safe colours, max width 600 px
- footer: disclaimer ("Research and education only, not personalised investment advice; we're not SEBI-registered" until that changes; configurable), why you got this, manage alerts link, one-click unsubscribe link, postal address placeholder from settings

Snapshot tests for both formats (golden files).

### 4. Delivery
`wealth/alerts/email.py`: provider adapters behind one interface (env `EMAIL_PROVIDER` = `brevo` | `resend` | `dry_run`; default `dry_run`). HTTP APIs only (`httpx`), with:
- `List-Unsubscribe` + `List-Unsubscribe-Post: List-Unsubscribe=One-Click` headers
- `EMAIL_FROM` from settings
- retries on 429/5xx

`dry_run` writes `.eml` files to `out/outbox/` (gitignored) for review. **No real email is sent in tests.**

`wealth/alerts/push.py`: Web Push via `pywebpush>=2.0` with VAPID keys from env. API endpoints `POST /v1/push/subscriptions` (store endpoint + keys; validate the https endpoint host is a known push service) and `DELETE /v1/push/subscriptions/{id}`. A 404/410 from the push service deletes the subscription.

### 5. Consent and unsubscribe (DPDP)
- `consents` rows: `weekly_email`, `reminder_email`, `push`, `urgent_alerts`.
  - API `GET/PUT /v1/me/consents`.
  - Defaults: weekly_email on at sign-up only if the user ticks it in onboarding (PWA). The API default is **off**.
- `GET /v1/email/unsubscribe?token=...` (no auth): an HMAC-signed token (`EMAIL_LINK_SECRET` env, `user_id`, `kind`, expiry 60 days) withdraws that consent and returns a tiny HTML confirmation. Also `POST` for one-click. Never reveals the email address.
- Every sent message → `alerts_log` (user, kind, subject, provider message id, sha256 of the rendered body, sent_at). **Don't store the full body** (privacy); the hash proves what was sent.

### 6. Jobs
- `python -m wealth.alerts.weekly --week 2026-W45 [--dry-run]`: for each user with `weekly_email` consent and a `weekly_plans` row for the week, build → render → send. Idempotent via `alerts_log` (one weekly email per user per week).
- `python -m wealth.alerts.daily [--dry-run]`:
  - reminders from the maturity calendar at −30/−7 days and on the day (dedupe via the `reminders` table `sent_at`)
  - urgent alerts: `sentinel_events.urgent` from the last 24 h for securities the user holds, max 1 per user per day
  - each over email/push per consent

### 6b. Follow-up from the C20 review
Add `GET /v1/me/checklist` (returns `{categories: [...]}`) and include `checklist` in `GET /v1/me`, so the PWA can restore the Money Map tiles on another device. Test both and isolation.

### 7. Inbox API
- `GET /v1/inbox?limit=`: briefs (subject, week, status), reminders and alerts for the user, newest first, from `alerts_log` + `weekly_plans` (no bodies).
- `GET /v1/brief/{week}` re-renders the brief from the stored plan (deterministic) for in-app viewing.

### 8. Tests
- Template coverage.
- `no_llm_numbers` extraction check.
- HTML/text golden snapshots for an ACTIONS week and a NONE week.
- Dry-run outbox contents and headers.
- Unsubscribe token: valid / expired / tampered.
- Weekly idempotency; reminders dedupe and −30/−7 timing with a frozen clock.
- Urgent alert limited to holders and capped at 1/day.
- Push subscription validation and 410 cleanup (mock).
- Inbox isolation.
- Consent off → nothing sent.

## Out of scope
Scheduling (C18 GitHub Actions). Track record (Scorekeeper, C16). PWA screens (C22/C23).

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass; DB tests run; no network
2. `.venv\Scripts\python -m ruff check src tests` and `.venv\Scripts\python -m ruff format --check src tests` → clean
3. New migration(s) up/down; OpenAPI refreshed; `out/` gitignored
4. Produce two sample `.eml` files with `--dry-run` for the golden user (ACTIONS and NONE weeks) and list their paths in the result file (don't commit them)
