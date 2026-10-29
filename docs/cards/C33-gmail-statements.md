# C33: Statements arrive by themselves: a Gmail helper script, with manual upload as the fallback

**Requested by the product owner:** "Is it possible to read the CAMS and other statements from Gmail? Manual upload can be the fallback."
**Builds on:** the existing statement import and review flow (CAS PDF parser, staging, de-dup, "Whose is this?"), C27 households, C31 Help, C32 sessions. Read `AGENTS.md` first.

## Decision (made; follow it)
- **No Gmail API from our server.** Reading mail needs Google's restricted Gmail scopes: a public app needs a paid security assessment, and testing-mode tokens expire every 7 days.
- **Instead: a Google Apps Script in each person's own Google account** (free, no Google verification). It runs on a daily time trigger and does three things:
  1. Searches Gmail for statement emails with PDF attachments.
  2. Posts each PDF to our API with a household **statement upload key**.
  3. Labels the thread `Wealth/Imported`.
- **The upload key can only add statements; it can never read data.**
- **Manual upload stays exactly as it is and is the fallback everywhere.**

## Scope

### 1. Statement upload keys (backend)
- **Table `statement_inbox_keys`.** Next free migration, revising the current head. It follows the household RLS pattern from AGENTS.md (policy, audit trigger).
  - Columns:
    - `id`, `user_id` (household owner)
    - `created_by` (login)
    - `label` (e.g. "Sudipta's Gmail")
    - `key_hash` (SHA-256, unique)
    - `created_at`, `last_used_at`, `revoked_at`
  - The plain key (32 random bytes, base64url, prefixed `wsk_`) is shown once at creation.
- **API (owner or partner):**
  - `POST /v1/statement-inbox/keys` returns `{id, label, key}` once.
  - `GET /v1/statement-inbox/keys` lists labels and last used dates.
  - `DELETE /v1/statement-inbox/keys/{id}` revokes.
- **`POST /v1/statement-inbox/uploads`** (no user session; `Authorization: StatementKey <key>`):
  - **Request:** multipart with one PDF, plus `sender`, `subject`, `received_at` and the Gmail `message_id` (for de-duplication).
  - **Checks:**
    - PDF magic bytes
    - ≤ `MAX_UPLOAD_MB`
    - one file per request
    - a per-key rate limit (e.g. 20 a day) and the per-IP limiter
  - **Lookup:** resolve the key via a narrowly scoped `SECURITY DEFINER` function, returning the household only for an unrevoked hash, the same pattern as C32's refresh lookup.
  - **Storage:** keep the PDF as a pending inbox item in the existing import staging storage (encrypted at rest if the import pipeline already does that; otherwise store it the same way manual uploads are stored). Record its sender type: `cams` | `kfintech` | `mfcentral` | `nsdl` | `cdsl` | `other`.
  - **Responses:** an already-seen Gmail `message_id` + file hash returns `200 {"status": "duplicate"}`, not a new item. Otherwise `202 {"status": "received"}`. Never echo parsed content.
  - **Never log** the key, the PDF content or the password.

### 2. Passwords and processing
- **Statement PDFs are password-protected.** An inbox item starts as "Needs password".
- **The person opens Inbox → "New statement from NSDL" → enters the password.**
  - The existing CAS parser runs.
  - The result goes into the **existing review screen**. Nothing is saved to holdings until they confirm, as with manual upload.
- **Optional "Remember this password for NSDL statements"** (default off). Stored per household + sender type, encrypted with AES-GCM using the env key `STATEMENT_PASSWORD_KEY` (32 bytes, base64; `.env.example` + `render.yaml` `sync: false` + deploy checker).
  - When a remembered password exists, the item is decrypted and parsed automatically. It still lands in review, never auto-committed.
  - "Forget saved passwords" lives in Settings.
- **Wrong password** → a plain message, and the item stays waiting. **Unreadable PDF** → "We couldn't read this file. Try uploading it in Money" with a link to manual upload.
- **Items older than 30 days without action** are deleted with their PDF, and there's a note in Inbox.

### 3. The Gmail helper script
- **Source in the repo:** `tools/gmail-statements/Code.gs` (Apps Script). It is plain, commented code with no external libraries.
  - `setup()`: creates the label and a daily trigger at about 07:00 local time.
  - `run()`:
    - searches `has:attachment filename:pdf newer_than:45d -label:Wealth/Imported` from the known senders (configurable list at the top of the script: CAMS, KFintech, MFCentral, NSDL, CDSL)
    - posts each PDF with `UrlFetchApp`, retrying once (the free API may take ~50 s to wake)
    - labels threads only after a 202 or `duplicate`
  - Constants at the top: `API_URL` and `UPLOAD_KEY`.
- **The PWA renders the script with the person's new key already filled in**, via Settings → "Get statements from Gmail":
  - a 4-step guide: open script.google.com, paste, run `setup`, approve
  - a copy button
  - a list of keys, each with "Last received …" and Revoke
- **Help article "Get statements from Gmail automatically"** (C31 help file):
  - how to request the CAMS/KFintech statement by email, and the monthly NSDL/CDSL statement
  - what the script can and can't do
  - how to stop it (delete the trigger, revoke the key)
  - that each adult sets it up in their own Gmail or forwards statements to one Gmail
  - Plain words, ≤ 20 words per sentence, and the Google "unverified app" warning explained calmly ("It's your own script, so Google asks you to confirm").
- **Inbox (C31) shows inbound statements** at the top, one primary action each ("Enter password" / "Review").

### 4. Manual upload stays the fallback
Money → Upload statement is unchanged, and every inbound failure message links to it.

## Tests (no network)
- **Key lifecycle:**
  - create shows the key once
  - list never shows it
  - revoke stops uploads
  - only owner/partner can create keys; a viewer can't
  - another household's key can't reach this household (RLS + definer function scope)
- **Upload endpoint:**
  - accepts a synthetic PDF
  - rejects non-PDF, oversize and missing/invalid keys
  - de-duplicates a repeated `message_id`
  - rate limit
  - no key or PDF content in logs or audit
- **Passwords:**
  - a wrong password keeps the item waiting
  - a remembered password is encrypted at rest (the DB value is not the plaintext) and decrypts with the key
  - a missing `STATEMENT_PASSWORD_KEY` disables remembering, with a plain message
  - forget clears it
- **The flow:** an inbound synthetic CAS reaches the same review screen as a manual upload, and nothing is committed before confirmation.
- **Cleanup** deletes stale items.
- **The Apps Script:** a Node unit test loads `Code.gs` with small fakes for `GmailApp`, `UrlFetchApp` and `PropertiesService`. It checks the search query, the posting, labelling only on success, and the retry.
- **PWA:** the Settings flow (create key → script with the key → revoke), the Inbox item with the password prompt, help copy passing the banned-terms and sentence-length checks, and the budget.

## Acceptance
1. Python: full `pytest` and `ruff` (`--no-cache`) pass; alembic upgrade/downgrade/upgrade; OpenAPI refreshed.
2. PWA: `gen:api`, `gen:copy`, `typecheck`, `lint`, `format:check`, full `npm test`, `build`, `budget` (≤ 170 KB initial; the new Settings screen is lazy).
3. `scripts/check_deploy_config.py` passes. The result file names the env var the reviewer must set (`STATEMENT_PASSWORD_KEY`) and gives the exact setup steps a person follows.
