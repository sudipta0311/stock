# C21: PWA: statement import flow (upload, Android share target, progress, review & merge, confirm, column mapping)

**Plan:** `docs/UPGRADE_PLAN.md` §3.3 (imports, review, privacy copy, monthly nudge), §8.1 (share target, cold-start UX), §8.4 wireframe "Import review". Read `AGENTS.md` first. Builds on C19/C20 (shell, design system, API client, money helpers) and the import API from C08b (and C09 `needs_mapping` + `/v1/imports/{id}/mapping` **if present**: feature-detect via the OpenAPI types and hide mapping UI if absent).

## Goal
"Upload your CAS" takes under 2 minutes end to end on a phone, including a server that's waking up. The user sees exactly what was found before anything is saved. The file and password never touch storage.

## Scope

### 1. `/import`: choose and upload
- **Cards:** "NSDL / CDSL CAS (PDF)", "CAMS / KFintech CAS (PDF)", "Broker statement (CSV/XLSX)" (disabled with "coming soon" if the API lacks broker support).
- Each card has a short "How to get it" disclosure (NSDL: monthly email; CAMS: mfcentral/camsonline request; broker: holdings/tradebook export path), written plainly.
- **File picker:** `accept` limited to `.pdf` (and `.csv`, `.xlsx` when broker import exists). Client-side size check against 10 MB with a friendly message.
- **Password field:** `type=password`, `autocomplete="off"`, with a hint ("For NSDL CAS this is usually your PAN in capitals"). Never stored. Sent only in the multipart request, cleared from state immediately after.
- **Progress states** (single screen): Uploading → Reading your statement → Ready to review. The cold-start "waking up" chip from C19 appears when relevant. Show a cancel button (aborts the fetch).
- **Error mapping by `code`:**
  - `wrong_password` → "That password didn't open the file" + retry
  - `not_a_pdf` / `too_large` / `unsupported_format` (with the CDSL coming-soon detail) / `parse_timeout` → friendly text + what to try next
- **Privacy line** under the button: "Your file and password are used once to read your holdings and are not stored."

### 2. Android share target
- In the PWA manifest, add `share_target` (`action: /import/share`, `method: POST`, `enctype: multipart/form-data`, files: `application/pdf`).
- The service worker intercepts the POST, stores the shared file **in memory/IndexedDB only until the import page picks it up** (delete it immediately after reading), and redirects to `/import?shared=1`. That page pre-fills the file and asks for the password.
- iOS has no share target: the regular picker is the fallback.
- Test the SW handler logic as a pure function.

### 3. `/import/:jobId`: review
- **Header:** source, statement date, counts chips (new / changed / duplicate / unknown).
- **Grouped list:**
  - New: name, quantity, statement value, sleeve badge, masked account
  - Changed: old → new quantity diff
  - Duplicate: collapsed by default, "Already up to date"
  - Unknown: needs a decision. The user picks a kind/sleeve (`PATCH` override) or rejects.
- Per-item accept/reject toggles; select-all for New.
- Sticky bottom bar: "Confirm import (n items)" → `POST /confirm` → success screen: "Added n holdings · Net worth updated" with a link to Money. 409 (already confirmed) → show the confirmed state.
- "Discard" → `DELETE` with confirm.
- **Broker `needs_mapping`** (only if the API supports it): show the detected headers and masked samples, a dropdown per canonical field pre-filled with the suggestion, a "Save as" name → `POST mapping` → review.

### 4. Imports history
`/money` → "Imports" sheet listing recent jobs (`GET /v1/imports`) with status and date. Monthly nudge banner on Money: "Your new CAS usually arrives around the 10th, upload it" when the last confirmed CAS is > 30 days old.

### 5. Tests (vitest + MSW)
- Size check.
- The password field is cleared after submit, and the password is never written to storage (spy on `localStorage`, `sessionStorage` and IndexedDB writes).
- Error-code → message mapping table.
- Progress state machine including cold-start retries (fake timers).
- Cancel aborts.
- Share-target handler: stores, redirects, deletes after pickup.
- Review grouping and counts; unknown-item override PATCH; confirm → success; 409 handling; discard.
- The mapping UI is hidden when the API types lack the endpoint.
- Nudge banner logic.
- Axe check on the upload and review screens.

## Out of scope
Backend changes (none expected; if one is truly needed, stop and list it in the result file instead of changing Python code).

## Acceptance (inside `apps/web`, plus the repo-root Python suite)
1. `npm run typecheck`, `lint`, `test`, `build`, `budget` → pass (the import route is code-split)
2. `.venv\Scripts\python -m pytest` still green
3. The result file includes a screen-by-screen note and the measured gzipped size of the import route chunk
