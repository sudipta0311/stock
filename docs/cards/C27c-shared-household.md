# C27c: Shared household access: spouse/partner logs in with their own account and sees and edits the same household (both directions); bring-my-data merge; RLS by household grant

**Requested by the product owner:** "I should be able to see my wife's investments and wealth and vice versa." Two adults, each with their own login, share one household with equal access. **This is core scope**, not optional. Runs after C27a.
**Plan:** `docs/UPGRADE_PLAN.md` §7.3 (RLS is the only gate), §7.4 (security), §1.3 P0-2 (tenant isolation must never regress). Read `AGENTS.md` first. Builds on C05 (RLS, `wealth_app`, `wealth_api`, `app_resolve_user`), C06 (auth, `user_session`), C27a (members with residency).

## Model (follow it)
- **The tenant key stays `user_id` on every row and means the household owner's id.** No table changes its tenant column.
- New table `household_access(owner_user_id uuid, grantee_user_id uuid, role text CHECK in ('partner','viewer'), created_at, revoked_at)`, unique on `(owner_user_id, grantee_user_id)`.
  - `partner` = full read/write, same as the owner, except managing access and deleting the household.
  - `viewer` = read-only.
- New table `household_invites(id, owner_user_id, email_lower, role, token_sha256, expires_at, accepted_at, revoked_at, created_at)`. The token is random (≥ 32 bytes, URL-safe); only its SHA-256 is stored; single use; expires in 7 days; bound to the invited email (the acceptor's verified login email must match, case-insensitive).
- Session context: `app.user_id` = the logged-in user (as today) **plus** `app.household_owner` = the household being viewed. The API sets it from a header/query (`X-Household: <owner id>`, default = own id). It isn't trusted by itself: the policies check it.

## 0. Fit with C27a (read first: the reviewer found these gaps)
C27a (migration `0010_household_planning`) added:
- `household_members.is_self` (exactly one per `user_id`, `relationship = 'self'`; a trigger forbids changing it)
- per-member `residency`/`tax`/`investing`
- a SECURITY DEFINER trigger `default_self_member_for_owner` that sets `member_id` to the **owner's** self member when an insert omits it
- `/v1/me/residency` and `/v1/me/investing` editing the self member

With shared access, these must follow the **logged-in person**, not the household owner:
- **Link each grant to a member:** `household_access.member_id` (FK to `household_members(user_id=owner, id)`) is the grantee's member in the owner's household.
  - Invites take an optional `member_id`: "This invite is for: Anita". The owner usually created that member already, so accepting links to it and never creates a duplicate.
  - Without `member_id`, accepting creates a new `spouse` member named after the grantee's display name.
- **Acting member:** new migration. Replace `default_self_member_for_owner()` (keep SECURITY DEFINER and the pinned `search_path`):
  - when `member_id` is null and `app.user_id` is the owner → the owner's self member (as now)
  - when the writer is a grantee → their linked member from `household_access`
  - Test both, and that a grantee can't attach rows to a member of another household.
- **"Me" endpoints act on the viewer's own member in the current household:**
  - `/v1/me/residency` and `/v1/me/investing` → the owner's self member for the owner; the linked member for a grantee
  - A grantee changing their own residency must not change the owner's. Test it.
  - `GET /v1/me` and `GET /v1/plan` add `me_member_id`. Copy that says "You" (actions, briefs, the PWA) is relative to the viewer: for Anita, "You" is her member and the owner appears by name.
- **Merge (§3) can't change `is_self`:** the trigger forbids it and the one-self-per-user index blocks it. So:
  - re-point the grantee's rows to their linked member in the target household (`user_id = owner`, `member_id = linked member`)
  - copy the grantee's self-member `residency`/`tax`/`investing` onto the linked member: the grantee's own values win (they are the authority on their own tax residence); report the overwritten fields in the response
  - leave the grantee's own `self` member in their own (now empty) household
- **The weekly brief (§4)** for a grantee uses their linked member as "You".

## Scope

### 1. Migration (next free number)
- Tables above, with RLS: a user sees access rows where they are owner or grantee, and invites they own.
- SECURITY DEFINER helpers (owned by the migration owner, pinned `search_path`, `REVOKE ALL FROM PUBLIC`, `GRANT EXECUTE TO wealth_app`):
  - `app_household_role(p_owner uuid) RETURNS text`: `'owner'` when `p_owner = current_setting('app.user_id')::uuid`; else the active (non-revoked) grant role of the current user on `p_owner`; else NULL.
  - `app_accept_invite(p_token text) RETURNS uuid`:
    - hashes the token and finds a valid invite (not expired, accepted or revoked)
    - checks the current user's email equals the invite email
    - creates or reactivates the grant and marks the invite accepted
    - returns the owner id
    - **constant-time-ish behaviour**: same error for not-found/expired/wrong-email (`invalid_invite`)
- **Replace the isolation policy on every user-owned table:**
  - `USING (user_id = nullif(current_setting('app.household_owner', true),'')::uuid AND app_household_role(user_id) IS NOT NULL)`
  - `WITH CHECK (user_id = <same household> AND app_household_role(user_id) IN ('owner','partner'))`
  - Keep the `users` table rule as it is (only your own user row) plus a narrow view `household_people(owner_id, user_id, display_name, role)` so partners can see each other's display names.
  - Downgrade restores the C05 policies exactly.
- **Safety:** if `app.household_owner` is unset, the API must set it to the user's own id, so behaviour equals today. A test proves an unset value yields 0 rows (no fallback to the owner via the policy).

### 2. API
- **Dependency:**
  - reads `X-Household` (default: own id)
  - inside `user_session` sets both settings
  - calls `app_household_role` once; 404 if NULL (don't reveal existence); exposes the role on `request.state`
  - mutating endpoints require owner/partner (403 problem `read_only` for viewers)
- **Endpoints:**
  - `GET /v1/households`: the households the user can open, with `{owner_id, owner_name, role, member_count}` (own first)
  - `POST /v1/households/invites` (owner only) `{email, role}` → creates an invite and returns a share link `https://<pwa>/join?token=...`. If `EMAIL_PROVIDER` is configured and the owner opts in, also send the invite email (dry-run in tests).
  - `GET /v1/households/invites` (owner) and `DELETE /v1/households/invites/{id}` (revoke)
  - `POST /v1/households/join` `{token, bring_my_data: bool}` → `app_accept_invite` → if `bring_my_data`, merge (§3) → returns the owner id
  - `GET /v1/households/access` (owner): grantees with roles; `PATCH` (change role) and `DELETE` (revoke) access. Revoke takes effect on the next request.
  - `DELETE /v1/households/leave` (grantee leaves)
- Every response that lists actions, assets, plan, etc. is for the household in `X-Household`. Add `household_owner_id` and `role` to `GET /v1/me`.
- **Audit:** invite created/revoked, accepted, role changed, access revoked/left, merge performed. Every write by a grantee is written to `audit_log` of the household, with `actor_user_id`.

### 3. Bring-my-data merge (SECURITY DEFINER `app_merge_into_household(p_owner uuid)`)
For the current user who **just** accepted a partner grant on `p_owner`:
- moves all rows of the user's own household (`user_id = current user`) in every user-owned table to `user_id = p_owner`
- the user's own `self` member becomes a member of the target household with relationship `spouse` (editable later)
- also moves profiles' investing prefs and residency onto that member (C27a member fields)
- leaves the `users` row and that user's own consents/push subscriptions (personal, per login) in place

Transactional, idempotent, refused for viewers, refused if the user owns grants to others (they must revoke first). Return counts per table. **Tests are mandatory.**

### 4. Emails and personalisation
- The weekly job (C17b) iterates **logins with consent**. For each, it uses the household they own, or (if they are a partner and own no household data) the shared household. Each partner receives their own copy. The brief header says "Family plan · <owner name>'s household".
- The plan/personaliser runs **per household** (owner id), not per login, so both partners see the same plan and actions. Marking an action done by either partner updates it for both (`done_by` recorded).

### 5. Tests (DB, using the non-owner API login fixture)
Two partners A (owner) and B (spouse), a stranger C, and a viewer V:
- **Before acceptance:** B sees nothing of A (404 with `X-Household: A`).
- **Accept with the right email:** B lists households and sees A's assets/plan; B can add an asset to A's household; the audit row has actor B.
- **Wrong email, expired, reused or revoked token** → the same `invalid_invite` error. Brute-force check: tokens are 43+ chars and only hashes are stored.
- **Viewer:** V reads A's household; every write → 403; a direct SQL write attempt as V (through `user_session` with `app.household_owner=A`) fails the WITH CHECK.
- **Stranger:** C with `X-Household: A` → 404. Direct SQL with `app.household_owner = A` as C → 0 rows. C can't call `app_accept_invite` for someone else's email.
- **Revocation:** A revokes B → B's next request gets 404; SQL as B returns 0 rows.
- **Escalation:** B (partner) can't create invites, change roles or revoke A; B can't set `app.household_owner` to C's household; `RESET ROLE` still gives no access (C05 property preserved).
- **Unset household setting** → 0 rows.
- **Merge:**
  - B had their own assets/holdings/imports → after join with `bring_my_data` they belong to A's household and are visible to both, with counts returned
  - B's old household is empty
  - B's consents remain B's
  - repeat merge → no-op
- **Plan:** the same plan id/actions seen by A and B; mark done by B → A sees done with `done_by=B`.
- **Linked member (§0):**
  - an invite for the existing member "Anita" → accept links to it, with no duplicate member
  - B adds an FD without `member_id` → owned by Anita's member, not A's self
  - B sets her residency to RI-IN via `/v1/me/residency` → only Anita's member changes; A stays NRI-DK
  - `me_member_id` differs for A and B on the same plan
  - merge with B's own data → rows re-pointed to Anita's member, B's residency kept, B's self member left in B's own household
- **Downgrade/upgrade** restores the C05 policies and passes the original isolation tests.

## Acceptance
1. `.venv\Scripts\python -m pytest` → all pass, including all original C05/C06/C07/C08b isolation tests unchanged
2. `ruff` clean; migration up/down; OpenAPI refreshed
3. The result file lists every policy and function added, with a short security rationale
