# Save endpoint coverage

Each mutating API route is listed once. C58 tests run only in the narrow 360 × 740 project. Existing specs are named where they cover a save flow.

| Method and path | Screen action or reason | End-to-end coverage |
| --- | --- | --- |
| `PUT /v1/me/consents` | Settings → App and privacy → change an email choice | `save-settings.spec.ts`; signed unsubscribe in `save-unsubscribe.spec.ts` |
| `POST /v1/push/subscriptions` | Settings → enable push alerts; needs a real browser push service and VAPID setup | No subscription is created; `save-settings.spec.ts` checks the no-permission state |
| `DELETE /v1/push/subscriptions/{subscription_id}` | Settings → disable a real push subscription; no subscription exists in the local stack | No service-backed subscription test |
| `GET /v1/email/unsubscribe` | Open a signed email link | `save-unsubscribe.spec.ts` |
| `POST /v1/email/unsubscribe` | Submit a signed one-click email link | `save-unsubscribe.spec.ts` |
| `POST /v1/auth/refresh` | Automatic session renewal; no user save action | No deterministic screen action to assert |
| `POST /v1/auth/logout` | Settings → App and privacy → Sign out | `save-settings.spec.ts` |
| `POST /v1/auth/logout-all` | Settings → App and privacy → Sign out on all devices | `save-settings.spec.ts` |
| `PUT /v1/fund-check/sips/{security_id}` | Money → Monthly amounts → save a fund amount | `save-members.spec.ts` |
| `POST /v1/auth/google/exchange` | Google sign-in needs an external OAuth identity token | No network or real OAuth test; dev-token sign-in is covered by existing specs |
| `POST /v1/households/invites` | Settings → You and your family → create a share link | `save-household.spec.ts`; partner and viewer invites also appear in `household.spec.ts` |
| `DELETE /v1/households/invites/{invite_id}` | Settings → revoke a pending invite | `save-household.spec.ts` |
| `POST /v1/households/join` | Open a valid share link and join the family plan | `save-household.spec.ts`; existing partner/viewer join in `household.spec.ts` |
| `PATCH /v1/households/access/{grantee_user_id}` | Settings → change an invited person's access | `save-household.spec.ts` |
| `DELETE /v1/households/access/{grantee_user_id}` | Settings → remove a person's access | `save-household.spec.ts` |
| `DELETE /v1/households/leave` | Settings → leave the family plan | `save-household.spec.ts` |
| `POST /v1/imports` | Upload a synthetic CAS or broker file | `cas-import.spec.ts`, `broker-import.spec.ts`, `save-imports.spec.ts` |
| `POST /v1/imports/{job_id}/mapping` | Import review → Fix columns → Use these columns | `save-imports.spec.ts` |
| `PATCH /v1/imports/{job_id}/items/{item_id}` | Import review → include or leave out a row | `save-imports.spec.ts` |
| `POST /v1/imports/{job_id}/confirm` | Confirm selected rows from a statement | `cas-import.spec.ts`, `broker-import.spec.ts` |
| `DELETE /v1/imports/{job_id}` | Import review → discard the import | `save-imports.spec.ts` |
| `POST /v1/loans/{loan_id}/events` | Loan details → record an extra payment or rate change | `save-loans.spec.ts` |
| `DELETE /v1/loans/{loan_id}/events/{event_id}` | Loan details → delete a recorded update | `save-loans.spec.ts` |
| `POST /v1/loans/{loan_id}/prepay-preview` | Loan details → compare extra-payment options | `save-loans.spec.ts` |
| `PUT /v1/me/profile` | Settings → Your plan → save plan details | `save-settings.spec.ts`; tax setting in `onboarding.spec.ts` |
| `PUT /v1/me/risk-answers` | Home → choose comfort with ups and downs → save answers | `onboarding.spec.ts` |
| `DELETE /v1/me` | Settings → delete account | `delete-account.spec.ts` |
| `POST /v1/members` | Settings → You and your family → add a person | `save-members.spec.ts` |
| `PATCH /v1/members/{member_id}` | Settings → edit family member details; also assign an asset | `save-members.spec.ts` |
| `DELETE /v1/members/{member_id}` | Settings → remove a person after moving or closing their asset | `save-members.spec.ts` |
| `POST /v1/accounts` | No PWA screen creates accounts; imports can select an existing account | No screen action calls this route |
| `PATCH /v1/accounts/{account_id}` | No PWA screen edits account details | No screen action calls this route |
| `DELETE /v1/accounts/{account_id}` | No PWA screen deletes accounts | No screen action calls this route |
| `POST /v1/assets` | Money → Add manually → save an asset form | `save-money.spec.ts`; existing FD and savings in `fd.spec.ts` and `plan-home.spec.ts` |
| `PATCH /v1/assets/{asset_id}` | Asset details → edit; also edit while offline and sync | `save-money.spec.ts` |
| `DELETE /v1/assets/{asset_id}` | Asset details → mark as closed | `save-money.spec.ts`, `save-members.spec.ts` |
| `POST /v1/assets/quick-add` | Money → quick add a short note | `quick-add.spec.ts` |
| `POST /v1/goals` | Home → Add a goal | `onboarding.spec.ts` saves, reloads, and checks Plan |
| `PATCH /v1/goals/{goal_id}` | No goal edit action or update API client is offered in the PWA | No screen action calls this route |
| `DELETE /v1/goals/{goal_id}` | No goal delete action or delete API client is offered in the PWA | No screen action calls this route |
| `PUT /v1/me/checklist` | Money → Add manually → select a checklist item | `save-money.spec.ts` |
| `PUT /v1/me/residency` | No PWA screen calls this route; member details use `PATCH /v1/members/{member_id}` | No screen action calls this route |
| `PUT /v1/me/investing` | Settings → Your plan → save monthly amount and this week's amount | `save-settings.spec.ts`; `plan-home.spec.ts` also saves the Home amount prompt |
| `POST /v1/highlights/{key}/dismiss` | Home → hide a displayed highlight; plan-dependent highlights may be absent | `save-highlights.spec.ts` uses seeded holdings and skips when none display |
| `PATCH /v1/actions/{action_id}` | Home → mark a proposed step done | `plan-home.spec.ts` checks the result after reload |
| `POST /v1/actions/{action_id}/bought` | Home → I bought it → save the purchase | `bought-it.spec.ts` |
| `PATCH /v1/recorded-buys/{recorded_buy_id}` | Money → edit a recorded purchase | `save-recorded-buy.spec.ts` |
| `DELETE /v1/recorded-buys/{recorded_buy_id}` | Money → remove a recorded purchase | `save-recorded-buy.spec.ts` |
| `POST /v1/statement-inbox/keys` | Settings → Get statements from Gmail → create an upload key | `save-statement-inbox.spec.ts` |
| `DELETE /v1/statement-inbox/keys/{key_id}` | Settings → revoke an upload key | `save-statement-inbox.spec.ts` |
| `POST /v1/statement-inbox/uploads` | Gmail integration submits a generated local CAS through its upload key | `save-statement-inbox.spec.ts` |
| `POST /v1/statement-inbox/items/{item_id}/process` | Inbox → enter the synthetic statement password → review statement | `save-statement-inbox.spec.ts` |
| `DELETE /v1/statement-inbox/items/{item_id}` | Inbox → remove a synthetic non-statement email | `save-statement-inbox.spec.ts` |
| `DELETE /v1/statement-inbox/passwords` | Settings → Get statements from Gmail → forget saved passwords | `save-statement-inbox.spec.ts` |

The PWA offers only a Done action for proposed non-buy steps. It has no skip or undo control. US investments link to broker upload rather than a manual-entry form. No dedicated account CRUD, goal edit/delete, or residency screen exists.
