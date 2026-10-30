# Deploy Wealth Copilot

1. **Use the prepared database.** The `wealth` database is in Frankfurt (`aws-eu-central-1`), on Postgres 18. Migrations 0001–0011 are already applied. Leave it in place. The API uses the pooled connection for the `wealth_api` login. Set `DATABASE_URL` in Render to that connection. Do not use the owner login for the API.

   For a future new database, create it in Frankfurt and save its owner connection as GitHub secret `OWNER_DATABASE_URL`. Run the **Database migrations** workflow. Set the `wealth_api` password in Neon’s SQL editor with `ALTER ROLE wealth_api WITH LOGIN PASSWORD '<new-password>';`, then use its pooled connection for Render `DATABASE_URL`.

2. **Add GitHub values.** Open the repository’s **Settings → Secrets and variables → Actions**. Add these repository secrets:

   - `OWNER_DATABASE_URL`: the owner connection for the `wealth` database.
   - `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY`, `FOUNDRY_MODEL_WORKHORSE`, `FOUNDRY_MODEL_REDTEAM`: from Azure AI Foundry.
   - `SEC_USER_AGENT`: the app name and contact details required for SEC data access.
   - `EMAIL_PROVIDER`, `EMAIL_API_KEY`, `EMAIL_FROM`: from your email provider. Set `EMAIL_PROVIDER` to `brevo`, `resend`, or `dry_run`.
   - `EMAIL_LINK_SECRET`: a long random value. Use the same value in Render.
   - `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`: generated in step 5.
   - `PRICE_API_KEY`: only if the chosen overseas price source needs a key.

   Optional Actions variable: `RESEARCH_BUDGET_USD`. The default is `5`.

3. **Create the Render services.** In Render, choose **New → Blueprint**, select this repository and branch `redesign/v2`, then apply `render.yaml`. The API service region is `frankfurt`. The static site has no region field.

   In `wealth-pwa`, set `VITE_API_URL` to the API URL, `VITE_OIDC_AUTHORITY` to `https://accounts.google.com`, and both `VITE_OIDC_CLIENT_ID` and `VITE_OIDC_AUDIENCE` to the Google client ID. Set `VITE_VAPID_PUBLIC_KEY` to the public push key.

   In `wealth-api`, set `DATABASE_URL` to the pooled `wealth_api` connection; `AUTH_JWKS_URL` to `https://www.googleapis.com/oauth2/v3/certs`; `AUTH_ISSUER` to `https://accounts.google.com`; `AUTH_AUDIENCE` and `GOOGLE_CLIENT_ID` to the Google client ID; `GOOGLE_CLIENT_SECRET` to the Google secret; `ALLOWED_EMAILS` to the approved addresses, separated by commas; `ALLOWED_ORIGINS` to the PWA URL; and `EMAIL_LINK_SECRET` to the GitHub value. Keep `ALLOWED_EMAILS` filled in. Other API fields are set by the Blueprint.

   The free API sleeps after quiet periods. Its first response may take up to a minute. The PWA can open while the API wakes.

4. **Allow the PWA in Google.** In the existing Google OAuth web client, add this JavaScript origin: `https://wealth-pwa.onrender.com`. Add this redirect URI: `https://wealth-pwa.onrender.com/auth/callback`. If Render gave the site a different address (for example with a suffix), use that address instead, here and in `ALLOWED_ORIGINS`. Keep the `localhost` entries used for development.

5. **Set up email and push.** Verify your email domain with SPF, DKIM, and DMARC. Check the provider’s free sending limit. Generate push keys in a terminal with `python -m wealth.alerts.push --generate-vapid`. Put the public key in Render `VITE_VAPID_PUBLIC_KEY` and GitHub `VAPID_PUBLIC_KEY`; put the private key only in GitHub `VAPID_PRIVATE_KEY`.

6. **Run the first checks.** Open `https://wealth-api.onrender.com/healthz?deep=1`. Sign in and finish setup. Upload an account statement (CAS). Run `marketdata-daily`, `marketdata-weekly` and `policy-weekly` from **Actions**. Set GitHub `EMAIL_PROVIDER` to `dry_run` and run `alerts-weekly` once. Restore your email provider before sending real emails.

7. **Check privacy before inviting anyone.** Keep the API and database in EU regions. Accept the data agreements from Render, Neon, and the email provider. Azure AI Foundry (`stockllm`, East US) receives market facts only. The CI guard `test_research_model_calls_do_not_read_personal_records_or_prompt_fields` blocks personal record fields in research model calls. The PWA notice must explain why data is used, the legal basis, how long it is kept, service providers, and access, correction, export, and deletion rights. Email must be opt-in. Device storage is for app functions only; do not add tracking cookies. Replace the breach-contact placeholder before launch.

8. **Plan for growth and launch.** Upgrade the API if sleep delays or slow imports become a problem. Upgrade Neon if storage or compute limits are reached. Move jobs if GitHub Actions minutes run out. EU investment recommendations may require MiFID II authorisation or a regulated partner in Denmark. The app is information-only; get regulatory advice before a public launch.
