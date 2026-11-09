# Live look-only check

1. Set LIVE_PROFILE_DIR and LIVE_OUT_DIR to folders outside this repository.
2. The helper opens installed Google Chrome. LIVE_CHROME_PATH can select a specific chrome.exe.
3. Run npm run live:login from e2e. Sign in, wait for Home, then close Chrome.
4. Run npm run live from e2e. Keep that profile's sign-in window closed during the check.
5. Open findings.md, findings.json, and screens/ under LIVE_OUT_DIR.
6. To end the saved session, open Settings, then Signed-in devices, and remove this device.

The check uses one Google Chrome context and blocks app writes. It stores no screenshots or reports in this repository.
Full-page screenshots can repeat the sticky header and bottom navigation mid-page; this is a Playwright screenshot artifact.
