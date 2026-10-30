import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const e2eDir = fileURLToPath(new URL(".", import.meta.url));
const repositoryRoot = resolve(e2eDir, "..");
process.env.PLAYWRIGHT_BROWSERS_PATH ??= resolve(
  repositoryRoot,
  ".playwright-browsers",
);

export default defineConfig({
  testDir: "./tests",
  globalSetup: "./setup/start-stack.mjs",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 2,
  timeout: 60_000,
  globalTimeout: 20 * 60_000,
  expect: { timeout: 10_000 },
  reporter: [
    ["list"],
    ["html", { outputFolder: "playwright-report", open: "never" }],
  ],
  outputDir: "test-results",
  use: {
    browserName: "chromium",
    headless: true,
    serviceWorkers: "block",
    actionTimeout: 10_000,
    navigationTimeout: 30_000,
    trace: "off",
  },
  projects: [
    {
      name: "mobile",
      use: {
        viewport: { width: 412, height: 915 },
        deviceScaleFactor: 1,
        isMobile: true,
        hasTouch: true,
      },
    },
    {
      name: "narrow",
      use: {
        viewport: { width: 360, height: 740 },
        deviceScaleFactor: 1,
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
});
