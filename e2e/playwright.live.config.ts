import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
import { getLiveEnvironment } from "./live/environment.mjs";

function liveEnvironment() {
  try {
    return getLiveEnvironment();
  } catch (error) {
    console.error(
      error instanceof Error
        ? error.message
        : "The live check is not configured.",
    );
    process.exit(1);
  }
}

const environment = liveEnvironment();

export default defineConfig({
  testDir: "./live",
  workers: 1,
  fullyParallel: false,
  retries: 0,
  timeout: 60 * 60 * 1_000,
  expect: { timeout: 10_000 },
  reporter: "list",
  outputDir: resolve(environment.outDir, "playwright-output"),
  use: {
    browserName: "chromium",
    headless: !environment.headed,
    viewport: { width: 360, height: 740 },
    deviceScaleFactor: 1,
    isMobile: true,
    hasTouch: true,
    serviceWorkers: "block",
    actionTimeout: 10_000,
    navigationTimeout: 60_000,
    trace: "off",
    video: "off",
    screenshot: "off",
  },
});
