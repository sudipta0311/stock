import { createHmac, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test as playwrightTest } from "@playwright/test";
import type { BrowserContext, Page, Route } from "@playwright/test";

const repositoryRoot = resolve(
  fileURLToPath(new URL("../..", import.meta.url)),
);

export interface TestIdentity {
  id: string;
  email: string;
  displayName: string;
}

export const test = playwrightTest.extend<{ localNetworkFence: void }>({
  localNetworkFence: [
    async ({ context }, use) => {
      const assertLocalOnly = await guardLocalNetwork(context);
      await use();
      await assertLocalOnly();
    },
    { auto: true },
  ],
});

export { expect };

export function appUrl(path = "/"): string {
  const base = process.env.E2E_BASE_URL;
  if (!base) throw new Error("The e2e app URL is not configured.");
  return new URL(path, base).toString();
}

export function makeIdentity(label: string): TestIdentity {
  const safeLabel = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, 18);
  const id = `${safeLabel}-${randomUUID()}`;
  return {
    id,
    email: `e2e-${id}@example.test`,
    displayName: "Synthetic test user",
  };
}

export function mintDevToken(identity: TestIdentity): string {
  const secret = process.env.E2E_DEV_AUTH_SECRET;
  if (!secret) throw new Error("The e2e sign-in secret is not configured.");
  const header = Buffer.from(
    JSON.stringify({ alg: "HS256", typ: "JWT" }),
  ).toString("base64url");
  const now = Math.floor(Date.now() / 1_000);
  const payload = Buffer.from(
    JSON.stringify({
      sub: `e2e-${identity.id}`,
      email: identity.email,
      name: identity.displayName,
      exp: now + 60 * 60,
    }),
  ).toString("base64url");
  const signed = `${header}.${payload}`;
  const signature = createHmac("sha256", secret)
    .update(signed)
    .digest("base64url");
  return `${signed}.${signature}`;
}

export async function guardLocalNetwork(
  context: BrowserContext,
): Promise<() => Promise<void>> {
  const blocked = new Set<string>();
  const handler = async (route: Route) => {
    let url: URL;
    try {
      url = new URL(route.request().url());
    } catch {
      blocked.add("invalid-url");
      await route.abort();
      return;
    }
    if (
      ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) &&
      ["http:", "https:"].includes(url.protocol)
    ) {
      await route.continue();
      return;
    }
    blocked.add(url.origin);
    await route.abort();
  };
  await context.route("**/*", handler);
  return async () => {
    await context.unroute("**/*", handler);
    expect([...blocked], "the e2e browser must stay on localhost").toEqual([]);
  };
}

// A wide row makes a phone lay the page out wider than its screen, so scrollWidth still
// equals innerWidth; compare the layout width with the visible width instead.
export async function expectFitsScreen(page: Page): Promise<void> {
  const [layoutWidth, screenWidth] = await page.evaluate(() => [
    window.innerWidth,
    Math.round(window.visualViewport?.width ?? window.innerWidth),
  ]);
  expect(layoutWidth, "page is wider than the screen").toBe(screenWidth);
}

export async function gotoApp(page: Page, path: string): Promise<void> {
  await page.goto(appUrl(path));
}

export async function signIn(
  page: Page,
  identity: TestIdentity,
): Promise<void> {
  // The journeys start as a returning user who has seen the first-visit tour. Its
  // storage key holds the server user id, unknown here, so every tour reads as seen.
  await page.addInitScript((prefix: string) => {
    const getItem = Storage.prototype.getItem;
    Storage.prototype.getItem = function (this: Storage, key: string) {
      return key.startsWith(prefix) ? "true" : getItem.call(this, key);
    };
  }, "wealth.tour.seen:");
  await gotoApp(page, "/login");
  await page.evaluate(() => {
    window.sessionStorage.clear();
    window.localStorage.clear();
  });
  await page.reload();
  await page.getByLabel("Development token").fill(mintDevToken(identity));
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Your money, in one clear place" }),
  ).toBeVisible();
}

export async function completeMinimalOnboarding(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Set up my Money Map" }).click();
  await expect(
    page.getByRole("heading", { name: "Who are we planning for?" }),
  ).toBeVisible();
  await page.getByRole("button", { name: /^Just me/ }).click();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Your plan is ready" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "View your plan" }).click();
  await expect(page).toHaveURL(/\/plan$/);
}

export async function startReadyAccount(
  page: Page,
  label: string,
): Promise<TestIdentity> {
  const identity = makeIdentity(label);
  await signIn(page, identity);
  await completeMinimalOnboarding(page);
  return identity;
}

export async function skipGuidedTour(page: Page): Promise<void> {
  const skip = page
    .getByRole("dialog", { name: "Home" })
    .getByRole("button", { name: "Skip" });
  if (await skip.isVisible().catch(() => false)) await skip.click();
}

export async function openFixedDepositForm(page: Page): Promise<void> {
  await gotoApp(page, "/money");
  await page
    .getByRole("button", { name: "Fixed deposits", exact: true })
    .click();
  await page.getByRole("button", { name: /Continue \(1 selected\)/ }).click();
  await expect(
    page.getByRole("heading", { name: "Fixed deposit", exact: true }),
  ).toBeVisible();
}

export async function completeFixedDeposit(page: Page): Promise<number> {
  const startedAt = Date.now();
  await page.getByLabel("Bank or post office").fill("Synthetic Bank");
  await page.getByLabel("Amount deposited").fill("5L");
  await page.getByLabel("Interest rate (%)").fill("7.1");
  await page.getByLabel("Matures (month and year)").fill("2027-03");
  await page.getByRole("button", { name: "Save now" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: /^Saved$/ }),
  ).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText(/^Value today ≈/)).toBeVisible();
  return Date.now() - startedAt;
}

export async function addDanishSavings(page: Page): Promise<void> {
  await gotoApp(page, "/money/add/savings");
  await page.getByLabel("Bank").fill("Synthetic Bank");
  await page.getByLabel("Currency").selectOption("DKK");
  await page.getByLabel("Current balance").fill("20,000");
  await page.getByRole("button", { name: "Save now" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: /^Saved$/ }),
  ).toBeVisible();
}

export async function uploadSyntheticCas(
  page: Page,
  password = process.env.E2E_CAS_PASSWORD,
): Promise<void> {
  const path = process.env.E2E_CAS_PATH;
  if (!path || !password)
    throw new Error("The synthetic statement is not configured.");
  await page.getByLabel("Statement file").setInputFiles(path);
  await page.getByLabel(/^Statement password/).fill(password);
  await page.getByRole("button", { name: "Upload statement" }).click();
  await expect(
    page.getByRole("heading", { name: "Review your import" }),
  ).toBeVisible({
    timeout: 120_000,
  });
}

export async function writeReport(
  reportName: string,
  projectName: string,
  value: unknown,
): Promise<void> {
  const reportDir = join(repositoryRoot, "out", "e2e");
  const fileName =
    projectName === "all"
      ? reportName + ".json"
      : reportName + "-" + projectName + ".json";
  await mkdir(reportDir, { recursive: true });
  await writeFile(
    join(reportDir, fileName),
    JSON.stringify(value, null, 2) + "\n",
    "utf8",
  );
}

function pythonExecutable(): string {
  const localPython = join(repositoryRoot, ".venv", "Scripts", "python.exe");
  return (
    process.env.PYTHON ?? (existsSync(localPython) ? localPython : "python")
  );
}

function runDatabaseCommand(args: string[]): Promise<void> {
  return new Promise((resolveCommand, rejectCommand) => {
    const environment = { ...process.env, PYTHONDONTWRITEBYTECODE: "1" };
    delete environment.OWNER_DATABASE_URL;
    delete environment.DATABASE_URL;
    const child = spawn(
      pythonExecutable(),
      ["-m", "scripts.e2e_database", ...args],
      {
        cwd: repositoryRoot,
        env: environment,
        stdio: ["ignore", "ignore", "pipe"],
        windowsHide: true,
      },
    );
    let stderr = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString("utf8")).slice(-2000);
    });
    // The last error line says why, with any database address removed.
    const reason = () =>
      stderr
        .trim()
        .split(/\r?\n/)
        .pop()
        ?.replace(/postgres(?:ql)?(?:\+\w+)?:\/\/\S+/gi, "<database>") ?? "";
    child.once("error", () =>
      rejectCommand(new Error("The e2e database helper failed.")),
    );
    child.once("exit", (code) =>
      code === 0
        ? resolveCommand()
        : rejectCommand(
            new Error(`The e2e database helper failed: ${reason()}`),
          ),
    );
  });
}

export function seedHomeAction(email: string): Promise<void> {
  return runDatabaseCommand(["seed-action", email]);
}

export function verifyAccountDeleted(email: string): Promise<void> {
  return runDatabaseCommand(["verify-deleted", email]);
}
