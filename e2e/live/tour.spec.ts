import { expect, type Locator } from "@playwright/test";
import { test, type LiveRuntime } from "./runtime";

const tourPaths = ["/", "/money", "/plan", "/ideas"];
const fullLoadPaths = new Set([
  "/does-not-exist",
  "/login",
  "/join",
  "/onboarding",
]);
const addTypes = [
  "epf",
  "ppf",
  "fd",
  "po_scheme",
  "nps",
  "insurance",
  "gold",
  "bond",
  "savings",
  "loan",
  "property",
  "us_investments",
];

interface Inventory {
  assets: Set<string>;
  holdings: Set<string>;
  loans: Set<string>;
}

function emptyInventory(): Inventory {
  return { assets: new Set(), holdings: new Set(), loans: new Set() };
}

function rows(value: unknown, names: string[]): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    return value.filter(
      (item): item is Record<string, unknown> =>
        typeof item === "object" && item !== null && !Array.isArray(item),
    );
  }
  if (typeof value !== "object" || value === null) return [];
  const record = value as Record<string, unknown>;
  for (const name of names) {
    if (Array.isArray(record[name])) return rows(record[name], names);
  }
  return [];
}

function fieldId(row: Record<string, unknown>, names: string[]): string | null {
  for (const name of names) {
    const value = row[name];
    if (typeof value === "string" && value) return value;
  }
  return null;
}

function idsFrom(
  value: unknown,
  arrays: string[],
  fields: string[],
): Set<string> {
  return new Set(
    rows(value, arrays)
      .map((row) => fieldId(row, fields))
      .filter((value): value is string => value !== null),
  );
}

function checkSet(
  runtime: LiveRuntime,
  name: string,
  expected: Set<string>,
  actual: Set<string>,
): void {
  const missing = [...expected].filter((value) => !actual.has(value));
  const extra = [...actual].filter((value) => !expected.has(value));
  if (missing.length || extra.length) {
    runtime.addFinding(
      "fail",
      "list coverage",
      name +
        ": missing IDs [" +
        missing.join(", ") +
        "]; unexpected IDs [" +
        extra.join(", ") +
        "].",
    );
  }
}

async function hrefs(
  page: LiveRuntime["page"] | Locator,
  prefix: string,
): Promise<string[]> {
  const values = await page
    .locator('a[href^="' + prefix + '"]')
    .evaluateAll((nodes) =>
      nodes
        .map((node) => (node as HTMLAnchorElement).getAttribute("href") ?? "")
        .filter(Boolean),
    );
  return [...new Set(values)];
}

function idsFromPaths(paths: string[], prefix: string): Set<string> {
  return new Set(
    paths
      .map((path) => {
        const pathname = path.split(/[?#]/)[0] ?? "";
        if (!pathname.startsWith(prefix)) return null;
        const encodedId = pathname.slice(prefix.length).split("/")[0];
        if (!encodedId) return null;
        try {
          return decodeURIComponent(encodedId);
        } catch {
          return encodedId;
        }
      })
      .filter((value): value is string => value !== null),
  );
}

async function navigateWithinApp(
  runtime: LiveRuntime,
  path: string,
  clickVisibleLink = true,
): Promise<void> {
  const { page, environment } = runtime;
  const target = new URL(path, environment.pwaUrl);
  const current = new URL(page.url());
  if (current.origin !== target.origin) {
    throw new Error("The requested route is outside the app origin.");
  }
  if (
    current.pathname === target.pathname &&
    current.search === target.search &&
    current.hash === target.hash
  ) {
    return;
  }

  if (clickVisibleLink) {
    const links = page.locator("a[href]");
    const linkCount = await links.count();
    for (let index = 0; index < linkCount; index += 1) {
      const link = links.nth(index);
      if (!(await link.isVisible().catch(() => false))) continue;
      const href = await link.getAttribute("href");
      if (!href) continue;
      let linkUrl: URL;
      try {
        linkUrl = new URL(href, current);
      } catch {
        continue;
      }
      if (
        linkUrl.origin !== target.origin ||
        linkUrl.pathname !== target.pathname ||
        linkUrl.search !== target.search
      ) {
        continue;
      }
      try {
        await link.click({ timeout: 5_000 });
        await page.waitForFunction(
          (expectedPath) => window.location.pathname === expectedPath,
          target.pathname,
          { timeout: 5_000 },
        );
        return;
      } catch {
        break;
      }
    }
  }

  const targetPath = target.pathname + target.search + target.hash;
  await page.evaluate((nextPath) => {
    window.history.pushState(null, "", nextPath);
    window.dispatchEvent(
      new PopStateEvent("popstate", { state: window.history.state }),
    );
  }, targetPath);
  await page.waitForFunction(
    (expectedPath) => window.location.pathname === expectedPath,
    target.pathname,
    { timeout: 5_000 },
  );
}

async function visit(runtime: LiveRuntime, path: string): Promise<void> {
  runtime.setScreen(path);
  const startedAt = Date.now();
  let opened = false;
  try {
    if (fullLoadPaths.has(new URL(path, runtime.environment.pwaUrl).pathname)) {
      await runtime.page.goto(
        new URL(path, runtime.environment.pwaUrl).toString(),
        {
          waitUntil: "domcontentloaded",
          timeout: 60_000,
        },
      );
    } else {
      await navigateWithinApp(runtime, path);
    }
    opened = true;
  } catch {
    runtime.addFinding(
      "fail",
      "navigation",
      "The requested screen could not be opened.",
    );
  }
  if (opened && isListDetailPath(path)) {
    const heading = runtime.page.locator("main h1, main h2").first();
    try {
      await heading.waitFor({ state: "visible", timeout: 8_000 });
      const text = (await heading.innerText()).toLowerCase();
      if (/(?:couldn['’]t|could not|failed to) load|not found/.test(text)) {
        runtime.addFinding(
          "fail",
          "detail page",
          "The list item detail page reported that its item could not load.",
        );
      }
    } catch {
      runtime.addFinding(
        "fail",
        "detail page",
        "The list item detail page did not show its heading.",
      );
    }
  }
  await runtime.checkScreen(startedAt);
}

function isListDetailPath(path: string): boolean {
  const pathname = path.split(/[?#]/)[0] || "/";
  return /^\/(?:money\/(?:asset|holding|loan)\/[^/]+|import\/[^/]+|inbox\/brief\/[^/]+)$/.test(
    pathname,
  );
}

async function clickTab(
  runtime: LiveRuntime,
  label: string,
  path: string,
): Promise<void> {
  runtime.setScreen(path);
  const startedAt = Date.now();
  try {
    await runtime.page
      .getByRole("navigation", { name: "Primary" })
      .getByRole("link", { name: label, exact: true })
      .click({ timeout: 5_000 });
  } catch {
    runtime.addFinding(
      "fail",
      "bottom navigation",
      "A primary tab could not be opened.",
    );
    await navigateWithinApp(runtime, path, false).catch(() => undefined);
  }
  await runtime.checkScreen(startedAt);
}

async function openAllReadOnlyStates(runtime: LiveRuntime): Promise<void> {
  const { page } = runtime;
  let opened = 0;
  while (opened < 300) {
    const summary = page
      .locator("details:not([open]) > summary:visible")
      .first();
    if (!(await summary.count())) break;
    const startedAt = Date.now();
    try {
      await summary.click({ timeout: 5_000 });
    } catch {
      runtime.addFinding(
        "fail",
        "expander",
        "A visible section could not be opened.",
      );
      break;
    }
    opened += 1;
    runtime.count("expanders");
    await runtime.checkScreen(startedAt);
  }
  if (
    opened === 300 &&
    (await page.locator("details:not([open]) > summary:visible").count())
  ) {
    runtime.addFinding(
      "fail",
      "expander coverage",
      "The visible section limit was reached before every section opened.",
    );
  }

  const termButtons = page
    .locator("[data-term-copy]")
    .locator("xpath=..")
    .getByRole("button", { name: "What's this?", exact: true });
  const termCount = await termButtons.count();
  for (let index = 0; index < termCount; index += 1) {
    const button = termButtons.nth(index);
    if (!(await button.isVisible().catch(() => false))) continue;
    if ((await button.getAttribute("aria-expanded")) === "true") continue;
    const startedAt = Date.now();
    await button.click().catch(() => undefined);
    if ((await button.getAttribute("aria-expanded")) !== "true") {
      runtime.addFinding(
        "fail",
        "term explainer",
        "A visible What's this? explanation could not be opened.",
      );
      await runtime.checkScreen(startedAt);
      continue;
    }
    runtime.count("expanders");
    await runtime.checkScreen(startedAt);
  }

  const tabs = page.getByRole("tab");
  const tabCount = await tabs.count();
  for (let index = 0; index < tabCount; index += 1) {
    const tab = tabs.nth(index);
    if (!(await tab.isVisible().catch(() => false))) continue;
    if ((await tab.getAttribute("aria-selected")) === "true") continue;
    const startedAt = Date.now();
    await tab.click().catch(() => undefined);
    if ((await tab.getAttribute("aria-selected")) !== "true") {
      runtime.addFinding("fail", "tab", "A visible tab could not be opened.");
      await runtime.checkScreen(startedAt);
      continue;
    }
    runtime.count("expanders");
    await runtime.checkScreen(startedAt);
  }
}

async function showAllHoldings(runtime: LiveRuntime): Promise<void> {
  let opened = 0;
  while (opened < 20) {
    const button = runtime.page
      .getByRole("button", { name: /^Show all \d+$/ })
      .first();
    if (!(await button.count())) break;
    const startedAt = Date.now();
    try {
      await button.click();
    } catch {
      runtime.addFinding(
        "fail",
        "holding list",
        "A Show all control could not be opened.",
      );
      break;
    }
    opened += 1;
    runtime.count("expanders");
    await runtime.checkScreen(startedAt);
  }
  if (
    opened === 20 &&
    (await runtime.page.getByRole("button", { name: /^Show all \d+$/ }).count())
  ) {
    runtime.addFinding(
      "fail",
      "holding list",
      "The Show all limit was reached before every group opened.",
    );
  }
}

async function clickAndCancelDialog(
  runtime: LiveRuntime,
  trigger: Locator,
  check: string,
): Promise<boolean> {
  if (!(await trigger.count())) return true;
  const button = trigger.first();
  if (!(await button.isVisible().catch(() => false))) return true;
  const startedAt = Date.now();
  const dialogIndex = await runtime.page.getByRole("dialog").count();
  await button.click().catch(() => undefined);
  const dialog = runtime.page.getByRole("dialog").nth(dialogIndex);
  if (!(await waitForDialogOpen(runtime, dialog, check))) return false;
  runtime.count("dialogs");
  await runtime.checkScreen(startedAt);
  const cancel = dialog.getByRole("button", { name: /^cancel$/i }).first();
  if (await cancel.isVisible().catch(() => false)) {
    await cancel.click().catch(() => undefined);
  } else {
    await runtime.page.keyboard.press("Escape").catch(() => undefined);
  }
  return waitForDialogClose(runtime, dialog, check);
}

async function waitForDialogOpen(
  runtime: LiveRuntime,
  dialog: Locator,
  check: string,
): Promise<boolean> {
  try {
    await dialog.waitFor({ state: "visible", timeout: 5_000 });
    return true;
  } catch {
    runtime.addFinding("fail", "dialog", check + " did not open.");
    return false;
  }
}

async function waitForDialogClose(
  runtime: LiveRuntime,
  dialog: Locator,
  check: string,
): Promise<boolean> {
  try {
    await dialog.waitFor({ state: "hidden", timeout: 5_000 });
    return true;
  } catch {
    runtime.addFinding("fail", "dialog", check + " did not close.");
    return false;
  }
}

async function openEverySettingsDialog(runtime: LiveRuntime): Promise<void> {
  const { page } = runtime;
  if (
    !(await clickAndCancelDialog(
      runtime,
      page.getByRole("button", { name: /remove person/i }),
      "Remove person confirmation",
    ))
  )
    return;
  if (
    !(await clickAndCancelDialog(
      runtime,
      page.getByRole("button", { name: /risk questions/i }),
      "Risk questions",
    ))
  )
    return;
  if (
    !(await clickAndCancelDialog(
      runtime,
      page.getByRole("button", { name: /delete (my )?account/i }),
      "Delete account confirmation",
    ))
  )
    return;
  if (
    !(await clickAndCancelDialog(
      runtime,
      page.getByRole("button", { name: /sign out on all devices/i }),
      "Signed-in devices confirmation",
    ))
  )
    return;
  await clickAndCancelDialog(
    runtime,
    page.getByRole("button", { name: /^sign out$/i }),
    "Settings sign-out confirmation",
  );
}

async function exerciseAccountMenu(runtime: LiveRuntime): Promise<void> {
  const trigger = runtime.page.getByRole("button", {
    name: /account menu/i,
  });
  if (!(await trigger.count())) {
    runtime.addFinding(
      "fail",
      "account menu",
      "The account menu button was not found on Home.",
    );
    return;
  }
  const startedAt = Date.now();
  const dialogIndex = await runtime.page.getByRole("dialog").count();
  await trigger.click().catch(() => undefined);
  const dialog = runtime.page.getByRole("dialog").nth(dialogIndex);
  if (!(await waitForDialogOpen(runtime, dialog, "Account menu"))) return;
  runtime.count("dialogs");
  await runtime.checkScreen(startedAt);
  const signOut = dialog.getByRole("button", { name: /^sign out$/i });
  if (await signOut.isVisible().catch(() => false)) {
    const signOutStartedAt = Date.now();
    await signOut.click().catch(() => undefined);
    // The confirmation replaces the menu inside the same sheet. It has its own
    // "Sign out" button, which must never be pressed: only Cancel.
    const question = dialog.getByText(/^Sign out of .+ on this phone\?$/);
    const cancel = dialog.getByRole("button", { name: "Cancel", exact: true });
    try {
      await question.waitFor({ state: "visible", timeout: 5_000 });
      runtime.count("dialogs");
      await runtime.checkScreen(signOutStartedAt);
      await cancel.click({ timeout: 5_000 });
      await question.waitFor({ state: "hidden", timeout: 5_000 });
    } catch {
      runtime.addFinding(
        "fail",
        "dialog",
        "The sign-out confirmation did not open or close in the account sheet.",
      );
    }
  }
  await runtime.page.keyboard.press("Escape").catch(() => undefined);
  await waitForDialogClose(runtime, dialog, "Account menu");
  await runtime.checkScreen();
}

async function runGuidedTour(runtime: LiveRuntime): Promise<void> {
  runtime.setScreen("/");
  const dialogIndex = await runtime.page.getByRole("dialog").count();
  await runtime.page.evaluate(() => {
    window.dispatchEvent(new Event("wealth:show-tour"));
  });
  const dialog = runtime.page.getByRole("dialog").nth(dialogIndex);
  if (!(await waitForDialogOpen(runtime, dialog, "Guided tour"))) return;
  for (let index = 0; index < tourPaths.length; index += 1) {
    runtime.setScreen(tourPaths[index] ?? "/");
    runtime.count("dialogs");
    await runtime.checkScreen();
    const buttonName = index + 1 === tourPaths.length ? "Finish" : "Next";
    await dialog
      .getByRole("button", { name: buttonName, exact: true })
      .click()
      .catch(() => undefined);
  }
  await waitForDialogClose(runtime, dialog, "Guided tour");
}

async function saveMoneyInventory(
  runtime: LiveRuntime,
  inventory: Inventory,
): Promise<void> {
  await showAllHoldings(runtime);
  const assetPaths = await hrefs(runtime.page, "/money/asset/");
  const holdingPaths = await hrefs(runtime.page, "/money/holding/");
  const loanPaths = await hrefs(runtime.page, "/money/loan/");
  assetPaths.forEach((path) => inventory.assets.add(path));
  holdingPaths.forEach((path) => inventory.holdings.add(path));
  loanPaths.forEach((path) => inventory.loans.add(path));

  const assetSnapshot = runtime.snapshots.get("/v1/assets");
  if (assetSnapshot !== undefined) {
    const expected = new Set(
      rows(assetSnapshot, ["assets", "items"])
        .filter((row) => row.asset_type !== "loan")
        .map((row) => fieldId(row, ["id", "asset_id"]))
        .filter((value): value is string => value !== null),
    );
    const actual = new Set(
      assetPaths.map((path) =>
        decodeURIComponent(path.split("/").at(-1) ?? ""),
      ),
    );
    checkSet(runtime, "assets", expected, actual);
  }

  const networth = runtime.snapshots.get("/v1/networth");
  if (networth !== undefined) {
    const expected = new Set(
      rows(networth, ["holdings"])
        .filter((row) => !row.recorded_buy_id)
        .map((row) => fieldId(row, ["security_id"]))
        .filter((value): value is string => value !== null),
    );
    const actual = idsFromPaths(holdingPaths, "/money/holding/");
    checkSet(runtime, "holdings", expected, actual);
  }

  const loanSnapshot = runtime.snapshots.get("/v1/loans");
  if (loanSnapshot !== undefined) {
    const expected = idsFrom(
      loanSnapshot,
      ["loans", "items"],
      ["id", "loan_id"],
    );
    const actual = new Set(
      loanPaths.map((path) => decodeURIComponent(path.split("/").at(-1) ?? "")),
    );
    checkSet(runtime, "loans", expected, actual);
  }
}

async function currencyViews(
  runtime: LiveRuntime,
  inventory: Inventory,
): Promise<void> {
  await visit(runtime, "/settings");
  await openAllReadOnlyStates(runtime);
  const selector = runtime.page.locator("#primary-currency");
  if (!(await selector.count())) return;
  const initial = await selector.inputValue();
  const options = await selector
    .locator("option")
    .evaluateAll((items) =>
      items.map((item) => (item as HTMLOptionElement).value).filter(Boolean),
    );
  try {
    for (const currency of options) {
      const startedAt = Date.now();
      await selector.selectOption(currency).catch(() => undefined);
      await runtime.checkScreen(startedAt);
      await visit(runtime, "/money");
      await openAllReadOnlyStates(runtime);
      await saveMoneyInventory(runtime, inventory);
    }
  } finally {
    await visit(runtime, "/settings");
    await openAllReadOnlyStates(runtime);
    const restore = runtime.page.locator("#primary-currency");
    if (await restore.count()) {
      const startedAt = Date.now();
      await restore.selectOption(initial).catch(() => undefined);
      await runtime.checkScreen(startedAt);
    }
  }
}

async function memberViews(
  runtime: LiveRuntime,
  inventory: Inventory,
): Promise<void> {
  await visit(runtime, "/money");
  const selector = runtime.page.locator("#header-member-filter");
  if (!(await selector.count())) {
    await openAllReadOnlyStates(runtime);
    await saveMoneyInventory(runtime, inventory);
    return;
  }
  const initial = await selector.inputValue();
  const values = await selector
    .locator("option")
    .evaluateAll((items) =>
      items.map((item) => (item as HTMLOptionElement).value).filter(Boolean),
    );
  try {
    for (const value of values) {
      const startedAt = Date.now();
      await selector.selectOption(value).catch(() => undefined);
      await runtime.checkScreen(startedAt);
      await openAllReadOnlyStates(runtime);
      await saveMoneyInventory(runtime, inventory);
    }
  } finally {
    const startedAt = Date.now();
    await selector.selectOption(initial).catch(() => undefined);
    await runtime.checkScreen(startedAt);
  }
}

async function followDetailRoutes(
  runtime: LiveRuntime,
  source: string,
  paths: string[],
): Promise<void> {
  for (const path of [...new Set(paths)]) {
    runtime.setScreen(source);
    runtime.count("listItems");
    await visit(runtime, path);
    await openAllReadOnlyStates(runtime);
  }
}

async function inspectImportHistory(runtime: LiveRuntime): Promise<string[]> {
  const summary = runtime.page
    .locator("main summary")
    .filter({ hasText: /^\s*Previous uploads\s*$/ });
  if (!(await summary.count())) {
    runtime.addFinding(
      "fail",
      "past imports",
      "The Previous uploads list was not available on Money.",
    );
    return [];
  }

  const history = summary.locator("xpath=..");
  if ((await history.getAttribute("open")) === null) {
    const startedAt = Date.now();
    await summary.click().catch(() => undefined);
    if ((await history.getAttribute("open")) === null) {
      runtime.addFinding(
        "fail",
        "past imports",
        "The Previous uploads list could not be opened.",
      );
      return [];
    }
    runtime.count("expanders");
    await runtime.checkScreen(startedAt);
  }

  const importPaths = await hrefs(history, "/import/");
  const importSnapshot = runtime.snapshots.get("/v1/imports");
  if (importSnapshot !== undefined) {
    const expectedImports = idsFrom(
      importSnapshot,
      ["jobs", "imports", "items"],
      ["id", "job_id"],
    );
    checkSet(
      runtime,
      "past imports",
      expectedImports,
      idsFromPaths(importPaths, "/import/"),
    );
  }

  return importPaths;
}

async function closeImportHistoryIfOpen(runtime: LiveRuntime): Promise<void> {
  const summary = runtime.page
    .locator("main summary")
    .filter({ hasText: /^\s*Previous uploads\s*$/ });
  if (!(await summary.count())) return;
  const history = summary.locator("xpath=..");
  if ((await history.getAttribute("open")) === null) return;

  const startedAt = Date.now();
  await summary.click().catch(() => undefined);
  if ((await history.getAttribute("open")) !== null) {
    runtime.addFinding(
      "fail",
      "past imports",
      "The Previous uploads list could not be closed.",
    );
    return;
  }
  await runtime.checkScreen(startedAt);
}

async function openRecordedBuyDetails(runtime: LiveRuntime): Promise<void> {
  const buttons = runtime.page
    .locator("main section[aria-label] button")
    .filter({ has: runtime.page.locator("[data-security]") });
  const count = await buttons.count();
  for (let index = 0; index < count; index += 1) {
    const button = buttons.nth(index);
    const startedAt = Date.now();
    const dialogIndex = await runtime.page.getByRole("dialog").count();
    await button.click().catch(() => undefined);
    const dialog = runtime.page.getByRole("dialog").nth(dialogIndex);
    if (
      !(await waitForDialogOpen(runtime, dialog, "Recorded purchase details"))
    )
      continue;
    runtime.count("dialogs");
    runtime.count("listItems");
    await runtime.checkScreen(startedAt);
    const cancel = dialog.getByRole("button", { name: /^cancel$/i }).first();
    if (await cancel.isVisible().catch(() => false)) {
      await cancel.click().catch(() => undefined);
    } else {
      await runtime.page.keyboard.press("Escape").catch(() => undefined);
    }
    if (
      !(await waitForDialogClose(runtime, dialog, "Recorded purchase details"))
    ) {
      break;
    }
    await runtime.checkScreen();
  }
}

async function readMoneyValue(
  page: LiveRuntime["page"],
  kind: "home" | "money",
): Promise<string | null> {
  if (kind === "home") {
    return page.evaluate(() => {
      const links = Array.from(document.querySelectorAll('a[href="/money"]'));
      const link = links.find((item) =>
        /[₹]|DKK|kr\s*[\d]/i.test(item.textContent ?? ""),
      );
      const match = link?.textContent?.match(
        /(?:₹|DKK|kr)\s*[\d.,]+(?:\s*[A-Za-z]+)?/i,
      );
      return match?.[0]?.replace(/\s+/g, " ").trim() ?? null;
    });
  }
  return page.evaluate(() => {
    const labels = Array.from(document.querySelectorAll("main p"));
    const label = labels.find((item) =>
      /^net worth\b/i.test(item.textContent?.trim() ?? ""),
    );
    const value = label?.parentElement?.querySelector("p.text-lg");
    return value?.textContent?.trim() ?? null;
  });
}

async function checkHomeMoneyTotal(
  runtime: LiveRuntime,
  homeValue: string | null,
): Promise<void> {
  const moneyValue = await readMoneyValue(runtime.page, "money");
  if (homeValue && moneyValue && homeValue !== moneyValue) {
    runtime.addFinding(
      "fail",
      "matching totals",
      "Home: " + homeValue + "; Money: " + moneyValue + ".",
    );
  }
}

test("live account screens can be reviewed without saving", async ({
  live,
}, testInfo) => {
  testInfo.setTimeout(60 * 60 * 1_000);
  const { page, environment } = live;
  const inventory = emptyInventory();
  const planHoldingPaths = new Set<string>();
  const homeAssetPaths = new Set<string>();

  live.setScreen("/");
  const firstLoad = Date.now();
  try {
    await page.goto(new URL("/", environment.pwaUrl).toString(), {
      waitUntil: "domcontentloaded",
      timeout: 60_000,
    });
  } catch {
    live.addFinding("fail", "navigation", "Home could not be opened.");
  }
  await page
    .waitForFunction(
      () =>
        window.location.pathname === "/login" ||
        document.querySelector('nav[aria-label="Primary"]') !== null,
      undefined,
      { timeout: 60_000 },
    )
    .catch(() => undefined);
  const firstPath = await page.evaluate(() => window.location.pathname);
  if (firstPath === "/login") {
    throw new Error("Not signed in: run npm run live:login first.");
  }
  await live.checkScreen(firstLoad);
  await runGuidedTour(live);

  await clickTab(live, "Home", "/");
  await openAllReadOnlyStates(live);
  const homeTotal = await readMoneyValue(page, "home");
  for (const path of await hrefs(page, "/money/asset/"))
    homeAssetPaths.add(path);
  const actionCards = page.locator('article[data-testid^="action-"]');
  const actionCount = await actionCards.count();
  for (let index = 0; index < actionCount; index += 1) {
    const action = actionCards.nth(index);
    const showSteps = action.getByRole("button").first();
    const startedAt = Date.now();
    const dialogIndex = await page.getByRole("dialog").count();
    await showSteps.click().catch(() => undefined);
    const dialog = page.getByRole("dialog").nth(dialogIndex);
    if (!(await waitForDialogOpen(live, dialog, "Plan step details"))) continue;
    live.count("dialogs");
    await live.checkScreen(startedAt);
    await openAllReadOnlyStates(live);
    await page.keyboard.press("Escape").catch(() => undefined);
    if (!(await waitForDialogClose(live, dialog, "Plan step details"))) break;
  }

  await clickTab(live, "Money", "/money");
  await openAllReadOnlyStates(live);
  await saveMoneyInventory(live, inventory);
  await checkHomeMoneyTotal(live, homeTotal);
  await openRecordedBuyDetails(live);
  await currencyViews(live, inventory);
  await memberViews(live, inventory);
  const importPaths = await inspectImportHistory(live);
  await followDetailRoutes(live, "/money", importPaths);
  await visit(live, "/money");
  await closeImportHistoryIfOpen(live);
  for (const path of await hrefs(page, "/money/asset/"))
    homeAssetPaths.add(path);
  for (const type of addTypes) {
    await visit(live, "/money/add/" + type);
  }

  await clickTab(live, "Plan", "/plan");
  await openAllReadOnlyStates(live);
  for (const path of await hrefs(page, "/money/holding/"))
    planHoldingPaths.add(path);

  await clickTab(live, "Ideas", "/ideas");
  await openAllReadOnlyStates(live);
  await visit(live, "/ideas/world");
  await openAllReadOnlyStates(live);

  await clickTab(live, "Inbox", "/inbox");
  await openAllReadOnlyStates(live);
  const inboxSnapshot = live.snapshots.get("/v1/inbox");
  const briefPaths = await hrefs(page, "/inbox/brief/");
  if (inboxSnapshot !== undefined) {
    const expectedBriefs = new Set(
      rows(inboxSnapshot, ["briefs"])
        .map((item) => fieldId(item, ["week"]))
        .filter((value): value is string => value !== null),
    );
    const actualBriefs = new Set(
      briefPaths.map((path) =>
        decodeURIComponent(path.split("/").at(-1) ?? ""),
      ),
    );
    checkSet(live, "weekly briefs", expectedBriefs, actualBriefs);
  }
  const statementSnapshot = live.snapshots.get("/v1/statement-inbox/items");
  if (statementSnapshot !== undefined) {
    const inboxItems = rows(statementSnapshot, ["items"]);
    const statementIds = inboxItems
      .map((item) => fieldId(item, ["id", "item_id"]))
      .filter((value): value is string => value !== null);
    const visibleItems = await page
      .locator('section[aria-labelledby="gmail-statement-inbox-heading"] h3')
      .count();
    if (inboxItems.length !== visibleItems) {
      live.addFinding(
        "fail",
        "statement row count",
        "statement inbox response IDs [" +
          statementIds.join(", ") +
          "]; " +
          inboxItems.length +
          " response item(s), " +
          visibleItems +
          " visible item(s).",
      );
    }
  }
  await followDetailRoutes(live, "/inbox", briefPaths);

  await visit(live, "/import");
  await openAllReadOnlyStates(live);

  await visit(live, "/settings");
  await openAllReadOnlyStates(live);
  await openEverySettingsDialog(live);

  await visit(live, "/settings/gmail");
  await openAllReadOnlyStates(live);

  await visit(live, "/help");
  await openAllReadOnlyStates(live);

  await visit(live, "/");
  await exerciseAccountMenu(live);

  const allAssetPaths = new Set([...inventory.assets, ...homeAssetPaths]);
  const allHoldingPaths = new Set([...inventory.holdings, ...planHoldingPaths]);
  const allLoanPaths = [...inventory.loans];
  await visit(live, "/money");
  await openAllReadOnlyStates(live);
  await followDetailRoutes(live, "/money", [...allAssetPaths]);
  await followDetailRoutes(live, "/money", [...allHoldingPaths]);
  await followDetailRoutes(live, "/money", allLoanPaths);

  for (const path of ["/does-not-exist", "/login", "/join", "/onboarding"]) {
    await visit(live, path);
  }

  const fails = live.findings.filter(
    (finding) => finding.severity === "fail",
  ).length;
  const notes = live.findings.filter(
    (finding) => finding.severity === "note",
  ).length;
  const routes = [
    ...new Set(live.findings.map((finding) => finding.screen)),
  ].sort();
  console.log(
    "Live check counts: " +
      fails +
      " fail, " +
      notes +
      " note; routes: " +
      (routes.join(", ") || "none"),
  );
  expect(
    fails,
    "Live look-only finding count: " + fails + " fail, " + notes + " note.",
  ).toBe(0);
});
