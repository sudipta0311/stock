import {
  completeFixedDeposit,
  expectFitsScreen,
  gotoApp,
  openFixedDepositForm,
  seedMoneyListHoldings,
  startReadyAccount,
} from "../support/app";
import { allowConsoleErrors, expect, test } from "../support/save-test";
import type { Page } from "@playwright/test";
async function addFromMenu(
  page: Page,
  tile: string,
  fill: () => Promise<void>,
  label: string,
  needsOfficialRate: boolean = false,
): Promise<void> {
  await gotoApp(page, "/money");
  await expectFitsScreen(page);
  const checklist = page.locator("#money-checklist");
  if (
    !(await checklist.evaluate((node) => (node as HTMLDetailsElement).open))
  ) {
    await checklist.locator(":scope > summary").click();
  }
  const selectedTiles = checklist.locator('button[aria-pressed="true"]');
  while (await selectedTiles.count()) {
    const saveSelection = page.waitForResponse(
      (response) =>
        response.request().method() === "PUT" &&
        new URL(response.url()).pathname === "/v1/me/checklist",
    );
    await selectedTiles.first().click();
    expect((await saveSelection).ok()).toBeTruthy();
  }
  const targetTile = checklist.getByRole("button", { name: tile, exact: true });
  const saveTarget = page.waitForResponse(
    (response) =>
      response.request().method() === "PUT" &&
      new URL(response.url()).pathname === "/v1/me/checklist",
  );
  await targetTile.click();
  expect((await saveTarget).ok()).toBeTruthy();
  await expect(targetTile).toHaveAttribute("aria-pressed", "true");
  await expectFitsScreen(page);
  await page.getByRole("button", { name: /Continue \(1 selected\)/ }).click();
  await expectFitsScreen(page);
  await fill();
  if (needsOfficialRate) {
    const expectedRateError =
      /^console: Failed to load resource: the server responded with a status of 422 .*\/v1\/assets$/;
    allowConsoleErrors(page, [expectedRateError, expectedRateError]);
    await page.getByRole("button", { name: "Save and return" }).click();
    const rate = page.getByLabel("Interest rate (%)");
    await expect(rate).toBeVisible();
    await expect(rate).toBeFocused();
    await expectFitsScreen(page);
    await rate.fill("6.8");
  }
  const detailsSummary = page.getByText("More details (for a closer estimate)");
  const details = detailsSummary.locator("xpath=ancestor::details[1]");
  if (!(await details.evaluate((node) => (node as HTMLDetailsElement).open))) {
    await detailsSummary.click();
  }
  await page.getByLabel("Name this item").fill(label);
  await page.getByRole("button", { name: "Save and return" }).click();
  await expect(page).toHaveURL(/\/money$/);
  await page.reload();
  await expect(page.getByText(label, { exact: true })).toBeVisible();
  await expectFitsScreen(page);
}

test("money add menu saves every remaining asset form family", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-money-forms");

  await addFromMenu(
    page,
    "EPF",
    async () => {
      await page.getByLabel("Current EPF balance").fill("95000");
    },
    "Synthetic EPF",
  );
  await addFromMenu(
    page,
    "PPF",
    async () => {
      await page.getByLabel("Current PPF balance").fill("125000");
      await page.getByLabel("Year opened").fill("2020");
    },
    "Synthetic PPF",
  );
  await addFromMenu(
    page,
    "Post Office",
    async () => {
      await page.getByLabel("Post Office scheme").selectOption("nsc");
      await page.getByLabel("Amount invested").fill("30000");
      await page.getByLabel("Purchased (month and year)").fill("2024-01");
    },
    "Synthetic certificate",
    true,
  );
  await addFromMenu(
    page,
    "NPS",
    async () => {
      await page.getByLabel("NPS Tier I balance").fill("45000");
    },
    "Synthetic NPS",
  );
  await addFromMenu(
    page,
    "LIC / Insurance",
    async () => {
      await page.getByLabel("Insurance type").selectOption("term_insurance");
      await page.getByLabel("Insurer").fill("Synthetic Cover Co");
      await page.getByLabel("Life cover amount").fill("1000000");
      await page.getByLabel("Cover ends in year").fill("2040");
    },
    "Synthetic term cover",
  );
  await addFromMenu(
    page,
    "Gold",
    async () => {
      await page.getByLabel("Type of gold").selectOption("physical");
      await page.getByLabel("Weight in grams").fill("10");
      await page.getByLabel("Purity (karat)").selectOption("22");
    },
    "Synthetic gold",
  );
  await addFromMenu(
    page,
    "Bonds",
    async () => {
      await page.getByLabel("Bond type").selectOption("gsec");
      await page.getByLabel("Amount invested").fill("50000");
      await page.getByLabel("Matures (month and year)").fill("2029-06");
    },
    "Synthetic bond",
  );
  await addFromMenu(
    page,
    "Property",
    async () => {
      await page.getByLabel("Property type").fill("Home");
      await page.getByLabel("Estimated value").fill("2500000");
    },
    "Synthetic property",
  );
});

test("asset edits, offline replay, deletion, and checklist survive reload", async ({
  page,
  context,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-money-edits");
  await openFixedDepositForm(page);
  await completeFixedDeposit(page);
  await expectFitsScreen(page);

  const assetLink = page.getByRole("link", {
    name: /Fixed deposit at Synthetic Bank/,
  });
  await assetLink.click();
  await expect(
    page.getByRole("heading", { name: "Fixed deposit" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Edit details" }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Amount deposited").fill("600000");
  await page.getByRole("button", { name: "Save and return" }).click();
  await page.reload();
  await expect(page.getByText(/6,00,000/)).toBeVisible();
  await expectFitsScreen(page);

  await page.getByRole("button", { name: "Edit details" }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Amount deposited").fill("700000");
  await context.setOffline(true);
  await page.getByRole("button", { name: "Save and return" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: /1 change waiting to sync/ }),
  ).toBeVisible();
  await expectFitsScreen(page);
  const replay = page.waitForResponse(
    (response) =>
      response.request().method() === "PATCH" &&
      new URL(response.url()).pathname.startsWith("/v1/assets/"),
  );
  await context.setOffline(false);
  expect((await replay).ok()).toBeTruthy();
  await page.reload();
  await expect(page.getByText(/7,00,000/)).toBeVisible();

  await page.getByText("Edit history", { exact: true }).click();
  await page.getByRole("button", { name: "Mark as closed" }).first().click();
  await page.getByRole("button", { name: "Mark as closed" }).last().click();
  await expect(page).toHaveURL(/\/money$/);
  await page.reload();
  await expect(
    page.getByRole("link", { name: /Fixed deposit at Synthetic Bank/ }),
  ).toHaveCount(0);

  await page.locator("#money-checklist").locator(":scope > summary").click();
  await expectFitsScreen(page);
  const ppf = page.getByRole("button", { name: "PPF", exact: true });
  const checklistSave = page.waitForResponse(
    (response) =>
      response.request().method() === "PUT" &&
      new URL(response.url()).pathname === "/v1/me/checklist",
  );
  await ppf.click();
  await expect(ppf).toHaveAttribute("aria-pressed", "true");
  expect((await checklistSave).ok()).toBeTruthy();
  await page.reload();
  await page.getByText("Add manually", { exact: true }).click();
  await expect(
    page.getByRole("button", { name: "PPF", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expectFitsScreen(page);
});

test("show all opens only the selected holding list", async ({ page }) => {
  const identity = await startReadyAccount(page, "money-holding-lists");
  await seedMoneyListHoldings(identity.email);
  await gotoApp(page, "/money");

  const funds = page.getByRole("region", { name: "Funds" });
  const shares = page.getByRole("region", { name: "Shares" });
  const fundToggle = funds.getByRole("button", { name: "Show all 9" });
  await expect(fundToggle).toBeVisible();
  await expect(shares.getByRole("button", { name: "Show all 9" })).toBeVisible();
  await expect(funds.getByText("Synthetic fund 1", { exact: true })).toHaveCount(0);
  await expect(shares.getByText("Synthetic share 1", { exact: true })).toHaveCount(0);

  await shares.getByRole("button", { name: "Show all 9" }).click();

  await expect(shares.getByText("Synthetic share 1", { exact: true })).toBeVisible();
  await expect(funds.getByText("Synthetic fund 1", { exact: true })).toHaveCount(0);
  await expect(fundToggle).toBeVisible();
});
