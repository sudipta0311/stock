import { expectFitsScreen, gotoApp, startReadyAccount } from "../support/app";
import { allowConsoleErrors, expect, test } from "../support/save-test";

test("quick add prefills EPF and explains an unreadable note", async ({
  page,
}) => {
  await startReadyAccount(page, "quick-add");
  await gotoApp(page, "/money");
  await expectFitsScreen(page);
  await page.getByText("Add manually", { exact: true }).click();
  await expectFitsScreen(page);
  await page.locator("summary").filter({ hasText: "Quick add" }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Quick add").fill("EPF 12L");
  await page.getByRole("button", { name: "Add details" }).click();
  await expect(page).toHaveURL(/\/money\/add\/epf$/);
  await expect(page.getByLabel("Current EPF balance")).toHaveValue(
    /12,00,000|1200000/,
  );
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Save and return" }).click();
  await expect(page).toHaveURL(/\/money$/);
  await page.reload();
  await expect(page.getByRole("link", { name: /EPF \/ VPF/ })).toBeVisible();
  await expectFitsScreen(page);

  await gotoApp(page, "/money");
  await page.getByText("Add manually", { exact: true }).click();
  await page.locator("summary").filter({ hasText: "Quick add" }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Quick add").fill("random words");
  allowConsoleErrors(page, [
    /^console: Failed to load resource: the server responded with a status of 422 \(Unprocessable Content\) @ http:\/\/127\.0\.0\.1:\d+\/v1\/assets\/quick-add$/,
  ]);
  await page.getByRole("button", { name: "Add details" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Try a short line" }),
  ).toBeVisible();
});
