import { expect, startReadyAccount, test } from "../support/app";
import { gotoApp } from "../support/app";

test("quick add prefills EPF and explains an unreadable note", async ({
  page,
}) => {
  await startReadyAccount(page, "quick-add");
  await gotoApp(page, "/money");
  await page.getByLabel("Quick add").fill("EPF 12L");
  await page.getByRole("button", { name: "Add details" }).click();
  await expect(page).toHaveURL(/\/money\/add\/epf$/);
  await expect(page.getByLabel("Current EPF balance")).toHaveValue(
    /12,00,000|1200000/,
  );
  await page.getByRole("button", { name: "Save now" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: /^Saved$/ }),
  ).toBeVisible();

  await gotoApp(page, "/money");
  await page.getByLabel("Quick add").fill("random words");
  await page.getByRole("button", { name: "Add details" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Try a short line" }),
  ).toBeVisible();
});
