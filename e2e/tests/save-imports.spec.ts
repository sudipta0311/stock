import { randomUUID } from "node:crypto";
import { expectFitsScreen, gotoApp, startReadyAccount } from "../support/app";
import { expect, test } from "../support/save-test";

test("column mapping, import row correction, and discard survive reload", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-import-review");
  await gotoApp(page, "/import");
  await expectFitsScreen(page);
  await page
    .getByRole("button", { name: "Broker file (Excel or CSV)" })
    .click();
  await page
    .getByLabel("Where did you get this file?")
    .selectOption("indmoney");
  const symbol = `SYNTH${randomUUID().replace(/\W/g, "").slice(0, 6).toUpperCase()}`;
  await page.getByLabel("Statement file").setInputFiles({
    name: "synthetic-save-review.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(
      `Symbol,Name,Quantity,Invested Value,Current Market Price\n${symbol},Synthetic review row,3,90.00,40.00\n`,
      "utf8",
    ),
  });
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Upload file" }).click();
  await expect(
    page.getByRole("heading", { name: "Review your import" }),
  ).toBeVisible({ timeout: 30_000 });
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Fix columns" }).click();
  await expect(
    page.getByRole("heading", { name: "Match the file columns" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByLabel("Security code").selectOption("");
  await page.getByLabel("Share symbol").selectOption("Symbol");
  await page.getByLabel("Company name").selectOption("Name");
  await page.locator("#mapping-quantity").selectOption("Quantity");
  await page.getByLabel("Save this format as").fill("Synthetic broker format");
  await page.getByRole("button", { name: "Use these columns" }).click();
  await expect(
    page.getByText("Synthetic review row", { exact: true }),
  ).toBeVisible();
  await expectFitsScreen(page);

  const row = page.getByRole("checkbox").first();
  if (!(await row.isChecked())) {
    const include = page.waitForResponse(
      (response) =>
        response.request().method() === "PATCH" &&
        new URL(response.url()).pathname.includes("/v1/imports/") &&
        new URL(response.url()).pathname.includes("/items/"),
    );
    await row.check();
    expect((await include).ok()).toBeTruthy();
  }
  await expect(row).toBeChecked();
  const exclude = page.waitForResponse(
    (response) =>
      response.request().method() === "PATCH" &&
      new URL(response.url()).pathname.includes("/v1/imports/") &&
      new URL(response.url()).pathname.includes("/items/"),
  );
  await row.uncheck();
  expect((await exclude).ok()).toBeTruthy();
  await expect(row).not.toBeChecked();
  await page.reload();
  await expect(page.getByRole("checkbox").first()).not.toBeChecked();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Discard", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Discard this import?" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Discard import" }).click();
  await expect(page).toHaveURL(/\/money$/);
  await page.reload();
  await expect(page.getByRole("heading", { name: "Your money" })).toBeVisible();
  await page.getByText("Previous uploads", { exact: true }).click();
  await expect(
    page.getByRole("link", { name: /Statement import .* discarded/ }),
  ).toBeVisible();
  await expectFitsScreen(page);
});
