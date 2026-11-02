import {
  expect,
  expectFitsScreen,
  gotoApp,
  seedDenmarkBuyAction,
  skipGuidedTour,
  startReadyAccount,
  test,
} from "../support/app";

test("records a Denmark fund purchase and keeps it in Money after reload", async ({
  page,
}) => {
  const identity = await startReadyAccount(page, "bought-it-denmark");
  await seedDenmarkBuyAction(identity.email);
  await gotoApp(page, "/");
  await skipGuidedTour(page);

  const buyAction = page.getByTestId("action-BUY_ABROAD");
  await buyAction.getByRole("button", { name: "Mark done" }).click();
  const sheet = page.getByRole("dialog", { name: "What did you buy?" });
  const fundRow = sheet.getByTestId("buy-row-0");
  await expect(fundRow).toBeVisible();
  const fundName = await fundRow.locator(".line-clamp-2").innerText();
  expect(fundName.length).toBeGreaterThan(20);
  await expect(sheet.getByLabel("Amount (DKK)")).toHaveValue(/\d+/);
  const details = sheet.locator("details");
  await expect(details).not.toHaveAttribute("open");
  await expectFitsScreen(page);

  await sheet.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByText("Done ✓")).toBeVisible();
  await expect(page.getByText("Added to your money.")).toBeVisible();

  await page.reload();
  await skipGuidedTour(page);
  await expect(page.getByText("Done ✓")).toBeVisible();
  await expect(page.getByText("Added to your money.")).toBeVisible();
  await expect(page.getByTestId("action-BUY_ABROAD")).toHaveCount(0);

  await gotoApp(page, "/money");
  const funds = page.getByRole("region", { name: "Funds" });
  const recordedFund = funds.getByRole("button").filter({ hasText: fundName });
  await expect(recordedFund).toBeVisible();
  await expect(recordedFund.getByText("Added by you")).toBeVisible();
  await expectFitsScreen(page);
});
