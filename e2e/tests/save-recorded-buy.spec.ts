import {
  expectFitsScreen,
  gotoApp,
  seedDenmarkBuyAction,
  skipGuidedTour,
  startReadyAccount,
} from "../support/app";
import { expect, test } from "../support/save-test";

test("recorded purchase edits and removal persist", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  const identity = await startReadyAccount(page, "save-recorded-buy");
  await seedDenmarkBuyAction(identity.email);
  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await expectFitsScreen(page);
  await page
    .getByTestId("action-BUY_ABROAD")
    .getByRole("button", { name: "I bought it" })
    .click();
  const purchaseSheet = page.getByRole("dialog", { name: "What did you buy?" });
  await expectFitsScreen(page);
  const fundName = await purchaseSheet
    .getByTestId("buy-row-0")
    .locator(".line-clamp-2")
    .innerText();
  await purchaseSheet
    .getByRole("button", { name: "Save", exact: true })
    .click();
  await expect(page.getByText("Added to your money.")).toBeVisible();
  await page.reload();
  await gotoApp(page, "/money");
  await expectFitsScreen(page);
  const fundRow = page.getByRole("button").filter({ hasText: fundName });
  await expect(fundRow).toBeVisible();
  await fundRow.click();
  const details = page.getByRole("dialog", { name: "Added by you" });
  await details.getByRole("button", { name: "Edit" }).click();
  await expectFitsScreen(page);
  const amount = details.getByLabel("Amount (DKK)");
  await amount.fill("425");
  await details.getByRole("button", { name: "Save changes" }).click();
  await expect(details).toBeHidden();
  await page.reload();
  await page.getByRole("button").filter({ hasText: fundName }).click();
  const reloadedDetails = page.getByRole("dialog", { name: "Added by you" });
  await expect(reloadedDetails.getByText(/425(?:\.00)? DKK/)).toBeVisible();
  await expectFitsScreen(page);

  await reloadedDetails
    .getByRole("button", { name: "Remove purchase" })
    .click();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Remove purchase" }).click();
  await expect(reloadedDetails).toBeHidden();
  await page.reload();
  await expect(
    page.getByRole("button").filter({ hasText: fundName }),
  ).toHaveCount(0);
  await expectFitsScreen(page);
});
