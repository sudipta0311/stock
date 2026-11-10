import {
  expectFitsScreen,
  gotoApp,
  seedHoldingReviews,
  skipGuidedTour,
  startReadyAccount,
} from "../support/app";
import { expect, test } from "../support/save-test";

test("a shown Home highlight stays hidden after reload", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  const identity = await startReadyAccount(page, "save-highlight");
  await seedHoldingReviews(identity.email);
  await gotoApp(page, "/");
  await skipGuidedTour(page);
  const dismissButton = page.getByRole("button", { name: /^Hide:/ }).first();
  if (!(await dismissButton.count())) {
    test.skip(
      true,
      "This plan has no dismissible Home highlight for the synthetic data.",
    );
  }
  await expectFitsScreen(page);
  const label = await dismissButton.getAttribute("aria-label");
  expect(label).toBeTruthy();
  await dismissButton.click();
  await expect(page.getByRole("button", { name: label! })).toHaveCount(0);
  await page.reload();
  await expect(page.getByRole("button", { name: label! })).toHaveCount(0);
  await expectFitsScreen(page);
});
