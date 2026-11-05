import AxeBuilder from "@axe-core/playwright";
import {
  expect,
  expectFitsScreen,
  gotoApp,
  seedHoldingReviews,
  skipGuidedTour,
  startReadyAccount,
  test,
  writeReport,
} from "../support/app";

test("held shares show weekly reviews and Home has one sell step", async ({
  page,
}, testInfo) => {
  const identity = await startReadyAccount(page, "holding-review");
  await seedHoldingReviews(identity.email);

  await gotoApp(page, "/money");
  await expect(page.getByText("Synthetic share to add", { exact: true })).toBeVisible();
  await expect(page.getByText("Add more", { exact: true })).toBeVisible();
  await expect(page.getByText("Synthetic share to sell", { exact: true })).toBeVisible();
  await expect(page.getByText("Sell", { exact: true })).toBeVisible();
  await expectFitsScreen(page);

  await page.getByText("Synthetic share to add", { exact: true }).click();
  await expect(page.getByRole("heading", { name: "Our view" })).toBeVisible();
  await expect(page.getByText("Add more", { exact: true })).toBeVisible();
  await expect(page.getByText("How we check", { exact: true })).toBeVisible();
  await expect(page.getByText("Company results", { exact: true })).not.toBeVisible();
  await expect(page.getByText("We don't follow this company's results yet", { exact: false }))
    .not.toBeVisible();
  await expectFitsScreen(page);

  const addViolations = (await new AxeBuilder({ page }).analyze()).violations.filter(
    (violation) => violation.impact === "critical" || violation.impact === "serious",
  );
  await writeReport("holding-review-add-axe", testInfo.project.name, addViolations);
  expect(addViolations).toEqual([]);

  await gotoApp(page, "/money");
  await page.getByText("Synthetic share to sell", { exact: true }).click();
  await expect(page.getByRole("heading", { name: "Our view" })).toBeVisible();
  await expect(page.getByText("Sell", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Company revenue fell for 2 quarters in a row.", { exact: true }),
  ).toBeVisible();
  await expectFitsScreen(page);

  const sellViolations = (
    await new AxeBuilder({ page }).analyze()
  ).violations.filter(
    (violation) => violation.impact === "critical" || violation.impact === "serious",
  );
  await writeReport("holding-review-sell-axe", testInfo.project.name, sellViolations);
  expect(sellViolations).toEqual([]);

  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await expect(page.getByTestId("action-SELL_EXIT")).toHaveCount(1);
  await expectFitsScreen(page);
});
