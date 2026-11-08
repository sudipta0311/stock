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
  await page.setViewportSize({ width: 360, height: 740 });
  const identity = await startReadyAccount(page, "holding-review");
  await seedHoldingReviews(identity.email);

  await gotoApp(page, "/money");
  await expect(
    page.getByText("Synthetic share to add", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Add more", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Synthetic share to sell", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Sell", { exact: true })).toBeVisible();
  await expectFitsScreen(page);

  await page.getByText("Synthetic share to add", { exact: true }).click();
  await expect(page.getByRole("heading", { name: "Our view" })).toBeVisible();
  await expect(page.getByText("Add more", { exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Back to Money" })).toBeVisible();
  const buyingPrice = page.getByText("Buying price not recorded", {
    exact: true,
  });
  await expect(buyingPrice).toBeVisible();
  await expect(buyingPrice).toHaveClass(/text-muted-foreground/);
  await expect(buyingPrice).not.toHaveClass(/font-semibold/);
  await expect(
    page
      .getByText("Why this suggestion?", { exact: true })
      .locator("xpath=ancestor::details"),
  ).not.toHaveAttribute("open");
  await expect(
    page.getByText("How we check", { exact: true }),
  ).not.toBeVisible();
  await expect(
    page.getByText("Recent checks support adding more shares.", {
      exact: true,
    }),
  ).toBeVisible();
  await page.getByText("Why this suggestion?", { exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "How we check", level: 3 }),
  ).toBeVisible();
  const companyResultsCheck = page
    .getByRole("listitem")
    .filter({ hasText: "Company results" });
  await expect(companyResultsCheck).toHaveCount(0);
  await expect(
    page.getByText(/^Waiting for data: .*Company results/),
  ).toBeVisible();
  await expectFitsScreen(page);

  const addViolations = (
    await new AxeBuilder({ page }).analyze()
  ).violations.filter(
    (violation) =>
      violation.impact === "critical" || violation.impact === "serious",
  );
  await writeReport(
    "holding-review-add-axe",
    testInfo.project.name,
    addViolations,
  );
  expect(addViolations).toEqual([]);

  await gotoApp(page, "/money");
  await page.getByText("Synthetic share to sell", { exact: true }).click();
  await expect(page.getByRole("heading", { name: "Our view" })).toBeVisible();
  await expect(page.getByText("Sell", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Company revenue fell for 2 quarters in a row.", {
      exact: true,
    }),
  ).toBeVisible();
  await expectFitsScreen(page);

  const sellViolations = (
    await new AxeBuilder({ page }).analyze()
  ).violations.filter(
    (violation) =>
      violation.impact === "critical" || violation.impact === "serious",
  );
  await writeReport(
    "holding-review-sell-axe",
    testInfo.project.name,
    sellViolations,
  );
  expect(sellViolations).toEqual([]);

  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await expect(page.getByTestId("action-SELL_EXIT")).toHaveCount(1);
  const sellStep = page.getByTestId("action-SELL_EXIT");
  await expect(
    sellStep.getByRole("button", { name: "I sold it" }),
  ).toBeVisible();
  await sellStep.getByRole("button", { name: "Show steps" }).click();
  const instructions = page.getByRole("dialog", {
    name: "How to take this step",
  });
  await expect(instructions).toBeVisible();
  await expect(instructions).not.toContainText(
    /invest as planned|monthly investment|buy order/i,
  );
  await expectFitsScreen(page);
});
