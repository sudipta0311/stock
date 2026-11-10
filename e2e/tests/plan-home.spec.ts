import {
  addDanishSavings,
  expect,
  expectFitsScreen,
  gotoApp,
  seedHomeAction,
  skipGuidedTour,
  startReadyAccount,
  test,
} from "../support/app";

test("Plan and Home show household values, closed details, and action status", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  const identity = await startReadyAccount(page, "plan-home");
  await addDanishSavings(page);

  await gotoApp(page, "/plan");
  await expect(
    page.getByRole("heading", { name: "Your money vs your plan" }),
  ).toBeVisible();
  await expect(page.getByText("You have", { exact: true })).toBeVisible();
  await expect(page.getByText("Target", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Current amount", { exact: true }),
  ).not.toBeVisible();
  await expectFitsScreen(page);

  await seedHomeAction(identity.email);
  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await expect(page.getByRole("heading", { name: "This week" })).toBeVisible();
  await expect(page.getByRole("article", { name: /For you/ })).toBeVisible();
  await page.getByRole("button", { name: "Done" }).click();
  await expect(
    page.getByText("How much can you invest each month?", { exact: true }),
  ).toBeVisible();
  // The amount field shares its row with the currency switch.
  await expectFitsScreen(page);
  await page.getByLabel("Monthly amount").fill("25000");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  const newMonthlyStep = page.getByRole("article", {
    name: /Change your monthly investments.*For you/,
  });
  await expect(newMonthlyStep).toBeVisible();
  await expect(newMonthlyStep).toContainText("25,000");
  await page.reload();
  await expect(newMonthlyStep).toBeVisible();
  await expect(newMonthlyStep).toContainText("25,000");
  await expectFitsScreen(page);

  await gotoApp(page, "/money");
  await expect(page.getByRole("heading", { name: "Your money" })).toBeVisible();
  await expectFitsScreen(page);
});
