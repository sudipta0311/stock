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
  const identity = await startReadyAccount(page, "plan-home");
  await addDanishSavings(page);

  await gotoApp(page, "/plan");
  await expect(
    page.getByRole("heading", { name: "Your money vs your plan" }),
  ).toBeVisible();
  await expect(page.getByText(/Your safe money target is/)).toBeVisible();
  await expect(
    page.getByText("Current amount", { exact: true }),
  ).not.toBeVisible();
  await expectFitsScreen(page);

  await seedHomeAction(identity.email);
  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await expect(page.getByRole("heading", { name: "This week" })).toBeVisible();
  await expect(page.getByRole("article", { name: /For you/ })).toBeVisible();
  await page.getByRole("button", { name: "Mark done" }).click();
  await expect(
    page.getByText("How much can you invest each month?", { exact: true }),
  ).toBeVisible();
  // The amount field shares its row with the currency switch.
  await expectFitsScreen(page);
  await page.getByLabel("Monthly amount").fill("25000");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByText("Nothing to buy this week", { exact: true }),
  ).toBeVisible();

  await gotoApp(page, "/money");
  await expect(page.getByRole("heading", { name: "Your money" })).toBeVisible();
  await expectFitsScreen(page);
});
