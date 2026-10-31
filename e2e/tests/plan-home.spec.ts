import {
  addDanishSavings,
  expect,
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

  await seedHomeAction(identity.email);
  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await expect(page.getByRole("heading", { name: "This week" })).toBeVisible();
  await expect(page.getByRole("article", { name: /For you/ })).toBeVisible();
  await page.getByRole("button", { name: "Mark done" }).click();
  await expect(
    page.getByText("How much can you invest each month?", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Monthly amount").fill("25000");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByText("Nothing to buy this week", { exact: true }),
  ).toBeVisible();
});
