import {
  expect,
  gotoApp,
  makeIdentity,
  signIn,
  skipGuidedTour,
  test,
  writeReport,
} from "../support/app";

test("short household setup reaches the first plan in five setup taps", async ({
  page,
}) => {
  const identity = makeIdentity("onboarding");
  await signIn(page, identity);

  // The dev-token submit authenticates this test account. Count the setup taps after it.
  let taps = 0;
  await page.getByRole("button", { name: "Set up my Money Map" }).click();
  taps += 1;
  await expect(
    page.getByRole("heading", { name: "Who are we planning for?" }),
  ).toBeVisible();
  await page.getByRole("button", { name: /^Me and my spouse/ }).click();
  taps += 1;

  const residence = page.getByLabel("Tax residence");
  await expect(residence).toHaveCount(2);
  await residence.nth(0).selectOption("NRI");
  taps += 1;
  await expect(residence.nth(1)).toHaveValue("RI");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  taps += 1;
  await expect(
    page.getByRole("heading", { name: "Your plan is ready" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "View your plan" }).click();
  taps += 1;
  await expect(page).toHaveURL(/\/plan$/);
  expect(taps).toBeLessThanOrEqual(5);

  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await page.getByText("Make it more accurate", { exact: true }).click();
  await page.getByRole("link", { name: "Add a goal" }).click();
  await expect(
    page.getByRole("heading", { name: "What are you saving for?" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Education abroad" }).click();
  await page.getByLabel("Currency").selectOption("USD");
  await page.getByLabel("Target amount (USD)").fill("50000");
  await page.getByLabel("Target year").fill("2034");
  await page.getByRole("button", { name: "Add Education abroad" }).click();
  await expect(page.getByText("Education abroad · 2034")).toBeVisible();
  await page.getByRole("link", { name: "Home" }).click();
  await page.getByText("Make it more accurate", { exact: true }).click();
  await page
    .getByRole("link", { name: "Choose your comfort with ups and downs" })
    .click();
  await expect(
    page.getByRole("heading", { name: "How do you feel about risk?" }),
  ).toBeVisible();
  const questions = page.getByRole("group");
  // The questions load after the heading; counting earlier finds none.
  await expect(questions.first()).toBeVisible();
  for (let index = 0; index < (await questions.count()); index += 1) {
    await questions.nth(index).getByRole("button").first().click();
  }
  await page.getByRole("button", { name: "Save answers" }).click();
  await expect(page.getByText(/^Your plan:/)).toBeVisible();
  await gotoApp(page, "/settings");
  await page.getByText("Fine-tune your plan", { exact: true }).click();
  await page.getByLabel("Tax regime").selectOption("new");
  await page.getByRole("button", { name: "Save plan details" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: /saved/i }),
  ).toBeVisible();

  await writeReport("onboarding", "all", {
    setupTapsAfterAuthentication: taps,
    optionalGoal: "Education abroad, USD 50000, 2034",
    riskAnswersSaved: true,
    taxRegime: "new",
  });
});
