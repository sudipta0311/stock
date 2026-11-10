import {
  expect,
  gotoApp,
  expectFitsScreen,
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
  await page.setViewportSize({ width: 360, height: 740 });
  await signIn(page, identity);

  // The dev-token submit authenticates this test account. Count the setup taps after it.
  let taps = 0;
  await page.getByRole("button", { name: "Get started" }).click();
  taps += 1;
  await expect(
    page.getByRole("heading", { name: "Who are we planning for?" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Another adult/ }),
  ).toBeVisible();
  await expect(
    page.getByText("Your tax residence helps tailor the plan."),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: /^Me and my spouse/ }).click();
  taps += 1;
  await expect(
    page.getByText("Your spouse", { selector: "legend" }),
  ).toBeVisible();
  await expect(page.getByLabel("Relationship")).toHaveCount(0);
  await expect(
    page.getByText("Your tax residence helps tailor the plan."),
  ).toHaveCount(1);

  const residence = page.getByLabel("Tax residence");
  await expect(residence).toHaveCount(2);
  await residence.nth(0).selectOption("NRI");
  taps += 1;
  await expect(residence.nth(1)).toHaveValue("RI");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  taps += 1;
  await expect(
    page.getByRole("heading", { name: "You're ready to start" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Explore the app" }).click();
  taps += 1;
  await expect(page).toHaveURL(/\/$/);
  expect(taps).toBeLessThanOrEqual(5);

  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await page.getByText("Make it more accurate", { exact: true }).click();
  await page.getByRole("link", { name: "Add a goal" }).click();
  await expect(page.getByRole("heading", { name: "Add a goal" })).toBeVisible();
  await expect(page.getByRole("banner").getByText("Add a goal")).toBeVisible();
  await expect(
    page.getByText("Pick a goal, then add a rough target and year."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Education abroad" }).click();
  await page.getByLabel("Currency").selectOption("USD");
  await page.getByLabel("Target amount (USD)").fill("50000");
  await page.getByLabel("Target year").fill("2034");
  await page.getByRole("button", { name: "Save goal" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByText("Goal saved", { exact: true })).toBeVisible();
  await page.reload();
  await gotoApp(page, "/plan");
  const savedGoal = page.getByText("Education abroad", { exact: true }).last();
  await savedGoal.locator("xpath=ancestor::details[1]/summary").click();
  await expect(savedGoal).toBeVisible();
  await expectFitsScreen(page);
  await gotoApp(page, "/");
  await page.getByText("Make it more accurate", { exact: true }).click();
  await page
    .getByRole("link", { name: "Choose your comfort with ups and downs" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Your comfort with risk" }),
  ).toBeVisible();
  await expect(
    page.getByRole("progressbar", { name: "Onboarding progress" }),
  ).toHaveCount(0);
  await expect(
    page.getByText("Choose the answer that feels closest."),
  ).toBeVisible();
  const firstNext = page.getByRole("button", { name: "Next" });
  const firstSkip = page.getByRole("button", { name: "Skip for now" });
  const firstNextBox = await firstNext.boundingBox();
  const firstSkipBox = await firstSkip.boundingBox();
  expect(firstNextBox).not.toBeNull();
  expect(firstSkipBox).not.toBeNull();
  expect(firstNextBox!.y).toBeLessThan(firstSkipBox!.y);
  expect(firstNextBox!.x).toBeGreaterThan(100);
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Skip for now" }).click();
  await expect(page).toHaveURL(/\/$/);
  await page.getByText("Make it more accurate", { exact: true }).click();
  await page
    .getByRole("link", { name: "Choose your comfort with ups and downs" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Your comfort with risk" }),
  ).toBeVisible();
  const question = page.getByRole("group");
  await expect(question).toBeVisible();
  await expectFitsScreen(page);
  for (let index = 1; index <= 6; index += 1) {
    await expect(page.getByText(`Question ${index} of 6`)).toBeVisible();
    if (index > 1) {
      await expect(question.locator("legend")).toBeFocused();
      const backBox = await page
        .getByRole("button", { name: "Back" })
        .boundingBox();
      const nextBox = await page
        .getByRole("button", {
          name: index === 6 ? "Save answers" : "Next",
        })
        .boundingBox();
      expect(backBox).not.toBeNull();
      expect(nextBox).not.toBeNull();
      expect(Math.abs(backBox!.y - nextBox!.y)).toBeLessThan(2);
      expect(backBox!.x).toBeLessThan(nextBox!.x);
    }
    await question.getByRole("button").first().click();
    if (index < 6) {
      await page.getByRole("button", { name: "Next" }).click();
    } else {
      await page.getByRole("button", { name: "Save answers" }).click();
    }
  }
  await expect(page.getByText(/^Your plan:/)).toBeVisible();
  await gotoApp(page, "/settings");
  await page.getByText("Your plan", { exact: true }).click();
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

test("setup risk skip continues to consent without saving", async ({
  page,
}) => {
  const identity = makeIdentity("risk-skip");
  await page.setViewportSize({ width: 360, height: 740 });
  await signIn(page, identity);
  await gotoApp(page, "/onboarding/risk");

  await expect(page.getByText("Question 1 of 6")).toBeVisible();
  await expect(
    page.getByRole("progressbar", { name: "Onboarding progress" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Skip for now" }).click();

  await expect(page).toHaveURL(/\/onboarding\/consent$/);
  await expect(
    page.getByRole("heading", { name: "How should we keep you updated?" }),
  ).toBeVisible();
});
