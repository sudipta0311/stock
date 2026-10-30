import {
  addDanishSavings,
  completeFixedDeposit,
  completeMinimalOnboarding,
  expect,
  gotoApp,
  makeIdentity,
  openFixedDepositForm,
  signIn,
  skipGuidedTour,
  test,
  uploadSyntheticCas,
  writeReport,
} from "../support/app";

test("five narrow-screen tasks meet their timing and visibility targets", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "Usability timing is measured at 360 by 740.",
  );
  const tasks: Array<{ task: string; elapsedMs: number; targetMs: number }> =
    [];

  await signIn(page, makeIdentity("usability-fd"));
  await completeMinimalOnboarding(page);
  let startedAt = Date.now();
  await openFixedDepositForm(page);
  await completeFixedDeposit(page);
  tasks.push({
    task: "Add fixed deposit",
    elapsedMs: Date.now() - startedAt,
    targetMs: 30_000,
  });

  await signIn(page, makeIdentity("usability-cas"));
  await completeMinimalOnboarding(page);
  await gotoApp(page, "/import");
  startedAt = Date.now();
  await uploadSyntheticCas(page);
  tasks.push({
    task: "Reach statement review",
    elapsedMs: Date.now() - startedAt,
    targetMs: 120_000,
  });

  await signIn(page, makeIdentity("usability-home"));
  await completeMinimalOnboarding(page);
  startedAt = Date.now();
  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await expect(
    page.getByRole("heading", { name: "This week" }),
  ).toBeInViewport();
  tasks.push({
    task: "Find this week",
    elapsedMs: Date.now() - startedAt,
    targetMs: 10_000,
  });

  await signIn(page, makeIdentity("usability-plan"));
  await completeMinimalOnboarding(page);
  await addDanishSavings(page);
  startedAt = Date.now();
  await gotoApp(page, "/plan");
  const safeMoney = page.getByText(/Your safe money target is/);
  await expect(safeMoney).toBeInViewport();
  tasks.push({
    task: "Find safe money target",
    elapsedMs: Date.now() - startedAt,
    targetMs: 15_000,
  });

  await signIn(page, makeIdentity("usability-household"));
  await expect(
    page.getByRole("heading", { name: "Your money, in one clear place" }),
  ).toBeVisible();
  startedAt = Date.now();
  await page.getByRole("button", { name: "Set up my Money Map" }).click();
  await page.getByRole("button", { name: /^Me and my spouse/ }).click();
  const residences = page.getByLabel("Tax residence");
  await residences.nth(0).selectOption("NRI");
  await residences.nth(1).selectOption("RI");
  await expect(residences.nth(1)).toHaveValue("RI");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Your plan is ready" }),
  ).toBeVisible();
  tasks.push({
    task: "Add spouse and set tax residence",
    elapsedMs: Date.now() - startedAt,
    targetMs: 60_000,
  });

  await writeReport("usability", "all", {
    viewport: { width: 360, height: 740 },
    tasks,
    observations: [
      "Tasks used current button and field labels without help.",
      "This week and the safe money answer were checked in the first viewport.",
    ],
  });
  for (const task of tasks)
    expect(task.elapsedMs, task.task).toBeLessThan(task.targetMs);
});
