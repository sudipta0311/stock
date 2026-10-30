import AxeBuilder from "@axe-core/playwright";
import {
  completeMinimalOnboarding,
  expect,
  gotoApp,
  makeIdentity,
  signIn,
  test,
  uploadSyntheticCas,
  writeReport,
} from "../support/app";

test("required screens have no serious or critical accessibility violations", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "Axe runs once at the narrow phone viewport.",
  );
  const reports: Array<{ screen: string; result: unknown }> = [];
  const failures: Array<{
    screen: string;
    id: string;
    impact: string | null;
    help: string;
  }> = [];
  const scan = async (screen: string) => {
    const result = await new AxeBuilder({ page }).analyze();
    reports.push({ screen, result });
    failures.push(
      ...result.violations
        .filter(
          (violation) =>
            violation.impact === "serious" || violation.impact === "critical",
        )
        .map((violation) => ({
          screen,
          id: violation.id,
          impact: violation.impact,
          help: violation.help,
        })),
    );
  };

  await gotoApp(page, "/login");
  await scan("Login");
  await signIn(page, makeIdentity("axe"));
  await scan("Onboarding welcome");
  await page.getByRole("button", { name: "Set up my Money Map" }).click();
  await scan("Onboarding household");
  await page.getByRole("button", { name: /^Just me/ }).click();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Your plan is ready" }),
  ).toBeVisible();
  await scan("Onboarding done");
  await page.getByRole("button", { name: "View your plan" }).click();
  await expect(page).toHaveURL(/\/plan$/);

  await gotoApp(page, "/onboarding/goals");
  await expect(
    page.getByRole("heading", { name: "What are you saving for?" }),
  ).toBeVisible();
  await scan("Onboarding goals");
  await gotoApp(page, "/onboarding/risk");
  await expect(
    page.getByRole("heading", { name: "How do you feel about risk?" }),
  ).toBeVisible();
  await scan("Onboarding risk");
  await gotoApp(page, "/onboarding/consent");
  await expect(
    page.getByRole("heading", { name: "How should we keep you updated?" }),
  ).toBeVisible();
  await scan("Onboarding updates");

  await gotoApp(page, "/money");
  await scan("Money");
  await page
    .getByRole("button", { name: "Fixed deposits", exact: true })
    .click();
  await page.getByRole("button", { name: /Continue \(1 selected\)/ }).click();
  await scan("Fixed deposit form");

  await gotoApp(page, "/import");
  await scan("Statement upload");
  await uploadSyntheticCas(page);
  await scan("Statement review");

  await gotoApp(page, "/plan");
  await scan("Plan");
  await gotoApp(page, "/");
  const tourSkip = page
    .getByRole("dialog", { name: "Home" })
    .getByRole("button", { name: "Skip" });
  if (await tourSkip.isVisible().catch(() => false)) await tourSkip.click();
  await scan("Home");
  await gotoApp(page, "/ideas");
  await scan("Ideas");
  await gotoApp(page, "/inbox");
  await scan("Inbox");
  await gotoApp(page, "/settings");
  await scan("Settings");

  await writeReport("axe", "all", {
    scans: reports,
    severeOrCritical: failures,
  });
  expect(failures).toEqual([]);
});
