import {
  addDanishSavings,
  expectFitsScreen,
  gotoApp,
  startReadyAccount,
} from "../support/app";
import { expect, test } from "../support/save-test";

test("monthly plan settings and email consent persist; push stays off without permission", async ({
  page,
  context,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-settings");
  await addDanishSavings(page);
  await expectFitsScreen(page);

  await gotoApp(page, "/settings");
  await page.getByText("Your plan", { exact: true }).click();
  await page.getByText("Fine-tune your plan", { exact: true }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Monthly investing amount").fill("65000");
  await page.getByLabel("I have money to invest this week").fill("15000");
  // An earlier "saved" status can still be on screen, so wait for this save itself.
  const savedInvesting = page.waitForResponse(
    (response) =>
      response.request().method() === "PUT" &&
      new URL(response.url()).pathname === "/v1/me/investing",
  );
  await page.getByRole("button", { name: "Save plan details" }).click();
  expect((await savedInvesting).ok()).toBeTruthy();
  await expect(
    page.getByRole("status").filter({ hasText: /saved/i }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.reload();
  await page.getByText("Your plan", { exact: true }).click();
  await page.getByText("Fine-tune your plan", { exact: true }).click();
  await expect(page.getByLabel("Monthly investing amount")).toHaveValue(
    "65000.00",
  );
  await expect(page.getByLabel("I have money to invest this week")).toHaveValue(
    "15000.00",
  );
  await expectFitsScreen(page);
  await context.grantPermissions([], { origin: new URL(page.url()).origin });
  await gotoApp(page, "/settings?section=notifications");
  await expectFitsScreen(page);
  const weekly = page.getByLabel("Weekly email");
  const consentSave = page.waitForResponse(
    (response) =>
      response.request().method() === "PUT" &&
      new URL(response.url()).pathname === "/v1/me/consents",
  );
  await weekly.check();
  await expect(weekly).toBeChecked();
  expect((await consentSave).ok()).toBeTruthy();
  await expect(page.getByLabel("Push alerts on this device")).not.toBeChecked();
  await page.reload();
  await expect(page.getByLabel("Weekly email")).toBeChecked();
  await expect(page.getByLabel("Push alerts on this device")).not.toBeChecked();
  await expectFitsScreen(page);
});

test("sign out on this phone returns to sign in", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-logout-phone");
  await gotoApp(page, "/settings");
  await page.getByText("App and privacy", { exact: true }).click();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  const confirmation = page.getByRole("dialog", { name: "Sign out" });
  await expect(confirmation).toBeVisible();
  await expectFitsScreen(page);
  await confirmation
    .getByRole("button", { name: "Sign out", exact: true })
    .click();
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByLabel("Development token")).toBeVisible();
});

test("sign out everywhere returns to sign in", async ({ page }, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-logout-everywhere");
  await gotoApp(page, "/settings");
  await page.getByText("App and privacy", { exact: true }).click();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Sign out on all devices" }).click();
  await expect(
    page.getByRole("heading", { name: "Sign out on all devices?" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Sign out everywhere" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByLabel("Development token")).toBeVisible();
  await expectFitsScreen(page);
});
