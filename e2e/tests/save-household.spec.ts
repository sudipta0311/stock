import {
  completeMinimalOnboarding,
  expectFitsScreen,
  gotoApp,
  guardLocalNetwork,
  makeIdentity,
  signIn,
  skipGuidedTour,
  startReadyAccount,
} from "../support/app";
import { expect, test, watchBrowserErrors } from "../support/save-test";
import type { Page, TestInfo } from "@playwright/test";

async function createInvite(
  page: Page,
  email: string,
  role: "partner" | "viewer",
): Promise<string> {
  await gotoApp(page, "/settings");
  await page.getByText("You and your family", { exact: true }).click();
  await page.getByLabel("Their email").fill(email);
  await page.getByLabel("Access", { exact: true }).first().selectOption(role);
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Create share link" }).click();
  return page.getByLabel("Share link").inputValue();
}

async function acceptInvite(
  page: Page,
  link: string,
  identity: ReturnType<typeof makeIdentity>,
): Promise<void> {
  await signIn(page, identity);
  await completeMinimalOnboarding(page);
  await followInvite(page, link);
}

async function followInvite(page: Page, link: string): Promise<void> {
  const token = new URL(link).searchParams.get("token");
  if (!token) throw new Error("The synthetic invite has no token.");
  await gotoApp(page, `/join?token=${encodeURIComponent(token)}`);
  await expect(
    page.getByRole("heading", { name: "Join this family plan" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Join family plan" }).click();
  await expect(page).toHaveURL(/\/$/);
  await skipGuidedTour(page);
}

async function expectJustLeftSettings(
  page: Page,
  testInfo: TestInfo,
): Promise<void> {
  try {
    await page.getByText("You and your family", { exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Household", level: 2 }),
    ).toBeVisible();
    await expect(
      page.getByText("App and privacy", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByText("1 person · Access: only you", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Leave family plan" }),
    ).toHaveCount(0);
    await expectFitsScreen(page);
  } catch (error) {
    await testInfo.attach("joining-page-after-leaving-settings", {
      body: await page.locator("body").ariaSnapshot(),
      contentType: "text/plain",
    });
    throw error;
  }
}

test("invites revoke, members join and leave, and owners remove access", async ({
  page,
  browser,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-household-owner");
  const revokedIdentity = makeIdentity("save-invite-revoked");
  const revokedLink = await createInvite(page, revokedIdentity.email, "viewer");
  expect(new URL(await revokedLink).searchParams.get("token")).toBeTruthy();
  await page.reload();
  await page.getByText("You and your family", { exact: true }).click();
  await expect(
    page.getByText(revokedIdentity.email, { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Revoke invite" }).click();
  await expect(
    page.getByText(revokedIdentity.email, { exact: false }),
  ).toHaveCount(0);
  await page.reload();
  await page.getByText("You and your family", { exact: true }).click();
  await expect(
    page.getByText(revokedIdentity.email, { exact: false }),
  ).toHaveCount(0);
  await expectFitsScreen(page);

  const viewport = page.viewportSize() ?? { width: 360, height: 740 };
  const joiningContext = await browser.newContext({
    viewport,
    isMobile: true,
    hasTouch: true,
    serviceWorkers: "block",
  });
  let joinNetworkCheck: (() => Promise<void>) | undefined;
  let joinErrorCheck: ReturnType<typeof watchBrowserErrors> | undefined;
  try {
    joinNetworkCheck = await guardLocalNetwork(joiningContext);
    const joiningPage = await joiningContext.newPage();
    joinErrorCheck = watchBrowserErrors(joiningPage);
    const joiningIdentity = makeIdentity("save-partner-joins");
    const partnerLink = await createInvite(
      page,
      joiningIdentity.email,
      "partner",
    );
    await acceptInvite(joiningPage, partnerLink, joiningIdentity);
    await gotoApp(joiningPage, "/settings");
    await joiningPage.getByText("You and your family", { exact: true }).click();
    await expect(
      joiningPage.getByRole("button", { name: "Leave family plan" }),
    ).toBeVisible();
    await expectFitsScreen(joiningPage);
    await joiningPage
      .getByRole("button", { name: "Leave family plan" })
      .click();
    await expect(
      joiningPage
        .getByRole("status")
        .filter({ hasText: "left the family plan" }),
    ).toBeVisible();
    await joiningPage.reload();
    await expect(
      joiningPage.getByRole("heading", {
        name: "Who are we planning for?",
      }),
    ).toBeVisible();
    await completeMinimalOnboarding(joiningPage);
    await skipGuidedTour(joiningPage);
    await gotoApp(joiningPage, "/settings");
    await expectJustLeftSettings(joiningPage, testInfo);

    const viewerLink = await createInvite(
      page,
      joiningIdentity.email,
      "viewer",
    );
    await followInvite(joiningPage, viewerLink);
    await gotoApp(page, "/settings");
    await page.getByText("You and your family", { exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Remove access" }),
    ).toBeVisible();
    await expectFitsScreen(page);
    const accessRoles = page.getByLabel("Access role", { exact: true });
    await accessRoles.last().selectOption("partner");
    await expect(accessRoles.last()).toBeEnabled();
    await page.reload();
    await page.getByText("You and your family", { exact: true }).click();
    await expect(
      page.getByLabel("Access role", { exact: true }).last(),
    ).toHaveValue("partner");
    await page
      .getByLabel("Access role", { exact: true })
      .last()
      .selectOption("viewer");
    await expect(
      page.getByLabel("Access role", { exact: true }).last(),
    ).toBeEnabled();
    await page.reload();
    await page.getByText("You and your family", { exact: true }).click();
    await expect(
      page.getByLabel("Access role", { exact: true }).last(),
    ).toHaveValue("viewer");
    await page.getByRole("button", { name: "Remove access" }).click();
    await expect(
      page.getByRole("button", { name: "Remove access" }),
    ).toHaveCount(0);
    await page.reload();
    await page.getByText("You and your family", { exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Remove access" }),
    ).toHaveCount(0);
    await expectFitsScreen(page);
  } finally {
    try {
      if (joinNetworkCheck) await joinNetworkCheck();
    } finally {
      try {
        if (joinErrorCheck) joinErrorCheck.assertNoUnexpectedErrors();
      } finally {
        await joiningContext.close();
      }
    }
  }
});
