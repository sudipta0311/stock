import {
  completeFixedDeposit,
  expect,
  gotoApp,
  makeIdentity,
  openFixedDepositForm,
  completeMinimalOnboarding,
  signIn,
  guardLocalNetwork,
  skipGuidedTour,
  startReadyAccount,
  test,
} from "../support/app";
import type { BrowserContext, Page } from "@playwright/test";

async function createInvite(
  page: Page,
  email: string,
  role: "partner" | "viewer",
): Promise<string> {
  await gotoApp(page, "/settings");
  await page.getByLabel("Their email").fill(email);
  // A member who already joined has an "Access" control too; the invite form's is first.
  await page
    .getByRole("combobox", { name: "Access" })
    .first()
    .selectOption(role);
  await page.getByRole("button", { name: "Create share link" }).click();
  const shareLink = await page.getByLabel("Share link").inputValue();
  expect(new URL(shareLink).searchParams.get("token")).toBeTruthy();
  return shareLink;
}

async function acceptInvite(
  page: Page,
  shareLink: string,
  identity: ReturnType<typeof makeIdentity>,
): Promise<void> {
  await signIn(page, identity);
  await completeMinimalOnboarding(page);
  const token = new URL(shareLink).searchParams.get("token");
  if (!token) throw new Error("The synthetic invite has no token.");
  await gotoApp(page, "/join?token=" + encodeURIComponent(token));
  await expect(
    page.getByRole("heading", { name: "Join this family plan" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Join family plan" }).click();
  await expect(page).toHaveURL(/\/$/);
  await skipGuidedTour(page);
}

test("partner and viewer access share the same family values", async ({
  page,
  browser,
}) => {
  await startReadyAccount(page, "household-owner");
  await openFixedDepositForm(page);
  await completeFixedDeposit(page);

  const viewport = page.viewportSize() ?? { width: 412, height: 915 };
  const partnerContext = await browser.newContext({
    viewport,
    isMobile: true,
    hasTouch: true,
    serviceWorkers: "block",
  });
  const verifyPartnerNetwork = await guardLocalNetwork(partnerContext);
  const partnerPage = await partnerContext.newPage();
  let viewerContext: BrowserContext | undefined;
  let verifyViewerNetwork: (() => Promise<void>) | undefined;
  try {
    const partner = makeIdentity("partner");
    const partnerLink = await createInvite(page, partner.email, "partner");
    await acceptInvite(partnerPage, partnerLink, partner);

    await gotoApp(page, "/");
    await skipGuidedTour(page);
    const ownerNetWorth = await page.getByText(/₹/).first().innerText();
    await expect(
      partnerPage.getByRole("heading", { name: "This week" }),
    ).toBeVisible();
    const partnerNetWorth = await partnerPage
      .getByText(/₹/)
      .first()
      .innerText();
    expect(partnerNetWorth).toBe(ownerNetWorth);

    viewerContext = await browser.newContext({
      viewport,
      isMobile: true,
      hasTouch: true,
      serviceWorkers: "block",
    });
    verifyViewerNetwork = await guardLocalNetwork(viewerContext);
    const viewerPage = await viewerContext.newPage();
    const viewer = makeIdentity("viewer");
    const viewerLink = await createInvite(page, viewer.email, "viewer");
    await acceptInvite(viewerPage, viewerLink, viewer);
    await gotoApp(viewerPage, "/money");
    await expect(
      viewerPage.getByRole("button", { name: "Fixed deposits", exact: true }),
    ).toBeDisabled();
    await expect(viewerPage.getByLabel("Quick add")).toHaveCount(0);
    await gotoApp(viewerPage, "/money/add/fd");
    await expect(
      viewerPage.getByText("View only", { exact: true }),
    ).toBeVisible();
    await expect(viewerPage.getByLabel("Bank or post office")).toBeDisabled();
  } finally {
    try {
      await verifyPartnerNetwork();
    } finally {
      try {
        if (viewerContext && verifyViewerNetwork) await verifyViewerNetwork();
      } finally {
        try {
          if (viewerContext) await viewerContext.close();
        } finally {
          await partnerContext.close();
        }
      }
    }
  }
});
