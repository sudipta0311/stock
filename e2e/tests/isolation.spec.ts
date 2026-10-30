import {
  completeFixedDeposit,
  expect,
  gotoApp,
  guardLocalNetwork,
  makeIdentity,
  openFixedDepositForm,
  signIn,
  startReadyAccount,
  test,
} from "../support/app";

test("another account sees no assets and cannot open a saved item", async ({
  page,
  browser,
}) => {
  await startReadyAccount(page, "isolation-owner");
  await openFixedDepositForm(page);
  await completeFixedDeposit(page);
  // In the setup flow the next step finishes setup; the saved item opens from Money.
  await page.getByRole("button", { name: "Finish Money Map setup" }).click();
  await page.getByRole("link", { name: /Synthetic Bank/ }).first().click();
  await expect(page).toHaveURL(/\/money\/asset\//);
  const ownerAssetPath = new URL(page.url()).pathname;

  const secondContext = await browser.newContext({
    viewport: page.viewportSize() ?? { width: 412, height: 915 },
    isMobile: true,
    hasTouch: true,
    serviceWorkers: "block",
  });
  const verifyNetwork = await guardLocalNetwork(secondContext);
  try {
    const secondPage = await secondContext.newPage();
    await signIn(secondPage, makeIdentity("isolation-second"));
    await expect(
      secondPage.getByRole("heading", {
        name: "Your money, in one clear place",
      }),
    ).toBeVisible();
    await secondPage
      .getByRole("button", { name: "Set up my Money Map" })
      .click();
    await secondPage.getByRole("button", { name: /^Just me/ }).click();
    await secondPage
      .getByRole("button", { name: "Continue", exact: true })
      .click();
    await secondPage.getByRole("button", { name: "View your plan" }).click();
    await gotoApp(secondPage, "/money");
    await expect(
      secondPage.getByText("0 items", { exact: true }),
    ).toBeVisible();
    await expect(
      secondPage.getByText("Synthetic Bank", { exact: true }),
    ).toHaveCount(0);

    const assetResponse = secondPage.waitForResponse(
      (response) =>
        response.request().method() === "GET" &&
        response.url().includes("/v1/assets/"),
    );
    await gotoApp(secondPage, ownerAssetPath);
    expect((await assetResponse).status()).toBe(404);
  } finally {
    try {
      await verifyNetwork();
    } finally {
      await secondContext.close();
    }
  }
});
