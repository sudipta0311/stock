import {
  expect,
  gotoApp,
  openFixedDepositForm,
  seedHoldingReviews,
  startReadyAccount,
  test,
} from "../support/app";

test("button, link, field, and disclosure styles stay readable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await startReadyAccount(page, "styles-controls");

  await gotoApp(page, "/money/add/us_investments");
  const primaryLink = page.getByRole("link", { name: "Upload a broker file" });
  const button = page.getByRole("button", { name: "Continue to next item" });
  await expect(primaryLink).toBeVisible();
  await expect(button).toBeVisible();
  const [linkStyle, buttonStyle] = await Promise.all([
    primaryLink.evaluate((element) => {
      const style = getComputedStyle(element);
      return [style.fontSize, style.fontWeight];
    }),
    button.evaluate((element) => {
      const style = getComputedStyle(element);
      return [style.fontSize, style.fontWeight];
    }),
  ]);
  expect(linkStyle).toEqual(["15px", "600"]);
  expect(buttonStyle).toEqual(["15px", "600"]);

  await openFixedDepositForm(page);
  const amountField = page.getByLabel("Amount deposited");
  const [fieldHeight, fieldFontSize] = await amountField.evaluate((element) => [
    element.getBoundingClientRect().height,
    getComputedStyle(element).fontSize,
  ]);
  expect(fieldHeight).toBeGreaterThanOrEqual(48);
  expect(fieldFontSize).toBe("15px");

  await gotoApp(page, "/money");
  const summary = page.locator("#money-checklist > summary");
  await expect(summary).toBeVisible();
  const summaryHeight = await summary.evaluate(
    (element) => getComputedStyle(element).minHeight,
  );
  expect(Number.parseFloat(summaryHeight)).toBeGreaterThanOrEqual(44);
});

test("holding card and disclosure headings follow their size classes", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  const identity = await startReadyAccount(page, "styles-holding");
  await seedHoldingReviews(identity.email);
  await gotoApp(page, "/money");
  await page.getByText("Synthetic share to add", { exact: true }).click();

  const cardTitle = page.getByRole("heading", {
    name: "Our view",
    exact: true,
  });
  await expect(cardTitle).toBeVisible();
  await page.getByText("Why this suggestion?", { exact: true }).click();
  const sectionHeading = page.getByRole("heading", {
    name: "How we check",
    level: 3,
  });
  await expect(sectionHeading).toBeVisible();

  const [titleSize, sectionSize] = await Promise.all([
    cardTitle.evaluate((element) => getComputedStyle(element).fontSize),
    sectionHeading.evaluate((element) => getComputedStyle(element).fontSize),
  ]);
  expect(titleSize).toBe("17px");
  expect(sectionSize).toBe("13px");
});

test("focused form controls stay above the fixed bottom navigation", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await startReadyAccount(page, "styles-focus-visibility");
  await gotoApp(page, "/import");
  await page
    .getByRole("button", { name: "Broker file (Excel or CSV)" })
    .click();
  await page.getByLabel("Where did you get this file?").selectOption("nordnet");

  const accountSelect = page.getByLabel("Account (optional)");
  await expect(accountSelect).toBeVisible();
  await page.getByLabel("Statement file").setInputFiles({
    name: "synthetic-focus.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("Name,Quantity\nSynthetic share,1\n"),
  });

  const uploadButton = page.getByRole("button", {
    name: "Upload file",
    exact: true,
  });
  const bottomNavigation = page.locator('nav[aria-label="Primary"]');
  await expect(uploadButton).toBeEnabled();
  await uploadButton.focus();
  await expect(uploadButton).toBeFocused();

  const [controlBottom, navigationTop] = await Promise.all([
    uploadButton.evaluate((element) => element.getBoundingClientRect().bottom),
    bottomNavigation.evaluate((element) => element.getBoundingClientRect().top),
  ]);
  expect(controlBottom).toBeLessThanOrEqual(navigationTop);
});
