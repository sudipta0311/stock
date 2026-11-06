import {
  expect,
  expectFitsScreen,
  gotoApp,
  makeIdentity,
  seedResearchPack,
  signIn,
  test,
} from "../support/app";

const longCompanyName = "Synthetic Long Company Name for Phone Layouts";

test("Ideas keeps long names and the world fund rows on screen", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 740 });
  const identity = makeIdentity("ideas-layout");
  await signIn(page, identity);
  await page.getByRole("button", { name: "Get started" }).click();
  await page.getByRole("button", { name: /^Me and my spouse/ }).click();
  const residences = page.getByLabel("Tax residence");
  await expect(residences).toHaveCount(2);
  await residences.nth(0).selectOption("NRI");
  await residences.nth(1).selectOption("RI");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "You're ready to start" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Explore the app" }).click();
  await expect(page).toHaveURL(/\/$/);

  await seedResearchPack(identity.email);
  await gotoApp(page, "/ideas");
  const longName = page.getByRole("heading", {
    name: longCompanyName,
    exact: true,
  });
  await expect(longName).toBeVisible();
  const marketHeading = page.getByRole("heading", { name: "India", exact: true });
  await expect(marketHeading).toHaveClass(/text-xs/);
  await expect(marketHeading).toHaveClass(/text-muted-foreground/);
  const ideaArticle = longName.locator("xpath=ancestor::article");
  const referencePrice = ideaArticle.getByText(/Reference price/);
  await expect(referencePrice).toBeVisible();
  await expect(referencePrice.locator("xpath=preceding-sibling::*[1]")).toContainText(
    "Buy in 3 parts",
  );
  await expect(
    page
      .getByRole("article")
      .filter({ hasText: "Synthetic US Test Company One" })
      .getByText(/You already own about DKK .* through your funds\./),
  ).toBeVisible();
  await expect(longName).toHaveText(longCompanyName);
  await expect(longName).not.toHaveClass(/line-clamp/);
  await expectFitsScreen(page);

  const fundsHeading = page.getByRole("heading", {
    name: "US and world funds",
    exact: true,
  });
  await expect(fundsHeading).toBeVisible();
  await expect(fundsHeading.locator("..").getByRole("listitem")).toHaveCount(1);
  await expect(page.getByRole("heading", { name: "World fund", exact: true })).toBeVisible();
  const otherFunds = page.getByText("Other funds to compare", { exact: true });
  await expect(otherFunds).toBeVisible();
  await expect(otherFunds.locator("xpath=ancestor::details")).not.toHaveAttribute("open");
  await expect(page.getByRole("heading", { name: "S&P 500 fund", exact: true })).not.toBeVisible();
  await expect(page.getByRole("heading", { name: "Nasdaq 100 fund", exact: true })).not.toBeVisible();
  await otherFunds.click();
  for (const name of ["S&P 500 fund", "Nasdaq 100 fund"]) {
    await expect(page.getByRole("heading", { name, exact: true })).toBeVisible();
  }
  await expectFitsScreen(page);
  const alreadyOwned = page.getByText("You already own enough (1)");
  await expect(alreadyOwned).toBeVisible();
  expect(
    await alreadyOwned.evaluate(
      (element) =>
        (element.closest("details") as HTMLDetailsElement | null)?.open,
    ),
  ).toBe(false);
  await expect(
    page.getByRole("heading", { name: "Synthetic US Test Company Two" }),
  ).not.toBeVisible();
  await expectFitsScreen(page);
});
