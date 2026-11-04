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
  const identity = makeIdentity("ideas-layout");
  await signIn(page, identity);
  await page.getByRole("button", { name: "Set up my Money Map" }).click();
  await page.getByRole("button", { name: /^Me and my spouse/ }).click();
  const residences = page.getByLabel("Tax residence");
  await expect(residences).toHaveCount(2);
  await residences.nth(0).selectOption("NRI");
  await residences.nth(1).selectOption("RI");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Your plan is ready" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "View your plan" }).click();
  await expect(page).toHaveURL(/\/plan$/);

  await seedResearchPack(identity.email);
  await gotoApp(page, "/ideas");
  const longName = page.getByRole("heading", {
    name: longCompanyName,
    exact: true,
  });
  await expect(longName).toBeVisible();
  await expect(
    page
      .getByRole("article")
      .filter({ hasText: "Synthetic US Test Company One" })
      .getByText(/You already own about DKK .* through your funds\./),
  ).toBeVisible();
  await expect
    .poll(() =>
      longName.evaluate((element) =>
        getComputedStyle(element).getPropertyValue("-webkit-line-clamp"),
      ),
    )
    .toBe("1");

  const fundsHeading = page.getByRole("heading", {
    name: "US and world funds",
    exact: true,
  });
  await expect(fundsHeading).toBeVisible();
  await expect(fundsHeading.locator("..").getByRole("listitem")).toHaveCount(3);
  for (const name of ["World fund", "S&P 500 fund", "Nasdaq 100 fund"]) {
    await expect(
      page.getByRole("heading", { name, exact: true }),
    ).toBeVisible();
  }
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
  await page.setViewportSize({ width: 384, height: 844 });
  await expectFitsScreen(page);
});
