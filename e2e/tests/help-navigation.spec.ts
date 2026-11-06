import {
  expect,
  expectFitsScreen,
  gotoApp,
  startReadyAccount,
  skipGuidedTour,
  test,
} from "../support/app";

test("Home help opens this week's article and returns Home", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 });
  await startReadyAccount(page, "help-navigation");
  await gotoApp(page, "/");
  await skipGuidedTour(page);
  await page.getByRole("link", { name: "Help with Home" }).click();

  await expect(page).toHaveURL(/\/help#this-week$/);
  const heading = page.getByRole("heading", { name: "Follow this week" });
  await expect(heading).toBeVisible();
  await expect(page.locator("#this-week details")).toHaveAttribute("open", "");
  await expect(heading).toBeFocused();
  await expectFitsScreen(page);

  await page.getByLabel("Search help articles").fill("extra payment");
  const searchResult = page
    .getByRole("article")
    .filter({ has: page.getByRole("heading", { name: "Add and update loans" }) });
  await expect(searchResult.locator("summary")).not.toContainText("Read steps");
  await searchResult.getByRole("heading", { name: "Add and update loans" }).click();
  await expect(searchResult.locator("details")).toHaveAttribute("open", "");

  await page.getByRole("link", { name: "Back" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "This week" })).toBeVisible();
  await expectFitsScreen(page);
});
