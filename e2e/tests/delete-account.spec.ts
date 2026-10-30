import {
  expect,
  gotoApp,
  startReadyAccount,
  test,
  verifyAccountDeleted,
} from "../support/app";

test("account deletion returns to sign-in and removes the user row", async ({
  page,
}) => {
  const identity = await startReadyAccount(page, "delete-account");
  await gotoApp(page, "/settings");
  // Development tokens create a user on any request, so let Settings finish loading
  // first; app-session tokens can't recreate a deleted user (server test covers it).
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: "Delete my account" }).click();
  await page.getByLabel("Type DELETE to confirm.").fill("DELETE");
  await page.getByRole("button", { name: "Delete account" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByLabel("Development token")).toBeVisible();
  await verifyAccountDeleted(identity.email);
});
