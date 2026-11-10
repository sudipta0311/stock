import {
  expectFitsScreen,
  gotoApp,
  localApiUrl,
  makeEmailUnsubscribeToken,
  mintDevToken,
  startReadyAccount,
} from "../support/app";
import { expect, test } from "../support/save-test";

test("signed email unsubscribe links turn consent off for GET and POST", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  const identity = await startReadyAccount(page, "save-email-unsubscribe");
  await gotoApp(page, "/settings?section=notifications");
  const weekly = page.getByLabel("Weekly email");
  await weekly.check();
  await expect(weekly).toBeChecked();
  await expectFitsScreen(page);

  const meResponse = await page.request.get(localApiUrl("/v1/me"), {
    headers: { Authorization: `Bearer ${mintDevToken(identity)}` },
  });
  expect(meResponse.ok()).toBeTruthy();
  const me = (await meResponse.json()) as { id: string };
  const token = makeEmailUnsubscribeToken(me.id, "weekly_email");
  const url = localApiUrl(
    `/v1/email/unsubscribe?token=${encodeURIComponent(token)}`,
  );
  await page.goto(url);
  await expect(
    page.getByRole("heading", { name: "Alerts updated" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await gotoApp(page, "/settings?section=notifications");
  await expect(page.getByLabel("Weekly email")).not.toBeChecked();
  await expectFitsScreen(page);

  await page.getByLabel("Weekly email").check();
  await expect(page.getByLabel("Weekly email")).toBeChecked();
  const post = await page.request.post(url);
  expect(post.ok()).toBeTruthy();
  expect(await post.text()).toContain("Alerts updated");
  await page.reload();
  await expect(page.getByLabel("Weekly email")).not.toBeChecked();
  await expectFitsScreen(page);
});
