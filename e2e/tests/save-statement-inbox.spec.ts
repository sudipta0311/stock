import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import type { Page } from "@playwright/test";
import {
  expectFitsScreen,
  gotoApp,
  localApiUrl,
  mintDevToken,
  startReadyAccount,
} from "../support/app";
import { expect, test } from "../support/save-test";

async function uploadSyntheticStatement(
  page: Page,
  key: string,
  subject: string,
): Promise<void> {
  const fixturePath = process.env.E2E_CAS_PATH;
  if (!fixturePath)
    throw new Error("The synthetic statement fixture is not configured.");
  const buffer = await readFile(fixturePath);
  const response = await page.request.post(
    localApiUrl("/v1/statement-inbox/uploads"),
    {
      headers: { Authorization: `StatementKey ${key}` },
      multipart: {
        file: {
          name: "synthetic-cas.pdf",
          mimeType: "application/pdf",
          buffer,
        },
        sender: "synthetic@cams.example.test",
        subject,
        received_at: new Date().toISOString(),
        message_id: `synthetic-${randomUUID()}`,
      },
    },
  );
  expect(response.status()).toBe(202);
}

test("Gmail key, saved password, inbox processing, and removal save", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  const identity = await startReadyAccount(page, "save-statement-inbox");
  await gotoApp(page, "/settings/gmail");
  await expectFitsScreen(page);
  await page.getByLabel("Name this connection").fill("Synthetic forwarder");
  const keyResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/v1/statement-inbox/keys",
  );
  await page.getByRole("button", { name: "Create an upload key" }).click();
  const keyPayload = (await (await keyResponse).json()) as {
    id: string;
    key: string;
  };
  await expect(
    page.getByRole("heading", { name: "Your new key is ready" }),
  ).toBeVisible();
  await expectFitsScreen(page);

  await uploadSyntheticStatement(
    page,
    keyPayload.key,
    "Consolidated Account Statement November 2026",
  );
  await gotoApp(page, "/inbox");
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Enter password" }).click();
  await expectFitsScreen(page);
  const password = process.env.E2E_CAS_PASSWORD;
  if (!password)
    throw new Error("The synthetic statement password is not configured.");
  await page.getByLabel("Statement password").fill(password);
  await page.getByLabel(/Remember this password for CAMS statements/).check();
  await page.getByRole("button", { name: "Review statement" }).click();
  await expect(page.getByRole("link", { name: "Review" })).toBeVisible({
    timeout: 30_000,
  });
  await expectFitsScreen(page);
  await page.reload();
  await expect(page.getByRole("link", { name: "Review" })).toBeVisible();

  const processedItems = await page.request.get(
    localApiUrl("/v1/statement-inbox/items"),
    {
      headers: { Authorization: `Bearer ${mintDevToken(identity)}` },
    },
  );
  expect(processedItems.ok()).toBeTruthy();
  const processedInbox = (await processedItems.json()) as {
    items: Array<{ saved_password: boolean }>;
  };
  expect(processedInbox.items.some((item) => item.saved_password)).toBe(true);

  await gotoApp(page, "/settings/gmail");
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Forget saved passwords" }).click();
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: "Saved statement passwords were removed." }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.reload();
  const savedPasswordState = await page.request.get(
    localApiUrl("/v1/statement-inbox/items"),
    {
      headers: { Authorization: `Bearer ${mintDevToken(identity)}` },
    },
  );
  expect(savedPasswordState.ok()).toBeTruthy();
  const inbox = (await savedPasswordState.json()) as {
    items: Array<{ saved_password: boolean }>;
  };
  expect(inbox.items.every((item) => !item.saved_password)).toBe(true);

  await uploadSyntheticStatement(
    page,
    keyPayload.key,
    "Synthetic purchase note",
  );
  await gotoApp(page, "/inbox");
  await page.getByText("Other emails (1)", { exact: true }).click();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Remove", exact: true }).click();
  await page.reload();
  await expect(page.getByText("Other emails (1)", { exact: true })).toHaveCount(
    0,
  );
  await expectFitsScreen(page);

  await gotoApp(page, "/settings/gmail");
  await expectFitsScreen(page);
  await page
    .getByRole("button", { name: "Revoke Synthetic forwarder" })
    .click();
  await expect(
    page.getByRole("status").filter({ hasText: "upload key was revoked" }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Revoke Synthetic forwarder" }),
  ).toHaveCount(0);
  await expectFitsScreen(page);
});
