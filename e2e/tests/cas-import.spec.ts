import {
  expect,
  startReadyAccount,
  test,
  uploadSyntheticCas,
  writeReport,
} from "../support/app";

test("CAS password feedback, review, and confirmation update Money", async ({
  page,
}, testInfo) => {
  await startReadyAccount(page, "cas-import");
  await page.goto(new URL("/import", process.env.E2E_BASE_URL).toString());
  await page
    .getByLabel("Statement file")
    .setInputFiles(process.env.E2E_CAS_PATH ?? "");
  await page.getByLabel(/^Statement password/).fill("wrong synthetic password");
  await page.getByRole("button", { name: "Upload statement" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "That password didn't open the file.",
    {
      timeout: 120_000,
    },
  );

  await page
    .getByLabel("Statement file")
    .setInputFiles(process.env.E2E_CAS_PATH ?? "");
  const startedAt = Date.now();
  await uploadSyntheticCas(page);
  await expect(page.getByText(/^New \d+$/)).toBeVisible();
  await page.getByRole("button", { name: /Confirm import \(/ }).click();
  await expect(
    page.getByRole("heading", {
      name: /Added \d+ holdings? · Net worth updated/,
    }),
  ).toBeVisible();
  const elapsedMs = Date.now() - startedAt;
  expect(elapsedMs).toBeLessThan(120_000);
  await page.getByRole("link", { name: "Go to Money" }).click();
  await expect(page.getByText("Synthetic Tools Limited")).toBeVisible();
  await writeReport("cas-import", testInfo.project.name, {
    elapsedMs,
    targetMs: 120_000,
  });
});
