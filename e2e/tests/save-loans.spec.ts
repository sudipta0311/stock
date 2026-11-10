import { expectFitsScreen, gotoApp, startReadyAccount } from "../support/app";
import { expect, test } from "../support/save-test";

test("loan payments, rate changes, preview, and event deletion persist", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-loan-events");
  await gotoApp(page, "/money");
  await expectFitsScreen(page);
  await page.getByText("Add manually", { exact: true }).click();
  await page.getByRole("button", { name: "Loans", exact: true }).click();
  await page.getByRole("button", { name: /Continue \(1 selected\)/ }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Loan type").selectOption("home");
  await page.getByLabel("Outstanding balance").fill("1200000");
  await page.getByLabel("Interest rate (%)").fill("5.2");
  await page.getByLabel("Monthly payment").fill("18000");
  await page.getByLabel("Currency").selectOption("INR");
  await page.getByText("More details (for a closer estimate)").click();
  await page.getByLabel("Name this item").fill("Synthetic home loan");
  await page.getByRole("button", { name: "Save and return" }).click();
  await expect(page).toHaveURL(/\/money$/);
  await page.reload();
  const loan = page.getByRole("link", { name: /Synthetic home loan/ });
  await expect(loan).toBeVisible();
  await loan.click();
  await expect(
    page.getByRole("heading", { name: "Synthetic home loan" }),
  ).toBeVisible();
  await expectFitsScreen(page);

  await page.getByRole("button", { name: "Record a payment" }).click();
  await page.getByLabel("What happened?").selectOption("prepayment");
  await page.getByLabel("Date").fill("2026-11-01");
  await page.getByLabel("Amount (INR)").fill("25000");
  await page.getByRole("button", { name: "Save update" }).click();
  await expect(page.getByText("Extra payment", { exact: true })).toBeVisible();
  await expectFitsScreen(page);

  await page.getByRole("button", { name: "Record a payment" }).click();
  await page.getByLabel("What happened?").selectOption("rate_change");
  await page.getByLabel("Date").fill("2026-11-02");
  await page.getByLabel("New interest rate (%)").fill("4.8");
  await page.getByRole("button", { name: "Save update" }).click();
  await expect(page.getByText(/4\.8%/)).toBeVisible();

  await page.getByRole("button", { name: "What if I pay extra?" }).click();
  await page.getByLabel("Extra amount (INR)").fill("50000");
  await page.getByRole("button", { name: "Compare options" }).click();
  await expect(page.getByText(/Finish \d+ months sooner/)).toBeVisible();
  await expectFitsScreen(page);

  await page.reload();
  await expect(page.getByText(/4\.8%/)).toBeVisible();
  await page.getByRole("button", { name: "Delete Extra payment" }).click();
  await expect(page.getByText("Extra payment", { exact: true })).toHaveCount(0);
  await page.reload();
  await expect(page.getByText(/4\.8%/)).toBeVisible();
  await expect(page.getByText("Extra payment", { exact: true })).toHaveCount(0);
  await expectFitsScreen(page);
});
