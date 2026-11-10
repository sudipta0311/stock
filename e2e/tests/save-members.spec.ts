import {
  expectFitsScreen,
  gotoApp,
  seedResearchPack,
  startReadyAccount,
} from "../support/app";
import type { Page } from "@playwright/test";
import { expect, test } from "../support/save-test";

function householdMemberCard(page: Page, name: string) {
  return page
    .getByRole("region", { name: "Household" })
    .locator("div.rounded-xl.border")
    .filter({ hasText: name });
}

test("member details, asset assignment, filter, and removal save", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  await startReadyAccount(page, "save-household-member");
  await gotoApp(page, "/settings");
  await page.getByText("You and your family", { exact: true }).click();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Add person" }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Name", { exact: true }).fill("Synthetic Partner");
  await page.getByLabel("Relationship").selectOption("spouse");
  await page.getByRole("button", { name: "Add person" }).last().click();
  await expect(
    householdMemberCard(page, "Synthetic Partner").getByText(
      "Synthetic Partner",
      { exact: true },
    ),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.reload();
  await page.getByText("You and your family", { exact: true }).click();
  const memberCard = householdMemberCard(page, "Synthetic Partner");
  await expect(memberCard).toBeVisible();
  await memberCard.getByRole("button", { name: "Edit details" }).click();
  await expectFitsScreen(page);
  await memberCard
    .getByLabel("Name", { exact: true })
    .fill("Synthetic Co-owner");
  await memberCard.getByRole("button", { name: "Save details" }).click();
  const updatedMemberCard = householdMemberCard(page, "Synthetic Co-owner");
  await expect(
    updatedMemberCard.getByText("Synthetic Co-owner", { exact: true }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.reload();
  await page.getByText("You and your family", { exact: true }).click();
  await expect(
    householdMemberCard(page, "Synthetic Co-owner").getByText(
      "Synthetic Co-owner",
      { exact: true },
    ),
  ).toBeVisible();
  await expectFitsScreen(page);

  await gotoApp(page, "/money");
  await expectFitsScreen(page);
  await page.getByText("Add manually", { exact: true }).click();
  await expectFitsScreen(page);
  await page
    .getByRole("button", { name: "Fixed deposits", exact: true })
    .click();
  await page.getByRole("button", { name: /Continue \(1 selected\)/ }).click();
  await expectFitsScreen(page);
  await page.getByLabel("Bank or post office").fill("Synthetic Bank");
  await page.getByLabel("Amount deposited").fill("80000");
  await page.getByLabel("Interest rate (%)").fill("6.5");
  await page.getByLabel("Matures (month and year)").fill("2028-04");
  await page
    .getByLabel("Whose is this?")
    .selectOption({ label: "Synthetic Co-owner" });
  await page.getByText("More details (for a closer estimate)").click();
  await page.getByLabel("Name this item").fill("Synthetic partner deposit");
  await page.getByRole("button", { name: "Save and return" }).click();
  await expect(page).toHaveURL(/\/money$/);
  await expect(
    page.getByText("Synthetic partner deposit", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByText("Synthetic partner deposit", { exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Choose family or person")
    .selectOption({ label: "Synthetic Co-owner" });
  await expect(
    page.getByText("Synthetic partner deposit", { exact: true }),
  ).toBeVisible();
  await expectFitsScreen(page);

  await page.getByRole("link", { name: /Synthetic partner deposit/ }).click();
  await expectFitsScreen(page);
  await page.getByText("Edit history", { exact: true }).click();
  await page.getByRole("button", { name: "Mark as closed" }).first().click();
  await page.getByRole("button", { name: "Mark as closed" }).last().click();
  await gotoApp(page, "/settings");
  await page.getByText("You and your family", { exact: true }).click();
  await expectFitsScreen(page);
  const renamedMember = householdMemberCard(page, "Synthetic Co-owner");
  await renamedMember.getByRole("button", { name: "Remove person" }).click();
  await expect(
    page.getByRole("heading", { name: "Remove person" }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.getByRole("button", { name: "Remove person" }).last().click();
  await expect(
    page
      .getByRole("region", { name: "Household" })
      .getByText("Synthetic Co-owner", { exact: true }),
  ).toHaveCount(0);
  await page.reload();
  await page.getByText("You and your family", { exact: true }).click();
  await expect(
    page
      .getByRole("region", { name: "Household" })
      .getByText("Synthetic Co-owner", { exact: true }),
  ).toHaveCount(0);
  await expectFitsScreen(page);
});

test("SIP amount saves from Money and survives reload", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "C58 save flows use the narrow phone layout.",
  );
  const identity = await startReadyAccount(page, "save-fund-sip");
  await seedResearchPack(identity.email);
  await gotoApp(page, "/money");
  await expectFitsScreen(page);
  await page.getByText("Monthly amounts", { exact: true }).click();
  const amount = page.getByLabel("Synthetic S&P 500 fund");
  await expect(amount).toBeVisible();
  await amount.fill("3200");
  await page.getByRole("button", { name: "Save amount" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Monthly amount saved." }),
  ).toBeVisible();
  await expectFitsScreen(page);
  await page.reload();
  await page.getByText("Monthly amounts", { exact: true }).click();
  await expect(page.getByLabel("Synthetic S&P 500 fund")).toHaveValue(
    "3200.00",
  );
  await expectFitsScreen(page);
});
