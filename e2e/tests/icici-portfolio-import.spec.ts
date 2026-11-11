import { readFile } from "node:fs/promises";
import {
  appUrl,
  expect,
  expectFitsScreen,
  seedBrokerImportAccount,
  seedIciciPortfolioSecurities,
  startReadyAccount,
  test,
} from "../support/app";

test("ICICI portfolio import recognizes names and lets me find one share", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "narrow",
    "Import review save flows use the narrow phone layout.",
  );
  const identity = await startReadyAccount(page, "icici-portfolio-import");
  await seedBrokerImportAccount(identity.email);
  await seedIciciPortfolioSecurities(identity.email);
  await page.goto(appUrl("/import"));
  await expectFitsScreen(page);
  await page
    .getByRole("button", { name: "Broker file (Excel or CSV)" })
    .click();
  const brokerPicker = page.getByLabel("Where did you get this file?");
  await expect(brokerPicker).toHaveValue("nordnet");
  await page.getByLabel("Statement file").setInputFiles({
    name: "icici_direct_portfolio.csv",
    mimeType: "text/csv",
    buffer: await readFile(
      new URL(
        "../../tests/fixtures/brokers/icici_direct_portfolio.csv",
        import.meta.url,
      ),
    ),
  });
  const reviewResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "GET" &&
      /\/v1\/imports\/[^/]+$/.test(new URL(response.url()).pathname),
  );
  await page.getByRole("button", { name: "Upload file" }).click();
  await expect(
    page.getByRole("heading", { name: "Review your import" }),
  ).toBeVisible({ timeout: 30_000 });
  await expectFitsScreen(page);
  const detail = (await (await reviewResponse).json()) as {
    broker: string;
    detected_broker: string;
    market_default: string;
    currency_default: string;
    items: Array<{ name: string; status: string; matched_by?: string | null }>;
  };
  expect(detail).toMatchObject({
    broker: "icici_direct",
    detected_broker: "icici_direct",
    market_default: "IN",
    currency_default: "INR",
  });
  expect(
    detail.items.filter(
      (item) => item.status === "new" && item.matched_by === "name",
    ),
  ).toHaveLength(4);
  expect(
    detail.items.find((item) => item.name === "Mystery Basket ETF")?.status,
  ).toBe("unknown");
  await expect(
    page.getByText("We read this as a file from ICICI Direct.", {
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Find this share", exact: true })
    .click();
  const search = page.getByLabel("Share name or NSE symbol");
  await search.fill("SYNTH-ICICI-PICK");
  const result = page.getByRole("button", {
    name: /SYNTH-ICICI-PICK.*Listed Synthetic Index Fund/,
  });
  await expect(result).toBeVisible();
  await result.click();
  await expect(page.getByRole("heading", { name: "Need review" })).toHaveCount(
    0,
  );
  await expectFitsScreen(page);

  await page.getByRole("button", { name: /Confirm import \(/ }).click();
  await expect(
    page.getByRole("heading", {
      name: /Added 5 holdings? · Net worth updated/,
    }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Go to Money" }).click();
  // The chosen share is saved under its listed name, not the broker's row name.
  const importedHolding = page.getByRole("link", {
    name: /^Listed Synthetic Index Fund 5 units/,
  });
  await expect(importedHolding).toBeVisible();
  await expectFitsScreen(page);
  await importedHolding.click();
  const profitLoss = page.getByText("Profit or loss", { exact: true }).locator("..");
  await expect(profitLoss).not.toContainText("Buying price not recorded");
  await expect(profitLoss).toContainText(/\+.*· \+\d+%/);
});
