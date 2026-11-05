import { randomUUID } from "node:crypto";
import type { Page } from "@playwright/test";
import {
  appUrl,
  expect,
  expectFitsScreen,
  seedBrokerImportAccount,
  startReadyAccount,
  test,
} from "../support/app";

type BrokerImportDetail = {
  account_id: string | null;
  auto_mapped: boolean;
  items: Array<{
    name: string;
    symbol: string | null;
    market: string | null;
    currency: string | null;
  }>;
};

function syntheticIsin(country: string): string {
  const body = randomUUID().replace(/\D/g, "").slice(0, 9).padEnd(9, "0");
  const expanded = `${country}${body}`
    .split("")
    .map((character) =>
      /[A-Z]/.test(character)
        ? String(character.charCodeAt(0) - 55)
        : character,
    )
    .join("");
  let sum = 0;
  let doubleDigit = true;
  for (let index = expanded.length - 1; index >= 0; index -= 1) {
    const value = Number(expanded[index]) * (doubleDigit ? 2 : 1);
    sum += Math.floor(value / 10) + (value % 10);
    doubleDigit = !doubleDigit;
  }
  return `${country}${body}${(10 - (sum % 10)) % 10}`;
}

async function uploadBrokerFile(
  page: Page,
  options: {
    broker: string;
    filename: string;
    mimeType: string;
    buffer: Buffer;
    account?: string;
    fromInbox?: boolean;
  },
): Promise<BrokerImportDetail> {
  if (options.fromInbox) {
    await page.goto(appUrl("/inbox"));
    await page.getByRole("link", { name: "Upload a file" }).click();
  } else {
    await page.goto(appUrl("/import"));
  }
  await page
    .getByRole("button", { name: "Broker file (Excel or CSV)" })
    .click();
  await page
    .getByLabel("Where did you get this file?")
    .selectOption(options.broker);
  if (options.account) {
    await page
      .getByLabel("Account (optional)")
      .selectOption({ label: options.account });
  }
  await page.getByLabel("Statement file").setInputFiles({
    name: options.filename,
    mimeType: options.mimeType,
    buffer: options.buffer,
  });
  const reviewResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "GET" &&
      /\/v1\/imports\/[^/]+$/.test(new URL(response.url()).pathname),
  );
  await page.getByRole("button", { name: "Upload file" }).click();
  await expect(
    page.getByRole("heading", { name: "Review your import" }),
  ).toBeVisible({ timeout: 120_000 });
  await expect(
    page.getByText("We matched your file's columns automatically."),
  ).toBeVisible();
  await expectFitsScreen(page);
  return (await (await reviewResponse).json()) as BrokerImportDetail;
}

test("Inbox upload reaches the broker import review", async ({ page }) => {
  await startReadyAccount(page, "broker-import-from-inbox");
  const isin = syntheticIsin("IN");
  const htmlXls = Buffer.from(
    `<html><table><tr><th>Scrip Name</th><th>ISIN</th><th>Qty</th><th>Avg Cost Price</th><th>LTP</th></tr><tr><td>Synthetic inbox share</td><td>${isin}</td><td>2</td><td>100.00</td><td>120.00</td></tr></table></html>`,
    "utf8",
  );
  const detail = await uploadBrokerFile(page, {
    broker: "hdfc_securities",
    filename: "synthetic-inbox.xls",
    mimeType: "application/vnd.ms-excel",
    buffer: htmlXls,
    fromInbox: true,
  });
  expect(detail.auto_mapped).toBe(true);
  expect(detail.items[0]?.name).toBe("Synthetic inbox share");
});

async function confirmAndFindInMoney(page: Page, name: string): Promise<void> {
  await page.getByRole("button", { name: /Confirm import \(/ }).click();
  await expect(
    page.getByRole("heading", {
      name: /Added \d+ holdings? · Net worth updated/,
    }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Go to Money" }).click();
  await expect(page.getByText(name, { exact: true })).toBeVisible();
  await expectFitsScreen(page);
}

test("three broker files auto-map, confirm, and appear in Money", async ({
  page,
}) => {
  const identity = await startReadyAccount(page, "broker-import");
  const indianIsin = syntheticIsin("IN");
  const htmlXls = Buffer.from(
    `<html><table><tr><th>Scrip Name</th><th>ISIN</th><th>Qty</th><th>Avg Cost Price</th><th>LTP</th></tr><tr><td>Synthetic India share</td><td>${indianIsin}</td><td>2</td><td>100.00</td><td>120.00</td></tr></table></html>`,
    "utf8",
  );
  const indianDetail = await uploadBrokerFile(page, {
    broker: "hdfc_securities",
    filename: "synthetic-india.xls",
    mimeType: "application/vnd.ms-excel",
    buffer: htmlXls,
  });
  expect(indianDetail.auto_mapped).toBe(true);
  expect(indianDetail.items[0]).toMatchObject({
    name: "Synthetic India share",
    market: "IN",
    currency: "INR",
  });
  await confirmAndFindInMoney(page, "Synthetic India share");

  const usSymbol = `SYNTH${randomUUID().replace(/\W/g, "").slice(0, 6).toUpperCase()}`;
  const usCsv = Buffer.from(
    `Symbol,Name,Quantity,Invested Value,Current Market Price\n${usSymbol},Synthetic US share,3,90.00,40.00\n`,
    "utf8",
  );
  const usDetail = await uploadBrokerFile(page, {
    broker: "indmoney",
    filename: "synthetic-us.csv",
    mimeType: "text/csv",
    buffer: usCsv,
  });
  expect(usDetail.auto_mapped).toBe(true);
  expect(usDetail.items[0]).toMatchObject({
    name: "Synthetic US share",
    symbol: usSymbol,
    market: "US",
    currency: "USD",
  });
  await confirmAndFindInMoney(page, "Synthetic US share");

  await seedBrokerImportAccount(identity.email);
  const europeanIsin = syntheticIsin("IE");
  const danishUsIsin = syntheticIsin("US");
  const danishTable = [
    "Værdipapir\tSymbol\tISIN\tAntal\tGAK\tKurs\tMarkedsværdi\tValuta",
    `Synthetic UCITS ETF\tSYNTHETF\t${europeanIsin}\t2\t1.234,56\t1.300,00\t2.600,00\tEUR`,
    `Synthetic Danish US share\tSYNTHDKUS\t${danishUsIsin}\t1\t100,00\t120,00\t120,00\tUSD`,
  ].join("\n");
  const danishUtf16 = Buffer.concat([
    Buffer.from([0xff, 0xfe]),
    Buffer.from(danishTable, "utf16le"),
  ]);
  const danishDetail = await uploadBrokerFile(page, {
    broker: "nordnet",
    filename: "synthetic-denmark.tsv",
    mimeType: "text/tab-separated-values",
    buffer: danishUtf16,
    account: "Synthetic Nordnet account · investment",
  });
  expect(danishDetail.auto_mapped).toBe(true);
  expect(danishDetail.account_id).toBeTruthy();
  expect(danishDetail.items).toHaveLength(2);
  expect(
    danishDetail.items.find((item) => item.symbol === "SYNTHDKUS"),
  ).toMatchObject({ market: "US", currency: "USD" });
  await page.getByLabel("Yes", { exact: true }).check();
  await confirmAndFindInMoney(page, "Synthetic Danish US share");
  await expect(
    page.getByText("Synthetic UCITS ETF", { exact: true }),
  ).toBeVisible();
});
