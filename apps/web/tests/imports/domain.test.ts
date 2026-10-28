import { describe, expect, it } from "vitest";
import type { ImportItem, ImportJob } from "../../src/imports/api";
import {
  countImportItems,
  groupImportItems,
  importErrorCopy,
  maskAccountReference,
  quantityChange,
  shouldShowCasNudge,
  validateImportFile,
} from "../../src/imports/domain";

function item(overrides: Partial<ImportItem> = {}): ImportItem {
  return {
    id: "item-one",
    kind: "equity",
    isin: "IN0000000001",
    name: "Example Fund",
    quantity: "12.00",
    statement_value: "1200.00",
    account_ref: "account-0123",
    sleeve: "equity_india",
    status: "new",
    diff: null,
    match: {},
    ...overrides,
  };
}

function job(overrides: Partial<ImportJob> = {}): ImportJob {
  return {
    id: "job-one",
    doc_type: "nsdl_cas",
    status: "confirmed",
    error_code: null,
    source: "nsdl",
    statement_date: "2026-09-01",
    counts: { new: 1 },
    created_at: "2026-09-02T00:00:00Z",
    confirmed_at: "2026-09-02T00:00:00Z",
    ...overrides,
  };
}

describe("import file checks and copy", () => {
  it("accepts a PDF up to ten megabytes and rejects larger files", () => {
    expect(
      validateImportFile({ name: "cas.pdf", size: 10 * 1024 * 1024 }),
    ).toBeNull();
    expect(
      validateImportFile({ name: "cas.pdf", size: 10 * 1024 * 1024 + 1 })?.code,
    ).toBe("too_large");
  });

  it("allows broker formats only when the API feature is available", () => {
    expect(
      validateImportFile({ name: "holdings.csv", size: 100 }, false)?.code,
    ).toBe("unsupported_format");
    expect(
      validateImportFile({ name: "holdings.xlsx", size: 100 }, true),
    ).toBeNull();
  });

  it.each([
    ["wrong_password", "That password didn't open the file."],
    ["not_a_pdf", "We couldn't read this as a PDF."],
    ["too_large", "This file is over 10 MB."],
    ["unsupported_format", "This statement format isn't supported yet."],
    ["parse_timeout", "The statement is taking too long to read."],
  ])("maps %s to friendly next steps", (code, message) => {
    expect(importErrorCopy(code).message).toBe(message);
    expect(importErrorCopy(code).nextStep.length).toBeGreaterThan(10);
  });
});

describe("import review data", () => {
  it("groups new, changed, duplicate, and unknown items and counts each group", () => {
    const items = [
      item({ id: "new", status: "new" }),
      item({
        id: "changed",
        status: "changed",
        diff: { old_quantity: "10.00", new_quantity: "12.00" },
      }),
      item({ id: "duplicate", status: "duplicate" }),
      item({ id: "unknown", status: "unknown", kind: "unknown", sleeve: null }),
    ];
    const groups = groupImportItems(items);
    expect(groups.new.map(({ id }) => id)).toEqual(["new"]);
    expect(groups.changed.map(({ id }) => id)).toEqual(["changed"]);
    expect(groups.duplicate.map(({ id }) => id)).toEqual(["duplicate"]);
    expect(groups.unknown.map(({ id }) => id)).toEqual(["unknown"]);
    expect(countImportItems(items)).toEqual({
      new: 1,
      changed: 1,
      duplicate: 1,
      unknown: 1,
    });
    expect(quantityChange(items[1]!)).toBe("10.00 → 12.00");
  });

  it("masks account references before they are displayed", () => {
    expect(maskAccountReference("account-0123")).toBe("•••• 0123");
    expect(maskAccountReference(null)).toBe("Account hidden");
  });
});

describe("monthly CAS nudge", () => {
  const now = new Date("2026-10-28T00:00:00Z");

  it("shows when there has never been a confirmed CAS or it is older than thirty days", () => {
    expect(shouldShowCasNudge([], now)).toBe(true);
    expect(
      shouldShowCasNudge([job({ confirmed_at: "2026-09-20T00:00:00Z" })], now),
    ).toBe(true);
  });

  it("hides after a recent CAS confirmation", () => {
    expect(
      shouldShowCasNudge([job({ confirmed_at: "2026-10-10T00:00:00Z" })], now),
    ).toBe(false);
  });
});
