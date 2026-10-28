import { expect, it } from "vitest";
import {
  formatInr,
  formatUsd,
  parseInr,
  pct,
  roundPaise,
} from "../src/lib/money";

it.each([
  ["500000", "500000.00"],
  ["5,00,000", "500000.00"],
  ["500,000", "500000.00"],
  ["1,23,45,678.50", "12345678.50"],
  ["₹5,00,000", "500000.00"],
  ["Rs5,00,000", "500000.00"],
  ["Rs. 5,00,000", "500000.00"],
  ["INR 500,000", "500000.00"],
  ["5L", "500000.00"],
  ["5 lakh", "500000.00"],
  ["1.2 cr", "12000000.00"],
  ["2.5Cr", "25000000.00"],
  ["50k", "50000.00"],
  ["1 thousand", "1000.00"],
  ["10 lakhs", "1000000.00"],
  ["1 crore", "10000000.00"],
  ["0.001k", "1.00"],
  ["0.00001k", "0.01"],
  ["1.230", "1.23"],
])("parseInr accepts %s", (text, expected) => {
  expect(parseInr(text)).toBe(expected);
});

it.each([
  "",
  "   ",
  "abc",
  "5L5",
  "1..2",
  "5 lakh crore",
  "-5",
  "₹-5",
  "1,23,4",
  "1234,567",
  "1,000,00",
  "1.001",
  "0.00001",
])("parseInr rejects invalid or over-precise input %s", (text) =>
  expect(() => parseInr(text)).toThrow(),
);

it.each([
  ["12345678.5", "full", 2, "₹1,23,45,678.50"],
  ["12345678.5", "full", 0, "₹1,23,45,679"],
  ["12345.675", "full", 2, "₹12,345.68"],
  ["0", "full", 2, "₹0.00"],
  ["99999", "compact", 2, "₹99,999"],
  ["540000", "compact", 2, "₹5.4 L"],
  ["1235000", "compact", 2, "₹12.35 L"],
  ["12400000", "compact", 2, "₹1.24 Cr"],
  ["120000000", "compact", 2, "₹12 Cr"],
  ["-5000", "full", 2, "-₹5,000.00"],
  ["-540000", "compact", 2, "-₹5.4 L"],
  ["-12345678.5", "full", 2, "-₹1,23,45,678.50"],
  ["9999999", "compact", 2, "₹1 Cr"],
  ["99999.6", "compact", 2, "₹1 L"],
  ["9999999.995", "compact", 2, "₹1 Cr"],
  ["9999999999", "compact", 2, "₹1000 Cr"],
] as const)(
  "formatInr formats %s with %s style",
  (value, style, decimals, expected) => {
    expect(formatInr(value, style, decimals)).toBe(expected);
  },
);

it("formats USD with international grouping", () => {
  expect(formatUsd("1234.5")).toBe("$1,234.50");
  expect(formatUsd("-1234.5", 0)).toBe("-$1,235");
});

it("rounds to paise and validates percentage boundaries", () => {
  expect(roundPaise("1.005")).toBe("1.01");
  expect(pct("7.10")).toBe("7.10");
  expect(pct(100)).toBe("100");
  for (const invalid of ["-0.1", "100.01", "NaN", 3.14])
    expect(() => pct(invalid)).toThrow();
});
