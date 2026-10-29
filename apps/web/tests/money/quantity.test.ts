import { describe, expect, it } from "vitest";
import { quantityLabel } from "../../src/money/quantity";

describe("quantityLabel", () => {
  it("groups large unit counts the Indian way and trims trailing zeros", () => {
    expect(quantityLabel("38647.3730", "mf")).toBe("38,647.373 units");
    expect(quantityLabel("1234567", "equity")).toBe("12,34,567 shares");
    expect(quantityLabel("82.000", "etf")).toBe("82 units");
  });
});
