import { describe, expect, it } from "vitest";
import { exactQuantity, quantityLabel } from "../../src/money/quantity";

describe("quantityLabel", () => {
  it("rounds fund and bond units while preserving share counts", () => {
    expect(quantityLabel("38647.3730", "mf")).toBe("38,647 units");
    expect(quantityLabel("1234567", "equity")).toBe("12,34,567 shares");
    expect(quantityLabel("82.000", "etf")).toBe("82 units");
    expect(quantityLabel("12.5", "bond")).toBe("13 units");
    expect(quantityLabel("4.4", "sgb")).toBe("4 units");
  });

  it("shows fractional holdings below one unit without rounding to zero", () => {
    expect(quantityLabel("0.42", "mf")).toBe("0.42 units");
    expect(quantityLabel("0.0004", "mf")).toBe("<0.001 units");
    expect(quantityLabel("0.999", "mf")).toBe("0.999 units");
  });

  it("keeps the exact holding quantity for details", () => {
    expect(exactQuantity("0.423700")).toBe("0.4237");
    expect(exactQuantity("82.000")).toBe("82");
    expect(exactQuantity("1200")).toBe("1200");
  });
});
