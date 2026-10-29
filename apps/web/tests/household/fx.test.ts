import { describe, expect, it } from "vitest";
import { formatDkk, formatRateToTwoDecimals } from "../../src/household/fx";

describe("formatDkk", () => {
  it("rounds summary amounts to whole kroner with international grouping", () => {
    expect(formatDkk("0.00")).toBe("DKK 0");
    expect(formatDkk("35242.29")).toBe("DKK 35,242");
    expect(formatDkk("35242.50")).toBe("DKK 35,243");
    expect(formatDkk("-0.49")).toBe("DKK 0");
  });
});

describe("formatRateToTwoDecimals", () => {
  it("rounds reference rates to two decimal places without float math", () => {
    expect(formatRateToTwoDecimals("14.1875")).toBe("14.19");
    expect(formatRateToTwoDecimals("14.184")).toBe("14.18");
    expect(formatRateToTwoDecimals("14")).toBe("14.00");
  });
});
