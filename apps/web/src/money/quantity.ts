/** Indian digit grouping for a plain number string: "38647.373" → "38,647.373". */
function groupIndian(value: string): string {
  const [whole = "0", fraction] = value.split(".");
  const lastThree = whole.slice(-3);
  const rest = whole.slice(0, -3).replace(/\B(?=(\d{2})+(?!\d))/g, ",");
  const grouped = rest ? `${rest},${lastThree}` : lastThree;
  return fraction ? `${grouped}.${fraction}` : grouped;
}

function roundedWhole(quantity: string): string {
  const match = /^(\d+)(?:\.(\d+))?$/.exec(quantity);
  if (!match) return quantity;
  const whole = BigInt(match[1] ?? "0");
  const fraction = match[2] ?? "";
  return (whole + ((fraction[0] ?? "0") >= "5" ? 1n : 0n)).toString();
}

function roundedSubUnit(quantity: string): string {
  const match = /^(\d+)(?:\.(\d+))?$/.exec(quantity);
  if (!match) return quantity;
  const fraction = match[2] ?? "";
  const thousandths =
    BigInt(match[1] ?? "0") * 1000n +
    BigInt(fraction.slice(0, 3).padEnd(3, "0") || "0") +
    ((fraction[3] ?? "0") >= "5" ? 1n : 0n);
  if (thousandths === 0n && /[1-9]/.test(quantity)) return "<0.001";
  const whole = thousandths / 1000n;
  const decimals = (thousandths % 1000n).toString().padStart(3, "0");
  const trimmedDecimals = decimals.replace(/0+$/, "");
  return trimmedDecimals ? `${whole}.${trimmedDecimals}` : whole.toString();
}

/** Exact quantity for closed holding details, with insignificant zeros removed. */
export function exactQuantity(quantity: string): string {
  return quantity.includes(".") ? quantity.replace(/\.?0+$/, "") : quantity;
}

/** "82 units" for funds, "1,200 shares" for stocks, without trailing zeros. */
export function quantityLabel(
  quantity: string,
  securityType: string,
  options: { exact?: boolean } = {},
): string {
  const isUnitCount = ["mf", "etf", "bond", "sgb"].includes(securityType);
  const isSubUnit = /^(?:0\.\d*[1-9]\d*|0\.0*[1-9]\d*)$/.test(quantity);
  const trimmed = options.exact
    ? exactQuantity(quantity)
    : isUnitCount
      ? isSubUnit
        ? roundedSubUnit(quantity)
        : roundedWhole(quantity)
      : exactQuantity(quantity);
  const unit = isUnitCount
    ? trimmed === "1"
      ? "unit"
      : "units"
    : trimmed === "1"
      ? "share"
      : "shares";
  return `${/^\d+(?:\.\d+)?$/.test(trimmed) ? groupIndian(trimmed) : trimmed} ${unit}`;
}
