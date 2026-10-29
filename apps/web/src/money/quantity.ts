/** Indian digit grouping for a plain number string: "38647.373" → "38,647.373". */
function groupIndian(value: string): string {
  const [whole = "0", fraction] = value.split(".");
  const lastThree = whole.slice(-3);
  const rest = whole.slice(0, -3).replace(/\B(?=(\d{2})+(?!\d))/g, ",");
  const grouped = rest ? `${rest},${lastThree}` : lastThree;
  return fraction ? `${grouped}.${fraction}` : grouped;
}

/** "82 units" for funds, "1,200 shares" for stocks, without trailing zeros. */
export function quantityLabel(quantity: string, securityType: string): string {
  const trimmed = quantity.includes(".")
    ? quantity.replace(/\.?0+$/, "")
    : quantity;
  const unit =
    securityType === "mf" || securityType === "etf" ? "units" : "shares";
  return `${/^\d+(?:\.\d+)?$/.test(trimmed) ? groupIndian(trimmed) : trimmed} ${unit}`;
}
