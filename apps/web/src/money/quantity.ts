/** "82 units" for funds, "20 shares" for stocks, without trailing zeros. */
export function quantityLabel(quantity: string, securityType: string): string {
  const trimmed = quantity.includes(".")
    ? quantity.replace(/\.?0+$/, "")
    : quantity;
  const unit =
    securityType === "mf" || securityType === "etf" ? "units" : "shares";
  return `${trimmed} ${unit}`;
}
