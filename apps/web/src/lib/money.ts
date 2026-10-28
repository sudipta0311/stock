const INTEGER_PATTERN = /^(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3})$/;
const AMOUNT_PATTERN = /^(\d+(?:,\d+)*)(?:\.(\d+))?$/;
const CURRENCY_PREFIX = /^(?:₹|rs\.?|inr)\s*/i;
const SUFFIX_PATTERN = /\s*(thousand|lakhs?|lac|cr|crores?|[kl])\s*$/i;

type Grouping = "indian" | "international" | "none";

function indianGroup(integer: string): string {
  if (integer.length <= 3) return integer;
  const lastThree = integer.slice(-3);
  const leading = integer.slice(0, -3);
  const groups: string[] = [];
  for (let end = leading.length; end > 0; end -= 2) {
    groups.unshift(leading.slice(Math.max(0, end - 2), end));
  }
  return `${groups.join(",")},${lastThree}`;
}

function formatScaled(
  value: bigint,
  decimals: number,
  grouping: Grouping,
  trim = false,
): string {
  const sign = value < 0n ? "-" : "";
  const magnitude = value < 0n ? -value : value;
  const digits = magnitude.toString().padStart(decimals + 1, "0");
  const integer = decimals === 0 ? digits : digits.slice(0, -decimals);
  const fraction = decimals === 0 ? "" : digits.slice(-decimals);
  const grouped =
    grouping === "indian"
      ? indianGroup(integer)
      : grouping === "international"
        ? integer.replace(/\B(?=(\d{3})+(?!\d))/g, ",")
        : integer;
  const shownFraction = trim ? fraction.replace(/0+$/, "") : fraction;
  const number = shownFraction ? `${grouped}.${shownFraction}` : grouped;
  return `${sign}${number}`;
}

function scaledInteger(
  value: string,
  decimals: number,
  decimalShift = 0,
): bigint {
  if (!Number.isInteger(decimals) || decimals < 0 || decimals > 100) {
    throw new RangeError(
      "decimals must be a non-negative integer no greater than 100",
    );
  }
  const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value.trim());
  if (!match) throw new TypeError("amount must be a plain decimal string");
  const whole = match[2] ?? "";
  const fraction = match[3] ?? "";
  let scaled = BigInt(`${whole}${fraction}`);
  const power = decimalShift + decimals - fraction.length;
  if (power >= 0) {
    scaled *= 10n ** BigInt(power);
  } else {
    const divisor = 10n ** BigInt(-power);
    const remainder = scaled % divisor;
    scaled /= divisor;
    if (remainder * 2n >= divisor) scaled += 1n;
  }
  return scaled;
}

function hasNegativeSign(value: string): boolean {
  const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value.trim());
  return match?.[1] === "-" && BigInt(`${match[2]}${match[3] ?? ""}`) !== 0n;
}

function normalizeMoneyText(text: string): { digits: string; scale: number } {
  if (typeof text !== "string") throw new TypeError("INR amount must be text");
  let cleaned = text.trim();
  if (!cleaned) throw new TypeError("INR amount cannot be empty");
  if (cleaned.startsWith("-"))
    throw new TypeError("INR amount cannot be negative");
  cleaned = cleaned.replace(CURRENCY_PREFIX, "");
  if (cleaned.startsWith("-"))
    throw new TypeError("INR amount cannot be negative");

  let suffixPower = 0;
  const suffix = SUFFIX_PATTERN.exec(cleaned);
  if (suffix) {
    cleaned = cleaned.slice(0, suffix.index).trimEnd();
    const suffixName = (suffix[1] ?? "").toLowerCase();
    suffixPower = ["k", "thousand"].includes(suffixName)
      ? 3
      : ["l", "lac", "lakh", "lakhs"].includes(suffixName)
        ? 5
        : 7;
  }

  const match = AMOUNT_PATTERN.exec(cleaned);
  const integerText = match?.[1];
  if (!match || !integerText || !INTEGER_PATTERN.test(integerText))
    throw new TypeError(`Invalid INR amount: ${text}`);
  const whole = integerText.replaceAll(",", "");
  const fraction = (match[2] ?? "").replace(/0+$/, "");
  const scale = 2 + suffixPower - fraction.length;
  if (scale < 0) {
    throw new TypeError(
      "INR amount cannot have more than 2 decimal places after scaling",
    );
  }
  return { digits: `${whole}${fraction}`, scale };
}

/** Parse supported Indian rupee text into an exact canonical amount string. */
export function parseInr(text: string): string {
  const { digits, scale } = normalizeMoneyText(text);
  const paise = BigInt(digits) * 10n ** BigInt(scale);
  const whole = paise / 100n;
  const fraction = (paise % 100n).toString().padStart(2, "0");
  return `${whole}.${fraction}`;
}

/** Format an exact decimal string using Indian grouping and half-up rounding. */
export function formatInr(
  value: string,
  style: "full" | "compact" = "full",
  decimals = 2,
): string {
  const sign = hasNegativeSign(value) ? "-" : "";
  const magnitude = value.trim().replace(/^-/, "");
  if (style === "full") {
    return `${sign}₹${formatScaled(scaledInteger(magnitude, decimals), decimals, "indian")}`;
  }
  if (style !== "compact")
    throw new TypeError("style must be 'full' or 'compact'");

  const roundedRupees = scaledInteger(magnitude, 0);
  if (roundedRupees < 100_000n) {
    return `${sign}₹${formatScaled(roundedRupees, 0, "indian")}`;
  }
  const lakhs = scaledInteger(magnitude, 2, -5);
  if (lakhs < 10_000n)
    return `${sign}₹${formatScaled(lakhs, 2, "none", true)} L`;
  const crores = scaledInteger(magnitude, 2, -7);
  return `${sign}₹${formatScaled(crores, 2, "none", true)} Cr`;
}

/** Format a USD decimal string with international grouping. */
export function formatUsd(value: string, decimals = 2): string {
  const sign = hasNegativeSign(value) ? "-" : "";
  const magnitude = value.trim().replace(/^-/, "");
  return `${sign}$${formatScaled(scaledInteger(magnitude, decimals), decimals, "international")}`;
}

export function roundPaise(value: string): string {
  return formatScaled(
    scaledInteger(value.trim().replace(/^-/, ""), 2) *
      (hasNegativeSign(value) ? -1n : 1n),
    2,
    "international",
  );
}

/** Add rupee values without converting them to binary floating point. */
export function sumInr(values: readonly string[]): string {
  const total = values.reduce((sum, value) => {
    const magnitude = value.trim().replace(/^-/, "");
    const sign = hasNegativeSign(value) ? -1n : 1n;
    return sum + scaledInteger(magnitude, 2) * sign;
  }, 0n);
  return formatScaled(total, 2, "none");
}

export function pct(value: string | number | bigint): string {
  const text =
    typeof value === "bigint"
      ? value.toString()
      : typeof value === "number" && Number.isSafeInteger(value)
        ? value.toString()
        : value;
  if (typeof text !== "string" || !/^\d+(?:\.\d+)?$/.test(text)) {
    throw new TypeError(
      "percentage must be text, an integer, or a decimal string",
    );
  }
  const [whole, fraction = ""] = text.split(".");
  const integer = BigInt(whole ?? "");
  if (integer > 100n || (integer === 100n && /[1-9]/.test(fraction))) {
    throw new RangeError("percentage must be between 0 and 100");
  }
  return text;
}
