export interface DkkInrRate {
  date: string;
  rate: string;
}

const FRANKFURTER_RATE_URL = "https://api.frankfurter.dev/v2/rate/dkk/inr";

export async function getDkkInrRate(): Promise<DkkInrRate> {
  const response = await fetch(FRANKFURTER_RATE_URL, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok)
    throw new Error("The reference exchange rate is unavailable.");
  const payload: unknown = await response.json();
  if (typeof payload !== "object" || payload === null) {
    throw new Error("The reference exchange rate is unavailable.");
  }
  const body = payload as Record<string, unknown>;
  const rate = body.rate;
  if (
    typeof body.date !== "string" ||
    (typeof rate !== "number" && typeof rate !== "string") ||
    !/^\d{4}-\d{2}-\d{2}$/.test(body.date) ||
    !/^\d+(?:\.\d+)?$/.test(String(rate))
  ) {
    throw new Error("The reference exchange rate is unavailable.");
  }
  return { date: body.date, rate: String(rate) };
}

function scaledDecimal(value: string, scale: number): bigint | null {
  const match = /^(\d+)(?:\.(\d+))?$/.exec(value.trim());
  if (!match) return null;
  const decimals = match[2] ?? "";
  if (decimals.length > scale) return null;
  return BigInt(`${match[1]}${decimals.padEnd(scale, "0")}`);
}

function roundRatioHalfUp(numerator: bigint, denominator: bigint): bigint {
  if (denominator <= 0n) return 0n;
  return (numerator + denominator / 2n) / denominator;
}

export function inrToDkk(amountInr: string, rate: DkkInrRate): string | null {
  const paise = scaledDecimal(amountInr, 2);
  const rateScaled = scaledDecimal(rate.rate, 8);
  if (paise === null || rateScaled === null || rateScaled === 0n) return null;
  const ore = roundRatioHalfUp(paise * 100_000_000n, rateScaled);
  return `${ore / 100n}.${String(ore % 100n).padStart(2, "0")}`;
}

export function dkkToInr(amountDkk: string, rate: DkkInrRate): string | null {
  const ore = scaledDecimal(amountDkk, 2);
  const rateScaled = scaledDecimal(rate.rate, 8);
  if (ore === null || rateScaled === null) return null;
  const paise = roundRatioHalfUp(ore * rateScaled, 100_000_000n);
  return `${paise / 100n}.${String(paise % 100n).padStart(2, "0")}`;
}

export function formatDkk(value: string): string {
  const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value.trim());
  if (!match) return "DKK 0";
  const integer = match[2];
  if (!integer) return "DKK 0";
  let rounded = BigInt(integer);
  if ((match[3]?.[0] ?? "0") >= "5") rounded += 1n;
  const sign = rounded === 0n ? "" : (match[1] ?? "");
  const grouped = rounded.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `${sign}DKK ${grouped}`;
}

export function formatRateToTwoDecimals(value: string): string {
  const match = /^(\d+)(?:\.(\d+))?$/.exec(value.trim());
  if (!match) return "0.00";
  const integer = match[1];
  if (!integer) return "0.00";
  const fraction = match[2] ?? "";
  let hundredths = BigInt(integer) * 100n;
  hundredths += BigInt(fraction.slice(0, 2).padEnd(2, "0") || "0");
  if ((fraction[2] ?? "0") >= "5") hundredths += 1n;
  return `${hundredths / 100n}.${String(hundredths % 100n).padStart(2, "0")}`;
}
