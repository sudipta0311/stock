import { apiFetch } from "../api/client";

export interface DkkInrRate {
  date: string;
  rate: string;
}

/** The ECB DKK/INR rate the server stores; the CSP lets the app call only its API. */
export async function getDkkInrRate(): Promise<DkkInrRate> {
  const body = await apiFetch<{ date?: unknown; rate?: unknown }>(
    "/v1/fx/dkk-inr",
  );
  if (
    typeof body.date !== "string" ||
    typeof body.rate !== "string" ||
    !/^\d{4}-\d{2}-\d{2}$/.test(body.date) ||
    !/^\d+(?:\.\d+)?$/.test(body.rate)
  ) {
    throw new Error("The reference exchange rate is unavailable.");
  }
  return { date: body.date, rate: body.rate };
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
