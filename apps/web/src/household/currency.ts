import { formatInr } from "../lib/money";
import type { PrimaryCurrency } from "./HouseholdContext";
import { dkkToInr, formatDkk, inrToDkk } from "./fx";
import type { DkkInrRate } from "./fx";

export function convertInrToDkk(inr: string, rate: DkkInrRate): string | null {
  return inrToDkk(inr, rate);
}

export function convertDkkToInr(dkk: string, rate: DkkInrRate): string | null {
  return dkkToInr(dkk, rate);
}

export { formatDkk };

export function formatHouseholdAmount(
  inr: string,
  primary: PrimaryCurrency,
  rate: DkkInrRate | undefined,
  compact = true,
): string {
  if (primary === "DKK" && rate) {
    const dkk = inrToDkk(inr, rate);
    if (dkk !== null) return formatDkk(dkk);
  }
  return formatInr(inr, compact ? "compact" : "full");
}

export function formatSecondaryAmount(
  inr: string,
  primary: PrimaryCurrency,
  rate: DkkInrRate | undefined,
  compact = true,
): string | null {
  if (!rate) return null;
  if (primary === "INR") {
    const dkk = inrToDkk(inr, rate);
    return dkk === null ? null : formatDkk(dkk);
  }
  return formatInr(inr, compact ? "compact" : "full");
}
