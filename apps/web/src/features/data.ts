import englishCopy from "../copy/en.json";
import { instrumentLabel as sharedInstrumentLabel } from "../lib/labels";

export type JsonRecord = Record<string, unknown>;

export function asRecord(value: unknown): JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function textValue(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return fallback;
}

export function isPositiveAmount(value: unknown): boolean {
  if (typeof value !== "string" || !/^-?\d+(?:\.\d+)?$/.test(value.trim())) {
    return false;
  }
  const digits = value.trim().replace(/[.-]/g, "");
  return BigInt(digits || "0") > 0n;
}

export function compareDecimalStrings(left: string, right: string): -1 | 0 | 1 {
  const toScaled = (value: string): bigint => {
    const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value.trim());
    if (!match) return 0n;
    const fraction = (match[3] ?? "").padEnd(2, "0").slice(0, 2);
    const absolute = BigInt(`${match[2]}${fraction}`);
    return match[1] === "-" ? -absolute : absolute;
  };
  const a = toScaled(left);
  const b = toScaled(right);
  return a < b ? -1 : a > b ? 1 : 0;
}

export function reasonCopy(code: string): string {
  const reasons = asRecord(englishCopy.reasons);
  const known = reasons[code];
  if (typeof known === "string") return known;
  console.warn(`Unknown reason code: ${code}`);
  return "This action follows your plan. Open the details to learn more.";
}

export function messageCopy(code: string, fallback: string): string {
  const messages = asRecord(englishCopy.messages);
  const known = messages[code];
  return typeof known === "string" ? known : fallback;
}

export function plainName(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function instrumentLabel(code: string): string {
  return sharedInstrumentLabel(code);
}

export function friendlyRouteReason(code: string): string {
  return reasonCopy(code);
}

export function routeHowTo(route: string, amount: string): string {
  const instructions: Record<string, string> = {
    A: englishCopy.household_ui.howto_global_a,
    B: englishCopy.household_ui.howto_global_b,
    C: englishCopy.household_ui.howto_global_c,
  };
  const instruction =
    instructions[route] ?? englishCopy.household_ui.howto_global_generic;
  return instruction.replace(
    "{amount}",
    amount ? `for ${amount}` : "as planned",
  );
}

export function friendlyRouteWarning(code: string): string {
  const normalized = code.toLowerCase();
  const key = normalized.includes("tcs")
    ? "tcs"
    : normalized.includes("schedule_fa")
      ? "schedule_fa"
      : normalized.includes("estate_tax")
        ? "estate_tax"
        : normalized.includes("fx_markup")
          ? "fx_markup"
          : "";
  const warnings = asRecord(englishCopy.household_ui.global_warnings);
  return textValue(
    warnings[key],
    englishCopy.household_ui.global_warning_default,
  );
}

export function formatScenarioLabel(code: string): string {
  const labels: Record<string, string> = {
    inr_minus_10: "If the rupee falls 10%",
    india_equity_minus_25: "If Indian shares fall 25%",
    us_equity_minus_30: "If US shares fall 30%",
    oil_shock: "If there is an oil shock",
    global_risk_off: "If global markets fall together",
  };
  return labels[code] ?? plainName(code);
}
