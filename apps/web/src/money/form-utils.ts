import { formatInr, parseInr } from "../lib/money";
import type { GeneratedAssetField } from "./forms.generated";
import type { AssetType } from "./forms.generated";

export type AssetFormValues = Record<
  string,
  string | number | boolean | string[] | null | undefined
>;

export function initialAssetValues(
  assetType: AssetType,
  fields: readonly GeneratedAssetField[],
  details: Record<string, unknown> = {},
  memberId?: string | null,
): AssetFormValues {
  const values: AssetFormValues = { ...details, asset_type: assetType };
  for (const field of fields) {
    if (values[field.name] === undefined && field.defaultValue !== undefined) {
      values[field.name] = field.defaultValue as AssetFormValues[string];
    }
  }
  values.member_id = memberId ?? values.member_id ?? null;
  return values;
}

/**
 * Amounts in any currency. Rupees accept the shorthand used everywhere else
 * ("38L", "1,25,000"); kroner and euro accept "120,000", but "120.000" stays an
 * error rather than being read as 120.
 */
export function normalizeAmount(value: string, currency: unknown): string {
  const text = value.trim();
  if ((currency ?? "INR") === "INR") return parseInr(text);
  const plain = /^\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?$/.test(text)
    ? text.replaceAll(",", "")
    : text;
  if (!/^\d+(?:\.\d{1,2})?$/.test(plain)) throw new Error("Invalid amount");
  return plain;
}

/** Format a saved draft like a person would type it into its currency field. */
export function formatAmountInput(value: string, currency: unknown): string {
  const normalized = normalizeAmount(value, currency);
  const [whole = "0", rawFraction = ""] = normalized.split(".");
  const fraction = /[1-9]/.test(rawFraction) ? rawFraction.padEnd(2, "0") : "";
  if ((currency ?? "INR") === "INR") {
    return formatInr(normalized, "full", fraction ? 2 : 0).slice(1);
  }
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return fraction ? `${grouped}.${fraction}` : grouped;
}

export function validateAssetValues(
  fields: readonly GeneratedAssetField[],
  values: AssetFormValues,
): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const field of fields) {
    const raw = values[field.name];
    const value = typeof raw === "string" ? raw.trim() : raw;
    if (
      field.required &&
      (value === undefined || value === null || value === "")
    ) {
      errors[field.name] = "This field is required.";
      continue;
    }
    if (value === undefined || value === null || value === "") continue;
    if (field.kind === "rupees") {
      try {
        parseInr(String(value));
      } catch {
        errors[field.name] = "Enter a rupee amount such as 5 lakh or 50k.";
      }
    } else if (field.kind === "amount") {
      try {
        normalizeAmount(String(value), values.currency);
      } catch {
        errors[field.name] =
          (values.currency ?? "INR") === "INR"
            ? "Enter a rupee amount such as 5 lakh or 50k."
            : "Enter an amount with up to two decimal places.";
      }
    } else if (
      field.kind === "rate_pct" &&
      !/^\d+(?:\.\d{1,4})?$/.test(String(value))
    ) {
      errors[field.name] = "Enter a number followed by a percent rate.";
    } else if (
      field.kind === "month_year" &&
      !/^\d{4}-(0[1-9]|1[0-2])$/.test(String(value))
    ) {
      errors[field.name] = "Choose a month and year.";
    } else if (field.kind === "number" && !Number.isFinite(Number(value))) {
      errors[field.name] = "Enter a number.";
    }
  }
  return errors;
}

export function serializeAssetValues(
  assetType: AssetType,
  fields: readonly GeneratedAssetField[],
  values: AssetFormValues,
  existingDetails: Record<string, unknown> = {},
): Record<string, unknown> {
  const body: Record<string, unknown> = { asset_type: assetType };
  for (const field of fields) {
    if (field.name === "asset_type" || field.kind === "hidden") continue;
    const raw = values[field.name];
    if (field.name === "member_id") {
      body.member_id = raw || null;
      continue;
    }
    if (raw === undefined || raw === null || raw === "") {
      if (Object.prototype.hasOwnProperty.call(existingDetails, field.name)) {
        body[field.name] = null;
      }
      continue;
    }
    if (field.kind === "rupees") {
      body[field.name] = parseInr(String(raw));
    } else if (field.kind === "amount") {
      body[field.name] = normalizeAmount(String(raw), values.currency);
    } else if (
      field.kind === "number" ||
      (field.kind === "enum" &&
        field.options?.some((option) => typeof option === "number"))
    ) {
      // Selects hand back text; numeric choices (gold purity, deposit years) must stay numbers.
      body[field.name] = Number(raw);
    } else if (field.kind === "list" && typeof raw === "string") {
      body[field.name] = raw
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
    } else {
      body[field.name] = raw;
    }
  }
  return body;
}
