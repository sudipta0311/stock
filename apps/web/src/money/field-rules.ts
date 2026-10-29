import type { AssetFormValues } from "./form-utils";
import type { AssetType, GeneratedAssetField } from "./forms.generated";

interface KindRule {
  show: readonly string[];
  require?: readonly string[];
}

interface DependentFields {
  on: string;
  fields: readonly string[];
  rules: Record<string, KindRule>;
}

// Mirrors GoldAsset.validate_kind_fields and PostOfficeAsset.validate_td_years in
// wealth.schemas.assets: the API needs some fields for one kind and rejects them
// for the others, which the generated (flat) form definition can't express.
const DEPENDENT_FIELDS: Partial<Record<AssetType, DependentFields>> = {
  gold: {
    on: "kind",
    fields: [
      "units",
      "grams",
      "purity",
      "series",
      "issue",
      "issue_price_per_gram",
      "purpose",
    ],
    rules: {
      physical: {
        show: ["grams", "purity", "purpose"],
        require: ["grams", "purity"],
      },
      digital: { show: ["grams"], require: ["grams"] },
      sgb: { show: ["units", "series", "issue", "issue_price_per_gram"] },
      etf: { show: ["units", "series"] },
      mf: { show: ["units", "series"] },
    },
  },
  po_scheme: {
    on: "scheme",
    fields: ["td_years"],
    rules: { td: { show: ["td_years"], require: ["td_years"] } },
  },
};

function hiddenFields(
  assetType: AssetType,
  values: AssetFormValues,
): Set<string> {
  const config = DEPENDENT_FIELDS[assetType];
  if (!config) return new Set();
  const rule = config.rules[String(values[config.on] ?? "")];
  return new Set(config.fields.filter((name) => !rule?.show.includes(name)));
}

/** The fields to show for the current choice, with the ones it needs marked required. */
export function applyFieldRules<T extends GeneratedAssetField>(
  assetType: AssetType,
  fields: readonly T[],
  values: AssetFormValues,
): T[] {
  const config = DEPENDENT_FIELDS[assetType];
  if (!config) return [...fields];
  const rule = config.rules[String(values[config.on] ?? "")];
  const hidden = hiddenFields(assetType, values);
  return fields
    .filter((field) => !hidden.has(field.name))
    .map((field) =>
      rule?.require?.includes(field.name)
        ? { ...field, required: true }
        : field,
    );
}

/** Blanks values of fields that don't apply to the current choice, so they are not sent. */
export function withoutHiddenValues(
  assetType: AssetType,
  values: AssetFormValues,
): AssetFormValues {
  const hidden = hiddenFields(assetType, values);
  if (!hidden.size) return values;
  return Object.fromEntries(
    Object.entries(values).map(([name, value]) => [
      name,
      hidden.has(name) ? "" : value,
    ]),
  ) as AssetFormValues;
}
