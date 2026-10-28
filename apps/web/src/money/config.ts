import type { AssetType } from "./forms.generated";

export interface ChecklistTile {
  category: string;
  label: string;
  shortLabel: string;
  assetType?: AssetType | "insurance";
}

export const checklistTiles: readonly ChecklistTile[] = [
  { category: "epf", label: "EPF", shortLabel: "EPF", assetType: "epf" },
  { category: "ppf", label: "PPF", shortLabel: "PPF", assetType: "ppf" },
  {
    category: "fd",
    label: "Fixed deposits",
    shortLabel: "Fixed deposits",
    assetType: "fd",
  },
  {
    category: "post_office",
    label: "Post Office",
    shortLabel: "Post Office",
    assetType: "po_scheme",
  },
  { category: "nps", label: "NPS", shortLabel: "NPS", assetType: "nps" },
  {
    category: "insurance",
    label: "LIC / Insurance",
    shortLabel: "LIC / Insurance",
    assetType: "insurance",
  },
  { category: "gold", label: "Gold", shortLabel: "Gold", assetType: "gold" },
  { category: "bonds", label: "Bonds", shortLabel: "Bonds", assetType: "bond" },
  {
    category: "savings",
    label: "Savings",
    shortLabel: "Savings",
    assetType: "savings",
  },
  { category: "loans", label: "Loans", shortLabel: "Loans", assetType: "loan" },
  {
    category: "property",
    label: "Property",
    shortLabel: "Property",
    assetType: "property",
  },
  {
    category: "us_investments",
    label: "US investments",
    shortLabel: "US investments",
  },
];

export interface ChecklistWorkflow {
  categories: string[];
  index: number;
}

export const CHECKLIST_STORAGE_KEY = "wealth.money-map-checklist.v1";
export const CHECKLIST_PENDING_KEY = "wealth.money-map-checklist.pending.v1";
export const ONBOARDING_STORAGE_KEY = "wealth.onboarding.v1";

export function categoryLabel(category: string): string {
  return (
    checklistTiles.find((tile) => tile.category === category)?.label ?? category
  );
}

export function addPathForCategory(category: string): string {
  const tile = checklistTiles.find(
    (candidate) => candidate.category === category,
  );
  return `/money/add/${tile?.assetType ?? "us_investments"}`;
}
