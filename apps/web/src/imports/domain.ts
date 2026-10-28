import { ApiError } from "../api/client";
import type { ImportItem, ImportJob } from "./api";

export const MAX_IMPORT_BYTES = 10 * 1024 * 1024;

export type ImportCategory = "new" | "changed" | "duplicate" | "unknown";

export interface ImportCounts {
  new: number;
  changed: number;
  duplicate: number;
  unknown: number;
}

export interface ImportFileProblem {
  code: "too_large" | "unsupported_format";
  message: string;
  nextStep: string;
}

export function validateImportFile(
  file: Pick<File, "name" | "size">,
  allowBroker = false,
): ImportFileProblem | null {
  if (file.size > MAX_IMPORT_BYTES) {
    return {
      code: "too_large",
      message: "This file is over 10 MB.",
      nextStep: "Choose a smaller statement file and try again.",
    };
  }
  const name = file.name.toLocaleLowerCase();
  const allowed =
    name.endsWith(".pdf") || (allowBroker && /\.(csv|xlsx)$/.test(name));
  if (!allowed) {
    return {
      code: "unsupported_format",
      message: "This file type is not supported yet.",
      nextStep: "Choose a CAS PDF. CDSL format support is coming soon.",
    };
  }
  return null;
}

export function importErrorCopy(code: string): {
  message: string;
  nextStep: string;
} {
  switch (code) {
    case "wrong_password":
      return {
        message: "That password didn't open the file.",
        nextStep: "Check the password on your statement and try again.",
      };
    case "not_a_pdf":
      return {
        message: "We couldn't read this as a PDF.",
        nextStep: "Download a fresh CAS PDF and select it here.",
      };
    case "too_large":
      return {
        message: "This file is over 10 MB.",
        nextStep: "Choose a smaller statement file and try again.",
      };
    case "unsupported_format":
      return {
        message: "This statement format isn't supported yet.",
        nextStep:
          "Try an NSDL CAS or CAMS/KFintech CAS PDF. CDSL format support is coming soon.",
      };
    case "parse_timeout":
      return {
        message: "The statement is taking too long to read.",
        nextStep: "Try a fresh download or a smaller file, then upload again.",
      };
    case "network_error":
      return {
        message: "We couldn't reach the import service.",
        nextStep: "Check your connection and try again.",
      };
    default:
      return {
        message: "We couldn't read this statement.",
        nextStep: "Check the file and password, then try again.",
      };
  }
}

export function errorCode(error: unknown): string {
  return error instanceof ApiError ? error.code : "network_error";
}

export function categoryForItem(item: ImportItem): ImportCategory {
  if (item.status === "duplicate") return "duplicate";
  if (item.status === "changed" || (item.status === "accepted" && item.diff))
    return "changed";
  if (item.status === "unknown" || (!item.sleeve && item.status === "accepted"))
    return "unknown";
  if (item.status === "new" || item.status === "accepted") return "new";
  if (item.diff) return "changed";
  if (item.kind === "unknown" || !item.sleeve) return "unknown";
  return "new";
}

export function groupImportItems(
  items: readonly ImportItem[],
): Record<ImportCategory, ImportItem[]> {
  const groups: Record<ImportCategory, ImportItem[]> = {
    new: [],
    changed: [],
    duplicate: [],
    unknown: [],
  };
  for (const item of items) groups[categoryForItem(item)].push(item);
  return groups;
}

export function countImportItems(items: readonly ImportItem[]): ImportCounts {
  const groups = groupImportItems(items);
  return {
    new: groups.new.length,
    changed: groups.changed.length,
    duplicate: groups.duplicate.length,
    unknown: groups.unknown.length,
  };
}

export function maskAccountReference(value: string | null): string {
  if (!value?.trim()) return "Account hidden";
  const suffix = value.replace(/[^a-z\d]/gi, "").slice(-4);
  return suffix ? `•••• ${suffix}` : "Account hidden";
}

export function quantityChange(item: ImportItem): string {
  const diff = item.diff;
  const textValue = (...keys: string[]): string | undefined => {
    for (const key of keys) {
      const value = diff?.[key];
      if (typeof value === "string" || typeof value === "number")
        return String(value);
    }
    return undefined;
  };
  const before = textValue("old_quantity", "previous_quantity", "old", "from");
  const after =
    textValue("new_quantity", "quantity", "current", "to") ?? item.quantity;
  return `${before ?? "Previous quantity"} → ${after}`;
}

export function shouldShowCasNudge(
  jobs: readonly ImportJob[],
  now = new Date(),
): boolean {
  const confirmedDates = jobs
    .filter((job) => job.status === "confirmed" && job.source !== null)
    .map((job) => job.confirmed_at ?? job.statement_date)
    .filter((date): date is string => date !== null)
    .map((date) =>
      new Date(date.length === 10 ? `${date}T00:00:00Z` : date).getTime(),
    )
    .filter(Number.isFinite);
  if (confirmedDates.length === 0) return true;
  const latest = Math.max(...confirmedDates);
  return now.getTime() - latest > 30 * 24 * 60 * 60 * 1000;
}

export function sumImportCounts(counts: Record<string, number>): number {
  return Object.values(counts).reduce(
    (total, count) => total + (Number.isFinite(count) ? Math.max(0, count) : 0),
    0,
  );
}
